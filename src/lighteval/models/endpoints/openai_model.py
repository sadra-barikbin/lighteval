from functools import singledispatchmethod
from typing import cast, Coroutine, Dict, List, Optional

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.endpoints.endpoint_model import EndpointModel, EndpointResponse
from lighteval.models.model_output import GenerateReturn, LoglikelihoodReturn
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    ContextType,
)
from lighteval.utils import is_openai_available

if is_openai_available():
    import tiktoken
    from openai.types import Completion
    from openai import NOT_GIVEN
    from tiktoken import Encoding


SOMEWHAT_OPEN_OPENAI_MODELS =  ["davinci-002", "babbage-002"]


class OpenAIModel(EndpointModel):

    _DEFAULT_MAX_LENGTH = 2048
    
    def __init__(self, model_id: str):
        import openai
        self.async_client: openai.AsyncOpenAI = openai.AsyncOpenAI()
        self.client : openai.OpenAI = openai.OpenAI()
        self.tokenizer: OpenAIModel.Tokenizer = self.Tokenizer(model_id)
        self.name = model_id
        

    async def _async_process_request(
        self, context: str, stop_tokens: list[str], max_tokens: int
    ) -> Coroutine[None, None, Completion]:
        
        if self.name in SOMEWHAT_OPEN_OPENAI_MODELS:
            logprobs = 1
        else:
            logprobs = None
        
        return await self.async_client.completions.create(
            prompt=context,
            stop=stop_tokens or NOT_GIVEN,
            logprobs=logprobs,
            max_tokens=max_tokens,
            model=self.name,
            temperature=0.,
            echo=True,
            seed=42,
        )
    
    def _process_request(self, context: str, stop_tokens: list[str], max_tokens: int) -> Completion:

        if self.name in SOMEWHAT_OPEN_OPENAI_MODELS:
            logprobs = 1
        else:
            logprobs = NOT_GIVEN

        return self.async_client.completions.create(
            prompt=context,
            stop=stop_tokens or NOT_GIVEN,
            logprobs=logprobs,
            max_tokens=max_tokens,
            model=self.name,
            temperature=0.,
            echo=True,
            seed=42,
        )
    
    @singledispatchmethod
    def _process_endpoint_response(self, request, response):
        ...
    
    @_process_endpoint_response.register
    def _(self, request: GreedyUntilRequest, response: EndpointResponse) -> GenerateReturn:
        response: Completion = cast(Completion, response)
        completion = response.choices[0]
        prompt_length = response.usage.prompt_tokens
        returns_logits = request.use_logits

        if self.name in SOMEWHAT_OPEN_OPENAI_MODELS:
            input_tokens = [self.tokenizer.convert_token_to_id(t) for t in completion.logprobs.tokens[:prompt_length]]
            generated_tokens = [self.tokenizer.convert_token_to_id(t) for t in completion.logprobs.tokens[prompt_length:]]
            logits = completion.logprobs.token_logprobs[1:] if returns_logits else None
        else:
            input_tokens = generated_tokens = logits = None

        return GenerateReturn(
            result=completion.text,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            logits=logits,
            truncated_tokens_count=-1,
            padded_tokens_count=-1,
        )
    
    @_process_endpoint_response.register
    def _(self, request: LoglikelihoodRequest, response: EndpointResponse) -> LoglikelihoodReturn:
        response: Completion = cast(Completion, response)
        completion = response.choices[0]

        len_choice = len(request.tokenized_continuation)
        logits = completion.logprobs.token_logprobs[-len_choice-1: -1]
        greedy_tokens = list(list(zip(*completion.logprobs.top_logprobs[-len_choice-1:-1]))[0])
        max_equal = greedy_tokens == completion.logprobs.tokens[-len_choice-1:-1]
        prompt_length = len(request.tokenized_context)
        return LoglikelihoodReturn(
            result=(sum(logits), max_equal),
            input_tokens=[self.tokenizer.convert_token_to_id(t) for t in completion.logprobs.tokens[:prompt_length]],
            generated_tokens=[self.tokenizer.convert_token_to_id(t) for t in completion.logprobs.tokens[prompt_length:-1]],
            truncated_tokens_count=-1,
            padded_tokens_count=-1,
        )
        
    def greedy_until(
        self,
        requests: List[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> List[GenerateReturn]:
        if self.name not in SOMEWHAT_OPEN_OPENAI_MODELS and any(req.use_logits for req in requests):
            raise ValueError("OpenAI models could not process requests with `use_logits=True` except the models 'davinci-002' and 'babbage-002'.")
        return super(OpenAIModel, self).greedy_until(requests, override_bs)
            
    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        if self.name not in SOMEWHAT_OPEN_OPENAI_MODELS:
            raise ValueError("OpenAI models could not be evaluated by non-generative metrics except the models 'davinci-002' and 'babbage-002'.")
        return super(OpenAIModel, self).loglikelihood(requests, override_bs)

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        if self.name not in SOMEWHAT_OPEN_OPENAI_MODELS:
            raise ValueError("OpenAI models could not be evaluated by non-generative metrics except the models 'davinci-002' and 'babbage-002'.")
        return super(OpenAIModel, self).loglikelihood_rolling(requests, override_bs)

    @property
    def add_special_tokens(self):
        return True

    @property
    def max_length(self) -> int:
        return OpenAIModel._DEFAULT_MAX_LENGTH
    
    class Tokenizer(LightevalModel.Tokenizer):
        encoding: Encoding
        model_id: str

        def __init__(self, model_id: str):
            self.model_id = model_id
            self.encoding = tiktoken.encoding_for_model(model_id)
        
        def encode(self, input_text: str, add_special_tokens: bool= True) -> List[int]:
            return self.encoding.encode(input_text, allowed_special= 'all' if add_special_tokens else set())
        
        def convert_token_to_id(self, token:str) -> int:
            return self.encoding.encode_single_token(token)
        
        def get_token_count(self, input: ContextType) -> int:
            if isinstance(input, str):
                return len(self.encoding.encode(input))
            else:
                # Adapted from https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
                match self.model_id:
                    case "gpt-3.5-turbo-0613" | "gpt-3.5-turbo-16k-0613" | "gpt-4-0314" | "gpt-4-32k-0314" | "gpt-4-0613" | "gpt-4-32k-0613":
                        tokens_per_message = 3
                        tokens_per_name = 1
                    case "gpt-3.5-turbo-0301":
                        tokens_per_message = 4
                        tokens_per_name = -1
                    case model if "gpt-3.5-turbo" in model or "gpt-4" in model:
                            tokens_per_message = 3
                            tokens_per_name = 1
                num_tokens = 0
                for message in input:
                    num_tokens += tokens_per_message
                    for key, value in message.items():
                        if key in ("role", "content"):
                            num_tokens += len(self.encoding.encode(value))
                        elif key == "name":
                            num_tokens += tokens_per_name
                num_tokens += 3
                return num_tokens

        @property
        def eos_token_id(self):
            return self.encoding.eot_token
        
        @property
        def eos_token(self):
            return self.encoding.decode_single_token_bytes(self.eos_token_id).decode()
