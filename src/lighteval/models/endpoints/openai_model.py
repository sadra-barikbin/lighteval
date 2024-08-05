from functools import singledispatchmethod
from typing import cast, Coroutine, Optional
from transformers import PreTrainedTokenizerFast

from lighteval.models.endpoints.endpoint_model import EndpointModel, EndpointResponse
from lighteval.models.model_output import GenerateReturn, LoglikelihoodReturn
from lighteval.tasks.requests import GreedyUntilRequest, LoglikelihoodRequest, LoglikelihoodRollingRequest
from lighteval.utils import is_openai_available

if is_openai_available():
    from openai.types import Completion


SOMEWHAT_OPEN_OPENAI_MODELS =  ["davinci-002", "babbage-002"]


class OpenAIModel(EndpointModel):

    _DEFAULT_MAX_LENGTH = 2048
    
    def __init__(self, model_id: str):
        import openai
        self.async_client: openai.AsyncOpenAI = openai.AsyncOpenAI()
        self.client : openai.OpenAI = openai.OpenAI()
        # self._tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.client.get_tokenizer())
        self.model_id = model_id
        

    async def _async_process_request(
        self, context: str, stop_tokens: list[str], max_tokens: int
    ) -> Coroutine[None, None, Completion]:
        
        if self.model_id in SOMEWHAT_OPEN_OPENAI_MODELS:
            logprobs = 1
        else:
            logprobs = None
        
        return await self.async_client.completions.create(
            prompt=context,
            stop=stop_tokens,
            logprobs=logprobs,
            max_tokens=max_tokens,
            model=self.model_id,
            temperature=0.,
            echo=True,
            seed=42,
        )
    
    def _process_request(self, context: str, stop_tokens: list[str], max_tokens: int) -> Completion:

        if self.model_id in SOMEWHAT_OPEN_OPENAI_MODELS:
            logprobs = 1
        else:
            logprobs = None

        return self.async_client.completions.create(
            prompt=context,
            stop=stop_tokens,
            logprobs=logprobs,
            max_tokens=max_tokens,
            model=self.model_id,
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
        return GenerateReturn(
            result=completion.text,
            input_tokens=[self.tokenizer.convert_token_to_id(t) for t in completion.logprobs.tokens[:prompt_length]],
            generated_tokens=[self.tokenizer.convert_token_to_id(t) for t in completion.logprobs.tokens[prompt_length:]],
            logits=completion.logprobs.token_logprobs[1:] if returns_logits else None,
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
        
    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        if self.model_id not in:
            raise ValueError("OpenAI models could not be evaluated by non-generative metrics except the models 'davinci-002' and 'babbage-002'.")
        return super(OpenAIModel, self).loglikelihood(requests, override_bs)

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        if self.model_id not in ["davinci-002", "babbage-002"]:
            raise ValueError("OpenAI models could not be evaluated by non-generative metrics except the models 'davinci-002' and 'babbage-002'.")
        return super(OpenAIModel, self).loglikelihood_rolling(requests, override_bs)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return True

    @property
    def max_length(self) -> int:
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        return AnthropicModel._DEFAULT_MAX_LENGTH
