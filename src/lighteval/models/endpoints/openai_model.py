# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Coroutine, List, Optional, cast

from huggingface_hub import TextGenerationInput

from lighteval.models.endpoints.endpoint_model import EndpointInput, EndpointModel, EndpointOutput, OpenAIOutput
from lighteval.models.model_output import GenerativeResponse, LoglikelihoodResponse
from lighteval.tasks.requests import (
    Conversation,
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    Request,
)
from lighteval.utils.imports import is_openai_available


if is_openai_available():
    import tiktoken
    from openai import NOT_GIVEN
    from openai.types import Completion
    from openai.types.chat import ChatCompletion
    from tiktoken import Encoding


SOMEWHAT_OPEN_OPENAI_MODELS = ["davinci-002", "babbage-002"]


class OpenAIModel(EndpointModel):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(self, model_id: str):
        import openai

        self.async_client: openai.AsyncOpenAI = openai.AsyncOpenAI()
        self.client: openai.OpenAI = openai.OpenAI()
        self._tokenizer = self.Tokenizer(model_id)
        self.name = model_id

    @property
    def tokenizer(self):
        return self._tokenizer

    def _process_request(
        self, prepared_request: EndpointInput, request: Request
    ) -> Coroutine[None, None, OpenAIOutput] | OpenAIOutput:
        client = self.async_client if self.use_async else self.client
        if self.name in SOMEWHAT_OPEN_OPENAI_MODELS:
            logprobs = 1
        else:
            logprobs = NOT_GIVEN

        if isinstance(prepared_request, TextGenerationInput):
            return client.completions.create(
                prompt=prepared_request.inputs,
                stop=prepared_request.parameters.stop or NOT_GIVEN,
                logprobs=logprobs,
                max_tokens=prepared_request.parameters.max_new_tokens or NOT_GIVEN,
                model=self.name,
                temperature=0.0,
                echo=True,
                seed=42,
            )
        else:
            return client.chat.completions.create(
                **prepared_request,
            )

    def _process_generate_response(self, response: EndpointOutput, request: GreedyUntilRequest) -> GenerativeResponse:
        if isinstance(response, Completion):
            completion = response.choices[0]
            prompt_length = response.usage.prompt_tokens
            returns_logits = request.use_logits

            if self.name in SOMEWHAT_OPEN_OPENAI_MODELS:
                input_tokens = [
                    self.tokenizer.convert_token_to_id(t) for t in completion.logprobs.tokens[:prompt_length]
                ]
                generated_tokens = [
                    self.tokenizer.convert_token_to_id(t) for t in completion.logprobs.tokens[prompt_length:]
                ]
                logits = completion.logprobs.token_logprobs[1:] if returns_logits else None
            else:
                input_tokens = generated_tokens = logits = None

            return GenerativeResponse(
                result=completion.text,
                input_tokens=input_tokens,
                generated_tokens=generated_tokens,
                logits=logits,
                truncated_tokens_count=-1,
                padded_tokens_count=-1,
            )
        else:
            response = cast(ChatCompletion, response)
            return GenerativeResponse(
                result=response.choices[0].message.content,
                truncated_tokens_count=-1,
                padded_tokens_count=-1,
            )

    def _process_logprob_response(
        self, response: Completion, request: LoglikelihoodRequest | LoglikelihoodRollingRequest
    ) -> LoglikelihoodResponse:
        completion = response.choices[0]

        len_choice = len(request.tokenized_continuation)
        logits = completion.logprobs.token_logprobs[-len_choice - 1 : -1]
        greedy_tokens = list(list(zip(*completion.logprobs.top_logprobs[-len_choice - 1 : -1]))[0])
        max_equal = greedy_tokens == completion.logprobs.tokens[-len_choice - 1 : -1]
        prompt_length = len(request.tokenized_context)
        return LoglikelihoodResponse(
            result=(sum(logits), max_equal),
            input_tokens=[self.tokenizer.convert_token_to_id(t) for t in completion.logprobs.tokens[:prompt_length]],
            generated_tokens=[
                self.tokenizer.convert_token_to_id(t) for t in completion.logprobs.tokens[prompt_length:-1]
            ],
            truncated_tokens_count=-1,
            padded_tokens_count=-1,
        )

    def greedy_until(
        self,
        requests: List[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> List[GenerativeResponse]:
        if self.name not in SOMEWHAT_OPEN_OPENAI_MODELS and any(req.use_logits for req in requests):
            raise ValueError(
                "OpenAI models could not process requests with `use_logits=True` except the models 'davinci-002' and 'babbage-002'."
            )
        return super(OpenAIModel, self).greedy_until(requests, override_bs)

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        if self.name not in SOMEWHAT_OPEN_OPENAI_MODELS:
            raise ValueError(
                "OpenAI models could not be evaluated by non-generative metrics except the models 'davinci-002' and 'babbage-002'."
            )
        return super(OpenAIModel, self).loglikelihood(requests, override_bs)

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        if self.name not in SOMEWHAT_OPEN_OPENAI_MODELS:
            raise ValueError(
                "OpenAI models could not be evaluated by non-generative metrics except the models 'davinci-002' and 'babbage-002'."
            )
        return super(OpenAIModel, self).loglikelihood_rolling(requests, override_bs)

    @property
    def add_special_tokens(self):
        return True

    @property
    def max_length(self) -> int:
        return OpenAIModel._DEFAULT_MAX_LENGTH

    class Tokenizer:
        encoding: Encoding
        model_id: str

        def __init__(self, model_id: str):
            self.model_id = model_id
            self.encoding = tiktoken.encoding_for_model(model_id)

        def encode(self, input_text: str, add_special_tokens: bool = True) -> List[int]:
            return self.encoding.encode(input_text, allowed_special="all" if add_special_tokens else set())

        def convert_token_to_id(self, token: str) -> int:
            return self.encoding.encode_single_token(token)

        def get_token_count(self, input: str | Conversation) -> int:
            if isinstance(input, str):
                return len(self.encoding.encode(input))
            else:
                # Adapted from https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
                match self.model_id:
                    case (
                        "gpt-3.5-turbo-0613"
                        | "gpt-3.5-turbo-16k-0613"
                        | "gpt-4-0314"
                        | "gpt-4-32k-0314"
                        | "gpt-4-0613"
                        | "gpt-4-32k-0613"
                    ):
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
