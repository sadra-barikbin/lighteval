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

from typing import Coroutine, Optional, cast

from huggingface_hub import ChatCompletionInput, TextGenerationInput
from transformers import PreTrainedTokenizerFast

from lighteval.models.endpoints.endpoint_model import AnthropicOutput, EndpointInput, EndpointModel, EndpointOutput
from lighteval.models.abstract_model import ModelInfo
from lighteval.models.model_output import GenerateReturn, LoglikelihoodReturn
from lighteval.tasks.requests import GreedyUntilRequest, LoglikelihoodRequest, LoglikelihoodRollingRequest, Request
from lighteval.utils import is_anthropic_available


if is_anthropic_available():
    from anthropic import NOT_GIVEN
    from anthropic.types import Completion, Message
    from anthropic.types.message_create_params import ToolChoiceToolChoiceTool
    from anthropic.types.tool_param import InputSchemaTyped, ToolParam


class AnthropicModel(EndpointModel):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(self, model_id: str):
        import anthropic

        self.async_client: anthropic.AsyncAnthropic = anthropic.AsyncAnthropic()
        self.client: anthropic.Anthropic = anthropic.Anthropic()
        self._tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.client.get_tokenizer())
        self.tokenizer.eos_token = "<EOT>"
        self.name = model_id
        self.model_info = ModelInfo(model_id)


    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self._tokenizer

    def hf_to_anthropic_tool_param(self, input: ChatCompletionInput) -> dict:
        if input.tool_choice:
            if isinstance(input.tool_choice, str):
                tool_choice = ToolChoiceToolChoiceTool(name=input.tool_choice, type="tool")
            else:
                tool_choice = ToolChoiceToolChoiceTool(name=input.tool_choice.function.name, type="tool")
        else:
            tool_choice = NOT_GIVEN
        if input.tools:
            tools = []
            for tool in input.tools:
                tools.append(
                    ToolParam(
                        input_schema=InputSchemaTyped(type="object", properties=tool.function.arguments),
                        name=tool.function.name,
                        description=tool.function.description,
                    )
                )
        else:
            tools = NOT_GIVEN
        return {"tool_choice": tool_choice, "tools": tools}

    def _process_request(
        self,
        prepared_request: EndpointInput,
        request: Request,
    ) -> Coroutine[None, None, AnthropicOutput]|AnthropicOutput:
        client = self.async_client if self.use_async else self.client
        if isinstance(prepared_request, TextGenerationInput):
            return client.completions.create(
                prompt=prepared_request.inputs,
                stop_sequences=prepared_request.parameters.stop or NOT_GIVEN,
                max_tokens_to_sample=prepared_request.parameters.max_new_tokens or self.max_length,
                model=self.name,
                temperature=0.0,
            )
        else:
            return client.messages.create(
                messages=prepared_request.messages,
                max_tokens=prepared_request.max_tokens or self.max_length,
                model=self.name,
                stop_sequences=prepared_request.stop or NOT_GIVEN,
                temperature=0.0,
                **self.hf_to_anthropic_tool_param(prepared_request),
            )

    def _process_generate_response(self, response: EndpointOutput, request: GreedyUntilRequest) -> GenerateReturn:
        if isinstance(response, Completion):
            return GenerateReturn(
                result=response.completion,
                truncated_tokens_count=-1,
                padded_tokens_count=-1,
            )
        else:
            response = cast(Message, response)
            return GenerateReturn(
                result=response.content[0].text,
                truncated_tokens_count=-1,
                padded_tokens_count=-1,
            )

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        raise ValueError("Anthropic models work only with generative metrics as the api does not provide the logits.")

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        raise ValueError("Anthropic models work only with generative metrics as the api does not provide the logits.")

    @property
    def add_special_tokens(self):
        return True

    @property
    def max_length(self) -> int:
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        return AnthropicModel._DEFAULT_MAX_LENGTH