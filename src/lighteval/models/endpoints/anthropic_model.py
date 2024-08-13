from functools import singledispatchmethod
from typing import cast, Coroutine, Optional
from transformers import PreTrainedTokenizerFast

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.endpoints.endpoint_model import EndpointModel, EndpointResponse
from lighteval.models.model_output import GenerateReturn, LoglikelihoodReturn
from lighteval.tasks.requests import GreedyUntilRequest, LoglikelihoodRequest, LoglikelihoodRollingRequest
from lighteval.utils import is_anthropic_available

if is_anthropic_available():
    from anthropic.types import ModelParam, Completion
    from anthropic import NOT_GIVEN

class AnthropicModel(EndpointModel):

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(self, model_id: "ModelParam"):
        import anthropic
        self.async_client: anthropic.AsyncAnthropic = anthropic.AsyncAnthropic()
        self.client : anthropic.Anthropic = anthropic.Anthropic()
        self.tokenizer = LightevalModel.Tokenizer.from_hf_tokenizer(
            PreTrainedTokenizerFast(tokenizer_object=self.client.get_tokenizer())
        )
        self.tokenizer.eos_token = "<EOT>"
        self.model_id = model_id
        

    async def _async_process_request(
        self, context: str, stop_tokens: list[str], max_tokens: int
    ) -> Coroutine[None, None, Completion]:
        
        return await self.async_client.completions.create(
            prompt=context,
            stop_sequences=stop_tokens or NOT_GIVEN,
            max_tokens_to_sample=max_tokens,
            model=self.model_id,
            temperature=0.
        )
    
    def _process_request(self, context: str, stop_tokens: list[str], max_tokens: int) -> Completion:
        return self.client.completions.create(
            prompt=context,
            stop_sequences=stop_tokens or NOT_GIVEN,
            max_tokens_to_sample=max_tokens,
            model=self.model_id,
            temperature=0.
        )
    
    @singledispatchmethod
    def _process_endpoint_response(self, request, response):
        ...
    
    @_process_endpoint_response.register
    def _(self, request: GreedyUntilRequest, response: EndpointResponse) -> GenerateReturn:
        response = cast(Completion, response)

        return GenerateReturn(
            result=response.completion,
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
