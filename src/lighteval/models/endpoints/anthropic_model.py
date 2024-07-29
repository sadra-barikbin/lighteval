from typing import Coroutine

from lighteval.models.endpoints import InferenceEndpointModel
from lighteval.models.model_output import GenerateReturn
from lighteval.tasks.requests import GreedyUntilRequest

class AnthropicModel(InferenceEndpointModel):
    def __init__(self):
        import anthropic
        self.client: anthropic.Anthropic = anthropic.Anthropic()

    def _async_process_request(
        self, context: str, stop_tokens: list[str], max_tokens: int
    ) -> Coroutine[None, list[TextGenerationOutput], str]:
        self.client
        return generated_text

    def _process_request(self, context: str, stop_tokens: list[str], max_tokens: int) -> TextGenerationOutput:
        # Todo: add an option to launch with conversational instead for chat prompts
        # https://huggingface.co/docs/huggingface_hub/v0.20.3/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient.conversational
        generated_text = self.client.text_generation(
            prompt=context,
            details=True,
            decoder_input_details=True,
            max_new_tokens=max_tokens,
            stop_sequences=stop_tokens,
            # truncate=,
        )

        return generated_text
    
    def greedy_until(self, requests: list[GreedyUntilRequest], override_bs: int | None = None) -> list[GenerateReturn]:
        return super().greedy_until(requests, override_bs)
        
        