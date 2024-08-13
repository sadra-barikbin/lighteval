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

import asyncio
from typing import Coroutine, List, Optional, Union, cast
from functools import singledispatchmethod

import torch
from huggingface_hub import (
    AsyncInferenceClient,
    InferenceClient,
    InferenceEndpoint,
    InferenceEndpointTimeoutError,
    TextGenerationOutput,
    create_inference_endpoint,
    get_inference_endpoint,
)
from transformers import AutoTokenizer

from lighteval.logging.hierarchical_logger import hlog, hlog_err, hlog_warn
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.endpoints.endpoint_model import EndpointModel, EndpointResponse
from lighteval.models.model_config import EnvConfig, InferenceEndpointModelConfig, InferenceModelConfig
from lighteval.models.model_output import GenerateReturn, LoglikelihoodReturn, LoglikelihoodSingleTokenReturn
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)


class InferenceEndpointModel(EndpointModel):
    """InferenceEndpointModels can be used both with the free inference client, or with inference
    endpoints, which will use text-generation-inference to deploy your model for the duration of the evaluation.
    """

    def __init__(
        self, config: Union[InferenceEndpointModelConfig, InferenceModelConfig], env_config: EnvConfig
    ) -> None:
        self.reuse_existing = getattr(config, "should_reuse_existing", True)
        if isinstance(config, InferenceEndpointModelConfig):
            if config.should_reuse_existing:
                self.endpoint = get_inference_endpoint(
                    name=config.name, token=env_config.token, namespace=config.namespace
                )
            else:
                self.endpoint: InferenceEndpoint = create_inference_endpoint(
                    name=config.name,
                    namespace=config.namespace,
                    repository=config.repository,
                    revision=config.revision,
                    framework=config.framework,
                    task="text-generation",
                    accelerator=config.accelerator,
                    vendor=config.vendor,
                    region=config.region,
                    type=config.endpoint_type,
                    instance_size=config.instance_size,
                    instance_type=config.instance_type,
                    token=env_config.token,
                    custom_image={
                        "health_route": "/health",
                        "env": {
                            # Documentaiton: https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/launcher
                            "MAX_BATCH_PREFILL_TOKENS": "2048",
                            "MAX_INPUT_LENGTH": "2047",
                            "MAX_TOTAL_TOKENS": "2048",
                            "MODEL_ID": "/repository",
                            "HF_MODEL_TRUST_REMOTE_CODE": "true",
                            **config.get_dtype_args(),
                            **config.get_custom_env_vars(),
                        },
                        "url": (config.image_url or "ghcr.io/huggingface/text-generation-inference:latest"),
                    },
                )
            hlog("Deploying your endpoint. Please wait.")
            try:
                self.endpoint.wait(timeout=600)  # Waits for the endpoint to be deployed
            except InferenceEndpointTimeoutError as e:
                hlog_err("Endpoint did not start within 10 minutes, there was a timeout.")
                raise e
            hlog("Endpoint successfully deployed!")
            self.name = config.repository
            self.revision = self.endpoint.revision
            self.async_client: AsyncInferenceClient = self.endpoint.async_client
            self.client: InferenceClient = self.endpoint.client

        else:  # Free inference client
            self.endpoint = None
            self.name = config.model
            self.revision = "default"
            self.async_client = AsyncInferenceClient(model=config.model, token=env_config.token)
            self.client = InferenceClient(model=config.model, token=env_config.token)

        self.tokenizer: LightevalModel.HFTokenizer = LightevalModel.HFTokenizer.from_hf_tokenizer(
            AutoTokenizer.from_pretrained(self.name)
        )
        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    def cleanup(self):
        if self.endpoint is not None and not self.reuse_existing:
            self.endpoint.delete()
            hlog_warn(
                "You deleted your endpoint after using it. You'll need to create it again if you need to reuse it."
            )

    def max_length(self):
        if self._max_length is not None:
            return self._max_length

        if hasattr(self.tokenizer.hf_tokenizer, "model_max_length"):
            self._max_length = self.tokenizer.hf_tokenizer.model_max_length
        else:
            self._max_length = 2048
        return self._max_length

    async def _async_process_request(
        self, context: str, stop_tokens: list[str], max_tokens: int
    ) -> Coroutine[None, None, TextGenerationOutput]:
        # Todo: add an option to launch with conversational instead for chat prompts
        # https://huggingface.co/docs/huggingface_hub/v0.20.3/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient.conversational
        generated_text = await self.async_client.text_generation(
            prompt=context,
            details=True,
            decoder_input_details=True,
            max_new_tokens=max_tokens,
            stop_sequences=stop_tokens,
            # truncate=,
        )

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
    
    @singledispatchmethod
    def _process_endpoint_response(self, request, response):
        ...
    
    @_process_endpoint_response.register
    def _(self, request: GreedyUntilRequest, response: EndpointResponse) -> GenerateReturn:
        response = cast(TextGenerationOutput, response)

        returns_logits = request.use_logits
        return GenerateReturn(
            result=response.generated_text,
            logits=[item.logprob for item in response.details.prefill] if returns_logits else None,
            truncated_tokens_count=-1,
            padded_tokens_count=-1,
        )
    
    @_process_endpoint_response.register
    def _(self, request: LoglikelihoodRequest, response: EndpointResponse) -> LoglikelihoodReturn:
        response = cast(TextGenerationOutput, response)

        cont_toks = torch.tensor(request.tokenized_continuation)
        len_choice = len(cont_toks)

        logits = [t.logprob for t in response.details.prefill[-len_choice:] if t.logprob is not None]

        greedy_tokens = torch.tensor(logits).argmax(dim=-1)
        max_equal = (greedy_tokens == cont_toks).all().squeeze(0)
        return LoglikelihoodReturn(
            result=(sum(logits), bool(max_equal)),
            input_tokens=[t.id for t in response.details.prefill[:-len_choice]],
            generated_tokens=[t.id for t in response.details.prefill[-len_choice:]],
            truncated_tokens_count=-1,
            padded_tokens_count=-1,
        )

    @_process_endpoint_response.register
    def _(self, request: LoglikelihoodRollingRequest, response: EndpointResponse) -> LoglikelihoodReturn:
        response = cast(TextGenerationOutput, response)

        logits = [t.logprob for t in response.details.tokens[:-1]]
        return LoglikelihoodReturn(
            result=sum(logits),
            input_tokens=[t.id for t in response.details.prefill],
            generated_tokens=[t.id for t in response.details.tokens[:-1]],
            truncated_tokens_count=-1,
            padded_tokens_count=-1,
        )

    def loglikelihood_single_token(
        self,
        requests: list[LoglikelihoodSingleTokenRequest],
        override_bs: Optional[int] = None,
    ) -> list[LoglikelihoodSingleTokenReturn]:
        raise ValueError("Endpoint models can't use single token metrics. Change the metric to the standard version")
