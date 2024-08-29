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
from dataclasses import asdict
from typing import Coroutine, List, Optional, TypeAlias, Union, cast

from huggingface_hub import (
    AsyncInferenceClient,
    ChatCompletionInput,
    ChatCompletionInputMessage,
    ChatCompletionOutput,
    InferenceClient,
    InferenceEndpoint,
    InferenceEndpointTimeoutError,
    TextGenerationInput,
    TextGenerationInputGenerateParameters,
    TextGenerationOutput,
    create_inference_endpoint,
    get_inference_endpoint,
)
from transformers import AutoTokenizer

from lighteval.logging.hierarchical_logger import hlog, hlog_err, hlog_warn
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.endpoints.endpoint_model import EndpointInput, EndpointOutput
from lighteval.models.model_config import InferenceEndpointModelConfig, InferenceModelConfig
from lighteval.models.model_output import GenerativeResponse, LoglikelihoodResponse
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    Request,
)
from lighteval.utils.utils import EnvConfig


BATCH_SIZE = 50


class InferenceEndpointModel(LightevalModel):
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

        self.use_async = True  # set to False for debug - async use is faster

        self._tokenizer = AutoTokenizer.from_pretrained(self.name)
        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False

        self.model_info = ModelInfo(
            model_name=self.name,
            model_sha=self.revision,
            model_dtype=config.model_dtype or "default",
            model_size=-1,
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def disable_tqdm(self) -> bool:
        False  # no accelerator = this is the main process

    def cleanup(self):
        if self.endpoint is not None and not self.reuse_existing:
            self.endpoint.delete()
            hlog_warn(
                "You deleted your endpoint after using it. You'll need to create it again if you need to reuse it."
            )

    def max_length(self):
        if self._max_length is not None:
            return self._max_length

        if hasattr(self.tokenizer, "model_max_length"):
            self._max_length = self.tokenizer.model_max_length
        else:
            self._max_length = 2048
        return self._max_length

    def _process_request(
        self, prepared_request: EndpointInput, request: Request
    ) -> EndpointOutput | Coroutine[None, None, EndpointOutput]:
        client = self.async_client if self.use_async else self.client
        if isinstance(prepared_request, TextGenerationInput):
            # https://github.com/huggingface/huggingface_hub/issues/2471
            request_as_dict = asdict(prepared_request)
            request_as_dict["parameters"]["stop_sequences"] = request_as_dict["parameters"]["stop"]
            del request_as_dict["parameters"]["stop"]

            return client.text_generation(prepared_request.inputs, **request_as_dict["parameters"])
        elif isinstance(prepared_request, ChatCompletionInput):
            return client.chat_completion(**prepared_request)

    def _process_generate_response(self, response: EndpointOutput, request: GreedyUntilRequest) -> GenerativeResponse:
        is_chat = isinstance(response, ChatCompletionOutput)
        if is_chat:
            logits = [t.logprob for t in response.choices[0].logprobs.content]
            input_tokens = request.tokenized_context
            generated_tokens = self.tokenizer.convert_tokens_to_ids(
                [t.token for t in response.choices[0].logprobs.content]
            )
        else:
            logits = [t.logprob for t in response.details.tokens]
            input_tokens = [t.id for t in response.details.prefill]
            generated_tokens = [t.id for t in response.details.tokens]
        return GenerativeResponse(
            result=response.choices[0].message.content if is_chat else response.generated_text,
            logits=logits if request.use_logits else None,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            truncated_tokens_count=-1,
            padded_tokens_count=-1,
        )

    def _process_logprob_response(
        self, response: TextGenerationOutput, request: LoglikelihoodRequest | LoglikelihoodRollingRequest
    ) -> LoglikelihoodResponse:
        len_choice = len(request.tokenized_continuation)
        logits = sum([t.logprob for t in response.details.prefill[1:][-len_choice:]])
        return LoglikelihoodResponse(
            result=(logits, True) if isinstance(request, LoglikelihoodRequest) else logits,
            input_tokens=[t.id for t in response.details.prefill[:-len_choice]],
            generated_tokens=-1,
            truncated_tokens_count=-1,
            padded_tokens_count=-1,
        )
