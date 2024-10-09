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
from abc import abstractmethod
from typing import Coroutine, List, Optional, TypeAlias, Union, cast

from huggingface_hub import (
    ChatCompletionInput,
    ChatCompletionInputMessage,
    ChatCompletionOutput,
    TextGenerationInput,
    TextGenerationInputGenerateParameters,
    TextGenerationOutput,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Never

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.logging.hierarchical_logger import hlog_err
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import GenerateReturn, LoglikelihoodReturn, LoglikelihoodSingleTokenReturn
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
    Request,
)
from lighteval.utils import is_anthropic_available, is_openai_available
from lighteval.utils import as_list


if is_anthropic_available():
    from anthropic.types import Completion, Message

    AnthropicOutput: TypeAlias = Completion | Message
else:
    AnthropicOutput: TypeAlias = Never
if is_openai_available():
    from openai.types import Completion
    from openai.types.chat import ChatCompletion

    OpenAIOutput: TypeAlias = Completion | ChatCompletion
else:
    OpenAIOutput: TypeAlias = Never

EndpointInput: TypeAlias = TextGenerationInput | ChatCompletionInput
EndpointOutput: TypeAlias = TextGenerationOutput | ChatCompletionOutput | OpenAIOutput | AnthropicOutput


BATCH_SIZE = 50


class EndpointModel(LightevalModel):
    """Abstract endpoint model."""

    name: str
    use_async = True  # set to False for debug - async use is faster

    @property
    def disable_tqdm(self) -> bool:
        return False  # no accelerator = this is the main process

    async def _async_process_batch(
        self,
        requests: list[Request],
    ) -> list[EndpointOutput]:
        return await asyncio.gather(
            *[
                cast(
                    Coroutine[None, None, EndpointOutput],
                    self._process_request(self._prepare_request(request), request),
                )
                for request in requests
            ]
        )

    def _process_batch(
        self,
        requests: list[Request],
    ) -> list[EndpointOutput]:
        return [
            cast(EndpointOutput, self._process_request(self._prepare_request(request), request))
            for request in requests
        ]

    @abstractmethod
    def _process_request(
        self,
        prepared_request: EndpointInput,
        request: Request,
    ) -> Coroutine[None, None, EndpointOutput]|EndpointOutput:
        ...

    def _prepare_request(self, request: Request) -> EndpointInput:
        if isinstance(request, GreedyUntilRequest):
            stop = as_list(request.stop_sequence) or None
            max_tokens = request.generation_size
            context = request.context
            grammar = request.generation_grammar
        elif isinstance(request, (LoglikelihoodRequest, LoglikelihoodRollingRequest)):
            stop = None
            max_tokens = 1
            grammar = None
            rolling = isinstance(request, LoglikelihoodRollingRequest)
            if rolling:
                context = request.context
            elif isinstance(request.context, str):
                context = request.context + request.choice
            else:
                context = request.context + [ChatCompletionInputMessage(role="assistant", content=request.choice)]

        if isinstance(context, str):
            prepared_request = TextGenerationInput(
                inputs=context,
                parameters=TextGenerationInputGenerateParameters(
                    details=True,
                    decoder_input_details=True,
                    do_sample=False,
                    seed=42,
                    max_new_tokens=max_tokens,
                    stop=stop,
                    return_full_text=False,
                    grammar=grammar,
                    top_n_tokens=1,
                ),
            )
        else:
            prepared_request = ChatCompletionInput(
                messages=context,
                model=self.name,
                logprobs=True,
                stop=stop,
                max_tokens=max_tokens,
                response_format=grammar,
                seed=42,
                temperature=0.0,
                top_logprobs=1,
                stream=False,
            )
        return prepared_request

    @abstractmethod
    def _process_generate_response(self, response: EndpointOutput, request: GreedyUntilRequest) -> GenerateReturn:
        ...

    def _process_logprob_response(
        self, response: EndpointOutput, request: LoglikelihoodRequest | LoglikelihoodRollingRequest
    ) -> LoglikelihoodReturn:
        ...

    def greedy_until(
        self,
        requests: List[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> List[GenerateReturn]:
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)
            request.stop_sequence = (
                (as_list(request.stop_sequence)
                if request.stop_sequence is not None
                else []) + [self.tokenizer.eos_token]
            )

        dataset = GenerativeTaskDataset(requests=requests, dataset_splits=self.DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[str] = []

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=self.DATASET_SPLITS,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(
                dataloader, desc="Greedy generation", position=1, leave=False, disable=self.disable_tqdm
            ):
                num_samples = batch[0].num_samples
                if num_samples > 1:
                    hlog_err(
                        "Inference endpoints does not allow sampling evaluations - this is likely to fail or provide problematic results"
                    )

                if self.use_async:
                    responses = asyncio.run(self._async_process_batch(batch))
                else:
                    responses = self._process_batch(batch)
                for response, request in zip(responses, batch):
                    results.append(self._process_generate_response(response, request))

        return dataset.get_original_order(results)

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)
            request.tokenized_continuation = self.tok_encode(request.choice)
        dataset = LoglikelihoodDataset(requests=requests, dataset_splits=self.DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[str] = []

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=self.DATASET_SPLITS,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(dataloader, desc="Loglikelihoods", position=1, leave=False, disable=self.disable_tqdm):
                if self.use_async:
                    responses = asyncio.run(self._async_process_batch(batch))
                else:
                    responses = self._process_batch(batch)
                for cur_request, response in zip(batch, responses):
                    results.append(self._process_logprob_response(cast(TextGenerationOutput, response), cur_request))

        return dataset.get_original_order(results)

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs=None
    ) -> list[LoglikelihoodReturn]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        for request in requests:
            request.tokenized_context = [self.tokenizer.eos_token_id]
            request.tokenized_continuation = self.tok_encode(request.context)

        dataset = LoglikelihoodDataset(requests=requests, dataset_splits=self.DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[str] = []

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=self.DATASET_SPLITS,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(
                dataloader, desc="Loglikelihoods, rolling", position=1, leave=False, disable=self.disable_tqdm
            ):
                if self.use_async:
                    responses = asyncio.run(self._async_process_batch(batch))
                else:
                    responses = self._process_batch(batch)
                for response, request in zip(responses, batch):
                    results.append(self._process_logprob_response(cast(TextGenerationOutput, response), request))

        return dataset.get_original_order(results)

    def loglikelihood_single_token(
        self,
        requests: list[LoglikelihoodSingleTokenRequest],
        override_bs: Optional[int] = None,
    ) -> list[LoglikelihoodSingleTokenReturn]:
        raise ValueError("Endpoint models can't use single token metrics. Change the metric to the standard version")
