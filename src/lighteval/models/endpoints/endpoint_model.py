import asyncio
from abc import abstractmethod
from typing import Coroutine, List, Optional, Union, NewType, cast

from huggingface_hub import (
    TextGenerationInput,
    TextGenerationInputGenerateParameters,
    ChatCompletionInput,
    TextGenerationOutput,
    ChatCompletionInputMessage,
    ChatCompletionOutput,
    TextGenerationInput,
)
from huggingface_hub.inference._generated.types import BaseInferenceType
from torch.utils.data import DataLoader
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.logging.hierarchical_logger import hlog_err
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import ModelReturn, GenerateReturn, LoglikelihoodReturn, LoglikelihoodSingleTokenReturn
from lighteval.tasks.requests import (
    Request,
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
    GreedyUntilMultiTurnRequest,
)
from lighteval.utils import as_list

# For Anthropic and OpenAI models.
Completion = NewType("Completion", object)
ChatCompletion = NewType("ChatCompletion", object)
EndpointResponse = Union[TextGenerationOutput, ChatCompletionOutput, Completion, ChatCompletion]
LoglikelihoodRequests = Union[LoglikelihoodRequest, LoglikelihoodRollingRequest]

BATCH_SIZE = 50


class EndpointModel(LightevalModel):
    """Abstract endpoint model.
    """
    name: str
    use_async = True  # set to False for debug - async use is faster

    @property
    def disable_tqdm(self) -> bool:
        return False  # no accelerator = this is the main process
    
    @abstractmethod
    def _process_request(
        self, prepared_request: BaseInferenceType, request: Request,
    ) -> Union[Coroutine[None, None, ModelReturn], ModelReturn]:
        ...

    def _prepare_request(self, request: Request) -> TextGenerationInput|ChatCompletionInput:
        if isinstance(request, (GreedyUntilRequest, GreedyUntilMultiTurnRequest)):
            stop = as_list(request.stop_sequence) or None
            max_tokens = request.generation_size
            context = request.context
        elif isinstance(request, (LoglikelihoodRequest, LoglikelihoodRollingRequest)):
            stop = None
            max_tokens = 1
            rolling = isinstance(request, LoglikelihoodRollingRequest)
            if rolling:
                context = request.context
            else:
                if isinstance(request.context, str):
                    context = request.context + request.choice
                else:
                    context = request.context + ChatCompletionInputMessage("assistant", request.choice)

        if isinstance(request.context, str):
            client_input = TextGenerationInput(
                inputs=context,
                parameters=TextGenerationInputGenerateParameters(
                    details=True,
                    decoder_input_details=True,
                    do_sample=False,
                    seed=42,
                    max_new_tokens=max_tokens,
                    stop=stop
                )
            )
        else:
            client_input = ChatCompletionInput(
                messages=context,
                model=self.name,
                logprobs=True,
                stop=stop,
                max_tokens=max_tokens,
                seed=42,
                temperature=0.,
            )
        return client_input

    async def _async_process_batch(
        self,
        requests: list[Request],
    ) -> Coroutine[None, None, list[ModelReturn]]:
        return await asyncio.gather(
            *[
                cast(
                    Coroutine[None, None, ModelReturn],
                    self._process_request(self._prepare_request(request), request)
                )
                for request in requests
            ]
        )

    def _process_batch(
        self,
        requests: list[Request],
    ) -> list[ModelReturn]:
        return [
            cast(
                ModelReturn,
                self._process_request(self._prepare_request(request), request)
            )
            for request in requests
        ]

    def greedy_until(
        self,
        requests: List[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> List[GenerateReturn]:
        for request in requests:
            request.tokenized_context = self.tokenizer.tok_encode(request.context, self.add_special_tokens)
            request.stop_sequence = as_list(request.stop_sequence) + [self.tokenizer.eos_token]

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[GenerateReturn] = []

        if any(req.num_samples > 1 for req in requests):
            hlog_err(
                "Endpoint models do not allow sampling evaluations - this is likely to fail or provide problematic results"
            )

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(
                dataloader, desc="Greedy generation", position=1, leave=False, disable=self.disable_tqdm
            ):
                if self.use_async:
                    results.extend(asyncio.run(self._async_process_batch(batch)))
                else:
                    results.extend(self._process_batch(batch))

        return dataset.get_original_order(results)

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        for request in requests:
            request.tokenized_context = self.tokenizer.tok_encode(request.context, self.add_special_tokens)
            request.tokenized_continuation = self.tokenizer.tok_encode(request.choice, self.add_special_tokens)
        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[LoglikelihoodReturn] = []

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(dataloader, desc="Loglikelihoods", position=1, leave=False, disable=self.disable_tqdm):
                if self.use_async:
                    results.extend(asyncio.run(self._async_process_batch(batch)))
                else:
                    results.extend(self._process_batch(batch))

        return dataset.get_original_order(results)

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] =None
    ) -> list[LoglikelihoodReturn]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        for request in requests:
            request.tokenized_context = [self.tokenizer.eos_token_id]
            request.tokenized_continuation = self.tokenizer.tok_encode(request.context, self.add_special_tokens)

        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[LoglikelihoodReturn] = []

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(
                dataloader, desc="Loglikelihoods, rolling", position=1, leave=False, disable=self.disable_tqdm
            ):
                if self.use_async:
                    results.extend(asyncio.run(self._async_process_batch(batch, rolling=True)))
                else:
                    results.extend(self._process_batch(batch, rolling=True))

        return dataset.get_original_order(results)

    def loglikelihood_single_token(
        self,
        requests: list[LoglikelihoodSingleTokenRequest],
        override_bs: Optional[int] = None,
    ) -> list[LoglikelihoodSingleTokenReturn]:
        raise ValueError("Endpoint models can't use single token metrics. Change the metric to the standard version")
