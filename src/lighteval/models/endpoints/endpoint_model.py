import asyncio
from abc import abstractmethod
from typing import Coroutine, List, Optional, Union, NewType

from huggingface_hub import TextGenerationOutput
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
)
from lighteval.utils import as_list

# For Anthropic and OpenAI models.
Completion = NewType("Completion", object) 
EndpointResponse = Union[TextGenerationOutput, Completion]
LoglikelihoodRequests = Union[LoglikelihoodRequest, LoglikelihoodRollingRequest]

BATCH_SIZE = 50


class EndpointModel(LightevalModel):
    """Abstract endpoint model.
    """

    use_async = True  # set to False for debug - async use is faster

    @property
    def disable_tqdm(self) -> bool:
        return False  # no accelerator = this is the main process

    @abstractmethod
    async def _async_process_request(
        self, context: str, stop_tokens: list[str], max_tokens: int
    ) -> Coroutine[None, None, EndpointResponse]:
        ...
    
    @abstractmethod
    def _process_request(
        self, context: str, stop_tokens: list[str], max_tokens: int
    ) -> EndpointResponse:
        ...

    @abstractmethod
    def _process_endpoint_response(self, request: Request, response: EndpointResponse) -> ModelReturn:
        ...

    async def _async_process_batch_generate(
        self,
        requests: list[GreedyUntilRequest],
    ) -> Coroutine[None, None, list[GenerateReturn]]:
        responses = await asyncio.gather(
            *[
                self._async_process_request(
                    context=request.context,
                    stop_tokens=as_list(request.stop_sequence),
                    max_tokens=request.generation_size,
                )
                for request in requests
            ]
        )
        return [self._process_endpoint_response(req, res) for req, res in zip(requests, responses)]

    def _process_batch_generate(
        self,
        requests: list[GreedyUntilRequest],
    ) -> list[GenerateReturn]:
        return [
            self._process_endpoint_response(
                request,
                self._process_request(
                    context=request.context,
                    stop_tokens=as_list(request.stop_sequence),
                    max_tokens=request.generation_size,
                )
            )
            for request in requests
        ]

    async def _async_process_batch_logprob(
        self, requests: list[LoglikelihoodRequests], rolling: bool = False
    ) -> Coroutine[None, None, list[LoglikelihoodReturn]]:
        responses = await asyncio.gather(
            *[
                self._async_process_request(
                    context=request.context if rolling else request.context + request.choice,
                    stop_tokens=[],
                    max_tokens=1,
                )
                for request in requests
            ]
        )
        return [self._process_endpoint_response(req, res) for req, res in zip(requests, responses)]

    def _process_batch_logprob(
        self, requests: list[LoglikelihoodRequests], rolling: bool = False
    ) -> list[LoglikelihoodReturn]:
        return [
            self._process_endpoint_response(
                request,
                self._process_request(
                    context=request.context if rolling else request.context + request.choice,
                    stop_tokens=[],
                    max_tokens=1,
                )
            )
            for request in requests
        ]

    def greedy_until(
        self,
        requests: List[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> List[GenerateReturn]:
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)
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
                    results.extend(asyncio.run(self._async_process_batch_generate(batch)))
                else:
                    results.extend(self._process_batch_generate(batch))

        return dataset.get_original_order(results)

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodReturn]:
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)
            request.tokenized_continuation = self.tok_encode(request.choice)
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
                    results.extend(asyncio.run(self._async_process_batch_logprob(batch)))
                else:
                    results.extend(self._process_batch_logprob(batch))

        return dataset.get_original_order(results)

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] =None
    ) -> list[LoglikelihoodReturn]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        for request in requests:
            request.tokenized_context = [self.tokenizer.eos_token_id]
            request.tokenized_continuation = self.tok_encode(request.context)

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
                    results.extend(asyncio.run(self._async_process_batch_logprob(batch, rolling=True)))
                else:
                    results.extend(self._process_batch_logprob(batch, rolling=True))

        return dataset.get_original_order(results)

    def loglikelihood_single_token(
        self,
        requests: list[LoglikelihoodSingleTokenRequest],
        override_bs: Optional[int] = None,
    ) -> list[LoglikelihoodSingleTokenReturn]:
        raise ValueError("Endpoint models can't use single token metrics. Change the metric to the standard version")
