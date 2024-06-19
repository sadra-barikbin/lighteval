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
import math
from typing import Coroutine, List, Tuple, Union

import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from lighteval.utils import NO_TGI_ERROR_MSG, as_list, is_tgi_available
from lighteval.models.model_output import LoglikelihoodReturn, GenerateReturn
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest
)


if is_tgi_available():
    from text_generation import Client


BATCH_SIZE = 50


def divide_chunks(array, n):
    # looping till length array
    for i in range(0, len(array), n):
        yield array[i : i + n]


class ModelClient:

    def __init__(
        self,
        address,
        auth_token=None,
    ) -> None:
        if not is_tgi_available():
            raise ImportError(NO_TGI_ERROR_MSG)
        headers = {} if auth_token is None else {"Authorization": f"Basic {auth_token}"}

        self.client = Client(address, headers=headers, timeout=240)
        self.model_info = requests.get(f"{address}/info").json()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_info["model_id"])

    def __process_request_generate(self, request: GreedyUntilRequest) -> Coroutine[None, List, str]:
        request.stop_sequence = (
            (as_list(request.stop_sequence)
            if request.stop_sequence is not None
            else []) + [self.tokenizer.eos_token]
        )

        generated_text = self.client.generate(
            request.context,
            max_new_tokens=request.generation_size,
            decoder_input_details=True,
            stop_sequences=request.stop_sequence,
            seed=42,
            truncate=self.max_length,
        )

        return generated_text

    def greedy_until(self, requests: List[GreedyUntilRequest], override_bs=None) -> List[GenerateReturn]:
        generated_texts: List[GenerateReturn] = []

        batch_size = override_bs if override_bs > 0 else BATCH_SIZE

        for batch in tqdm(
            divide_chunks(requests, batch_size), total=math.ceil(len(requests) // batch_size), maxinterval=2
        ):
            results = [self.__process_request_generate(req) for req in batch]
            generated_texts.extend([GenerateReturn(result=result.generated_text) for result in results])
        return generated_texts

    def __process_request_logprob(self, request: Tuple[str, str]) -> Coroutine[None, List, str]:
        context, choice = request
        out = self.client.generate(context + choice, max_new_tokens=1, decoder_input_details=True)
        return out

    def loglikelihood(self, requests: List[LoglikelihoodRequest], override_bs=None) -> List[LoglikelihoodReturn]:
        res: List[LoglikelihoodReturn] = []

        batch_size = override_bs if override_bs > 0 else BATCH_SIZE

        for batch in tqdm(
            divide_chunks(requests, batch_size), total=math.ceil(len(requests) // batch_size), maxinterval=1
        ):
            batch = [(req.context, req.choice) for req in batch]
            # results = run(self.__process_batch_logprob(batch))
            results = [self.__process_request_logprob(req) for req in batch]
            details = [result.details.prefill for result in results]

            for detail, (context, choice) in zip(details, batch):
                tokenized_context = self.tokenizer.tokenize(context, add_special_tokens=True)
                tokenized_input = self.tokenizer.tokenize(context + choice, add_special_tokens=True)

                i = 0
                while i < len(tokenized_context) and tokenized_input[i] == tokenized_context[i]:
                    i += 1

                logprobs = [token.logprob for token in detail[i:]]

                logit_sum: float = np.sum(logprobs)
                res.append(LoglikelihoodReturn(result=(logit_sum, False)))

        return res
    
    @property
    def max_length(self) -> int:
        return self.model_info["max_input_length"]
    
    def set_cache_hook(self, cache_hook):
        self.cache_hook = cache_hook
