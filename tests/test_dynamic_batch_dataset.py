import pytest
from dataclasses import dataclass
from lighteval.data import DynamicBatchDataset


@dataclass
class DummyRequest:
    tokenized_context: str


class DummyDynamicBatchDataset(DynamicBatchDataset):
    def _sorting_criteria(self, request) -> int:
        return 0
    

@pytest.mark.parametrize("total_size, num_splits", [(5, 4), (5, 5), (4, 5), (6, 3)])
def test_dataset_iteration(total_size: int, num_splits: int):
    reqs = [DummyRequest(f"{i}") for i in range(total_size)]
    ds = DummyDynamicBatchDataset(reqs, dataset_splits=num_splits)
    retrieved_reqs = []
    for _ in ds.splits_start_end_iterator():
        for i in range(len(ds)):
            retrieved_reqs.append(ds[i].tokenized_context)
    assert len(retrieved_reqs) == total_size and len(set(retrieved_reqs)) == total_size

