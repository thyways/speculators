import pickle
from types import SimpleNamespace

from speculators.train.dataloader import _make_worker_dataset


def test_worker_dataset_drops_large_sampler_only_lengths_without_mutating_source():
    dataset = SimpleNamespace(
        approx_lengths=list(range(10_000)),
        marker="preserved",
    )

    worker_dataset = _make_worker_dataset(dataset)

    assert worker_dataset is not dataset
    assert len(dataset.approx_lengths) == 10_000
    assert worker_dataset.approx_lengths == []
    assert worker_dataset.marker == "preserved"
    assert len(pickle.dumps(worker_dataset)) < len(pickle.dumps(dataset)) / 10
