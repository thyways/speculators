"""Trainer tests for within-sequence forward/backward accumulation."""

from types import SimpleNamespace
from typing import cast

import torch

from speculators.model import SpeculatorModel
from speculators.train.trainer import Trainer, TrainerConfig, _StepTimer


class _PartitionedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.grads_seen_before_forward: list[float | None] = []

    def forward(self, input_ids, factor, **_kwargs):
        del input_ids
        grad = self.weight.grad
        self.grads_seen_before_forward.append(
            None if grad is None else float(grad.item())
        )
        loss = self.weight * factor
        metrics = {
            "loss_sum": factor.detach().clone(),
            "loss_total": torch.ones_like(factor),
        }
        return None, loss, metrics


def test_each_partition_backwards_before_the_next_forward():
    model = _PartitionedModel()
    trainer = Trainer.__new__(Trainer)
    trainer.model = cast("SpeculatorModel", model)
    trainer.device_type = "cpu"
    trainer.config = cast(
        "TrainerConfig", SimpleNamespace(hidden_states_dtype=torch.bfloat16)
    )

    partitions = [
        {"factor": torch.tensor(value, dtype=torch.float32)}
        for value in (1.0, 2.0, 3.0)
    ]
    loss, metrics = Trainer._partitioned_forward_backward(
        trainer,
        {"input_ids": torch.ones(1, 1, dtype=torch.long)},
        {},
        partitions,
        _StepTimer(enabled=False),
    )

    assert model.grads_seen_before_forward == [None, 1.0, 3.0]
    assert model.weight.grad is not None
    assert model.weight.grad.item() == 6.0
    assert loss.item() == 6.0
    assert metrics["loss_sum"].item() == 6.0
    assert metrics["loss_total"].item() == 3.0
