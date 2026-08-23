from types import SimpleNamespace

import pytest
import torch
from torch import nn

from speculators.train.optimizers import build_optimizers
from speculators.train.trainer import TrainerConfig


class _ToyLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.head = nn.Linear(4, 4)


class _ToyGatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_matrix = nn.Parameter(torch.zeros(4, 4))
        self.norm_weight = nn.Parameter(torch.ones(4))
        self.scalar_gate = nn.Parameter(torch.zeros(()))


def _config(**kwargs):
    values = {
        "optimizer": "adamw",
        "lr": 6e-4,
        "weight_decay": 0.01,
        "weight_decay_exclude_1d": False,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_trainer_config_carries_the_weight_decay_exclusion_flag():
    """build_optimizers reads this off TrainerConfig, not the CLI config.

    TrainerConfig is a NamedTuple with an explicit field list, so a flag that
    exists on the CLI but not here reaches the optimizer as its default and
    silently does nothing.
    """
    assert "weight_decay_exclude_1d" in TrainerConfig._fields
    config = TrainerConfig(
        lr=6e-4, num_epochs=1, save_path="x", weight_decay_exclude_1d=True
    )
    [optimizer] = build_optimizers(_ToyGatedModel(), config)
    assert [group["name"] for group in optimizer.param_groups] == [
        "base",
        "base_no_decay",
    ]
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.0)


def test_weight_decay_exclusion_is_off_by_default():
    model = _ToyGatedModel()
    [optimizer] = build_optimizers(model, _config(weight_decay_exclude_1d=False))

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)


def test_weight_decay_exclusion_puts_norms_and_scalar_gates_in_a_zero_group():
    model = _ToyGatedModel()
    [optimizer] = build_optimizers(model, _config(weight_decay_exclude_1d=True))

    assert [group["name"] for group in optimizer.param_groups] == [
        "base",
        "base_no_decay",
    ]
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == pytest.approx(0.01)
    assert no_decay["weight_decay"] == pytest.approx(0.0)
    assert decay["param_names"] == ["weight_matrix"]
    assert sorted(no_decay["param_names"]) == [
        "norm_weight",
        "scalar_gate",
    ]
    # Same LR either way -- this is a regularization split, not an LR split.
    assert decay["lr"] == pytest.approx(no_decay["lr"])


def test_weight_decay_exclusion_routes_every_bias_to_the_no_decay_group():
    model = _ToyLinearModel()
    [optimizer] = build_optimizers(model, _config(weight_decay_exclude_1d=True))

    by_name = {group["name"]: group["param_names"] for group in optimizer.param_groups}
    lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert lrs["base"] == pytest.approx(lrs["base_no_decay"]) == pytest.approx(6e-4)
    # Every bias is 1D, so the no-decay group holds exactly the biases here.
    assert sorted(by_name["base_no_decay"]) == ["base.bias", "head.bias"]
    assert sorted(by_name["base"]) == ["base.weight", "head.weight"]
