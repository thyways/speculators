from types import SimpleNamespace

import pytest
import torch
from torch import nn

from speculators.train.optimizers import build_optimizers


class _ToyBridgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.layer = nn.Module()
        self.layer.kv_bridge = nn.Linear(4, 4)


def _config(**kwargs):
    values = {
        "optimizer": "adamw",
        "lr": 6e-4,
        "kv_bridge_lr": 6e-5,
        "weight_decay": 0.01,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_adamw_uses_independent_kv_bridge_parameter_group():
    model = _ToyBridgeModel()
    [optimizer] = build_optimizers(model, _config())

    assert [group["name"] for group in optimizer.param_groups] == [
        "base",
        "kv_bridge",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [6e-4, 6e-5]
    )
    bridge_params = set(model.layer.kv_bridge.parameters())
    assert set(optimizer.param_groups[1]["params"]) == bridge_params
    assert not set(optimizer.param_groups[0]["params"]) & bridge_params


def test_scheduler_preserves_base_to_bridge_lr_ratio():
    model = _ToyBridgeModel()
    [optimizer] = build_optimizers(model, _config())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.5)

    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()

    base_lr, bridge_lr = [group["lr"] for group in optimizer.param_groups]
    assert base_lr == pytest.approx(3e-4)
    assert bridge_lr == pytest.approx(3e-5)
    assert bridge_lr / base_lr == pytest.approx(0.1)


def test_kv_bridge_lr_rejects_a_model_without_bridge_parameters():
    with pytest.raises(ValueError, match="no trainable '.kv_bridge.' parameters"):
        build_optimizers(nn.Linear(4, 4), _config())
