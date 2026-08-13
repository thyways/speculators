"""Tests for the trainer's training-progress hook.

Per-forward kwargs are fixed for a whole run, so schedules that depend on
training progress (Domino's decaying base-loss weight) are driven by
``SpeculatorModel.on_training_step``, which the trainer calls each step.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.model import SpeculatorModel
from speculators.models.domino import DominoDraftModel, DominoSpeculatorConfig
from speculators.train.trainer import Trainer, TrainerConfig


def _domino_model() -> DominoDraftModel:
    transformer_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
    )
    return DominoDraftModel(
        DominoSpeculatorConfig(
            transformer_layer_config=transformer_config,
            draft_vocab_size=64,
            block_size=4,
            aux_hidden_state_layer_ids=[0],
            gru_hidden_dim=8,
            logits_correction_emb_dim=4,
            pure_draft_prefix_len=1,
        )
    )


def _horizon(**overrides) -> int | None:
    config = TrainerConfig(
        lr=1e-4,
        num_epochs=overrides.pop("num_epochs", 4),
        save_path="/tmp/unused",
        **overrides,
    )
    steps_per_epoch = 25
    return Trainer._training_step_horizon(
        SimpleNamespace(config=config),  # type: ignore[arg-type]
        steps_per_epoch,
    )


def test_horizon_defaults_to_epochs_times_steps():
    assert _horizon(num_epochs=4) == 100


def test_horizon_prefers_the_explicit_scheduler_total():
    """One flag should set the horizon for both the LR schedule and lambda."""
    assert _horizon(num_epochs=4, scheduler_total_steps=60) == 60


def test_horizon_is_clamped_by_max_steps():
    """A run cut short must compress the schedule, not stretch past the run."""
    assert _horizon(num_epochs=4, max_steps=30) == 30
    assert _horizon(num_epochs=4, scheduler_total_steps=60, max_steps=30) == 30
    assert _horizon(num_epochs=4, max_steps=500) == 100


def test_horizon_is_none_when_there_is_nothing_to_schedule_over():
    assert _horizon(num_epochs=0) is None


def test_base_hook_is_a_no_op():
    """Models that ignore training progress must not need to implement it."""
    SpeculatorModel.on_training_step(object(), 5, 100)  # type: ignore[arg-type]


def test_domino_hook_updates_the_schedule_buffer_in_place():
    """In-place tensor updates avoid a compile guard on a Python float."""
    model = _domino_model()
    buffer = model.lambda_base

    model.on_training_step(0, 100)
    assert float(buffer) == pytest.approx(1.0)
    assert model._base_loss_active is True

    model.on_training_step(40, 100)
    assert float(buffer) == pytest.approx(0.2)

    model.on_training_step(50, 100)
    assert float(buffer) == pytest.approx(0.0)
    assert model._base_loss_active is False

    # Same tensor object throughout, and float32 so bf16 casts elsewhere in the
    # pipeline cannot quantize the schedule.
    assert model.lambda_base is buffer
    assert buffer.dtype == torch.float32


def test_schedule_survives_a_module_wide_dtype_cast():
    """`Module.to(dtype)` casts float buffers, which would quantize lambda.

    bfloat16 has ~1/256 resolution near 1.0, enough to freeze a long decay
    schedule for its first dozens of steps.
    """
    model = _domino_model()
    model.to(torch.bfloat16)  # type: ignore[arg-type]
    assert (
        model.lambda_base.dtype == torch.bfloat16
    )  # the cast really happened

    model.on_training_step(1, 2000)

    assert model.lambda_base.dtype == torch.float32
    assert float(model.lambda_base) == pytest.approx(0.999)


def test_schedule_state_is_not_persisted():
    model = _domino_model()

    assert "lambda_base" not in model.state_dict()
