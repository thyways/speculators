"""vLLM config translation tests for token-latent feedback."""

import pytest
import torch

from speculators.vllm.token_latent_feedback import _update_token_latent_feedback


def _config(**overrides):
    config = {
        "speculators_model_type": "token_latent_feedback",
        "aux_hidden_state_layer_ids": [2, 11, 20, 29, 38],
        "draft_vocab_size": 128,
        "target_hidden_size": 16,
        "block_size": 8,
        "mask_token_id": 127,
        "sample_from_anchor": False,
        "latent_dim": 128,
        "prefix_mixer_mode": "full",
        "use_reliability_gate": True,
        "strict_causal_prefix": True,
    }
    config.update(overrides)
    return config


def test_translation_routes_to_parallel_dflash_architecture():
    translated = {"hidden_size": 16, "rope_parameters": {"mrope_section": [1, 1]}}
    _update_token_latent_feedback(_config(), translated)
    assert translated["architectures"] == ["Qwen3TokenLatentFeedbackModel"]
    assert translated["model_arch"] == "token_latent_feedback"
    assert translated["block_size"] == 8
    assert translated["latent_dim"] == 128
    assert translated["dflash_config"]["target_layer_ids"] == [1, 10, 19, 28, 37]
    assert "mrope_section" not in translated["rope_parameters"]


def test_translation_rejects_anchor_sampling():
    with pytest.raises(ValueError, match="sample_from_anchor=False"):
        _update_token_latent_feedback(_config(sample_from_anchor=True), {})


def test_vllm_loader_drops_training_only_projection(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.model_executor.models.qwen3_dflash import (  # noqa: PLC0415
        DFlashQwen3ForCausalLM,
    )

    from speculators.vllm.token_latent_feedback_model import (  # noqa: PLC0415
        Qwen3TokenLatentFeedbackForCausalLM,
    )

    captured = []

    def fake_loader(_self, weights):
        captured.extend(weights)
        return {name for name, _ in captured}

    monkeypatch.setattr(DFlashQwen3ForCausalLM, "load_weights", fake_loader)
    model = object.__new__(Qwen3TokenLatentFeedbackForCausalLM)
    loaded = model.load_weights(
        [
            ("_target_code_projection", torch.ones(2, 2)),
            ("model.token_latent_head.intent_gate_proj.weight", torch.ones(2, 2)),
        ]
    )
    assert loaded == {"model.token_latent_head.intent_gate_proj.weight"}
    assert [name for name, _ in captured] == [
        "model.token_latent_head.intent_gate_proj.weight"
    ]
