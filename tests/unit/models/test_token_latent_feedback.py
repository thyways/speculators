"""Tests for the block-parallel token-latent feedback head."""

import pytest
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.token_latent_feedback import (
    TokenLatentFeedbackDraftModel,
    TokenLatentFeedbackHead,
    TokenLatentFeedbackSpeculatorConfig,
    build_causal_toeplitz_matrix,
)
from speculators.proposals import GreedyTokenProposalConfig


def _config(**overrides) -> TokenLatentFeedbackSpeculatorConfig:
    transformer = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    values = {
        "transformer_layer_config": transformer,
        "draft_vocab_size": 64,
        "block_size": 4,
        "aux_hidden_state_layer_ids": [0, 1],
        "mask_token_id": 0,
        "speculators_config": SpeculatorsConfig(
            algorithm="token_latent_feedback",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
    }
    values.update(overrides)
    return TokenLatentFeedbackSpeculatorConfig(**values)


def test_config_defaults_match_v12_design():
    config = _config()
    assert config.latent_dim == 128
    assert config.feedback_stages == 1
    assert config.resolved_prefix_mixer_mode == "full"
    assert config.sample_from_anchor is False
    assert config.feedback_output_projection_init_mode == "constant"
    assert config.position_scale_parameterization == "direct"
    assert config.prefix_latent_loss_alpha == 0.0


def test_config_rejects_anchor_sampling():
    def build_invalid():
        base = _config().model_dump()
        base["architectures"] = ["TokenLatentFeedbackDraftModel"]
        base["sample_from_anchor"] = True
        return TokenLatentFeedbackSpeculatorConfig(**base)

    with pytest.raises(ValueError, match="sample_from_anchor=False"):
        build_invalid()


def test_prefix_mixer_does_not_leak_future_slots():
    torch.manual_seed(0)
    head = TokenLatentFeedbackHead(
        hidden_size=8,
        latent_dim=4,
        block_size=4,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
    )
    hidden = torch.randn(2, 4, 8)
    changed = hidden.clone()
    changed[:, 3] += 100.0
    first = head(hidden).prefix_latents
    second = head(changed).prefix_latents
    torch.testing.assert_close(first[:, :3], second[:, :3])


def test_toeplitz_matrix_has_only_strict_prefix_entries():
    matrix = build_causal_toeplitz_matrix(torch.tensor([1.0, 2.0, 3.0]))
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    torch.testing.assert_close(matrix, expected)


def test_zero_feedback_projection_preserves_hidden_and_has_gradients():
    torch.manual_seed(0)
    head = TokenLatentFeedbackHead(
        hidden_size=8,
        latent_dim=4,
        block_size=4,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
    )
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    output = head(hidden)
    torch.testing.assert_close(output.hidden_states, hidden)
    output.hidden_states.square().mean().backward()
    assert head.feedback_up_proj.weight.grad is not None


def test_v13_head_has_nonzero_feedback_and_positive_scale_floor():
    torch.manual_seed(0)
    head = TokenLatentFeedbackHead(
        hidden_size=8,
        latent_dim=4,
        block_size=4,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_reliability_gate=False,
        position_scale_init=0.05,
        position_scale_parameterization="softplus_floor",
        position_scale_min=0.02,
        feedback_output_projection_init_mode="xavier_normal",
    )
    assert head.intent_gate_proj.out_features == 4
    assert torch.count_nonzero(head.feedback_up_proj.weight) > 0
    torch.testing.assert_close(
        head.effective_position_scale(),
        torch.tensor([0.0, 0.05, 0.05, 0.05]),
    )
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    output = head(hidden)
    assert not torch.equal(output.hidden_states, hidden)
    output.hidden_states.square().mean().backward()
    assert head.intent_gate_proj.weight.grad is not None
    assert head.feedback_up_proj.weight.grad is not None
    assert head.toeplitz_coeff.grad is not None
    assert head.position_scale.grad is not None


def test_v13_config_requires_scale_init_above_floor():
    with pytest.raises(ValueError, match="position_scale_init > position_scale_min"):
        _config(
            position_scale_init=0.02,
            position_scale_parameterization="softplus_floor",
            position_scale_min=0.02,
        )


def test_target_projection_follows_the_global_training_seed():
    torch.manual_seed(19)
    first = TokenLatentFeedbackDraftModel(_config())
    torch.manual_seed(19)
    second = TokenLatentFeedbackDraftModel(_config())
    torch.testing.assert_close(
        first._target_code_projection,
        second._target_code_projection,
    )


def test_model_forward_uses_ce_tv_and_backpropagates():
    torch.manual_seed(0)
    model = TokenLatentFeedbackDraftModel(
        _config(
            use_reliability_gate=False,
            feedback_output_projection_init_mode="xavier_normal",
            position_scale_init=0.05,
            position_scale_parameterization="softplus_floor",
            position_scale_min=0.02,
            source_latent_loss_alpha=0.05,
            prefix_latent_loss_alpha=0.1,
        )
    )
    nn.init.normal_(model.embed_tokens.weight)
    nn.init.normal_(model.lm_head.weight)
    nn.init.normal_(model.verifier_lm_head.weight)
    nn.init.ones_(model.verifier_norm.weight)
    sequence_length = 32
    batch = {
        "hidden_states": torch.randn(1, sequence_length, 32),
        "input_ids": torch.randint(0, 64, (1, sequence_length)),
        "loss_mask": torch.ones(1, sequence_length),
        "verifier_last_hidden_states": torch.randn(1, sequence_length, 16),
        "document_ids": torch.zeros(1, sequence_length, dtype=torch.long),
    }
    # Calling the original callable avoids compiling a CPU test through Triton.
    forward = getattr(model.forward, "_torchdynamo_orig_callable", model.forward)
    _, loss, metrics = forward(
        model,
        **batch,
        max_anchors=4,
        loss_config=resolve_loss_config('{"ce": 0.1, "tv": 0.9}', "eager"),
    )
    assert torch.isfinite(loss)
    assert "tv_loss_sum" in metrics
    assert "source_latent_loss_sum" in metrics
    assert "prefix_latent_loss_sum" in metrics
    loss.backward()
    assert model.token_latent_head.intent_gate_proj.weight.grad is not None
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused == []
