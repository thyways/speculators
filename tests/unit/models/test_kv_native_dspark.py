import json

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.kv_native_dspark.config import (
    KVNativeDSparkSpeculatorConfig,
)
from speculators.models.kv_native_dspark.core import KVNativeDSparkDraftModel
from speculators.models.kv_native_dspark.metrics import (
    _attention_weighted_value_loss,
    _query_sensitive_key_loss,
    add_kv_native_losses,
)
from speculators.models.kv_native_dspark.model_definitions import (
    KVSpaceAdapter,
    VerifierRotaryEmbedding,
    apply_partial_rotary_pos_emb,
)


def test_partial_rotary_preserves_pass_through_dimensions():
    tensor = torch.randn(1, 2, 3, 8)
    rotary = VerifierRotaryEmbedding(
        head_dim=8,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
        mrope_section=[1, 1, 0],
    )
    cos, sin = rotary(tensor, torch.arange(3).unsqueeze(0))
    rotated = apply_partial_rotary_pos_emb(tensor, cos, sin)
    assert torch.equal(rotated[..., 4:], tensor[..., 4:])
    assert rotated.shape == tensor.shape


def test_kv_losses_are_finite_and_differentiable():
    base = torch.tensor(1.0, requires_grad=True)
    metrics = {"loss_sum": base.detach(), "loss_total": torch.tensor(1.0)}
    predicted = torch.zeros(1, 4, 2, 1, 2, requires_grad=True)
    teacher = torch.ones_like(predicted)
    queries = torch.randn(1, 4, 2, 2, 2, requires_grad=True)
    loss, output_metrics = add_kv_native_losses(
        base,
        metrics,
        predicted_keys=predicted,
        predicted_values=predicted,
        teacher_keys=teacher,
        teacher_values=teacher,
        queries=queries,
        loss_mask=torch.ones(1, 4),
        block_size=2,
        local_kv_loss_alpha=1.0,
        query_key_loss_alpha=1.0,
        attention_value_loss_alpha=1.0,
    )
    assert torch.isfinite(loss)
    assert "local_kv_loss_sum" in output_metrics
    loss.backward()
    assert predicted.grad is not None


def test_attention_weighted_value_loss_follows_teacher_attention():
    teacher_keys = torch.tensor([[[[[4.0, 0.0]]], [[[0.0, 4.0]]]]])
    queries = torch.tensor([[[[[4.0, 0.0]]], [[[4.0, 0.0]]]]])
    teacher_values = torch.zeros(1, 2, 1, 1, 2)
    error_on_attended_value = teacher_values.clone()
    error_on_attended_value[:, 0] = 1.0
    error_on_ignored_value = teacher_values.clone()
    error_on_ignored_value[:, 1] = 1.0
    loss_mask = torch.ones(1, 2)

    attended_loss = _attention_weighted_value_loss(
        teacher_keys=teacher_keys,
        predicted_values=error_on_attended_value,
        teacher_values=teacher_values,
        queries=queries,
        loss_mask=loss_mask,
        block_size=2,
    )
    ignored_loss = _attention_weighted_value_loss(
        teacher_keys=teacher_keys,
        predicted_values=error_on_ignored_value,
        teacher_values=teacher_values,
        queries=queries,
        loss_mask=loss_mask,
        block_size=2,
    )

    assert attended_loss > 1000 * ignored_loss


def test_query_sensitive_key_loss_uses_cross_position_scores():
    teacher_keys = torch.zeros(1, 2, 1, 1, 2)
    predicted_keys = teacher_keys.clone()
    predicted_keys[:, 1, 0, 0, 0] = 1.0
    queries = torch.zeros(1, 2, 1, 1, 2)
    queries[:, 0, 0, 0, 0] = 1.0

    loss = _query_sensitive_key_loss(
        predicted_keys,
        teacher_keys,
        queries,
        torch.ones(1, 2),
        block_size=2,
    )

    assert loss > 0


def test_kv_adapter_starts_as_identity():
    adapter = KVSpaceAdapter(
        num_key_value_heads=2,
        num_key_value_groups=2,
        head_dim=4,
    )
    query = torch.randn(1, 4, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    output = torch.randn(1, 3, 4, 4)
    assert torch.equal(adapter.adapt_query(query), query)
    assert torch.equal(adapter.adapt_local_key(key), key)
    assert torch.equal(adapter.adapt_local_value(key), key)
    assert torch.equal(adapter.adapt_output(output), output)


def _tiny_kv_native_model(
    sample_from_anchor: bool = True,
    mapping: list[int] | None = None,
) -> KVNativeDSparkDraftModel:
    transformer_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=8,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 1.0,
        },
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = KVNativeDSparkSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=64,
        block_size=2,
        aux_hidden_state_layer_ids=[],
        mask_token_id=0,
        markov_rank=0,
        enable_confidence_head=False,
        confidence_head_with_markov=False,
        verifier_kv_layer_ids=[1, 3],
        verifier_kv_layer_mapping=[1, 3] if mapping is None else mapping,
        verifier_num_key_value_heads=1,
        verifier_head_dim=8,
        verifier_partial_rotary_factor=0.5,
        verifier_rope_theta=10_000.0,
        verifier_mrope_section=[1, 1, 0],
        sample_from_anchor=sample_from_anchor,
        num_speculative_tokens=2 if sample_from_anchor else 1,
    )
    model = KVNativeDSparkDraftModel(config)
    torch.nn.init.normal_(model.embed_tokens.weight, std=0.02)
    torch.nn.init.normal_(model.lm_head.weight, std=0.02)
    torch.nn.init.normal_(model.verifier_lm_head.weight, std=0.02)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model


def test_tiny_kv_native_forward_backward_smoke():
    torch.manual_seed(0)
    model = _tiny_kv_native_model().train()
    seq_len = 8
    _, loss, metrics = model(
        hidden_states=torch.empty(1, seq_len, 0),
        input_ids=torch.randint(0, 64, (1, seq_len)),
        loss_mask=torch.ones(1, seq_len),
        verifier_last_hidden_states=torch.randn(1, seq_len, 16),
        document_ids=torch.zeros(1, seq_len, dtype=torch.long),
        verifier_keys=torch.randn(1, seq_len, 2, 1, 8),
        verifier_values=torch.randn(1, seq_len, 2, 1, 8),
        verifier_kv_layer_ids=torch.tensor([1, 3]),
        max_anchors=2,
        local_kv_loss_alpha=0.1,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.layers[0].self_attn.q_proj.weight.grad is not None
    assert model.kv_native_layers[0].kv_adapter.query_map.grad is not None
    assert all(torch.isfinite(value) for value in metrics.values())


def test_from_scratch_model_has_no_hidden_context_projection():
    model = _tiny_kv_native_model()
    assert model.target_layer_ids == []
    assert not hasattr(model, "fc")
    assert not hasattr(model, "hidden_norm")
    assert all("kv_adapter.gate" not in name for name, _ in model.named_parameters())


@torch.no_grad()
def test_targets_follow_sample_from_anchor_shift():
    torch.manual_seed(0)
    verifier_hidden = torch.randn(1, 8, 16)
    anchored_indices = torch.tensor([2, 3, 4, 5])

    for sample_from_anchor in (False, True):
        model = _tiny_kv_native_model(sample_from_anchor).eval()
        actual = model._teacher_targets(verifier_hidden, anchored_indices)

        full_logits = model.verifier_lm_head(model.verifier_norm(verifier_hidden))
        if not sample_from_anchor:
            full_logits = torch.roll(full_logits, 1, dims=1)
        expected = full_logits[:, anchored_indices]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


def test_mapping_resolves_exported_layer_positions():
    model = _tiny_kv_native_model(mapping=[3, 1])
    assert model._verifier_kv_mapping_indices == (1, 0)

    with pytest.raises(ValueError, match="non-exported layers"):
        _tiny_kv_native_model(mapping=[1, 7])
    with pytest.raises(ValueError, match="one layer ID per draft layer"):
        _tiny_kv_native_model(mapping=[1])


def test_config_class_is_default_constructible():
    config = KVNativeDSparkSpeculatorConfig()
    assert config.speculators_model_type == "kv_native_dspark"
    assert config.aux_hidden_state_layer_ids == []
    assert set(config.verifier_kv_layer_mapping) <= set(config.verifier_kv_layer_ids)


def test_kv_native_checkpoint_contains_only_pure_kv_architecture(tmp_path):
    model = _tiny_kv_native_model(mapping=[3, 1])
    model.save_pretrained(tmp_path / "ckpt")

    saved = json.loads((tmp_path / "ckpt" / "config.json").read_text())
    assert saved["speculators_model_type"] == "kv_native_dspark"
    assert saved["verifier_kv_layer_mapping"] == [3, 1]
    assert saved["aux_hidden_state_layer_ids"] == []
    assert "context_source" not in saved
