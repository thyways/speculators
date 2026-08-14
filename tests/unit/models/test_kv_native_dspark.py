import json
import math

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.kv_native_dspark.config import (
    KVNativeDSparkSpeculatorConfig,
)
from speculators.models.kv_native_dspark.core import KVNativeDSparkDraftModel
from speculators.models.kv_native_dspark.model_definitions import (
    KVSpaceAdapter,
    TargetToDraftKVBridge,
    VerifierRotaryEmbedding,
    apply_partial_rotary_pos_emb,
    remove_partial_rotary_pos_emb,
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


def test_partial_rotary_round_trip():
    tensor = torch.randn(2, 3, 5, 8)
    rotary = VerifierRotaryEmbedding(
        head_dim=8,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
        mrope_section=[1, 1, 0],
    )
    cos, sin = rotary(tensor, torch.arange(5).expand(2, -1))
    rotated = apply_partial_rotary_pos_emb(tensor, cos, sin)
    restored = remove_partial_rotary_pos_emb(rotated, cos, sin)
    torch.testing.assert_close(restored, tensor, atol=1e-5, rtol=1e-5)


def test_target_to_draft_bridge_starts_from_uniform_all_layer_fusion():
    bridge = TargetToDraftKVBridge(
        num_source_layers=2,
        num_key_value_heads=2,
        head_dim=8,
        rank=4,
    )
    keys = torch.randn(1, 2, 2, 5, 8)
    values = torch.randn_like(keys)
    rotary = VerifierRotaryEmbedding(
        head_dim=8,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
        mrope_section=[1, 1, 0],
    )
    cos, sin = rotary(keys[:, 0], torch.arange(5).unsqueeze(0))
    bridge_output = bridge(keys, values, (cos, sin))
    mapped_keys = bridge_output.keys
    mapped_values = bridge_output.values
    torch.testing.assert_close(mapped_keys, keys.mean(dim=1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mapped_values, values.mean(dim=1))
    key_weights, value_weights = bridge.fusion_weights()
    expected_weights = torch.full((2, 2), 0.5)
    torch.testing.assert_close(key_weights, expected_weights)
    torch.testing.assert_close(value_weights, expected_weights)
    assert bridge_output.key_correction_ratio == 0
    assert bridge_output.value_correction_ratio == 0
    assert bridge_output.key_gate_entropy.item() == pytest.approx(math.log(2))
    assert bridge_output.value_gate_entropy.item() == pytest.approx(math.log(2))
    assert not bridge_output.key_correction_ratio.requires_grad
    assert not bridge_output.value_correction_ratio.requires_grad
    assert not bridge_output.key_gate_entropy.requires_grad
    assert not bridge_output.value_gate_entropy.requires_grad


def test_target_to_draft_bridge_scales_only_the_low_rank_correction():
    torch.manual_seed(0)
    full_scale = TargetToDraftKVBridge(
        num_source_layers=2,
        num_key_value_heads=1,
        head_dim=8,
        rank=4,
        residual_scale=1.0,
    )
    tenth_scale = TargetToDraftKVBridge(
        num_source_layers=2,
        num_key_value_heads=1,
        head_dim=8,
        rank=4,
        residual_scale=0.1,
    )
    tenth_scale.load_state_dict(full_scale.state_dict())
    torch.nn.init.normal_(full_scale.key_mappers[0].down.weight)
    torch.nn.init.normal_(full_scale.value_mappers[0].down.weight)
    tenth_scale.load_state_dict(full_scale.state_dict())

    keys = torch.randn(1, 2, 1, 5, 8)
    values = torch.randn_like(keys)
    rotary = VerifierRotaryEmbedding(
        head_dim=8,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
        mrope_section=[1, 1, 0],
    )
    cos, sin = rotary(keys[:, 0], torch.arange(5).unsqueeze(0))
    full_output = full_scale(keys, values, (cos, sin))
    tenth_output = tenth_scale(keys, values, (cos, sin))
    key_base = keys.mean(dim=1)
    value_base = values.mean(dim=1)

    torch.testing.assert_close(
        tenth_output.keys - key_base,
        0.1 * (full_output.keys - key_base),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        tenth_output.values - value_base,
        0.1 * (full_output.values - value_base),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        tenth_output.key_correction_ratio,
        0.1 * full_output.key_correction_ratio,
    )
    torch.testing.assert_close(
        tenth_output.value_correction_ratio,
        0.1 * full_output.value_correction_ratio,
    )


def test_target_to_draft_bridge_caps_correction_ratio_and_keeps_gradients():
    torch.manual_seed(0)
    bridge = TargetToDraftKVBridge(
        num_source_layers=2,
        num_key_value_heads=1,
        head_dim=8,
        rank=4,
        max_correction_ratio=0.25,
    )
    torch.nn.init.normal_(bridge.key_mappers[0].down.weight, std=10.0)
    torch.nn.init.normal_(bridge.value_mappers[0].down.weight, std=10.0)
    keys = torch.randn(1, 2, 1, 5, 8)
    values = torch.randn_like(keys)
    rotary = VerifierRotaryEmbedding(
        head_dim=8,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
        mrope_section=[1, 1, 0],
    )
    cos, sin = rotary(keys[:, 0], torch.arange(5).unsqueeze(0))

    output = bridge(keys, values, (cos, sin))

    assert output.key_correction_ratio.item() <= 0.25 + 1e-5
    assert output.value_correction_ratio.item() <= 0.25 + 1e-5
    (output.keys.square().mean() + output.values.square().mean()).backward()
    assert torch.isfinite(bridge.key_mappers[0].down.weight.grad).all()
    assert torch.isfinite(bridge.value_mappers[0].down.weight.grad).all()


@torch.no_grad()
def test_target_to_draft_bridge_can_rms_normalize_mapped_keys_only():
    bridge = TargetToDraftKVBridge(
        num_source_layers=2,
        num_key_value_heads=1,
        head_dim=8,
        rank=4,
        normalize_keys=True,
    )
    keys = torch.randn(1, 2, 1, 5, 8)
    values = torch.randn_like(keys)
    rotary = VerifierRotaryEmbedding(
        head_dim=8,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
        mrope_section=[1, 1, 0],
    )
    cos, sin = rotary(keys[:, 0], torch.arange(5).unsqueeze(0))

    output = bridge(keys, values, (cos, sin))

    key_rms = output.keys.float().square().mean(dim=-1).sqrt()
    torch.testing.assert_close(key_rms, torch.ones_like(key_rms))
    torch.testing.assert_close(output.values, values.mean(dim=1))


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
    *,
    bridge: bool = False,
    bridge_max_correction_ratio: float | None = None,
    bridge_normalize_keys: bool = False,
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
        use_sliding_window=True,
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
        kv_bridge_enabled=bridge,
        kv_bridge_rank=4,
        kv_bridge_max_correction_ratio=bridge_max_correction_ratio,
        kv_bridge_normalize_keys=bridge_normalize_keys,
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
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.layers[0].self_attn.q_proj.weight.grad is not None
    assert model.kv_native_layers[0].kv_adapter.query_map.grad is not None
    assert all(torch.isfinite(value) for value in metrics.values())


def test_tiny_kv_bridge_forward_backward_smoke():
    torch.manual_seed(0)
    model = _tiny_kv_native_model(bridge=True).train()
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
    )
    assert torch.isfinite(loss)
    loss.backward()
    layer = model.kv_native_layers[0]
    assert layer.kv_adapter is None
    assert layer.kv_bridge is not None
    down = layer.kv_bridge.key_mappers[0].down
    assert down.weight.grad is not None
    gate_grad = layer.kv_bridge.key_mappers[0].gate_logits.grad
    assert gate_grad is not None
    assert torch.count_nonzero(gate_grad) > 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    for name in (
        "key_correction_ratio",
        "value_correction_ratio",
        "key_gate_entropy",
        "value_gate_entropy",
    ):
        assert f"kv_bridge_{name}_sum" in metrics
        assert f"kv_bridge_{name}_total" in metrics


def test_tiny_bounded_bridge_uses_dspark_loss():
    torch.manual_seed(0)
    model = _tiny_kv_native_model(
        bridge=True,
        bridge_max_correction_ratio=0.5,
        bridge_normalize_keys=True,
    ).train()
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
    )

    loss.backward()
    bridge = model.kv_native_layers[0].kv_bridge
    assert bridge is not None
    assert bridge.key_mappers[0].down.weight.grad is not None
    assert metrics["kv_bridge_key_correction_ratio_sum"].item() <= 0.5 + 1e-5


@torch.no_grad()
def test_bridge_cache_conversion_is_sliding_window_aware():
    model = _tiny_kv_native_model(bridge=True).eval()
    seq_len = 12
    verifier_keys = torch.randn(1, seq_len, 2, 1, 8)
    verifier_values = torch.randn_like(verifier_keys)
    caches = model.convert_verifier_kv_cache(
        verifier_keys,
        verifier_values,
    )
    assert len(caches) == 2
    assert caches[0][0].shape == (1, 1, 8, 8)
    assert caches[0][1].shape == (1, 1, 8, 8)
    assert caches[1][0].shape == (1, 1, seq_len, 8)
    assert caches[1][1].shape == (1, 1, seq_len, 8)
    expected_sliding_keys = verifier_keys[:, -8:].mean(dim=2).transpose(1, 2)
    expected_sliding_values = verifier_values[:, -8:].mean(dim=2).transpose(1, 2)
    expected_full_keys = verifier_keys.mean(dim=2).transpose(1, 2)
    expected_full_values = verifier_values.mean(dim=2).transpose(1, 2)
    torch.testing.assert_close(
        caches[0][0], expected_sliding_keys, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(caches[0][1], expected_sliding_values)
    torch.testing.assert_close(caches[1][0], expected_full_keys, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(caches[1][1], expected_full_values)

    mrope_positions = torch.arange(seq_len).expand(3, 1, -1)
    mrope_caches = model.convert_verifier_kv_cache(
        verifier_keys,
        verifier_values,
        position_ids=mrope_positions,
    )
    for cache, mrope_cache in zip(caches, mrope_caches, strict=True):
        torch.testing.assert_close(cache[0], mrope_cache[0])
        torch.testing.assert_close(cache[1], mrope_cache[1])


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


def test_bridge_uses_all_exported_layers_and_exposes_learned_weights():
    model = _tiny_kv_native_model(bridge=True, mapping=[7])
    assert model._verifier_kv_mapping_indices == ()
    assert all(
        layer.kv_bridge is not None
        and layer.kv_bridge.num_source_layers == len(model.config.verifier_kv_layer_ids)
        for layer in model.kv_native_layers
    )
    weights = model.get_kv_bridge_fusion_weights()
    assert weights["keys"].shape == (2, 1, 2)
    assert weights["values"].shape == (2, 1, 2)
    torch.testing.assert_close(
        weights["keys"].sum(dim=-1),
        torch.ones(2, 1),
    )
    torch.testing.assert_close(
        weights["values"].sum(dim=-1),
        torch.ones(2, 1),
    )


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


def test_kv_bridge_checkpoint_serializes_bridge_contract(tmp_path):
    model = _tiny_kv_native_model(bridge=True)
    model.save_pretrained(tmp_path / "ckpt")

    saved = json.loads((tmp_path / "ckpt" / "config.json").read_text())
    assert saved["kv_bridge_enabled"] is True
    assert saved["kv_bridge_rank"] == 4
    assert saved["kv_bridge_residual_scale"] == 1.0
    assert saved["kv_bridge_max_correction_ratio"] is None
    assert saved["kv_bridge_normalize_keys"] is False
    assert "kv_bridge_type" not in saved
    assert "kv_bridge_cross_head" not in saved
    assert "kv_bridge_top_k" not in saved
    assert "kv_bridge_layer_mapping" not in saved
