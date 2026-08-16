import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorModel, SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.kv_native_dflash.config import (
    KVNativeDFlashSpeculatorConfig,
)
from speculators.models.kv_native_dflash.core import KVNativeDFlashDraftModel
from speculators.models.kv_native_dflash.model_definitions import (
    DualStreamAttentionMasks,
    KVSpaceAdapter,
    VerifierRotaryEmbedding,
    context_scale_from_logit,
    dual_stream_raw_kv_attention_forward,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _tiny_kv_native_dflash(**config_overrides) -> KVNativeDFlashDraftModel:
    transformer_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        layer_types=["full_attention", "full_attention"],
        use_sliding_window=False,
        sliding_window=8,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5,
        },
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = KVNativeDFlashSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=64,
        block_size=2,
        aux_hidden_state_layer_ids=[],
        mask_token_id=0,
        verifier_kv_layer_ids=[1, 3],
        verifier_kv_layer_mapping=[1, 3],
        verifier_num_key_value_heads=1,
        verifier_head_dim=8,
        verifier_partial_rotary_factor=0.5,
        verifier_rope_theta=10_000.0,
        sample_from_anchor=False,
        num_speculative_tokens=1,
        speculators_config=SpeculatorsConfig(
            algorithm="kv_native_dflash",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
        **config_overrides,
    )
    model = KVNativeDFlashDraftModel(config)
    torch.nn.init.normal_(model.embed_tokens.weight, std=0.02)
    torch.nn.init.normal_(model.lm_head.weight, std=0.02)
    torch.nn.init.normal_(model.verifier_lm_head.weight, std=0.02)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model


def test_config_is_final_dual_stream_raw_kv_type():
    config = KVNativeDFlashSpeculatorConfig()
    assert config.speculators_model_type == "kv_native_dflash"
    assert config.kv_native_architecture == "dual_stream_raw_kv"
    assert config.verifier_kv_layer_ids == [3, 11, 19, 27, 35]
    assert config.verifier_kv_layer_mapping == [3, 11, 19, 27, 35]
    assert config.sample_from_anchor is False
    assert config.anchor_hidden_injection is False
    assert not hasattr(config, "kv_bridge_enabled")
    assert not hasattr(config, "anchor_representation_loss_weight")

    with pytest.raises(ValueError, match="sample_from_anchor=false"):
        KVNativeDFlashSpeculatorConfig(sample_from_anchor=True)


def test_training_resolves_text_rope_from_verifier_config(monkeypatch):
    target_config = SimpleNamespace(
        layer_types=["full_attention", "full_attention"],
        num_key_value_heads=1,
        head_dim=8,
        hidden_size=16,
        num_attention_heads=2,
        partial_rotary_factor=0.25,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
        },
    )
    monkeypatch.setattr(
        "speculators.models.kv_native_dflash.core.get_verifier_config",
        lambda _path: target_config,
    )
    monkeypatch.setattr(
        VerifierConfig,
        "from_pretrained",
        classmethod(
            lambda cls, name_or_path, **_kwargs: cls(
                name_or_path=name_or_path,
                architectures=["Qwen3ForCausalLM"],
            )
        ),
    )
    monkeypatch.setattr(
        KVNativeDFlashDraftModel,
        "load_verifier_weights",
        lambda _self: None,
    )
    draft_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10_000_000.0},
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    model = KVNativeDFlashDraftModel.from_training_args(
        verifier_config=draft_config,
        verifier_name_or_path="mock-verifier",
        draft_vocab_size=64,
        block_size=2,
        mask_token_id=0,
        verifier_kv_layer_ids=[0, 1],
        verifier_kv_layer_mapping=[0, 1],
        verifier_num_key_value_heads=1,
        verifier_head_dim=8,
        sample_from_anchor=False,
        num_speculative_tokens=1,
        draft_attn_impl="eager",
    )
    assert model.config.verifier_partial_rotary_factor == pytest.approx(0.25)
    assert model.config.transformer_layer_config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.25,
    }


def test_verifier_rope_rejects_multidimensional_position_ids():
    rotary = VerifierRotaryEmbedding(
        head_dim=8, partial_rotary_factor=0.5, rope_theta=10_000.0
    )
    with pytest.raises(ValueError, match=r"\[batch, tokens\]"):
        rotary(torch.empty(1, 2, 8), torch.zeros(3, 1, 2, dtype=torch.long))


@torch.no_grad()
def test_verifier_rope_is_deterministic_after_checkpoint_reload(tmp_path):
    model = _tiny_kv_native_dflash().eval()
    reference = torch.randn(1, 5, 8)
    positions = torch.tensor([[0, 1, 7, 19, 63]], dtype=torch.long)
    expected = model.verifier_rotary_emb(reference, positions)
    checkpoint = tmp_path / "ckpt"
    model.save_pretrained(checkpoint)
    loaded = SpeculatorModel.from_pretrained(checkpoint).eval()
    actual = loaded.verifier_rotary_emb(reference, positions)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


def test_adapter_is_identity_initialized_and_has_no_local_kv_maps():
    adapter = KVSpaceAdapter(num_key_value_heads=2, num_key_value_groups=2, head_dim=4)
    query = torch.randn(1, 4, 3, 4)
    output = torch.randn(1, 3, 4, 4)
    torch.testing.assert_close(adapter.adapt_query(query), query)
    torch.testing.assert_close(adapter.adapt_output(output), output)
    assert not hasattr(adapter, "local_key_map")
    assert not hasattr(adapter, "local_value_map")


def test_dual_stream_attention_calls_two_independent_softmaxes(monkeypatch):
    model = _tiny_kv_native_dflash().eval()
    layer = model.kv_native_layers[0]
    calls = []

    def fake_attention(_module, query, key, _value, _mask, **_kwargs):
        calls.append(key.shape[-2])
        fill = 1.0 if key.shape[-2] == query.shape[-2] else 2.0
        output = torch.full(
            (query.shape[0], query.shape[2], query.shape[1], query.shape[3]),
            fill,
            dtype=query.dtype,
        )
        return output, None

    monkeypatch.setattr(
        "speculators.models.kv_native_dflash.model_definitions._resolve_attention_forward",
        lambda _attn: fake_attention,
    )
    with torch.no_grad():
        layer.self_attn.o_proj.weight.copy_(torch.eye(16))
        layer.context_gate_logit.zero_()
    hidden = torch.randn(1, 4, 16)
    keys = torch.randn(1, 1, 6, 8)
    values = torch.randn_like(keys)
    cos = torch.ones(1, 10, 4)
    sin = torch.zeros_like(cos)
    output = dual_stream_raw_kv_attention_forward(
        layer.self_attn,
        layer.kv_adapter,
        layer.context_gate_logit,
        hidden,
        keys,
        values,
        (cos, sin),
        DualStreamAttentionMasks(prefix=None, local=None),
    )
    assert calls == [4, 6]
    torch.testing.assert_close(output, torch.full_like(output, 3.0))


def test_context_scale_is_bounded_and_identity_at_zero():
    assert context_scale_from_logit(torch.tensor(0.0)).item() == pytest.approx(1.0)
    assert 0.0 <= context_scale_from_logit(torch.tensor(-100.0)).item() < 1e-6
    assert 1.999 < context_scale_from_logit(torch.tensor(100.0)).item() <= 2.0


def test_dual_stream_masks_have_separate_prefix_and_local_axes():
    torch.manual_seed(0)
    model = _tiny_kv_native_dflash()
    full, _, _, _ = model._build_dual_stream_attention_masks(
        torch.ones(1, 8),
        2,
        torch.zeros(1, 8, dtype=torch.long),
        torch.device("cpu"),
    )
    assert full is not None
    assert full.prefix.shape == (1, 1, 4, 8)
    assert full.local.shape == (1, 1, 4, 4)
    assert torch.isneginf(full.local[0, 0, :2, 2:]).all()
    assert torch.isneginf(full.local[0, 0, 2:, :2]).all()


def test_forward_backward_updates_final_architecture_parameters():
    torch.manual_seed(0)
    model = _tiny_kv_native_dflash().train()
    seq_len = 8
    keys = torch.randn(1, seq_len, 2, 1, 8)
    values = torch.randn_like(keys)
    keys_before = keys.clone()
    values_before = values.clone()
    _, loss, metrics = model(
        hidden_states=torch.empty(1, seq_len, 0),
        input_ids=torch.randint(0, 64, (1, seq_len)),
        loss_mask=torch.ones(1, seq_len),
        verifier_last_hidden_states=torch.randn(1, seq_len, 16),
        document_ids=torch.zeros(1, seq_len, dtype=torch.long),
        verifier_keys=keys,
        verifier_values=values,
        verifier_kv_layer_ids=torch.tensor([1, 3]),
        max_anchors=2,
        loss_config=resolve_loss_config("ce", "eager"),
        per_position_loss_weight="dpace",
    )
    assert torch.isfinite(loss)
    loss.backward()
    layer = model.kv_native_layers[0]
    assert layer.self_attn.q_proj.weight.grad is not None
    assert layer.kv_adapter.query_map.grad is not None
    assert layer.kv_adapter.output_map.grad is not None
    assert layer.context_gate_logit.grad is not None
    assert model.horizon_embedding.weight.grad is not None
    torch.testing.assert_close(keys, keys_before)
    torch.testing.assert_close(values, values_before)
    assert "loss_sum" in metrics
    assert "context_scale_layer_0_sum" in metrics
    assert "anchor_representation_loss_sum" not in metrics


def _write_fake_verifier_attention(
    directory,
    layer_ids,
    *,
    num_heads=2,
    head_dim=8,
    hidden_size=16,
    gated=True,
):
    """Write a checkpoint holding only the mapped layers' query path."""
    tensors = {}
    for layer_id in layer_ids:
        rows = num_heads * head_dim * (2 if gated else 1)
        prefix = f"model.language_model.layers.{layer_id}.self_attn"
        tensors[f"{prefix}.q_proj.weight"] = (
            torch.arange(rows * hidden_size, dtype=torch.float32).reshape(
                rows, hidden_size
            )
            + layer_id
        )
        tensors[f"{prefix}.q_norm.weight"] = torch.full((head_dim,), layer_id + 0.5)
    save_file(tensors, str(directory / "model.safetensors"))
    return directory


def test_verifier_query_projection_extracts_the_gated_query_half():
    num_heads, head_dim, hidden_size = 2, 8, 16
    rows = num_heads * head_dim * 2
    weight = torch.arange(rows * hidden_size, dtype=torch.float32).reshape(
        rows, hidden_size
    )

    query = KVNativeDFlashDraftModel._verifier_query_projection(
        weight, num_heads, head_dim
    )

    assert query is not None
    assert query.shape == (num_heads * head_dim, hidden_size)
    # Gated q_proj packs [query | gate] inside each head's slice, so the query
    # rows are the first head_dim rows of every 2*head_dim block -- not the
    # leading half of the matrix.
    torch.testing.assert_close(query[:head_dim], weight[:head_dim])
    torch.testing.assert_close(query[head_dim:], weight[2 * head_dim : 3 * head_dim])


def test_verifier_query_projection_passes_through_ungated_and_rejects_junk():
    plain = torch.zeros(16, 16)
    assert KVNativeDFlashDraftModel._verifier_query_projection(plain, 2, 8) is plain
    assert KVNativeDFlashDraftModel._verifier_query_projection(plain, 2, 7) is None
    assert (
        KVNativeDFlashDraftModel._verifier_query_projection(torch.zeros(16), 2, 8)
        is None
    )


def test_warm_start_copies_the_mapped_verifier_query_path(tmp_path):
    model = _tiny_kv_native_dflash()
    mapping = list(model.config.verifier_kv_layer_mapping)
    _write_fake_verifier_attention(tmp_path, mapping)

    model.warm_start_context_queries(str(tmp_path))

    for layer_index, verifier_layer in enumerate(mapping):
        attn = model.kv_native_layers[layer_index].self_attn
        source = (
            torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16) + verifier_layer
        )
        expected = source.reshape(2, 16, 16)[:, :8].reshape(16, 16)
        torch.testing.assert_close(attn.q_proj.weight, expected)
        torch.testing.assert_close(
            attn.q_norm.weight, torch.full((8,), verifier_layer + 0.5)
        )


def test_warm_start_leaves_the_query_path_alone_when_the_verifier_is_unreadable(
    tmp_path,
):
    model = _tiny_kv_native_dflash()
    before = [layer.self_attn.q_proj.weight.clone() for layer in model.kv_native_layers]

    model.warm_start_context_queries(str(tmp_path / "does-not-exist"))

    for layer, original in zip(model.kv_native_layers, before, strict=True):
        torch.testing.assert_close(layer.self_attn.q_proj.weight, original)


def test_anchor_hidden_injection_starts_as_an_exact_no_op():
    model = _tiny_kv_native_dflash(anchor_hidden_injection=True)
    assert bool(torch.all(model.anchor_hidden_proj.weight == 0))

    block_embeddings = torch.randn(1, 4, 16)
    injected = model._inject_anchor_hidden(
        block_embeddings,
        torch.randn(1, 8, 16),
        torch.tensor([3, 5]),
    )

    torch.testing.assert_close(injected, block_embeddings)


def test_anchor_hidden_injection_reads_the_last_verified_position():
    model = _tiny_kv_native_dflash(anchor_hidden_injection=True)
    with torch.no_grad():
        torch.nn.init.eye_(model.anchor_hidden_proj.weight)
    verifier_hidden = torch.randn(1, 8, 16)
    anchor_positions = torch.tensor([3, 5])

    injected = model._inject_anchor_hidden(
        torch.zeros(1, 4, 16),
        verifier_hidden,
        anchor_positions,
    )

    # The anchor is the bonus token, so the state comes from anchor - 1.
    expected = model.anchor_hidden_norm(verifier_hidden[:, anchor_positions - 1])
    torch.testing.assert_close(injected[:, ::2], expected)
    assert bool(torch.all(injected[:, 1::2] == 0))


def test_anchor_hidden_injection_receives_gradient_from_the_first_step():
    torch.manual_seed(0)
    model = _tiny_kv_native_dflash(anchor_hidden_injection=True).train()
    seq_len = 8
    _, loss, _ = model(
        hidden_states=torch.empty(1, seq_len, 0),
        input_ids=torch.randint(0, 64, (1, seq_len)),
        loss_mask=torch.ones(1, seq_len),
        verifier_last_hidden_states=torch.randn(1, seq_len, 16),
        document_ids=torch.zeros(1, seq_len, dtype=torch.long),
        verifier_keys=torch.randn(1, seq_len, 2, 1, 8),
        verifier_values=torch.randn(1, seq_len, 2, 1, 8),
        verifier_kv_layer_ids=torch.tensor([1, 3]),
        max_anchors=2,
        loss_config=resolve_loss_config("ce", "eager"),
        per_position_loss_weight="dpace",
    )
    assert torch.isfinite(loss)
    loss.backward()

    # Zero-initialized, so the forward is a no-op, but the projection still gets
    # gradient: the anchor slot reaches the loss through the block-local stream.
    gradient = model.anchor_hidden_proj.weight.grad
    assert gradient is not None
    assert bool(torch.any(gradient != 0))


def test_model_has_no_auxiliary_hidden_projection():
    model = _tiny_kv_native_dflash()
    assert model.target_layer_ids == []
    assert not hasattr(model, "fc")
    assert not hasattr(model, "hidden_norm")


@torch.no_grad()
def test_teacher_targets_use_dflash_anchor_shift():
    torch.manual_seed(0)
    model = _tiny_kv_native_dflash().eval()
    verifier_hidden = torch.randn(1, 8, 16)
    anchored_indices = torch.tensor([2, 3, 4, 5])
    actual = model._teacher_targets(verifier_hidden, anchored_indices)
    full_logits = model.verifier_lm_head(model.verifier_norm(verifier_hidden))
    expected = torch.roll(full_logits, 1, dims=1)[:, anchored_indices]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


def test_checkpoint_serializes_only_final_architecture(tmp_path):
    model = _tiny_kv_native_dflash()
    model.save_pretrained(tmp_path / "ckpt")
    saved = json.loads((tmp_path / "ckpt" / "config.json").read_text())
    assert saved["kv_native_architecture"] == "dual_stream_raw_kv"
    assert "anchor_representation_loss_weight" not in saved
    assert "kv_bridge_enabled" not in saved
    assert "kv_bridge_rank" not in saved
    state = model.state_dict()
    assert "horizon_embedding.weight" in state
    assert "layers.0.kv_adapter.query_map" in state
    assert "layers.0.kv_adapter.output_map" in state
    assert "layers.0.context_gate_logit" in state
    assert not any("kv_bridge" in name for name in state)
    loaded = SpeculatorModel.from_pretrained(tmp_path / "ckpt")
    assert isinstance(loaded, KVNativeDFlashDraftModel)
