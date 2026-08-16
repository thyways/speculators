from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import vllm.config as vllm_config_module
from hs_connectors.verifier_kv import SelectedVerifierKV
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

import speculators.vllm.kv_native_dflash as plugin
import speculators.vllm.kv_native_dflash_model as serving_model
from speculators.models.kv_native_dflash.model_definitions import (
    KVSpaceAdapter,
    VerifierRotaryEmbedding,
    apply_partial_rotary_pos_emb,
)
from speculators.vllm._dflash_family import (
    finalize_speculative_config,
    register_speculative_config_updater,
    register_speculative_method_alias,
)
from speculators.vllm.kv_native_dflash import (
    _bind_verifier_kv_runtime,
    _disable_kv_native_aux_hidden_states,
    _finalize_kv_native_speculative_config,
    _install_v1_dflash_proposer_patch,
    _install_v2_dflash_speculator_patch,
    _snapshot_verifier_slot_mappings,
    _update_kv_native_dflash,
)
from speculators.vllm.kv_native_dflash_model import (
    Qwen3KVNativeDFlashForCausalLM,
    Qwen3KVNativeDFlashModel,
    dual_stream_attention_forward_vllm,
    extract_context_kv_from_paged_cache,
    extract_verifier_kv_from_paged_cache,
    grouped_query_attention,
)


def _source(**overrides: Any) -> dict[str, Any]:
    source: dict[str, Any] = {
        "block_size": 16,
        "draft_vocab_size": 32000,
        "kv_native_architecture": "dual_stream_raw_kv",
        "mask_token_id": 248077,
        "num_speculative_tokens": 15,
        "sample_from_anchor": False,
        "speculators_model_type": "kv_native_dflash",
        "verifier_head_dim": 256,
        "verifier_kv_layer_ids": [3, 11, 19, 27, 35],
        "verifier_kv_layer_mapping": [3, 11, 19, 27, 35],
        "verifier_num_key_value_heads": 2,
        "verifier_partial_rotary_factor": 0.25,
        "verifier_rope_theta": 10_000_000.0,
    }
    source.update(overrides)
    return source


def _translated_base() -> dict[str, Any]:
    return {
        "hidden_size": 2048,
        "layer_types": ["full_attention"] * 5,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 1.0,
        },
    }


def test_kv_native_config_translation_preserves_final_training_contract():
    translated = _translated_base()

    _update_kv_native_dflash(_source(), translated)

    assert translated["architectures"] == ["Qwen3KVNativeDFlashForCausalLM"]
    assert translated["model_arch"] == "kv_native_dflash"
    assert translated["kv_native_architecture"] == "dual_stream_raw_kv"
    assert translated["sample_from_anchor"] is False
    assert translated["eagle_aux_hidden_state_layer_ids"] == []
    assert translated["num_target_layers"] == 0
    assert translated["verifier_kv_layer_ids"] == [3, 11, 19, 27, 35]
    assert translated["verifier_kv_layer_mapping"] == [3, 11, 19, 27, 35]
    assert not any("kv_bridge" in key for key in translated)
    assert translated["dflash_config"] == {
        "mask_token_id": 248077,
        "target_layer_ids": [],
        "use_aux_hidden_state": False,
        "sample_from_anchor": False,
        "linear_position_ids": True,
        "causal": False,
    }
    assert translated["partial_rotary_factor"] == pytest.approx(0.25)
    assert translated["rope_parameters"] == {
        "rope_type": "default",
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.25,
    }


def test_kv_native_translation_rejects_legacy_or_incompatible_variants():
    legacy = _source()
    legacy.pop("kv_native_architecture")
    with pytest.raises(ValueError, match="predates the final"):
        _update_kv_native_dflash(legacy, _translated_base())

    with pytest.raises(ValueError, match="sample_from_anchor=false"):
        _update_kv_native_dflash(
            _source(sample_from_anchor=True),
            _translated_base(),
        )

    with pytest.raises(ValueError, match="one source per draft layer"):
        _update_kv_native_dflash(
            _source(verifier_kv_layer_mapping=[3, 11]),
            _translated_base(),
        )

    with pytest.raises(ValueError, match="non-exported layers"):
        _update_kv_native_dflash(
            _source(verifier_kv_layer_mapping=[3, 11, 19, 27, 999]),
            _translated_base(),
        )

    mixed = _translated_base()
    mixed["layer_types"][-1] = "sliding_attention"
    with pytest.raises(NotImplementedError, match="full_attention"):
        _update_kv_native_dflash(_source(), mixed)


def test_formal_speculative_width_matches_trained_query_block():
    register_speculative_method_alias("kv_native_dflash", "dflash")
    register_speculative_config_updater(
        "kv_native_dflash",
        _finalize_kv_native_speculative_config,
    )

    actual = finalize_speculative_config(
        _source(),
        {"method": "kv_native_dflash", "num_speculative_tokens": 15},
    )
    assert actual == {"method": "dflash", "num_speculative_tokens": 15}

    with pytest.raises(ValueError, match="block_size-1"):
        _finalize_kv_native_speculative_config(
            _source(num_speculative_tokens=14),
            {"num_speculative_tokens": 14},
        )


def test_runtime_disables_target_aux_hidden_states():
    config = SimpleNamespace(architectures=["Qwen3KVNativeDFlashForCausalLM"])
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(hf_config=config),
            ),
        ),
        use_aux_hidden_state_outputs=True,
    )

    _disable_kv_native_aux_hidden_states(runner)

    assert runner.use_aux_hidden_state_outputs is False


def test_custom_model_remains_native_dflash_proposer_compatible():
    assert issubclass(Qwen3KVNativeDFlashForCausalLM, DFlashQwen3ForCausalLM)


def test_inherited_dflash_loader_routes_final_architecture_weights():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kv_adapter = KVSpaceAdapter(
                num_key_value_heads=1,
                num_key_value_groups=2,
                head_dim=4,
            )
            self.context_gate_logit = torch.nn.Parameter(torch.zeros(()))

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])
            self.horizon_embedding = torch.nn.Embedding(3, 8)
            self.use_aux_hidden_state = False
            self.has_separate_mask_embedding = False
            self.mask_token_id = None
            self.fused_buffers_built = False

        def _build_fused_kv_buffers(self):
            self.fused_buffers_built = True

    model = Qwen3KVNativeDFlashForCausalLM.__new__(Qwen3KVNativeDFlashForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = Draft()
    model.draft_id_to_target_id = None
    model.draft_model_config = SimpleNamespace(model="unused", revision=None)

    expected_query_map = torch.randn(1, 4, 4)
    expected_output_map = torch.randn(1, 4, 4)
    expected_gate = torch.tensor(0.75)
    expected_horizon = torch.randn(3, 8)
    model.load_weights(
        [
            ("layers.0.kv_adapter.query_map", expected_query_map),
            ("layers.0.kv_adapter.output_map", expected_output_map),
            ("layers.0.context_gate_logit", expected_gate),
            ("horizon_embedding.weight", expected_horizon),
        ]
    )

    layer = model.model.layers[0]
    torch.testing.assert_close(layer.kv_adapter.query_map, expected_query_map)
    torch.testing.assert_close(layer.kv_adapter.output_map, expected_output_map)
    torch.testing.assert_close(layer.context_gate_logit, expected_gate)
    torch.testing.assert_close(model.model.horizon_embedding.weight, expected_horizon)
    assert model.model.fused_buffers_built is True


def test_runtime_binding_accepts_selected_layers_across_cache_groups(monkeypatch):
    class ModelSpy:
        def __init__(self):
            self.bound = None

        def bind_verifier_kv(self, **kwargs):
            self.bound = kwargs

    spec = SimpleNamespace(block_size=16, num_kv_heads=2, head_size=256)
    groups = [SimpleNamespace(layer_names=[], kv_cache_spec=object()) for _ in range(9)]
    names_by_group = {
        6: ("model.layers.7.self_attn", "model.layers.31.self_attn"),
        7: ("model.layers.23.self_attn", "model.layers.39.self_attn"),
        8: ("model.layers.15.self_attn",),
    }
    for group_id, names in names_by_group.items():
        groups[group_id] = SimpleNamespace(
            layer_names=list(names),
            kv_cache_spec=spec,
        )
    all_attention = {
        name: SimpleNamespace(kv_cache=torch.empty(0))
        for names in names_by_group.values()
        for name in names
    }
    monkeypatch.setattr(
        vllm_config_module,
        "get_layers_from_vllm_config",
        lambda *_args, **_kwargs: all_attention,
    )
    model = ModelSpy()
    speculator = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Qwen3KVNativeDFlashForCausalLM"],
                verifier_kv_layer_ids=[7, 15, 23, 31, 39],
            )
        ),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                use_ubatching=False,
            )
        ),
        model=model,
        max_num_tokens=32,
        device=torch.device("cpu"),
    )

    _bind_verifier_kv_runtime(
        speculator,
        SimpleNamespace(kv_cache_groups=groups),
    )

    metadata = model.bound["verifier_metadata"]
    assert metadata.layer_ids == (7, 15, 23, 31, 39)
    assert metadata.cache_group_ids == (6, 8, 7, 6, 7)
    assert model.bound["slot_mapping_capacity"] == 32
    assert model.bound["slot_mapping_device"] == torch.device("cpu")


def test_text_query_rope_matches_training_verifier_rope():
    torch.manual_seed(0)
    num_tokens = 5
    head_dim = 8
    positions = torch.tensor([0, 1, 7, 19, 63], dtype=torch.long)
    query = torch.randn(num_tokens, 4, head_dim)
    key = torch.randn(num_tokens, 2, head_dim)

    with set_current_vllm_config(VllmConfig()):
        serving_rope = get_rope(
            head_dim,
            max_position=64,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.5,
            },
            dtype=torch.float32,
        )
    serving_query, serving_key = serving_rope.forward_native(
        positions,
        query.clone(),
        key.clone(),
    )

    training_rope = VerifierRotaryEmbedding(
        head_dim=head_dim,
        partial_rotary_factor=0.5,
        rope_theta=10_000_000.0,
    )
    query_heads_first = query.permute(1, 0, 2).unsqueeze(0)
    key_heads_first = key.permute(1, 0, 2).unsqueeze(0)
    cos, sin = training_rope(key_heads_first, positions.unsqueeze(0))
    expected_query = (
        apply_partial_rotary_pos_emb(query_heads_first, cos, sin)
        .squeeze(0)
        .permute(1, 0, 2)
    )
    expected_key = (
        apply_partial_rotary_pos_emb(key_heads_first, cos, sin)
        .squeeze(0)
        .permute(1, 0, 2)
    )

    torch.testing.assert_close(serving_query, expected_query)
    torch.testing.assert_close(serving_key, expected_key)


def test_extracts_cross_group_paged_verifier_kv_in_checkpoint_order():
    block_size = 2
    num_heads = 1
    head_dim = 2
    content = 2 * head_dim
    layer_ids = (7, 15, 23, 31, 39)
    layer_names = tuple(f"model.layers.{layer_id}.self_attn" for layer_id in layer_ids)
    group_ids = (6, 8, 7, 6, 7)
    caches = {
        name: torch.arange(
            4 * num_heads * block_size * content,
            dtype=torch.float32,
        ).reshape(4, num_heads, block_size, content)
        + 10_000 * source_index
        for source_index, name in enumerate(layer_names)
    }
    slots_by_group = {
        6: torch.tensor([-1, 3, 4], dtype=torch.long),
        8: torch.tensor([2, -1, 6], dtype=torch.long),
        7: torch.tensor([1, 5, -1], dtype=torch.long),
    }

    keys, values = extract_verifier_kv_from_paged_cache(
        caches,
        layer_names,
        group_ids,
        slots_by_group,
        block_size=block_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
    )

    assert keys.shape == (3, len(layer_names), num_heads, head_dim)
    assert values.shape == keys.shape
    for source_index, (layer_name, group_id) in enumerate(
        zip(layer_names, group_ids, strict=True)
    ):
        slots = slots_by_group[group_id]
        safe_slots = slots.clamp_min(0)
        gathered = caches[layer_name][
            safe_slots // block_size,
            :,
            safe_slots % block_size,
            :,
        ]
        expected_keys, expected_values = gathered.split(head_dim, dim=-1)
        invalid = slots < 0
        expected_keys = expected_keys.masked_fill(invalid[:, None, None], 0)
        expected_values = expected_values.masked_fill(invalid[:, None, None], 0)
        torch.testing.assert_close(keys[:, source_index], expected_keys)
        torch.testing.assert_close(values[:, source_index], expected_values)


def test_extracts_context_cache_with_correct_batch_token_axis_order():
    block_size = 2
    num_heads = 1
    head_dim = 2
    cache = torch.arange(4 * num_heads * block_size * 2 * head_dim).reshape(
        4,
        num_heads,
        block_size,
        2 * head_dim,
    )
    block_table = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32)
    context_lengths = torch.tensor([3, 1], dtype=torch.int32)

    keys, values, valid = extract_context_kv_from_paged_cache(
        cache,
        block_table,
        context_lengths,
        max_context_length=3,
        num_kv_heads=num_heads,
        head_dim=head_dim,
    )

    assert keys.shape == (2, 3, num_heads, head_dim)
    assert valid.tolist() == [[True, True, True], [True, False, False]]
    for request_index in range(2):
        for position in range(3):
            if position >= int(context_lengths[request_index]):
                torch.testing.assert_close(
                    keys[request_index, position],
                    torch.zeros_like(keys[request_index, position]),
                )
                torch.testing.assert_close(
                    values[request_index, position],
                    torch.zeros_like(values[request_index, position]),
                )
                continue
            block = int(block_table[request_index, position // block_size])
            packed = cache[block, :, position % block_size]
            expected_key, expected_value = packed.split(head_dim, dim=-1)
            torch.testing.assert_close(keys[request_index, position], expected_key)
            torch.testing.assert_close(values[request_index, position], expected_value)


def test_grouped_query_attention_matches_explicit_gqa_reference():
    torch.manual_seed(1)
    query = torch.randn(2, 3, 4, 2)
    key = torch.randn(2, 4, 2, 2)
    value = torch.randn(2, 4, 2, 2)
    valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
    scaling = 0.5

    actual = grouped_query_attention(
        query,
        key,
        value,
        scaling=scaling,
        key_valid=valid,
    )

    expanded_key = key.repeat_interleave(2, dim=2)
    expanded_value = value.repeat_interleave(2, dim=2)
    scores = (
        torch.einsum(
            "bqhd,bkhd->bhqk",
            query.float(),
            expanded_key.float(),
        )
        * scaling
    )
    scores = scores.masked_fill(
        ~valid[:, None, None, :],
        torch.finfo(scores.dtype).min,
    )
    probabilities = torch.softmax(scores, dim=-1)
    expected = torch.einsum(
        "bhqk,bkhd->bqhd",
        probabilities,
        expanded_value.float(),
    ).to(query.dtype)

    torch.testing.assert_close(actual, expected)


def test_serving_dual_stream_uses_two_independent_attention_calls(monkeypatch):
    calls: list[torch.Tensor | None] = []

    def fake_grouped_attention(query, key, value, *, scaling, key_valid=None):
        del key, value, scaling
        calls.append(key_valid)
        fill = 1.0 if key_valid is None else 2.0
        return torch.full_like(query, fill)

    monkeypatch.setattr(
        serving_model, "grouped_query_attention", fake_grouped_attention
    )
    monkeypatch.setattr(
        serving_model,
        "extract_context_kv_from_paged_cache",
        lambda *_args, **_kwargs: (
            torch.zeros(1, 1, 1, 2),
            torch.zeros(1, 1, 1, 2),
            torch.ones(1, 1, dtype=torch.bool),
        ),
    )
    monkeypatch.setattr(
        serving_model,
        "_layer_attention_metadata",
        lambda _name: SimpleNamespace(
            query_start_loc=torch.tensor([0, 2]),
            seq_lens=torch.tensor([3]),
            max_seq_len=3,
            max_query_len=2,
            block_table=torch.tensor([[0]]),
        ),
    )

    class QKVProjection:
        def __call__(self, hidden_states):
            projected = torch.cat((hidden_states, hidden_states, hidden_states), dim=-1)
            return projected, None

    class Rotary:
        def __call__(self, _positions, query, key):
            return query, key

    class OutputProjection:
        def __call__(self, hidden_states):
            return hidden_states, None

    class IdentityAdapter:
        @staticmethod
        def adapt_query(query):
            return query

        @staticmethod
        def adapt_output(output):
            return output

    attention = SimpleNamespace(
        qkv_proj=QKVProjection(),
        q_size=2,
        kv_size=2,
        q_norm=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
        rotary_emb=Rotary(),
        scaling=2**-0.5,
        attn=SimpleNamespace(
            layer_name="draft.layers.0.self_attn.attn",
            kv_cache=torch.ones(1, 1, 1, 4),
        ),
        o_proj=OutputProjection(),
    )

    output = dual_stream_attention_forward_vllm(
        attention,
        IdentityAdapter(),
        torch.tensor(0.0),
        torch.randn(2, 2),
        torch.tensor([5, 6]),
        block_size=2,
    )

    torch.testing.assert_close(output, torch.full((2, 2), 3.0))
    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1] is not None


def test_horizon_embedding_repeats_for_each_query_block():
    model = Qwen3KVNativeDFlashModel.__new__(Qwen3KVNativeDFlashModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(block_size=3)
    model.embed_tokens = torch.nn.Embedding(8, 2)
    model.embed_tokens.weight.data.zero_()
    model.has_separate_mask_embedding = False
    model.mask_token_id = None
    model.horizon_embedding = torch.nn.Embedding(3, 2)
    model.horizon_embedding.weight.data.copy_(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )

    actual = model.embed_input_ids(torch.tensor([1, 2, 3, 4, 5, 6]))

    expected = model.horizon_embedding.weight.detach().repeat(2, 1)
    torch.testing.assert_close(actual, expected)


def _bare_snapshot_model(metadata: SelectedVerifierKV, capacity: int = 8):
    model = Qwen3KVNativeDFlashModel.__new__(Qwen3KVNativeDFlashModel)
    torch.nn.Module.__init__(model)
    unique_groups = tuple(dict.fromkeys(metadata.cache_group_ids))
    model._verifier_metadata = metadata
    model._verifier_slot_buffer = torch.empty(
        (len(unique_groups), capacity),
        dtype=torch.int64,
    )
    model._verifier_slot_group_rows = {
        group_id: index for index, group_id in enumerate(unique_groups)
    }
    model._verifier_slot_representatives = {
        group_id: next(
            name
            for name, layer_group_id in zip(
                metadata.layer_names,
                metadata.cache_group_ids,
                strict=True,
            )
            if layer_group_id == group_id
        )
        for group_id in unique_groups
    }
    model._verifier_slot_valid_length = None
    return model


def test_target_slot_snapshot_is_private_and_consumed_once():
    metadata = SelectedVerifierKV(
        layer_ids=(7, 15),
        layer_names=(
            "model.layers.7.self_attn",
            "model.layers.15.self_attn",
        ),
        cache_group_ids=(6, 8),
        block_size=2,
        num_kv_heads=1,
        head_dim=2,
    )
    model = _bare_snapshot_model(metadata)
    shared_slots = torch.arange(10 * 8, dtype=torch.int64).reshape(10, 8)
    slots_by_layer = {
        name: shared_slots[group_id]
        for name, group_id in zip(
            metadata.layer_names,
            metadata.cache_group_ids,
            strict=True,
        )
    }

    model.snapshot_verifier_slot_mappings(slots_by_layer, num_tokens=4)
    expected = shared_slots[8, :4].clone()
    shared_slots[8, :4].fill_(999_999)

    snapshot = model._current_verifier_slot_mappings(4)
    torch.testing.assert_close(snapshot[8], expected)
    model.clear_verifier_slot_mappings()
    with pytest.raises(RuntimeError, match="missing or stale"):
        model._current_verifier_slot_mappings(4)


def test_proposer_snapshot_uses_target_context_length_and_rejects_ubatching():
    class SnapshotSpy:
        def __init__(self):
            self.events = []

        def clear_verifier_slot_mappings(self):
            self.events.append("clear")

        def snapshot_verifier_slot_mappings(self, mappings, num_tokens):
            self.events.append(("snapshot", mappings, num_tokens))

    model = SnapshotSpy()
    speculator = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Qwen3KVNativeDFlashForCausalLM"],
            )
        ),
        model=model,
    )
    mappings = {"model.layers.7.self_attn": torch.arange(6)}

    returned = _snapshot_verifier_slot_mappings(speculator, 4, mappings)

    assert returned is model
    assert model.events == ["clear", ("snapshot", mappings, 4)]

    with pytest.raises(ValueError, match="ubatched"):
        _snapshot_verifier_slot_mappings(speculator, 4, [mappings])


def test_raw_verifier_kv_is_written_to_draft_cache_without_mapping():
    num_tokens = 3
    num_sources = 2
    num_heads = 2
    head_dim = 4
    source_keys = torch.randn(num_tokens, num_sources, num_heads, head_dim)
    source_values = torch.randn_like(source_keys)
    keys_before = source_keys.clone()
    values_before = source_values.clone()

    class CacheUpdateSpy:
        def __init__(self):
            self.calls = []

        def do_kv_cache_update(self, _attn, keys, values, _cache, slots):
            self.calls.append((keys.clone(), values.clone(), slots.clone()))
            assert keys.is_contiguous()
            assert values.is_contiguous()

    model = Qwen3KVNativeDFlashModel.__new__(Qwen3KVNativeDFlashModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        verifier_kv_layer_ids=[7, 15],
        verifier_num_key_value_heads=num_heads,
        verifier_head_dim=head_dim,
    )
    adapter = KVSpaceAdapter(
        num_key_value_heads=num_heads,
        num_key_value_groups=1,
        head_dim=head_dim,
    )
    model.layers = [SimpleNamespace(kv_adapter=adapter)]
    model._verifier_kv_mapping_indices = (1,)
    spy = CacheUpdateSpy()
    attention = SimpleNamespace(impl=spy, kv_cache=torch.empty(0))
    model._attn_layers = [attention]
    model._num_attn_layers = 1
    model._source_kv = lambda *_args: (source_keys, source_values)
    slots = torch.tensor([9, 10, 11], dtype=torch.long)

    model.precompute_and_store_context_kv(
        torch.zeros(num_tokens, head_dim),
        torch.arange(num_tokens),
        slots,
    )

    assert len(spy.calls) == 1
    actual_keys, actual_values, actual_slots = spy.calls[0]
    torch.testing.assert_close(actual_keys, source_keys[:, 1])
    torch.testing.assert_close(actual_values, source_values[:, 1])
    torch.testing.assert_close(actual_slots, slots)
    torch.testing.assert_close(source_keys, keys_before)
    torch.testing.assert_close(source_values, values_before)


def test_v2_patch_skips_verifier_snapshot_during_dummy_run(monkeypatch):
    original_propose = DFlashSpeculator.propose
    original_set_attn = DFlashSpeculator.set_attn
    original_init_cudagraph = DFlashSpeculator.init_cudagraph_manager
    marker = plugin._V2_SPECULATOR_PATCH_MARKER
    marker_was_set = getattr(DFlashSpeculator, marker, False)

    def fake_propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        num_tokens_across_dp=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        del (
            self,
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile,
        )
        return "dummy-ok"

    def fail_snapshot(*_args, **_kwargs):
        raise AssertionError("dummy run must not snapshot verifier slots")

    try:
        DFlashSpeculator.propose = fake_propose
        if hasattr(DFlashSpeculator, marker):
            delattr(DFlashSpeculator, marker)
        monkeypatch.setattr(plugin, "_snapshot_verifier_slot_mappings", fail_snapshot)
        _install_v2_dflash_speculator_patch()

        result = DFlashSpeculator.propose(
            SimpleNamespace(),
            SimpleNamespace(num_tokens=4),
            {},
            {},
            torch.empty(0),
            None,
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            dummy_run=True,
            skip_attn_for_dummy_run=True,
        )
        assert result == "dummy-ok"
    finally:
        DFlashSpeculator.propose = original_propose
        DFlashSpeculator.set_attn = original_set_attn
        DFlashSpeculator.init_cudagraph_manager = original_init_cudagraph
        if marker_was_set:
            setattr(DFlashSpeculator, marker, True)
        elif hasattr(DFlashSpeculator, marker):
            delattr(DFlashSpeculator, marker)


def test_v1_and_v2_patches_install_idempotently_with_original_signatures():
    v1_propose_signature = inspect.signature(DFlashProposer.propose)
    v2_propose_signature = inspect.signature(DFlashSpeculator.propose)

    _install_v1_dflash_proposer_patch()
    _install_v1_dflash_proposer_patch()
    _install_v2_dflash_speculator_patch()
    _install_v2_dflash_speculator_patch()

    assert getattr(DFlashProposer, plugin._V1_PROPOSER_PATCH_MARKER)
    assert getattr(DFlashSpeculator, plugin._V2_SPECULATOR_PATCH_MARKER)
    assert inspect.signature(DFlashProposer.propose) == v1_propose_signature
    assert inspect.signature(DFlashSpeculator.propose) == v2_propose_signature
    assert hasattr(DFlashProposer.propose, "__wrapped__")
    assert hasattr(DFlashSpeculator.propose, "__wrapped__")
