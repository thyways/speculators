# SPDX-License-Identifier: Apache-2.0
"""vLLM model for dual-stream raw-KV DFlash checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from hs_connectors.verifier_kv import SelectedVerifierKV
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.utils import maybe_prefix

from speculators.models.kv_native_dflash.model_definitions import (
    KVSpaceAdapter,
    context_scale_from_logit,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_PACKED_KV_CACHE_NDIM = 4
_ATTENTION_TENSOR_NDIM = 4


def extract_verifier_kv_from_paged_cache(
    verifier_caches: Mapping[str, torch.Tensor],
    layer_names: Sequence[str],
    layer_cache_group_ids: Sequence[int],
    slot_mappings_by_group: Mapping[int, torch.Tensor],
    *,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather newly computed raw verifier rows in checkpoint layer order."""

    if not layer_names:
        raise ValueError("No verifier KV layers were selected")
    if len(layer_names) != len(layer_cache_group_ids):
        raise ValueError(
            "Verifier layer names and cache-group IDs must have matching lengths"
        )

    group_indices: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for group_id in dict.fromkeys(layer_cache_group_ids):
        if group_id not in slot_mappings_by_group:
            raise KeyError(f"Missing verifier slot mapping for cache group {group_id}")
        slot_mapping = slot_mappings_by_group[group_id]
        if slot_mapping.ndim != 1:
            raise ValueError(
                "verifier slot_mapping must be one-dimensional, got "
                f"{tuple(slot_mapping.shape)} for cache group {group_id}"
            )
        safe_slots = slot_mapping.to(dtype=torch.long).clamp_min(0)
        group_indices[group_id] = (
            slot_mapping >= 0,
            safe_slots // block_size,
            safe_slots % block_size,
        )

    expected_content = 2 * head_dim
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for layer_name, group_id in zip(
        layer_names,
        layer_cache_group_ids,
        strict=True,
    ):
        if layer_name not in verifier_caches:
            raise KeyError(f"Missing verifier KV cache {layer_name!r}")
        cache = verifier_caches[layer_name]
        expected_shape = (
            cache.shape[0],
            num_kv_heads,
            block_size,
            expected_content,
        )
        if cache.ndim != _PACKED_KV_CACHE_NDIM or tuple(cache.shape) != expected_shape:
            raise ValueError(
                "Unsupported verifier KV cache layout for "
                f"{layer_name!r}: expected {expected_shape}, got "
                f"{tuple(cache.shape)}"
            )

        valid_slots, block_indices, block_offsets = group_indices[group_id]
        if (
            not cache.is_cuda
            and block_indices.numel()
            and int(block_indices.max()) >= cache.shape[0]
        ):
            raise ValueError(
                f"Verifier slot mapping references block {int(block_indices.max())}, "
                f"but {layer_name!r} has only {cache.shape[0]} blocks."
            )
        token_kv = cache[block_indices, :, block_offsets, :]
        token_kv = token_kv.masked_fill(~valid_slots[:, None, None], 0)
        token_keys, token_values = token_kv.split(head_dim, dim=-1)
        keys.append(token_keys)
        values.append(token_values)
    return torch.stack(keys, dim=1), torch.stack(values, dim=1)


def extract_context_kv_from_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lengths: torch.Tensor,
    *,
    max_context_length: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize the raw prefix needed by the small DFlash query block."""

    if max_context_length <= 0:
        shape = (context_lengths.shape[0], 0, num_kv_heads, head_dim)
        empty = cache.new_empty(shape)
        valid = torch.empty(
            (context_lengths.shape[0], 0), dtype=torch.bool, device=cache.device
        )
        return empty, empty, valid
    if cache.ndim != _PACKED_KV_CACHE_NDIM:
        raise ValueError(f"draft context cache must be rank four, got {cache.shape}")
    block_size = cache.shape[2]
    expected_tail = (num_kv_heads, block_size, 2 * head_dim)
    if tuple(cache.shape[1:]) != expected_tail:
        raise ValueError(
            f"draft context cache tail must be {expected_tail}, got {cache.shape[1:]}"
        )
    positions = torch.arange(max_context_length, device=cache.device)
    logical_blocks = positions // block_size
    if logical_blocks.numel() and logical_blocks[-1] >= block_table.shape[1]:
        raise ValueError("draft block table is too short for the requested context")
    physical_blocks = block_table[:, logical_blocks].long()
    block_offsets = positions % block_size
    valid = positions.unsqueeze(0) < context_lengths.unsqueeze(1)
    safe_blocks = physical_blocks.clamp_min(0)
    token_kv = cache[safe_blocks, :, block_offsets.unsqueeze(0), :]
    token_kv = token_kv.masked_fill(~valid[:, :, None, None], 0)
    keys, values = token_kv.split(head_dim, dim=-1)
    return keys, values, valid


def grouped_query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    key_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Small dense GQA used for each independent serving-side softmax."""

    batch, query_length, num_query_heads, head_dim = query.shape
    if key.shape != value.shape or key.ndim != _ATTENTION_TENSOR_NDIM:
        raise ValueError("key/value must be matching [B,K,KVH,D] tensors")
    if key.shape[0] != batch or key.shape[-1] != head_dim:
        raise ValueError("query and key/value batch/head dimensions are incompatible")
    num_kv_heads = key.shape[2]
    if num_query_heads % num_kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    groups = num_query_heads // num_kv_heads
    grouped_query = query.reshape(
        batch, query_length, num_kv_heads, groups, head_dim
    ).permute(0, 2, 3, 1, 4)
    keys = key.permute(0, 2, 1, 3)
    values = value.permute(0, 2, 1, 3)
    scores = (
        torch.einsum("bhgqd,bhkd->bhgqk", grouped_query.float(), keys.float()) * scaling
    )
    if key_valid is not None:
        scores = scores.masked_fill(
            ~key_valid[:, None, None, None, :],
            torch.finfo(scores.dtype).min,
        )
    probabilities = torch.softmax(scores, dim=-1)
    if key_valid is not None:
        probabilities = probabilities * key_valid[:, None, None, None, :]
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(probabilities.dtype).eps)
    output = torch.einsum("bhgqk,bhkd->bhgqd", probabilities, values.float())
    return (
        output.permute(0, 3, 1, 2, 4)
        .reshape(batch, query_length, num_query_heads, head_dim)
        .to(query.dtype)
    )


def _layer_attention_metadata(layer_name: str) -> Any | None:
    if not is_forward_context_available():
        return None
    metadata = get_forward_context().attn_metadata
    if metadata is None:
        return None
    if isinstance(metadata, list):
        raise ValueError("KV-native DFlash does not support ubatched metadata")
    return metadata.get(layer_name)


def dual_stream_attention_forward_vllm(
    attention: nn.Module,
    adapter: KVSpaceAdapter,
    context_gate_logit: torch.Tensor,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Serving twin of the training-side dual-stream raw-KV attention."""

    num_tokens = hidden_states.shape[0]
    qkv, _ = attention.qkv_proj(hidden_states)
    query, key, value = qkv.split(
        [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
    )
    query_shape, key_shape = query.shape, key.shape
    query = attention.q_norm(
        query.reshape(num_tokens, attention.num_heads, attention.head_dim)
    )
    key = attention.k_norm(
        key.reshape(num_tokens, attention.num_kv_heads, attention.head_dim)
    )
    value = value.reshape(num_tokens, attention.num_kv_heads, attention.head_dim)

    local_query, local_key = attention.rotary_emb(
        positions,
        query.reshape(query_shape),
        key.reshape(key_shape),
    )
    local_query = local_query.reshape(
        num_tokens, attention.num_heads, attention.head_dim
    )
    local_key = local_key.reshape(
        num_tokens, attention.num_kv_heads, attention.head_dim
    )
    context_query = adapter.adapt_query(query)
    context_query, _ = attention.rotary_emb(
        positions,
        context_query.reshape(query_shape),
        None,
    )
    context_query = context_query.reshape_as(local_query)

    num_blocks = (num_tokens + block_size - 1) // block_size
    padded_tokens = num_blocks * block_size
    pad_tokens = padded_tokens - num_tokens
    if pad_tokens:
        local_query = torch.cat(
            (local_query, local_query.new_zeros(pad_tokens, *local_query.shape[1:])),
            dim=0,
        )
        local_key = torch.cat(
            (local_key, local_key.new_zeros(pad_tokens, *local_key.shape[1:])),
            dim=0,
        )
        value = torch.cat((value, value.new_zeros(pad_tokens, *value.shape[1:])), dim=0)
        context_query = torch.cat(
            (
                context_query,
                context_query.new_zeros(pad_tokens, *context_query.shape[1:]),
            ),
            dim=0,
        )
    local_output = grouped_query_attention(
        local_query.reshape(num_blocks, block_size, *local_query.shape[1:]),
        local_key.reshape(num_blocks, block_size, *local_key.shape[1:]),
        value.reshape(num_blocks, block_size, *value.shape[1:]),
        scaling=attention.scaling,
    ).reshape(padded_tokens, attention.num_heads, attention.head_dim)

    context_output = torch.zeros_like(local_output)
    metadata = _layer_attention_metadata(attention.attn.layer_name)
    if metadata is not None and attention.attn.kv_cache.numel():
        query_start = metadata.query_start_loc
        metadata_reqs = max(0, query_start.shape[0] - 1)
        num_context_reqs = min(num_blocks, metadata_reqs)
        query_lengths = (
            query_start[1 : num_context_reqs + 1] - query_start[:num_context_reqs]
        )
        context_lengths = (
            metadata.seq_lens[:num_context_reqs] - query_lengths
        ).clamp_min(0)
        max_context_length = max(
            0,
            int(metadata.max_seq_len) - int(metadata.max_query_len),
        )
        if num_context_reqs and max_context_length:
            context_keys, context_values, valid = extract_context_kv_from_paged_cache(
                attention.attn.kv_cache,
                metadata.block_table[:num_context_reqs],
                context_lengths,
                max_context_length=max_context_length,
                num_kv_heads=attention.num_kv_heads,
                head_dim=attention.head_dim,
            )
            attended = grouped_query_attention(
                context_query[: num_context_reqs * block_size].reshape(
                    num_context_reqs,
                    block_size,
                    attention.num_heads,
                    attention.head_dim,
                ),
                context_keys,
                context_values,
                scaling=attention.scaling,
                key_valid=valid,
            )
            attended = adapter.adapt_output(attended).reshape(
                num_context_reqs * block_size,
                attention.num_heads,
                attention.head_dim,
            )
            context_output[: attended.shape[0]] = attended

    combined = (
        local_output + context_scale_from_logit(context_gate_logit) * context_output
    )
    projected, _ = attention.o_proj(combined[:num_tokens].reshape(num_tokens, -1))
    return projected


class Qwen3KVNativeDFlashModel(DFlashQwen3Model):
    """DFlash backbone with raw verifier K/V context and a native local stream."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError(
                "KV-native DFlash requires tensor_parallel_size=1 and "
                "pipeline_parallel_size=1."
            )
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        config = self.config
        if getattr(config, "kv_native_architecture", None) != "dual_stream_raw_kv":
            raise ValueError(
                "Qwen3KVNativeDFlashModel requires the dual_stream_raw_kv checkpoint"
            )
        if bool(getattr(self, "use_aux_hidden_state", True)):
            raise ValueError(
                "KV-native DFlash must disable auxiliary hidden-state context."
            )
        del self.hidden_norm

        source_lookup = {
            int(layer_id): index
            for index, layer_id in enumerate(config.verifier_kv_layer_ids)
        }
        self._verifier_kv_mapping_indices = tuple(
            source_lookup[int(layer_id)]
            for layer_id in config.verifier_kv_layer_mapping
        )
        if len(self._verifier_kv_mapping_indices) != len(self.layers):
            raise ValueError(
                "verifier_kv_layer_mapping must contain one source per draft layer"
            )
        model_dtype = vllm_config.model_config.dtype
        for layer in self.layers:
            layer.kv_adapter = KVSpaceAdapter(
                num_key_value_heads=int(config.verifier_num_key_value_heads),
                num_key_value_groups=layer.self_attn.num_heads
                // layer.self_attn.num_kv_heads,
                head_dim=int(config.verifier_head_dim),
            ).to(dtype=model_dtype)
            layer.kv_adapter.requires_grad_(False)
            layer.context_gate_logit = nn.Parameter(
                torch.zeros((), dtype=model_dtype), requires_grad=False
            )
        self.horizon_embedding = nn.Embedding(
            int(config.block_size),
            int(config.hidden_size),
            dtype=model_dtype,
        )
        self.horizon_embedding.requires_grad_(False)

        self._verifier_attention: Mapping[str, Any] | None = None
        self._verifier_metadata: SelectedVerifierKV | None = None
        self._verifier_slot_buffer: torch.Tensor | None = None
        self._verifier_slot_group_rows: dict[int, int] = {}
        self._verifier_slot_representatives: dict[int, str] = {}
        self._verifier_slot_valid_length: int | None = None

    def _build_fused_kv_buffers(self) -> None:
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]
        self._num_attn_layers = len(self._attn_layers)
        first = self.layers[0].self_attn
        self._head_dim = first.head_dim
        self._num_kv_heads = first.num_kv_heads

    def bind_verifier_kv(
        self,
        *,
        verifier_attention: Mapping[str, Any],
        verifier_metadata: SelectedVerifierKV,
        slot_mapping_capacity: int,
        slot_mapping_device: torch.device,
    ) -> None:
        config = self.config
        expected_ids = tuple(int(value) for value in config.verifier_kv_layer_ids)
        if verifier_metadata.layer_ids != expected_ids:
            raise ValueError(
                "Bound verifier layer IDs do not match the checkpoint: "
                f"{verifier_metadata.layer_ids} != {expected_ids}"
            )
        expected_shape = (
            int(config.verifier_num_key_value_heads),
            int(config.verifier_head_dim),
        )
        actual_shape = (
            verifier_metadata.num_kv_heads,
            verifier_metadata.head_dim,
        )
        if actual_shape != expected_shape:
            raise ValueError(
                "Bound verifier KV head shape does not match the checkpoint: "
                f"{actual_shape} != {expected_shape}"
            )
        if slot_mapping_capacity <= 0:
            raise ValueError("Verifier slot-mapping capacity must be positive")

        group_ids = tuple(dict.fromkeys(verifier_metadata.cache_group_ids))
        self._verifier_attention = verifier_attention
        self._verifier_metadata = verifier_metadata
        self._verifier_slot_buffer = torch.empty(
            (len(group_ids), slot_mapping_capacity),
            dtype=torch.int64,
            device=slot_mapping_device,
        )
        self._verifier_slot_group_rows = {
            group_id: index for index, group_id in enumerate(group_ids)
        }
        self._verifier_slot_representatives = {}
        for layer_name, group_id in zip(
            verifier_metadata.layer_names,
            verifier_metadata.cache_group_ids,
            strict=True,
        ):
            self._verifier_slot_representatives.setdefault(group_id, layer_name)
        self._verifier_slot_valid_length = None

    def clear_verifier_slot_mappings(self) -> None:
        self._verifier_slot_valid_length = None

    def snapshot_verifier_slot_mappings(
        self,
        slot_mappings_by_layer: Mapping[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        self.clear_verifier_slot_mappings()
        buffer = self._verifier_slot_buffer
        if buffer is None or self._verifier_metadata is None:
            raise RuntimeError("Verifier KV caches were not bound before slot snapshot")
        if not 0 <= num_tokens <= buffer.shape[1]:
            raise ValueError("Verifier slot snapshot length exceeds its capacity")
        for group_id, layer_name in self._verifier_slot_representatives.items():
            if layer_name not in slot_mappings_by_layer:
                raise KeyError(
                    f"Missing target slot mapping for verifier layer {layer_name!r}"
                )
            source = slot_mappings_by_layer[layer_name]
            if source.ndim != 1 or source.dtype != torch.int64:
                raise ValueError(
                    "Target verifier slot mappings must be one-dimensional int64"
                )
            if source.device != buffer.device or source.shape[0] < num_tokens:
                raise ValueError("Target verifier slot mapping is incompatible")
            row = self._verifier_slot_group_rows[group_id]
            buffer[row, :num_tokens].copy_(source[:num_tokens])
        self._verifier_slot_valid_length = num_tokens

    def _current_verifier_slot_mappings(
        self, num_context: int
    ) -> dict[int, torch.Tensor]:
        buffer = self._verifier_slot_buffer
        if buffer is None or self._verifier_metadata is None:
            raise RuntimeError(
                "Live verifier KV caches were not bound before inference"
            )
        if self._verifier_slot_valid_length != num_context:
            raise RuntimeError(
                "Verifier source-slot snapshot is missing or stale: "
                f"{self._verifier_slot_valid_length} != {num_context}"
            )
        return {
            group_id: buffer[row, :num_context]
            for group_id, row in self._verifier_slot_group_rows.items()
        }

    def _source_kv(
        self,
        num_context: int,
        context_states: torch.Tensor,
        context_slot_mapping: torch.Tensor | Sequence[torch.Tensor | None] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        num_sources = len(config.verifier_kv_layer_ids)
        num_heads = int(config.verifier_num_key_value_heads)
        head_dim = int(config.verifier_head_dim)
        if context_slot_mapping is None:
            self.clear_verifier_slot_mappings()
            shape = (num_context, num_sources, num_heads, head_dim)
            zeros = context_states.new_zeros(shape)
            return zeros, zeros
        if self._verifier_attention is None or self._verifier_metadata is None:
            raise RuntimeError(
                "Live verifier KV caches were not bound before inference"
            )
        source_slots = self._current_verifier_slot_mappings(num_context)
        verifier_caches = {
            name: self._verifier_attention[name].kv_cache
            for name in self._verifier_metadata.layer_names
        }
        try:
            return extract_verifier_kv_from_paged_cache(
                verifier_caches,
                self._verifier_metadata.layer_names,
                self._verifier_metadata.cache_group_ids,
                source_slots,
                block_size=self._verifier_metadata.block_size,
                num_kv_heads=self._verifier_metadata.num_kv_heads,
                head_dim=self._verifier_metadata.head_dim,
            )
        finally:
            self.clear_verifier_slot_mappings()

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: (
            torch.Tensor | Sequence[torch.Tensor | None] | None
        ) = None,
    ) -> None:
        """Copy depth-matched raw verifier K/V into persistent draft context caches."""

        if not hasattr(self, "_num_attn_layers"):
            self._build_fused_kv_buffers()
        num_context = context_states.shape[0]
        if context_positions.shape != (num_context,):
            raise ValueError("context_positions must align with context_states")
        source_keys, source_values = self._source_kv(
            num_context,
            context_states,
            context_slot_mapping,
        )
        expected_dtype = self.layers[0].kv_adapter.query_map.dtype
        if source_keys.dtype != expected_dtype:
            raise ValueError(
                "Verifier KV cache dtype must match the draft dtype; got "
                f"{source_keys.dtype}, expected {expected_dtype}."
            )
        per_layer_slots = isinstance(context_slot_mapping, (list, tuple))
        for layer_index, source_index in enumerate(self._verifier_kv_mapping_indices):
            if context_slot_mapping is None:
                continue
            destination_slots = (
                context_slot_mapping[layer_index]
                if per_layer_slots
                else context_slot_mapping
            )
            if destination_slots is None:
                continue
            raw_keys = source_keys[:, source_index].contiguous()
            raw_values = source_values[:, source_index].contiguous()
            attn = self._attn_layers[layer_index]
            attn.impl.do_kv_cache_update(
                attn,
                raw_keys,
                raw_values,
                attn.kv_cache,
                destination_slots,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = super().embed_input_ids(input_ids)
        horizon_ids = torch.arange(input_ids.shape[0], device=input_ids.device)
        horizon_ids = horizon_ids % int(self.config.block_size)
        return embeddings + self.horizon_embedding(horizon_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            if residual is not None:
                normalized, residual = layer.input_layernorm(hidden_states, residual)
            else:
                residual = hidden_states
                normalized = layer.input_layernorm(hidden_states)
            attention_output = dual_stream_attention_forward_vllm(
                layer.self_attn,
                layer.kv_adapter,
                layer.context_gate_logit,
                normalized,
                positions,
                int(self.config.block_size),
            )
            hidden_states, residual = layer.post_attention_layernorm(
                attention_output, residual
            )
            hidden_states = layer.mlp(hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3KVNativeDFlashForCausalLM(DFlashQwen3ForCausalLM):
    """Top-level raw-KV DFlash model using vLLM's DFlash loader."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Qwen3KVNativeDFlashModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=logit_scale,
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def bind_verifier_kv(
        self,
        *,
        verifier_attention: Mapping[str, Any],
        verifier_metadata: SelectedVerifierKV,
        slot_mapping_capacity: int,
        slot_mapping_device: torch.device,
    ) -> None:
        self.model.bind_verifier_kv(
            verifier_attention=verifier_attention,
            verifier_metadata=verifier_metadata,
            slot_mapping_capacity=slot_mapping_capacity,
            slot_mapping_device=slot_mapping_device,
        )

    def clear_verifier_slot_mappings(self) -> None:
        self.model.clear_verifier_slot_mappings()

    def snapshot_verifier_slot_mappings(
        self,
        slot_mappings_by_layer: Mapping[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        self.model.snapshot_verifier_slot_mappings(
            slot_mappings_by_layer,
            num_tokens,
        )


__all__ = [
    "Qwen3KVNativeDFlashForCausalLM",
    "Qwen3KVNativeDFlashModel",
    "dual_stream_attention_forward_vllm",
    "extract_context_kv_from_paged_cache",
    "extract_verifier_kv_from_paged_cache",
    "grouped_query_attention",
]
