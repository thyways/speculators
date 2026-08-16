from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS as HF_ATTENTION_FUNCTIONS,
)
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

from speculators.models.attention import ALL_ATTENTION_FUNCTIONS
from speculators.models.dflash.model_definitions import (
    Qwen3DFlashAttention,
    Qwen3DFlashDecoderLayer,
)

if TYPE_CHECKING:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

__all__ = [
    "DualStreamAttentionMasks",
    "KVSpaceAdapter",
    "Qwen3KVNativeDecoderLayer",
    "VerifierRotaryEmbedding",
    "apply_partial_rotary_pos_emb",
    "context_scale_from_logit",
    "dual_stream_raw_kv_attention_forward",
    "remove_partial_rotary_pos_emb",
]

_TEXT_POSITION_IDS_NDIM = 2
_SERVING_TENSOR_NDIM = 3
_BATCHED_TENSOR_NDIM = 4


class DualStreamAttentionMasks(NamedTuple):
    """Independent masks for verifier-prefix and synthetic-block attention."""

    prefix: Any
    local: Any


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_partial_rotary_pos_emb(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply verifier partial RoPE to ``[batch, heads, tokens, dim]``."""

    rotary_dim = cos.shape[-1]
    rotated, passthrough = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotated = rotated * cos + _rotate_half(rotated) * sin
    return torch.cat((rotated, passthrough), dim=-1)


def remove_partial_rotary_pos_emb(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Remove verifier partial RoPE from ``[batch, heads, tokens, dim]``."""

    rotary_dim = cos.shape[-1]
    rotated, passthrough = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    unrotated = rotated * cos - _rotate_half(rotated) * sin
    return torch.cat((unrotated, passthrough), dim=-1)


class VerifierRotaryEmbedding(nn.Module):
    """One-dimensional partial RoPE matching the verifier text attention."""

    def __init__(
        self,
        *,
        head_dim: int,
        partial_rotary_factor: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        rotary_dim = int(head_dim * partial_rotary_factor)
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError(
                f"verifier rotary dim must be a positive even integer, got {rotary_dim}"
            )
        self.rotary_dim = rotary_dim
        self.rope_theta = float(rope_theta)

    def forward(
        self, reference: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != _TEXT_POSITION_IDS_NDIM:
            raise ValueError(
                "verifier text position_ids must be [batch, tokens], got "
                f"{tuple(position_ids.shape)}"
            )
        frequency_indices = torch.arange(
            0,
            self.rotary_dim,
            2,
            dtype=torch.float32,
            device=reference.device,
        )
        inv_freq = 1.0 / (self.rope_theta ** (frequency_indices / self.rotary_dim))
        frequencies = torch.einsum("f,bt->btf", inv_freq, position_ids.float())
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return (
            embedding.cos().to(reference.dtype),
            embedding.sin().to(reference.dtype),
        )


class KVSpaceAdapter(nn.Module):
    """Identity-initialized maps around the raw verifier-KV context stream.

    Local Q/K/V remain entirely in draft space.  Only the context query is
    adapted into verifier key space, and only the context output is adapted
    back into draft value space.  The verifier K/V tensors themselves are never
    changed.
    """

    def __init__(
        self,
        *,
        num_key_value_heads: int,
        num_key_value_groups: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        shape = (num_key_value_heads, head_dim, head_dim)
        self.query_map = nn.Parameter(torch.empty(shape))
        self.output_map = nn.Parameter(torch.empty(shape))
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.head_dim = head_dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        identity = torch.eye(
            self.head_dim,
            device=self.query_map.device,
            dtype=self.query_map.dtype,
        )
        with torch.no_grad():
            self.query_map.copy_(identity.expand_as(self.query_map))
            self.output_map.copy_(identity.expand_as(self.output_map))

    def adapt_query(self, query: torch.Tensor) -> torch.Tensor:
        """Adapt ``[B,H,T,D]`` or serving-layout ``[T,H,D]`` queries."""

        serving_layout = query.ndim == _SERVING_TENSOR_NDIM
        if serving_layout:
            query = query.transpose(0, 1).unsqueeze(0)
        if query.ndim != _BATCHED_TENSOR_NDIM:
            raise ValueError(f"query must be [B,H,T,D] or [T,H,D], got {query.shape}")
        batch, num_heads, tokens, head_dim = query.shape
        expected_heads = self.num_key_value_heads * self.num_key_value_groups
        if num_heads != expected_heads or head_dim != self.head_dim:
            raise ValueError(
                "query shape is incompatible with the verifier KV adapter: "
                f"got {tuple(query.shape)}, expected heads={expected_heads}, "
                f"head_dim={self.head_dim}"
            )
        grouped = query.reshape(
            batch,
            self.num_key_value_heads,
            self.num_key_value_groups,
            tokens,
            head_dim,
        )
        adapted = torch.einsum("bhgtd,hde->bhgte", grouped, self.query_map)
        adapted = adapted.reshape(batch, num_heads, tokens, head_dim)
        if serving_layout:
            return adapted.squeeze(0).transpose(0, 1).contiguous()
        return adapted

    def adapt_output(self, output: torch.Tensor) -> torch.Tensor:
        """Adapt ``[B,T,H,D]`` or serving-layout ``[T,H,D]`` outputs."""

        serving_layout = output.ndim == _SERVING_TENSOR_NDIM
        if serving_layout:
            output = output.unsqueeze(0)
        if output.ndim != _BATCHED_TENSOR_NDIM:
            raise ValueError(f"output must be [B,T,H,D] or [T,H,D], got {output.shape}")
        batch, tokens, num_heads, head_dim = output.shape
        expected_heads = self.num_key_value_heads * self.num_key_value_groups
        if num_heads != expected_heads or head_dim != self.head_dim:
            raise ValueError(
                "output shape is incompatible with the verifier KV adapter: "
                f"got {tuple(output.shape)}, expected heads={expected_heads}, "
                f"head_dim={self.head_dim}"
            )
        grouped = output.reshape(
            batch,
            tokens,
            self.num_key_value_heads,
            self.num_key_value_groups,
            head_dim,
        )
        adapted = torch.einsum("bthgd,hde->bthge", grouped, self.output_map)
        adapted = adapted.reshape(batch, tokens, num_heads, head_dim)
        return adapted.squeeze(0) if serving_layout else adapted


def context_scale_from_logit(logit: torch.Tensor) -> torch.Tensor:
    """Bound the context residual scale to ``(0, 2)``; zero initializes to one."""

    return 2.0 * torch.sigmoid(logit)


def _resolve_attention_forward(attn: Qwen3DFlashAttention):
    implementation = attn.config._attn_implementation or "eager"  # noqa: SLF001
    if implementation == "simple_flex_attention":
        return ALL_ATTENTION_FUNCTIONS[implementation]
    if implementation == "eager":
        return eager_attention_forward
    return HF_ATTENTION_FUNCTIONS[implementation]


def dual_stream_raw_kv_attention_forward(
    attn: Qwen3DFlashAttention,
    adapter: KVSpaceAdapter,
    context_gate_logit: torch.Tensor,
    hidden_states: torch.Tensor,
    verifier_keys: torch.Tensor,
    verifier_values: torch.Tensor,
    verifier_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_masks: DualStreamAttentionMasks,
    **kwargs: Any,
) -> torch.Tensor:
    """Run independent local and raw-verifier-prefix attention softmaxes."""

    batch_size, query_length, _ = hidden_states.shape
    num_heads = attn.config.num_attention_heads
    num_kv_heads = attn.config.num_key_value_heads
    head_dim = attn.head_dim
    expected_prefix = (batch_size, num_kv_heads, verifier_keys.shape[2], head_dim)
    if verifier_keys.shape != verifier_values.shape:
        raise ValueError("raw verifier key/value shapes must match")
    if tuple(verifier_keys.shape) != expected_prefix:
        raise ValueError(
            "raw verifier K/V must be [B,KVH,prefix,D], got "
            f"{tuple(verifier_keys.shape)}"
        )

    query = attn.q_proj(hidden_states).view(
        batch_size, query_length, num_heads, head_dim
    )
    query = attn.q_norm(query).transpose(1, 2)
    local_keys = attn.k_proj(hidden_states).view(
        batch_size, query_length, num_kv_heads, head_dim
    )
    local_keys = attn.k_norm(local_keys).transpose(1, 2)
    local_values = attn.v_proj(hidden_states).view(
        batch_size, query_length, num_kv_heads, head_dim
    )
    local_values = local_values.transpose(1, 2)

    cos, sin = verifier_position_embeddings
    query_cos = cos[:, -query_length:]
    query_sin = sin[:, -query_length:]
    local_query = apply_partial_rotary_pos_emb(query, query_cos, query_sin)
    local_keys = apply_partial_rotary_pos_emb(local_keys, query_cos, query_sin)
    context_query = apply_partial_rotary_pos_emb(
        adapter.adapt_query(query), query_cos, query_sin
    )

    attention_forward = _resolve_attention_forward(attn)
    common_kwargs = {
        "dropout": 0.0 if not attn.training else attn.attention_dropout,
        "scaling": attn.scaling,
        "sliding_window": attn.sliding_window,
        **kwargs,
    }
    local_output, _ = attention_forward(
        attn,
        local_query,
        local_keys,
        local_values,
        attention_masks.local,
        **common_kwargs,
    )
    context_output, _ = attention_forward(
        attn,
        context_query,
        verifier_keys,
        verifier_values,
        attention_masks.prefix,
        **common_kwargs,
    )
    context_output = torch.nan_to_num(context_output)
    context_output = adapter.adapt_output(context_output)
    combined = (
        local_output + context_scale_from_logit(context_gate_logit) * context_output
    )
    return attn.o_proj(combined.reshape(batch_size, query_length, -1))


class Qwen3KVNativeDecoderLayer(Qwen3DFlashDecoderLayer):
    """One dual-stream layer reading an untouched verifier K/V prefix."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        verifier_num_key_value_heads: int,
        verifier_head_dim: int,
    ) -> None:
        super().__init__(config, layer_idx)
        if verifier_head_dim != self.self_attn.head_dim:
            raise ValueError(
                "KV-native attention requires matched head dimensions: "
                f"draft={self.self_attn.head_dim}, verifier={verifier_head_dim}"
            )
        if config.num_key_value_heads != verifier_num_key_value_heads:
            raise ValueError(
                "KV-native attention requires matched KV-head counts: "
                f"draft={config.num_key_value_heads}, "
                f"verifier={verifier_num_key_value_heads}"
            )
        self.layer_idx = layer_idx
        self.self_attn.is_causal = False
        self.kv_adapter = KVSpaceAdapter(
            num_key_value_heads=verifier_num_key_value_heads,
            num_key_value_groups=self.self_attn.num_key_value_groups,
            head_dim=verifier_head_dim,
        )
        self.context_gate_logit = nn.Parameter(torch.zeros(()))

    def reset_dual_stream_parameters(self) -> None:
        self.kv_adapter.reset_parameters()
        nn.init.zeros_(self.context_gate_logit)

    @property
    def context_scale(self) -> torch.Tensor:
        return context_scale_from_logit(self.context_gate_logit)

    def forward(  # type: ignore[override]
        self,
        *,
        hidden_states: torch.Tensor,
        verifier_keys: torch.Tensor,
        verifier_values: torch.Tensor,
        verifier_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_masks: DualStreamAttentionMasks,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attention_output = dual_stream_raw_kv_attention_forward(
            self.self_attn,
            self.kv_adapter,
            self.context_gate_logit,
            normalized,
            verifier_keys,
            verifier_values,
            verifier_position_embeddings,
            attention_masks,
            **kwargs,
        )
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
