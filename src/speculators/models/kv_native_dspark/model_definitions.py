from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    "KVLayerArtifacts",
    "KVSpaceAdapter",
    "Qwen3KVNativeDecoderLayer",
    "VerifierRotaryEmbedding",
    "apply_partial_rotary_pos_emb",
    "kv_native_attention_forward",
]

_TEXT_POSITION_IDS_NDIM = 2
_MROPE_POSITION_IDS_NDIM = 3
_MROPE_COORDINATES = 3


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
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


class VerifierRotaryEmbedding(nn.Module):
    """Qwen3.5 partial interleaved MRoPE used by exported verifier Keys."""

    def __init__(
        self,
        *,
        head_dim: int,
        partial_rotary_factor: float,
        rope_theta: float,
        mrope_section: list[int],
    ) -> None:
        super().__init__()
        rotary_dim = int(head_dim * partial_rotary_factor)
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError(
                f"verifier rotary dim must be a positive even integer, got {rotary_dim}"
            )
        if sum(mrope_section) != rotary_dim // 2:
            raise ValueError(
                "verifier_mrope_section must sum to half the rotary dimension, "
                f"got sum={sum(mrope_section)} and rotary_dim={rotary_dim}"
            )
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(int(value) for value in mrope_section)

    def forward(
        self, reference: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == _TEXT_POSITION_IDS_NDIM:
            position_ids = position_ids.unsqueeze(0).expand(_MROPE_COORDINATES, -1, -1)
        if (
            position_ids.ndim != _MROPE_POSITION_IDS_NDIM
            or position_ids.shape[0] != _MROPE_COORDINATES
        ):
            raise ValueError(
                "verifier MRoPE position_ids must be [batch, tokens] or "
                f"[3, batch, tokens], got {tuple(position_ids.shape)}"
            )
        frequencies = torch.einsum(
            "f,cbt->cbtf", self.inv_freq.float(), position_ids.float()
        )
        interleaved = frequencies[0].clone()
        for coordinate, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[coordinate] * 3
            interleaved[..., offset:length:3] = frequencies[
                coordinate, ..., offset:length:3
            ]
        embedding = torch.cat((interleaved, interleaved), dim=-1)
        return (
            embedding.cos().to(reference.dtype),
            embedding.sin().to(reference.dtype),
        )


class KVSpaceAdapter(nn.Module):
    """Trainable per-head maps around the DSpark Q/K/V/O path.

    The prefix cache remains in verifier space. Query and local K/V maps move
    draft activations into that space, while the output map applies the Value
    mapping after attention (``A @ V @ W == (A @ V) @ W``). This avoids a
    second full set of Q/K/V/O projections and never materializes mapped prefix
    K/V.
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
        self.local_key_map = nn.Parameter(torch.empty(shape))
        self.local_value_map = nn.Parameter(torch.empty(shape))
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
            for weight in (
                self.query_map,
                self.local_key_map,
                self.local_value_map,
                self.output_map,
            ):
                weight.copy_(identity.expand_as(weight))

    def adapt_query(self, query: torch.Tensor) -> torch.Tensor:
        batch, num_heads, tokens, head_dim = query.shape
        expected_heads = self.num_key_value_heads * self.num_key_value_groups
        if num_heads != expected_heads or head_dim != self.head_dim:
            raise ValueError(
                "query shape is incompatible with the verifier KV adapter: "
                f"got {tuple(query.shape)}, expected heads={expected_heads}, "
                f"head_dim={self.head_dim}"
            )
        grouped = query.view(
            batch,
            self.num_key_value_heads,
            self.num_key_value_groups,
            tokens,
            head_dim,
        )
        adapted = torch.einsum("bhgtd,hde->bhgte", grouped, self.query_map)
        return adapted.reshape(batch, num_heads, tokens, head_dim)

    @staticmethod
    def _adapt_kv(tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bhtd,hde->bhte", tensor, weight)

    def adapt_local_key(self, key: torch.Tensor) -> torch.Tensor:
        return self._adapt_kv(key, self.local_key_map)

    def adapt_local_value(self, value: torch.Tensor) -> torch.Tensor:
        return self._adapt_kv(value, self.local_value_map)

    def adapt_output(self, output: torch.Tensor) -> torch.Tensor:
        batch, tokens, num_heads, head_dim = output.shape
        grouped = output.view(
            batch,
            tokens,
            self.num_key_value_heads,
            self.num_key_value_groups,
            head_dim,
        )
        adapted = torch.einsum("bthgd,hde->bthge", grouped, self.output_map)
        return adapted.reshape(batch, tokens, num_heads, head_dim)


def kv_native_attention_forward(  # noqa: PLR0917
    attn: Qwen3DFlashAttention,
    adapter: KVSpaceAdapter,
    hidden_states: torch.Tensor,
    verifier_keys: torch.Tensor,
    verifier_values: torch.Tensor,
    verifier_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one direct-read attention over verifier-space prefix K/V.

    ``hidden_states`` must already be layernormed. ``verifier_keys`` and
    ``verifier_values`` are ``[batch, kv_heads, prefix_tokens, head_dim]`` taken
    verbatim from the verifier cache, so the Keys are already rotated by the
    verifier's partial MRoPE. Draft activations are moved into that space by the
    adapter instead of the prefix being moved into draft space.

    Returns ``(attention_output, local_keys, local_values, queries)`` where the
    three trailing tensors are in verifier space and shaped
    ``[batch, kv_heads_or_heads, tokens, head_dim]``.
    """

    batch_size, query_length, _ = hidden_states.shape
    num_heads = attn.config.num_attention_heads
    num_kv_heads = attn.config.num_key_value_heads
    head_dim = attn.head_dim

    query = attn.q_proj(hidden_states).view(
        batch_size, query_length, num_heads, head_dim
    )
    query = attn.q_norm(query).transpose(1, 2)
    query = adapter.adapt_query(query)

    local_keys = attn.k_proj(hidden_states).view(
        batch_size, query_length, num_kv_heads, head_dim
    )
    local_keys = attn.k_norm(local_keys).transpose(1, 2)
    local_keys = adapter.adapt_local_key(local_keys)

    local_values = attn.v_proj(hidden_states).view(
        batch_size, query_length, num_kv_heads, head_dim
    )
    local_values = local_values.transpose(1, 2)
    local_values = adapter.adapt_local_value(local_values)

    cos, sin = verifier_position_embeddings
    query_cos = cos[:, -query_length:]
    query_sin = sin[:, -query_length:]
    query = apply_partial_rotary_pos_emb(query, query_cos, query_sin)
    local_keys = apply_partial_rotary_pos_emb(local_keys, query_cos, query_sin)

    key = torch.cat((verifier_keys, local_keys), dim=2)
    value = torch.cat((verifier_values, local_values), dim=2)
    attn_impl = attn.config._attn_implementation or "eager"  # noqa: SLF001
    if attn_impl == "simple_flex_attention":
        attn_fn = ALL_ATTENTION_FUNCTIONS[attn_impl]
    elif attn_impl == "eager":
        attn_fn = eager_attention_forward
    else:
        attn_fn = HF_ATTENTION_FUNCTIONS[attn_impl]
    attn_output, _ = attn_fn(
        attn,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0 if not attn.training else attn.attention_dropout,
        scaling=attn.scaling,
        sliding_window=attn.sliding_window,
        **kwargs,
    )
    attn_output = adapter.adapt_output(attn_output)
    attn_output = attn.o_proj(attn_output.reshape(batch_size, query_length, -1))
    return attn_output, local_keys, local_values, query


@dataclass
class KVLayerArtifacts:
    local_keys: torch.Tensor
    local_values: torch.Tensor
    queries: torch.Tensor


class Qwen3KVNativeDecoderLayer(Qwen3DFlashDecoderLayer):
    """A DSpark layer that reads verifier K/V as external prefix memory."""

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
                "KV-native direct-read currently requires matched head dimensions: "
                f"draft={self.self_attn.head_dim}, verifier={verifier_head_dim}"
            )
        draft_kv_heads = config.num_key_value_heads
        if draft_kv_heads != verifier_num_key_value_heads:
            raise ValueError(
                "KV-native direct-read currently requires matched KV-head counts: "
                f"draft={draft_kv_heads}, verifier={verifier_num_key_value_heads}"
            )
        self.layer_idx = layer_idx
        self.self_attn.is_causal = False
        self.kv_adapter = KVSpaceAdapter(
            num_key_value_heads=verifier_num_key_value_heads,
            num_key_value_groups=self.self_attn.num_key_value_groups,
            head_dim=verifier_head_dim,
        )

    def _kv_attention(
        self,
        hidden_states: torch.Tensor,
        verifier_keys: torch.Tensor,
        verifier_values: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return kv_native_attention_forward(
            self.self_attn,
            self.kv_adapter,
            hidden_states,
            verifier_keys,
            verifier_values,
            position_embeddings,
            attention_mask,
            **kwargs,
        )

    def forward(  # type: ignore[override]
        self,
        *,
        hidden_states: torch.Tensor,
        verifier_keys: torch.Tensor,
        verifier_values: torch.Tensor,
        verifier_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, KVLayerArtifacts]:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attention_output, local_keys, local_values, queries = self._kv_attention(
            normalized,
            verifier_keys,
            verifier_values,
            verifier_position_embeddings,
            attention_mask,
            **kwargs,
        )
        artifacts = KVLayerArtifacts(
            local_keys=local_keys.transpose(1, 2),
            local_values=local_values.transpose(1, 2),
            queries=queries.transpose(1, 2),
        )

        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, artifacts
