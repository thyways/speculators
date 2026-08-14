from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, cast

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
    "KVBridgeDiagnostics",
    "KVSpaceAdapter",
    "Qwen3KVNativeDecoderLayer",
    "TargetToDraftKVBridge",
    "VerifierRotaryEmbedding",
    "apply_partial_rotary_pos_emb",
    "kv_bridge_attention_forward",
    "kv_native_attention_forward",
    "remove_partial_rotary_pos_emb",
]

_TEXT_POSITION_IDS_NDIM = 2
_MROPE_POSITION_IDS_NDIM = 3
_MROPE_COORDINATES = 3
_GATED_BRIDGE_FEATURE_NDIM = 4
_RMS_EPSILON = torch.finfo(torch.float32).eps


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.float()
    return torch.sqrt(torch.mean(values * values))


def _rms_normalize_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Parameter-free RMS normalization over the head dimension."""

    values = tensor.float()
    inverse_rms = torch.rsqrt(
        values.square().mean(dim=-1, keepdim=True).clamp_min(_RMS_EPSILON)
    )
    return tensor * inverse_rms.to(tensor.dtype)


def _cap_residual_ratio(
    base: torch.Tensor,
    residual: torch.Tensor,
    max_ratio: float | None,
) -> torch.Tensor:
    """Bound each token's residual RMS relative to its fused-base RMS."""

    if max_ratio is None:
        return residual
    base_values = base.float()
    residual_values = residual.float()
    base_rms = torch.sqrt(
        base_values.square().mean(dim=-1, keepdim=True).clamp_min(_RMS_EPSILON)
    )
    residual_rms = torch.sqrt(
        residual_values.square().mean(dim=-1, keepdim=True).clamp_min(_RMS_EPSILON)
    )
    multiplier = (max_ratio * base_rms / residual_rms).clamp(max=1.0)
    return residual * multiplier.to(residual.dtype)


class _BridgeMapperOutput(NamedTuple):
    mapped: torch.Tensor
    correction_ratio: torch.Tensor
    gate_entropy: torch.Tensor


class _BridgeMappingOutput(NamedTuple):
    keys: torch.Tensor
    values: torch.Tensor
    key_correction_ratio: torch.Tensor
    value_correction_ratio: torch.Tensor
    key_gate_entropy: torch.Tensor
    value_gate_entropy: torch.Tensor


class KVBridgeDiagnostics(NamedTuple):
    key_correction_ratio: torch.Tensor
    value_correction_ratio: torch.Tensor
    key_gate_entropy: torch.Tensor
    value_gate_entropy: torch.Tensor


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


class _GatedLowRankBridgeMapper(nn.Module):
    """Learn an all-source softmax fusion plus a low-rank correction."""

    def __init__(
        self,
        num_source_layers: int,
        head_dim: int,
        *,
        rank: int,
        residual_scale: float,
        max_correction_ratio: float | None,
    ) -> None:
        super().__init__()
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")
        if max_correction_ratio is not None and max_correction_ratio <= 0.0:
            raise ValueError("max_correction_ratio must be positive when set")
        self.num_source_layers = num_source_layers
        self.residual_scale = residual_scale
        self.max_correction_ratio = max_correction_ratio
        self.gate_logits = nn.Parameter(torch.empty(num_source_layers))
        # One independent D -> rank projection per verifier source layer. This is
        # algebraically the source-block decomposition of a concat -> bottleneck FC,
        # but exposes an explicit softmax gate over layers before the latent sum.
        self.source_up = nn.Parameter(torch.empty(num_source_layers, rank, head_dim))
        self.down = nn.Linear(rank, head_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.gate_logits)
        for source_weight in self.source_up:
            nn.init.xavier_uniform_(source_weight)
        nn.init.zeros_(self.down.weight)

    def fusion_weights(self) -> torch.Tensor:
        return torch.softmax(self.gate_logits.float(), dim=0)

    def forward(self, features: torch.Tensor) -> _BridgeMapperOutput:
        """Fuse ``[batch,tokens,source_layers,head_dim]`` into one head."""

        expected_tail = (self.num_source_layers, self.source_up.shape[-1])
        if (
            features.ndim != _GATED_BRIDGE_FEATURE_NDIM
            or tuple(features.shape[-2:]) != expected_tail
        ):
            raise ValueError(
                "Gated bridge mapper expects [batch,tokens,source_layers,head_dim] "
                f"with tail {expected_tail}, got {tuple(features.shape)}"
            )
        weights = self.fusion_weights().to(features.dtype)
        weighted_features = features * weights.view(1, 1, -1, 1)
        fused_base = weighted_features.sum(dim=2)
        latent = torch.einsum(
            "btsd,srd->btr",
            weighted_features,
            self.source_up,
        )
        correction = self.down(torch.nn.functional.silu(latent))
        scaled_correction = self.residual_scale * correction
        scaled_correction = _cap_residual_ratio(
            fused_base,
            scaled_correction,
            self.max_correction_ratio,
        )
        # These values are diagnostics only. Keeping their sqrt/log branches out of
        # AOTAutograd avoids zero-cotangent times singular-derivative NaNs at the
        # zero-initialized correction.
        with torch.no_grad():
            correction_ratio = _rms(scaled_correction) / _rms(fused_base).clamp_min(
                _RMS_EPSILON
            )
            gate_entropy = -(weights * weights.clamp_min(_RMS_EPSILON).log()).sum()
        return _BridgeMapperOutput(
            mapped=fused_base + scaled_correction,
            correction_ratio=correction_ratio,
            gate_entropy=gate_entropy,
        )


class TargetToDraftKVBridge(nn.Module):
    """Map all exported verifier K/V layers into the draft attention space.

    Source Keys arrive with verifier RoPE already applied. The bridge strips that
    rotation, then each draft KV head learns independent softmax weights over all
    verifier layers. Every source is projected into a low-rank latent space before
    the gated sum, retaining source-specific alignment capacity. The fused Key is
    rotated with the same verifier MRoPE used by draft Q/local-K; Values are fused
    directly.
    """

    def __init__(
        self,
        *,
        num_source_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        rank: int,
        residual_scale: float = 1.0,
        max_correction_ratio: float | None = None,
        normalize_keys: bool = False,
    ) -> None:
        super().__init__()
        if num_source_layers <= 0:
            raise ValueError("num_source_layers must be positive")
        self.num_source_layers = num_source_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_correction_ratio = max_correction_ratio
        self.normalize_keys = normalize_keys
        self.key_mappers = nn.ModuleList(
            [
                _GatedLowRankBridgeMapper(
                    num_source_layers,
                    head_dim,
                    rank=rank,
                    residual_scale=residual_scale,
                    max_correction_ratio=max_correction_ratio,
                )
                for _ in range(num_key_value_heads)
            ]
        )
        self.value_mappers = nn.ModuleList(
            [
                _GatedLowRankBridgeMapper(
                    num_source_layers,
                    head_dim,
                    rank=rank,
                    residual_scale=residual_scale,
                    max_correction_ratio=max_correction_ratio,
                )
                for _ in range(num_key_value_heads)
            ]
        )

    def reset_parameters(self) -> None:
        for mapper in (*self.key_mappers, *self.value_mappers):
            mapper.reset_parameters()

    def fusion_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return learned K/V layer weights as ``[kv_heads, source_layers]``."""

        key_weights = torch.stack(
            [
                cast("_GatedLowRankBridgeMapper", mapper).fusion_weights()
                for mapper in self.key_mappers
            ]
        )
        value_weights = torch.stack(
            [
                cast("_GatedLowRankBridgeMapper", mapper).fusion_weights()
                for mapper in self.value_mappers
            ]
        )
        return key_weights, value_weights

    def _validate_sources(
        self,
        source_keys: torch.Tensor,
        source_values: torch.Tensor,
    ) -> None:
        if source_keys.shape != source_values.shape:
            raise ValueError(
                "KV bridge source key/value shapes differ: "
                f"{tuple(source_keys.shape)} vs {tuple(source_values.shape)}"
            )
        if source_keys.ndim != 5:  # noqa: PLR2004
            raise ValueError(
                "KV bridge expects [batch,source_layers,heads,tokens,dim], got "
                f"{tuple(source_keys.shape)}"
            )
        expected = (
            self.num_source_layers,
            self.num_key_value_heads,
            self.head_dim,
        )
        actual = (source_keys.shape[1], source_keys.shape[2], source_keys.shape[4])
        if actual != expected:
            raise ValueError(
                "KV bridge source shape is incompatible with its configuration: "
                f"got layers/heads/dim={actual}, expected {expected}"
            )

    def _map_tensor(
        self,
        tensor: torch.Tensor,
        mappers: nn.ModuleList,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [B, source_layers, source_heads, tokens, D] -> [B, heads, tokens, D].
        outputs: list[torch.Tensor] = []
        correction_ratios: list[torch.Tensor] = []
        gate_entropies: list[torch.Tensor] = []
        for target_head, mapper in enumerate(mappers):
            features = tensor[:, :, target_head].permute(0, 2, 1, 3)
            mapper_output = mapper(features)
            outputs.append(mapper_output.mapped)
            correction_ratios.append(mapper_output.correction_ratio)
            gate_entropies.append(mapper_output.gate_entropy)
        return (
            torch.stack(outputs, dim=1),
            torch.stack(correction_ratios).mean(),
            torch.stack(gate_entropies).mean(),
        )

    def forward(
        self,
        source_keys: torch.Tensor,
        source_values: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> _BridgeMappingOutput:
        self._validate_sources(source_keys, source_values)
        cos, sin = position_embeddings
        num_tokens = source_keys.shape[3]
        expected_position_shape = (
            source_keys.shape[0],
            num_tokens,
            cos.shape[-1],
        )
        if cos.shape != sin.shape or cos.shape != expected_position_shape:
            raise ValueError(
                "KV bridge position embeddings must be matching [batch,tokens,dim] "
                f"tensors: expected {expected_position_shape}, got "
                f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
            )

        content_keys = torch.stack(
            [
                remove_partial_rotary_pos_emb(source_keys[:, index], cos, sin)
                for index in range(self.num_source_layers)
            ],
            dim=1,
        )
        (
            mapped_content_keys,
            key_correction_ratio,
            key_gate_entropy,
        ) = self._map_tensor(
            content_keys,
            self.key_mappers,
        )
        if self.normalize_keys:
            mapped_content_keys = _rms_normalize_last_dim(mapped_content_keys)
        mapped_keys = apply_partial_rotary_pos_emb(
            mapped_content_keys,
            cos,
            sin,
        )
        (
            mapped_values,
            value_correction_ratio,
            value_gate_entropy,
        ) = self._map_tensor(
            source_values,
            self.value_mappers,
        )
        return _BridgeMappingOutput(
            keys=mapped_keys,
            values=mapped_values,
            key_correction_ratio=key_correction_ratio,
            value_correction_ratio=value_correction_ratio,
            key_gate_entropy=key_gate_entropy,
            value_gate_entropy=value_gate_entropy,
        )


def kv_native_attention_forward(  # noqa: PLR0917
    attn: Qwen3DFlashAttention,
    adapter: KVSpaceAdapter,
    hidden_states: torch.Tensor,
    verifier_keys: torch.Tensor,
    verifier_values: torch.Tensor,
    verifier_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Run one direct-read attention over verifier-space prefix K/V.

    ``hidden_states`` must already be layernormed. ``verifier_keys`` and
    ``verifier_values`` are ``[batch, kv_heads, prefix_tokens, head_dim]`` taken
    verbatim from the verifier cache, so the Keys are already rotated by the
    verifier's partial MRoPE. Draft activations are moved into that space by the
    adapter instead of the prefix being moved into draft space.

    Returns the projected attention output. Local Q/K/V remain internal to the
    attention computation because training is supervised only at the DSpark output.
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
    return attn_output


def kv_bridge_attention_forward(  # noqa: PLR0917
    attn: Qwen3DFlashAttention,
    bridge: TargetToDraftKVBridge,
    hidden_states: torch.Tensor,
    verifier_keys: torch.Tensor,
    verifier_values: torch.Tensor,
    verifier_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, KVBridgeDiagnostics]:
    """Attend with native draft Q/local-K/V over a mapped verifier prefix cache."""

    batch_size, query_length, _ = hidden_states.shape
    num_heads = attn.config.num_attention_heads
    num_kv_heads = attn.config.num_key_value_heads
    head_dim = attn.head_dim

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
    prefix_length = verifier_keys.shape[3]
    prefix_embeddings = (cos[:, :prefix_length], sin[:, :prefix_length])
    bridge_output = bridge(
        verifier_keys,
        verifier_values,
        prefix_embeddings,
    )
    mapped_keys = bridge_output.keys
    mapped_values = bridge_output.values
    query_cos = cos[:, -query_length:]
    query_sin = sin[:, -query_length:]
    query = apply_partial_rotary_pos_emb(query, query_cos, query_sin)
    local_keys = apply_partial_rotary_pos_emb(local_keys, query_cos, query_sin)

    key = torch.cat((mapped_keys, local_keys), dim=2)
    value = torch.cat((mapped_values, local_values), dim=2)
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
    attn_output = attn.o_proj(attn_output.reshape(batch_size, query_length, -1))
    with torch.no_grad():
        bridge_diagnostics = KVBridgeDiagnostics(
            key_correction_ratio=bridge_output.key_correction_ratio,
            value_correction_ratio=bridge_output.value_correction_ratio,
            key_gate_entropy=bridge_output.key_gate_entropy,
            value_gate_entropy=bridge_output.value_gate_entropy,
        )
    return attn_output, bridge_diagnostics


class Qwen3KVNativeDecoderLayer(Qwen3DFlashDecoderLayer):
    """A DSpark layer that reads verifier K/V as external prefix memory."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        verifier_num_key_value_heads: int,
        verifier_head_dim: int,
        kv_bridge_enabled: bool = False,
        kv_bridge_num_source_layers: int = 1,
        kv_bridge_rank: int = 32,
        kv_bridge_residual_scale: float = 1.0,
        kv_bridge_max_correction_ratio: float | None = None,
        kv_bridge_normalize_keys: bool = False,
    ) -> None:
        super().__init__(config, layer_idx)
        if verifier_head_dim != self.self_attn.head_dim:
            raise ValueError(
                "KV-native attention currently requires matched head dimensions: "
                f"draft={self.self_attn.head_dim}, verifier={verifier_head_dim}"
            )
        draft_kv_heads = config.num_key_value_heads
        if draft_kv_heads != verifier_num_key_value_heads:
            raise ValueError(
                "KV-native attention currently requires matched KV-head counts: "
                f"draft={draft_kv_heads}, verifier={verifier_num_key_value_heads}"
            )
        self.layer_idx = layer_idx
        self.self_attn.is_causal = False
        self.kv_adapter = (
            None
            if kv_bridge_enabled
            else KVSpaceAdapter(
                num_key_value_heads=verifier_num_key_value_heads,
                num_key_value_groups=self.self_attn.num_key_value_groups,
                head_dim=verifier_head_dim,
            )
        )
        self.kv_bridge = (
            TargetToDraftKVBridge(
                num_source_layers=kv_bridge_num_source_layers,
                num_key_value_heads=verifier_num_key_value_heads,
                head_dim=verifier_head_dim,
                rank=kv_bridge_rank,
                residual_scale=kv_bridge_residual_scale,
                max_correction_ratio=kv_bridge_max_correction_ratio,
                normalize_keys=kv_bridge_normalize_keys,
            )
            if kv_bridge_enabled
            else None
        )

    def _kv_attention(
        self,
        *,
        hidden_states: torch.Tensor,
        verifier_keys: torch.Tensor,
        verifier_values: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, KVBridgeDiagnostics | None]:
        if self.kv_bridge is not None:
            return kv_bridge_attention_forward(
                self.self_attn,
                self.kv_bridge,
                hidden_states,
                verifier_keys,
                verifier_values,
                position_embeddings,
                attention_mask,
                **kwargs,
            )
        if self.kv_adapter is None:
            raise RuntimeError("KV-native layer has neither an adapter nor a bridge")
        output = kv_native_attention_forward(
            self.self_attn,
            self.kv_adapter,
            hidden_states,
            verifier_keys,
            verifier_values,
            position_embeddings,
            attention_mask,
            **kwargs,
        )
        return output, None

    def forward(  # type: ignore[override]
        self,
        *,
        hidden_states: torch.Tensor,
        verifier_keys: torch.Tensor,
        verifier_values: torch.Tensor,
        verifier_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, KVBridgeDiagnostics | None]:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attention_output, bridge_diagnostics = self._kv_attention(
            hidden_states=normalized,
            verifier_keys=verifier_keys,
            verifier_values=verifier_values,
            position_embeddings=verifier_position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )

        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, bridge_diagnostics
