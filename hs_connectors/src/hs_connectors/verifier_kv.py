"""Verifier KV extraction helpers used by the file-based vLLM connector.

The public payload uses token-major tensors independent of vLLM's physical
cache layout::

    verifier_keys:   [tokens, selected_layers, kv_heads, head_dim]
    verifier_values: [tokens, selected_layers, kv_heads, head_dim]
    verifier_kv_layer_ids: [selected_layers]

vLLM's attention backends expose logical paged-cache views with shape
``[blocks, kv_heads, block_size, 2 * head_dim]``.  NHD versus HND changes the
physical strides, not those logical axes, so extraction must index the logical
view instead of guessing a layout from equal-sized dimensions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

__all__ = [
    "SelectedVerifierKV",
    "build_slot_mapping",
    "discover_selected_verifier_kv",
    "extract_selected_verifier_kv",
]


_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_STANDARD_KV_CACHE_NDIM = 4


@dataclass(frozen=True)
class SelectedVerifierKV:
    """Resolved selected verifier cache metadata in checkpoint layer order."""

    layer_ids: tuple[int, ...]
    layer_names: tuple[str, ...]
    cache_group_ids: tuple[int, ...]
    block_size: int
    num_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if not (
            len(self.layer_ids) == len(self.layer_names) == len(self.cache_group_ids)
        ):
            raise ValueError(
                "verifier layer IDs, names, and cache-group IDs must have "
                "matching lengths"
            )

    @property
    def cache_group_id(self) -> int:
        """Return the unique group used by single-block-table consumers."""
        group_ids = set(self.cache_group_ids)
        if len(group_ids) != 1:
            raise ValueError(
                "selected verifier KV layers span multiple cache groups: "
                f"{self.cache_group_ids}"
            )
        return next(iter(group_ids))


def _extract_layer_id(layer_name: str) -> int | None:
    match = _LAYER_INDEX_RE.search(layer_name)
    return int(match.group(1)) if match is not None else None


def _find_requested_layers(
    kv_cache_config: KVCacheConfig,
    requested: tuple[int, ...],
) -> dict[int, tuple[int, str]]:
    found: dict[int, tuple[int, str]] = {}
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        for layer_name in group.layer_names:
            layer_id = _extract_layer_id(layer_name)
            if layer_id not in requested:
                continue
            if layer_id in found:
                raise ValueError(
                    f"verifier layer {layer_id} maps to multiple KV cache layers: "
                    f"{found[layer_id][1]!r} and {layer_name!r}"
                )
            found[layer_id] = (group_id, layer_name)
    return found


def _available_layer_ids(kv_cache_config: KVCacheConfig) -> list[int]:
    return sorted(
        layer_id
        for group in kv_cache_config.kv_cache_groups
        for name in group.layer_names
        if (layer_id := _extract_layer_id(name)) is not None
    )


def discover_selected_verifier_kv(
    kv_cache_config: KVCacheConfig,
    selected_layer_ids: list[int] | tuple[int, ...],
) -> SelectedVerifierKV:
    """Resolve selected attention-cache layers and their cache groups.

    Selected groups may differ at serving time because hybrid cache grouping
    also includes draft layers. Their block/head geometry must remain identical.
    """

    requested = tuple(int(layer_id) for layer_id in selected_layer_ids)
    if not requested:
        raise ValueError("selected verifier KV layer IDs must not be empty")
    if len(requested) != len(set(requested)):
        raise ValueError(
            f"selected verifier KV layer IDs contain duplicates: {requested}"
        )

    found = _find_requested_layers(kv_cache_config, requested)

    missing = [layer_id for layer_id in requested if layer_id not in found]
    if missing:
        available = _available_layer_ids(kv_cache_config)
        raise ValueError(
            f"selected verifier KV layers {missing} were not found in vLLM's KV "
            f"cache groups; available attention layer IDs: {available}"
        )

    cache_group_ids = tuple(found[layer_id][0] for layer_id in requested)
    geometry: tuple[int, int, int] | None = None
    geometry_by_group: dict[int, tuple[int, int, int]] = {}
    for group_id in dict.fromkeys(cache_group_ids):
        spec = kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
        for attribute in ("block_size", "num_kv_heads", "head_size"):
            if not hasattr(spec, attribute):
                raise TypeError(
                    f"selected verifier cache group {group_id} is not a standard "
                    "attention KV cache: missing "
                    f"{attribute!r} on {type(spec).__name__}"
                )
        group_geometry = (
            int(spec.block_size),
            int(spec.num_kv_heads),
            int(spec.head_size),
        )
        geometry_by_group[group_id] = group_geometry
        if geometry is None:
            geometry = group_geometry
        elif group_geometry != geometry:
            raise ValueError(
                "selected verifier cache groups must use matching "
                "block/head geometry, got "
                f"{geometry_by_group}"
            )

    if geometry is None:
        raise RuntimeError("selected verifier KV geometry was not resolved")
    block_size, num_kv_heads, head_dim = geometry
    return SelectedVerifierKV(
        layer_ids=requested,
        layer_names=tuple(found[layer_id][1] for layer_id in requested),
        cache_group_ids=cache_group_ids,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def build_slot_mapping(
    block_ids: list[int] | torch.Tensor,
    block_size: int,
    num_tokens: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the logical cache slots for a request's ordered block table."""

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")

    block_ids_t = torch.as_tensor(block_ids, dtype=torch.long, device=device)
    if block_ids_t.ndim != 1:
        raise ValueError(
            f"block_ids must be one-dimensional, got shape {tuple(block_ids_t.shape)}"
        )
    if torch.any(block_ids_t < 0):
        raise ValueError(f"block_ids must be non-negative, got {block_ids_t.tolist()}")
    capacity = block_ids_t.numel() * block_size
    if num_tokens > capacity:
        raise ValueError(
            f"request has {num_tokens} tokens but its {block_ids_t.numel()} cache "
            f"blocks only hold {capacity} tokens"
        )
    offsets = torch.arange(block_size, dtype=torch.long, device=device)
    slots = block_ids_t[:, None] * block_size + offsets[None, :]
    return slots.reshape(-1)[:num_tokens]


def _validate_layer_cache(
    kv_cache: torch.Tensor,
    metadata: SelectedVerifierKV,
) -> torch.Tensor:
    """Validate and return vLLM's logical packed K/V cache view."""

    if kv_cache.ndim != _STANDARD_KV_CACHE_NDIM:
        raise ValueError(
            f"expected a rank-4 standard KV cache, got shape {tuple(kv_cache.shape)}"
        )
    expected_content = 2 * metadata.head_dim
    if kv_cache.shape[-1] != expected_content:
        raise ValueError(
            "unsupported verifier KV layout: expected packed K/V in the final "
            f"dimension ({expected_content}), got shape {tuple(kv_cache.shape)}"
        )

    expected_shape = (
        kv_cache.shape[0],
        metadata.num_kv_heads,
        metadata.block_size,
        expected_content,
    )
    if tuple(kv_cache.shape) != expected_shape:
        raise ValueError(
            "unsupported verifier KV logical shape: expected "
            f"{expected_shape}, got {tuple(kv_cache.shape)}. vLLM exposes "
            "[blocks, heads, tokens, K+V] even when physical strides are NHD."
        )
    return kv_cache


def extract_selected_verifier_kv(
    kv_caches: dict[str, torch.Tensor],
    metadata: SelectedVerifierKV,
    block_ids: list[int] | torch.Tensor,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract selected verifier K/V in token-major training format."""

    if not metadata.layer_names:
        raise ValueError("no selected verifier KV layer names were resolved")
    # The training connector supplies one block table. Fail closed if a future
    # training topology splits the selected verifier layers across groups.
    _ = metadata.cache_group_id
    missing = [name for name in metadata.layer_names if name not in kv_caches]
    if missing:
        raise KeyError(f"selected verifier KV cache tensors are missing: {missing}")

    first_cache = kv_caches[metadata.layer_names[0]]
    slots = build_slot_mapping(
        block_ids,
        metadata.block_size,
        num_tokens,
        device=first_cache.device,
    )
    block_indices = slots // metadata.block_size
    offsets = slots % metadata.block_size

    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for layer_name in metadata.layer_names:
        cache = _validate_layer_cache(kv_caches[layer_name], metadata)
        if cache.device != first_cache.device:
            raise ValueError("all selected verifier KV caches must be on one device")
        if block_indices.numel() and int(block_indices.max()) >= cache.shape[0]:
            raise ValueError(
                f"request references verifier KV block {int(block_indices.max())}, "
                f"but layer {layer_name!r} only has {cache.shape[0]} blocks"
            )
        token_kv = cache[block_indices, :, offsets, :]
        key, value = token_kv.split(metadata.head_dim, dim=-1)
        keys.append(key)
        values.append(value)

    # Each indexed layer is [tokens, heads, dim].
    return torch.stack(keys, dim=1), torch.stack(values, dim=1)
