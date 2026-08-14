from __future__ import annotations

from typing import Any

import torch

__all__ = ["add_kv_native_losses"]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _group_queries(
    queries: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    batch, tokens, layers, num_heads, head_dim = queries.shape
    if num_heads % num_kv_heads:
        raise ValueError(
            f"query heads ({num_heads}) must be divisible by KV heads ({num_kv_heads})"
        )
    groups = num_heads // num_kv_heads
    return queries.view(batch, tokens, layers, num_kv_heads, groups, head_dim)


def _query_sensitive_key_loss(
    predicted_keys: torch.Tensor,
    teacher_keys: torch.Tensor,
    queries: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    batch, tokens, layers, num_kv_heads, head_dim = predicted_keys.shape
    if tokens % block_size:
        raise ValueError(
            f"token count ({tokens}) must be divisible by block size ({block_size})"
        )
    num_blocks = tokens // block_size
    grouped_queries = _group_queries(queries.float(), num_kv_heads).view(
        batch,
        num_blocks,
        block_size,
        layers,
        num_kv_heads,
        -1,
        head_dim,
    )
    key_error = (predicted_keys.float() - teacher_keys.float()).view(
        batch,
        num_blocks,
        block_size,
        layers,
        num_kv_heads,
        head_dim,
    )
    score_error = torch.einsum(
        "bnqlhgd,bnklhd->bnqlhgk",
        grouped_queries,
        key_error,
    ) * (head_dim**-0.5)
    query_mask = loss_mask.to(score_error.dtype).view(batch, num_blocks, block_size)
    key_mask = query_mask[:, :, None, None, None, None, :]
    per_query = (score_error.square() * key_mask).sum(dim=-1) / key_mask.sum(
        dim=-1
    ).clamp_min(1.0)
    per_query = per_query.mean((-1, -2, -3)).view(batch, tokens)
    return _masked_mean(per_query, loss_mask)


def _attention_weighted_value_loss(
    *,
    teacher_keys: torch.Tensor,
    predicted_values: torch.Tensor,
    teacher_values: torch.Tensor,
    queries: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    batch, tokens, layers, num_kv_heads, _ = teacher_keys.shape
    if tokens % block_size:
        raise ValueError(
            f"token count ({tokens}) must be divisible by block size ({block_size})"
        )
    num_blocks = tokens // block_size
    grouped_queries = _group_queries(queries.float(), num_kv_heads).view(
        batch,
        num_blocks,
        block_size,
        layers,
        num_kv_heads,
        -1,
        teacher_keys.shape[-1],
    )
    teacher_key_blocks = teacher_keys.float().view(
        batch,
        num_blocks,
        block_size,
        layers,
        num_kv_heads,
        teacher_keys.shape[-1],
    )
    attention_scores = torch.einsum(
        "bnqlhgd,bnklhd->bnqlhgk",
        grouped_queries,
        teacher_key_blocks,
    ) * (teacher_keys.shape[-1] ** -0.5)
    key_mask = loss_mask.bool().view(batch, num_blocks, block_size)
    attention_scores = attention_scores.masked_fill(
        ~key_mask[:, :, None, None, None, None, :],
        torch.finfo(attention_scores.dtype).min,
    )
    attention_weights = attention_scores.softmax(dim=-1).detach()

    value_error = (predicted_values.float() - teacher_values.float()).view(
        batch,
        num_blocks,
        block_size,
        layers,
        num_kv_heads,
        teacher_values.shape[-1],
    )
    weighted_error = torch.einsum(
        "bnqlhgk,bnklhd->bnqlhgd",
        attention_weights,
        value_error,
    ).square()
    per_query = weighted_error.mean((-1, -2, -3, -4)).view(batch, tokens)
    return _masked_mean(per_query, loss_mask)


def add_kv_native_losses(
    base_loss: torch.Tensor,
    metrics: dict[str, Any],
    *,
    predicted_keys: torch.Tensor,
    predicted_values: torch.Tensor,
    teacher_keys: torch.Tensor,
    teacher_values: torch.Tensor,
    queries: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    local_kv_loss_alpha: float,
    query_key_loss_alpha: float,
    attention_value_loss_alpha: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Add attention-aware local-K/V objectives."""

    key_mse = (
        (predicted_keys.float() - teacher_keys.float()).square().mean((-1, -2, -3))
    )
    value_mse = (
        (predicted_values.float() - teacher_values.float()).square().mean((-1, -2, -3))
    )
    local_kv_loss = _masked_mean(key_mse + value_mse, loss_mask)
    query_key_loss = _query_sensitive_key_loss(
        predicted_keys,
        teacher_keys,
        queries,
        loss_mask,
        block_size,
    )
    attention_value_loss = _attention_weighted_value_loss(
        teacher_keys=teacher_keys,
        predicted_values=predicted_values,
        teacher_values=teacher_values,
        queries=queries,
        loss_mask=loss_mask,
        block_size=block_size,
    )
    loss = (
        base_loss
        + local_kv_loss_alpha * local_kv_loss
        + query_key_loss_alpha * query_key_loss
        + attention_value_loss_alpha * attention_value_loss
    )

    one = torch.ones((), device=base_loss.device)
    metrics["loss_sum"] = loss.detach().clone()
    for name, value in (
        ("local_kv_loss", local_kv_loss),
        ("query_key_loss", query_key_loss),
        ("attention_value_loss", attention_value_loss),
    ):
        metrics[f"{name}_sum"] = value.detach()
        metrics[f"{name}_total"] = one.clone()
    return loss, metrics
