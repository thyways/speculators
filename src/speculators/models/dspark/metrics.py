"""Loss and metrics for the DSpark draft model.

loss = compound_loss(logits, targets) + conf_alpha * BCE(confidence, accept_rate)

The confidence target ``accept_rate = sum_v min(q_v, p_v) = 1 - d_TV`` is the
analytical acceptance rate (the overlap ``tv_loss`` already computes).
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from speculators.losses import (
    LossConfig,
    compound_loss,
    dflash_loss_decay,
    dpace_loss_decay,
    tv_loss,
)
from speculators.models.metrics import compute_accuracy_multi_step

__all__ = ["compute_metrics"]

_EPS = 1e-8


def _masked_decayed_mean(
    elementwise: torch.Tensor,  # [1, T]
    loss_mask: torch.Tensor,  # [1, T]
    pos_idx: torch.Tensor,  # [1, T]
    decay_fn: Callable[..., torch.Tensor] | None,
) -> torch.Tensor:
    """Masked, optionally position-decayed mean of a precomputed per-position term."""
    loss_mask = loss_mask.to(elementwise.dtype)
    weighted = elementwise * loss_mask
    if decay_fn is not None:
        weighted = weighted * decay_fn(
            pos_idx.to(weighted.dtype), elementwise_loss=elementwise
        )
    denominator = loss_mask.sum(dim=1) + _EPS
    return (weighted.sum(dim=1) / denominator).mean()


def compute_metrics(
    logits: torch.Tensor,  # [1, T, draft_vocab_size] (Markov-corrected)
    targets: torch.Tensor,  # [1, T, draft_vocab_size]
    confidence_logits: torch.Tensor | None,  # [1, T] or None
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    loss_config: LossConfig,
    tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
    gamma: float = 4.0,
    confidence_head_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs)."""

    return _compute_metrics_impl(
        logits,
        targets,
        confidence_logits,
        loss_mask,
        block_size,
        loss_config,
        tv_loss_fn=tv_loss_fn,
        gamma=gamma,
        confidence_head_alpha=confidence_head_alpha,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )


def _compute_metrics_impl(
    logits: torch.Tensor,
    targets: torch.Tensor,
    confidence_logits: torch.Tensor | None,
    loss_mask: torch.Tensor,
    block_size: int,
    loss_config: LossConfig,
    *,
    tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    gamma: float,
    confidence_head_alpha: float,
    per_position_loss_weight: str,
    dpace_alpha: float,
    sample_from_anchor: bool,
) -> tuple[torch.Tensor, dict]:
    device = logits.device
    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)
    start_pos = 0 if sample_from_anchor else 1
    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )

    loss, term_losses = compound_loss(
        logits, targets, loss_mask, pos_idx, loss_config=loss_config, decay_fn=decay_fn
    )

    # Analytical per-position acceptance rate = distributional overlap. It is a
    # detached metric/confidence target and does not alter the draft loss graph;
    # the fused implementation avoids full-vocabulary fp32 softmax tensors.
    with torch.no_grad():
        accept_rate = 1.0 - tv_loss_fn(logits, targets)
    metric_accept_rate = accept_rate.detach()
    num_blocks = seq_len // block_size
    accept_blocks = metric_accept_rate.view(num_blocks, block_size)[:, start_pos:]
    draft_mask = loss_mask.to(metric_accept_rate.dtype).view(num_blocks, block_size)[
        :, start_pos:
    ]
    # A masked slot is absent, not a rejected proposal: use multiplicative
    # identity inside cumprod, then exclude it from the length sum.
    accept_prefix = (accept_blocks * draft_mask + (1.0 - draft_mask)).cumprod(
        dim=-1
    ) * draft_mask

    metrics: dict[str, Any] = {}
    if confidence_logits is not None:
        c_star = metric_accept_rate.to(confidence_logits.dtype)
        bce = binary_cross_entropy_with_logits(
            confidence_logits, c_star, reduction="none"
        )  # [1, T]
        conf_loss = _masked_decayed_mean(bce, loss_mask, pos_idx, decay_fn)
        loss = loss + confidence_head_alpha * conf_loss

        with torch.no_grad():
            mask_f = loss_mask.to(metric_accept_rate.dtype)
            mask_total = mask_f.sum().clamp_min(1.0)
            conf_prob = confidence_logits.float().sigmoid()
            metrics["confidence_loss_sum"] = conf_loss.detach().clone()
            metrics["confidence_loss_total"] = torch.ones((), device=device)
            metrics["confidence_abs_error_sum"] = (
                (conf_prob - metric_accept_rate).abs() * mask_f
            ).sum()
            metrics["confidence_abs_error_total"] = mask_total
            # Mean predicted vs. observed acceptance — a calibration sanity check.
            metrics["confidence_pred_mean_sum"] = (conf_prob * mask_f).sum()
            metrics["confidence_pred_mean_total"] = mask_total.clone()
            # Calibration of the cumulative acceptance product, which is what
            # dynamic draft-length thresholding consumes (signed pred - target).
            conf_prefix = (
                conf_prob.view(num_blocks, block_size)[:, start_pos:] * draft_mask
            ).cumprod(dim=-1)
            metrics["confidence_cumprod_bias_sum"] = (
                (conf_prefix - accept_prefix) * draft_mask
            ).sum()
            metrics["confidence_cumprod_bias_total"] = draft_mask.sum().clamp_min(1.0)

    ones = torch.ones((), device=device)
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    # Mean acceptance rate of the (Markov-corrected) drafter.
    with torch.no_grad():
        mask_f = loss_mask.to(metric_accept_rate.dtype)
        metrics["accept_rate_sum"] = (metric_accept_rate * mask_f).sum()
        metrics["accept_rate_total"] = mask_f.sum().clamp_min(1.0)

    # Expected accepted draft length per block (DSpark's tau): the cumulative
    # acceptance product summed over draft slots, plus the always-emitted anchor.
    with torch.no_grad():
        per_block_len = accept_prefix.sum(dim=-1) + 1.0
        block_valid = (draft_mask.sum(dim=-1) > 0).to(metric_accept_rate.dtype)
        metrics["accept_len_sum"] = (per_block_len * block_valid).sum()
        metrics["accept_len_total"] = block_valid.sum().clamp_min(1.0)

    # Per-position greedy accuracy
    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()
    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    return loss, metrics
