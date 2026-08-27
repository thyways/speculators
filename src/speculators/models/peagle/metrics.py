"""Metrics and globally normalized loss for P-EAGLE training."""

from typing import Any

import torch

from speculators.losses import LossConfig, kl_div_loss
from speculators.models.metrics import compute_accuracy_multi_step

_LOSS_REDUCTION_EPS = 1e-5
_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    anchor_pos: torch.Tensor,
    depth: torch.Tensor,
    num_depths: int,
    loss_config: LossConfig | None = None,
    supervision_mask: torch.Tensor | None = None,
    global_loss_count: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute P-EAGLE loss and counted metrics.

    Partitioned training calls this once per segment. ``global_loss_count`` keeps
    every segment normalized by the number of supervised positions in the full
    COD sample, so summing segment losses exactly recovers the unpartitioned
    objective. ``supervision_mask`` excludes cumulative depth-0 prefix entries
    that are repeated only to provide causal context.
    """
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG

    orig_positions = anchor_pos + depth
    sampled_loss_mask = loss_mask[:, orig_positions]
    if supervision_mask is not None:
        sampled_loss_mask = sampled_loss_mask * supervision_mask.unsqueeze(0)
    sampled_loss_mask = sampled_loss_mask.to(dtype=torch.float32)

    local_loss_count = sampled_loss_mask.sum()
    if global_loss_count is None:
        global_loss_count = local_loss_count
    denominator = global_loss_count.to(dtype=torch.float32) + _LOSS_REDUCTION_EPS

    total_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    weighted_loss_sum = torch.zeros_like(total_loss)
    term_sums: dict[str, torch.Tensor] = {}
    report_terms = len(loss_config) > 1
    for name, (loss_fn, weight) in loss_config.items():
        elementwise_loss = loss_fn(logits, targets)
        term_sum = (elementwise_loss * sampled_loss_mask).sum()
        total_loss = total_loss + weight * term_sum / denominator
        weighted_loss_sum = weighted_loss_sum + weight * term_sum.detach()
        if report_terms:
            term_sums[f"{name}_loss"] = term_sum.detach()

    with torch.no_grad():
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)
        correct_per_pos, total_per_pos = compute_accuracy_multi_step(
            pred_ids,
            target_ids,
            sampled_loss_mask,
            depth.unsqueeze(0),
            num_depths,
        )

    metrics: dict[str, Any] = {
        "loss_sum": weighted_loss_sum,
        "loss_total": local_loss_count.detach(),
        "full_acc_sum": correct_per_pos.sum(),
        "full_acc_total": total_per_pos.sum(),
    }
    for term_name, term_sum in term_sums.items():
        metrics[f"{term_name}_sum"] = term_sum
        metrics[f"{term_name}_total"] = local_loss_count.detach().clone()
    for d in range(num_depths):
        metrics[f"position_{d}_acc_sum"] = correct_per_pos[d]
        metrics[f"position_{d}_acc_total"] = total_per_pos[d]

    return total_loss, metrics
