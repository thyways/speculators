"""Losses and metrics for sampled token-latent candidate training."""

from functools import partial

import torch
from torch.nn import functional

from speculators.losses import (
    dflash_loss_decay,
    dpace_loss_decay,
    masked_decayed_mean,
)
from speculators.models.metrics import compute_accuracy_multi_step
from speculators.models.token_latent_ssm.model_definitions import (
    TokenLatentTrainingOutput,
)

__all__ = ["compute_token_latent_metrics"]


def _candidate_ce(logits: torch.Tensor) -> torch.Tensor:
    candidate_count = logits.shape[-1]
    labels = torch.zeros(
        logits.shape[:-1],
        dtype=torch.long,
        device=logits.device,
    )
    return functional.cross_entropy(
        logits.reshape(-1, candidate_count).float(),
        labels.reshape(-1),
        reduction="none",
    ).reshape(logits.shape[:-1])


def _weighted_mean(
    elementwise: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    *,
    per_position_loss_weight: str,
    dpace_alpha: float,
    gamma: float,
) -> torch.Tensor:
    positions = torch.arange(
        elementwise.shape[1],
        device=elementwise.device,
    ).remainder(block_size)
    positions = positions.unsqueeze(0)
    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay,
            gamma=gamma,
            sample_from_anchor=False,
        )
    return masked_decayed_mean(
        elementwise,
        loss_mask,
        positions,
        decay_fn,
    )


def compute_token_latent_metrics(
    output: TokenLatentTrainingOutput,
    target_draft_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    *,
    retrieval_loss_weight: float,
    conditional_loss_weight: float,
    per_position_loss_weight: str,
    dpace_alpha: float,
    gamma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hidden_ce = _candidate_ce(output.hidden_retrieval_logits)
    transition_ce = _candidate_ce(output.transition_retrieval_logits)
    retrieval_elementwise = 0.5 * (hidden_ce + transition_ce)
    conditional_elementwise = _candidate_ce(output.conditional_logits)

    retrieval_loss = _weighted_mean(
        retrieval_elementwise.reshape(1, -1),
        loss_mask,
        block_size,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        gamma=gamma,
    )
    conditional_loss = _weighted_mean(
        conditional_elementwise.reshape(1, -1),
        loss_mask,
        block_size,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        gamma=gamma,
    )
    loss = (
        retrieval_loss_weight * retrieval_loss
        + conditional_loss_weight * conditional_loss
    )

    selected_index = output.conditional_logits.argmax(dim=-1, keepdim=True)
    predicted_ids = output.candidate_ids.gather(2, selected_index).squeeze(-1)
    flat_predicted = predicted_ids.reshape(1, -1)
    flat_targets = target_draft_ids.reshape(1, -1)
    positions = torch.arange(
        flat_predicted.shape[1],
        device=flat_predicted.device,
    ).remainder(block_size)
    positions = positions.unsqueeze(0)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        flat_predicted,
        flat_targets,
        loss_mask,
        positions,
        block_size,
    )

    one = torch.ones((), device=loss.device)
    metrics = {
        "loss_sum": loss.detach().clone(),
        "loss_total": one,
        "retrieval_loss_sum": retrieval_loss.detach().clone(),
        "retrieval_loss_total": one.clone(),
        "conditional_loss_sum": conditional_loss.detach().clone(),
        "conditional_loss_total": one.clone(),
        "full_acc_sum": correct_per_pos[1:].sum(),
        "full_acc_total": total_per_pos[1:].sum(),
    }
    expected_accepted = torch.zeros((), device=loss.device)
    prefix = torch.ones((), device=loss.device)
    for position in range(1, block_size):
        metrics[f"position_{position}_acc_sum"] = correct_per_pos[position]
        metrics[f"position_{position}_acc_total"] = total_per_pos[position]
        accuracy = correct_per_pos[position] / total_per_pos[position].clamp(min=1.0)
        prefix = prefix * accuracy
        expected_accepted = expected_accepted + prefix
    metrics["eal_sum"] = expected_accepted
    metrics["eal_total"] = one.clone()
    return loss, metrics
