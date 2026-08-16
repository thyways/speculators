"""Optimizer construction for speculator training.

Provides a single entry point, :func:`build_optimizers`, that returns the list of
optimizers the trainer should drive. The default ("adamw") returns a single AdamW
optimizer over all parameters, preserving the historical behavior. The "muon" option
returns two optimizers: ``torch.optim.Muon`` over the 2D weight matrices (which is all
Muon supports) and ``torch.optim.AdamW`` over everything else (norms, biases, and the
embedding / LM-head matrices, following standard Muon practice).

Muon works transparently for both single-GPU and multi-GPU (FSDP2) training: when the
model is sharded with ``fully_shard`` the parameters become ``DTensor``s and Muon's
Newton-Schulz orthogonalization dispatches across ranks automatically.
"""

import logging

import torch
from torch import Tensor
from torch.nn import Module

logger = logging.getLogger("speculators")

# Names of parameters that are 2D but should still be optimized with AdamW rather than
# Muon, following the convention from Keller Jordan's Muon (embeddings and the output
# head are excluded from the orthogonalized update).
_ADAMW_NAME_HINTS = (
    "embed_tokens",
    # Domino's recurrent head: gate-stacked RNN matrices are not the kind of
    # matrix Newton-Schulz orthogonalization is meant for, and embed_proj ends
    # in a vocabulary output projection -- the same reason lm_head is excluded.
    "embed_proj",
    "layer_fusion_weights",
    "lm_head",
    "prefix_gru",
)

# Muon only orthogonalizes 2D weight matrices.
_MATRIX_NDIM = 2


def split_named_params_for_kv_bridge(
    model: Module,
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split trainable parameters into base and target-to-draft bridge groups."""

    base_params: list[tuple[str, Tensor]] = []
    bridge_params: list[tuple[str, Tensor]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bridge = any(part.startswith("kv_bridge") for part in name.split("."))
        target = bridge_params if is_bridge else base_params
        target.append((name, param))
    return base_params, bridge_params


def _named_param_group(
    named_params: list[tuple[str, Tensor]],
    *,
    name: str,
    lr: float,
) -> dict:
    return {
        "params": [param for _, param in named_params],
        "param_names": [param_name for param_name, _ in named_params],
        "name": name,
        "lr": lr,
    }


def split_named_params_for_muon(
    model: Module,
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split a model's trainable parameters into Muon and AdamW groups.

    A parameter goes to Muon iff it requires gradients, is a 2D matrix with both
    dimensions > 1, and is not an embedding or LM-head weight; everything else goes to
    AdamW. Degenerate 2D weights (``[1, N]`` / ``[N, 1]`` vectors) route to AdamW --
    Muon orthogonalizes matrices, not vectors, and crashes on them under FSDP2.

    :param model: The model whose parameters should be partitioned.
    :return: A ``(muon_params, adamw_params)`` tuple of named parameter lists.
    """
    muon_params: list[tuple[str, Tensor]] = []
    adamw_params: list[tuple[str, Tensor]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim == _MATRIX_NDIM
            and min(param.shape) > 1  # exclude degenerate [1, N] / [N, 1] vectors
            and not any(hint in name for hint in _ADAMW_NAME_HINTS)
        ):
            muon_params.append((name, param))
        else:
            adamw_params.append((name, param))
    return muon_params, adamw_params


def build_optimizers(model: Module, config) -> list[torch.optim.Optimizer]:
    """Build the optimizer(s) for a training run based on ``config.optimizer``.

    :param model: The model to optimize.
    :param config: A ``TrainerConfig`` holding the optimizer hyperparameters.
    :return: A list of optimizers for the trainer to step in tandem. The default
        "adamw" returns a single optimizer; "muon" returns ``[Muon, AdamW]``.
    """
    if config.optimizer == "adamw":
        if config.kv_bridge_lr is not None:
            base_params, bridge_params = split_named_params_for_kv_bridge(model)
            if not bridge_params:
                raise ValueError(
                    "kv_bridge_lr was set, but the model has no trainable "
                    "KV-bridge parameters."
                )
            param_groups = []
            if base_params:
                param_groups.append(
                    _named_param_group(base_params, name="base", lr=config.lr)
                )
            param_groups.append(
                _named_param_group(
                    bridge_params,
                    name="kv_bridge",
                    lr=config.kv_bridge_lr,
                )
            )
            logger.info(
                "AdamW optimizer: %d base params at %.3g LR, %d KV-bridge params "
                "at %.3g LR.",
                len(base_params),
                config.lr,
                len(bridge_params),
                config.kv_bridge_lr,
            )
            return [
                torch.optim.AdamW(
                    param_groups,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
            ]
        return [
            torch.optim.AdamW(
                model.named_parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        ]

    if config.optimizer == "muon":
        if config.kv_bridge_lr is not None:
            raise ValueError("kv_bridge_lr is currently supported only with AdamW.")
        muon_params, adamw_params = split_named_params_for_muon(model)
        logger.info(
            "Muon optimizer: %d 2D params via Muon, %d params via AdamW.",
            len(muon_params),
            len(adamw_params),
        )

        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(
                torch.optim.Muon(
                    muon_params,
                    lr=config.muon_lr,
                    momentum=config.muon_momentum,
                    weight_decay=config.muon_weight_decay,
                    ns_steps=config.muon_ns_steps,
                    adjust_lr_fn=config.muon_adjust_lr_fn,
                )
            )
        if adamw_params:
            optimizers.append(
                torch.optim.AdamW(
                    adamw_params,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
            )
        if not optimizers:
            raise ValueError("No trainable parameters found to optimize.")
        return optimizers

    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")
