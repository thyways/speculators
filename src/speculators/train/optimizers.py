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
# Muon, following the convention from Keller Jordan's Muon (embeddings, embedding-like
# codebooks, Markov vocabulary factors, recurrent heads, and output heads are excluded
# from the orthogonalized update).
_ADAMW_NAME_HINTS = (
    "embed_tokens",
    # Domino's recurrent head: gate-stacked RNN matrices are not the kind of
    # matrix Newton-Schulz orthogonalization is meant for, and embed_proj ends
    # in a vocabulary output projection -- the same reason lm_head is excluded.
    "embed_proj",
    "layer_fusion_weights",
    "lm_head",
    "markov_w1",
    "markov_w2",
    "prefix_gru",
    # DFlash2 predecessor/successor tables are embedding-like codebooks.
    "codebook",
)

# Muon only orthogonalizes 2D weight matrices.
_MATRIX_NDIM = 2

# Parameters at or below this rank have no weight matrix to regularize.
_NO_DECAY_MAX_NDIM = 1


def split_named_params_for_weight_decay(
    named_params: list[tuple[str, Tensor]],
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split a named parameter list into decayed and undecayed halves.

    Parameters with ``ndim <= 1`` -- RMSNorm weights, biases, and scalar gates --
    have no weight matrix to regularize. Decaying them only drags norms toward
    zero and pins gates at whatever value corresponds to a zero parameter.

    :param named_params: ``(name, parameter)`` pairs to partition.
    :return: A ``(decay, no_decay)`` tuple of named parameter lists.
    """
    decay: list[tuple[str, Tensor]] = []
    no_decay: list[tuple[str, Tensor]] = []
    for entry in named_params:
        target = no_decay if entry[1].ndim <= _NO_DECAY_MAX_NDIM else decay
        target.append(entry)
    return decay, no_decay


def _named_param_group(
    named_params: list[tuple[str, Tensor]],
    *,
    name: str,
    lr: float,
    weight_decay: float | None = None,
) -> dict:
    group = {
        "params": [param for _, param in named_params],
        "param_names": [param_name for param_name, _ in named_params],
        "name": name,
        "lr": lr,
    }
    if weight_decay is not None:
        group["weight_decay"] = weight_decay
    return group


def _weight_decay_param_groups(
    named_params: list[tuple[str, Tensor]],
    *,
    name: str,
    lr: float,
    weight_decay: float,
    exclude_1d: bool,
) -> list[dict]:
    """Build one parameter group, or two when 1D params skip weight decay.

    Returns an empty list for an empty input so callers never hand AdamW a
    parameter group with no parameters.
    """

    if not named_params:
        return []
    if not exclude_1d:
        return [_named_param_group(named_params, name=name, lr=lr)]
    decay, no_decay = split_named_params_for_weight_decay(named_params)
    return [
        _named_param_group(entries, name=group_name, lr=lr, weight_decay=group_decay)
        for group_name, entries, group_decay in (
            (name, decay, weight_decay),
            (f"{name}_no_decay", no_decay, 0.0),
        )
        if entries
    ]


def split_named_params_for_muon(
    model: Module,
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split a model's trainable parameters into Muon and AdamW groups.

    A parameter goes to Muon iff it requires gradients, is a 2D matrix with both
    dimensions > 1, and is not an embedding, codebook, recurrent-head, or
    vocabulary-output weight; everything else goes to AdamW. Degenerate 2D weights
    (``[1, N]`` / ``[N, 1]`` vectors) route to AdamW -- Muon orthogonalizes matrices,
    not vectors, and crashes on them under FSDP2.

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


def _build_adamw_optimizer(
    model: Module, config, *, exclude_1d: bool
) -> torch.optim.Optimizer:
    """Build the single AdamW optimizer for ``--optimizer adamw``."""

    if not exclude_1d:
        # Historical path: one implicit group over every parameter.
        return torch.optim.AdamW(
            model.named_parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    trainable: list[tuple[str, Tensor]] = [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]
    param_groups = _weight_decay_param_groups(
        trainable,
        name="base",
        lr=config.lr,
        weight_decay=config.weight_decay,
        exclude_1d=True,
    )
    logger.info(
        "AdamW optimizer: %s.",
        ", ".join(
            f"{len(group['params'])} params in {group['name']} at "
            f"weight_decay={group['weight_decay']:.3g}"
            for group in param_groups
        ),
    )
    return torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )


def build_optimizers(model: Module, config) -> list[torch.optim.Optimizer]:
    """Build the optimizer(s) for a training run based on ``config.optimizer``.

    :param model: The model to optimize.
    :param config: A ``TrainerConfig`` holding the optimizer hyperparameters.
    :return: A list of optimizers for the trainer to step in tandem. The default
        "adamw" returns a single optimizer; "muon" returns ``[Muon, AdamW]``.
    """
    # Read directly rather than via getattr: a config that forgets to carry this
    # field should raise here, not silently train without the exclusion.
    exclude_1d = bool(config.weight_decay_exclude_1d)
    if config.optimizer == "adamw":
        return [_build_adamw_optimizer(model, config, exclude_1d=exclude_1d)]

    if config.optimizer == "muon":
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
                    (
                        _weight_decay_param_groups(
                            adamw_params,
                            name="base",
                            lr=config.lr,
                            weight_decay=config.weight_decay,
                            exclude_1d=True,
                        )
                        if exclude_1d
                        else adamw_params
                    ),
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
            )
        if not optimizers:
            raise ValueError("No trainable parameters found to optimize.")
        return optimizers

    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")
