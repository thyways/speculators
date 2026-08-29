"""Parallel token-latent feedback speculator (方案设计 v1.2)."""

from .config import (
    LatentFeedbackSpeculatorConfig,
    ParallelTokenLatentFeedbackSpeculatorConfig,
    ParallelTokenLatentSpeculatorConfig,
    TokenLatentFeedbackSpeculatorConfig,
)
from .core import (
    LatentFeedbackDraftModel,
    ParallelTokenLatentDraftModel,
    ParallelTokenLatentFeedbackDraftModel,
    TokenLatentFeedbackDraftModel,
)
from .metrics import compute_latent_cosine_loss
from .model_definitions import (
    TokenLatentFeedbackHead,
    TokenLatentFeedbackOutput,
    build_causal_toeplitz_matrix,
)

__all__ = [
    "LatentFeedbackDraftModel",
    "LatentFeedbackSpeculatorConfig",
    "ParallelTokenLatentDraftModel",
    "ParallelTokenLatentFeedbackDraftModel",
    "ParallelTokenLatentFeedbackSpeculatorConfig",
    "ParallelTokenLatentSpeculatorConfig",
    "TokenLatentFeedbackDraftModel",
    "TokenLatentFeedbackHead",
    "TokenLatentFeedbackOutput",
    "TokenLatentFeedbackSpeculatorConfig",
    "build_causal_toeplitz_matrix",
    "compute_latent_cosine_loss",
]
