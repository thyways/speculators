# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM model implementation for Speculators Domino checkpoints.

Domino's backbone is plain DFlash, so only the recurrent logit-correction head
is new here. That head is imported verbatim from the training module
(:mod:`speculators.models.domino.model_definitions`) -- it is pure PyTorch, and
sharing it is what guarantees serving cannot drift from training. It is
replicated rather than tensor-parallel sharded: at ``gru_hidden_dim`` 1024 and
``emb_dim`` 256 it is a few tens of MB, and replicating keeps the arithmetic
bit-identical to the trained model.

Runtime dispatch uses ``method="dspark"`` (the sequential in-block sampler).
Checkpoint config should set ``architectures=["Qwen3DominoModel"]`` and
``model_arch="domino"``.
"""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM
from vllm.model_executor.models.utils import maybe_prefix

from speculators.models.domino.config import (
    DEFAULT_GRU_HIDDEN_DIM,
    DEFAULT_LOGITS_CORRECTION_EMB_DIM,
    DEFAULT_PURE_DRAFT_PREFIX_LEN,
    resolve_suffix_start,
)
from speculators.models.domino.model_definitions import DominoLogitsCorrection

logger = init_logger(__name__)

# Weights that only exist for training; the draft never evaluates them.
_TRAINING_ONLY_WEIGHTS = (
    "confidence_head",
    "markov_head",
    "markov_w",
    "t2d",
    "verifier_lm_head",
    "verifier_norm",
)


class Qwen3DominoModel(DFlashQwen3Model):
    """DFlash Qwen3 backbone plus Domino's recurrent logit-correction head."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        self.confidence_head = None
        config = self.config
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )

        # Fall back to the training-side defaults, never to a different value.
        gru_hidden_dim = (
            getattr(config, "gru_hidden_dim", None) or DEFAULT_GRU_HIDDEN_DIM
        )
        emb_dim = (
            getattr(config, "logits_correction_emb_dim", None)
            or DEFAULT_LOGITS_CORRECTION_EMB_DIM
        )

        self.logits_correction = DominoLogitsCorrection(
            hidden_size=int(config.hidden_size),
            gru_hidden_dim=int(gru_hidden_dim),
            emb_dim=int(emb_dim),
            draft_vocab_size=int(draft_vocab_size),
            initializer_range=float(getattr(config, "initializer_range", None) or 0.02),
        ).to(dtype=vllm_config.model_config.dtype)
        self.logits_correction.requires_grad_(False)


class Qwen3DominoForCausalLM(Qwen3DSparkForCausalLM):
    """Top-level Domino draft model.

    Reuses DSpark's LM head, draft-vocab remapping, and weight loader; replaces
    the memoryless Markov bias with the GRU-driven correction consumed by the
    shared sequential sampler.
    """

    # None for full-vocabulary drafts, where the remap is the identity.
    draft_id_to_target_id: nn.Parameter | None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Qwen3DominoModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

        # Resolved with the same helper and the same defaults as training, so a
        # config that omits a field cannot shift which slots get corrected.
        self.sample_from_anchor = bool(
            getattr(self.config, "sample_from_anchor", True),
        )
        self.suffix_start = resolve_suffix_start(
            sample_from_anchor=self.sample_from_anchor,
            pure_draft_prefix_len=int(
                getattr(
                    self.config,
                    "pure_draft_prefix_len",
                    DEFAULT_PURE_DRAFT_PREFIX_LEN,
                )
            ),
        )

    def has_markov(self) -> bool:
        return False

    def has_hidden_correction(self) -> bool:
        return False

    def has_recurrent_logits_correction(self) -> bool:
        return True

    def init_recurrent_state(
        self,
        num_reqs: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Zero GRU state, reset at the start of every block."""
        return self.model.logits_correction.prefix_gru.initial_state(
            reference,
            num_reqs,
        )

    def advance_recurrent_state(
        self,
        prev_token_ids: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Consume the previously sampled token, mirroring the training scan."""
        gru = self.model.logits_correction.prefix_gru
        embeds = self.model.embed_input_ids(prev_token_ids)
        return gru.step(gru.project_inputs(embeds.to(state.dtype)), state)

    def recurrent_logit_correction(
        self,
        step: int,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor | None:
        """Additive correction for this step, or None for uncorrected slots.

        ``step`` counts draft tokens, while ``suffix_start`` is expressed in
        block slots; with ``sample_from_anchor=False`` slot 0 is the anchor and
        is never drafted, so the two differ by one.
        """
        slot = step + (0 if self.sample_from_anchor else 1)
        if slot < self.suffix_start:
            return None
        return self.model.logits_correction(hidden_states, state)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load Domino training-export weights via DSpark's loader.

        Training-only tensors are dropped first: DSpark's loader would prefix
        them with ``model.`` and then fail to place them.
        """
        return super().load_weights(
            (name, weight)
            for name, weight in weights
            if not any(part in name for part in _TRAINING_ONLY_WEIGHTS)
        )


__all__ = [
    "Qwen3DominoForCausalLM",
    "Qwen3DominoModel",
]
