from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import LossConfig
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.domino.config import (
    DEFAULT_GRU_HIDDEN_DIM,
    DEFAULT_LOGITS_CORRECTION_EMB_DIM,
    DEFAULT_PURE_DRAFT_PREFIX_LEN,
    DominoSpeculatorConfig,
)
from speculators.models.domino.metrics import compute_metrics
from speculators.models.domino.model_definitions import DominoLogitsCorrection
from speculators.models.utils import conditional_torch_compile

__all__ = [
    "DominoDraftModel",
    "linear_lambda_base",
]


def linear_lambda_base(
    global_step: int,
    total_steps: int | None,
    lambda_start: float = 1.0,
    decay_ratio: float = 0.5,
) -> float:
    """Domino's base-loss weight: linear decay to zero, then flat zero.

    Decays from ``lambda_start`` to 0 over the first ``total_steps *
    decay_ratio`` steps and stays at 0 afterwards. Without a schedule horizon
    there is nothing to decay over, so the base term is disabled outright
    rather than pinned at its starting value for the whole run.
    """
    if not total_steps or total_steps <= 0:
        return 0.0
    decay_steps = max(1, int(total_steps * decay_ratio))
    progress = min(global_step / decay_steps, 1.0)
    return max(0.0, min(1.0, lambda_start * (1.0 - progress)))


@SpeculatorModel.register("domino")
class DominoDraftModel(DFlashDraftModel):
    """DFlash backbone plus a GRU-driven logit correction.

    Block-parallel drafting predicts every slot in a block independently, so
    acceptance decays toward the end of the block. Domino restores the
    dependency: a GRU consumes the block's token embeddings and its state at
    each position, concatenated with that position's draft hidden state, is
    projected to an additive logit correction. Slots before ``suffix_start``
    keep the uncorrected DFlash logits.

    Training blends the corrected and uncorrected objectives with a weight that
    decays over the run (see :func:`linear_lambda_base`). Everything else --
    the backbone, the anchored-block masking, the loss terms -- is DFlash's.
    """

    config_class: ClassVar[type[DominoSpeculatorConfig]] = DominoSpeculatorConfig  # type: ignore[misc,assignment]

    lambda_base: torch.Tensor

    def __init__(self, config: DominoSpeculatorConfig) -> None:
        super().__init__(config=config)

        self.suffix_start = config.suffix_start
        if self.suffix_start >= config.block_size:
            raise ValueError(
                "Domino needs at least one corrected slot per block, but "
                f"suffix_start={self.suffix_start} leaves none for "
                f"block_size={config.block_size}. Lower "
                "pure_draft_prefix_len or raise block_size."
            )

        self.logits_correction = DominoLogitsCorrection(
            hidden_size=config.transformer_layer_config.hidden_size,
            gru_hidden_dim=config.gru_hidden_dim,
            emb_dim=config.logits_correction_emb_dim,
            draft_vocab_size=self.draft_vocab_size,
            initializer_range=float(
                config.transformer_layer_config.initializer_range or 0.02
            ),
        )

        # Schedule state, not weights: kept off the checkpoint (non-persistent)
        # and held as a tensor so in-place updates do not retrigger a
        # torch.compile guard the way a changing Python float would.
        self.register_buffer(
            "lambda_base",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self._base_loss_active = False

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DominoDraftModel":
        """Create a Domino model from training arguments (mirrors DFlash)."""
        config = DominoSpeculatorConfig(
            **cls._build_base_config_kwargs("domino", verifier_config, **kwargs),
            gru_hidden_dim=kwargs.get("gru_hidden_dim", DEFAULT_GRU_HIDDEN_DIM),
            logits_correction_emb_dim=kwargs.get(
                "logits_correction_emb_dim", DEFAULT_LOGITS_CORRECTION_EMB_DIM
            ),
            pure_draft_prefix_len=kwargs.get(
                "pure_draft_prefix_len", DEFAULT_PURE_DRAFT_PREFIX_LEN
            ),
            lambda_base_start=kwargs.get("lambda_base_start", 1.0),
            lambda_base_decay_ratio=kwargs.get("lambda_base_decay_ratio", 0.5),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    def on_training_step(
        self,
        global_step: int,
        total_steps: int | None = None,
    ) -> None:
        """Refresh the decaying base-loss weight for the upcoming step."""
        weight = linear_lambda_base(
            global_step,
            total_steps,
            self.config.lambda_base_start,
            self.config.lambda_base_decay_ratio,
        )
        if self.lambda_base.dtype != torch.float32:
            # `Module.to(dtype)` casts floating-point buffers too, and a bf16
            # lambda has only ~1/256 resolution near 1.0 -- enough to freeze the
            # schedule for the first dozens of steps of a long decay. Restore
            # float32 rather than silently quantizing. Rebinding the buffer
            # invalidates one torch.compile guard, which the dtype cast that got
            # us here would have done anyway.
            self.lambda_base = torch.zeros(
                (),
                dtype=torch.float32,
                device=self.lambda_base.device,
            )
        self.lambda_base.fill_(weight)
        self._base_loss_active = weight > 0.0

    def _block_token_ids(
        self,
        input_ids: torch.Tensor,
        anchored_block_indices: torch.Tensor,
        num_blocks: int,
    ) -> torch.Tensor:
        """Ground-truth tokens at every block slot, shaped [N, block_size].

        This is the GRU's input sequence: slot ``k``'s state must have consumed
        exactly the tokens preceding slot ``k``'s label, which holds for both
        anchor modes because the block tokens are ``p+0 .. p+block_size-1`` and
        the label offset shifts with ``sample_from_anchor``. Note this is *not*
        DFly's ``_previous_token_ids`` -- that duplicates slot 0's token, which
        is correct for a memoryless bias but would shift the whole recurrent
        state sequence by one position.
        """
        return input_ids[0, anchored_block_indices].view(num_blocks, self.block_size)

    def _correct_suffix_logits(
        self,
        hidden: torch.Tensor,  # [1, N*block_size, hidden_size]
        base_logits: torch.Tensor,  # [1, N*block_size, draft_vocab_size]
        block_tokens: torch.Tensor,  # [N, block_size]
        num_blocks: int,
    ) -> torch.Tensor:
        """Return the final (corrected) logits, shaped like ``base_logits``."""
        block = self.block_size
        suffix_start = self.suffix_start

        states = self.logits_correction.block_states(self.embed_tokens(block_tokens))
        # Slot k reads the state that has consumed tokens up to its predecessor:
        # index k when the anchor slot also predicts, else k-1. Running the scan
        # over all block_size positions and offsetting the slice is equivalent
        # to upstream's shorter scan (the GRU is unidirectional, so the trailing
        # state is simply never read) and keeps one compiled code path.
        lo = suffix_start - (0 if self.config.sample_from_anchor else 1)
        states_suffix = states[:, lo : lo + block - suffix_start]

        base_blocks = base_logits.view(num_blocks, block, -1)
        correction = self.logits_correction(
            hidden.view(num_blocks, block, -1)[:, suffix_start:],
            states_suffix,
        )
        # In place on the correction (never on base_logits, whose graph the base
        # loss still needs) so the suffix logits are materialized once.
        correction.add_(base_blocks[:, suffix_start:])
        return torch.cat([base_blocks[:, :suffix_start], correction], dim=1).view(
            1, num_blocks * block, -1
        )

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        if per_position_loss_weight == "dpace":
            # D-PACE derives its per-position weights from the elementwise loss
            # of the logits being scored, so the final and base terms would be
            # weighted differently and their blend would be meaningless.
            # Upstream Domino excludes it the same way (loss_type is pinned).
            raise ValueError(
                "Domino does not support --per-position-loss-weight=dpace; its "
                "blended base/final objective requires a shared per-position "
                "weighting. Use fixed-exp-decay."
            )

        (
            hidden,
            base_logits,
            targets,
            aligned_loss_mask,
            anchored_block_indices,
        ) = self._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            **kwargs,
        )

        num_blocks = max_anchors
        final_logits = self._correct_suffix_logits(
            hidden,
            base_logits,
            self._block_token_ids(input_ids, anchored_block_indices, num_blocks),
            num_blocks,
        )

        # Validation always scores the pure corrected objective so val loss is
        # comparable across epochs and --save-best stays meaningful. This is a
        # deliberate divergence from upstream, which evaluates with the live
        # training-time lambda.
        include_base = self.training and self._base_loss_active
        lambda_base = (
            self.lambda_base if include_base else torch.zeros_like(self.lambda_base)
        )

        loss, metrics = compute_metrics(
            final_logits,
            base_logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            lambda_base=lambda_base,
            include_base=include_base,
            loss_config=loss_config,
            gamma=gamma,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
        )
        return None, loss, metrics
