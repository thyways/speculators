from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DEFAULT_GRU_HIDDEN_DIM",
    "DEFAULT_LOGITS_CORRECTION_EMB_DIM",
    "DEFAULT_PURE_DRAFT_PREFIX_LEN",
    "DominoSpeculatorConfig",
    "resolve_suffix_start",
]

# Single source of truth for the head's defaults. The serving-side config
# translation reads the same constants: a fallback that disagreed with the
# training default would silently shift which slots get corrected whenever a
# checkpoint config omits the field.
DEFAULT_GRU_HIDDEN_DIM = 1024
DEFAULT_LOGITS_CORRECTION_EMB_DIM = 256
DEFAULT_PURE_DRAFT_PREFIX_LEN = 1


def resolve_suffix_start(
    *,
    sample_from_anchor: bool,
    pure_draft_prefix_len: int,
) -> int:
    """First block slot that receives the recurrent logit correction.

    Slots are counted from the start of the block, so when the anchor slot is a
    bonus token rather than a prediction the correction starts one slot later.
    Training and serving must agree on this, so both call this helper.
    """
    return pure_draft_prefix_len + (0 if sample_from_anchor else 1)


@SpeculatorModelConfig.register("domino")
class DominoSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash config plus Domino's recurrent logit-correction head.

    Domino restores the intra-block token dependency that block-parallel
    drafting drops: a GRU runs over the block's token embeddings and its state
    at each position feeds an MLP whose output is added to the DFlash logits.
    All DFlash fields are inherited unchanged.
    """

    speculators_model_type: Literal["domino"] = "domino"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DominoSpeculator"],
        description="Model architectures that can load these weights",
    )

    sample_from_anchor: bool = Field(
        default=True,
        description=(
            "Whether to sample from the anchor position. "
            "False: anchor is the bonus token, only mask tokens predict "
            "(block_size-1 speculative tokens). "
            "True: sample from anchor and all mask positions "
            "(block_size speculative tokens). "
            "Default True matches the upstream Domino recipe "
            "(dflash_config.shift_label=true)."
        ),
    )

    gru_hidden_dim: int = Field(
        default=DEFAULT_GRU_HIDDEN_DIM,
        gt=0,
        description=(
            "Hidden width of the single-layer GRU that carries the intra-block "
            "recurrent state over previous token embeddings."
        ),
    )
    logits_correction_emb_dim: int = Field(
        default=DEFAULT_LOGITS_CORRECTION_EMB_DIM,
        gt=0,
        description=(
            "Bottleneck width of the logit-correction MLP "
            "([hidden + gru_hidden] -> emb_dim -> draft_vocab_size). "
            "Upstream Domino calls this `emb_dim`."
        ),
    )
    pure_draft_prefix_len: int = Field(
        default=DEFAULT_PURE_DRAFT_PREFIX_LEN,
        ge=0,
        description=(
            "Number of leading predicted positions in each block that keep the "
            "uncorrected DFlash logits. The correction starts at slot "
            "`pure_draft_prefix_len` when sample_from_anchor is True, and at "
            "`pure_draft_prefix_len + 1` otherwise (slot 0 is the anchor)."
        ),
    )

    # Schedule for the blended base/final objective. These are training
    # hyperparameters rather than architecture, but they live on the config so
    # every construction path (from_training_args, from_pretrained, a
    # config-only directory) resolves them identically, and so a checkpoint
    # records the schedule it was trained under. Serving ignores them.
    lambda_base_start: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Initial weight of the uncorrected (base) loss term. It decays "
            "linearly to 0 over the first `lambda_base_decay_ratio` of "
            "training, anchoring early steps on the backbone alone. Set to 0 "
            "to train the corrected objective from the start."
        ),
    )
    lambda_base_decay_ratio: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of total training steps over which `lambda_base_start` "
            "decays to 0."
        ),
    )

    @property
    def suffix_start(self) -> int:
        """First block slot that receives the recurrent logit correction."""
        return resolve_suffix_start(
            sample_from_anchor=self.sample_from_anchor,
            pure_draft_prefix_len=self.pure_draft_prefix_len,
        )
