from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.losses import LossConfig, kl_div_loss, resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash2.config import DFlash2SpeculatorConfig
from speculators.models.dflash2.metrics import compute_metrics
from speculators.models.dflash2.model_definitions import (
    CandidateSelector,
    Qwen3DFlash2DecoderLayer,
)
from speculators.models.utils import conditional_torch_compile

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "DFlash2DraftModel",
]


@SpeculatorModel.register("dflash2")
class DFlash2DraftModel(DFlashDraftModel):
    """DFlash backbone plus a local convolution and a candidate selector.

    Ports the training side of vllm-project/vllm#52816. Two things change relative
    to DFlash, both aimed at the same weakness: a DFlash block drafts every slot
    from mask tokens in one pass, so slot ``k`` cannot see what landed in slot
    ``k-1``.

    * A **grouped dynamic depthwise convolution** wrapped around each attention and
      each MLP sublayer mixes a slot's hidden state with the ones before it inside
      the block, with taps zeroed at the block boundary.
    * A **candidate selector** adds a low-rank, predecessor-conditioned correction
      to the logits: ``bias[p, c] = <A[p] * project(h), B[c]>``. At inference the
      correction ranks the target head's top-K per slot and the drafter walks the
      best path from the verified anchor.

    The loss mirrors that split, because inference makes two decisions and only one
    of them is the selector's (see :mod:`speculators.models.dflash2.metrics`):
    DFlash's full-vocabulary loss on the unary logits decides the candidate set,
    ``topK(unary)``, and a cross-entropy over the kept candidates -- scored as the
    walk scores them, ``unary[c] + bias[prev, c]`` -- decides which one wins.
    Slots whose target the draft head left outside the top-K carry no selector loss;
    ``candidate_recall`` reports how often that happens and bounds the selector,
    ``selector_accept_len`` vs ``unary_accept_len`` reports what it is earning.

    The block layout already matches inference: ``_backbone_forward`` lays the
    draft out as ``max_anchors`` contiguous blocks of ``block_size`` query slots
    whose slot 0 holds the verified anchor token, which is exactly vLLM's
    ``1 + num_speculative_tokens`` query block per request.

    Memory: unlike DFlash, DFlash2 cannot prune the draft vocabulary, so every
    full-vocabulary logit tensor in the forward is sized by the verifier's vocab.
    Scale ``--max-anchors`` down accordingly -- the peak holds two such tensors
    (the targets and the unary logits), as DFlash's does. The selector adds only
    ``top_k``-wide tensors.

    Starting from DFlash weights: ``--from-pretrained`` does not cross algorithms,
    because each speculator config pins its own ``speculators_model_type``. Build
    the DFlash2 checkpoint first, then train from it::

        dflash = SpeculatorModel.from_pretrained(dflash_checkpoint)
        dflash2 = DFlash2DraftModel.from_training_args(...)   # or a saved config
        dflash2.load_state_dict(dflash.state_dict(), strict=False)  # conv/selector
        dflash2.save_pretrained(warm_start_dir)                     # keep their init

    The conv and the selector are the only keys ``strict=False`` skips, and their
    initialization makes the result compute exactly what the DFlash checkpoint did
    (``tests/integration/models/test_dflash2_cuda.py`` pins that, bit for bit).
    """

    config_class: ClassVar[type[DFlash2SpeculatorConfig]] = DFlash2SpeculatorConfig  # type: ignore[misc,assignment]
    _no_split_modules: ClassVar[list[str]] = ["Qwen3DFlash2DecoderLayer"]  # type: ignore[misc]

    def __init__(self, config: DFlash2SpeculatorConfig) -> None:
        self._validate_config(config)
        super().__init__(config=config)

        self.candidate_selector = CandidateSelector(
            verifier_vocab_size=self.verifier_vocab_size,
            draft_vocab_size=self.draft_vocab_size,
            hidden_size=config.transformer_layer_config.hidden_size,  # type: ignore[arg-type]
            rank=config.selector_rank,
            top_k=config.selector_top_k,
        )
        self.input_embedding_scale = config.input_embedding_scale
        self.output_multiplier = config.output_multiplier
        self.final_logit_softcapping = config.final_logit_softcapping

    @staticmethod
    def _validate_config(config: DFlash2SpeculatorConfig) -> None:
        """Reject configs that would train fine but draft wrongly once served."""
        if config.sample_from_anchor:
            raise ValueError(
                "dflash2 requires sample_from_anchor=False. The convolution's block "
                "boundary is the inference query block, which is "
                "1 + num_speculative_tokens; that equals block_size only when the "
                "anchor is the bonus token."
            )
        verifier_vocab_size = config.transformer_layer_config.vocab_size
        if config.draft_vocab_size != verifier_vocab_size:
            raise ValueError(
                f"dflash2 requires the full vocabulary: draft_vocab_size "
                f"({config.draft_vocab_size}) must equal the verifier vocab_size "
                f"({verifier_vocab_size}). The candidate selector emits the top-K "
                "ids of the draft head directly as draft tokens, with no d2t remap "
                "on the inference side, so a pruned draft vocabulary would draft "
                "the wrong tokens. Pass --draft-vocab-size "
                f"{verifier_vocab_size} (or omit it)."
            )
        if config.conv_kernel_size > config.block_size:
            raise ValueError(
                f"conv_kernel_size ({config.conv_kernel_size}) must not exceed "
                f"block_size ({config.block_size}); taps past the block boundary "
                "are always zero."
            )

    def _build_layer(self, layer_idx: int) -> nn.Module:
        return Qwen3DFlash2DecoderLayer(
            self.config.transformer_layer_config,  # type: ignore[arg-type]
            layer_idx,
            conv_kernel_size=self.config.conv_kernel_size,
            conv_group_size=self.config.conv_group_size,
            block_size=self.config.block_size,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlash2DraftModel":
        """Create a DFlash2 model from training arguments (mirrors DFlash)."""
        config = DFlash2SpeculatorConfig(
            **cls._build_base_config_kwargs("dflash2", verifier_config, **kwargs),
            conv_kernel_size=kwargs.get("conv_kernel_size", 3),
            conv_group_size=kwargs.get("conv_group_size", 64),
            selector_rank=kwargs.get("selector_rank", 256),
            selector_top_k=kwargs.get("selector_top_k", 16),
            input_embedding_scale=kwargs.get("input_embedding_scale", 1.0),
            output_multiplier=kwargs.get("output_multiplier", 1.0),
            final_logit_softcapping=kwargs.get("final_logit_softcapping"),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """DFlash's knobs plus the selector term's weight; the selector's shape
        comes from the config, not from call kwargs."""
        loss_config = resolve_loss_config(
            kwargs["loss_fn"], kwargs.get("loss_implementation", "fused")
        )
        shared = {
            "loss_config": loss_config,
            "gamma": kwargs.get("dflash_decay_gamma", 4.0),
            "max_anchors": kwargs.get("max_anchors", 512),
            "per_position_loss_weight": kwargs.get(
                "per_position_loss_weight", "fixed-exp-decay"
            ),
            "dpace_alpha": kwargs.get("dpace_alpha", 0.5),
            "selector_loss_weight": kwargs.get("selector_loss_weight", 1.0),
        }
        return dict(shared), dict(shared)

    def _scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply ``output_multiplier`` and the softcap, as ``compute_candidates``
        does to the candidate values it returns. Both default to no-ops.

        ``compute_candidates`` scales after its top-K and this scales before, which
        selects the same candidates: both transforms are monotonic.
        """
        if self.output_multiplier != 1.0:
            logits = logits * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

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
        max_anchors: int = 512,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        selector_loss_weight: float = 1.0,
        **kwargs,
    ):
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **kwargs,
            )
        )

        num_blocks = max_anchors
        block = self.block_size
        # Ground-truth block tokens (verifier vocab); slot 0 is the anchor.
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        # Slot k predicts token p+k, so its predecessor within the block is slot
        # k-1. Slot 0 is the anchor and carries no loss, so its (unused) entry is
        # itself; slot 1's predecessor is the anchor, matching the first step of
        # the inference walk.
        prev_token_ids = torch.cat(
            [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
        )  # [num_blocks, block]

        loss, metrics = compute_metrics(
            self._scale_logits(logits),
            targets,
            aligned_loss_mask,
            block,
            selector=self.candidate_selector,
            hidden_states=hidden,
            prev_token_ids=prev_token_ids,
            anchor_token_ids=block_tokens[:, 0],
            gamma=gamma,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            selector_loss_weight=selector_loss_weight,
        )
        return None, loss, metrics
