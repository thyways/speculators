"""HashGram DFlash training model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import LossConfig, kl_div_loss, resolve_loss_config, tv_loss
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash2.metrics import (
    compute_metrics,
    selector_training_candidates,
)
from speculators.models.dspark.model_definitions import MarkovHead
from speculators.models.hashgram.config import HashGramSpeculatorConfig
from speculators.models.hashgram.model_definitions import HashGramSelector
from speculators.models.utils import conditional_torch_compile

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["HashGramDraftModel"]

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


@SpeculatorModel.register("hashgram")
class HashGramDraftModel(DFlashDraftModel):
    """DFlash backbone with DSpark recall and hashed vector n-gram reranking."""

    config_class: ClassVar[type[HashGramSpeculatorConfig]] = (  # type: ignore[misc,assignment]
        HashGramSpeculatorConfig
    )

    def __init__(self, config: HashGramSpeculatorConfig) -> None:
        target_vocab_size = config.transformer_layer_config.vocab_size
        if config.draft_vocab_size != target_vocab_size:
            raise ValueError(
                "HashGram currently requires the full verifier vocabulary: "
                f"draft_vocab_size={config.draft_vocab_size}, "
                f"verifier_vocab_size={target_vocab_size}."
            )
        if config.hashgram_top_k > target_vocab_size:
            raise ValueError(
                f"hashgram_top_k ({config.hashgram_top_k}) cannot exceed "
                f"verifier_vocab_size ({target_vocab_size})."
            )
        super().__init__(config=config)

        initializer_range = float(
            getattr(config.transformer_layer_config, "initializer_range", 0.02)
        )
        self.markov_head: MarkovHead | None = None
        if config.hashgram_use_markov_recall and config.hashgram_markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=target_vocab_size,
                draft_vocab_size=target_vocab_size,
                markov_rank=config.hashgram_markov_rank,
                hidden_size=config.transformer_layer_config.hidden_size,
                head_type="vanilla",
            )
        self.hashgram_selector = HashGramSelector(
            vocab_size=target_vocab_size,
            hidden_size=config.transformer_layer_config.hidden_size,
            rank=config.hashgram_rank,
            top_k=config.hashgram_top_k,
            bigram_buckets=config.hashgram_bigram_buckets,
            trigram_buckets=config.hashgram_trigram_buckets,
            num_hashes=config.hashgram_num_hashes,
            initializer_range=initializer_range,
            hidden_refine=config.hashgram_hidden_refine,
            use_bigram=config.hashgram_use_bigram,
            use_trigram=config.hashgram_use_trigram,
        )

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> HashGramDraftModel:
        """Create a HashGram model from the shared training arguments."""
        config = HashGramSpeculatorConfig(
            **cls._build_base_config_kwargs("hashgram", verifier_config, **kwargs),
            hashgram_rank=kwargs.get("hashgram_rank", 128),
            hashgram_top_k=kwargs.get("hashgram_top_k", 16),
            hashgram_bigram_buckets=kwargs.get("hashgram_bigram_buckets", 1_048_576),
            hashgram_trigram_buckets=kwargs.get("hashgram_trigram_buckets", 1_048_576),
            hashgram_num_hashes=kwargs.get("hashgram_num_hashes", 1),
            hashgram_loss_alpha=kwargs.get("hashgram_loss_alpha", 1.0),
            hashgram_markov_rank=kwargs.get("hashgram_markov_rank", 256),
            hashgram_use_markov_recall=kwargs.get("hashgram_use_markov_recall", True),
            hashgram_hidden_refine=kwargs.get("hashgram_hidden_refine", False),
            hashgram_use_bigram=kwargs.get("hashgram_use_bigram", True),
            hashgram_use_trigram=kwargs.get("hashgram_use_trigram", True),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve unary and HashGram selector objectives."""
        implementation = kwargs.get("loss_implementation", "fused")
        loss_config = resolve_loss_config(kwargs["loss_fn"], implementation)
        tv_loss_fn = resolve_loss_config("tv", implementation)["tv"][0]
        shared = {
            "loss_config": loss_config,
            "tv_loss_fn": tv_loss_fn,
            "gamma": kwargs.get("dflash_decay_gamma", 4.0),
            "max_anchors": kwargs.get("max_anchors", 512),
            "per_position_loss_weight": kwargs.get(
                "per_position_loss_weight", "fixed-exp-decay"
            ),
            "dpace_alpha": kwargs.get("dpace_alpha", 0.5),
            "selector_loss_alpha": kwargs.get("hashgram_loss_alpha", 1.0),
        }
        return dict(shared), dict(shared)

    def _previous_ids(
        self,
        input_ids: torch.Tensor,
        document_ids: torch.Tensor,
        anchored_block_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the previous one/two token IDs for every aligned block slot."""
        block_size = self.block_size
        num_blocks = anchored_block_indices.numel() // block_size
        block_positions = anchored_block_indices.view(num_blocks, block_size)
        target_positions = block_positions + int(self.config.sample_from_anchor)
        prev1_positions = target_positions - 1
        prev2_positions = target_positions - 2

        seq_len = input_ids.shape[1]
        target_valid = (target_positions >= 0) & (target_positions < seq_len)
        safe_targets = target_positions.clamp(min=0, max=max(seq_len - 1, 0))
        target_docs = document_ids[0, safe_targets]
        target_valid &= target_docs >= 0

        def gather_positions(positions: torch.Tensor) -> torch.Tensor:
            valid = (positions >= 0) & (positions < seq_len)
            safe = positions.clamp(min=0, max=max(seq_len - 1, 0))
            values = input_ids[0, safe]
            same_document = document_ids[0, safe] == target_docs
            valid &= target_valid & same_document
            return torch.where(
                valid,
                values,
                torch.full_like(values, self.mask_token_id),
            )

        return (
            gather_positions(prev1_positions).reshape(1, -1),
            gather_positions(prev2_positions).reshape(1, -1),
        )

    def _same_document_block_mask(
        self,
        document_ids: torch.Tensor,
        anchored_block_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Mask predictions whose target token leaves the anchor document."""
        block_size = self.block_size
        num_blocks = anchored_block_indices.numel() // block_size
        block_positions = anchored_block_indices.view(num_blocks, block_size)
        target_positions = block_positions + int(self.config.sample_from_anchor)
        seq_len = document_ids.shape[1]
        valid = (target_positions >= 0) & (target_positions < seq_len)
        safe_targets = target_positions.clamp(min=0, max=max(seq_len - 1, 0))
        target_docs = document_ids[0, safe_targets]
        anchor_docs = document_ids[0, block_positions[:, :1]]
        valid &= (anchor_docs >= 0) & (target_docs == anchor_docs)
        return valid.reshape(1, -1)

    def _sequential_greedy_predictions(
        self,
        *,
        hidden: torch.Tensor,
        unary_logits: torch.Tensor,
        previous_ids: torch.Tensor,
        previous2_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the lightweight selector chain without another backbone pass.

        Unlike the teacher-forced training objective, every slot after the first
        consumes the token actually chosen for the preceding slot.
        """
        num_blocks = unary_logits.shape[1] // self.block_size
        hidden_blocks = hidden.view(num_blocks, self.block_size, -1)
        unary_blocks = unary_logits.view(num_blocks, self.block_size, -1)
        previous_blocks = previous_ids.view(num_blocks, self.block_size)
        previous2_blocks = previous2_ids.view(num_blocks, self.block_size)
        predictions = torch.full(
            (num_blocks, self.block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=unary_logits.device,
        )

        start_pos = 0 if self.config.sample_from_anchor else 1
        if start_pos >= self.block_size:
            return predictions.reshape(1, -1)
        prev1 = previous_blocks[:, start_pos]
        prev2 = previous2_blocks[:, start_pos]

        for position in range(start_pos, self.block_size):
            slot_hidden = hidden_blocks[:, position]
            recall_logits = unary_blocks[:, position]
            if self.markov_head is not None:
                recall_logits = recall_logits + self.markov_head.block_bias(
                    prev_token_ids=prev1.unsqueeze(1),
                    hidden_states=slot_hidden.unsqueeze(1),
                ).squeeze(1)

            candidate_ids = recall_logits.topk(
                self.hashgram_selector.top_k, dim=-1
            ).indices
            candidate_logits = self.hashgram_selector.score_candidates(
                unary_logits=recall_logits,
                hidden_states=slot_hidden,
                previous_ids=prev1,
                previous2_ids=prev2,
                candidate_ids=candidate_ids,
                lm_head_weight=self.lm_head.weight
                if self.config.hashgram_hidden_refine
                else None,
            )
            selected = candidate_ids.gather(
                -1, candidate_logits.argmax(dim=-1, keepdim=True)
            ).squeeze(-1)
            predictions[:, position] = selected
            prev2, prev1 = prev1, selected

        return predictions.reshape(1, -1)

    def _greedy_rollout_metrics(
        self,
        *,
        predictions: torch.Tensor,
        target_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute raw slot accuracy and prefix acceptance for greedy rollout."""
        num_blocks = predictions.shape[1] // self.block_size
        pred_blocks = predictions.view(num_blocks, self.block_size)
        target_blocks = target_ids.view(num_blocks, self.block_size)
        valid_blocks = loss_mask.to(torch.bool).view(num_blocks, self.block_size)
        correct_blocks = pred_blocks.eq(target_blocks)
        metric_dtype = torch.float32
        start_pos = 0 if self.config.sample_from_anchor else 1

        result: dict[str, torch.Tensor] = {}
        full_correct = torch.zeros((), dtype=metric_dtype, device=loss_mask.device)
        full_total = torch.zeros_like(full_correct)
        prefix_alive = torch.ones(num_blocks, dtype=torch.bool, device=loss_mask.device)
        accepted_lengths = torch.zeros(
            num_blocks, dtype=metric_dtype, device=loss_mask.device
        )

        for position in range(start_pos, self.block_size):
            valid = valid_blocks[:, position]
            correct = correct_blocks[:, position] & valid
            prefix_alive = prefix_alive & correct
            valid_total = valid.to(metric_dtype).sum()
            correct_total = correct.to(metric_dtype).sum()
            accepted_total = prefix_alive.to(metric_dtype).sum()
            accepted_lengths += prefix_alive.to(metric_dtype)
            full_correct += correct_total
            full_total += valid_total

            prefix = f"hashgram_greedy_position_{position}"
            result[f"{prefix}_acc_sum"] = correct_total
            result[f"{prefix}_acc_total"] = valid_total
            result[f"{prefix}_accept_sum"] = accepted_total
            result[f"{prefix}_accept_total"] = valid_total.clone()

        block_valid = valid_blocks[:, start_pos:].any(dim=-1)
        result["hashgram_greedy_full_acc_sum"] = full_correct
        result["hashgram_greedy_full_acc_total"] = full_total
        result["hashgram_greedy_eal_sum"] = (
            accepted_lengths * block_valid.to(metric_dtype)
        ).sum()
        result["hashgram_greedy_eal_total"] = block_valid.to(metric_dtype).sum()
        return result

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        loss_config: LossConfig | None = None,
        tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
        gamma: float = 4.0,
        max_anchors: int = 512,
        hashgram_loss_alpha: float | None = None,
        selector_loss_alpha: float | None = None,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ) -> tuple[None, torch.Tensor, dict[str, Any]]:
        hidden, unary_logits, targets, aligned_loss_mask, block_indices = (
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
        previous_ids, previous2_ids = self._previous_ids(
            input_ids,
            document_ids,
            block_indices,
        )
        aligned_loss_mask = aligned_loss_mask * self._same_document_block_mask(
            document_ids,
            block_indices,
        ).to(aligned_loss_mask.dtype)

        recall_logits = unary_logits
        if self.markov_head is not None:
            num_blocks = block_indices.numel() // self.block_size
            hidden_blocks = hidden.view(num_blocks, self.block_size, -1)
            prev_blocks = previous_ids.view(num_blocks, self.block_size)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_blocks,
                hidden_states=hidden_blocks,
            ).reshape_as(unary_logits)
            recall_logits = unary_logits + markov_bias

        base_candidate_ids = None
        if not self.training:
            base_candidate_ids = unary_logits.topk(
                self.hashgram_selector.top_k, dim=-1
            ).indices
        candidate_ids = recall_logits.topk(self.hashgram_selector.top_k, dim=-1).indices
        target_ids = targets.argmax(dim=-1)
        training_candidate_ids, target_positions, contains_target = (
            selector_training_candidates(candidate_ids, target_ids)
        )
        candidate_logits = self.hashgram_selector.score_candidates(
            unary_logits=recall_logits,
            hidden_states=hidden,
            previous_ids=previous_ids,
            previous2_ids=previous2_ids,
            candidate_ids=training_candidate_ids,
            lm_head_weight=self.lm_head.weight
            if self.config.hashgram_hidden_refine
            else None,
        )

        selector_alpha = (
            selector_loss_alpha
            if selector_loss_alpha is not None
            else hashgram_loss_alpha
        )
        if selector_alpha is None:
            selector_alpha = self.config.hashgram_loss_alpha
        loss, metrics = compute_metrics(
            unary_logits=recall_logits,
            targets=targets,
            training_candidate_ids=training_candidate_ids,
            candidate_logits=candidate_logits,
            target_positions=target_positions,
            contains_target=contains_target,
            loss_mask=aligned_loss_mask,
            block_size=self.block_size,
            top_k=self.hashgram_selector.top_k,
            sample_from_anchor=self.config.sample_from_anchor,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            tv_loss_fn=tv_loss_fn,
            gamma=gamma,
            selector_loss_alpha=selector_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
        )
        with torch.no_grad():
            valid = aligned_loss_mask.to(torch.bool)
            valid_float = valid.to(unary_logits.dtype)
            valid_total = valid_float.sum()
            metric_candidate_sets = [("recall", candidate_ids)]
            if base_candidate_ids is not None:
                metric_candidate_sets.insert(0, ("base", base_candidate_ids))
            for name, ids in metric_candidate_sets:
                contains = ids.eq(target_ids.unsqueeze(-1)).any(dim=-1)
                prefix = f"{name}_candidate_recall_at_{self.hashgram_selector.top_k}"
                metrics[f"{prefix}_sum"] = (
                    contains.to(valid_float.dtype) * valid_float
                ).sum()
                metrics[f"{prefix}_total"] = valid_total.clone()
            if not self.training:
                greedy_predictions = self._sequential_greedy_predictions(
                    hidden=hidden,
                    unary_logits=unary_logits,
                    previous_ids=previous_ids,
                    previous2_ids=previous2_ids,
                )
                metrics.update(
                    self._greedy_rollout_metrics(
                        predictions=greedy_predictions,
                        target_ids=target_ids,
                        loss_mask=aligned_loss_mask,
                    )
                )
        return None, loss, metrics
