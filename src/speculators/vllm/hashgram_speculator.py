"""Sequential HashGram proposal runtime for vLLM 0.28."""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash.utils import load_dflash_model
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def sample_hashgram_block(
    speculator: Any,
    num_reqs: int,
    head_hidden: torch.Tensor,
) -> None:
    """Sample a HashGram block left-to-right from one backbone forward."""
    num_steps = speculator.num_speculative_steps
    num_samples = num_reqs * num_steps
    sample_hidden = head_hidden[speculator.sample_indices[:num_samples]]
    hidden_per_step = sample_hidden.view(num_reqs, num_steps, -1)
    unary_logits = speculator.model.compute_draft_logits(sample_hidden).view(
        num_reqs,
        num_steps,
        -1,
    )

    idx_map = speculator.sample_idx_mapping[:num_samples].view(num_reqs, num_steps)
    sample_pos = speculator.sample_pos[:num_samples].view(num_reqs, num_steps)
    previous = speculator.input_buffers.input_ids[
        speculator._anchor_idx[:num_reqs]  # noqa: SLF001
    ].long()
    previous2 = speculator.previous2_ids[:num_reqs]

    for step in range(num_steps):
        hidden_step = hidden_per_step[:, step]
        recall_logits = unary_logits[:, step]
        if speculator.model.has_markov():
            markov_embed = speculator.model.markov_embed(previous)
            recall_logits = recall_logits + speculator.model.markov_bias(markov_embed)

        candidate_ids = recall_logits.topk(
            speculator.model.model.hashgram_selector.top_k,
            dim=-1,
        ).indices
        candidate_scores = speculator.model.score_hashgram_candidates(
            unary_logits=recall_logits,
            hidden_states=hidden_step,
            previous_ids=previous,
            previous2_ids=previous2,
            candidate_ids=candidate_ids,
        )

        proposal_logits = speculator.hashgram_logits[:num_reqs]
        proposal_logits.fill_(float("-inf"))
        proposal_logits.scatter_(
            1,
            candidate_ids,
            candidate_scores.to(proposal_logits.dtype),
        )
        sampled = speculator._sample_logits(  # noqa: SLF001
            proposal_logits,
            idx_map[:, step],
            sample_pos[:, step],
            step,
        )
        speculator.draft_tokens[:num_reqs, step] = sampled
        previous2, previous = previous, sampled


class HashGramSpeculator(DSparkSpeculator):
    """DSpark block preparation with HashGram candidate reranking."""

    _speculator_name = "HashGram"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        if self._draft_topk is not None:
            raise ValueError(
                "dspark_draft_topk cannot be combined with HashGram's own top-k."
            )
        if self.enable_adaptive_verification:
            raise ValueError(
                "HashGram checkpoints do not contain a DSpark confidence head; "
                "disable adaptive verification."
            )

        config = self.draft_model_config.hf_config
        mask_token_id = int(config.mask_token_id)
        self.previous2_ids = torch.full(
            (self.max_num_reqs,),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )
        self.hashgram_logits = torch.empty(
            (self.max_num_reqs, self.vocab_size),
            dtype=vllm_config.model_config.head_dtype,
            device=device,
        )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],  # noqa: ARG002
    ) -> torch.nn.Module:
        return load_dflash_model(target_model, self.vllm_config)

    def _capture_previous2(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
    ) -> None:
        """Save the token immediately before each DFlash query anchor."""
        num_reqs = input_batch.num_reqs
        starts = input_batch.query_start_loc[:num_reqs]
        valid_ends = (
            input_batch.query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        )
        has_context = valid_ends > starts
        indices = (valid_ends - 1).clamp(
            min=0,
            max=input_batch.input_ids.shape[0] - 1,
        )
        values = input_batch.input_ids[indices.long()].long()
        values = torch.where(
            has_context,
            values,
            torch.full_like(
                values, int(self.draft_model_config.hf_config.mask_token_id)
            ),
        )
        self.previous2_ids[:num_reqs].copy_(values)

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        sample_hashgram_block(self, num_reqs, head_hidden)

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        self._capture_previous2(input_batch, num_rejected)
        return super().propose(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings,
            last_hidden_states=last_hidden_states,
            aux_hidden_states=aux_hidden_states,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            temperature=temperature,
            seeds=seeds,
            num_tokens_across_dp=num_tokens_across_dp,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
        )


__all__ = ["HashGramSpeculator", "sample_hashgram_block"]
