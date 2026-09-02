"""vLLM model implementation for Speculators HashGram checkpoints."""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import maybe_prefix

from speculators.models.hashgram.model_definitions import HashGramSelector


class Qwen3HashGramModel(DFlashQwen3Model):
    """DFlash Qwen3 backbone with HashGram recall and reranking modules."""

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
        config = self.config
        draft_vocab_size = int(
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        if draft_vocab_size != int(config.vocab_size):
            raise ValueError(
                "HashGram requires the full verifier vocabulary: "
                f"draft_vocab_size={draft_vocab_size}, "
                f"verifier_vocab_size={config.vocab_size}."
            )

        top_k = int(config.hashgram_top_k)
        if top_k > draft_vocab_size:
            raise ValueError(
                f"hashgram_top_k ({top_k}) cannot exceed vocab size "
                f"({draft_vocab_size})."
            )

        self.markov_head: DSparkMarkovHead | None = None
        use_markov = bool(getattr(config, "hashgram_use_markov_recall", True))
        markov_rank = int(getattr(config, "hashgram_markov_rank", 256))
        if use_markov and markov_rank > 0:
            self.markov_head = DSparkMarkovHead(
                int(config.vocab_size),
                draft_vocab_size,
                markov_rank,
                prefix=maybe_prefix(prefix, "markov_head"),
                quant_config=self.quant_config,
            )
            self.markov_head.requires_grad_(False)

        hidden_refine = bool(getattr(config, "hashgram_hidden_refine", False))
        if hidden_refine and get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError(
                "HashGram hidden_refine requires draft tensor parallel size 1."
            )
        self.hashgram_selector = HashGramSelector(
            vocab_size=draft_vocab_size,
            hidden_size=int(config.hidden_size),
            rank=int(config.hashgram_rank),
            top_k=top_k,
            bigram_buckets=int(config.hashgram_bigram_buckets),
            trigram_buckets=int(config.hashgram_trigram_buckets),
            num_hashes=int(getattr(config, "hashgram_num_hashes", 1)),
            initializer_range=float(getattr(config, "initializer_range", 0.02)),
            hidden_refine=hidden_refine,
            use_bigram=bool(getattr(config, "hashgram_use_bigram", True)),
            use_trigram=bool(getattr(config, "hashgram_use_trigram", True)),
        ).to(dtype=vllm_config.model_config.dtype)
        self.hashgram_selector.requires_grad_(False)


class Qwen3HashGramForCausalLM(DFlashQwen3ForCausalLM):
    """Top-level HashGram draft model consumed by ``HashGramSpeculator``."""

    model_cls = Qwen3HashGramModel

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        # HashGram checkpoints require a full vocabulary, so the map is identity.
        return draft_ids

    def has_markov(self) -> bool:
        return self.model.markov_head is not None

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.model.markov_head is None:
            raise RuntimeError("HashGram Markov recall is disabled.")
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        if self.model.markov_head is None:
            raise RuntimeError("HashGram Markov recall is disabled.")
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def score_hashgram_candidates(
        self,
        *,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        previous_ids: torch.Tensor,
        previous2_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        selector = self.model.hashgram_selector
        lm_head_weight = self.lm_head.weight if selector.hidden_refine else None
        return selector.score_candidates(
            unary_logits=unary_logits,
            hidden_states=hidden_states,
            previous_ids=previous_ids,
            previous2_ids=previous2_ids,
            candidate_ids=candidate_ids,
            lm_head_weight=lm_head_weight,
        )


__all__ = ["Qwen3HashGramForCausalLM", "Qwen3HashGramModel"]
