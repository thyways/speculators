"""Token-code retrieval and diagonal token-conditioned SSM modules."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from speculators.models.latent_scan.model_definitions import LatentRMSNorm

__all__ = [
    "TokenLatentSSMHead",
    "TokenLatentTrainingOutput",
]


@dataclass
class TokenLatentTrainingOutput:
    hidden_retrieval_logits: torch.Tensor
    transition_retrieval_logits: torch.Tensor
    conditional_logits: torch.Tensor
    candidate_ids: torch.Tensor


class TokenLatentSSMHead(nn.Module):
    """Low-dimensional retrieval plus a real-token-conditioned diagonal SSM."""

    def __init__(
        self,
        *,
        target_vocab_size: int,
        draft_vocab_size: int,
        hidden_size: int,
        code_dim: int,
        state_dim: int,
        block_size: int,
        hidden_candidate_count: int,
        transition_candidate_count: int,
        logit_scale_init: float,
        rms_norm_eps: float,
        initializer_range: float,
    ) -> None:
        super().__init__()
        self.target_vocab_size = target_vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.hidden_size = hidden_size
        self.code_dim = code_dim
        self.state_dim = state_dim
        self.block_size = block_size
        self.hidden_candidate_count = hidden_candidate_count
        self.transition_candidate_count = transition_candidate_count

        self.token_codebook = nn.Embedding(target_vocab_size, code_dim)
        self.hidden_norm = LatentRMSNorm(hidden_size, rms_norm_eps)
        self.state_norm = LatentRMSNorm(state_dim, rms_norm_eps)

        self.hidden_retrieval_proj = nn.Linear(hidden_size, code_dim, bias=False)
        self.transition_retrieval_proj = nn.Linear(code_dim, code_dim, bias=False)

        self.init_hidden_proj = nn.Linear(hidden_size, state_dim, bias=False)
        self.init_token_proj = nn.Linear(code_dim, state_dim, bias=False)
        self.ssm_hidden_proj = nn.Linear(hidden_size, state_dim, bias=False)
        self.ssm_token_proj = nn.Linear(code_dim, state_dim, bias=False)
        self.delta_proj = nn.Linear(state_dim, state_dim, bias=True)
        self.update_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.output_gate_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.slot_embedding = nn.Parameter(torch.zeros(block_size, state_dim))
        self.a_log = nn.Parameter(torch.empty(state_dim))

        self.query_hidden_proj = nn.Linear(hidden_size, state_dim, bias=False)
        self.query_state_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.query_token_proj = nn.Linear(code_dim, state_dim, bias=False)
        self.query_to_code = nn.Linear(state_dim, code_dim, bias=False)

        log_scale = math.log(logit_scale_init)
        self.retrieval_logit_scale = nn.Parameter(torch.tensor(log_scale))
        self.conditional_logit_scale = nn.Parameter(torch.tensor(log_scale))
        self.register_buffer(
            "codebook_initialized",
            torch.tensor(False, dtype=torch.bool),
        )

        self.reset_parameters(initializer_range)

    def reset_parameters(self, initializer_range: float) -> None:
        nn.init.normal_(
            self.token_codebook.weight,
            mean=0.0,
            std=initializer_range,
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=initializer_range,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        positions = torch.linspace(0.0, 1.0, self.state_dim)
        timescales = torch.exp(positions * math.log(float(self.block_size)))
        with torch.no_grad():
            self.a_log.copy_(timescales.reciprocal().log())

    @torch.no_grad()
    def initialize_codebook(self, source_weights: torch.Tensor) -> None:
        """Initialize token codes from a verifier vocabulary-weight subspace.

        Selecting evenly spaced columns gives a cheap rank-``code_dim`` factor
        ``C = W[:, column_ids]`` without materializing a full SVD of the very
        large verifier LM head.  The codebook remains fully trainable afterward.
        """
        if source_weights.shape != (
            self.target_vocab_size,
            self.hidden_size,
        ):
            raise ValueError(
                "Unexpected verifier vocabulary-weight shape: "
                f"{tuple(source_weights.shape)}."
            )
        column_ids = (
            torch.linspace(
                0,
                self.hidden_size - 1,
                self.code_dim,
                device=source_weights.device,
            )
            .round()
            .long()
        )
        codes = source_weights.index_select(1, column_ids).float()
        codes = functional.normalize(codes, dim=-1)
        self.token_codebook.weight.copy_(
            codes.to(
                device=self.token_codebook.weight.device,
                dtype=self.token_codebook.weight.dtype,
            )
        )
        self.codebook_initialized.fill_(True)

    @staticmethod
    def draft_to_target_ids(
        draft_ids: torch.Tensor,
        draft_to_target_offset: torch.Tensor | None,
    ) -> torch.Tensor:
        if draft_to_target_offset is None:
            return draft_ids
        return draft_ids + draft_to_target_offset[draft_ids]

    def context_codes(self, target_ids: torch.Tensor) -> torch.Tensor:
        return functional.normalize(
            self.token_codebook(target_ids).float(),
            dim=-1,
        ).to(self.token_codebook.weight.dtype)

    def candidate_codes(
        self,
        draft_ids: torch.Tensor,
        draft_to_target_offset: torch.Tensor | None,
    ) -> torch.Tensor:
        target_ids = self.draft_to_target_ids(
            draft_ids,
            draft_to_target_offset,
        )
        return self.context_codes(target_ids)

    def all_candidate_target_ids(
        self,
        device: torch.device,
        draft_to_target_offset: torch.Tensor | None,
    ) -> torch.Tensor:
        draft_ids = torch.arange(
            self.draft_vocab_size,
            dtype=torch.long,
            device=device,
        )
        return self.draft_to_target_ids(draft_ids, draft_to_target_offset)

    def all_candidate_codes(
        self,
        reference: torch.Tensor,
        draft_to_target_offset: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_ids = self.all_candidate_target_ids(
            reference.device,
            draft_to_target_offset,
        )
        codes = self.context_codes(target_ids).to(reference.dtype)
        return target_ids, codes

    @staticmethod
    def _scaled_cosine(
        query: torch.Tensor,
        codes: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        scale = logit_scale.float().exp().clamp(max=100.0).to(query.dtype)
        return torch.einsum("...d,...kd->...k", query, codes) * scale

    def retrieval_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return functional.normalize(
            self.hidden_retrieval_proj(self.hidden_norm(hidden_states)).float(),
            dim=-1,
        ).to(hidden_states.dtype)

    def transition_query(self, previous_target_ids: torch.Tensor) -> torch.Tensor:
        previous_codes = self.context_codes(previous_target_ids)
        return functional.normalize(
            self.transition_retrieval_proj(previous_codes).float(),
            dim=-1,
        ).to(previous_codes.dtype)

    def initial_state(
        self,
        anchor_hidden: torch.Tensor,
        anchor_target_ids: torch.Tensor,
    ) -> torch.Tensor:
        anchor_codes = self.context_codes(anchor_target_ids)
        return torch.tanh(
            self.init_hidden_proj(self.hidden_norm(anchor_hidden))
            + self.init_token_proj(anchor_codes)
        )

    def advance_state(
        self,
        hidden_states: torch.Tensor,
        previous_target_ids: torch.Tensor,
        state: torch.Tensor,
        slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous_codes = self.context_codes(previous_target_ids)
        inputs = (
            self.ssm_hidden_proj(self.hidden_norm(hidden_states))
            + self.ssm_token_proj(previous_codes)
            + self.slot_embedding[slot].to(hidden_states.dtype)
        )
        delta = functional.softplus(self.delta_proj(inputs).float())
        continuous_a = -torch.exp(self.a_log.float())
        decay = torch.exp(delta * continuous_a).to(inputs.dtype)
        update = functional.silu(self.update_proj(inputs))
        state = decay * state + (1.0 - decay) * update
        output = torch.sigmoid(self.output_gate_proj(inputs)) * state
        return state, output

    def conditional_query(
        self,
        hidden_states: torch.Tensor,
        previous_target_ids: torch.Tensor,
        state_output: torch.Tensor,
    ) -> torch.Tensor:
        previous_codes = self.context_codes(previous_target_ids)
        query_state = (
            self.query_hidden_proj(self.hidden_norm(hidden_states))
            + self.query_state_proj(state_output)
            + self.query_token_proj(previous_codes)
        )
        return functional.normalize(
            self.query_to_code(self.state_norm(query_state)).float(),
            dim=-1,
        ).to(hidden_states.dtype)

    def score_retrieval_candidates(
        self,
        query: torch.Tensor,
        candidate_ids: torch.Tensor,
        draft_to_target_offset: torch.Tensor | None,
    ) -> torch.Tensor:
        codes = self.candidate_codes(candidate_ids, draft_to_target_offset).to(
            query.dtype
        )
        return self._scaled_cosine(
            query,
            codes,
            self.retrieval_logit_scale,
        )

    def score_conditional_candidates(
        self,
        query: torch.Tensor,
        candidate_ids: torch.Tensor,
        draft_to_target_offset: torch.Tensor | None,
    ) -> torch.Tensor:
        codes = self.candidate_codes(candidate_ids, draft_to_target_offset).to(
            query.dtype
        )
        return self._scaled_cosine(
            query,
            codes,
            self.conditional_logit_scale,
        )

    def forward(
        self,
        hidden_blocks: torch.Tensor,
        block_target_token_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        draft_to_target_offset: torch.Tensor | None,
    ) -> TokenLatentTrainingOutput:
        if hidden_blocks.shape[1] != self.block_size:
            raise ValueError(
                f"Expected block_size={self.block_size}, got {hidden_blocks.shape[1]}."
            )
        previous_ids = torch.cat(
            [
                block_target_token_ids[:, :1],
                block_target_token_ids[:, :-1],
            ],
            dim=1,
        )
        hidden_query = self.retrieval_query(hidden_blocks)
        transition_query = self.transition_query(previous_ids)
        hidden_logits = self.score_retrieval_candidates(
            hidden_query,
            candidate_ids,
            draft_to_target_offset,
        )
        transition_logits = self.score_retrieval_candidates(
            transition_query,
            candidate_ids,
            draft_to_target_offset,
        )

        state = self.initial_state(
            hidden_blocks[:, 0],
            block_target_token_ids[:, 0],
        )
        conditional_logits = [torch.zeros_like(hidden_logits[:, 0])]
        for slot in range(1, self.block_size):
            state, output = self.advance_state(
                hidden_blocks[:, slot],
                previous_ids[:, slot],
                state,
                slot,
            )
            query = self.conditional_query(
                hidden_blocks[:, slot],
                previous_ids[:, slot],
                output,
            )
            conditional_logits.append(
                self.score_conditional_candidates(
                    query,
                    candidate_ids[:, slot],
                    draft_to_target_offset,
                )
            )

        return TokenLatentTrainingOutput(
            hidden_retrieval_logits=hidden_logits,
            transition_retrieval_logits=transition_logits,
            conditional_logits=torch.stack(conditional_logits, dim=1),
            candidate_ids=candidate_ids,
        )

    def prepare_inference(
        self,
        hidden_per_step: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_target_ids: torch.Tensor,
        draft_to_target_offset: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        candidate_target_ids, candidate_codes = self.all_candidate_codes(
            hidden_per_step,
            draft_to_target_offset,
        )
        hidden_query = self.retrieval_query(hidden_per_step)
        flat_query = hidden_query.flatten(0, 1)
        hidden_scores = flat_query @ candidate_codes.transpose(0, 1)
        count = min(self.hidden_candidate_count, self.draft_vocab_size)
        hidden_candidate_ids = hidden_scores.topk(count, dim=-1).indices.view(
            hidden_per_step.shape[0],
            hidden_per_step.shape[1],
            count,
        )
        return {
            "ssm_state": self.initial_state(anchor_hidden, anchor_target_ids),
            "hidden_candidate_ids": hidden_candidate_ids,
            "candidate_target_ids": candidate_target_ids,
            "candidate_codes": candidate_codes,
        }

    def select_inference_token(
        self,
        *,
        step: int,
        hidden_states: torch.Tensor,
        previous_target_ids: torch.Tensor,
        inference_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        state, output = self.advance_state(
            hidden_states,
            previous_target_ids,
            inference_state["ssm_state"],
            step + 1,
        )
        inference_state["ssm_state"] = state

        transition_query = self.transition_query(previous_target_ids)
        transition_scores = transition_query @ inference_state[
            "candidate_codes"
        ].transpose(0, 1)
        transition_count = min(
            self.transition_candidate_count,
            self.draft_vocab_size,
        )
        transition_ids = transition_scores.topk(
            transition_count,
            dim=-1,
        ).indices
        candidate_ids = torch.cat(
            [
                inference_state["hidden_candidate_ids"][:, step],
                transition_ids,
            ],
            dim=-1,
        )
        query = self.conditional_query(
            hidden_states,
            previous_target_ids,
            output,
        )
        candidate_codes = inference_state["candidate_codes"][candidate_ids]
        scores = self._scaled_cosine(
            query,
            candidate_codes,
            self.conditional_logit_scale,
        )
        selected_draft_ids = candidate_ids.gather(
            1,
            scores.argmax(dim=-1, keepdim=True),
        ).squeeze(1)
        return inference_state["candidate_target_ids"][selected_draft_ids]
