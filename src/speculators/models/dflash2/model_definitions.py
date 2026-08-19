"""Local convolution and candidate selector for the DFlash2 draft model.

Both modules mirror ``vllm/model_executor/models/qwen3_dflash2.py`` from
vllm-project/vllm#52816 -- parameter names, shapes and math are the ones the
inference side loads and evaluates, so a checkpoint trained here drafts with the
same arithmetic it was trained with. :func:`grouped_conv` and :func:`score_edges`
are deliberate verbatim ports of the reference ``_grouped_conv`` / ``_score_edges``
so the parity tests can compare against the upstream reference implementations.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    FlashAttentionKwargs,
    Qwen3Config,
)
from typing_extensions import Unpack

from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer

__all__ = [
    "CandidateSelector",
    "DFlashGroupedConv",
    "Qwen3DFlash2DecoderLayer",
    "grouped_conv",
    "score_edges",
]


def _keep_initialization(module: nn.Module) -> None:
    """Shield an explicitly initialized subtree from HF's ``post_init``.

    ``PreTrainedModel.init_weights`` walks children first, so the flag has to sit
    on every module that owns parameters, not just the root of the subtree.
    """
    for submodule in module.modules():
        submodule._is_hf_initialized = True  # type: ignore[assignment]  # noqa: SLF001


def grouped_conv(
    hidden_states: torch.Tensor,  # [num_blocks*block_size, hidden_size]
    delta: torch.Tensor,  # [num_blocks*block_size, taps, num_groups]
    base: torch.Tensor,  # [taps, hidden_size]
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:  # [num_blocks*block_size, hidden_size]
    """``out[i, c] = sum_t (base[t, c] + delta[i, t, g(c)]) * x[i - t, c]``.

    Taps are zeroed across the block boundary, so position ``i`` only ever mixes
    in positions from its own draft block. Channels share a dynamic coefficient
    within a group of ``group_size``; the static part is per channel.
    """
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output = output + coefficients[:, tap] * shifted * (position >= tap).view(
            -1, 1, 1
        )
    return output.flatten(-2)


def score_edges(
    predecessor_table: torch.Tensor,  # [vocab_size, rank]
    successor_table: torch.Tensor,  # [vocab_size, rank]
    candidate_ids: torch.Tensor,  # [num_blocks, steps, top_k]
    unary_logits: torch.Tensor,  # [num_blocks, steps, top_k]
    hidden: torch.Tensor,  # [num_blocks, steps, rank]
    anchor_token_ids: torch.Tensor,  # [num_blocks]
    top_k: int,
) -> torch.Tensor:  # [num_blocks, steps, top_k, top_k]
    """``edge(p -> c) = <A[p] * project(h), B[c]> + unary[c]`` over the kept top-K.

    Step 0's predecessors are the verified anchor token; every later step's are the
    previous step's candidates. Mirrors the reference implementation exactly and is
    only used for the inference-shaped diagnostics; the training loss goes through
    :meth:`CandidateSelector.block_bias`, which scores the whole vocabulary.
    """
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


class DFlashGroupedConv(nn.Module):
    """Grouped dynamic depthwise convolution wrapped around one sublayer.

    ``prepare`` convolves the sublayer's input and hands back the coefficients for
    ``finish``, which convolves the sublayer's output; both coefficient sets come
    from one projection of the input, and ``base_kernel[side]`` is the static part
    of each. Weight names match vLLM's ``DFlashGroupedConv`` so the exported
    checkpoint loads unchanged.
    """

    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
    ) -> None:
        super().__init__()
        if taps < 1:
            raise ValueError(f"conv_kernel_size must be >= 1, got {taps}")
        if taps > block_size:
            raise ValueError(
                f"conv_kernel_size={taps} must not exceed block_size={block_size}; "
                "taps beyond the block boundary are always zero."
            )
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, taps, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * taps * self.num_groups, bias=False
        )
        self.reset_parameters()
        # Keep the identity initialization: HF's post_init would otherwise
        # re-initialize these from initializer_range.
        _keep_initialization(self)

    def reset_parameters(self) -> None:
        """Initialize to the identity so a fresh DFlash2 block is a DFlash block.

        Tap 0's static coefficient is 1 and every other static tap is 0, and the
        projection producing the dynamic part starts at 0. The projection still
        receives gradient (the delta enters the output linearly), so it moves off
        zero from the first step -- this only fixes where training starts, which is
        what makes DFlash weights a usable starting point (see
        :class:`~speculators.models.dflash2.core.DFlash2DraftModel` for how to load
        them).
        """
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0].fill_(1.0)
            self.kernel_projection.weight.zero_()

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        output = grouped_conv(
            flat,
            delta.reshape(flat.shape[0], self.taps, self.num_groups),
            self.base_kernel[side].to(flat.dtype),
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )
        return output.view_as(hidden_states)

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolve the sublayer input; return it with ``finish``'s coefficients."""
        coefficients = self.kernel_projection(hidden_states).reshape(
            -1, 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        """Convolve the sublayer output with the coefficients from ``prepare``."""
        return self._convolve(hidden_states, coefficients, 1)


class CandidateSelector(nn.Module):
    """Low-rank, predecessor-conditioned logit correction.

    ``bias[p, c] = <A[p] * project(h), B[c]>`` where ``A`` is the predecessor
    codebook, ``B`` the successor codebook and ``p`` the token occupying the
    previous draft slot. Inference keeps only the target head's top-K per slot and
    walks the best path through those transitions; training scores the whole
    vocabulary (:meth:`block_bias`), which is the same function evaluated on every
    token instead of on the K the walk happens to see, and so gives every token
    gradient. :meth:`edge_scores` is the walk's own K-by-K view, used by the
    diagnostics that measure the restricted decision.

    ``predecessor_codebook`` is indexed by verifier-vocabulary ids (the previous
    token) and ``successor_codebook`` by draft-vocabulary ids (it adds onto the
    draft logits). vLLM's selector holds both as ``[vocab_size, rank]``, so a
    servable checkpoint needs the two vocabularies to be the same size --
    :class:`~speculators.models.dflash2.core.DFlash2DraftModel` enforces that.
    """

    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        hidden_size: int,
        rank: int,
        top_k: int,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"selector_rank must be > 0, got {rank}")
        if top_k < 2:  # noqa: PLR2004
            raise ValueError(
                f"selector_top_k must be >= 2 for a transition to exist, got {top_k}"
            )
        self.rank = rank
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(verifier_vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(draft_vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        self.reset_parameters()
        _keep_initialization(self)

    def reset_parameters(self) -> None:
        """Start with a zero correction, LoRA-style.

        ``successor_codebook`` is zero, so the bias is exactly 0 and a fresh DFlash2
        drafts identically to a DFlash -- which is what makes DFlash weights a
        usable starting point (see
        :class:`~speculators.models.dflash2.core.DFlash2DraftModel`).

        The bias is a product of three factors, so zeroing one of them necessarily
        zeroes the gradient to the other two on the first step: only the successor
        codebook (and the convolution) move at step 0, and the predecessor codebook
        and the projection start moving once the successor side is nonzero. This is
        LoRA's ``B = 0`` situation and costs one optimizer step; the alternative --
        three small nonzero factors -- would trade that for a random perturbation
        of the logits at initialization.
        """
        with torch.no_grad():
            self.predecessor_codebook.normal_(mean=0.0, std=self.rank**-0.5)
            self.successor_codebook.zero_()
            self.hidden_projection.weight.normal_(
                mean=0.0, std=self.hidden_projection.in_features**-0.5
            )

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The rank-space gate ``project(h)``, shape ``[..., rank]``."""
        return self.hidden_projection(hidden_states)

    def block_bias(
        self,
        prev_token_ids: torch.Tensor,  # [num_blocks, block_size]
        hidden_states: torch.Tensor,  # [num_blocks, block_size, hidden_size]
    ) -> torch.Tensor:  # [num_blocks, block_size, draft_vocab_size]
        """Full-vocabulary additive bias for the training loss."""
        gate = self.predecessor_codebook[prev_token_ids.long()] * self.project(
            hidden_states
        )
        return gate @ self.successor_codebook.transpose(0, 1).to(gate.dtype)

    def edge_scores(
        self,
        candidate_ids: torch.Tensor,  # [num_blocks, steps, top_k]
        unary_logits: torch.Tensor,  # [num_blocks, steps, top_k]
        hidden_states: torch.Tensor,  # [num_blocks, steps, hidden_size]
        anchor_token_ids: torch.Tensor,  # [num_blocks]
    ) -> torch.Tensor:  # [num_blocks, steps, top_k, top_k]
        """The inference-shaped K-by-K transition scores (diagnostics only)."""
        return score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            self.project(hidden_states),
            anchor_token_ids,
            self.top_k,
        )


class Qwen3DFlash2DecoderLayer(Qwen3DFlashDecoderLayer):
    """DFlash decoder layer with a grouped conv around attention and around the MLP.

    The convolution sees only the draft block's own query positions -- the target
    context enters through ``target_hidden`` in attention and is never convolved,
    matching the inference side where the context lives in the KV cache.
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        conv_kernel_size: int,
        conv_group_size: int,
        block_size: int,
    ) -> None:
        super().__init__(config, layer_idx)
        conv_args = {
            "hidden_size": config.hidden_size,
            "taps": conv_kernel_size,
            "group_size": conv_group_size,
            "block_size": block_size,
        }
        self.attention_conv = DFlashGroupedConv(**conv_args)  # type: ignore[arg-type]
        self.mlp_conv = DFlashGroupedConv(**conv_args)  # type: ignore[arg-type]

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        # necessary, but kept here for BC
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        assert hidden_states is not None  # noqa: S101
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return residual + hidden_states  # type: ignore[operator,return-value]
