"""DFlash2's proposal path for vLLM: the candidate walk.

A port of ``vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py`` from
vllm-project/vllm#52816. The draft runs as plain DFlash -- one forward over the
``1 + num_speculative_tokens`` query block -- and then, instead of an independent
argmax per slot, the selector scores every transition between adjacent slots'
top-K candidates and this walk picks a path from the verified anchor. The walk is
one Triton program per request: a slot's K scores stay in registers and the
slot-to-slot dependency is a loop inside the program rather than a kernel per
slot.

Two things the PR gets from vLLM that this module carries itself, so it runs
against a vLLM that predates the PR:

* :func:`gumbel_noised_argmax`, which #52816 factors out of
  ``gumbel_block_argmax`` into ``v1/worker/gpu/sample/gumbel.py``.
* the ``draft_logits_spec`` hook on ``DraftModelSpeculator``. The base allocates
  the proposal cache at the head dtype filled with zeros;
  :meth:`DFlash2Speculator._install_draft_logits` replaces it with the fp32,
  ``-inf`` buffer the walk needs. When the hook is present the base already
  allocated it correctly and only the fill is restated.
"""

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32, tl_rand64
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

__all__ = ["DFlash2Speculator"]


@triton.jit
def gumbel_noised_argmax(
    logits,
    keys,
    mask,
    seed,
    pos,
    temp,
    USE_FP64: tl.constexpr,  # noqa: N803
    APPLY_TEMPERATURE: tl.constexpr = True,  # noqa: N803
):
    """Argmax of logits under Gumbel-max sampling, or plain argmax at temp 0.

    ``keys`` indexes the noise, so the same token draws the same noise wherever it
    appears; ``pos`` and ``seed`` place the draw in the request's stream, which is
    what lets a draft and its verification agree.
    """
    if temp != 0.0 and APPLY_TEMPERATURE:
        # Match the behavior of _temperature_kernel: if that kernel uses
        # tl.div_rn, this must too.
        logits = logits / temp

    # fp32 is the default reduction dtype; fp64 is ~1/32-1/64x the throughput
    # on H100/Ada/Blackwell and empirically indistinguishable for Gumbel-max.
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        gumbel_seed = tl.randint(seed, pos)
        if USE_FP64:
            u = tl_rand64(gumbel_seed, keys, includes_zero=False)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(gumbel_seed, keys, includes_zero=False)
            # Draw the large-noise tail (which decides the argmax winner) from
            # u -> 0, where fp32 has fine resolution, instead of u -> 1, where
            # fp32 spacing is ~2**-24. The naive `-log(-log(u))` puts the winning
            # tail at u -> 1, hard-capping the noise at ~16.6 and coarsely
            # quantizing it; `log1p(-u)` == `log(1 - u)` keeps the tail in the
            # well-resolved region. `1 - u` would lose precision for small u, so
            # log1p is required.
            gumbel_noise = -tl.log(-tldevice.log1p(-u))
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    return tl.max(logits, axis=0, return_indices=True)


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,  # noqa: N803
    SAMPLE_PROBABILISTIC: tl.constexpr,  # noqa: N803
    USE_FP64: tl.constexpr,  # noqa: N803
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        # Load at the width the argmax will reduce in. Loading fp32 and letting
        # the noise promote to fp64 gives the two arms of that branch different
        # types, which Triton rejects on ROCm.
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float64 if USE_FP64 else tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        # The candidate token ids key the noise, so a token drawn at this slot
        # gets the same noise the target's own sampling would give it.
        position = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            position,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            USE_FP64=USE_FP64,
        )

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,  # noqa: N803
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    cache_base = (req_state * num_steps + step) * top_k
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._selector_scores = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            self.selector_top_k,
            dtype=torch.float32,
            device=device,
        )
        self._cached_candidate_ids = torch.zeros(
            self._selector_scores.shape, dtype=torch.int64, device=device
        )
        self._install_draft_logits(vllm_config, device)

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # fp32, not the head dtype. Rounding real selector scores to bf16 moves
        # the argmax of a candidate row 0.81% of the time and reverses the order
        # of 0.68% of candidate pairs, so the walk and the rejection that checks
        # it would no longer read the same distribution. The fill is -inf because
        # the cache kernel writes only the K candidates, and every column it
        # never touches has to read as impossible.
        return torch.float32, -float("inf")

    def _install_draft_logits(
        self, vllm_config: VllmConfig, device: torch.device
    ) -> None:
        """Give the walk a proposal cache it can write a subset of.

        Restated rather than trusted: a base that allocates at the head dtype, or
        a construction path that skips the allocation, would otherwise hand the
        walk memory where it needs every unwritten column to be impossible.
        """
        if self.draft_logits is None:
            return
        dtype, fill = self.draft_logits_spec(vllm_config)
        if self.draft_logits.dtype is not dtype:
            self.draft_logits = torch.full(
                tuple(self.draft_logits.shape), fill, dtype=dtype, device=device
            )
        else:
            self.draft_logits.fill_(fill)

    def _sample_path(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
    ) -> None:
        block_k = triton.next_power_of_2(self.selector_top_k)
        _selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self._selector_scores,
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=self.draft_logits is not None,
            USE_FP64=self.use_fp64_gumbel,
            num_warps=1,
        )

    def _cache_draft_logits(self, candidate_ids: torch.Tensor, num_sample: int) -> None:
        draft_logits = self.draft_logits
        assert draft_logits is not None  # noqa: S101
        block_k = triton.next_power_of_2(self.selector_top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            draft_logits,
            self._cached_candidate_ids,
            candidate_ids,
            self._selector_scores,
            self.sample_idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.selector_top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_token_ids,
        )
        self._sample_path(candidate_ids, scores, num_reqs)
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_sample)
