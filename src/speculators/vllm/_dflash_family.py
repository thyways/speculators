"""Shared vLLM plumbing for DFlash-family speculators that sample serially.

DFlash's block is sampled in parallel, but DSpark, DFly, and Domino all make
position ``k``'s logits depend on the token sampled at ``k-1``, so they run on
vLLM's DSpark speculator, which owns that left-to-right loop. Speculators
extends that loop by monkey-patching ``DSparkSpeculator._sample_sequential``.

That patch is a single method slot, and vLLM loads general plugins in an
unspecified order, so every algorithm **must** share one implementation and one
patch marker. Two plugins each installing their own version would silently
leave whichever loaded first broken. The same applies to the wrapper around
``SpeculatorsConfig.build_vllm_speculative_config``.

Algorithms therefore register into this module and let it patch vLLM once:

* :func:`register_config_updater` -- Speculators-config -> vLLM draft config.
* :func:`register_speculative_method_alias` -- route ``method`` to a runtime.
* :func:`register_init_hook` -- adjust a constructed speculator in place.
* :func:`register_speculative_config_updater` -- finalize the vLLM proposal config.
* :func:`install_config_patches` / :func:`install_speculator_patches` --
  idempotent, safe to call from every plugin's ``register()``.

The shared sampler supports three composable per-step corrections, discovered
by duck typing so a model only implements what it needs:

* ``has_hidden_correction`` -- correct the hidden state, then run the LM head
  (DFly). Forces a per-step LM head call.
* ``has_markov`` -- memoryless additive logit bias from the previous token
  (DSpark). Defaults to True, matching vLLM's own DSpark model.
* ``has_recurrent_logits_correction`` -- additive logit correction driven by a
  state carried across steps (Domino). Needs
  ``init_recurrent_state`` / ``advance_recurrent_state`` /
  ``recurrent_logit_correction``.
* ``has_candidate_selector`` -- bypass the full-vocabulary LM head and let a
  token-code candidate decoder select the next target-vocabulary token directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# vLLM 0.26+ treats a method=dspark architecture that does not also advertise
# Qwen3DSparkModel as a DeepSeek-v4 embedded head. Family models keep this as a
# secondary architecture sentinel to prevent that rewrite; registry resolution
# still picks the algorithm-specific architecture first.
DSPARK_COMPAT_ARCH = "Qwen3DSparkModel"

_CONFIG_PATCH_MARKER = "_speculators_family_config_patched"
_SPECULATOR_PATCH_MARKER = "_speculators_family_speculator_patched"

# method name in a checkpoint -> the vLLM runtime that can execute it
_METHOD_ALIASES: dict[str, str] = {}
# hooks run after DSparkSpeculator.__init__, in registration order
_INIT_HOOKS: list[Callable[[Any, torch.device], None]] = []
# algorithm name -> final mutation of vLLM's speculative-config dictionary
_SPECULATIVE_CONFIG_UPDATERS: dict[
    str,
    Callable[[dict[str, Any], dict[str, Any]], None],
] = {}


def register_speculative_method_alias(method: str, runtime: str) -> None:
    """Route a checkpoint's ``method`` onto an existing vLLM runtime."""
    _METHOD_ALIASES[method] = runtime


def register_init_hook(hook: Callable[[Any, torch.device], None]) -> None:
    """Register a post-``__init__`` fixup for the shared DSpark speculator.

    Hooks must inspect the speculator's draft config and no-op for models they
    do not own, since every hook runs for every constructed speculator.
    """
    if hook not in _INIT_HOOKS:
        _INIT_HOOKS.append(hook)


def register_speculative_config_updater(
    algorithm: str,
    updater: Callable[[dict[str, Any], dict[str, Any]], None],
) -> None:
    """Finalize one algorithm's vLLM speculative config after translation."""
    _SPECULATIVE_CONFIG_UPDATERS[algorithm] = updater


def register_config_updater(
    algorithm: str,
    updater: Callable[[dict[str, Any], dict[str, Any]], None],
) -> None:
    """Teach vLLM how to translate one Speculators algorithm's config."""
    from vllm.transformers_utils.configs.speculators.algos import (  # noqa: PLC0415
        SUPPORTED_SPECULATORS_TYPES,
    )

    SUPPORTED_SPECULATORS_TYPES.setdefault(algorithm, updater)


def map_speculative_method(config: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``method`` to the runtime registered for it, if any."""
    method = config.get("method")
    runtime = _METHOD_ALIASES.get(method) if isinstance(method, str) else None
    if runtime is None:
        return config
    config = dict(config)
    config["method"] = runtime
    return config


def finalize_speculative_config(
    config_dict: dict[str, Any],
    speculative_config: dict[str, Any],
) -> dict[str, Any]:
    """Apply the shared method alias and an algorithm-specific finalizer."""
    speculative_config = map_speculative_method(speculative_config)
    algorithm = config_dict.get("speculators_model_type")
    updater = (
        _SPECULATIVE_CONFIG_UPDATERS.get(algorithm)
        if isinstance(algorithm, str)
        else None
    )
    if updater is not None:
        speculative_config = dict(speculative_config)
        updater(config_dict, speculative_config)
    return speculative_config


def normalize_rope(pre_trained_config: dict[str, Any]) -> None:
    """Convert target M-RoPE metadata to the drafter's linear positions."""
    rope_parameters = pre_trained_config.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        return
    if "mrope_section" not in rope_parameters:
        return

    rope_parameters = dict(rope_parameters)
    rope_parameters.pop("mrope_section", None)
    rope_parameters.pop("mrope_interleaved", None)
    pre_trained_config["rope_parameters"] = rope_parameters


def preserve_dspark_anchor_mode(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Propagate the checkpoint's DSpark anchor sampling convention."""
    if "sample_from_anchor" not in config_dict:
        return

    sample_from_anchor = config_dict["sample_from_anchor"]
    if not isinstance(sample_from_anchor, bool):
        raise TypeError("DSpark sample_from_anchor must be a boolean.")

    pre_trained_config["sample_from_anchor"] = sample_from_anchor
    pre_trained_config["dspark_bonus_anchor"] = not sample_from_anchor


def resolve_dspark_target_hidden_size(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Fill the target width omitted by same-width DSpark checkpoints."""
    target_hidden_size = config_dict.get("target_hidden_size")
    if target_hidden_size is None:
        target_hidden_size = pre_trained_config.get("hidden_size")
    if not isinstance(target_hidden_size, int) or target_hidden_size <= 0:
        raise ValueError("DSpark target_hidden_size must be a positive integer.")
    pre_trained_config["target_hidden_size"] = target_hidden_size


def propagate_intra_block_causality(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
    default: bool | None = None,
) -> None:
    """Match vLLM's per-layer draft mask to the training-time mask.

    Training always makes full-attention draft layers non-causal. The
    ``sliding_window_non_causal`` flag only controls sliding-window layers.
    vLLM's ``dflash_config.causal`` is instead a *global* override, so writing
    ``causal=True`` when the flag is false incorrectly turns full-attention
    layers causal too. That is the train/serve mismatch that hurts acceptance
    for all-full checkpoints.

    When sliding-window attention was trained non-causally, set the global
    override to ``False``. Otherwise remove the override and let vLLM's
    per-layer fallback keep full-attention layers non-causal and sliding-window
    layers causal.

    ``default`` is the ``sliding_window_non_causal`` value to assume when the
    checkpoint omits the field. Leave it ``None`` to skip untouched -- correct
    for DSpark, whose updater also serves external checkpoints that never had
    the field and rely on vLLM's own fallback. Speculators-trained configs
    always serialize it, so this only affects hand-written configs.
    """
    non_causal = config_dict.get("sliding_window_non_causal", default)
    if non_causal is None:
        return
    if not isinstance(non_causal, bool):
        raise TypeError("sliding_window_non_causal must be a boolean.")

    had_dflash_config = "dflash_config" in pre_trained_config
    dflash_config = dict(pre_trained_config.get("dflash_config") or {})
    if non_causal:
        dflash_config["causal"] = False
    else:
        dflash_config.pop("causal", None)

    if dflash_config or had_dflash_config:
        pre_trained_config["dflash_config"] = dflash_config


def _emit_step_token(
    self: Any,
    *,
    num_reqs: int,
    logits_step: torch.Tensor,
    idx_map: torch.Tensor,
    sample_pos: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """Turn one step's logits into a target-vocabulary token id."""
    if self.draft_logits is None:
        return self.model.map_draft_to_target(logits_step.argmax(dim=-1))

    if self._d2t_scatter_index is not None:
        if self._draft_scatter_buf is None:
            raise RuntimeError("Missing reduced-vocabulary scatter buffer.")
        scatter = self._draft_scatter_buf[:num_reqs]
        scatter.index_copy_(
            1,
            self._d2t_scatter_index,
            logits_step.to(scatter.dtype),
        )
        logits_step = scatter

    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample  # noqa: PLC0415

    # sample_pos is the predicted token's position Q; the target verifies it
    # with the predecessor's Gumbel key, so pass Q-1.
    return gumbel_sample(
        logits_step,
        idx_map[:, step],
        self.temperature,
        self.seeds,
        sample_pos[:, step] - 1,
        apply_temperature=True,
        output_processed_logits=self.draft_logits,
        output_processed_logits_col=self._step_cols[step],
        use_fp64=self.use_fp64_gumbel,
    )


def _sample_candidate_block(
    self: Any,
    *,
    num_reqs: int,
    head_hidden: torch.Tensor,
    hidden_per_step: torch.Tensor,
) -> None:
    """Run a no-base-logits candidate decoder left to right."""
    if self.draft_logits is not None:
        raise NotImplementedError(
            "Token-latent candidate selection currently supports greedy drafting only."
        )
    prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]].long()
    anchor_hidden = head_hidden[self._anchor_idx[:num_reqs]]
    selector_state = self.model.init_candidate_selector(
        hidden_per_step,
        anchor_hidden,
        prev,
    )
    for step in range(self.num_speculative_steps):
        sampled = self.model.select_candidate_token(
            step,
            hidden_per_step[:, step],
            prev,
            selector_state,
        )
        self.draft_tokens[:num_reqs, step] = sampled
        prev = sampled


def sample_sequential_block(  # noqa: C901
    self: Any,
    num_reqs: int,
    head_hidden: torch.Tensor,
) -> None:
    """Sample one parallel block left to right for every DFlash-family model."""
    num_steps = self.num_speculative_steps
    num_samples = num_reqs * num_steps
    sample_hidden = head_hidden[self.sample_indices[:num_samples]]
    hidden_per_step = sample_hidden.view(num_reqs, num_steps, -1)

    model = self.model
    has_candidate_selector = bool(
        getattr(model, "has_candidate_selector", lambda: False)()
    )
    if has_candidate_selector:
        _sample_candidate_block(
            self,
            num_reqs=num_reqs,
            head_hidden=head_hidden,
            hidden_per_step=hidden_per_step,
        )
        return

    has_hidden_correction = bool(
        getattr(model, "has_hidden_correction", lambda: False)()
    )
    has_markov = bool(getattr(model, "has_markov", lambda: True)())
    has_recurrent = bool(
        getattr(model, "has_recurrent_logits_correction", lambda: False)()
    )

    base_logits = None
    if not has_hidden_correction:
        # Base logits do not depend on the sampled prefix, so the LM head runs
        # once for the whole block instead of once per step.
        logits = model.compute_draft_logits(sample_hidden)
        base_logits = logits.view(num_reqs, num_steps, -1)

    idx_map = self.sample_idx_mapping[:num_samples].view(num_reqs, num_steps)
    sample_pos = self.sample_pos[:num_samples].view(num_reqs, num_steps)
    prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]].long()

    state = None
    if has_recurrent:
        state = model.init_recurrent_state(num_reqs, hidden_per_step)

    for step in range(num_steps):
        if has_recurrent:
            # Consume the previous token before scoring this step, so the state
            # has seen exactly the tokens preceding this position -- the same
            # alignment the training-time scan produces.
            state = model.advance_recurrent_state(prev, state)

        if has_hidden_correction:
            corrected = model.apply_hidden_correction(hidden_per_step[:, step], prev)
            logits_step = model.compute_draft_logits(corrected)
        else:
            if base_logits is None:
                raise RuntimeError("Missing precomputed draft logits.")
            logits_step = base_logits[:, step]

        if has_recurrent:
            # None for the leading uncorrected slots of the block.
            correction = model.recurrent_logit_correction(
                step,
                hidden_per_step[:, step],
                state,
            )
            if correction is not None:
                logits_step = logits_step + correction

        if has_markov:
            markov_embed = model.markov_embed(prev)
            logits_step = logits_step + model.markov_bias(markov_embed)

        sampled = _emit_step_token(
            self,
            num_reqs=num_reqs,
            logits_step=logits_step,
            idx_map=idx_map,
            sample_pos=sample_pos,
            step=step,
        )
        self.draft_tokens[:num_reqs, step] = sampled
        prev = sampled


def install_config_patches() -> None:
    """Patch vLLM's speculators-config layer once, for all family algorithms."""
    from vllm.transformers_utils.configs.speculators.algos import (  # noqa: PLC0415
        SUPPORTED_SPECULATORS_TYPES,
    )
    from vllm.transformers_utils.configs.speculators.base import (  # noqa: PLC0415
        SpeculatorsConfig,
    )

    if getattr(SpeculatorsConfig, _CONFIG_PATCH_MARKER, False):
        return

    original_dflash_updater = SUPPORTED_SPECULATORS_TYPES["dflash"]
    original_dspark_updater = SUPPORTED_SPECULATORS_TYPES["dspark"]

    def update_dflash(
        config_dict: dict[str, Any],
        pre_trained_config: dict[str, Any],
    ) -> None:
        original_dflash_updater(config_dict, pre_trained_config)
        # Upstream currently translates ``False`` into global ``causal=True``.
        # Repair that override after running the rest of its config mapping.
        propagate_intra_block_causality(config_dict, pre_trained_config)

    def update_dspark(
        config_dict: dict[str, Any],
        pre_trained_config: dict[str, Any],
    ) -> None:
        original_dspark_updater(config_dict, pre_trained_config)
        preserve_dspark_anchor_mode(config_dict, pre_trained_config)
        resolve_dspark_target_hidden_size(config_dict, pre_trained_config)
        propagate_intra_block_causality(config_dict, pre_trained_config)

    SUPPORTED_SPECULATORS_TYPES["dflash"] = update_dflash
    SUPPORTED_SPECULATORS_TYPES["dspark"] = update_dspark

    original = SpeculatorsConfig.build_vllm_speculative_config.__func__

    @classmethod  # type: ignore[misc]
    def build_vllm_speculative_config(
        cls: type,
        config_dict: dict[str, Any],
    ) -> dict[str, Any]:
        return finalize_speculative_config(
            config_dict,
            original(cls, config_dict),
        )

    SpeculatorsConfig.build_vllm_speculative_config = build_vllm_speculative_config
    setattr(SpeculatorsConfig, _CONFIG_PATCH_MARKER, True)


def install_speculator_patches() -> None:
    """Install the shared sequential sampler and init-hook dispatch once."""
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (  # noqa: PLC0415
        DSparkSpeculator,
    )

    if getattr(DSparkSpeculator, _SPECULATOR_PATCH_MARKER, False):
        return

    original_init = DSparkSpeculator.__init__

    def init(self: Any, vllm_config: Any, device: torch.device) -> None:
        original_init(self, vllm_config, device)
        for hook in _INIT_HOOKS:
            hook(self, device)

    DSparkSpeculator.__init__ = init
    DSparkSpeculator._sample_sequential = (  # noqa: SLF001
        sample_sequential_block
    )
    setattr(DSparkSpeculator, _SPECULATOR_PATCH_MARKER, True)
