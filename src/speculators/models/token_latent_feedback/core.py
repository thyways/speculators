"""Training model for the block-parallel token-latent feedback method."""

from __future__ import annotations

from typing import ClassVar, Literal, cast

import torch
from torch.nn import functional
from transformers import PretrainedConfig

from speculators.losses import LossConfig, resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.token_latent_feedback.config import (
    DEFAULT_LATENT_DIM,
    DEFAULT_LATENT_LOSS_ALPHA,
    LatentFeedbackSpeculatorConfig,
    ParallelTokenLatentFeedbackSpeculatorConfig,
    ParallelTokenLatentSpeculatorConfig,
    TokenLatentFeedbackSpeculatorConfig,
)
from speculators.models.token_latent_feedback.metrics import (
    compute_latent_cosine_loss,
)
from speculators.models.token_latent_feedback.model_definitions import (
    TokenLatentFeedbackHead,
)
from speculators.models.utils import conditional_torch_compile

__all__ = [
    "LatentFeedbackDraftModel",
    "ParallelTokenLatentDraftModel",
    "ParallelTokenLatentFeedbackDraftModel",
    "TokenLatentFeedbackDraftModel",
]


_DEFAULT_MAIN_LOSS = '{"ce": 0.1, "tv": 0.9}'
_FLAT_HIDDEN_RANK = 2
_BLOCK_HIDDEN_RANK = 3
_DEFAULT_FUSED_LOSS_CONFIG: LossConfig = resolve_loss_config(
    _DEFAULT_MAIN_LOSS, "fused"
)
_DEFAULT_EAGER_LOSS_CONFIG: LossConfig = resolve_loss_config(
    _DEFAULT_MAIN_LOSS, "eager"
)


@SpeculatorModel.register("token_latent_feedback")
class TokenLatentFeedbackDraftModel(DFlashDraftModel):
    """DFlash with one parallel causal token-latent feedback stage."""

    config_class: ClassVar[type[TokenLatentFeedbackSpeculatorConfig]] = (  # type: ignore[misc,assignment]
        TokenLatentFeedbackSpeculatorConfig
    )
    algorithm_name: ClassVar[str] = "token_latent_feedback"
    _no_split_modules = ["TokenLatentFeedbackHead"]
    _target_code_projection: torch.Tensor

    def __init__(self, config: TokenLatentFeedbackSpeculatorConfig) -> None:
        super().__init__(config=config)
        transformer_config = config.transformer_layer_config
        self.token_latent_head = TokenLatentFeedbackHead(
            hidden_size=int(transformer_config.hidden_size),
            latent_dim=int(config.resolved_latent_dim),
            block_size=int(config.block_size),
            rms_norm_eps=float(transformer_config.rms_norm_eps),
            initializer_range=float(transformer_config.initializer_range or 0.02),
            prefix_mixer_mode=config.resolved_prefix_mixer_mode,
            use_reliability_gate=config.resolved_reliability_gate,
            strict_causal_prefix=config.strict_causal_prefix,
            position_scale_init=float(config.position_scale_init),
            feedback_output_projection_init=float(
                config.feedback_output_projection_init
            ),
        )

        # R in c_v = Normalize(R w_v) is training-only supervision.  Keeping it
        # as a frozen buffer makes resumed checkpoints exact. Initialization uses
        # the same global RNG seeded by the trainer's ordinary ``--seed`` flag;
        # the vLLM loader drops it because inference never uses token codes or
        # vocabulary retrieval.
        latent_dim = int(config.resolved_latent_dim)
        hidden_size = int(transformer_config.hidden_size)
        if latent_dim <= hidden_size:
            random_basis = torch.randn(
                hidden_size,
                latent_dim,
                dtype=torch.float32,
            )
            projection = torch.linalg.qr(random_basis, mode="reduced").Q.transpose(0, 1)
        else:
            projection = functional.normalize(
                torch.randn(
                    latent_dim,
                    hidden_size,
                    dtype=torch.float32,
                ),
                dim=-1,
            )
        self.register_buffer(
            "_target_code_projection",
            projection,
            persistent=True,
        )

    @property
    def latent_dim(self) -> int:
        """Width of the token-intent latent stream."""
        return int(self.token_latent_head.latent_dim)

    @property
    def feedback_head(self) -> TokenLatentFeedbackHead:
        """Compatibility alias for the v1.2 feedback module."""
        return self.token_latent_head

    def _reshape_feedback_blocks(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == _FLAT_HIDDEN_RANK:
            if hidden_states.shape[0] % self.block_size:
                raise ValueError(
                    "flattened hidden states must contain complete blocks: "
                    f"{hidden_states.shape[0]} % {self.block_size} != 0"
                )
            return hidden_states.reshape(-1, self.block_size, hidden_states.shape[-1])
        if (
            hidden_states.ndim != _BLOCK_HIDDEN_RANK
            or hidden_states.shape[1] != self.block_size
        ):
            raise ValueError(
                "hidden states must have shape [blocks, block_size, hidden] or "
                f"[blocks*{self.block_size}, hidden], got {tuple(hidden_states.shape)}"
            )
        return hidden_states

    def apply_token_latent_feedback(
        self,
        raw_hidden_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the head and return corrected hidden states plus source latents."""
        output = self.token_latent_head(raw_hidden_blocks)
        return output.hidden_states, output.latents

    def predict_token_latents(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return normalized token-intent latents for complete blocks."""
        blocks = self._reshape_feedback_blocks(hidden_states)
        return self.token_latent_head(blocks).latents

    def load_verifier_weights(self) -> None:
        """Load the frozen verifier projection used by latent supervision."""
        super().load_verifier_weights()

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> TokenLatentFeedbackDraftModel:
        algorithm = getattr(cls, "algorithm_name", "token_latent_feedback")
        base = cls._build_base_config_kwargs(algorithm, verifier_config, **kwargs)
        # v1.2 always treats the leading slot as the verified anchor.  Respect an
        # explicit true value by letting the config validator report a clear error.
        sample_from_anchor = kwargs.get("sample_from_anchor")
        base["sample_from_anchor"] = (
            False if sample_from_anchor is None else sample_from_anchor
        )
        latent_dim = kwargs.get("latent_dim")
        if latent_dim is None:
            latent_dim = DEFAULT_LATENT_DIM
        feedback_stages = kwargs.get("feedback_stages")
        if feedback_stages is None:
            feedback_stages = 1
        prefix_mixer_mode = kwargs.get("prefix_mixer_mode")
        if prefix_mixer_mode is None:
            prefix_mixer_mode = "full"
        prefix_mixer_mode = cast(
            "Literal['full', 'shifted', 'none']",
            prefix_mixer_mode,
        )
        use_reliability_gate = kwargs.get("use_reliability_gate")
        if use_reliability_gate is None:
            use_reliability_gate = True
        latent_loss_alpha = kwargs.get("latent_loss_alpha")
        if latent_loss_alpha is None:
            latent_loss_alpha = DEFAULT_LATENT_LOSS_ALPHA
        strict_causal_prefix = kwargs.get("strict_causal_prefix")
        if strict_causal_prefix is None:
            strict_causal_prefix = True
        position_scale_init = kwargs.get("position_scale_init")
        if position_scale_init is None:
            position_scale_init = 1.0
        feedback_output_init = kwargs.get("feedback_output_projection_init")
        if feedback_output_init is None:
            feedback_output_init = 0.0
        config = cls.config_class(
            **base,
            latent_dim=latent_dim,
            token_latent_dim=kwargs.get("token_latent_dim"),
            feedback_stages=feedback_stages,
            latent_feedback_stages=kwargs.get("latent_feedback_stages"),
            prefix_mixer_mode=prefix_mixer_mode,
            feedback_mode=kwargs.get("feedback_mode"),
            prefix_mixer=kwargs.get("prefix_mixer"),
            prefix_mixer_parameterization=kwargs.get("prefix_mixer_parameterization")
            or "toeplitz",
            use_reliability_gate=use_reliability_gate,
            reliability_gate=kwargs.get("reliability_gate"),
            strict_causal_prefix=strict_causal_prefix,
            feedback_output_projection_init=feedback_output_init,
            position_scale_init=position_scale_init,
            latent_loss_alpha=latent_loss_alpha,
            latent_loss_weight=kwargs.get("latent_loss_weight"),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        implementation = kwargs.get("loss_implementation", "fused")
        loss_spec = kwargs.get("loss_fn") or _DEFAULT_MAIN_LOSS
        loss_config = resolve_loss_config(loss_spec, implementation)
        shared = {
            "loss_config": loss_config,
            "gamma": kwargs.get("dflash_decay_gamma", 4.0),
            "max_anchors": kwargs.get("max_anchors", 512),
            "per_position_loss_weight": kwargs.get(
                "per_position_loss_weight", "fixed-exp-decay"
            ),
            "dpace_alpha": kwargs.get("dpace_alpha", 0.5),
            "latent_loss_alpha": kwargs.get(
                "latent_loss_alpha", DEFAULT_LATENT_LOSS_ALPHA
            ),
        }
        return dict(shared), dict(shared)

    def _compute_latent_loss(
        self,
        latents: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align predicted latents with hard verifier-token codes."""
        target_ids = targets.argmax(dim=-1)
        target_weights = self.verifier_lm_head.weight[target_ids]
        embedding_weights = self.embed_tokens.weight[target_ids]
        target_weights = torch.where(
            torch.isfinite(target_weights),
            target_weights,
            embedding_weights,
        )
        target_weights = torch.nan_to_num(target_weights)
        target_codes = functional.linear(
            target_weights.float(),
            self._target_code_projection.float(),
        )
        target_codes = functional.normalize(target_codes, dim=-1)
        return compute_latent_cosine_loss(latents, target_codes, loss_mask)

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
        gamma: float = 4.0,
        max_anchors: int = 512,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        latent_loss_alpha: float | None = None,
        **kwargs,
    ):
        raw_hidden, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_raw_hidden_forward(
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
        hidden_blocks = raw_hidden.reshape(-1, self.block_size, raw_hidden.shape[-1])
        feedback = self.token_latent_head(hidden_blocks)
        # Apply the existing DFlash final norm only after feedback.  At zero
        # output-projection initialization this is bit-for-bit the base path.
        corrected_hidden = feedback.hidden_states.reshape_as(raw_hidden)
        hidden = self.norm(corrected_hidden)
        logits = self.lm_head(hidden)

        if isinstance(loss_config, str):
            loss_config = resolve_loss_config(
                loss_config,
                "fused" if logits.is_cuda else "eager",
            )
        resolved_loss_config = loss_config or (
            _DEFAULT_FUSED_LOSS_CONFIG if logits.is_cuda else _DEFAULT_EAGER_LOSS_CONFIG
        )
        final_loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=resolved_loss_config,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=False,
        )
        latent_loss, latent_cosine = self._compute_latent_loss(
            feedback.latents.reshape(1, -1, self.latent_dim),
            targets,
            aligned_loss_mask,
        )
        alpha = (
            float(latent_loss_alpha)
            if latent_loss_alpha is not None
            else self.config.resolved_latent_loss_alpha
        )
        loss = final_loss + alpha * latent_loss

        one = torch.ones((), device=loss.device)
        metrics["final_loss_sum"] = final_loss.detach().clone()
        metrics["final_loss_total"] = one.clone()
        metrics["latent_loss_sum"] = latent_loss.detach().clone()
        metrics["latent_loss_total"] = one.clone()
        metrics["latent_cosine_sum"] = latent_cosine.detach().clone()
        metrics["latent_cosine_total"] = one.clone()
        reliability = feedback.reliability.reshape(1, -1)
        metrics["reliability_sum"] = (
            (reliability * aligned_loss_mask.float()).sum().detach().clone()
        )
        metrics["reliability_total"] = aligned_loss_mask.float().sum().clamp_min(1.0)
        metrics["loss_sum"] = loss.detach().clone()
        return None, loss, metrics


@SpeculatorModel.register("parallel_token_latent")
class ParallelTokenLatentDraftModel(TokenLatentFeedbackDraftModel):
    """Alias retaining the name used in the v1.2 design document."""

    config_class = ParallelTokenLatentSpeculatorConfig  # type: ignore[assignment,misc]
    algorithm_name: ClassVar[str] = "parallel_token_latent"


@SpeculatorModel.register("latent_feedback")
class LatentFeedbackDraftModel(TokenLatentFeedbackDraftModel):
    """Short compatibility alias for ``token_latent_feedback``."""

    config_class = LatentFeedbackSpeculatorConfig  # type: ignore[assignment,misc]
    algorithm_name: ClassVar[str] = "latent_feedback"


@SpeculatorModel.register("parallel_token_latent_feedback")
class ParallelTokenLatentFeedbackDraftModel(TokenLatentFeedbackDraftModel):
    """Long-form compatibility alias for the v1.2 method."""

    config_class = ParallelTokenLatentFeedbackSpeculatorConfig  # type: ignore[assignment,misc]
    algorithm_name: ClassVar[str] = "parallel_token_latent_feedback"
