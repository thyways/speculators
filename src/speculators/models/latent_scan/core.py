from typing import ClassVar, cast

import torch
from torch import nn
from torch.nn import functional
from transformers import PretrainedConfig

from speculators.losses import LossConfig, ce_loss
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.latent_scan.config import LatentScanSpeculatorConfig
from speculators.models.latent_scan.model_definitions import (
    LatentRMSNorm,
    Qwen3LatentScanDecoderLayer,
)
from speculators.models.utils import conditional_torch_compile

__all__ = [
    "LatentScanDraftModel",
]

_DEFAULT_LOSS_CONFIG: LossConfig = {"ce": (ce_loss, 1.0)}


@SpeculatorModel.register("latent_scan")
class LatentScanDraftModel(DFlashDraftModel):
    """All-layer DFlash token-latent scan with one final vocabulary projection."""

    config_class: ClassVar[type[LatentScanSpeculatorConfig]] = (  # type: ignore[misc,assignment]
        LatentScanSpeculatorConfig
    )
    _no_split_modules: ClassVar[list[str]] = ["Qwen3LatentScanDecoderLayer"]  # type: ignore[misc]

    def __init__(self, config: LatentScanSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = int(config.transformer_layer_config.hidden_size)
        eps = float(config.transformer_layer_config.rms_norm_eps)
        initializer_range = float(
            config.transformer_layer_config.initializer_range or 0.02
        )
        self.token_latent_norm = LatentRMSNorm(hidden_size, eps)
        self.token_latent_projection = nn.Linear(
            hidden_size,
            config.latent_dim,
            bias=False,
        )
        nn.init.normal_(
            self.token_latent_projection.weight,
            mean=0.0,
            std=initializer_range,
        )

        # Initialize the auxiliary target with PyTorch's global RNG, just like
        # the other randomly initialized draft weights. ``train.py`` seeds this
        # RNG from the common ``--seed`` argument before constructing the model.
        # Keep the buffer persistent so checkpoint/resume uses the exact same
        # target instead of drawing a fresh one after every load.
        self.register_buffer(
            "_token_target_projection",
            torch.empty(
                config.latent_dim,
                hidden_size,
                device=self.token_latent_projection.weight.device,
                dtype=torch.float32,
            ),
        )
        target_projection = torch.randn(
            config.latent_dim,
            hidden_size,
            dtype=torch.float32,
        )
        target_projection = functional.normalize(target_projection, dim=-1)
        self._token_target_projection = target_projection.to(
            device=self.token_latent_projection.weight.device
        )

    def _build_layer(self, layer_idx: int) -> nn.Module:
        return Qwen3LatentScanDecoderLayer(
            self.config.transformer_layer_config,  # type: ignore[arg-type]
            layer_idx,
            latent_dim=self.config.latent_dim,
            block_size=self.config.block_size,
            layer_scale_init=self.config.latent_layer_scale_init,
        )

    def _force_causal_local_attention(self) -> bool:
        return self.config.strict_causal_slots

    def _reshape_blocks(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        return hidden_states.reshape(-1, self.block_size, hidden_size)

    def _predict_token_latents(self, hidden_states: torch.Tensor) -> torch.Tensor:
        blocks = self._reshape_blocks(hidden_states)
        return self.token_latent_projection(self.token_latent_norm(blocks))

    def _init_draft_layer_state(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self._predict_token_latents(hidden_states)

    def _after_draft_layer(
        self,
        hidden_states: torch.Tensor,
        layer_state: torch.Tensor | None,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if layer_state is None:
            raise RuntimeError("LatentScan layer state was not initialized.")
        layer = cast("Qwen3LatentScanDecoderLayer", self.layers[layer_idx])
        hidden_blocks = self._reshape_blocks(hidden_states)
        current_latents = self._predict_token_latents(hidden_states)
        hidden_blocks, next_latents = layer.latent_scan(
            hidden_blocks,
            layer_state,
            current_latents,
        )
        return hidden_blocks.reshape_as(hidden_states), next_latents

    def _compute_latent_loss(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        anchored_block_indices: torch.Tensor,
        aligned_loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predicted = functional.normalize(
            self._predict_token_latents(hidden).float(),
            dim=-1,
        )
        block_token_ids = input_ids[:, anchored_block_indices].reshape(
            -1,
            self.block_size,
        )
        with torch.no_grad():
            token_embeddings = self.embed_tokens(block_token_ids).float()
            targets = functional.linear(
                token_embeddings,
                self._token_target_projection.float(),
            )
            targets = functional.normalize(targets, dim=-1)

        latent_mask = aligned_loss_mask.reshape(-1, self.block_size).float().clone()
        block_valid = (latent_mask.sum(dim=-1) > 0).to(latent_mask.dtype)
        latent_mask[:, 0] = block_valid
        cosine = (predicted * targets).sum(dim=-1)
        valid_count = latent_mask.sum()
        mean_cosine = (cosine * latent_mask).sum() / valid_count.clamp(min=1.0)
        has_valid_target = (valid_count > 0).to(mean_cosine.dtype)
        latent_loss = (1.0 - mean_cosine) * has_valid_target
        return latent_loss, mean_cosine

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "LatentScanDraftModel":
        config = LatentScanSpeculatorConfig(
            **cls._build_base_config_kwargs(
                "latent_scan",
                verifier_config,
                **kwargs,
            ),
            latent_dim=kwargs.get("latent_dim", 256),
            latent_layer_scale_init=kwargs.get("latent_layer_scale_init", 1e-3),
            strict_causal_slots=kwargs.get("strict_causal_slots", True),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        train_kwargs, validation_kwargs = DFlashDraftModel.get_trainer_kwargs(**kwargs)
        latent_loss_alpha = kwargs.get("latent_loss_alpha", 0.1)
        train_kwargs["latent_loss_alpha"] = latent_loss_alpha
        validation_kwargs["latent_loss_alpha"] = latent_loss_alpha
        return train_kwargs, validation_kwargs

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
        per_position_loss_weight: str = "dpace",
        dpace_alpha: float = 0.5,
        latent_loss_alpha: float = 0.1,
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
        final_loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=False,
        )
        latent_loss, latent_cosine = self._compute_latent_loss(
            hidden,
            input_ids,
            anchored_block_indices,
            aligned_loss_mask,
        )
        loss = final_loss + latent_loss_alpha * latent_loss

        ones = torch.ones((), device=loss.device)
        metrics["final_loss_sum"] = final_loss.detach().clone()
        metrics["final_loss_total"] = ones
        metrics["latent_loss_sum"] = latent_loss.detach().clone()
        metrics["latent_loss_total"] = ones.clone()
        metrics["latent_cosine_sum"] = latent_cosine.detach().clone()
        metrics["latent_cosine_total"] = ones.clone()
        metrics["loss_sum"] = loss.detach().clone()
        return None, loss, metrics
