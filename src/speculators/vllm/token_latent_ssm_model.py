# SPDX-License-Identifier: Apache-2.0
"""vLLM model for no-base-logits token-latent SSM checkpoints."""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    maybe_prefix,
    process_eagle_weight,
)

from speculators.models.token_latent_ssm.config import (
    DEFAULT_HIDDEN_CANDIDATES,
    DEFAULT_LOGIT_SCALE_INIT,
    DEFAULT_SSM_STATE_DIM,
    DEFAULT_TOKEN_CODE_DIM,
    DEFAULT_TRANSITION_CANDIDATES,
)
from speculators.models.token_latent_ssm.model_definitions import (
    TokenLatentSSMHead,
)

_TRAINING_ONLY_WEIGHTS = (
    "lm_head",
    "t2d",
    "verifier_lm_head",
    "verifier_norm",
)


class Qwen3TokenLatentSSMModel(DFlashQwen3Model):
    """Plain DFlash feature extractor plus the token-latent decoder head."""

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
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.token_latent_head = TokenLatentSSMHead(
            target_vocab_size=int(config.vocab_size),
            draft_vocab_size=int(draft_vocab_size),
            hidden_size=int(config.hidden_size),
            code_dim=int(getattr(config, "token_code_dim", DEFAULT_TOKEN_CODE_DIM)),
            state_dim=int(getattr(config, "ssm_state_dim", DEFAULT_SSM_STATE_DIM)),
            block_size=int(getattr(config, "block_size", 8)),
            hidden_candidate_count=int(
                getattr(
                    config,
                    "hidden_candidate_count",
                    DEFAULT_HIDDEN_CANDIDATES,
                )
            ),
            transition_candidate_count=int(
                getattr(
                    config,
                    "transition_candidate_count",
                    DEFAULT_TRANSITION_CANDIDATES,
                )
            ),
            logit_scale_init=float(
                getattr(
                    config,
                    "token_latent_logit_scale_init",
                    DEFAULT_LOGIT_SCALE_INIT,
                )
            ),
            rms_norm_eps=float(config.rms_norm_eps),
            initializer_range=float(getattr(config, "initializer_range", None) or 0.02),
        ).to(dtype=vllm_config.model_config.dtype)
        self.token_latent_head.requires_grad_(False)
        self.confidence_head = None


class Qwen3TokenLatentSSMForCausalLM(Qwen3DSparkForCausalLM):
    """Sequential candidate decoder without a draft LM-head computation."""

    draft_id_to_target_id: nn.Parameter | None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Qwen3TokenLatentSSMModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def has_markov(self) -> bool:
        return False

    def has_hidden_correction(self) -> bool:
        return False

    def has_recurrent_logits_correction(self) -> bool:
        return False

    def has_candidate_selector(self) -> bool:
        return True

    def init_candidate_selector(
        self,
        hidden_per_step: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_target_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.model.token_latent_head.prepare_inference(
            hidden_per_step,
            anchor_hidden,
            anchor_target_ids,
            self.draft_id_to_target_id,
        )

    def select_candidate_token(
        self,
        step: int,
        hidden_states: torch.Tensor,
        previous_target_ids: torch.Tensor,
        selector_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.model.token_latent_head.select_inference_token(
            step=step,
            hidden_states=hidden_states,
            previous_target_ids=previous_target_ids,
            inference_state=selector_state,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights: dict[str, torch.Tensor] = {}
        includes_mapping = False
        includes_embeddings = False
        for weight_name, loaded_weight in weights:
            if any(part in weight_name for part in _TRAINING_ONLY_WEIGHTS):
                continue
            if "d2t" in weight_name:
                name = weight_name.replace("d2t", "draft_id_to_target_id")
                includes_mapping = True
            else:
                name = "model." + weight_name
            if "embed_tokens" in name:
                includes_embeddings = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = ["mask_embedding", "lm_head"]
        if not includes_embeddings:
            skip_substrs.append("embed_tokens")
        if not includes_mapping:
            skip_substrs.append("draft_id_to_target_id")
        loaded = AutoWeightsLoader(
            self,
            skip_substrs=skip_substrs,
        ).load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()  # noqa: SLF001
        return loaded


__all__ = [
    "Qwen3TokenLatentSSMForCausalLM",
    "Qwen3TokenLatentSSMModel",
]
