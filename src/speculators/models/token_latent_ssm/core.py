from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import LossConfig
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.token_latent_ssm.config import (
    DEFAULT_CONDITIONAL_LOSS_WEIGHT,
    DEFAULT_HIDDEN_CANDIDATES,
    DEFAULT_LOGIT_SCALE_INIT,
    DEFAULT_RETRIEVAL_LOSS_WEIGHT,
    DEFAULT_SSM_STATE_DIM,
    DEFAULT_TOKEN_CODE_DIM,
    DEFAULT_TRAINING_NEGATIVES,
    DEFAULT_TRANSITION_CANDIDATES,
    TokenLatentSSMSpeculatorConfig,
)
from speculators.models.token_latent_ssm.metrics import (
    compute_token_latent_metrics,
)
from speculators.models.token_latent_ssm.model_definitions import (
    TokenLatentSSMHead,
)
from speculators.models.utils import conditional_torch_compile

__all__ = ["TokenLatentSSMDraftModel"]


@SpeculatorModel.register("token_latent_ssm")
class TokenLatentSSMDraftModel(DFlashDraftModel):
    """DFlash hidden features decoded by token retrieval and a small causal SSM."""

    config_class: ClassVar[type[TokenLatentSSMSpeculatorConfig]] = (  # type: ignore[misc,assignment]
        TokenLatentSSMSpeculatorConfig
    )

    def __init__(self, config: TokenLatentSSMSpeculatorConfig) -> None:
        super().__init__(config=config)
        # This architecture never consumes draft base logits.  DFlash keeps a
        # reduced-vocabulary LM head in ordinary checkpoints because its serving
        # path needs it; token_latent_ssm must omit that tensor for every vocab
        # layout, including reduced-vocabulary training.
        keys_to_ignore_on_save = list(self._keys_to_ignore_on_save)
        keys_to_ignore_on_load_missing = list(self._keys_to_ignore_on_load_missing)
        if "lm_head.weight" not in keys_to_ignore_on_save:
            keys_to_ignore_on_save.append("lm_head.weight")
        if "lm_head.weight" not in keys_to_ignore_on_load_missing:
            keys_to_ignore_on_load_missing.append("lm_head.weight")
        self.__dict__["_keys_to_ignore_on_save"] = keys_to_ignore_on_save
        self.__dict__["_keys_to_ignore_on_load_missing"] = (
            keys_to_ignore_on_load_missing
        )

        transformer_config = config.transformer_layer_config
        self.token_latent_head = TokenLatentSSMHead(
            target_vocab_size=int(transformer_config.vocab_size),
            draft_vocab_size=int(config.draft_vocab_size),
            hidden_size=int(transformer_config.hidden_size),
            code_dim=config.token_code_dim,
            state_dim=config.ssm_state_dim,
            block_size=config.block_size,
            hidden_candidate_count=config.hidden_candidate_count,
            transition_candidate_count=config.transition_candidate_count,
            logit_scale_init=config.token_latent_logit_scale_init,
            rms_norm_eps=float(transformer_config.rms_norm_eps),
            initializer_range=float(transformer_config.initializer_range or 0.02),
        )

    def load_verifier_weights(self) -> None:
        super().load_verifier_weights()
        if bool(self.token_latent_head.codebook_initialized.item()):
            return

        # The design initializes token codes from verifier output geometry.  A
        # full-vocabulary LM head has exactly the required [target_vocab, hidden]
        # shape.  A reduced draft head does not cover every possible anchor token,
        # so fall back to the verifier's full embedding table in that case.
        source_weights = self.lm_head.weight
        expected_shape = (
            self.token_latent_head.target_vocab_size,
            self.token_latent_head.hidden_size,
        )
        if source_weights.shape != expected_shape:
            source_weights = self.embed_tokens.weight

        # Draft-vocab tensors are initialized uniformly to NaN before verifier
        # loading, so one scalar is enough to distinguish an unloaded table
        # without materializing a vocabulary-sized finite mask.
        source_is_loaded = bool(torch.isfinite(source_weights.reshape(-1)[0]).item())
        if source_is_loaded:
            self.token_latent_head.initialize_codebook(source_weights)
        else:
            # Config-only construction without a verifier path has no source
            # weights yet.  Preserve the normal globally-seeded random
            # initialization rather than replacing it with NaNs.
            self.token_latent_head.codebook_initialized.fill_(True)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "TokenLatentSSMDraftModel":
        base_config = cls._build_base_config_kwargs(
            "token_latent_ssm",
            verifier_config,
            **kwargs,
        )
        # v1.1 uses DFlash as a block feature extractor. Keep its mask
        # bidirectional; token-level causality is supplied by the SSM decoder.
        base_config["sliding_window_non_causal"] = True
        config = TokenLatentSSMSpeculatorConfig(
            **base_config,
            token_code_dim=kwargs.get("token_code_dim", DEFAULT_TOKEN_CODE_DIM),
            ssm_state_dim=kwargs.get("ssm_state_dim", DEFAULT_SSM_STATE_DIM),
            hidden_candidate_count=kwargs.get(
                "hidden_candidate_count",
                DEFAULT_HIDDEN_CANDIDATES,
            ),
            transition_candidate_count=kwargs.get(
                "transition_candidate_count",
                DEFAULT_TRANSITION_CANDIDATES,
            ),
            training_negative_count=kwargs.get(
                "training_negative_count",
                DEFAULT_TRAINING_NEGATIVES,
            ),
            retrieval_loss_weight=kwargs.get(
                "retrieval_loss_weight",
                DEFAULT_RETRIEVAL_LOSS_WEIGHT,
            ),
            conditional_loss_weight=kwargs.get(
                "conditional_loss_weight",
                DEFAULT_CONDITIONAL_LOSS_WEIGHT,
            ),
            token_latent_logit_scale_init=kwargs.get(
                "token_latent_logit_scale_init",
                DEFAULT_LOGIT_SCALE_INIT,
            ),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        shared = {
            "loss_config": kwargs.get("loss_config"),
            "gamma": kwargs.get("dflash_decay_gamma", 4.0),
            "max_anchors": kwargs.get("max_anchors", 512),
            "per_position_loss_weight": kwargs.get(
                "per_position_loss_weight",
                "dpace",
            ),
            "dpace_alpha": kwargs.get("dpace_alpha", 0.5),
            "training_negative_count": kwargs.get(
                "training_negative_count",
                DEFAULT_TRAINING_NEGATIVES,
            ),
            "retrieval_loss_weight": kwargs.get(
                "retrieval_loss_weight",
                DEFAULT_RETRIEVAL_LOSS_WEIGHT,
            ),
            "conditional_loss_weight": kwargs.get(
                "conditional_loss_weight",
                DEFAULT_CONDITIONAL_LOSS_WEIGHT,
            ),
        }
        return dict(shared), dict(shared)

    def _sample_training_candidates(
        self,
        target_draft_ids: torch.Tensor,
        negative_count: int,
    ) -> torch.Tensor:
        flat_targets = target_draft_ids.reshape(-1)
        batch_negative_count = negative_count // 2
        random_negative_count = negative_count - batch_negative_count
        pieces = [flat_targets.unsqueeze(-1)]
        if batch_negative_count:
            positions = torch.randint(
                flat_targets.numel(),
                (flat_targets.numel(), batch_negative_count),
                device=flat_targets.device,
            )
            pieces.append(flat_targets[positions])
        if random_negative_count:
            pieces.append(
                torch.randint(
                    self.draft_vocab_size,
                    (flat_targets.numel(), random_negative_count),
                    device=flat_targets.device,
                )
            )
        return torch.cat(pieces, dim=-1).view(
            *target_draft_ids.shape,
            negative_count + 1,
        )

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
        training_negative_count: int = DEFAULT_TRAINING_NEGATIVES,
        retrieval_loss_weight: float = DEFAULT_RETRIEVAL_LOSS_WEIGHT,
        conditional_loss_weight: float = DEFAULT_CONDITIONAL_LOSS_WEIGHT,
        **kwargs,
    ):
        del loss_config
        hidden, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_hidden_forward(
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
        hidden_blocks = hidden.view(-1, self.block_size, hidden.shape[-1])
        target_draft_ids = targets.argmax(dim=-1).view(-1, self.block_size)
        block_target_token_ids = input_ids[0, anchored_block_indices].view(
            -1,
            self.block_size,
        )
        candidate_ids = self._sample_training_candidates(
            target_draft_ids,
            training_negative_count,
        )
        output = self.token_latent_head(
            hidden_blocks,
            block_target_token_ids,
            candidate_ids,
            self.d2t,
        )
        loss, metrics = compute_token_latent_metrics(
            output,
            target_draft_ids,
            aligned_loss_mask,
            self.block_size,
            retrieval_loss_weight=retrieval_loss_weight,
            conditional_loss_weight=conditional_loss_weight,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            gamma=gamma,
        )
        return None, loss, metrics
