from __future__ import annotations

from typing import cast

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.kv_native_dspark.config import (
    KVNativeDSparkSpeculatorConfig,
)
from speculators.models.kv_native_dspark.metrics import add_kv_native_losses
from speculators.models.kv_native_dspark.model_definitions import (
    KVLayerArtifacts,
    Qwen3KVNativeDecoderLayer,
    VerifierRotaryEmbedding,
)
from speculators.models.metrics import LossConfig, kl_div_loss, resolve_loss_config
from speculators.models.utils import conditional_torch_compile, get_verifier_config

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}
_TEXT_POSITION_IDS_NDIM = 2

__all__ = ["KVNativeDSparkDraftModel"]


@SpeculatorModel.register("kv_native_dspark")
class KVNativeDSparkDraftModel(DSparkDraftModel):
    """DSpark trained from scratch to read real verifier K/V prefix memory."""

    config_class = (  # type: ignore[misc,assignment]
        KVNativeDSparkSpeculatorConfig
    )
    _no_split_modules = ["Qwen3KVNativeDecoderLayer"]

    def __init__(self, config: KVNativeDSparkSpeculatorConfig) -> None:
        super().__init__(config=config)
        self.layers = torch.nn.ModuleList(
            [
                Qwen3KVNativeDecoderLayer(
                    config.transformer_layer_config,  # type: ignore[arg-type]
                    layer_idx,
                    verifier_num_key_value_heads=config.verifier_num_key_value_heads,
                    verifier_head_dim=config.verifier_head_dim,
                )
                for layer_idx in range(len(self.layers))
            ]
        )
        for layer in self.kv_native_layers:
            layer.apply(self._initialize_weights)
            layer.kv_adapter.reset_parameters()

        self.verifier_rotary_emb = VerifierRotaryEmbedding(
            head_dim=config.verifier_head_dim,
            partial_rotary_factor=config.verifier_partial_rotary_factor,
            rope_theta=config.verifier_rope_theta,
            mrope_section=config.verifier_mrope_section,
        )
        self._verifier_kv_mapping_indices = self._resolve_mapping_indices(
            config.verifier_kv_layer_ids,
            config.verifier_kv_layer_mapping,
            len(self.layers),
        )

    @property
    def kv_native_layers(self) -> list[Qwen3KVNativeDecoderLayer]:
        return cast("list[Qwen3KVNativeDecoderLayer]", list(self.layers))

    @staticmethod
    def _resolve_mapping_indices(
        exported_layer_ids: list[int],
        mapping: list[int],
        num_draft_layers: int,
    ) -> tuple[int, ...]:
        if len(mapping) != num_draft_layers:
            raise ValueError(
                "verifier_kv_layer_mapping must contain one layer ID per draft "
                f"layer: got {len(mapping)} for {num_draft_layers} layers"
            )
        lookup = {
            layer_id: index for index, layer_id in enumerate(exported_layer_ids)
        }
        unknown = sorted(set(mapping) - set(lookup))
        if unknown:
            raise ValueError(
                f"verifier_kv_layer_mapping references non-exported layers: {unknown}"
            )
        return tuple(lookup[layer_id] for layer_id in mapping)

    @property
    def verifier_kv_shape(self) -> tuple[int, int, int]:
        return (
            len(self.config.verifier_kv_layer_ids),
            self.config.verifier_num_key_value_heads,
            self.config.verifier_head_dim,
        )

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> KVNativeDSparkDraftModel:
        cls._validate_verifier_kv_contract(kwargs)
        enable_confidence_head = kwargs.get("enable_confidence_head")
        confidence_head_with_markov = kwargs.get("confidence_head_with_markov")
        base_config = cls._build_base_config_kwargs(
            "kv_native_dspark", verifier_config, **kwargs
        )
        base_config["aux_hidden_state_layer_ids"] = []
        config = KVNativeDSparkSpeculatorConfig(
            **base_config,
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=(
                True if enable_confidence_head is None else enable_confidence_head
            ),
            confidence_head_with_markov=(
                True
                if confidence_head_with_markov is None
                else confidence_head_with_markov
            ),
            verifier_kv_layer_ids=kwargs["verifier_kv_layer_ids"],
            verifier_kv_layer_mapping=kwargs["verifier_kv_layer_mapping"],
            verifier_num_key_value_heads=kwargs["verifier_num_key_value_heads"],
            verifier_head_dim=kwargs["verifier_head_dim"],
            verifier_partial_rotary_factor=kwargs["verifier_partial_rotary_factor"],
            verifier_rope_theta=kwargs["verifier_rope_theta"],
            verifier_mrope_section=kwargs["verifier_mrope_section"],
            num_speculative_tokens=kwargs["num_speculative_tokens"],
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def _validate_verifier_kv_contract(kwargs: dict) -> None:
        target_config = get_verifier_config(kwargs["verifier_name_or_path"])
        layer_types = getattr(target_config, "layer_types", None)
        if layer_types is None:
            raise ValueError(
                "Verifier config has no layer_types; full-attention K/V layers "
                "cannot be validated"
            )
        available = {
            index
            for index, layer_type in enumerate(layer_types)
            if layer_type == "full_attention"
        }
        invalid = sorted(set(kwargs["verifier_kv_layer_ids"]) - available)
        if invalid:
            raise ValueError(
                f"Selected verifier layers are not full-attention: {invalid}; "
                f"available={sorted(available)}"
            )

        expected_heads = int(target_config.num_key_value_heads)
        expected_head_dim = int(
            getattr(target_config, "head_dim", None)
            or target_config.hidden_size // target_config.num_attention_heads
        )
        if kwargs["verifier_num_key_value_heads"] != expected_heads:
            raise ValueError(
                "--verifier-num-key-value-heads does not match the verifier: "
                f"{kwargs['verifier_num_key_value_heads']} != {expected_heads}"
            )
        if kwargs["verifier_head_dim"] != expected_head_dim:
            raise ValueError(
                "--verifier-head-dim does not match the verifier: "
                f"{kwargs['verifier_head_dim']} != {expected_head_dim}"
            )

        rope = getattr(target_config, "rope_parameters", None) or getattr(
            target_config, "rope_scaling", None
        )
        if not isinstance(rope, dict):
            raise ValueError("Verifier config does not expose RoPE parameters")
        if not rope.get("mrope_interleaved", False):
            raise ValueError(
                "VerifierRotaryEmbedding implements the interleaved MRoPE frequency "
                "layout only, but the verifier does not set mrope_interleaved=true. "
                "Chunked [TTT..HHH..WWW] MRoPE would silently produce wrong Keys."
            )
        expected_partial = float(rope.get("partial_rotary_factor", 1.0))
        expected_theta = float(rope.get("rope_theta", 10_000.0))
        expected_section = list(rope.get("mrope_section", []))
        if kwargs["verifier_partial_rotary_factor"] != expected_partial:
            raise ValueError(
                "--verifier-partial-rotary-factor does not match the verifier"
            )
        if kwargs["verifier_rope_theta"] != expected_theta:
            raise ValueError("--verifier-rope-theta does not match the verifier")
        if kwargs["verifier_mrope_section"] != expected_section:
            raise ValueError(
                "--verifier-mrope-section does not match the verifier: "
                f"{kwargs['verifier_mrope_section']} != {expected_section}"
            )

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        shared = {
            "loss_config": resolve_loss_config(kwargs["loss_fn"]),
            "gamma": kwargs.get("dflash_decay_gamma", 4.0),
            "max_anchors": kwargs.get("max_anchors", 3072),
            "confidence_head_alpha": kwargs.get("confidence_head_alpha", 1.0),
            "per_position_loss_weight": kwargs.get(
                "per_position_loss_weight", "fixed-exp-decay"
            ),
            "dpace_alpha": kwargs.get("dpace_alpha", 0.5),
            "local_kv_loss_alpha": kwargs.get("local_kv_loss_alpha", 0.1),
            "query_key_loss_alpha": kwargs.get("query_key_loss_alpha", 1.0),
            "attention_value_loss_alpha": kwargs.get("attention_value_loss_alpha", 1.0),
        }
        return dict(shared), dict(shared)

    @staticmethod
    def _teacher_local_kv(
        tensor: torch.Tensor,
        anchored_indices: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        return tensor[:, anchored_indices, layer_index]

    def _teacher_targets(
        self,
        verifier_last_hidden_states: torch.Tensor,
        anchored_indices: torch.Tensor,
    ) -> torch.Tensor:
        total_seq_len = verifier_last_hidden_states.shape[1]
        with torch.no_grad():
            if anchored_indices.numel() < total_seq_len:
                target_indices = (
                    anchored_indices
                    if self.config.sample_from_anchor
                    else (anchored_indices - 1) % total_seq_len
                )
                target_hidden = verifier_last_hidden_states[:, target_indices]
                return self.verifier_lm_head(self.verifier_norm(target_hidden))

            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            if not self.config.sample_from_anchor:
                verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            return verifier_logits[:, anchored_indices]

    @conditional_torch_compile
    def forward(  # noqa: C901, PLR0917
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        verifier_keys: torch.Tensor,
        verifier_values: torch.Tensor,
        verifier_kv_layer_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        local_kv_loss_alpha: float = 0.1,
        query_key_loss_alpha: float = 1.0,
        attention_value_loss_alpha: float = 1.0,
        **kwargs,
    ):
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError(
                "Auxiliary hidden-state token axes must match input_ids, got "
                f"{tuple(hidden_states.shape[:2])} and {tuple(input_ids.shape)}"
            )
        if input_ids.shape[0] != 1:
            raise ValueError("KV-native DSpark expects packed batch size 1")
        if verifier_last_hidden_states.shape[:2] != input_ids.shape:
            raise ValueError(
                "Verifier final hidden-state token axes must match input_ids"
            )
        if verifier_keys.shape != verifier_values.shape:
            raise ValueError(
                "verifier key/value shapes differ: "
                f"{tuple(verifier_keys.shape)} vs {tuple(verifier_values.shape)}"
            )
        expected_shape = (*input_ids.shape, *self.verifier_kv_shape)
        if tuple(verifier_keys.shape) != expected_shape:
            raise ValueError(
                "Expected verifier K/V [batch,tokens,layers,heads,dim]="
                f"{expected_shape}, got {tuple(verifier_keys.shape)}"
            )
        # Payload validation already checked these IDs against the model config.
        del verifier_kv_layer_ids

        device = input_ids.device
        total_seq_len = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(total_seq_len, device=device).unsqueeze(0)
        if position_ids.ndim != _TEXT_POSITION_IDS_NDIM:
            raise ValueError(
                "Current text-only KV-native training expects position_ids [B,T], "
                f"got {tuple(position_ids.shape)}"
            )
        if position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids must match input_ids shape, got "
                f"{tuple(position_ids.shape)} and {tuple(input_ids.shape)}"
            )

        full_mask, sliding_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, max_anchors, document_ids, device)
        )
        mask_tokens_size = max_anchors * self.block_size
        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        combined_position_ids = torch.cat(
            (position_ids, mask_position_ids.unsqueeze(0)), dim=1
        )
        verifier_position_embeddings = self.verifier_rotary_emb(
            verifier_last_hidden_states, combined_position_ids
        )
        anchored_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )
        targets = self._teacher_targets(
            verifier_last_hidden_states,
            anchored_indices,
        )

        artifacts: list[KVLayerArtifacts] = []
        teacher_keys: list[torch.Tensor] = []
        teacher_values: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.kv_native_layers):
            source_index = self._verifier_kv_mapping_indices[layer_idx]
            attention_mask = (
                sliding_mask if layer_idx in self.sliding_window_indices else full_mask
            )
            noise_embedding, layer_artifacts = layer(
                hidden_states=noise_embedding,
                verifier_keys=verifier_keys[:, :, source_index].transpose(1, 2),
                verifier_values=verifier_values[:, :, source_index].transpose(1, 2),
                attention_mask=attention_mask,
                verifier_position_embeddings=verifier_position_embeddings,
                **kwargs,
            )
            artifacts.append(layer_artifacts)
            teacher_keys.append(
                self._teacher_local_kv(
                    verifier_keys, anchored_indices, source_index
                )
            )
            teacher_values.append(
                self._teacher_local_kv(
                    verifier_values, anchored_indices, source_index
                )
            )

        hidden = self.norm(noise_embedding)
        logits = self.lm_head(hidden)
        aligned_loss_mask = loss_mask[:, anchored_indices]
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )
        if not self.config.sample_from_anchor:
            aligned_loss_mask[:, :: self.block_size] = 0

        num_blocks = max_anchors
        block_tokens = input_ids[0, anchored_indices].view(num_blocks, self.block_size)
        if self.config.sample_from_anchor:
            prev_token_ids = block_tokens
        else:
            prev_token_ids = torch.cat(
                (block_tokens[:, :1], block_tokens[:, :-1]),
                dim=1,
            )
        hidden_blocks = hidden.view(num_blocks, self.block_size, -1)
        confidence_logits = None
        prev_emb = None
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits.view(num_blocks, self.block_size, -1) + bias).view(
                1, mask_tokens_size, -1
            )
        if self.confidence_head is not None:
            features = (
                torch.cat((hidden_blocks, prev_emb.to(hidden_blocks.dtype)), dim=-1)
                if self.config.confidence_head_with_markov and prev_emb is not None
                else hidden_blocks
            )
            confidence_logits = self.confidence_head(features).reshape(
                1, mask_tokens_size
            )

        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            self.block_size,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
        )
        loss, metrics = add_kv_native_losses(
            loss,
            metrics,
            predicted_keys=torch.stack(
                [item.local_keys for item in artifacts], dim=2
            ),
            predicted_values=torch.stack(
                [item.local_values for item in artifacts], dim=2
            ),
            teacher_keys=torch.stack(teacher_keys, dim=2),
            teacher_values=torch.stack(teacher_values, dim=2),
            queries=torch.stack([item.queries for item in artifacts], dim=2),
            loss_mask=aligned_loss_mask,
            block_size=self.block_size,
            local_kv_loss_alpha=local_kv_loss_alpha,
            query_key_loss_alpha=query_key_loss_alpha,
            attention_value_loss_alpha=attention_value_loss_alpha,
        )
        return None, loss, metrics
