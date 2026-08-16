from __future__ import annotations

import logging
from copy import deepcopy
from typing import ClassVar, cast

import torch
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from speculators.losses import LossConfig
from speculators.model import SpeculatorModel
from speculators.models.dflash.attention import create_dual_stream_anchor_mask_mod
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.kv_native_dflash.config import (
    KVNativeDFlashSpeculatorConfig,
)
from speculators.models.kv_native_dflash.model_definitions import (
    DualStreamAttentionMasks,
    Qwen3KVNativeDecoderLayer,
    VerifierRotaryEmbedding,
)
from speculators.models.utils import conditional_torch_compile, get_verifier_config

_TEXT_POSITION_IDS_NDIM = 2
_WEIGHT_NDIM = 2

logger = logging.getLogger(__name__)

__all__ = ["KVNativeDFlashDraftModel"]


@SpeculatorModel.register("kv_native_dflash")
class KVNativeDFlashDraftModel(DFlashDraftModel):
    """Dual-stream DFlash that reads untouched verifier K/V prefix memory."""

    config_class: ClassVar[type[KVNativeDFlashSpeculatorConfig]] = (  # type: ignore[misc]
        KVNativeDFlashSpeculatorConfig
    )
    _no_split_modules = ["Qwen3KVNativeDecoderLayer"]

    def __init__(self, config: KVNativeDFlashSpeculatorConfig) -> None:
        super().__init__(config=config)
        num_draft_layers = config.transformer_layer_config.num_hidden_layers
        mapping_indices = self._resolve_mapping_indices(
            config.verifier_kv_layer_ids,
            config.verifier_kv_layer_mapping,
            num_draft_layers,
        )
        self.layers = torch.nn.ModuleList(
            [
                Qwen3KVNativeDecoderLayer(
                    config.transformer_layer_config,  # type: ignore[arg-type]
                    layer_idx,
                    verifier_num_key_value_heads=config.verifier_num_key_value_heads,
                    verifier_head_dim=config.verifier_head_dim,
                )
                for layer_idx in range(num_draft_layers)
            ]
        )
        for layer in self.kv_native_layers:
            layer.apply(self._initialize_weights)
            layer.reset_dual_stream_parameters()

        hidden_size = config.transformer_layer_config.hidden_size
        self.horizon_embedding = torch.nn.Embedding(config.block_size, hidden_size)
        self._initialize_weights(self.horizon_embedding)
        self.verifier_rotary_emb = VerifierRotaryEmbedding(
            head_dim=config.verifier_head_dim,
            partial_rotary_factor=config.verifier_partial_rotary_factor,
            rope_theta=config.verifier_rope_theta,
        )
        self._verifier_kv_mapping_indices = mapping_indices
        if config.anchor_hidden_injection:
            self.anchor_hidden_norm = Qwen3RMSNorm(
                hidden_size,
                eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
            )
            self.anchor_hidden_proj = torch.nn.Linear(
                hidden_size, hidden_size, bias=False
            )
            self.reset_anchor_hidden_injection()

    def reset_anchor_hidden_injection(self) -> None:
        """Zero the anchor-state projection so the injection starts as a no-op.

        The verifier's residual stream is far larger in magnitude than a token
        embedding, so the state is normalized first and then passed through a
        zero-initialized projection: step 0 is bit-identical to a run without the
        injection, while the projection still receives gradient from step 1 and
        can grow the term at whatever scale training wants.
        """
        torch.nn.init.ones_(self.anchor_hidden_norm.weight)
        torch.nn.init.zeros_(self.anchor_hidden_proj.weight)

    @property
    def kv_native_layers(self) -> list[Qwen3KVNativeDecoderLayer]:
        return cast("list[Qwen3KVNativeDecoderLayer]", list(self.layers))

    @staticmethod
    def _verifier_query_projection(
        weight: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor | None:
        """Return the query half of a possibly gated verifier ``q_proj`` weight.

        Gated-attention verifiers (Qwen3.5's ``attn_output_gate``) size ``q_proj``
        at ``num_heads * head_dim * 2`` and split it *inside each head's slice*
        as ``[query | gate]``, so the query rows are
        ``weight.reshape(num_heads, 2 * head_dim, -1)[:, :head_dim]`` rather than
        the leading half of the matrix. Returns ``None`` when the shape matches
        neither the plain nor the gated layout.
        """
        if weight.ndim != _WEIGHT_NDIM:
            return None
        expected_rows = num_heads * head_dim
        hidden_size = weight.shape[1]
        if weight.shape[0] == expected_rows:
            return weight
        if weight.shape[0] == 2 * expected_rows:
            return weight.reshape(num_heads, 2 * head_dim, hidden_size)[
                :, :head_dim
            ].reshape(expected_rows, hidden_size)
        return None

    def warm_start_context_queries(self, verifier_name_or_path: str) -> None:
        """Initialize each draft layer's query path from its mapped verifier layer.

        The context stream matches ``q_norm(q_proj(x))`` against the *raw* verifier
        keys of ``verifier_kv_layer_mapping[i]``, and both the KV adapter and the
        context gate start as no-ops, so a randomly initialized query projection
        makes the cross-attention pure noise at step 0 -- the query-to-key
        alignment has to be discovered from scratch. Copying the verifier's own
        query path for that layer starts the queries in the right space instead.

        The draft's residual stream is not the verifier's layer-``mapping[i]``
        residual stream, so this is a warm start rather than an exact transplant:
        it calibrates the projection's output space and scale, not its input.

        Initialization only -- nothing here runs at inference. A verifier whose
        tensors cannot be resolved or shape-matched downgrades to a warning and
        keeps the random query path, so other verifiers still train.
        """
        from speculators.utils.loading import load_model_layers  # noqa: PLC0415

        mapping = list(self.config.verifier_kv_layer_mapping)
        requests = {
            layer_index: (
                f".layers.{verifier_layer}.self_attn.q_proj.weight",
                f".layers.{verifier_layer}.self_attn.q_norm.weight",
            )
            for layer_index, verifier_layer in enumerate(mapping)
        }
        names = sorted({name for pair in requests.values() for name in pair})
        try:
            weights = load_model_layers(names, verifier_name_or_path)
        except (ValueError, OSError) as error:
            logger.warning(
                "Skipping context-query warm start: could not read verifier "
                "attention weights from %s (%s)",
                verifier_name_or_path,
                error,
            )
            return

        warmed = []
        for layer_index, (query_name, norm_name) in requests.items():
            attn = self.kv_native_layers[layer_index].self_attn
            source = weights.get(query_name)
            query = (
                None
                if source is None
                else self._verifier_query_projection(
                    source,
                    attn.config.num_attention_heads,
                    attn.head_dim,
                )
            )
            if query is None or query.shape != attn.q_proj.weight.shape:
                logger.warning(
                    "Skipping context-query warm start for draft layer %d: "
                    "verifier tensor %s is missing or not shape-compatible.",
                    layer_index,
                    query_name,
                )
                continue
            with torch.no_grad():
                attn.q_proj.weight.copy_(query.to(attn.q_proj.weight))
                norm_weight = weights.get(norm_name)
                if (
                    norm_weight is not None
                    and norm_weight.shape == attn.q_norm.weight.shape
                ):
                    attn.q_norm.weight.copy_(norm_weight.to(attn.q_norm.weight))
            warmed.append(f"{layer_index}<-{mapping[layer_index]}")
        if warmed:
            logger.info(
                "Warm-started draft query paths from verifier layers: %s",
                ", ".join(warmed),
            )

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
        lookup = {layer_id: index for index, layer_id in enumerate(exported_layer_ids)}
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
    ) -> KVNativeDFlashDraftModel:
        partial_rotary_factor, rope_theta = cls._resolve_verifier_kv_contract(kwargs)
        verifier_config = cls._with_text_partial_rope(
            verifier_config,
            partial_rotary_factor=partial_rotary_factor,
            rope_theta=rope_theta,
        )
        base_config = cls._build_base_config_kwargs(
            "kv_native_dflash", verifier_config, **kwargs
        )
        base_config["aux_hidden_state_layer_ids"] = []
        num_speculative_tokens = kwargs.get("num_speculative_tokens")
        if num_speculative_tokens is None:
            num_speculative_tokens = base_config["block_size"] - 1
        config = KVNativeDFlashSpeculatorConfig(
            **base_config,
            verifier_kv_layer_ids=kwargs["verifier_kv_layer_ids"],
            verifier_kv_layer_mapping=kwargs["verifier_kv_layer_mapping"],
            verifier_num_key_value_heads=kwargs["verifier_num_key_value_heads"],
            verifier_head_dim=kwargs["verifier_head_dim"],
            verifier_partial_rotary_factor=partial_rotary_factor,
            verifier_rope_theta=rope_theta,
            anchor_hidden_injection=kwargs.get("anchor_hidden_injection", False),
            num_speculative_tokens=num_speculative_tokens,
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        if kwargs.get("warm_start_context_queries", True):
            model.warm_start_context_queries(kwargs["verifier_name_or_path"])
        return model

    @staticmethod
    def _with_text_partial_rope(
        draft_config: PretrainedConfig,
        *,
        partial_rotary_factor: float,
        rope_theta: float,
    ) -> PretrainedConfig:
        draft_config = deepcopy(draft_config)
        text_rope = {
            "rope_type": "default",
            "rope_theta": rope_theta,
            "partial_rotary_factor": partial_rotary_factor,
        }
        if hasattr(draft_config, "rope_parameters"):
            draft_config.rope_parameters = text_rope
        else:
            draft_config.rope_scaling = None
            draft_config.rope_theta = rope_theta
            draft_config.partial_rotary_factor = partial_rotary_factor
        return draft_config

    @staticmethod
    def _resolve_verifier_kv_contract(kwargs: dict) -> tuple[float, float]:
        target_config = get_verifier_config(kwargs["verifier_name_or_path"])
        target_config = getattr(target_config, "text_config", target_config)
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
        rope_type = rope.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(
                "KV-native DFlash text RoPE supports only rope_type='default', "
                f"got {rope_type!r}"
            )
        partial_rotary_factor = float(
            rope.get(
                "partial_rotary_factor",
                getattr(target_config, "partial_rotary_factor", 1.0),
            )
        )
        rope_theta = float(
            rope.get(
                "rope_theta",
                getattr(target_config, "rope_theta", 10_000.0),
            )
        )
        rotary_dim = int(expected_head_dim * partial_rotary_factor)
        if (
            partial_rotary_factor <= 0.0
            or partial_rotary_factor > 1.0
            or rotary_dim <= 0
            or rotary_dim % 2
        ):
            raise ValueError(
                "Verifier config must define a partial rotary factor in (0, 1] "
                "whose resulting rotary dimension is positive and even"
            )
        if rope_theta <= 0.0:
            raise ValueError("Verifier config rope_theta must be positive")
        return partial_rotary_factor, rope_theta

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        return DFlashDraftModel.get_trainer_kwargs(**kwargs)

    @torch.compiler.disable
    def _create_dual_stream_attention_masks(
        self,
        document_ids: torch.Tensor,
        total_seq_len: int,
        anchor_positions: torch.Tensor,
        device: torch.device,
        sliding_window: int | None = None,
        sliding_window_non_causal: bool = False,
    ) -> DualStreamAttentionMasks:
        prefix_mod, local_mod, query_length, prefix_length = (
            create_dual_stream_anchor_mask_mod(
                document_ids=document_ids.squeeze(0).to(device),
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                block_size=self.block_size,
                sliding_window=sliding_window,
                sliding_window_non_causal=sliding_window_non_causal,
            )
        )
        prefix = self._create_mask_fn(
            prefix_mod,
            B=None,
            H=None,
            Q_LEN=query_length,
            KV_LEN=prefix_length,
            device=device,
        )
        local = self._create_mask_fn(
            local_mod,
            B=None,
            H=None,
            Q_LEN=query_length,
            KV_LEN=query_length,
            device=device,
        )
        return DualStreamAttentionMasks(prefix=prefix, local=local)

    @torch.compiler.disable
    def _build_dual_stream_attention_masks(
        self,
        loss_mask: torch.Tensor,
        max_anchors: int,
        document_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[
        DualStreamAttentionMasks | None,
        DualStreamAttentionMasks | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        total_seq_len = loss_mask.shape[1]
        eligible_anchors = loss_mask.clone()
        eligible_anchors[:, 0] = 0
        has_same_document_prefix = (document_ids[:, 1:] == document_ids[:, :-1]) & (
            document_ids[:, 1:] != -1
        )
        eligible_anchors[:, 1:] = eligible_anchors[:, 1:] * (
            has_same_document_prefix.to(eligible_anchors.dtype)
        )
        anchor_positions, anchor_valid = select_anchors(
            eligible_anchors, max_anchors, self.block_size
        )
        full_masks = (
            self._create_dual_stream_attention_masks(
                document_ids,
                total_seq_len,
                anchor_positions,
                device,
            )
            if self.uses_full_attn
            else None
        )
        sliding_masks = (
            self._create_dual_stream_attention_masks(
                document_ids,
                total_seq_len,
                anchor_positions,
                device,
                sliding_window=self.sliding_window,
                sliding_window_non_causal=self.sliding_window_non_causal,
            )
            if self.uses_sliding_window_attn
            else None
        )
        return full_masks, sliding_masks, anchor_positions, anchor_valid

    def _teacher_targets(
        self,
        verifier_last_hidden_states: torch.Tensor,
        anchored_indices: torch.Tensor,
    ) -> torch.Tensor:
        total_seq_len = verifier_last_hidden_states.shape[1]
        with torch.no_grad():
            if anchored_indices.numel() < total_seq_len:
                target_indices = (anchored_indices - 1) % total_seq_len
                target_hidden = verifier_last_hidden_states[:, target_indices]
                return self.verifier_lm_head(self.verifier_norm(target_hidden))

            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            return verifier_logits[:, anchored_indices]

    def _inject_anchor_hidden(
        self,
        block_embeddings: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Add the verifier state at the last verified position into slot 0.

        The anchor itself is the verifier's bonus token: it is sampled from the
        logits of the preceding position and never runs through the verifier, so
        neither its K/V nor its hidden state exists at serving time. ``anchor - 1``
        is the deepest position that did run, which makes this the same
        conditioning EAGLE uses (state at the last verified position plus the
        embedding of the token sampled from it).

        Anchors are selected to have a same-document predecessor, so ``anchor - 1``
        is in range and in-document for every *valid* anchor; padded anchors are
        clamped and their slots are masked out of the loss anyway.
        """
        source_positions = (anchor_positions - 1).clamp(min=0)
        anchor_hidden = verifier_last_hidden_states[:, source_positions]
        injected = self.anchor_hidden_proj(self.anchor_hidden_norm(anchor_hidden))
        hidden_size = block_embeddings.shape[-1]
        blocks = block_embeddings.view(1, -1, self.block_size, hidden_size)
        blocks = torch.cat(
            (blocks[:, :, :1] + injected.unsqueeze(2), blocks[:, :, 1:]),
            dim=2,
        )
        return blocks.reshape(1, -1, hidden_size)

    @conditional_torch_compile
    def forward(  # noqa: C901
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
        max_anchors: int = 512,
        per_position_loss_weight: str = "dpace",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError(
                "Auxiliary hidden-state token axes must match input_ids, got "
                f"{tuple(hidden_states.shape[:2])} and {tuple(input_ids.shape)}"
            )
        if input_ids.shape[0] != 1:
            raise ValueError("KV-native DFlash expects packed batch size 1")
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

        full_masks, sliding_masks, anchor_positions, anchor_valid = (
            self._build_dual_stream_attention_masks(
                loss_mask, max_anchors, document_ids, device
            )
        )
        mask_tokens_size = max_anchors * self.block_size
        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        horizon_ids = torch.arange(mask_tokens_size, device=device) % self.block_size
        noise_embedding = self.embed_tokens(mask_token_ids)
        noise_embedding = noise_embedding + self.horizon_embedding(
            horizon_ids
        ).unsqueeze(0)
        if self.config.anchor_hidden_injection:
            noise_embedding = self._inject_anchor_hidden(
                noise_embedding,
                verifier_last_hidden_states,
                anchor_positions,
            )

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

        for layer_idx, layer in enumerate(self.kv_native_layers):
            attention_masks = (
                sliding_masks
                if layer_idx in self.sliding_window_indices
                else full_masks
            )
            if attention_masks is None:
                raise RuntimeError(f"draft layer {layer_idx} has no attention masks")
            source_index = self._verifier_kv_mapping_indices[layer_idx]
            layer_keys = verifier_keys[:, :, source_index].transpose(1, 2)
            layer_values = verifier_values[:, :, source_index].transpose(1, 2)
            noise_embedding = layer(
                hidden_states=noise_embedding,
                verifier_keys=layer_keys,
                verifier_values=layer_values,
                attention_masks=attention_masks,
                verifier_position_embeddings=verifier_position_embeddings,
                **kwargs,
            )

        final_hidden = self.norm(noise_embedding)
        logits = self.lm_head(final_hidden)
        aligned_loss_mask = loss_mask[:, anchored_indices]
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )
        aligned_loss_mask[:, :: self.block_size] = 0

        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=False,
        )

        one = torch.ones((), device=loss.device)
        for layer_index, layer in enumerate(self.kv_native_layers):
            metrics[f"context_scale_layer_{layer_index}_sum"] = (
                layer.context_scale.detach()
            )
            metrics[f"context_scale_layer_{layer_index}_total"] = one.clone()
        return None, loss, metrics
