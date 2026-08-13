"""Domino checkpoint converter.

Converts an external Domino checkpoint (the upstream ``DominoDraftModel``
layout, e.g. produced by SpecForge) to a Speculators checkpoint that loads with
``DominoDraftModel.from_pretrained(path)``.

The draft transformer body (``layers.*``, ``fc``, ``hidden_norm``, ``norm``) is
DFlash's and copies as-is. The recurrent head is the only structural
difference: upstream stores ``prefix_gru`` / ``embed_proj`` at the top level (or
under a ``logit_head`` container in early checkpoints), while Speculators groups
them under ``logits_correction``, so those keys are renamed. The GRU parameter
names and math are identical, so no weight surgery is needed. The external
checkpoint borrows the verifier's embedding and LM head at runtime, so
``embed_tokens`` / ``lm_head`` / ``verifier_lm_head`` / ``verifier_norm`` are
loaded from the verifier before saving.

Upstream's ``embed_proj`` output layer spans the full verifier vocabulary, so
the converted checkpoint only makes sense with
``draft_vocab_size == vocab_size``; the converter asserts this.
"""

from pathlib import Path

import torch
from loguru import logger
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.utils import (
    ensure_checkpoint_is_local,
    load_checkpoint_config,
    load_checkpoint_weights,
)
from speculators.models.domino import DominoDraftModel, DominoSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

__all__ = ["DominoConverter"]

# config.json keys that are not part of the draft transformer (Qwen3) config.
# ``emb_dim`` is duplicated at the top level by some upstream Domino configs
# alongside its canonical home in ``dflash_config``.
_NON_TRANSFORMER_KEYS = frozenset(
    {
        "architectures",
        "auto_map",
        "block_size",
        "dflash_config",
        "emb_dim",
        "num_target_layers",
    }
)

# state dict keys that are filled from the verifier (not the source checkpoint), so
# their absence from the source weights is expected, not a conversion error
_VERIFIER_FILLED_KEYS = frozenset(
    {
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "verifier_norm.weight",
        "t2d",
        "d2t",
    }
)

# Longest prefix first: the ``logit_head.`` layout must win over the bare one.
_HEAD_PREFIX_RENAMES = (
    ("logit_head.prefix_gru.", "logits_correction.prefix_gru."),
    ("logit_head.embed_proj.", "logits_correction.embed_proj."),
    ("prefix_gru.", "logits_correction.prefix_gru."),
    ("embed_proj.", "logits_correction.embed_proj."),
)


def remap_domino_head_keys(
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Move upstream head weights under the ``logits_correction`` container."""
    remapped: dict[str, torch.Tensor] = {}
    for name, tensor in weights.items():
        remapped_name = name
        for source_prefix, target_prefix in _HEAD_PREFIX_RENAMES:
            if name.startswith(source_prefix):
                remapped_name = target_prefix + name[len(source_prefix) :]
                break
        remapped[remapped_name] = tensor
    return remapped


class DominoConverter:
    """Convert an external Domino checkpoint to speculators format.

    Copies the DFlash draft body as-is, renames the recurrent head onto the
    ``logits_correction`` container, and fills the embedding, LM head, and
    verifier norm from the verifier so the saved checkpoint is self-contained.
    """

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        base_model: str,
        validate: bool = True,
        aux_hidden_state_layer_ids: list[int] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        logger.info(f"Converting Domino checkpoint: {input_path}")

        local_checkpoint_path = ensure_checkpoint_is_local(
            input_path, cache_dir
        )
        source_config = load_checkpoint_config(local_checkpoint_path)
        weights = load_checkpoint_weights(local_checkpoint_path)
        logger.info(f"Loaded {len(weights)} weights")

        config = self._build_config(
            source_config, base_model, aux_hidden_state_layer_ids
        )
        saved_path = self._save(
            config, remap_domino_head_keys(weights), output_path
        )
        logger.success(f"Saved to: {saved_path}")

        if validate:
            self._validate(saved_path)

    def _build_config(
        self,
        source_config: dict,
        base_model: str,
        aux_hidden_state_layer_ids: list[int] | None,
    ) -> DominoSpeculatorConfig:
        domino = source_config.get("dflash_config", {})
        projector_type = domino.get("projector_type")
        if projector_type is not None and projector_type != "domino":
            raise ValueError(
                "Checkpoint is not a Domino draft model: "
                f"dflash_config.projector_type={projector_type!r}"
            )
        # block_size lives at the top level in older checkpoints and inside
        # dflash_config in newer ones.
        block_size = source_config.get("block_size") or domino.get(
            "block_size"
        )
        if block_size is None:
            raise ValueError(
                "Checkpoint config has no `block_size` (checked both top-level "
                "and `dflash_config`)"
            )
        transformer_config = {
            k: v
            for k, v in source_config.items()
            if k not in _NON_TRANSFORMER_KEYS
        }

        verifier_config_dict, _ = PretrainedConfig.get_config_dict(base_model)
        source_hidden = transformer_config.get("hidden_size")
        target_hidden = verifier_config_dict.get("hidden_size")
        if source_hidden and target_hidden and source_hidden != target_hidden:
            raise ValueError(
                f"Architecture mismatch: source Domino checkpoint has "
                f"hidden_size={source_hidden} but base_model '{base_model}' has "
                f"hidden_size={target_hidden}. Dimensions must match."
            )

        # The head's output layer spans the source vocabulary, so the converted
        # draft has no vocabulary mapping. A verifier with a different vocab
        # would otherwise only fail later, while copying its embedding/LM head.
        source_vocab = transformer_config.get("vocab_size")
        target_vocab = verifier_config_dict.get("vocab_size")
        if source_vocab and target_vocab and source_vocab != target_vocab:
            raise ValueError(
                f"Vocabulary mismatch: source Domino checkpoint has "
                f"vocab_size={source_vocab} but base_model '{base_model}' has "
                f"vocab_size={target_vocab}. Domino conversion produces a "
                "full-vocabulary draft (draft_vocab_size == vocab_size), so the "
                "two must match."
            )

        if aux_hidden_state_layer_ids is None:
            target_layer_ids = domino.get("target_layer_ids")
            if target_layer_ids is None:
                raise ValueError(
                    "Checkpoint config has no `dflash_config.target_layer_ids`; "
                    "pass `aux_hidden_state_layer_ids` explicitly."
                )
            # Upstream reads hidden_states[layer_id + 1] (index 0 is the
            # embedding output) while speculators uses the layer id directly.
            aux_hidden_state_layer_ids = [i + 1 for i in target_layer_ids]

        # Upstream's shift_label is speculators' sample_from_anchor: both decide
        # whether the anchor slot predicts a token or is the bonus token.
        sample_from_anchor = bool(domino.get("shift_label", False))
        speculative_tokens = (
            block_size if sample_from_anchor else block_size - 1
        )

        speculators_config = SpeculatorsConfig(
            algorithm="domino",
            proposal_methods=[
                GreedyTokenProposalConfig(
                    speculative_tokens=speculative_tokens
                )
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=base_model,
                architectures=verifier_config_dict.get("architectures", []),
            ),
        )

        emb_dim = domino.get("emb_dim") or source_config.get("emb_dim")
        gru_hidden_dim = domino.get("gru_hidden_dim")
        if emb_dim is None or gru_hidden_dim is None:
            raise ValueError(
                "Checkpoint config is missing the Domino head dimensions "
                "(`dflash_config.emb_dim` / `dflash_config.gru_hidden_dim`)."
            )

        return DominoSpeculatorConfig(
            transformer_layer_config=transformer_config,  # type: ignore[arg-type]
            draft_vocab_size=transformer_config["vocab_size"],
            block_size=block_size,
            aux_hidden_state_layer_ids=aux_hidden_state_layer_ids,
            mask_token_id=domino.get("mask_token_id"),
            sample_from_anchor=sample_from_anchor,
            gru_hidden_dim=gru_hidden_dim,
            logits_correction_emb_dim=emb_dim,
            pure_draft_prefix_len=domino.get("pure_draft_prefix_len", 0),
            speculators_config=speculators_config,
        )

    def _save(
        self,
        config: DominoSpeculatorConfig,
        weights: dict[str, torch.Tensor],
        output_path: str | Path,
    ) -> Path:
        model = DominoDraftModel(config=config)

        body = {k: v for k, v in weights.items() if k not in ("t2d", "d2t")}
        missing, unexpected = model.load_state_dict(body, strict=False)
        if unexpected:
            raise ValueError(
                "Unexpected keys in checkpoint -- the structure does not match "
                f"DominoDraftModel. Unexpected keys: {unexpected}"
            )
        critical_missing = [
            k for k in missing if k not in _VERIFIER_FILLED_KEYS
        ]
        if critical_missing:
            raise ValueError(
                f"Draft weights missing after load: {critical_missing}"
            )
        logger.debug(f"Keys loaded from verifier at save time: {missing}")

        # embed_tokens / lm_head / verifier_lm_head / verifier_norm come from the
        # verifier; without this they would be saved as NaN.
        model.load_verifier_weights()

        model.to(dtype=next(iter(body.values())).dtype)  # type: ignore[call-arg]
        model.save_pretrained(str(output_path))
        return Path(output_path)

    def _validate(self, output_path: Path) -> None:
        logger.info("Validating converted Domino checkpoint...")
        try:
            model = DominoDraftModel.from_pretrained(str(output_path))
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error(f"Validation failed: {exc}")
            raise
        required = (
            "fc.weight",
            "lm_head.weight",
            "embed_tokens.weight",
            "logits_correction.prefix_gru.weight_ih_l0",
            "logits_correction.prefix_gru.weight_hh_l0",
            "logits_correction.embed_proj.0.weight",
            "logits_correction.embed_proj.2.weight",
        )
        state_dict = model.state_dict()
        for name in required:
            if name not in state_dict:
                raise ValueError(f"Converted checkpoint is missing {name}")
            if torch.isnan(state_dict[name]).any():
                raise ValueError(f"Converted checkpoint has NaN in {name}")
        logger.success("Validation succeeded")
