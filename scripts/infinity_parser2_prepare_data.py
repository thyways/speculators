#!/usr/bin/env python3
"""Prepare a deterministic ranked Infinity-Parser2 dataset for training."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from speculators.data_generation.preprocessing import (
    build_speculator_training_dataset,
    load_processor,
    load_raw_dataset,
)
from speculators.train.vocab_mapping import save_token_frequency_distribution

SELECTION_NAME = "ranked_selection.jsonl"
MANIFEST_NAME = "ranked_selection.json"
SHA256_HEX_LENGTH = 64
_PRESERVED_COLUMNS = ("_id", "_source_line_index", "_candidate_rank")
_HASH_COLUMNS = (
    "_sample_manifest_sha256",
    "_generation_config_sha256",
)
_OVERWRITE_FILES = {
    "dataset_info.json",
    "state.json",
    "token_freq.pt",
    SELECTION_NAME,
    MANIFEST_NAME,
}


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _flatten_ranked_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    record_id = row.get("id")
    provenance = row.get("provenance")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("each ranked row must contain a non-empty string id")
    if not isinstance(provenance, Mapping):
        raise ValueError("each ranked row must contain provenance metadata")

    source_index = provenance.get("source_line_index")
    candidate_rank = provenance.get("candidate_rank")
    if type(source_index) is not int or source_index < 0:
        raise ValueError("source_line_index must be a non-negative integer")
    if type(candidate_rank) is not int or candidate_rank < 0:
        raise ValueError("candidate_rank must be a non-negative integer")

    return {
        "_id": record_id,
        "_source_line_index": source_index,
        "_candidate_rank": candidate_rank,
        "_sample_manifest_sha256": _require_sha256(
            provenance.get("sample_manifest_sha256"),
            "sample_manifest_sha256",
        ),
        "_generation_config_sha256": _require_sha256(
            provenance.get("generation_config_sha256"),
            "generation_config_sha256",
        ),
    }


def _single_value(dataset: Dataset, column: str) -> str:
    values = dataset.unique(column)
    if len(values) != 1:
        raise ValueError(f"{column.removeprefix('_')} is not consistent")
    return str(values[0])


def _validate_ranked_rows(dataset: Dataset) -> tuple[str, str]:
    if len(dataset) == 0:
        raise ValueError("the ranked regeneration pool is empty")

    ranks = dataset.unique("_candidate_rank")
    if len(ranks) != len(dataset):
        raise ValueError("candidate_rank values are not unique")
    record_ids = dataset.unique("_id")
    if len(record_ids) != len(dataset):
        raise ValueError("record ids are not unique")

    return (
        _single_value(dataset, "_sample_manifest_sha256"),
        _single_value(dataset, "_generation_config_sha256"),
    )


def prepare_ranked_dataset(args: argparse.Namespace) -> Dataset:
    """Tokenize the ranked reserve and select the first surviving target rows."""
    if args.ranked_target_samples <= 0:
        raise ValueError("ranked_target_samples must be positive")
    if not 0.0 < args.token_freq_train_ratio <= 1.0:
        raise ValueError("token_freq_train_ratio must be in (0, 1]")
    if args.preprocessing_batch_size <= 0:
        raise ValueError("preprocessing_batch_size must be positive")

    raw_dataset, normalize_fn = load_raw_dataset(str(args.data))
    if normalize_fn is not None:
        raw_dataset = raw_dataset.map(
            normalize_fn,
            num_proc=args.num_preprocessing_workers,
            keep_in_memory=False,
        )
    raw_dataset = raw_dataset.map(
        _flatten_ranked_metadata,
        num_proc=args.num_preprocessing_workers,
        keep_in_memory=False,
        batch_size=args.preprocessing_batch_size,
        desc="Validating ranked regeneration metadata",
    )
    sample_sha256, generation_sha256 = _validate_ranked_rows(raw_dataset)
    raw_dataset = raw_dataset.sort(
        "_candidate_rank",
        keep_in_memory=False,
        writer_batch_size=args.preprocessing_batch_size,
    )

    processor = load_processor(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    processed = build_speculator_training_dataset(
        dataset=raw_dataset,
        processor=processor,
        max_length=args.seq_length,
        num_proc=args.num_preprocessing_workers,
        render_endpoint=args.render_endpoint,
        local_render=args.local_render,
        drop_clipped_rows=args.drop_clipped_rows,
        minimum_valid_tokens=args.minimum_valid_tokens,
        preserve_columns=_PRESERVED_COLUMNS,
        keep_in_memory=False,
        map_batch_size=args.preprocessing_batch_size,
        render_chat_template_kwargs={"enable_thinking": False},
    ).with_format(None)

    eligible_ranks = sorted(int(rank) for rank in processed.unique("_candidate_rank"))
    eligible_records = len(eligible_ranks)
    if eligible_records < args.ranked_target_samples:
        raise ValueError(
            "preprocessing survivors are below target: "
            f"{eligible_records} < {args.ranked_target_samples}"
        )

    selected_ranks = eligible_ranks[: args.ranked_target_samples]
    selected_cutoff = selected_ranks[-1]
    selected = processed.filter(
        lambda ranks: [rank <= selected_cutoff for rank in ranks],
        input_columns=["_candidate_rank"],
        batched=True,
        batch_size=args.preprocessing_batch_size,
        num_proc=args.num_preprocessing_workers,
        keep_in_memory=False,
        desc="Selecting ranked records after assistant-turn fan-out",
    )

    train_records = int(args.ranked_target_samples * args.token_freq_train_ratio)
    if train_records <= 0:
        raise ValueError("token_freq_train_ratio leaves an empty training prefix")
    train_cutoff = selected_ranks[train_records - 1]
    token_frequency_rows = selected.filter(
        lambda ranks: [rank <= train_cutoff for rank in ranks],
        input_columns=["_candidate_rank"],
        batched=True,
        batch_size=args.preprocessing_batch_size,
        num_proc=args.num_preprocessing_workers,
        keep_in_memory=False,
        desc="Selecting record prefix for token frequencies",
    )

    selected = selected.rename_columns(
        {
            "_id": "id",
            "_source_line_index": "source_line_index",
            "_candidate_rank": "candidate_rank",
        }
    )
    selected.info.description = json.dumps(
        {
            "ranked_preprocessing": {
                "dataset_order": "candidate_rank_ascending_assistant_turn",
                "eligible_records": eligible_records,
                "eligible_training_rows": len(processed),
                "generation_config_sha256": generation_sha256,
                "sample_manifest_sha256": sample_sha256,
                "selected_records": args.ranked_target_samples,
                "selected_training_rows": len(selected),
            }
        },
        sort_keys=True,
    )

    token_frequency_rows.set_format(type="torch")
    save_token_frequency_distribution(
        dataset=token_frequency_rows,
        output_path=Path(args.output) / "token_freq.pt",
    )
    selected.set_format(type="torch")
    return selected


def _ranked_metadata(dataset: Dataset) -> dict[str, Any]:
    try:
        description = json.loads(dataset.info.description or "{}")
        metadata = description["ranked_preprocessing"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("dataset is missing ranked preprocessing metadata") from exc
    if not isinstance(metadata, dict):
        raise ValueError("ranked preprocessing metadata must be an object")
    return metadata


def _unique_selection_rows(dataset: Dataset) -> list[tuple[int, int, str]]:
    """Collapse contiguous assistant-turn rows back to their source records."""
    plain = dataset.with_format(None)
    rows: list[tuple[int, int, str]] = []
    previous: tuple[int, int, str] | None = None
    for source_index, candidate_rank, record_id in zip(
        plain["source_line_index"],
        plain["candidate_rank"],
        plain["id"],
        strict=True,
    ):
        current = (int(source_index), int(candidate_rank), str(record_id))
        if previous is not None and current[1] == previous[1]:
            if current != previous:
                raise ValueError(
                    "assistant-turn rows disagree on source record metadata"
                )
            continue
        if previous is not None and current[1] < previous[1]:
            raise ValueError("prepared rows are not ordered by candidate rank")
        rows.append(current)
        previous = current
    return rows


def stage_selection(
    dataset: Dataset,
    *,
    output: Path,
    target_samples: int,
) -> tuple[Path, dict[str, Any]]:
    """Write the selection payload to a partial file before dataset publish."""
    rows = _unique_selection_rows(dataset)
    if len(rows) != target_samples:
        raise ValueError(
            f"selected dataset has {len(rows)} records, expected {target_samples}"
        )
    metadata = _ranked_metadata(dataset)
    sample_sha256 = _require_sha256(
        metadata.get("sample_manifest_sha256"),
        "sample_manifest_sha256",
    )
    generation_sha256 = _require_sha256(
        metadata.get("generation_config_sha256"),
        "generation_config_sha256",
    )
    if metadata.get("selected_records") != target_samples:
        raise ValueError("dataset metadata has a different selected record count")
    if metadata.get("selected_training_rows") != len(dataset):
        raise ValueError("dataset metadata has a different training row count")

    output.mkdir(parents=True, exist_ok=True)
    staged = output / f".{SELECTION_NAME}.{os.getpid()}.partial"
    staged.unlink(missing_ok=True)
    rows.sort()
    with staged.open("wt", encoding="utf-8") as handle:
        for source_index, candidate_rank, record_id in rows:
            handle.write(
                json.dumps(
                    {
                        "id": record_id,
                        "source_line_index": source_index,
                        "candidate_rank": candidate_rank,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    manifest = {
        "complete": True,
        "target_records": target_samples,
        "training_rows": len(dataset),
        "sample_manifest_sha256": sample_sha256,
        "generation_config_sha256": generation_sha256,
        "selection": {
            "path": SELECTION_NAME,
            "records": target_samples,
        },
    }
    return staged, manifest


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    staged = path.with_name(f".{path.name}.{os.getpid()}.partial")
    staged.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(staged, path)


def publish_selection(
    output: Path,
    *,
    staged_selection: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Publish the selection and write its completion manifest last."""
    os.replace(staged_selection, output / SELECTION_NAME)
    _write_json_atomic(output / MANIFEST_NAME, manifest)


def validate_existing_output(args: argparse.Namespace) -> None:
    """Validate a previously completed output before treating it as reusable."""
    output = Path(args.output)
    manifest_path = output / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"existing output is incomplete: missing {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read selection manifest: {manifest_path}") from exc
    selection = manifest.get("selection")
    training_rows = manifest.get("training_rows")
    if (
        manifest.get("complete") is not True
        or manifest.get("target_records") != args.ranked_target_samples
        or type(training_rows) is not int
        or training_rows < args.ranked_target_samples
        or not isinstance(selection, dict)
        or selection.get("path") != SELECTION_NAME
        or selection.get("records") != args.ranked_target_samples
    ):
        raise ValueError("existing prepared output was created with different settings")

    selection_path = output / SELECTION_NAME
    if not selection_path.is_file():
        raise ValueError(f"existing output is missing {selection_path}")
    with selection_path.open("rt", encoding="utf-8") as handle:
        selection_records = sum(1 for line in handle if line.strip())
    if selection_records != args.ranked_target_samples:
        raise ValueError("existing selection has a different record count")

    dataset = load_from_disk(output)
    if len(dataset) != training_rows:
        raise ValueError("existing dataset has a different training row count")
    metadata = _ranked_metadata(dataset)
    if (
        metadata.get("selected_records") != args.ranked_target_samples
        or metadata.get("selected_training_rows") != training_rows
    ):
        raise ValueError("existing dataset metadata has different record counts")


def _assert_safe_to_overwrite(output: Path) -> None:
    unexpected = []
    for path in output.iterdir():
        if path.is_file() and (
            path.suffix == ".arrow"
            or path.name in _OVERWRITE_FILES
            or (path.name.startswith(".") and path.name.endswith(".partial"))
        ):
            continue
        unexpected.append(path)
    if unexpected:
        paths = ", ".join(str(path) for path in unexpected)
        raise ValueError(
            "--overwrite would remove files outside the prepared-data "
            f"artifacts: {paths}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare ranked Infinity-Parser2 regeneration data"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, default=20480)
    parser.add_argument("--ranked-target-samples", type=int, required=True)
    parser.add_argument("--token-freq-train-ratio", type=float, default=0.99)
    render = parser.add_mutually_exclusive_group(required=True)
    render.add_argument("--render-endpoint")
    parser.add_argument(
        "--drop-clipped-rows",
        action="store_true",
        help="drop rows longer than --seq-length instead of truncating them. "
        "Required for online training, where the stored conversation is "
        "re-rendered and a truncated row's ids can never match; offline "
        "and text runs should leave it off and keep truncating.",
    )
    render.add_argument(
        "--local-render",
        action="store_true",
        help="tokenize with the processor in-process instead of calling the "
        "vLLM /render endpoint; same ids, far cheaper for image corpora",
    )
    parser.add_argument("--minimum-valid-tokens", type=int)
    parser.add_argument("--num-preprocessing-workers", type=int, default=8)
    parser.add_argument("--preprocessing-batch-size", type=int, default=1000)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    args.output = output
    if output.exists():
        if args.overwrite:
            _assert_safe_to_overwrite(output)
            shutil.rmtree(output)
        elif any(output.iterdir()):
            validate_existing_output(args)
            print(f"prepared dataset already complete: {output}")
            return 0
    output.mkdir(parents=True, exist_ok=True)

    dataset = prepare_ranked_dataset(args)
    staged_selection, manifest = stage_selection(
        dataset,
        output=output,
        target_samples=args.ranked_target_samples,
    )
    try:
        dataset.save_to_disk(output)
        publish_selection(
            output,
            staged_selection=staged_selection,
            manifest=manifest,
        )
    except Exception:
        staged_selection.unlink(missing_ok=True)
        raise
    print(
        f"prepared {len(dataset)} assistant-turn rows from "
        f"{args.ranked_target_samples} ranked records: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
