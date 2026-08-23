import argparse
import json
from pathlib import Path

import pytest
from datasets import Dataset

from scripts import infinity_parser2_prepare_data as prepare_module


class _FakeProcessor:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, *args, **kwargs):
        return ""

    def decode(self, _tokens):
        return "token"


def _raw_dataset(candidate_ranks: list[int]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "id": f"v1.12:{source_index:010d}",
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
                "provenance": {
                    "source_line_index": source_index,
                    "candidate_rank": candidate_rank,
                    "sample_manifest_sha256": "a" * 64,
                    "generation_config_sha256": "b" * 64,
                },
            }
            for source_index, candidate_rank in enumerate(candidate_ranks)
        ]
    )


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    raw: Dataset,
    *,
    dropped_ranks: set[int] | None = None,
    fanout_by_rank: dict[int, int] | None = None,
) -> None:
    dropped_ranks = dropped_ranks or set()
    fanout_by_rank = fanout_by_rank or {}

    def fake_build(dataset, *, preserve_columns, **_kwargs):
        rows = [
            dataset[index]
            for index in range(len(dataset))
            if dataset[index]["_candidate_rank"] not in dropped_ranks
            for _ in range(fanout_by_rank.get(dataset[index]["_candidate_rank"], 1))
        ]
        result = Dataset.from_dict(
            {
                "input_ids": [[10, 11] for _ in rows],
                "loss_mask": [[0, 1] for _ in rows],
                "seq_len": [2 for _ in rows],
                **{
                    column: [row[column] for row in rows] for column in preserve_columns
                },
            }
        )
        result.set_format(type="torch")
        return result

    monkeypatch.setattr(
        prepare_module,
        "load_processor",
        lambda *args, **kwargs: _FakeProcessor(),
    )
    monkeypatch.setattr(
        prepare_module,
        "load_raw_dataset",
        lambda _path: (raw, None),
    )
    monkeypatch.setattr(
        prepare_module,
        "build_speculator_training_dataset",
        fake_build,
    )


def _args(tmp_path: Path, target_samples: int) -> argparse.Namespace:
    return argparse.Namespace(
        model="target-model",
        data=tmp_path / "success-pool.jsonl",
        output=tmp_path,
        ranked_target_samples=target_samples,
        seq_length=8,
        token_freq_train_ratio=0.67,
        render_endpoint="http://127.0.0.1:8000",
        # Both are store_true flags, so False is their parser default. Rendering
        # goes through the endpoint set above, which --local-render is the
        # alternative to.
        local_render=False,
        drop_clipped_rows=False,
        minimum_valid_tokens=None,
        num_preprocessing_workers=1,
        preprocessing_batch_size=2,
        trust_remote_code=False,
    )


def test_ranked_preprocessing_backfills_and_uses_train_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _patch_pipeline(
        monkeypatch,
        _raw_dataset([4, 0, 3, 1, 2]),
        dropped_ranks={1},
        fanout_by_rank={0: 2, 2: 3},
    )
    captured = {}

    def capture_frequency(*, dataset, output_path):
        captured["dataset"] = dataset.with_format(None)
        captured["output_path"] = output_path

    monkeypatch.setattr(
        prepare_module,
        "save_token_frequency_distribution",
        capture_frequency,
    )
    args = _args(tmp_path, 3)
    dataset = prepare_module.prepare_ranked_dataset(args)

    plain = dataset.with_format(None)
    assert plain["candidate_rank"] == [0, 0, 2, 2, 2, 3]
    assert captured["dataset"]["_candidate_rank"] == [0, 0, 2, 2, 2]
    assert captured["output_path"] == tmp_path / "token_freq.pt"
    metadata = json.loads(dataset.info.description)["ranked_preprocessing"]
    assert metadata == {
        "dataset_order": "candidate_rank_ascending_assistant_turn",
        "eligible_records": 4,
        "eligible_training_rows": 7,
        "generation_config_sha256": "b" * 64,
        "sample_manifest_sha256": "a" * 64,
        "selected_records": 3,
        "selected_training_rows": 6,
    }


def test_ranked_preprocessing_checks_the_complete_reserve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _patch_pipeline(monkeypatch, _raw_dataset([0, 1, 2, 2]))
    monkeypatch.setattr(
        prepare_module,
        "save_token_frequency_distribution",
        lambda **kwargs: None,
    )

    with pytest.raises(ValueError, match="not unique"):
        prepare_module.prepare_ranked_dataset(_args(tmp_path, 2))


def test_ranked_preprocessing_requires_enough_survivors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _patch_pipeline(monkeypatch, _raw_dataset([0, 1]))
    monkeypatch.setattr(
        prepare_module,
        "save_token_frequency_distribution",
        lambda **kwargs: None,
    )

    with pytest.raises(ValueError, match="below target"):
        prepare_module.prepare_ranked_dataset(_args(tmp_path, 3))


def test_selection_manifest_is_minimal_and_published_last(tmp_path: Path):
    output = tmp_path / "prepared"
    output.mkdir()
    dataset = Dataset.from_dict(
        {
            "input_ids": [[10, 11], [10, 12], [12, 13]],
            "loss_mask": [[0, 1], [0, 1], [0, 1]],
            "seq_len": [2, 2, 2],
            "id": [
                "v1.12:0000000003",
                "v1.12:0000000003",
                "v1.12:0000000001",
            ],
            "source_line_index": [3, 3, 1],
            "candidate_rank": [0, 0, 2],
        }
    )
    dataset.info.description = json.dumps(
        {
            "ranked_preprocessing": {
                "sample_manifest_sha256": "a" * 64,
                "generation_config_sha256": "b" * 64,
                "selected_records": 2,
                "selected_training_rows": 3,
            }
        }
    )

    staged, manifest = prepare_module.stage_selection(
        dataset,
        output=output,
        target_samples=2,
    )
    assert manifest == {
        "complete": True,
        "target_records": 2,
        "training_rows": 3,
        "sample_manifest_sha256": "a" * 64,
        "generation_config_sha256": "b" * 64,
        "selection": {
            "path": prepare_module.SELECTION_NAME,
            "records": 2,
        },
    }
    assert not (output / prepare_module.MANIFEST_NAME).exists()

    dataset.save_to_disk(output)
    prepare_module.publish_selection(
        output,
        staged_selection=staged,
        manifest=manifest,
    )
    rows = [
        json.loads(line)
        for line in (output / prepare_module.SELECTION_NAME).read_text().splitlines()
    ]
    assert [row["source_line_index"] for row in rows] == [1, 3]
    assert not list(output.glob("*.partial"))

    args = argparse.Namespace(output=output, ranked_target_samples=2)
    prepare_module.validate_existing_output(args)
    args.ranked_target_samples = 3
    with pytest.raises(ValueError, match="different settings"):
        prepare_module.validate_existing_output(args)
