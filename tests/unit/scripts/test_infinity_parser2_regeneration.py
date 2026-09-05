"""Tests for the compact Infinity-Parser2 regeneration pipeline."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "infinity_parser2_regeneration"
    / "script.py"
)
SCRIPT_DIR = SCRIPT_PATH.parent


def load_module():
    name = "infinity_parser2_regeneration_for_tests"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


regen = load_module()


def source_row(image: Path, index: int) -> dict:
    return {
        "images": [str(image)],
        "conversations": [
            {"from": "human", "value": f"<image> question {index}"},
            {"from": "gpt", "value": f"old answer {index}"},
        ],
        "attributes": {"dataset_name": "unit-test", "row": index},
    }


def sample_args(source: Path, output_root: Path, media_root: Path):
    return regen.build_parser().parse_args(
        [
            "sample",
            "--source",
            str(source),
            "--output-root",
            str(output_root),
            "--population-size",
            "6",
            "--sample-size",
            "4",
            "--pilot-size",
            "2",
            "--reserve-size",
            "1",
            "--seed",
            "7",
            "--allowed-media-root",
            str(media_root),
        ]
    )


def populate_successes(output_root: Path) -> tuple[str, str]:
    connection = regen.open_state(output_root, write=True)
    sample_hash = regen.get_meta(connection, "sample_sha256")
    generation = {
        "sample_sha256": sample_hash,
        "model": "unit-test-model",
        "model_fingerprint": {
            "model_path": "/tmp/unit-test-model",
            "files": [],
            "sha256": "a" * 64,
        },
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    generation_hash = regen.sha256_json(generation)
    regen.set_meta(connection, "generation_config", generation)
    regen.set_meta(connection, "generation_sha256", generation_hash)

    rows = connection.execute(
        "SELECT id, rank, source_index, record FROM requests ORDER BY rank"
    ).fetchall()
    for row in rows:
        record = json.loads(row["record"])
        answer = f"new answer {row['rank']}"
        event = {
            "event": "success",
            "id": row["id"],
            "source_line_index": row["source_index"],
            "candidate_rank": row["rank"],
            "generation_config_sha256": generation_hash,
            "endpoint": "http://teacher.invalid/v1/chat/completions",
            "attempts": 1,
            "latency_seconds": 0.01,
            "turns": [
                {
                    "assistant_ordinal": 0,
                    "request_seed": regen.derive_request_seed(
                        42, row["source_index"], 0
                    ),
                    "attempts": 1,
                    "latency_seconds": 0.01,
                    "finish_reason": "stop",
                    "answer": answer,
                    "answer_sha256": regen.sha256_bytes(answer.encode()),
                    "usage": {},
                    "response_id": None,
                }
            ],
            "created_at": regen.utc_now(),
        }
        assert record["provenance"]["assistant_turns"] == 1
        connection.execute(
            "INSERT INTO generations(id, status, event) VALUES (?, ?, ?)",
            (row["id"], "success", regen.json_bytes(event).decode()),
        )
    connection.commit()
    connection.close()
    return sample_hash, generation_hash


def test_directory_contains_only_the_real_entrypoints():
    # Guards against scratch scripts accumulating alongside the real launchers.
    assert {path.name for path in SCRIPT_DIR.iterdir() if path.is_file()} == {
        "prepare_fast.sh",
        "run_4node_dp.sh",
        "run_all.sh",
        "script.py",
    }


@pytest.mark.parametrize("action", ["generate", "full"])
@pytest.mark.parametrize("retry_errors", [False, True])
def test_run_all_retries_errors_even_when_stage_is_complete(
    tmp_path: Path, action: str, retry_errors: bool
):
    # Stub the pipeline processes, leaving the real shell control flow intact.
    # status exits 0, as it does when every row has already returned HTTP 400.
    calls_path = tmp_path / "calls.jsonl"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['REGEN_TEST_CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    fake_python.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PARSER2_")
    }
    env.update(
        REGEN_TEST_CALLS=str(calls_path),
        PARSER2_PYTHON_BIN=str(fake_python),
        PARSER2_DATA_ROOT=str(tmp_path / "data"),
        PARSER2_ENDPOINTS="http://teacher.invalid/v1/chat/completions",
        PARSER2_PREPARE_MODE="render",
        PARSER2_RETRY_ERRORS=str(int(retry_errors)),
        PARSER2_ALLOW_CONFIG_CHANGE="1",
    )
    subprocess.run(  # noqa: S603
        ["/bin/bash", str(SCRIPT_DIR / "run_all.sh"), action],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    generations = [
        call for call in calls if call[0] == str(SCRIPT_PATH) and call[1] == "generate"
    ]
    assert len(generations) == int(retry_errors)
    if retry_errors:
        assert "--retry-errors" in generations[0]
        assert "--allow-config-change" in generations[0]
        assert generations[0][generations[0].index("--max-tokens") + 1] == "16384"


def test_multiturn_skeleton_drops_old_answers_and_preserves_image_order():
    images = [Path("/tmp/first.png"), Path("/tmp/second.png")]
    conversations = [
        {"from": "human", "value": "<image> first"},
        {"from": "gpt", "value": "old first"},
        {"from": "human", "value": "<image> second"},
        {"from": "gpt", "value": "old second"},
    ]

    skeleton, old_answers = regen.build_conversation_skeleton(conversations, images)

    assert old_answers == ["old first", "old second"]
    assert [turn["role"] for turn in skeleton] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert skeleton[0]["content"][0]["path"] == str(images[0])
    assert skeleton[2]["content"][0]["path"] == str(images[1])
    assert "old first" not in repr(skeleton)


def test_ranked_sample_is_deterministic_and_stages_are_nested():
    membership, ranks = regen.splitmix64_floyd_ranked(1000, 100, 123)
    membership_again, ranks_again = regen.splitmix64_floyd_ranked(1000, 100, 123)
    assert membership == membership_again
    assert ranks.tolist() == ranks_again.tolist()

    valid = bytearray((100 + 7) // 8)
    for rank in range(100):
        regen.set_bit(valid, rank)
    stages = regen.build_stages(
        valid_ranks=valid,
        candidate_size=100,
        valid_count=100,
        sample_size=80,
        pilot_size=40,
        reserve_size=10,
    )
    assert stages["smoke"]["candidate_rank_exclusive"] == 80
    assert stages["pilot"]["candidate_rank_exclusive"] == 50
    assert stages["full"]["candidate_rank_exclusive"] == 90


def test_sample_status_and_ranked_export(tmp_path: Path):
    media_root = tmp_path / "media"
    media_root.mkdir()
    rows = []
    for index in range(6):
        image = media_root / f"{index}.png"
        image.write_bytes(b"image")
        rows.append(source_row(image, index))
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(regen.json_bytes(row) + b"\n" for row in rows))
    output_root = tmp_path / "state"

    assert regen.command_sample(sample_args(source, output_root, media_root)) == 0
    connection = regen.open_state(output_root)
    assert connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 5
    connection.close()

    sample_hash, generation_hash = populate_successes(output_root)
    connection = regen.open_state(output_root)
    status = regen.status_payload(connection, "full")
    connection.close()
    assert status["complete"] is True
    assert status["success"] == 5

    pool = tmp_path / "pool.jsonl"
    pool_args = regen.build_parser().parse_args(
        [
            "export",
            "--output-root",
            str(output_root),
            "--stage",
            "full",
            "--output",
            str(pool),
            "--minimum-count",
            "4",
        ]
    )
    assert regen.command_export(pool_args) == 0
    pool_rows = [json.loads(line) for line in pool.read_text().splitlines()]
    assert len(pool_rows) == 5
    assert [row["provenance"]["candidate_rank"] for row in pool_rows] == list(range(5))
    assert "old answer" not in pool.read_text()

    selection_dir = tmp_path / "prepared"
    selection_dir.mkdir()
    connection = regen.open_state(output_root)
    selected_rows = connection.execute(
        "SELECT id, rank, source_index FROM requests "
        "WHERE rank < 4 ORDER BY source_index"
    ).fetchall()
    connection.close()
    selection_path = selection_dir / "ranked_selection.jsonl"
    selection_path.write_bytes(
        b"".join(
            regen.json_bytes(
                {
                    "id": row["id"],
                    "source_line_index": row["source_index"],
                    "candidate_rank": row["rank"],
                }
            )
            + b"\n"
            for row in selected_rows
        )
    )
    manifest_path = selection_dir / "ranked_selection.json"
    regen.write_json(
        manifest_path,
        {
            "complete": True,
            "target_records": 4,
            "sample_manifest_sha256": sample_hash,
            "generation_config_sha256": generation_hash,
            "selection": {"path": selection_path.name, "records": 4},
        },
    )

    final = tmp_path / "final.jsonl"
    final_args = regen.build_parser().parse_args(
        [
            "export",
            "--output-root",
            str(output_root),
            "--stage",
            "full",
            "--output",
            str(final),
            "--selection-manifest",
            str(manifest_path),
        ]
    )
    assert regen.command_export(final_args) == 0
    final_rows = [json.loads(line) for line in final.read_text().splitlines()]
    assert len(final_rows) == 4
    assert [row["provenance"]["candidate_rank"] for row in final_rows] == list(range(4))


@pytest.mark.parametrize("max_tokens", [128, 16384])
def test_generate_event_regenerates_assistant_turns_in_order(
    monkeypatch: pytest.MonkeyPatch,
    max_tokens: int,
):
    skeleton, _ = regen.build_conversation_skeleton(
        [
            {"from": "human", "value": "first"},
            {"from": "gpt", "value": "discard first"},
            {"from": "human", "value": "second"},
            {"from": "gpt", "value": "discard second"},
        ],
        [],
    )
    record = {
        "id": "v1.12:0000000017",
        "conversations": skeleton,
        "attributes": {},
        "provenance": {
            "source_line_index": 17,
            "candidate_rank": 3,
            "assistant_turns": 2,
        },
    }
    payloads = []

    async def fake_post_chat(session, endpoint, payload, *, max_retries):
        del session, endpoint, max_retries
        payloads.append(payload)
        index = len(payloads) - 1
        return (
            {
                "id": f"response-{index}",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": f"new {index}"},
                    }
                ],
                "usage": {},
            },
            1,
        )

    monkeypatch.setattr(regen, "post_chat", fake_post_chat)
    args = argparse.Namespace(
        model="unit-test",
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        max_retries=2,
    )
    event = asyncio.run(
        regen.generate_event(
            object(),
            "http://teacher.invalid/v1/chat/completions",
            record,
            generation_sha256="b" * 64,
            args=args,
        )
    )

    assert event["event"] == "success"
    assert [turn["answer"] for turn in event["turns"]] == ["new 0", "new 1"]
    assert [message["role"] for message in payloads[0]["messages"]] == ["user"]
    assert [message["role"] for message in payloads[1]["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert "discard" not in repr(payloads)
    assert all(payload["max_tokens"] == max_tokens for payload in payloads)


def test_async_generation_persists_results_incrementally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    media_root = tmp_path / "media"
    media_root.mkdir()
    rows = []
    for index in range(6):
        image = media_root / f"{index}.png"
        image.write_bytes(b"image")
        rows.append(source_row(image, index))
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(regen.json_bytes(row) + b"\n" for row in rows))
    output_root = tmp_path / "state"
    regen.command_sample(sample_args(source, output_root, media_root))

    async def fake_verify(session, endpoints, model):
        del session, endpoints, model

    async def fake_generate(
        session,
        endpoint,
        record,
        *,
        generation_sha256,
        args,
    ):
        del session, args
        return {
            "event": "success",
            "id": record["id"],
            "source_line_index": record["provenance"]["source_line_index"],
            "candidate_rank": record["provenance"]["candidate_rank"],
            "generation_config_sha256": generation_sha256,
            "endpoint": endpoint,
            "attempts": 1,
            "latency_seconds": 0.01,
            "turns": [],
            "created_at": regen.utc_now(),
        }

    monkeypatch.setattr(regen, "verify_endpoints", fake_verify)
    monkeypatch.setattr(regen, "generate_event", fake_generate)
    connection = regen.open_state(output_root)
    stage = regen.stage_config(connection, "full")
    connection.close()
    args = argparse.Namespace(
        concurrency_per_endpoint=2,
        timeout=10.0,
        connect_timeout=1.0,
        api_key_env="UNIT_TEST_API_KEY",
        retry_errors=False,
        commit_batch=2,
        model="unit-test",
    )
    asyncio.run(
        regen.generate_async(
            db_path=regen.state_path(output_root),
            rank_limit=stage["candidate_rank_exclusive"],
            pending=5,
            endpoints=["http://teacher.invalid/v1/chat/completions"],
            generation_sha256="c" * 64,
            args=args,
        )
    )
    connection = regen.open_state(output_root)
    assert regen.generation_counts(connection, stage["candidate_rank_exclusive"]) == {
        "success": 5,
        "error": 0,
        "pending": 0,
    }
    connection.close()


def test_cli_only_exposes_meaningful_commands():
    parser = regen.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == {
        "sample",
        "generate",
        "status",
        "export",
    }
