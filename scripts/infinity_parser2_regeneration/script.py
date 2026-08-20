#!/usr/bin/env python3
"""Sample and regenerate Infinity-Parser2 training conversations."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import functools
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import time
from array import array
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

DB_NAME = "state.sqlite3"
SCHEMA_VERSION = 1
SAMPLE_ALGORITHM = "splitmix64-floyd-ranked-v2"
SMOKE_SIZE = 100
UINT64_RANGE = 1 << 64
UINT64_MASK = UINT64_RANGE - 1
UINT32_SENTINEL = (1 << 32) - 1
RETRYABLE_STATUSES = {408, 409, 425, 429}
HTTP_OK = 200
HTTP_REDIRECT = 300
HTTP_SERVER_ERROR = 500
INVALID_EXAMPLE_LIMIT = 20
PROGRESS_BAR_WIDTH = 40
PROGRESS_LOG_INTERVAL = 30.0
BUILD_CACHE_KIB = 8 << 20
REQUEST_INSERT_BATCH_SIZE = 1000
SELECTION_INSERT_BATCH_SIZE = 10000
# Paged by rank so each page is its own read transaction. Holding one cursor
# open for the whole run keeps a WAL read snapshot alive, which blocks
# checkpointing; the WAL then grows without bound (observed: 433 MiB) until
# extending the mmapped -shm index fails on the shared filesystem and the
# process takes SIGBUS.
_PENDING_QUERY = (
    "SELECT r.rank, r.record FROM requests r "
    "LEFT JOIN generations g ON g.id = r.id "
    "WHERE r.rank > ? AND r.rank < ? AND g.id IS NULL "
    "ORDER BY r.rank LIMIT ?"
)
_RETRYABLE_PENDING_QUERY = (
    "SELECT r.rank, r.record FROM requests r "
    "LEFT JOIN generations g ON g.id = r.id "
    "WHERE r.rank > ? AND r.rank < ? AND (g.id IS NULL OR g.status = 'error') "
    "ORDER BY r.rank LIMIT ?"
)
PENDING_PAGE_SIZE = 2048
_PENDING_COUNT_QUERY = (
    "SELECT COUNT(*) AS count FROM requests r "
    "LEFT JOIN generations g ON g.id = r.id "
    "WHERE r.rank < ? AND g.id IS NULL"
)
_RETRYABLE_PENDING_COUNT_QUERY = (
    "SELECT COUNT(*) AS count FROM requests r "
    "LEFT JOIN generations g ON g.id = r.id "
    "WHERE r.rank < ? AND (g.id IS NULL OR g.status = 'error')"
)
UNRESOLVED_PLACEHOLDER = re.compile(
    r"<\s*(?:image|bbox|ref[-_ ]?object|object[-_ ]?ref|"
    r"grounding[-_ ]?(?:box|ref)|box|point|polygon|quad)"
    r"(?:\s[^>]*)?>",
    re.IGNORECASE,
)


class PipelineError(RuntimeError):
    """A user-facing pipeline error."""


class RequestError(PipelineError):
    """A generation failure recorded as an error event."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        attempts: int = 1,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.attempts = attempts
        self.status_code = status_code
        self.retryable = retryable


@dataclass(frozen=True)
class PrefixMap:
    source: str
    destination: str


class SplitMix64:
    """Small deterministic PRNG used only for exact index sampling."""

    def __init__(self, seed: int) -> None:
        if not 0 <= seed <= UINT64_MASK:
            raise PipelineError(f"seed must be in [0, {UINT64_MASK}]")
        self.state = seed

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & UINT64_MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
        return (value ^ (value >> 31)) & UINT64_MASK

    def randbelow(self, upper: int) -> int:
        if not 0 < upper <= UINT64_RANGE:
            raise PipelineError("randbelow upper bound is invalid")
        limit = UINT64_RANGE - UINT64_RANGE % upper
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


class ProgressBar:
    """Single-line stderr progress bar with rate and ETA."""

    def __init__(self, total: int, prefix: str, interval: float = 1.0) -> None:
        self.total = total
        self.prefix = prefix
        self.is_tty = sys.stderr.isatty()
        # Redirected to a file: one line every PROGRESS_LOG_INTERVAL instead of
        # thousands of carriage returns.
        self.interval = (
            interval if self.is_tty else max(interval, PROGRESS_LOG_INTERVAL)
        )
        self.started = time.monotonic()
        self.last_drawn = 0.0

    def update(self, done: int, suffix: str = "", *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_drawn < self.interval:
            return
        self.last_drawn = now
        elapsed = now - self.started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        fraction = done / self.total if self.total > 0 else 1.0
        filled = int(PROGRESS_BAR_WIDTH * fraction)
        bar = "=" * filled + " " * (PROGRESS_BAR_WIDTH - filled)
        body = (
            f"{self.prefix} [{bar}] {done}/{self.total} {fraction * 100:5.1f}% "
            f"{rate:7.1f}/s elapsed {format_duration(elapsed)} "
            f"eta {format_duration(eta)}{suffix}"
        )
        if self.is_tty:
            print(f"\r{body}\x1b[K", end="", file=sys.stderr, flush=True)
        else:
            print(body, file=sys.stderr, flush=True)

    def close(self) -> None:
        if self.is_tty:
            print("", file=sys.stderr, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"JSON file must contain an object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.partial")
    staged.write_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    os.replace(staged, path)


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError(f"another process owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def connect_db(
    path: Path, *, write: bool = False, build: bool = False
) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 60000")
    connection.execute("PRAGMA foreign_keys = ON")
    if build:
        # 64 KiB pages instead of the 4 KiB default. Only settable before the
        # first table exists, so it has to happen here. Every later full scan
        # (status, the generate producer, export) is bound by the shared
        # filesystem's small-read IOPS rather than its bandwidth, and larger
        # pages cut the syscall count for those scans by 16x.
        connection.execute("PRAGMA page_size = 65536")
        # Sample build only. The target is a .partial file that is deleted on
        # any failure and atomically renamed on success, so a rollback journal
        # and fsyncs buy nothing. The page cache matters much more: rank is a
        # random permutation of source order, so rows arrive in random rowid
        # order across three B-trees. With the stock 2 MiB cache every insert
        # dirties a different page and writeback thrashes the shared
        # filesystem; holding the whole database in memory instead lets it be
        # written out once at commit.
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute(f"PRAGMA cache_size = {-BUILD_CACHE_KIB}")
        connection.execute("PRAGMA temp_store = MEMORY")
    elif write:
        # Not WAL. WAL keeps its index in an mmapped -shm file, and this state
        # lives on a shared network filesystem where that mapping is neither
        # coherent nor safely growable: one run took SIGBUS when the index had
        # to grow past a 433 MiB WAL, and the next corrupted the database
        # outright (out-of-order rowids, pages referenced twice, scrambled
        # rows). A rollback journal touches no shared memory. It needs the
        # writer to take an exclusive lock, which is only viable because the
        # pending-record reader pages its queries instead of holding one cursor
        # open for the whole run.
        connection.execute("PRAGMA journal_mode = TRUNCATE")
        connection.execute("PRAGMA synchronous = FULL")
    return connection


def initialize_db(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE requests (
            rank INTEGER PRIMARY KEY,
            id TEXT UNIQUE NOT NULL,
            source_index INTEGER UNIQUE NOT NULL,
            record TEXT NOT NULL
        );
        CREATE TABLE generations (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('success', 'error')),
            event TEXT NOT NULL,
            FOREIGN KEY (id) REFERENCES requests(id)
        );
        CREATE INDEX generations_status ON generations(status);
        """)
    set_meta(connection, "schema_version", SCHEMA_VERSION)


_MISSING = object()


def get_meta(
    connection: sqlite3.Connection,
    key: str,
    default: Any = _MISSING,
) -> Any:
    row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        if default is _MISSING:
            raise PipelineError(f"state is missing metadata: {key}")
        return default
    return json.loads(row["value"])


def set_meta(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, json_bytes(value).decode("utf-8")),
    )


def state_path(output_root: str | Path) -> Path:
    return Path(output_root).expanduser().resolve() / DB_NAME


def open_state(output_root: str | Path, *, write: bool = False):
    path = state_path(output_root)
    if not path.is_file():
        raise PipelineError(f"sample state does not exist: {path}")
    connection = connect_db(path, write=write)
    if get_meta(connection, "schema_version") != SCHEMA_VERSION:
        connection.close()
        raise PipelineError(f"unsupported state schema: {path}")
    if get_meta(connection, "sample_complete", False) is not True:
        connection.close()
        raise PipelineError(f"sample state is incomplete: {path}")
    return connection


def bit_is_set(bitset: bytearray, index: int) -> bool:
    return bool(bitset[index >> 3] & (1 << (index & 7)))


def set_bit(bitset: bytearray, index: int) -> None:
    bitset[index >> 3] |= 1 << (index & 7)


def iter_selected_indices(bitset: bytearray, population_size: int) -> Iterator[int]:
    for byte_index, byte_value in enumerate(bitset):
        value = byte_value
        while value:
            lowest = value & -value
            bit_index = lowest.bit_length() - 1
            index = (byte_index << 3) + bit_index
            if index < population_size:
                yield index
            value ^= lowest


def splitmix64_floyd_ranked(
    population_size: int,
    candidate_size: int,
    seed: int,
) -> tuple[bytearray, array]:
    """Return an exact uniform sample and a random rank for each member."""
    if population_size <= 0:
        raise PipelineError("population size must be positive")
    if not 0 < candidate_size <= population_size:
        raise PipelineError("candidate size is outside the population")
    if population_size > UINT32_SENTINEL:
        raise PipelineError("population size exceeds the 32-bit rank format")

    random_source = SplitMix64(seed)
    selected = bytearray((population_size + 7) // 8)
    for candidate in range(population_size - candidate_size, population_size):
        draw = random_source.randbelow(candidate + 1)
        choice = candidate if bit_is_set(selected, draw) else draw
        set_bit(selected, choice)

    ranked = array("I", iter_selected_indices(selected, population_size))
    rank_seed = int.from_bytes(
        hashlib.sha256(f"{SAMPLE_ALGORITHM}\0rank\0{seed}".encode("ascii")).digest()[
            :8
        ],
        "big",
    )
    rank_random = SplitMix64(rank_seed)
    for right in range(candidate_size - 1, 0, -1):
        left = rank_random.randbelow(right + 1)
        ranked[right], ranked[left] = ranked[left], ranked[right]

    ranks = array("I", [UINT32_SENTINEL]) * population_size
    for rank, source_index in enumerate(ranked):
        ranks[source_index] = rank
    return selected, ranks


def rank_limit_for_count(
    valid_ranks: bytearray,
    candidate_size: int,
    target: int,
) -> int:
    count = 0
    for rank in range(candidate_size):
        if bit_is_set(valid_ranks, rank):
            count += 1
            if count == target:
                return rank + 1
    raise PipelineError(f"only {count} valid candidates remain for {target}")


def build_stages(
    *,
    valid_ranks: bytearray,
    candidate_size: int,
    valid_count: int,
    sample_size: int,
    pilot_size: int | None,
    reserve_size: int,
) -> dict[str, dict[str, int]]:
    stages: dict[str, dict[str, int]] = {}

    def add(name: str, target: int, reserve: int) -> None:
        pool = min(valid_count, target + reserve)
        if pool < target:
            raise PipelineError(
                f"stage {name} needs {target} rows, only {valid_count} remain"
            )
        stages[name] = {
            "target_records": target,
            "pool_records": pool,
            "candidate_rank_exclusive": rank_limit_for_count(
                valid_ranks, candidate_size, pool
            ),
        }

    add("smoke", min(SMOKE_SIZE, sample_size), 0)
    if pilot_size is not None and pilot_size < sample_size:
        add("pilot", pilot_size, reserve_size)
    add("full", sample_size, reserve_size)
    return stages


def parse_prefix_maps(values: Sequence[str] | None) -> list[PrefixMap]:
    mappings: list[PrefixMap] = []
    for value in values or []:
        source, separator, destination = value.partition("=")
        if not separator or not source or not destination:
            raise PipelineError(
                f"invalid --path-map {value!r}; expected ABS_FROM=ABS_TO"
            )
        if not Path(source).is_absolute() or not Path(destination).is_absolute():
            raise PipelineError(f"path-map values must be absolute: {value}")
        mappings.append(
            PrefixMap(
                source.rstrip(os.sep) + os.sep,
                destination.rstrip(os.sep) + os.sep,
            )
        )
    return sorted(mappings, key=lambda mapping: -len(mapping.source))


@functools.cache
def resolved_directory(directory: str) -> str:
    return os.path.realpath(directory)


def resolve_media_file(value: str) -> str:
    """realpath() of a media file, reusing a memoised parent directory.

    A plain realpath() re-walks and re-stats every path component. These
    datasets hold ~10^6 files under ~10^3 directories, so caching the parent
    turns each file into a single lstat, which is what lets threads scale here.
    Symlinked files fall back to the full walk.
    """
    directory, separator, name = value.rpartition(os.sep)
    if not separator:
        return os.path.realpath(value)
    info = os.lstat(value)
    if stat.S_ISLNK(info.st_mode):
        return os.path.realpath(value)
    if not stat.S_ISREG(info.st_mode):
        raise PipelineError(f"media path is not a regular file: {value}")
    return f"{resolved_directory(directory)}{os.sep}{name}"


def map_media_path(
    value: str,
    allowed_root: Path,
    mappings: Sequence[PrefixMap],
) -> Path:
    if not Path(value).is_absolute():
        raise PipelineError(f"media path is not absolute: {value!r}")

    candidates = [value]
    for mapping in mappings:
        if value.startswith(mapping.source):
            candidates.append(mapping.destination + value[len(mapping.source) :])
            break

    failure = f"media path has no matching path-map: {value}"
    for candidate in candidates:
        try:
            resolved = resolve_media_file(candidate)
        except OSError:
            failure = f"media file does not exist: {candidate}"
            continue
        if not Path(resolved).is_relative_to(allowed_root):
            failure = f"media file is outside {allowed_root}: {resolved}"
            continue
        # Keep the path as the source spells it so the export stays diffable
        # against the source dataset. Only the containment check needs the
        # resolved form, and vLLM resolves again before its own
        # --allowed-local-media-path check.
        return Path(candidate)
    raise PipelineError(failure)


def build_user_content(
    prompt: str, image_paths: Sequence[Path]
) -> list[dict[str, str]]:
    placeholder_count = prompt.count("<image>")
    if placeholder_count == 0:
        return [
            *({"type": "image", "path": str(path)} for path in image_paths),
            {"type": "text", "text": prompt},
        ]
    if placeholder_count != len(image_paths):
        raise PipelineError(
            f"{placeholder_count} image placeholders for {len(image_paths)} images"
        )

    fragments = prompt.split("<image>")
    content: list[dict[str, str]] = []
    for index, image_path in enumerate(image_paths):
        if fragments[index]:
            content.append({"type": "text", "text": fragments[index]})
        content.append({"type": "image", "path": str(image_path)})
    if fragments[-1]:
        content.append({"type": "text", "text": fragments[-1]})
    return content


def build_conversation_skeleton(
    conversations: Sequence[Mapping[str, Any]],
    image_paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not conversations or len(conversations) % 2:
        raise PipelineError("conversations must contain human/gpt pairs")

    prompts: list[str] = []
    old_answers: list[str] = []
    for index, turn in enumerate(conversations):
        expected = "human" if index % 2 == 0 else "gpt"
        if turn.get("from") != expected or not isinstance(turn.get("value"), str):
            raise PipelineError("conversation roles must alternate human/gpt")
        if expected == "human":
            prompts.append(turn["value"])
        else:
            old_answers.append(turn["value"])

    total_placeholders = sum(prompt.count("<image>") for prompt in prompts)
    if total_placeholders not in (0, len(image_paths)):
        raise PipelineError("image placeholders do not match the image list")

    skeleton: list[dict[str, Any]] = []
    image_offset = 0
    for prompt_index, prompt in enumerate(prompts):
        if total_placeholders:
            count = prompt.count("<image>")
            assigned = image_paths[image_offset : image_offset + count]
            image_offset += count
        else:
            assigned = image_paths if prompt_index == 0 else []
        skeleton.append(
            {"role": "user", "content": build_user_content(prompt, assigned)}
        )
        skeleton.append({"role": "assistant", "content": None})
    return skeleton, old_answers


def stable_record_id(source_version: str, source_index: int) -> str:
    return f"{source_version}:{source_index:010d}"


def convert_source_record(
    row: Any,
    *,
    source_index: int,
    candidate_rank: int,
    source_version: str,
    allowed_root: Path,
    mappings: Sequence[PrefixMap],
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise PipelineError("source row must be an object")
    images = row.get("images")
    conversations = row.get("conversations")
    attributes = row.get("attributes")
    if not isinstance(images, list) or not images:
        raise PipelineError("images must be a non-empty list")
    if not all(isinstance(image, str) and image for image in images):
        raise PipelineError("images must contain non-empty strings")
    if not isinstance(conversations, list):
        raise PipelineError("conversations must be a list")
    if not isinstance(attributes, dict):
        raise PipelineError("attributes must be an object")

    mapped = [map_media_path(image, allowed_root, mappings) for image in images]
    skeleton, old_answers = build_conversation_skeleton(conversations, mapped)
    return {
        "id": stable_record_id(source_version, source_index),
        "conversations": skeleton,
        "attributes": attributes,
        "provenance": {
            "source_version": source_version,
            "source_line_index": source_index,
            "candidate_rank": candidate_rank,
            "assistant_turns": len(old_answers),
            "original_assistant_sha256s": [
                sha256_bytes(answer.encode("utf-8")) for answer in old_answers
            ],
            "media": [str(path) for path in mapped],
        },
    }


def remove_sqlite_files(path: Path) -> None:
    for candidate in (
        path,
        Path(f"{path}-journal"),
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
    ):
        candidate.unlink(missing_ok=True)


def sample_config(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).expanduser().resolve(strict=True)
    allowed_root = Path(args.allowed_media_root).expanduser().resolve(strict=True)
    source_stat = source.stat()
    return {
        "algorithm": SAMPLE_ALGORITHM,
        "source": str(source),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "population_size": args.population_size,
        "sample_size": args.sample_size,
        "pilot_size": args.pilot_size,
        "reserve_size": args.reserve_size,
        "seed": args.seed,
        "source_version": args.source_version,
        "allowed_media_root": str(allowed_root),
        "path_maps": list(args.path_map or []),
    }


def validate_sample_args(args: argparse.Namespace) -> None:
    if args.population_size <= 0:
        raise PipelineError("--population-size must be positive")
    if not 0 < args.sample_size <= args.population_size:
        raise PipelineError("--sample-size is outside the population")
    if args.reserve_size < 0:
        raise PipelineError("--reserve-size must be non-negative")
    if args.sample_size + args.reserve_size > args.population_size:
        raise PipelineError("sample size plus reserve exceeds the population")
    if args.pilot_size is not None and not 0 < args.pilot_size < args.sample_size:
        raise PipelineError("--pilot-size must be smaller than --sample-size")
    if args.convert_workers <= 0:
        raise PipelineError("--convert-workers must be positive")
    if args.convert_chunk <= 0:
        raise PipelineError("--convert-chunk must be positive")


def convert_one(
    task: tuple[int, bytes],
    *,
    candidate_ranks: array,
    source_version: str,
    allowed_root: Path,
    mappings: Sequence[PrefixMap],
) -> tuple[int, int, dict[str, Any] | None, str | None]:
    """Convert one selected source line; safe to run on a worker thread.

    Most of the wall clock here is media stat/readlink syscalls against the
    shared filesystem, which release the GIL, so threads scale nearly linearly.
    """
    source_index, raw_line = task
    candidate_rank = int(candidate_ranks[source_index])
    try:
        record = convert_source_record(
            json.loads(raw_line),
            source_index=source_index,
            candidate_rank=candidate_rank,
            source_version=source_version,
            allowed_root=allowed_root,
            mappings=mappings,
        )
    except (json.JSONDecodeError, OSError, PipelineError) as exc:
        return source_index, candidate_rank, None, str(exc)
    return source_index, candidate_rank, record, None


def command_sample(args: argparse.Namespace) -> int:  # noqa: C901
    validate_sample_args(args)
    config = sample_config(args)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_path = output_root / DB_NAME
    partial_path = output_root / f".{DB_NAME}.partial"

    with file_lock(output_root / ".pipeline.lock"):
        if final_path.is_file():
            existing = open_state(output_root)
            existing_config = get_meta(existing, "sample_config")
            existing.close()
            if existing_config == config:
                print(f"sample already complete: {final_path}")
                return 0
            if not args.overwrite:
                raise PipelineError(
                    "sample configuration changed; rerun with --overwrite"
                )

        remove_sqlite_files(partial_path)
        connection = connect_db(partial_path, build=True)
        initialize_db(connection)
        set_meta(connection, "sample_config", config)
        connection.commit()

        candidate_size = args.sample_size + args.reserve_size
        selected, candidate_ranks = splitmix64_floyd_ranked(
            args.population_size,
            candidate_size,
            args.seed,
        )
        valid_ranks = bytearray((candidate_size + 7) // 8)
        source_digest = hashlib.sha256()
        request_digest = hashlib.sha256()
        mappings = parse_prefix_maps(args.path_map)
        allowed_root = Path(config["allowed_media_root"])
        selected_seen = 0
        valid_count = 0
        source_lines = 0
        invalid_examples: list[dict[str, Any]] = []
        insert_batch: list[tuple[int, str, int, str]] = []
        progress = ProgressBar(candidate_size, "sample  ")

        convert = functools.partial(
            convert_one,
            candidate_ranks=candidate_ranks,
            source_version=args.source_version,
            allowed_root=allowed_root,
            mappings=mappings,
        )
        pending: list[tuple[int, bytes]] = []

        def flush_pending(executor: ThreadPoolExecutor) -> None:
            """Convert one chunk in parallel, then book-keep in source order."""
            nonlocal valid_count
            if not pending:
                return
            # executor.map preserves input order, so request_digest stays
            # byte-identical to the sequential implementation.
            for source_index, candidate_rank, record, error in executor.map(
                convert, pending
            ):
                if record is None:
                    if len(invalid_examples) < INVALID_EXAMPLE_LIMIT:
                        invalid_examples.append(
                            {
                                "source_line_index": source_index,
                                "candidate_rank": candidate_rank,
                                "error": error,
                            }
                        )
                    continue
                record_data = json_bytes(record)
                request_digest.update(record_data + b"\n")
                insert_batch.append(
                    (
                        candidate_rank,
                        record["id"],
                        source_index,
                        record_data.decode("utf-8"),
                    )
                )
                set_bit(valid_ranks, candidate_rank)
                valid_count += 1
            pending.clear()
            if len(insert_batch) >= REQUEST_INSERT_BATCH_SIZE:
                # No commit here: the build is one transaction, made atomic by
                # the .partial rename rather than by sqlite.
                connection.executemany(
                    "INSERT INTO requests(rank, id, source_index, record) "
                    "VALUES (?, ?, ?, ?)",
                    insert_batch,
                )
                insert_batch.clear()
            progress.update(selected_seen, f" valid {valid_count}")

        try:
            with (
                ThreadPoolExecutor(max_workers=args.convert_workers) as executor,
                Path(config["source"]).open("rb") as source_file,
            ):
                for source_index, raw_line in enumerate(source_file):
                    source_lines += 1
                    source_digest.update(raw_line)
                    if source_index >= args.population_size or not bit_is_set(
                        selected, source_index
                    ):
                        continue

                    selected_seen += 1
                    pending.append((source_index, raw_line))
                    if len(pending) >= args.convert_chunk:
                        flush_pending(executor)
                flush_pending(executor)

            progress.update(selected_seen, f" valid {valid_count}", force=True)
            progress.close()
            if insert_batch:
                connection.executemany(
                    "INSERT INTO requests(rank, id, source_index, record) "
                    "VALUES (?, ?, ?, ?)",
                    insert_batch,
                )
            if source_lines != args.population_size:
                raise PipelineError(
                    f"source has {source_lines} lines, expected {args.population_size}"
                )
            if selected_seen != candidate_size:
                raise PipelineError(
                    f"saw {selected_seen} selected rows, expected {candidate_size}"
                )
            if valid_count < args.sample_size:
                raise PipelineError(
                    f"only {valid_count} valid rows remain; increase reserve size"
                )

            stages = build_stages(
                valid_ranks=valid_ranks,
                candidate_size=candidate_size,
                valid_count=valid_count,
                sample_size=args.sample_size,
                pilot_size=args.pilot_size,
                reserve_size=args.reserve_size,
            )
            summary = {
                "source_sha256": source_digest.hexdigest(),
                "requests_sha256": request_digest.hexdigest(),
                "candidate_records": candidate_size,
                "valid_records": valid_count,
                "invalid_records": candidate_size - valid_count,
                "invalid_examples": invalid_examples,
                "stages": stages,
            }
            sample_hash = sha256_json({"config": config, "summary": summary})
            set_meta(connection, "sample_summary", summary)
            set_meta(connection, "sample_sha256", sample_hash)
            set_meta(connection, "stages", stages)
            set_meta(connection, "sample_complete", True)
            connection.commit()
            connection.close()

            for suffix in ("-wal", "-shm"):
                Path(f"{final_path}{suffix}").unlink(missing_ok=True)
            os.replace(partial_path, final_path)
        except Exception:
            connection.close()
            remove_sqlite_files(partial_path)
            raise

    print(
        f"sample complete: {valid_count} valid rows, "
        f"{candidate_size - valid_count} invalid, state={final_path}"
    )
    return 0


def stage_config(connection: sqlite3.Connection, stage_name: str) -> dict[str, int]:
    stages = get_meta(connection, "stages")
    stage = stages.get(stage_name) if isinstance(stages, dict) else None
    if not isinstance(stage, dict):
        available = ", ".join(sorted(stages)) if isinstance(stages, dict) else ""
        raise PipelineError(
            f"unknown stage {stage_name!r}; available stages: {available}"
        )
    return {key: int(value) for key, value in stage.items()}


def fingerprint_model(model_path: Path) -> dict[str, Any]:
    files = []
    for name in (
        "config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        path = model_path / name
        if path.is_file():
            files.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not files:
        raise PipelineError(f"model has no semantic config files: {model_path}")
    return {
        "model_path": str(model_path),
        "files": files,
        "sha256": sha256_json(files),
    }


def generation_config(
    connection: sqlite3.Connection, args: argparse.Namespace
) -> dict[str, Any]:
    model_path = Path(args.model_path).expanduser().resolve(strict=True)
    return {
        "sample_sha256": get_meta(connection, "sample_sha256"),
        "model": args.model,
        "model_fingerprint": fingerprint_model(model_path),
        "max_tokens": args.max_tokens if args.max_tokens > 0 else None,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def ensure_generation_config(
    connection: sqlite3.Connection, args: argparse.Namespace
) -> tuple[dict[str, Any], str]:
    config = generation_config(connection, args)
    digest = sha256_json(config)
    existing = get_meta(connection, "generation_config", None)
    existing_digest = get_meta(connection, "generation_sha256", None)
    changed = existing is not None and existing != config
    if changed and not args.allow_config_change:
        raise PipelineError("generation configuration differs from the existing state")
    if changed:
        # Rows already in `generations` were produced under the superseded
        # config. Keep the history so the provenance of a mixed run is
        # auditable instead of silently overwritten.
        history = get_meta(connection, "superseded_generation_configs", [])
        history.append({"config": existing, "sha256": existing_digest})
        set_meta(connection, "superseded_generation_configs", history)
    elif existing_digest is not None and existing_digest != digest:
        raise PipelineError("generation configuration hash is inconsistent")
    if existing is None or changed:
        set_meta(connection, "generation_config", config)
        set_meta(connection, "generation_sha256", digest)
        connection.commit()
    return config, digest


def normalize_endpoint(value: str) -> str:
    value = value.rstrip("/")
    path = urlsplit(value).path
    if path.endswith("/chat/completions"):
        return value
    if path.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def models_endpoint(chat_endpoint: str) -> str:
    parsed = urlsplit(chat_endpoint)
    prefix = parsed.path.rsplit("/chat/completions", 1)[0]
    return urlunsplit(parsed._replace(path=f"{prefix}/models", query=""))


def authorization_headers(api_key_env: str) -> dict[str, str]:
    api_key = os.environ.get(api_key_env, "")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


async def verify_endpoints(
    session: aiohttp.ClientSession,
    endpoints: Sequence[str],
    model: str,
) -> None:
    async def verify(endpoint: str) -> None:
        try:
            async with session.get(models_endpoint(endpoint)) as response:
                body = await response.text()
                if response.status != HTTP_OK:
                    raise PipelineError(
                        f"{endpoint} /models returned HTTP {response.status}: "
                        f"{body[:500]}"
                    )
                payload = json.loads(body)
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            raise PipelineError(f"cannot reach teacher {endpoint}: {exc}") from exc
        if not isinstance(payload, dict):
            raise PipelineError(f"teacher {endpoint} returned invalid model data")
        models = {
            item.get("id")
            for item in payload.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if model not in models:
            raise PipelineError(
                f"teacher {endpoint} serves {sorted(models)}, expected {model!r}"
            )

    await asyncio.gather(*(verify(endpoint) for endpoint in endpoints))


def derive_request_seed(
    base_seed: int, source_index: int, assistant_ordinal: int
) -> int:
    material = f"{base_seed}\0{source_index}\0{assistant_ordinal}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def typed_content_to_openai(
    parts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in parts:
        if part.get("type") == "image":
            media_path = Path(str(part.get("path"))).resolve(strict=True)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": media_path.as_uri()},
                }
            )
        elif part.get("type") == "text":
            content.append({"type": "text", "text": str(part.get("text", ""))})
        else:
            raise PipelineError(f"unsupported content part: {part}")
    return content


async def post_chat(  # noqa: C901
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: Mapping[str, Any],
    *,
    max_retries: int,
) -> tuple[dict[str, Any], int]:
    for attempt in range(1, max_retries + 2):
        try:
            async with session.post(endpoint, json=payload) as response:
                body = await response.text()
                if HTTP_OK <= response.status < HTTP_REDIRECT:
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise RequestError(
                            "malformed_response",
                            "teacher returned invalid JSON",
                            attempts=attempt,
                            retryable=False,
                        ) from exc
                    if not isinstance(data, dict):
                        raise RequestError(
                            "malformed_response",
                            "teacher response is not an object",
                            attempts=attempt,
                            retryable=False,
                        )
                    return data, attempt
                retryable = (
                    response.status in RETRYABLE_STATUSES
                    or response.status >= HTTP_SERVER_ERROR
                )
                failure = RequestError(
                    "http_error",
                    f"HTTP {response.status}: {body[:1000]}",
                    attempts=attempt,
                    status_code=response.status,
                    retryable=retryable,
                )
                if not retryable:
                    raise failure
        except RequestError as exc:
            failure = exc
            if not exc.retryable:
                raise
        except asyncio.TimeoutError as exc:
            failure = RequestError(
                "timeout", str(exc), attempts=attempt, retryable=True
            )
        except aiohttp.ClientError as exc:
            failure = RequestError(
                "network_error",
                str(exc),
                attempts=attempt,
                retryable=True,
            )

        if attempt > max_retries:
            raise failure
        await asyncio.sleep(min(30.0, 2 ** (attempt - 1)))
    raise AssertionError("retry loop did not terminate")


def parse_response(data: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RequestError("malformed_response", "response has no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RequestError("malformed_response", "choice is not an object")
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise RequestError(
            "finish_reason",
            f"expected finish_reason='stop', got {finish_reason!r}",
        )
    message = choice.get("message")
    answer = message.get("content") if isinstance(message, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise RequestError("empty_response", "assistant response is empty")
    if UNRESOLVED_PLACEHOLDER.search(answer):
        raise RequestError(
            "unresolved_placeholder",
            "assistant response contains an unresolved placeholder",
        )
    usage = data.get("usage")
    return answer, finish_reason, usage if isinstance(usage, dict) else {}


async def generate_event(
    session: aiohttp.ClientSession,
    endpoint: str,
    record: Mapping[str, Any],
    *,
    generation_sha256: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_index = record["provenance"]["source_line_index"]
    started = time.perf_counter()
    messages: list[dict[str, Any]] = []
    generated_turns: list[dict[str, Any]] = []
    total_attempts = 0
    failed_ordinal = 0
    try:
        for turn in record["conversations"]:
            if turn["role"] == "user":
                messages.append(
                    {
                        "role": "user",
                        "content": typed_content_to_openai(turn["content"]),
                    }
                )
                continue

            failed_ordinal = len(generated_turns)
            request_seed = derive_request_seed(args.seed, source_index, failed_ordinal)
            payload = {
                "model": args.model,
                "messages": list(messages),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": request_seed,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            # Omitting max_tokens makes vLLM fall back to
            # max_model_len - prompt_tokens, i.e. the model's own ceiling.
            if args.max_tokens > 0:
                payload["max_tokens"] = args.max_tokens
            turn_started = time.perf_counter()
            data, attempts = await post_chat(
                session,
                endpoint,
                payload,
                max_retries=args.max_retries,
            )
            try:
                answer, finish_reason, usage = parse_response(data)
            except RequestError as exc:
                exc.attempts = attempts
                raise
            total_attempts += attempts
            generated_turns.append(
                {
                    "assistant_ordinal": failed_ordinal,
                    "request_seed": request_seed,
                    "attempts": attempts,
                    "latency_seconds": round(time.perf_counter() - turn_started, 6),
                    "finish_reason": finish_reason,
                    "answer": answer,
                    "answer_sha256": sha256_bytes(answer.encode("utf-8")),
                    "usage": usage,
                    "response_id": data.get("id"),
                }
            )
            messages.append({"role": "assistant", "content": answer})
        return {
            "event": "success",
            "id": record["id"],
            "source_line_index": source_index,
            "candidate_rank": record["provenance"]["candidate_rank"],
            "generation_config_sha256": generation_sha256,
            "endpoint": endpoint,
            "attempts": total_attempts,
            "latency_seconds": round(time.perf_counter() - started, 6),
            "turns": generated_turns,
            "created_at": utc_now(),
        }
    except RequestError as exc:
        return {
            "event": "error",
            "id": record["id"],
            "source_line_index": source_index,
            "candidate_rank": record["provenance"]["candidate_rank"],
            "generation_config_sha256": generation_sha256,
            "endpoint": endpoint,
            "attempts": total_attempts + exc.attempts,
            "latency_seconds": round(time.perf_counter() - started, 6),
            "completed_assistant_turns": len(generated_turns),
            "failed_assistant_ordinal": failed_ordinal,
            "error": {
                "kind": exc.kind,
                "message": str(exc),
                "status_code": exc.status_code,
            },
            "created_at": utc_now(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "event": "error",
            "id": record["id"],
            "source_line_index": source_index,
            "candidate_rank": record["provenance"]["candidate_rank"],
            "generation_config_sha256": generation_sha256,
            "endpoint": endpoint,
            "attempts": total_attempts,
            "latency_seconds": round(time.perf_counter() - started, 6),
            "completed_assistant_turns": len(generated_turns),
            "failed_assistant_ordinal": failed_ordinal,
            "error": {
                "kind": type(exc).__name__,
                "message": str(exc),
                "status_code": None,
            },
            "created_at": utc_now(),
        }


def generation_counts(
    connection: sqlite3.Connection, rank_limit: int
) -> dict[str, int]:
    counts = {"success": 0, "error": 0, "pending": 0}
    rows = connection.execute(
        "SELECT COALESCE(g.status, 'pending') AS status, COUNT(*) AS count "
        "FROM requests r LEFT JOIN generations g ON g.id = r.id "
        "WHERE r.rank < ? GROUP BY COALESCE(g.status, 'pending')",
        (rank_limit,),
    )
    for row in rows:
        counts[row["status"]] = int(row["count"])
    return counts


def pending_count(
    connection: sqlite3.Connection,
    rank_limit: int,
    *,
    retry_errors: bool,
) -> int:
    query = _RETRYABLE_PENDING_COUNT_QUERY if retry_errors else _PENDING_COUNT_QUERY
    row = connection.execute(query, (rank_limit,)).fetchone()
    return int(row["count"])


async def generate_async(  # noqa: C901
    *,
    db_path: Path,
    rank_limit: int,
    pending: int,
    endpoints: Sequence[str],
    generation_sha256: str,
    args: argparse.Namespace,
) -> None:
    worker_count = len(endpoints) * args.concurrency_per_endpoint
    request_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
        maxsize=worker_count * 2
    )
    result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
        maxsize=worker_count * 2
    )
    timeout = aiohttp.ClientTimeout(
        total=args.timeout,
        connect=args.connect_timeout,
    )
    connector = aiohttp.TCPConnector(limit=worker_count)

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers=authorization_headers(args.api_key_env),
    ) as session:
        await verify_endpoints(session, endpoints, args.model)

        async def producer() -> None:
            connection = connect_db(db_path)
            query = _RETRYABLE_PENDING_QUERY if args.retry_errors else _PENDING_QUERY
            produced = 0
            last_rank = -1
            try:
                while True:
                    # fetchall() completes the statement, which ends the implicit
                    # read transaction and lets the WAL checkpoint between pages.
                    rows = connection.execute(
                        query, (last_rank, rank_limit, PENDING_PAGE_SIZE)
                    ).fetchall()
                    if not rows:
                        break
                    for row in rows:
                        await request_queue.put(json.loads(row["record"]))
                        produced += 1
                        last_rank = int(row["rank"])
            finally:
                connection.close()
            if produced != pending:
                raise PipelineError(
                    f"pending query changed: expected {pending}, got {produced}"
                )
            for _ in range(worker_count):
                await request_queue.put(None)

        async def worker(endpoint: str) -> None:
            while True:
                record = await request_queue.get()
                if record is None:
                    request_queue.task_done()
                    return
                event = await generate_event(
                    session,
                    endpoint,
                    record,
                    generation_sha256=generation_sha256,
                    args=args,
                )
                await result_queue.put(event)
                request_queue.task_done()

        async def writer() -> None:
            connection = connect_db(db_path, write=True)
            batch: list[tuple[str, str, str]] = []
            progress = ProgressBar(pending, "generate")
            succeeded = 0
            failed = 0
            try:
                for completed in range(1, pending + 1):
                    event = await result_queue.get()
                    if event["event"] == "success":
                        succeeded += 1
                    else:
                        failed += 1
                    batch.append(
                        (
                            event["id"],
                            event["event"],
                            json_bytes(event).decode("utf-8"),
                        )
                    )
                    result_queue.task_done()
                    if len(batch) >= args.commit_batch or completed == pending:
                        connection.executemany(
                            "INSERT INTO generations(id, status, event) "
                            "VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                            "status = excluded.status, event = excluded.event",
                            batch,
                        )
                        connection.commit()
                        batch.clear()
                    progress.update(
                        completed,
                        f" ok {succeeded} err {failed}",
                        force=completed == pending,
                    )
            finally:
                progress.close()
                connection.close()

        tasks = [asyncio.create_task(producer())]
        for endpoint in endpoints:
            tasks.extend(
                asyncio.create_task(worker(endpoint))
                for _ in range(args.concurrency_per_endpoint)
            )
        tasks.append(asyncio.create_task(writer()))
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def command_generate(args: argparse.Namespace) -> int:
    if args.concurrency_per_endpoint <= 0 or args.commit_batch <= 0:
        raise PipelineError("concurrency and commit batch must be positive")
    if args.max_retries < 0:
        raise PipelineError("--max-retries must be non-negative")
    db_path = state_path(args.output_root)
    with file_lock(db_path.parent / ".pipeline.lock"):
        connection = open_state(args.output_root, write=True)
        stage = stage_config(connection, args.stage)
        _, config_hash = ensure_generation_config(connection, args)
        pending = pending_count(
            connection,
            stage["candidate_rank_exclusive"],
            retry_errors=args.retry_errors,
        )
        connection.close()

        if pending:
            endpoints = [normalize_endpoint(value) for value in args.endpoint]
            if not endpoints:
                raise PipelineError("at least one --endpoint is required")
            asyncio.run(
                generate_async(
                    db_path=db_path,
                    rank_limit=stage["candidate_rank_exclusive"],
                    pending=pending,
                    endpoints=endpoints,
                    generation_sha256=config_hash,
                    args=args,
                )
            )

        connection = open_state(args.output_root, write=True)
        counts = generation_counts(connection, stage["candidate_rank_exclusive"])
        complete = counts["pending"] == 0
        if complete:
            set_meta(
                connection,
                f"complete:{args.stage}",
                {
                    "complete": True,
                    "stage": args.stage,
                    "generation_sha256": config_hash,
                    "counts": counts,
                    "completed_at": utc_now(),
                },
            )
            connection.commit()
        connection.close()

    print(
        f"{args.stage}: {counts['success']} successes, "
        f"{counts['error']} errors, {counts['pending']} pending"
    )
    return 0 if complete else 1


def status_payload(connection: sqlite3.Connection, stage_name: str) -> dict[str, Any]:
    stage = stage_config(connection, stage_name)
    counts = generation_counts(connection, stage["candidate_rank_exclusive"])
    error_kinds: Counter[str] = Counter()
    if counts["error"]:
        rows = connection.execute(
            "SELECT g.event FROM requests r JOIN generations g ON g.id = r.id "
            "WHERE r.rank < ? AND g.status = 'error'",
            (stage["candidate_rank_exclusive"],),
        )
        for row in rows:
            event = json.loads(row["event"])
            error = event.get("error")
            kind = error.get("kind") if isinstance(error, dict) else "unknown"
            error_kinds[str(kind)] += 1
    return {
        "stage": stage_name,
        **stage,
        **counts,
        "complete": counts["pending"] == 0,
        "sample_sha256": get_meta(connection, "sample_sha256"),
        "generation_sha256": get_meta(connection, "generation_sha256", None),
        "error_kinds": dict(sorted(error_kinds.items())),
    }


def command_status(args: argparse.Namespace) -> int:
    connection = open_state(args.output_root)
    payload = status_payload(connection, args.stage)
    connection.close()
    if args.json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(
            f"{args.stage}: {payload['success']} successes, "
            f"{payload['error']} errors, {payload['pending']} pending; "
            f"target={payload['target_records']}"
        )
        if payload["error_kinds"]:
            print(json.dumps(payload["error_kinds"], sort_keys=True))
    if args.require_complete and not payload["complete"]:
        return 1
    return 0


def load_selection(  # noqa: C901
    connection: sqlite3.Connection,
    manifest_path: Path,
    *,
    sample_sha256: str,
    generation_sha256: str,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("complete") is not True:
        raise PipelineError(f"selection manifest is incomplete: {manifest_path}")
    if manifest.get("sample_manifest_sha256") not in (None, sample_sha256):
        raise PipelineError("selection uses a different sample")
    if manifest.get("generation_config_sha256") not in (
        None,
        generation_sha256,
    ):
        raise PipelineError("selection uses a different generation config")
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or not isinstance(selection.get("path"), str):
        raise PipelineError("selection manifest has no selection path")
    path = (manifest_path.parent / selection["path"]).resolve(strict=True)

    connection.execute("DROP TABLE IF EXISTS temp.selected")
    connection.execute(
        "CREATE TEMP TABLE selected (rank INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL)"
    )
    batch: list[tuple[int, str]] = []
    count = 0
    try:
        with path.open("rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                rank = row.get("candidate_rank")
                record_id = row.get("id")
                if type(rank) is not int or not isinstance(record_id, str):
                    raise PipelineError(f"invalid selection row {line_number}: {path}")
                batch.append((rank, record_id))
                count += 1
                if len(batch) >= SELECTION_INSERT_BATCH_SIZE:
                    connection.executemany(
                        "INSERT INTO selected(rank, id) VALUES (?, ?)", batch
                    )
                    batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO selected(rank, id) VALUES (?, ?)", batch
            )
    except (OSError, json.JSONDecodeError, sqlite3.IntegrityError) as exc:
        raise PipelineError(f"invalid selection data {path}: {exc}") from exc

    expected = manifest.get("target_records", selection.get("records"))
    if type(expected) is not int or count != expected:
        raise PipelineError(f"selection contains {count} rows, expected {expected}")
    return {
        "path": str(path),
        "records": count,
        "manifest_sha256": sha256_file(manifest_path),
        "selection_sha256": sha256_file(path),
    }


def build_final_record(
    request: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    sample_sha256: str,
    generation_sha256: str,
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    turns = event.get("turns")
    assistant_count = request["provenance"]["assistant_turns"]
    if not isinstance(turns, list) or len(turns) != assistant_count:
        raise PipelineError(f"generated turn count differs for {request['id']}")

    conversations: list[dict[str, Any]] = []
    assistant_ordinal = 0
    for turn in request["conversations"]:
        if turn["role"] == "user":
            conversations.append(turn)
            continue
        generated = turns[assistant_ordinal]
        answer = generated.get("answer")
        if not isinstance(answer, str) or UNRESOLVED_PLACEHOLDER.search(answer):
            raise PipelineError(f"invalid generated answer for {request['id']}")
        conversations.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        )
        assistant_ordinal += 1

    provenance = dict(request["provenance"])
    provenance.update(
        {
            "sample_manifest_sha256": sample_sha256,
            "generation_config_sha256": generation_sha256,
            "generator_model_sha256": generation["model_fingerprint"]["sha256"],
            "generation_endpoint": event["endpoint"],
            "generation_attempts": event["attempts"],
            "generation_latency_seconds": event["latency_seconds"],
            "generated_assistant_turns": [
                {key: value for key, value in turn.items() if key != "answer"}
                for turn in turns
            ],
        }
    )
    return {
        "id": request["id"],
        "conversations": conversations,
        "attributes": request["attributes"],
        "provenance": provenance,
    }


def existing_export_matches(
    output: Path,
    report_path: Path | None,
    signature: Mapping[str, Any],
) -> bool:
    if report_path is None or not output.is_file() or not report_path.is_file():
        return False
    try:
        report = read_json(report_path)
        return all(report.get(key) == value for key, value in signature.items()) and (
            report.get("bytes") == output.stat().st_size
        )
    except (OSError, PipelineError):
        return False


def command_export(args: argparse.Namespace) -> int:  # noqa: C901
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report).expanduser().resolve() if args.report else None
    connection = open_state(args.output_root)
    stage = stage_config(connection, args.stage)
    sample_sha256 = get_meta(connection, "sample_sha256")
    generation = get_meta(connection, "generation_config", None)
    generation_sha256 = get_meta(connection, "generation_sha256", None)
    if generation is None or generation_sha256 is None:
        connection.close()
        raise PipelineError("no generation configuration exists")

    selection = None
    if args.selection_manifest:
        selection = load_selection(
            connection,
            Path(args.selection_manifest).expanduser().resolve(strict=True),
            sample_sha256=sample_sha256,
            generation_sha256=generation_sha256,
        )

    signature = {
        "stage": args.stage,
        "sample_sha256": sample_sha256,
        "generation_sha256": generation_sha256,
        "selection_manifest_sha256": (
            selection["manifest_sha256"] if selection else None
        ),
    }
    if existing_export_matches(output, report_path, signature):
        connection.close()
        print(f"export already complete: {output}")
        return 0

    if selection:
        query = (
            "SELECT r.record, g.event FROM selected s "
            "JOIN requests r ON r.rank = s.rank AND r.id = s.id "
            "JOIN generations g ON g.id = r.id AND g.status = 'success' "
            "ORDER BY r.rank"
        )
        parameters: tuple[Any, ...] = ()
        expected_count = selection["records"]
    else:
        query = (
            "SELECT r.record, g.event FROM requests r "
            "JOIN generations g ON g.id = r.id AND g.status = 'success' "
            "WHERE r.rank < ? ORDER BY r.rank"
        )
        parameters = (stage["candidate_rank_exclusive"],)
        expected_count = None

    staged = output.with_name(f".{output.name}.{os.getpid()}.partial")
    staged.unlink(missing_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    count = 0
    previous_rank = -1
    try:
        with staged.open("wb") as handle:
            for row in connection.execute(query, parameters):
                request = json.loads(row["record"])
                event = json.loads(row["event"])
                rank = request["provenance"]["candidate_rank"]
                if rank <= previous_rank:
                    raise PipelineError("export order is not strictly ranked")
                previous_rank = rank
                final_record = build_final_record(
                    request,
                    event,
                    sample_sha256=sample_sha256,
                    generation_sha256=generation_sha256,
                    generation=generation,
                )
                data = json_bytes(final_record) + b"\n"
                handle.write(data)
                digest.update(data)
                byte_count += len(data)
                count += 1
                if count % 10000 == 0:
                    print(
                        f"export: {count} rows",
                        file=sys.stderr,
                        flush=True,
                    )
        if expected_count is not None and count != expected_count:
            raise PipelineError(
                f"selected export has {count} rows, expected {expected_count}"
            )
        if args.minimum_count is not None and count < args.minimum_count:
            raise PipelineError(
                f"export has {count} successes, needs {args.minimum_count}"
            )
        os.replace(staged, output)
    except Exception:
        staged.unlink(missing_ok=True)
        connection.close()
        raise
    connection.close()

    report = {
        **signature,
        "created_at": utc_now(),
        "output": str(output),
        "records": count,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "target_records": stage["target_records"],
        "pool_records": stage["pool_records"],
        "selection": selection,
    }
    if report_path is not None:
        write_json(report_path, report)
    print(f"exported {count} rows to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infinity-Parser2 response regeneration"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample")
    sample.add_argument("--source", required=True)
    sample.add_argument("--output-root", required=True)
    sample.add_argument("--population-size", type=int, required=True)
    sample.add_argument("--sample-size", type=int, required=True)
    sample.add_argument("--pilot-size", type=int)
    sample.add_argument("--reserve-size", type=int, default=20000)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--source-version", default="v1.12")
    sample.add_argument("--allowed-media-root", required=True)
    sample.add_argument("--path-map", action="append")
    sample.add_argument(
        "--convert-workers",
        type=int,
        default=64,
        help="threads validating media paths; the shared filesystem is "
        "latency bound so this is the main lever on sample wall clock",
    )
    sample.add_argument("--convert-chunk", type=int, default=4096)
    sample.add_argument("--overwrite", action="store_true")
    sample.set_defaults(handler=command_sample)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--output-root", required=True)
    generate.add_argument("--model-path", required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--endpoint", action="append", default=[])
    generate.add_argument("--stage", default="full")
    generate.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="0 or less leaves max_tokens unset so the teacher generates up to "
        "max_model_len minus the prompt length",
    )
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--concurrency-per-endpoint", type=int, default=4)
    generate.add_argument("--commit-batch", type=int, default=64)
    generate.add_argument("--timeout", type=float, default=600.0)
    generate.add_argument("--connect-timeout", type=float, default=30.0)
    generate.add_argument("--max-retries", type=int, default=5)
    generate.add_argument("--retry-errors", action="store_true")
    generate.add_argument(
        "--allow-config-change",
        action="store_true",
        help="accept a generation config that differs from the stored one, "
        "keeping already generated rows and recording the superseded config",
    )
    generate.add_argument("--api-key-env", default="OPENAI_API_KEY")
    generate.set_defaults(handler=command_generate)

    status = subparsers.add_parser("status")
    status.add_argument("--output-root", required=True)
    status.add_argument("--stage", default="full")
    status.add_argument("--json", dest="json_output", action="store_true")
    status.add_argument("--require-complete", action="store_true")
    status.set_defaults(handler=command_status)

    export = subparsers.add_parser("export")
    export.add_argument("--output-root", required=True)
    export.add_argument("--stage", default="full")
    export.add_argument("--output", "--final", dest="output", required=True)
    export.add_argument("--selection-manifest")
    export.add_argument("--minimum-count", type=int)
    export.add_argument("--report")
    export.set_defaults(handler=command_export)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
