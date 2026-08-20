"""Aggregate a draft-checkpoint sweep into a single comparison table.

Reads ``<run>/perf_results.csv`` for every ``<variant>_spec<N>`` subdirectory of a
sweep directory and prints the per-subset acceptance / throughput / latency tables
plus a drafts-weighted overall row.
"""

import argparse
import csv
from pathlib import Path

SUBSET_ORDER = [
    "HumanEval",
    "math_reasoning",
    "qa",
    "question",
    "rag",
    "summarization",
    "tool_call",
    "translation",
    "writing",
]

HEADER = """Qwen3.6-35B-A3B draft checkpoints, RedHatAI/speculator_benchmarks
single-GPU vLLM, concurrency 1, temperature 0, max_output_tokens 4096,
verifier Qwen/Qwen3.6-35B-A3B. Every subset runs its full row count
(HumanEval 164, tool_call 200, the other seven 80) -- the earlier
`max_requests 200/subset` cap never truncated any subset, and guidellm does
not resample, so capped and uncapped runs are the same measurement.
NOTE: `dflash_muon-*` was run alone on one card, sequentially; the other
      columns come from 8 concurrent single-GPU replicas. Acceptance metrics
      are per-draft and unaffected, but output_tps / itl / ttft carry a small
      bias because the 8-replica runs contend for host CPU.
NOTE: the dataset ships question.jsonl and writing.jsonl with identical
      content (same git oid), so those two rows are expected to match.
NOTE: acceptance positions are 0-indexed, so `pos14` is the 15th speculative
      position. A run that proposes fewer tokens than a position asks for
      leaves that cell blank rather than reporting a zero."""

SUBSET_WIDTH = 15
OVERALL_WIDTH = 22
COL_WIDTH = 15

# Acceptance rates are per-draft, so they aggregate across subsets weighted by
# each subset's draft count; throughput and latency medians do not.
DRAFTS_WEIGHTED_FIELDS = (
    "acceptance_length",
    "acceptance_at_pos_0",
    "acceptance_at_pos_6",
    "acceptance_at_pos_14",
)


def variant_sort_key(name: str) -> tuple[str, int]:
    """Sort ``dflash_spec15`` as ``("dflash", 15)`` so specs stay numeric."""
    variant, _, spec = name.partition("_spec")
    return variant, int(spec) if spec.isdigit() else 0


def label_of(run_dir: Path) -> str:
    variant, _, spec = run_dir.name.partition("_spec")
    return f"{variant}-{spec}"


def load_run(run_dir: Path) -> dict[str, dict[str, float]]:
    rows = {}
    with (run_dir / "perf_results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["subset"]] = {
                k: float(v)
                for k, v in row.items()
                if k not in ("subset", "strategy") and v not in ("", None)
            }
    return rows


def format_table(
    title: str,
    labels: list[str],
    data: dict[str, dict[str, dict[str, float]]],
    field: str,
    digits: int,
) -> list[str]:
    lines = [f"=== {title} ===", "subset".ljust(SUBSET_WIDTH)]
    lines[1] += "".join(label.rjust(COL_WIDTH) for label in labels)
    for subset in SUBSET_ORDER:
        line = subset.ljust(SUBSET_WIDTH)
        for label in labels:
            value = data[label].get(subset, {}).get(field)
            cell = "-" if value is None else f"{value:.{digits}f}"
            line += cell.rjust(COL_WIDTH)
        lines.append(line)
    return lines


def overall_row(rows: dict[str, dict[str, float]], field: str) -> float | None:
    """Drafts-weighted for acceptance fields, plain mean for perf fields.

    Returns ``None`` when no subset reports the field. That is how a run with
    fewer speculative positions than the row asks for stays blank: a spec=7 run
    has no ``acceptance_at_pos_14`` column at all, and averaging the positions
    it does have would silently answer a different question.
    """
    subsets = [s for s in SUBSET_ORDER if s in rows and field in rows[s]]
    if not subsets:
        return None
    if field in DRAFTS_WEIGHTED_FIELDS:
        drafts = sum(rows[s]["num_drafts"] for s in subsets)
        if field == "acceptance_length":
            # per-subset acceptance_length == accepted/drafts + 1
            accepted = sum(rows[s]["num_accepted_tokens"] for s in subsets)
            return accepted / drafts + 1.0
        return sum(rows[s]["num_drafts"] * rows[s][field] for s in subsets) / drafts
    return sum(rows[s][field] for s in subsets) / len(subsets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="sweep directory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="write to this file instead of stdout",
    )
    args = parser.parse_args()

    run_dirs = sorted(
        (p for p in args.run_dir.iterdir() if (p / "perf_results.csv").is_file()),
        key=lambda p: variant_sort_key(p.name),
    )
    if not run_dirs:
        raise SystemExit(f"no perf_results.csv found under {args.run_dir}")

    labels = [label_of(p) for p in run_dirs]
    data = {label_of(p): load_run(p) for p in run_dirs}

    lines = [HEADER, ""]
    for title, field, digits in (
        ("Acceptance length", "acceptance_length", 3),
        ("Output tokens/s (median)", "output_tps_median", 2),
        ("Inter-token latency ms (median)", "itl_median_ms", 2),
        ("TTFT ms (median)", "ttft_median_ms", 2),
    ):
        lines += format_table(title, labels, data, field, digits)
        lines.append("")

    lines.append(
        "=== Overall (drafts-weighted acceptance, "
        "arithmetic mean of subset medians) ==="
    )
    head = "metric".ljust(OVERALL_WIDTH)
    lines.append(head + "".join(label.rjust(COL_WIDTH) for label in labels))
    for metric, field, digits in (
        ("acceptance_length", "acceptance_length", 3),
        ("pos0 accept rate", "acceptance_at_pos_0", 4),
        ("pos6 accept rate", "acceptance_at_pos_6", 4),
        ("pos14 accept rate", "acceptance_at_pos_14", 4),
        ("output_tps", "output_tps_median", 2),
        ("itl_ms", "itl_median_ms", 2),
    ):
        line = metric.ljust(OVERALL_WIDTH)
        for label in labels:
            value = overall_row(data[label], field)
            cell = "" if value is None else f"{value:.{digits}f}"
            line += cell.rjust(COL_WIDTH)
        lines.append(line.rstrip())

    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
