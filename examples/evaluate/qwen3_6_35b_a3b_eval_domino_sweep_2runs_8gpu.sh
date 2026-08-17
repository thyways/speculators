#!/usr/bin/env bash
# Add the Qwen3.6-35B-A3B Domino checkpoint to an existing draft sweep produced
# by qwen3_6_35b_a3b_eval_draft_sweep_5runs_8gpu.sh.
#
# Two runs, executed one after another (each run occupies all eight GPUs as
# eight independent single-GPU replicas, per-replica concurrency 1, greedy):
#
#   1. domino, 7 speculative tokens
#   2. domino, 15 speculative tokens
#
# Unlike the five-run sweep this script does not create a new sweep root: it
# appends `domino_spec7` / `domino_spec15` directories and `sweep_summary.txt`
# rows to SWEEP_ROOT, which must already exist. Settings are kept identical to
# the five-run sweep so the resulting columns are directly comparable.
#
# The Domino checkpoint lives outside $WORKSPACE/model_weights/Qwen3_6_35B_A3B_draft,
# so the draft path is passed explicitly instead of being derived from the label.
#
# The GPU placeholder (`$WORKSPACE/training.py`) is stopped before the sweep and
# restarted afterwards. Only placeholder processes registered in
# `$WORKSPACE/.hold_gpu/*.json` are touched, per the workspace convention.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE="${WORKSPACE:-$(dirname -- "$REPO")}"
HOLD_DIR="$WORKSPACE/.hold_gpu"

DRAFT_MODEL="${DRAFT_MODEL:-$WORKSPACE/model_weights/domino_qwen3_6_35b_a3b_5full/checkpoints/0}"
SWEEP_ROOT="${SWEEP_ROOT:-$WORKSPACE/evaluation_results/qwen3_6_35b_a3b_draft_sweep_20260817_015126}"

MAX_REQUESTS="${MAX_REQUESTS:-200}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RESTART_PLACEHOLDER="${RESTART_PLACEHOLDER:-1}"
PLACEHOLDER_MEM="${PLACEHOLDER_MEM:-0.9}"
PLACEHOLDER_UTIL="${PLACEHOLDER_UTIL:-20}"
IDLE_WAIT_SECONDS="${IDLE_WAIT_SECONDS:-600}"

# label:algorithm:speculative_tokens
RUNS=(
    "domino_spec7:domino:7"
    "domino_spec15:domino:15"
)

log() {
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

stop_placeholder() {
    shopt -s nullglob
    local metas=("$HOLD_DIR"/*.json)
    shopt -u nullglob
    if ((${#metas[@]} == 0)); then
        log "No GPU placeholder registered in $HOLD_DIR; nothing to stop."
        return 0
    fi
    local meta pid
    for meta in "${metas[@]}"; do
        pid="$(basename -- "$meta" .json)"
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping GPU placeholder PID $pid"
            kill -TERM "$pid" 2>/dev/null || true
        else
            log "Placeholder PID $pid already gone; removing stale $meta"
            rm -f -- "$meta"
        fi
    done
}

# Waits until every GPU in GPU_IDS reports < 1 GiB used, so the next run's
# idle check does not trip on the previous run's servers still shutting down.
wait_for_idle_gpus() {
    local deadline=$((SECONDS + IDLE_WAIT_SECONDS))
    local gpu used busy
    while true; do
        busy=""
        for gpu in ${GPU_IDS//,/ }; do
            used="$(
                nvidia-smi --id="$gpu" --query-gpu=memory.used \
                    --format=csv,noheader,nounits | tr -d '[:space:]'
            )"
            if [[ ! "$used" =~ ^[0-9]+$ ]] || ((used > 1024)); then
                busy="$busy $gpu(${used:-?}MiB)"
            fi
        done
        if [[ -z "$busy" ]]; then
            return 0
        fi
        if ((SECONDS >= deadline)); then
            log "GPUs still busy after ${IDLE_WAIT_SECONDS}s:$busy"
            return 1
        fi
        sleep 10
    done
}

start_placeholder() {
    if [[ "$RESTART_PLACEHOLDER" != "1" ]]; then
        log "RESTART_PLACEHOLDER=$RESTART_PLACEHOLDER; leaving GPUs idle."
        return 0
    fi
    log "Starting GPU placeholder on $GPU_IDS (mem=$PLACEHOLDER_MEM util=$PLACEHOLDER_UTIL)"
    (
        cd "$WORKSPACE" && setsid nohup python3 training.py \
            -d "$GPU_IDS" -m "$PLACEHOLDER_MEM" -u "$PLACEHOLDER_UTIL" \
            >/tmp/hold_gpu.log 2>&1 </dev/null &
    )
}

if [[ ! -d "$SWEEP_ROOT" ]]; then
    echo "Sweep root does not exist: $SWEEP_ROOT" >&2
    exit 1
fi
for path in "$DRAFT_MODEL/config.json" "$DRAFT_MODEL/model.safetensors"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

SUMMARY="$SWEEP_ROOT/sweep_summary.txt"
touch "$SUMMARY"

log "Sweep root:  $SWEEP_ROOT"
log "Draft model: $DRAFT_MODEL"
stop_placeholder
if ! wait_for_idle_gpus; then
    echo "GPUs did not become idle before the sweep." >&2
    exit 1
fi

failed_runs=()
for entry in "${RUNS[@]}"; do
    IFS=':' read -r label algorithm spec_tokens <<< "$entry"
    runner="$SCRIPT_DIR/example_qwen3_6_35b_a3b_${algorithm}_speculator_benchmarks_8gpu.sh"
    output_root="$SWEEP_ROOT/$label"
    run_log="$SWEEP_ROOT/${label}.log"

    if [[ ! -f "$runner" ]]; then
        echo "Missing runner script: $runner" >&2
        failed_runs+=("$label (missing runner)")
        continue
    fi

    log "=== Run $label: $algorithm, $spec_tokens speculative tokens ==="
    log "    model=$DRAFT_MODEL"
    log "    log=$run_log"
    start_epoch="$SECONDS"
    if env \
        REPO="$REPO" \
        WORKSPACE="$WORKSPACE" \
        MODEL="$DRAFT_MODEL" \
        NUM_SPECULATIVE_TOKENS="$spec_tokens" \
        GPU_IDS="$GPU_IDS" \
        MAX_REQUESTS="$MAX_REQUESTS" \
        MAX_CONCURRENCY="$MAX_CONCURRENCY" \
        MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" \
        TEMPERATURE="$TEMPERATURE" \
        OUTPUT_ROOT="$output_root" \
        bash "$runner" >"$run_log" 2>&1
    then
        log "Run $label finished in $((SECONDS - start_epoch))s"
        printf 'OK    %-14s %-7s spec=%-3s %ss  %s\n' \
            "$label" "$algorithm" "$spec_tokens" \
            "$((SECONDS - start_epoch))" "$output_root" >> "$SUMMARY"
    else
        status=$?
        log "Run $label FAILED (exit $status) after $((SECONDS - start_epoch))s"
        printf 'FAIL  %-14s %-7s spec=%-3s exit=%s  %s\n' \
            "$label" "$algorithm" "$spec_tokens" "$status" "$run_log" >> "$SUMMARY"
        failed_runs+=("$label (exit $status)")
    fi

    wait_for_idle_gpus || log "Warning: GPUs not fully idle after $label"
done

log "=== Sweep summary ==="
cat "$SUMMARY"

start_placeholder

if ((${#failed_runs[@]} > 0)); then
    echo "Failed runs: ${failed_runs[*]}" >&2
    exit 1
fi

log "Both Domino runs complete: $SWEEP_ROOT"
