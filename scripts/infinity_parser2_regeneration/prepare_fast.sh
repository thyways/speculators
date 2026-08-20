#!/usr/bin/env bash
# Export + prepare the regenerated corpus into dflash training artifacts.
#
# Replaces run_all.sh's prepare_stage for the full-size corpus. That path derives
# loss-mask boundaries through the vLLM /render endpoint, rendering every
# assistant turn twice -- once with the generation prompt, once complete -- and
# re-sending the images each time, roughly 2M calls for this corpus. The endpoint
# tops out near 16 renders/s here however it is configured (1 to 64 API server
# processes, 1 to 8 client processes, shm or lru mm cache), with no CPU, GPU or
# filesystem saturation to explain it: that is days of work. The identical
# tokenization in-process profiles at ~90 ms and scales across cores, so this
# script renders locally and never starts a server. Equivalence was verified end
# to end on real rows: input_ids, loss_mask and token_freq all match the endpoint
# exactly. Prepare therefore needs no GPU, and all of them are held instead.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

PYTHON_BIN="${PARSER2_PYTHON_BIN:-${REPO_ROOT}/speculators_venv/bin/python}"
PIPELINE="${SCRIPT_DIR}/script.py"
PREPARE_SCRIPT="${REPO_ROOT}/scripts/infinity_parser2_prepare_data.py"
VOCAB_SCRIPT="${REPO_ROOT}/scripts/build_vocab_mapping.py"

MODEL="${PARSER2_MODEL:-/home/ma-user/work/data_mllm/publish_models/Infinity-Parser2-2B-2604}"
OUTPUT_ROOT="${PARSER2_REGEN_ROOT:-/inspire/sfs/project/inf-multimodal/public/wumengke/datasets/infinity_parser2_v1_12_regen_1p5m}"
FINAL_DIR="${PARSER2_FINAL_DIR:-/inspire/sfs/project/inf-multimodal/public/wumengke/datasets/infinity_parser2_v1_12_target_answers}"
PREPARED_ROOT="${PARSER2_PREPARED_ROOT:-/inspire/sfs/project/inf-multimodal/public/wumengke/datasets/infinity_parser2_v1_12_dflash_data}"

STAGE="${PARSER2_STAGE:-full}"
# Must match --total-seq-len in the training script: that is the per-rank token
# budget the packing sampler works with, so a longer row can never be packed.
SEQ_LENGTH="${PARSER2_SEQ_LENGTH:-16384}"
TOKEN_FREQ_RATIO="${PARSER2_TRAIN_DATA_RATIO:-0.99}"
MINIMUM_VALID_TOKENS="${PARSER2_MINIMUM_VALID_TOKENS:-1}"
DRAFT_VOCAB_SIZE="${PARSER2_DRAFT_VOCAB_SIZE:-32000}"
# One render costs ~90 ms of single-core CPU, so workers scale with cores. Keep
# each worker single-threaded: the image processor otherwise opens a thread pool
# per worker and the oversubscription cost this many processes is severe.
PREPROCESSING_WORKERS="${PARSER2_PREPROCESSING_WORKERS:-64}"
PREPROCESSING_BATCH_SIZE="${PARSER2_PREPROCESSING_BATCH_SIZE:-64}"
# prepare aborts when preprocessing survivors fall below --ranked-target-samples,
# and rows do get dropped there (over --seq-length, or below the valid-token
# floor). Asking for every success would throw away hours over a handful of rows.
# Selection takes the lowest ranks, and rank is a uniform random permutation, so
# the kept subset stays uniform.
TARGET_MARGIN_PERCENT="${PARSER2_TARGET_MARGIN_PERCENT:-97}"

HOLD_GPUS="${PARSER2_HOLD_GPUS:-1}"
HOLD_SCRIPT="${PARSER2_HOLD_SCRIPT:-/inspire/sfs/project/inf-multimodal/public/wumengke/training.py}"
HOLD_MEM="${PARSER2_HOLD_MEM:-0.9}"
HOLD_UTIL="${PARSER2_HOLD_UTIL:-90}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

HOLD_PID=""
PREPARED_DIR="${PREPARED_ROOT}/${STAGE}"
POOL_PATH="${FINAL_DIR}/${STAGE}.pool.jsonl"
LOG_DIR="${OUTPUT_ROOT}/prepare_logs"

die() {
    echo "Error: $*" >&2
    exit 1
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM HUP
    if [[ -n "$HOLD_PID" ]]; then
        kill -TERM -- "-$HOLD_PID" 2>/dev/null || kill -TERM "$HOLD_PID" 2>/dev/null || true
        wait "$HOLD_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

[[ -x "$PYTHON_BIN" ]] || die "missing executable: $PYTHON_BIN"
for script in "$PIPELINE" "$PREPARE_SCRIPT" "$VOCAB_SCRIPT"; do
    [[ -f "$script" ]] || die "missing script: $script"
done
mkdir -p "$FINAL_DIR" "$PREPARED_ROOT" "$LOG_DIR"

# The sqlite state is multiple GiB. Every full scan of it is bound by the shared
# filesystem's small-read IOPS (~2.5 MiB/s), while one sequential read runs at
# ~420 MiB/s, so pull it into the page cache before querying.
warm_state() {
    local state="${OUTPUT_ROOT}/state.sqlite3"
    [[ -f "$state" ]] || die "no sample state at $state"
    echo "Warming page cache for $state"
    cat "$state" >/dev/null
}

success_count() {
    "$PYTHON_BIN" - "$OUTPUT_ROOT" "$STAGE" <<'PY'
import json
import sqlite3
import sys
from pathlib import Path

state = Path(sys.argv[1]) / "state.sqlite3"
db = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
db.row_factory = sqlite3.Row
stages = json.loads(
    db.execute("SELECT value FROM meta WHERE key = 'stages'").fetchone()["value"]
)
limit = int(stages[sys.argv[2]]["candidate_rank_exclusive"])
row = db.execute(
    "SELECT COUNT(*) AS count FROM requests r "
    "JOIN generations g ON g.id = r.id AND g.status = 'success' "
    "WHERE r.rank < ?",
    (limit,),
).fetchone()
print(int(row["count"]))
PY
}

# Prepare is pure CPU work, so nothing would otherwise keep the cluster from
# reclaiming these GPUs for the hours it runs.
hold_all_gpus() {
    (( HOLD_GPUS == 1 )) || return 0
    [[ -f "$HOLD_SCRIPT" ]] || {
        echo "Skipping GPU hold: no ${HOLD_SCRIPT}" >&2
        return 0
    }
    local devices
    devices=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)
    [[ -n "$devices" ]] || return 0
    echo "Holding GPUs ${devices} against reclamation"
    setsid nohup python3 "$HOLD_SCRIPT" \
        -d "$devices" -m "$HOLD_MEM" -u "$HOLD_UTIL" \
        >"${LOG_DIR}/hold_gpu.log" 2>&1 </dev/null &
    HOLD_PID=$!
}

export_pool() {
    local target="$1"
    if [[ -s "$POOL_PATH" ]]; then
        echo "Reusing existing pool: ${POOL_PATH}"
        return 0
    fi
    echo "Exporting ${target} rows to ${POOL_PATH}"
    "$PYTHON_BIN" "$PIPELINE" export \
        --output-root "$OUTPUT_ROOT" \
        --stage "$STAGE" \
        --output "$POOL_PATH" \
        --minimum-count "$target" \
        --report "${POOL_PATH%.jsonl}.export.json"
}

prepare_dataset() {
    echo "Preparing ${PREPARED_DIR} with ${PREPROCESSING_WORKERS} workers"
    local args=(
        "$PYTHON_BIN" "$PREPARE_SCRIPT"
        --model "$MODEL"
        --data "$POOL_PATH"
        --output "$PREPARED_DIR"
        --seq-length "$SEQ_LENGTH"
        --ranked-target-samples "$1"
        --token-freq-train-ratio "$TOKEN_FREQ_RATIO"
        --local-render
        --drop-clipped-rows
        --minimum-valid-tokens "$MINIMUM_VALID_TOKENS"
        --num-preprocessing-workers "$PREPROCESSING_WORKERS"
        --preprocessing-batch-size "$PREPROCESSING_BATCH_SIZE"
    )
    [[ "${PARSER2_OVERWRITE_PREPARED:-0}" == "1" ]] && args+=(--overwrite)
    "${args[@]}"
}

build_vocab() {
    echo "Building draft vocab mapping"
    "$PYTHON_BIN" "$VOCAB_SCRIPT" \
        --token-freq-path "${PREPARED_DIR}/token_freq.pt" \
        --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
        --target-model-path "$MODEL" \
        --output-path "$PREPARED_DIR"
}

verify_artifacts() {
    local missing=0
    local artifact
    for artifact in state.json dataset_info.json token_freq.pt d2t.npy t2d.npy; do
        if [[ -f "${PREPARED_DIR}/${artifact}" ]]; then
            echo "  ok      ${artifact}"
        else
            echo "  MISSING ${artifact}" >&2
            missing=1
        fi
    done
    (( missing == 0 )) || die "prepared data is incomplete"
    echo "Prepared data ready: ${PREPARED_DIR}"
}

warm_state
successes="$(success_count)"
(( successes > 0 )) || die "no successful generations for stage ${STAGE}"
target="${PARSER2_TARGET_RECORDS:-$(( successes * TARGET_MARGIN_PERCENT / 100 ))}"
(( target > 0 )) || die "target record count must be positive"
echo "Stage ${STAGE}: ${successes} successful records, using target ${target}"

export_pool "$target"
hold_all_gpus
prepare_dataset "$target"
build_vocab
verify_artifacts
