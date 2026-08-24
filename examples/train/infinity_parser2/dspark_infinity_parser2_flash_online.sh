#!/usr/bin/env bash
# Online DSpark training for Infinity-Parser2-Flash on the prepared 800k corpus.

set -Eeuo pipefail

ROOT="/inspire/sfs/project/inf-multimodal/public/wumengke"
REPO="$ROOT/speculators"
MODEL="/home/ma-user/work/data_mllm/publish_models/Infinity-Parser2-2B-2604"
DATA_DIR="$ROOT/datasets/infinity_parser2_v1_12_dflash_data/full"
RUN_DIR="${RUN_DIR:-$REPO/output/dspark_infinity_parser2_flash_online_full}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-3}"
# Threads per worker fetching one batch's hidden states concurrently. A batch
# packed from short samples holds ~20 of them (seq_len p50 is 801 against a
# 16384-token budget) and each costs a blocking request, so serial fetching
# makes the step wait on the sum of those latencies. In-flight requests total
# NUM_WORKERS * FETCH_THREADS, spread over the vLLM data-parallel ranks.
FETCH_THREADS="${FETCH_THREADS:-4}"
# Per-request timeout. The tail matters more than the ceiling: a stuck request
# blocks its rank and every peer then waits at the gradient all-reduce, so fail
# fast and let --max-retries retry instead of holding the step for minutes.
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/model_weights/dspark_infinity_parser2_2b_v1_12}"
LOG_DIR="${LOG_DIR:-$RUN_DIR}"
WANDB_PROJECT="${WANDB_PROJECT:-infinity-parser2-flash}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$ROOT/.secrets/wandb_key}"
MEDIA_ROOT="/inspire/sfs/project/inf-multimodal/public"
VLLM_PORT="${VLLM_PORT:-8200}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MARKOV_RANK="${MARKOV_RANK:-256}"
MARKOV_HEAD_TYPE="${MARKOV_HEAD_TYPE:-vanilla}"
CONFIDENCE_HEAD_ALPHA="${CONFIDENCE_HEAD_ALPHA:-1.0}"
DEFAULT_LOSS_FN='{"ce": 0.1, "tv": 0.9}'
LOSS_FN="${LOSS_FN:-$DEFAULT_LOSS_FN}"
TARGET_LAYER_IDS=(2 7 12 17 22)
JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-$$}}"
HIDDEN_STATES_DIR=""
VLLM_LOG="$RUN_DIR/vllm_${JOB_TAG}.log"

SPEC_PYTHON="$REPO/speculators_venv/bin/python"
TORCHRUN="$REPO/speculators_venv/bin/torchrun"
VLLM_PYTHON="$REPO/vllm_venv/bin/python"

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR"

for executable in "$SPEC_PYTHON" "$TORCHRUN" "$VLLM_PYTHON"; do
    if [[ ! -x "$executable" ]]; then
        echo "Missing executable: $executable" >&2
        exit 1
    fi
done

if [[ ! -f "$MODEL/config.json" ]]; then
    echo "Missing model config: $MODEL/config.json" >&2
    exit 1
fi

if [[ ! -f "$WANDB_KEY_FILE" ]]; then
    echo "Missing W&B key file: $WANDB_KEY_FILE" >&2
    exit 1
fi
WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
if [[ -z "$WANDB_API_KEY" ]]; then
    echo "W&B key file is empty: $WANDB_KEY_FILE" >&2
    exit 1
fi
export WANDB_API_KEY

for path in \
    "$DATA_DIR/state.json" \
    "$DATA_DIR/dataset_info.json" \
    "$DATA_DIR/token_freq.pt"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing prepared-data artifact: $path" >&2
        exit 1
    fi
done

for command in curl flock pgrep setsid; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Missing command: $command" >&2
        exit 1
    fi
done

ALLOCATED_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "${ALLOCATED_GPUS//[[:space:]]/}"
if (( ${#GPU_LIST[@]} != 8 )); then
    echo "Expected exactly 8 GPUs, got: $ALLOCATED_GPUS" >&2
    exit 1
fi
# Online training fetches every hidden state from the vLLM side, so the split
# between generation and training GPUs is the main throughput knob. VLLM_GPU_COUNT
# takes the leading GPUs; the rest train. Note that changing the training count
# also changes the data-parallel degree, and therefore the tokens per step.
VLLM_GPU_COUNT="${VLLM_GPU_COUNT:-4}"
if (( VLLM_GPU_COUNT < 1 || VLLM_GPU_COUNT > ${#GPU_LIST[@]} - 1 )); then
    echo "VLLM_GPU_COUNT must be between 1 and $(( ${#GPU_LIST[@]} - 1 ))" >&2
    exit 1
fi
TRAIN_GPU_COUNT=$(( ${#GPU_LIST[@]} - VLLM_GPU_COUNT ))
VLLM_GPUS=$(IFS=,; printf '%s' "${GPU_LIST[*]:0:VLLM_GPU_COUNT}")
TRAIN_GPUS=$(IFS=,; printf '%s' "${GPU_LIST[*]:VLLM_GPU_COUNT:TRAIN_GPU_COUNT}")

exec 9>"$RUN_DIR/training.lock"
if ! flock -n 9; then
    echo "Another job is using $RUN_DIR" >&2
    exit 1
fi

VLLM_PID=""
TRAIN_PID=""

terminate_job() {
    local pid="$1"
    local alive
    local -a children=()

    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 0
    mapfile -t children < <(pgrep -P "$pid" 2>/dev/null || true)
    if (( ${#children[@]} )); then
        kill -TERM "${children[@]}" 2>/dev/null || true
    fi
    kill -TERM -- "-$pid" 2>/dev/null || true

    for _ in {1..30}; do
        alive=0
        kill -0 "$pid" 2>/dev/null && alive=1
        for child in "${children[@]}"; do
            kill -0 "$child" 2>/dev/null && alive=1
        done
        (( alive == 0 )) && return 0
        sleep 1
    done

    if (( ${#children[@]} )); then
        kill -KILL "${children[@]}" 2>/dev/null || true
    fi
    kill -KILL -- "-$pid" 2>/dev/null || true
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM HUP
    terminate_job "$TRAIN_PID"
    terminate_job "$VLLM_PID"
    [[ -n "$TRAIN_PID" ]] && wait "$TRAIN_PID" 2>/dev/null || true
    [[ -n "$VLLM_PID" ]] && wait "$VLLM_PID" 2>/dev/null || true
    case "$HIDDEN_STATES_DIR" in
        */dspark_parser2_flash_*) rm -rf -- "$HIDDEN_STATES_DIR" ;;
    esac
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

HIDDEN_STATES_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dspark_parser2_flash_${JOB_TAG}.XXXXXX")"

"$SPEC_PYTHON" - "$VLLM_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1)
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        raise SystemExit(f"Port {port} is already in use")
PY

echo "Model:         $MODEL"
echo "Data:          $DATA_DIR"
echo "Checkpoints:   $CHECKPOINT_DIR"
echo "vLLM GPUs:     $VLLM_GPUS"
echo "Training GPUs: $TRAIN_GPUS"
echo "Block size:    $BLOCK_SIZE"
echo "Markov head:   $MARKOV_HEAD_TYPE (rank $MARKOV_RANK)"
echo "DSpark loss:   $LOSS_FN"
echo "Fetch:         $NUM_WORKERS workers x $FETCH_THREADS threads, timeout ${REQUEST_TIMEOUT}s"

setsid env \
    CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
    PYTHONUNBUFFERED=1 \
    "$VLLM_PYTHON" "$REPO/scripts/launch_vllm.py" "$MODEL" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --tensor-parallel-size 1 \
    --data-parallel-size "$VLLM_GPU_COUNT" \
    --gpu-memory-utilization 0.9 \
    --max_model_len 65536 \
    --api-server-count 8 \
    --served-model-name "$MODEL" \
    --allowed-local-media-path "$MEDIA_ROOT" \
    --limit-mm-per-prompt '{"image":16}' \
    --port "$VLLM_PORT" \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

deadline=$((SECONDS + 1800))
until curl -fsS --max-time 10 "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        tail -n 100 "$VLLM_LOG" >&2 || true
        wait "$VLLM_PID"
    fi
    if (( SECONDS >= deadline )); then
        tail -n 100 "$VLLM_LOG" >&2 || true
        echo "Timed out waiting for vLLM" >&2
        exit 1
    fi
    sleep 2
done

setsid env \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    PYTHONUNBUFFERED=1 \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_MODE="$WANDB_MODE" \
    "$TORCHRUN" \
    --standalone \
    --nproc_per_node "$TRAIN_GPU_COUNT" \
    "$REPO/scripts/train.py" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_DIR" \
    --save-path "$CHECKPOINT_DIR" \
    --speculator-type dspark \
    --draft-arch qwen3 \
    --draft-hidden-act silu \
    --num-layers 5 \
    --mask-token-id 248077 \
    --block-size "$BLOCK_SIZE" \
    --sample-from-anchor \
    --max-anchors 3072 \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --draft-mrope-full-head-hack \
    --sliding-window 2048 \
    --sliding-window-non-causal \
    --draft-attn-impl simple_flex_attention \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --loss-fn "$LOSS_FN" \
    --dflash-decay-gamma 4.0 \
    --optimizer muon \
    --muon-lr 2e-4 \
    --lr 1e-4 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.04 \
    --epochs 3 \
    --checkpoint-freq 0.1 \
    --total-seq-len 16384 \
    --train-data-ratio 0.99 \
    --noise-std 0 \
    --hidden-states-dtype bfloat16 \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --fetch-threads "$FETCH_THREADS" \
    --hidden-states-backend file \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --vllm-endpoint "http://127.0.0.1:${VLLM_PORT}/v1" \
    --on-missing generate \
    --on-generate delete \
    --request-timeout "$REQUEST_TIMEOUT" \
    --max-retries 5 \
    --fail-on-hidden-state-error \
    --seed 42 \
    --logger wandb \
    --log-dir "$LOG_DIR" \
    --run-name "$JOB_TAG" &
TRAIN_PID=$!

wait "$TRAIN_PID"
