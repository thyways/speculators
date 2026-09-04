#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

export REPO="${REPO:-$DEFAULT_REPO}"
export ROOT="${ROOT:-$(dirname -- "$REPO")}"
export ENV_REPO="${ENV_REPO:-$ROOT/speculators}"

MODEL="${MODEL:-$ROOT/model_weights/Qwen--Qwen3.6-35B-A3B}"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/qwen3.6-35b-a3b/qwen3.6-35b-a3b_train_spec_len3072_fullvocab}"
export RUN_DIR="${RUN_DIR:-$ROOT/model_weights/dspark_qwen3_6-35b-a3b-perfectblend}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_DIR/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"
WANDB_PROJECT="${WANDB_PROJECT:-speculators-qwen3_6-35b-a3b-perfectblend}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$ROOT/.secrets/wandb_key}"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:${VLLM_PORT}/v1}"
VLLM_HEALTH_ENDPOINT="${VLLM_HEALTH_ENDPOINT:-http://localhost:${VLLM_PORT}/health}"
BLOCK_SIZE="${BLOCK_SIZE:-7}"
MAX_ANCHORS="${MAX_ANCHORS:-2048}"
MARKOV_RANK="${MARKOV_RANK:-256}"
MARKOV_HEAD_TYPE="${MARKOV_HEAD_TYPE:-vanilla}"
CONFIDENCE_HEAD_ALPHA="${CONFIDENCE_HEAD_ALPHA:-1.0}"
DEFAULT_LOSS_FN='{"ce": 0.1, "tv": 0.9}'
LOSS_FN="${LOSS_FN:-$DEFAULT_LOSS_FN}"
JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-$$}}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-/tmp/dspark_qwen3_6_35b_a3b_hidden_states}"
VLLM_LOG="${VLLM_LOG:-$RUN_DIR/logs/vllm_${JOB_TAG}.log}"

SPEC_PYTHON="${SPEC_PYTHON:-$ENV_REPO/speculators_venv/bin/python}"
TORCHRUN="${TORCHRUN:-$ENV_REPO/speculators_venv/bin/torchrun}"
VLLM_PYTHON="${VLLM_PYTHON:-$ENV_REPO/vllm_venv/bin/python}"
LAUNCH_VLLM="${LAUNCH_VLLM:-$REPO/scripts/launch_vllm.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO/scripts/train.py}"
LOCAL_PYTHONPATH="${LOCAL_PYTHONPATH:-$REPO/src:$REPO/hs_connectors/src}"

mkdir -p "$RUN_DIR" "$RUN_DIR/logs" "$CHECKPOINT_DIR" "$HIDDEN_STATES_DIR"

for executable in "$SPEC_PYTHON" "$TORCHRUN" "$VLLM_PYTHON"; do
    if [[ ! -x "$executable" ]]; then
        echo "Missing executable: $executable" >&2
        exit 1
    fi
done

exec 9>"$RUN_DIR/training.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
    echo "Another cluster job already holds $RUN_DIR/training.lock" >&2
    exit 1
fi

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
    "$DATA_DIR/dataset_info.json"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing prepared-data artifact: $path" >&2
        exit 1
    fi
done

for command in setsid curl; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "The cluster image must provide the '$command' command." >&2
        exit 1
    fi
done

ALLOCATED_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "$ALLOCATED_GPUS"
if (( ${#GPU_LIST[@]} != 8 )); then
    echo "Expected exactly 8 allocated GPUs, got: $ALLOCATED_GPUS" >&2
    exit 1
fi
VLLM_GPUS=$(IFS=,; printf '%s' "${GPU_LIST[*]:0:2}")
TRAIN_GPUS=$(IFS=,; printf '%s' "${GPU_LIST[*]:2:6}")

VLLM_PID=""
TRAIN_PID=""

terminate_group() {
    local pgid="$1"
    [[ -z "$pgid" ]] && return 0
    if ! kill -0 -- "-$pgid" 2>/dev/null; then
        return 0
    fi

    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in {1..30}; do
        if ! kill -0 -- "-$pgid" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    kill -KILL -- "-$pgid" 2>/dev/null || true
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    terminate_group "$TRAIN_PID"
    terminate_group "$VLLM_PID"
    rm -rf "$HIDDEN_STATES_DIR"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Repository:       $REPO"
echo "Model:            $MODEL"
echo "Data (800k):      $DATA_DIR"
echo "Checkpoints:      $CHECKPOINT_DIR"
echo "W&B project:      $WANDB_PROJECT"
echo "vLLM GPUs:        $VLLM_GPUS"
echo "Training GPUs:    $TRAIN_GPUS"
echo "Markov head:      $MARKOV_HEAD_TYPE (rank $MARKOV_RANK)"
echo "DSpark loss:      $LOSS_FN"
echo "vLLM log:         $VLLM_LOG"

"$SPEC_PYTHON" - "$VLLM_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1)
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        raise SystemExit(f"Port {port} is already in use; refusing to launch another server")
PY

echo "=== Launching vLLM ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
    PYTHONPATH="$LOCAL_PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    "$VLLM_PYTHON" \
    "$LAUNCH_VLLM" \
    "$MODEL" \
    --target-layer-ids 2 20 37 \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    -- \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --max-model-len 3088 \
    --gpu-memory-utilization 0.92 \
    --port "$VLLM_PORT" \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM on port $VLLM_PORT (PID/PGID $VLLM_PID)..."
deadline=$((SECONDS + 1800))
until curl -sf "$VLLM_HEALTH_ENDPOINT" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming healthy. Last log lines:" >&2
        tail -n 100 "$VLLM_LOG" >&2 || true
        wait "$VLLM_PID"
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out after 30 minutes waiting for vLLM. See $VLLM_LOG" >&2
        exit 1
    fi
    sleep 2
done

echo "vLLM is healthy."
echo "=== Launching DSpark training ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    PYTHONPATH="$LOCAL_PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_MODE="$WANDB_MODE" \
    "$TORCHRUN" \
    --standalone \
    --nproc_per_node 6 \
    "$TRAIN_SCRIPT" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_DIR" \
    --save-path "$CHECKPOINT_DIR" \
    --epochs 3 \
    --train-data-ratio 0.98 \
    --optimizer muon \
    --muon-lr 2e-4 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --noise-std 0.05 \
    --scheduler-type linear \
    --scheduler-warmup-ratio 0.01 \
    --total-seq-len 8192 \
    --speculator-type dspark \
    --block-size "$BLOCK_SIZE" \
    --sample-from-anchor \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers 5 \
    --dflash-decay-gamma 7 \
    --sliding-window 2048 \
    --sliding-window-non-causal \
    --target-layer-ids 2 20 37 \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --loss-fn "$LOSS_FN" \
    --vllm-endpoint "$VLLM_ENDPOINT" \
    --request-timeout 180 \
    --on-missing generate \
    --on-generate delete \
    --logger wandb \
    --log-dir "$LOG_DIR" \
    --checkpoint-freq 0.5 \
    --run-name dspark_5swa2048nc_muon_fullvocab &
TRAIN_PID=$!

wait "$TRAIN_PID"
