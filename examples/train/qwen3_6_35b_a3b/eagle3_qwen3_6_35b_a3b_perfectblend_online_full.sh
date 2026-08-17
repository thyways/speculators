#!/usr/bin/env bash
# Full-data online Eagle3 training for Qwen3.6-35B-A3B on one 8-GPU node.
# Reuses the prepared PerfectBlend data and vocabulary mapping from DFlash, but
# uses three verifier features, one full-attention draft layer,
# pre-FC/output normalization, and teacher-forced TTT steps.
# Run this script as the cluster job command; do not wrap it in nohup.
# The prepared data in DATA_DIR must use this model's tokenizer.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

export REPO="${REPO:-$DEFAULT_REPO}"
export ROOT="${ROOT:-$(dirname -- "$REPO")}"
export ENV_REPO="${ENV_REPO:-$ROOT/speculators}"

MODEL="${MODEL:-$ROOT/model_weights/Qwen/Qwen3.6-35B-A3B}"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/qwen3_6_35b_500k}"
export RUN_DIR="${RUN_DIR:-$ROOT/model_weights/eagle3_1_qwen3_6_35b_a3b_1full}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_DIR/checkpoints}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$RUN_DIR/tensorboard}"

VLLM_PORT="${VLLM_PORT:-8100}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:${VLLM_PORT}/v1}"
VLLM_HEALTH_ENDPOINT="${VLLM_HEALTH_ENDPOINT:-http://localhost:${VLLM_PORT}/health}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
MASK_TOKEN_ID="${MASK_TOKEN_ID:-248077}"
TTT_STEPS="${TTT_STEPS:-7}"
LR="${LR:-1e-4}"
JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-$$}}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-${TMPDIR:-/tmp}/eagle3_qwen3_6_35b_a3b_${JOB_TAG}_hidden_states}"
VLLM_LOG="${VLLM_LOG:-$RUN_DIR/vllm_${JOB_TAG}.log}"

SPEC_PYTHON="${SPEC_PYTHON:-$ENV_REPO/speculators_venv/bin/python}"
TORCHRUN="${TORCHRUN:-$ENV_REPO/speculators_venv/bin/torchrun}"
VLLM_PYTHON="${VLLM_PYTHON:-$ENV_REPO/vllm_venv/bin/python}"
LAUNCH_VLLM="${LAUNCH_VLLM:-$REPO/scripts/launch_vllm.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO/scripts/train.py}"
LOCAL_PYTHONPATH="${LOCAL_PYTHONPATH:-$REPO/src:$REPO/hs_connectors/src}"

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$TENSORBOARD_DIR" "$HIDDEN_STATES_DIR"

for executable in "$SPEC_PYTHON" "$TORCHRUN" "$VLLM_PYTHON"; do
    if [[ ! -x "$executable" ]]; then
        echo "Missing executable: $executable" >&2
        exit 1
    fi
done

# Prevent two invocations from writing the same checkpoint directory
# concurrently. The lock is released automatically when the job exits.
exec 9>"$RUN_DIR/training.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
    echo "Another cluster job already holds $RUN_DIR/training.lock" >&2
    exit 1
fi

if [[ ! -f "$MODEL/config.json" ]]; then
    echo "Missing model config: $MODEL/config.json" >&2
    exit 1
fi

for path in \
    "$DATA_DIR/state.json" \
    "$DATA_DIR/dataset_info.json" \
    "$DATA_DIR/token_freq.pt" \
    "$DATA_DIR/d2t.npy" \
    "$DATA_DIR/t2d.npy"; do
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

# Preserve the GPU allocation supplied by the scheduler. When no allocation
# variable is present, default to all eight local GPUs. Two GPUs each host one
# TP=1 vLLM data-parallel replica; the other six train the draft model.
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

echo "Repository:        $REPO"
echo "Model:             $MODEL"
echo "Data:              $DATA_DIR"
echo "Checkpoints:       $CHECKPOINT_DIR"
echo "TensorBoard:       $TENSORBOARD_DIR/$JOB_TAG"
echo "Eagle3 layers:     1"
echo "TTT steps:         $TTT_STEPS"
echo "Target layers:     2 20 37"
echo "Mask token ID:     $MASK_TOKEN_ID"
echo "Learning rate:     $LR"
echo "Full-attn layers:  0"
echo "vLLM GPUs:         $VLLM_GPUS"
echo "Training GPUs:     $TRAIN_GPUS"
echo "vLLM log:          $VLLM_LOG"

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
    --max-model-len 10000 \
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
echo "=== Launching Eagle3 training ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    PYTHONPATH="$LOCAL_PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    "$TORCHRUN" \
    --standalone \
    --nproc_per_node 6 \
    "$TRAIN_SCRIPT" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_DIR" \
    --save-path "$CHECKPOINT_DIR" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs 1 \
    --train-data-ratio 0.98 \
    --optimizer adamw \
    --lr "$LR" \
    --weight-decay 0.01 \
    --noise-std 0 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.04 \
    --total-seq-len 4096 \
    --hidden-states-dtype bfloat16 \
    --speculator-type eagle3 \
    --draft-arch qwen3 \
    --draft-hidden-act silu \
    --draft-mrope-full-head-hack \
    --draft-attn-impl simple_flex_attention \
    --num-layers 1 \
    --mask-token-id "$MASK_TOKEN_ID" \
    --ttt-steps "$TTT_STEPS" \
    --ttt-step-loss-decay 1.0 \
    --loss-fn kl_div \
    --norm-before-residual \
    --no-embed-requires-grad \
    --no-norm-before-fc \
    --fc-norm \
    --norm-output \
    --full-attention-indices 0 \
    --target-layer-ids 2 20 37 \
    --hidden-states-backend file \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --vllm-endpoint "$VLLM_ENDPOINT" \
    --request-timeout 300 \
    --max-retries 5 \
    --on-missing generate \
    --on-generate delete \
    --fail-on-hidden-state-error \
    --seed 42 \
    --logger tensorboard \
    --log-dir "$TENSORBOARD_DIR" \
    --checkpoint-freq 0.1 \
    --run-name eagle3_1_qwen3_6_35b_a3b_1full &
TRAIN_PID=$!

# Keep the cluster job attached to training. Its stdout/stderr is therefore
# visible in the cluster job log. On completion or cancellation, cleanup stops
# the full vLLM process group and removes transient hidden-state files.
wait "$TRAIN_PID"
