#!/usr/bin/env bash
# Full-data online DFly training for Qwen3.6-35B-A3B on one 8-GPU node.
# Reuses the Qwen3.6-35B-A3B DFlash target-feature and six-layer draft layout,
# adding DFly per-layer target fusion and previous-token hidden correction.
# Keeps the AdamW + cosine recipe (4% linear warmup, then cosine decay).
# Run this script as the cluster job command; do not wrap it in nohup.
# The prepared data in DATA_DIR must use this model's tokenizer.

set -Eeuo pipefail

REPO="/inspire/sfs/project/inf-multimodal/public/wumengke/speculators"
MODEL="/inspire/sfs/project/inf-multimodal/public/share_base_models/Qwen3.6/Qwen3.6-35B-A3B"
# DFly and DFlash share the prepared-data format, so reuse it by default.
DATA_DIR="${DATA_DIR:-$REPO/output/dflash_qwen3_6_35b_a3b_perfectblend_online_500k/data}"
RUN_DIR="$REPO/output/dfly_qwen3_6_35b_a3b_perfectblend_online_500k"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
TENSORBOARD_DIR="$RUN_DIR/tensorboard"
VLLM_PORT="${VLLM_PORT:-8100}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
# Train all 16 official block positions. Runtime block-4/block-8 benchmarks
# then use trained prefixes instead of extrapolating an 8-position checkpoint.
BLOCK_SIZE="${BLOCK_SIZE:-16}"
# z-lab's official target_layer_ids are 0-based transformer-layer indices:
#   1 6 11 16 22 27 32 37
# It reads hidden_states[layer_id + 1], whereas speculators/vLLM captures the
# hidden-state index directly, so the CLI values below must all be offset by 1.
TARGET_LAYER_IDS=(2 7 12 17 23 28 33 38)
JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-$$}}"
HIDDEN_STATES_DIR="${TMPDIR:-/tmp}/dfly_qwen3_6_35b_a3b_${JOB_TAG}_hidden_states"
VLLM_LOG="$RUN_DIR/vllm_${JOB_TAG}.log"

SPEC_PYTHON="$REPO/speculators_venv/bin/python"
TORCHRUN="$REPO/speculators_venv/bin/torchrun"
VLLM_PYTHON="$REPO/vllm_venv/bin/python"

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
# TP=1 vLLM data-parallel replica; the other six train the dense draft model.
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

echo "Repository:     $REPO"
echo "Model:          $MODEL"
echo "Data:           $DATA_DIR"
echo "Checkpoints:    $CHECKPOINT_DIR"
echo "TensorBoard:    $TENSORBOARD_DIR/$JOB_TAG"
echo "vLLM GPUs:      $VLLM_GPUS"
echo "Training GPUs:  $TRAIN_GPUS"
echo "vLLM log:       $VLLM_LOG"

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
    PYTHONUNBUFFERED=1 \
    "$VLLM_PYTHON" "$REPO/scripts/launch_vllm.py" "$MODEL" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    -- \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --max-model-len 8200 \
    --gpu-memory-utilization 0.92 \
    --port "$VLLM_PORT" \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM on port $VLLM_PORT (PID/PGID $VLLM_PID)..."
deadline=$((SECONDS + 1800))
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
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
echo "=== Launching DFly training ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    PYTHONUNBUFFERED=1 \
    "$TORCHRUN" \
    --standalone \
    --nproc_per_node 6 \
    "$REPO/scripts/train.py" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_DIR" \
    --save-path "$CHECKPOINT_DIR" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs 1 \
    --train-data-ratio 0.98 \
    --optimizer adamw \
    --lr 6e-4 \
    --weight-decay 0.01 \
    --noise-std 0 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.04 \
    --total-seq-len 4096 \
    --speculator-type dfly \
    --enable-hidden-correction \
    --block-size "$BLOCK_SIZE" \
    --max-anchors 1024 \
    --num-layers 6 \
    --sliding-window 2048 \
    --full-attention-indices 5 \
    --sliding-window-non-causal \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --request-timeout 300 \
    --on-missing generate \
    --on-generate delete \
    --logger tensorboard \
    --log-dir "$TENSORBOARD_DIR" \
    --run-name "$JOB_TAG" &
TRAIN_PID=$!

# Keep the cluster job attached to training. Its stdout/stderr is therefore
# visible in the cluster job log. On completion or cancellation, cleanup stops
# the full vLLM process group and removes transient hidden-state files.
wait "$TRAIN_PID"
