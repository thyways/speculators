#!/usr/bin/env bash
# Full-data online P-EAGLE training for Qwen3.6-35B-A3B on one 8-GPU node.
# Reuses the prepared PerfectBlend data, target-feature layout, six-layer draft
# shape, and vocabulary mapping from DFlash, while using P-EAGLE's parallel
# multi-depth prediction and COD sampling.
# Keeps the AdamW + cosine recipe (4% linear warmup, then cosine decay).
# Run this script as the cluster job command; do not wrap it in nohup.
# The prepared data in DATA_DIR must use this model's tokenizer.

set -Eeuo pipefail

REPO="/inspire/sfs/project/inf-multimodal/public/wumengke/speculators"
MODEL="/inspire/sfs/project/inf-multimodal/public/share_base_models/Qwen3.6/Qwen3.6-35B-A3B"
# P-EAGLE and DFlash use the same tokenized-data and vocabulary-map artifacts.
DATA_DIR="${DATA_DIR:-$REPO/output/dflash_qwen3_6_35b_a3b_perfectblend_online_500k/data}"
RUN_DIR="$REPO/output/peagle_qwen3_6_35b_a3b_perfectblend_online_500k"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
TENSORBOARD_DIR="$RUN_DIR/tensorboard"
VLLM_PORT="${VLLM_PORT:-8100}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
NUM_LAYERS=6
NUM_DEPTHS="${NUM_DEPTHS:-8}"
# COD sampling reduces deeper positions geometrically, so retain P-EAGLE's
# canonical 3072-anchor budget rather than DFlash block expansion's 1024.
MAX_ANCHORS="${MAX_ANCHORS:-3072}"
DOWN_SAMPLE_RATIO="${DOWN_SAMPLE_RATIO:-0.8}"
DOWN_SAMPLE_RATIO_MIN="${DOWN_SAMPLE_RATIO_MIN:-0.2}"
# Match DFlash's eight Qwen3.6 target hidden-state inputs exactly. The vLLM
# launcher appends layer 40 automatically as the verifier's final hidden state
# for the loss; only these eight auxiliary indices are passed to the trainer.
TARGET_LAYER_IDS=(2 7 12 17 23 28 33 38)
# Match DFlash's six-layer attention layout: layers 0-4 use SWA, layer 5 uses
# full attention.
FULL_ATTENTION_INDICES=(5)
JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-$$}}"
HIDDEN_STATES_DIR="${TMPDIR:-/tmp}/peagle_qwen3_6_35b_a3b_${JOB_TAG}_hidden_states"
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

echo "Repository:       $REPO"
echo "Model:            $MODEL"
echo "Data:             $DATA_DIR"
echo "Checkpoints:      $CHECKPOINT_DIR"
echo "TensorBoard:      $TENSORBOARD_DIR/$JOB_TAG"
echo "P-EAGLE layers:   $NUM_LAYERS"
echo "P-EAGLE depths:   $NUM_DEPTHS"
echo "Max anchors:      $MAX_ANCHORS"
echo "COD ratios:       $DOWN_SAMPLE_RATIO / $DOWN_SAMPLE_RATIO_MIN"
echo "Target layers:    ${TARGET_LAYER_IDS[*]}"
echo "Full-attn layers: ${FULL_ATTENTION_INDICES[*]}"
echo "vLLM GPUs:        $VLLM_GPUS"
echo "Training GPUs:    $TRAIN_GPUS"
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
echo "=== Launching P-EAGLE training ==="
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
    --speculator-type peagle \
    --draft-arch qwen3 \
    --num-layers "$NUM_LAYERS" \
    --num-depths "$NUM_DEPTHS" \
    --down-sample-ratio "$DOWN_SAMPLE_RATIO" \
    --down-sample-ratio-min "$DOWN_SAMPLE_RATIO_MIN" \
    --max-anchors "$MAX_ANCHORS" \
    --no-norm-before-residual \
    --sliding-window 2048 \
    --full-attention-indices "${FULL_ATTENTION_INDICES[@]}" \
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
