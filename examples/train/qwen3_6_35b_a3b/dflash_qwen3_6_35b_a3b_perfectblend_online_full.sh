#!/usr/bin/env bash
# Online DFlash Training Script (Qwen3.6-35B-A3B + open-perfectblend, full dataset)

set -Eeuo pipefail

COMMENT=""
while getopts "c:" opt; do
    case $opt in
        c) COMMENT="$OPTARG" ;;
        *) echo "Usage: $0 [-c comment]" >&2; exit 1 ;;
    esac
done

export WANDB_PROJECT="${WANDB_PROJECT:-speculators-qwen3_6-35b-a3b-perfectblend}"

# ============ Configuration ============
WS="/inspire/sfs/project/inf-multimodal/public/wumengke"
MODEL="$WS/model_weights/Qwen--Qwen3.6-35B-A3B"
OUTPUT_DIR="$WS/model_weights/dflash_qwen3_6-35b-a3b-perfectblend"
DATA_DIR="$WS/datasets/qwen3.6-35b-a3b/qwen3.6-35b-a3b_train_spec_len3072_fullvocab"
LOG_DIR="$OUTPUT_DIR/logs"
SEQ_LENGTH=3072                 # Data truncation length (prepare_data); fixed by the shared artifacts
PACK_SEQ_LEN=8192
EPOCHS=3
LR=1e-4
VLLM_PORT=8000
JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-$$}}"
HS_PATH="/tmp/hs_dflash_qwen3_6_35b_a3b_${JOB_TAG}"
VLLM_LOG="$OUTPUT_DIR/logs/vllm_${JOB_TAG}.log"

# Online training uses two GPUs for the verifier server and six for training.
VLLM_GPUS="0,1"
TRAIN_GPUS="2,3,4,5,6,7"
NUM_TRAIN_GPUS=6

# DFlash-specific parameters
SPECULATOR_TYPE="dflash"
NUM_LAYERS=5
FULL_ATTENTION_INDICES="0 1 2 3 4"
TARGET_LAYER_IDS="1 10 19 28 37"
BLOCK_SIZE="7"
MAX_ANCHORS="2048"
DECAY_GAMMA=7

SPEC_PYTHON="$WS/speculators/speculators_venv/bin/python"
TORCHRUN="$WS/speculators/speculators_venv/bin/torchrun"
VLLM_PYTHON="$WS/speculators/vllm_venv/bin/python"
LAUNCH_VLLM="$WS/speculators/scripts/launch_vllm.py"
TRAIN_SCRIPT="$WS/speculators/scripts/train.py"

mkdir -p "$OUTPUT_DIR/runs" "$OUTPUT_DIR/logs" "$HS_PATH"

echo "=== Step 2: Using prepared training data from $DATA_DIR ==="

# Step 3: Launch vLLM server in the background (generates hidden states on-the-fly)
# --target-layer-ids selects the auxiliary hidden-state layers; it must match
# the --target-layer-ids passed to the trainer below.
echo "=== Step 3: Launching vLLM server for hidden states ==="
echo "vLLM log: $VLLM_LOG"
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" "$VLLM_PYTHON" "$LAUNCH_VLLM" "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --hidden-states-path "$HS_PATH" \
    -- \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --max-model-len $((SEQ_LENGTH + 16)) \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.92 \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    rm -rf "$HS_PATH"
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
deadline=$((SECONDS + 1800))
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming healthy. Last log lines:" >&2
        tail -n 100 "$VLLM_LOG" >&2 || true
        wait "$VLLM_PID" || true
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out after 30 minutes waiting for vLLM. See $VLLM_LOG" >&2
        exit 1
    fi
    sleep 2
done
echo "vLLM server ready."

RUN_NAME="dflash-${COMMENT:-base}-$(date +%Y%m%d-%H%M%S)"
SAVE_DIR="$OUTPUT_DIR/runs/$RUN_NAME"
echo "=== Step 4: Training (max_anchors = $MAX_ANCHORS, stream $((MAX_ANCHORS * BLOCK_SIZE))) ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN" \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    "$TRAIN_SCRIPT" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_DIR" \
    --save-path "$SAVE_DIR" \
    --epochs "$EPOCHS" \
    --train-data-ratio 0.98 \
    --optimizer muon \
    --muon-lr 2e-4 \
    --lr "$LR" \
    --noise-std 0 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.01 \
    --total-seq-len "$PACK_SEQ_LEN" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --dflash-decay-gamma "$DECAY_GAMMA" \
    --loss-fn ce \
    --target-layer-ids $TARGET_LAYER_IDS \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --request-timeout 180 \
    --on-missing generate \
    --on-generate delete \
    --logger wandb \
    --log-dir "$LOG_DIR" \
    --checkpoint-freq 0.5 \
    --run-name "$RUN_NAME" \
    --hidden-states-backend file \
    --hidden-states-path "$HS_PATH" \
    --full-attention-indices $FULL_ATTENTION_INDICES

echo "Done. Checkpoints saved to $SAVE_DIR"
