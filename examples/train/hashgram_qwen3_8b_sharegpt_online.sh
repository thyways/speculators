#!/bin/bash
# Online HashGram training example.
#
# HashGram uses one DFlash backbone pass followed by a DSpark-style low-rank
# recall bias and hashed vector bigram/trigram candidate reranking. Trained
# checkpoints are served by the speculators_hashgram vLLM general plugin.
# For one inference node + one training node with Mooncake, use
# hashgram_qwen3_8b_sharegpt_2node_mooncake.sh instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="$(cd "$REPO_ROOT/.." && pwd)"
TRAIN_VENV="${TRAIN_VENV:-$REPO_ROOT/speculators_venv}"
VLLM_VENV="${VLLM_VENV:-$REPO_ROOT/vllm_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$TRAIN_VENV/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-$VLLM_VENV/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$TRAIN_VENV/bin/torchrun}"
VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-1800}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$ROOT/.secrets/wandb_key}"

# ============ Configuration ============
MODEL="/inspire/sfs/project/inf-multimodal/public/wumengke/model_weights/Qwen--Qwen3-8B"
DATA_DIR="/inspire/sfs/project/inf-multimodal/public/wumengke/datasets/jihwan1205--perfectblend-qwen3-8b-regen/800k-len8192"
OUTPUT_DIR="./output/hashgram_qwen3_8b_1node"
VLLM_PORT=8000
SEQ_LENGTH=8192
EPOCHS=3
TRAIN_DATA_RATIO=0.98
ADAMW_LR=1e-4
MUON_LR=2e-4
NOISE_STD=0
SCHEDULER_TYPE="cosine"
SCHEDULER_WARMUP_RATIO=0.04
CHECKPOINT_FREQ=0.1
LOGGER="wandb"
RUN_NAME="hashgram-qwen3-8b-perfectblend"
LOG_DIR="$OUTPUT_DIR/logs"
WANDB_PROJECT="${WANDB_PROJECT:-hashgram-qwen3-8b}"

SPECULATOR_TYPE="hashgram"
BLOCK_SIZE=8
MAX_ANCHORS=512
NUM_LAYERS=5
SLIDING_WINDOW=2048
TARGET_LAYER_IDS="1 9 17 25 33"

HASHGRAM_RANK=128
HASHGRAM_TOP_K=16
HASHGRAM_BIGRAM_BUCKETS=1048576
HASHGRAM_TRIGRAM_BUCKETS=1048576
HASHGRAM_NUM_HASHES=1
HASHGRAM_LOSS_ALPHA=1.0
HASHGRAM_MARKOV_RANK=256

VLLM_GPUS="0,1,2,3"
VLLM_DP_SIZE=4
TRAIN_GPUS="4,5,6,7"
NUM_TRAIN_GPUS=4
# =======================================

cd "$REPO_ROOT"

[[ -x "$TRAIN_PYTHON" ]] || {
  echo "Training Python is not executable: $TRAIN_PYTHON" >&2
  exit 1
}
[[ -x "$VLLM_PYTHON" ]] || {
  echo "vLLM Python is not executable: $VLLM_PYTHON" >&2
  exit 1
}
[[ -x "$TORCHRUN_BIN" ]] || {
  echo "torchrun is not executable: $TORCHRUN_BIN" >&2
  exit 1
}
"$TRAIN_PYTHON" -c 'import importlib.util as u; assert u.find_spec("speculators") and u.find_spec("wandb")' || {
  echo "Training environment must contain speculators and wandb: $TRAIN_VENV" >&2
  exit 1
}
"$VLLM_PYTHON" -c 'import importlib.util as u; assert u.find_spec("transformers") and u.find_spec("vllm")' || {
  echo "Inference environment must contain transformers and vllm: $VLLM_VENV" >&2
  exit 1
}
[[ -f "$MODEL/config.json" ]] || {
  echo "Missing model config: $MODEL/config.json" >&2
  exit 1
}
compgen -G "$DATA_DIR/*.arrow" > /dev/null || {
  echo "No prepared Arrow shards found in: $DATA_DIR" >&2
  exit 1
}
export WANDB_PROJECT
command -v curl >/dev/null 2>&1 || {
  echo "curl is required" >&2
  exit 1
}
command -v setsid >/dev/null 2>&1 || {
  echo "setsid is required" >&2
  exit 1
}
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<EOF
Resolved single-node HashGram topology:
  repo_root=$REPO_ROOT
  vllm_python=$VLLM_PYTHON
  train_python=$TRAIN_PYTHON
  torchrun=$TORCHRUN_BIN
  model=$MODEL
  data_dir=$DATA_DIR
  inference_gpus=$VLLM_GPUS
  inference_tp=1
  inference_dp=$VLLM_DP_SIZE
  training_gpus=$TRAIN_GPUS
  training_ddp=$NUM_TRAIN_GPUS
  logger=$LOGGER
  wandb_project=$WANDB_PROJECT
  wandb_key_file=$WANDB_KEY_FILE
EOF
  exit 0
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR/checkpoints"

VLLM_PID=""

terminate_process_group() {
  local process_id="${1:-}"
  [[ -n "$process_id" ]] || return 0
  kill -0 "$process_id" 2>/dev/null || return 0
  kill -TERM -- "-$process_id" 2>/dev/null || kill -TERM "$process_id" 2>/dev/null || true
  for _ in {1..30}; do
    kill -0 "$process_id" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-$process_id" 2>/dev/null || kill -KILL "$process_id" 2>/dev/null || true
}

load_wandb_api_key() {
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    return 0
  fi
  [[ -r "$WANDB_KEY_FILE" ]] || {
    echo "W&B key file is not readable: $WANDB_KEY_FILE" >&2
    return 1
  }
  local wandb_key=""
  IFS= read -r wandb_key < "$WANDB_KEY_FILE" || true
  [[ -n "$wandb_key" ]] || {
    echo "W&B key file is empty: $WANDB_KEY_FILE" >&2
    return 1
  }
  export WANDB_API_KEY="$wandb_key"
}

cleanup() {
  terminate_process_group "$VLLM_PID"
}
trap cleanup EXIT INT TERM

load_wandb_api_key

echo "=== Step 1: Launching vLLM server ==="
echo "vLLM log: $LOG_DIR/vllm.log"
setsid env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u ROLE_RANK -u MASTER_ADDR -u MASTER_PORT \
  -u WANDB_API_KEY \
  VLLM_PLUGINS="${VLLM_PLUGINS:-}" \
  CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
  "$VLLM_PYTHON" scripts/launch_vllm.py "$MODEL" \
  --target-layer-ids $TARGET_LAYER_IDS \
  -- \
  --port "$VLLM_PORT" \
  --tensor-parallel-size 1 \
  --data-parallel-size "$VLLM_DP_SIZE" \
  --data-parallel-backend mp \
  --api-server-count 1 \
  >"$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

vllm_deadline=$((SECONDS + VLLM_START_TIMEOUT))
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    wait "$VLLM_PID" || true
    tail -n 200 "$LOG_DIR/vllm.log" >&2 || true
    echo "vLLM exited before becoming healthy" >&2
    exit 1
  fi
  if (( SECONDS >= vllm_deadline )); then
    tail -n 200 "$LOG_DIR/vllm.log" >&2 || true
    echo "Timed out waiting for vLLM after ${VLLM_START_TIMEOUT}s" >&2
    exit 1
  fi
  sleep 2
done

echo "=== Step 2: Training HashGram ==="
echo "Training runs in the foreground; press Ctrl-C to stop."
train_status=0
env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u ROLE_RANK -u MASTER_ADDR -u MASTER_PORT \
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  "$TORCHRUN_BIN" \
  --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
  scripts/train.py \
  --verifier-name-or-path "$MODEL" \
  --data-path "$DATA_DIR" \
  --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
  --save-path "$OUTPUT_DIR/checkpoints" \
  --epochs "$EPOCHS" \
  --train-data-ratio "$TRAIN_DATA_RATIO" \
  --checkpoint-freq "$CHECKPOINT_FREQ" \
  --optimizer muon \
  --muon-lr "$MUON_LR" \
  --lr "$ADAMW_LR" \
  --noise-std "$NOISE_STD" \
  --scheduler-type "$SCHEDULER_TYPE" \
  --scheduler-warmup-ratio "$SCHEDULER_WARMUP_RATIO" \
  --total-seq-len "$SEQ_LENGTH" \
  --speculator-type "$SPECULATOR_TYPE" \
  --block-size "$BLOCK_SIZE" \
  --max-anchors "$MAX_ANCHORS" \
  --sample-from-anchor \
  --num-layers "$NUM_LAYERS" \
  --sliding-window "$SLIDING_WINDOW" \
  --sliding-window-non-causal \
  --target-layer-ids $TARGET_LAYER_IDS \
  --hashgram-rank "$HASHGRAM_RANK" \
  --hashgram-top-k "$HASHGRAM_TOP_K" \
  --hashgram-bigram-buckets "$HASHGRAM_BIGRAM_BUCKETS" \
  --hashgram-trigram-buckets "$HASHGRAM_TRIGRAM_BUCKETS" \
  --hashgram-num-hashes "$HASHGRAM_NUM_HASHES" \
  --hashgram-loss-alpha "$HASHGRAM_LOSS_ALPHA" \
  --hashgram-markov-rank "$HASHGRAM_MARKOV_RANK" \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --logger "$LOGGER" \
  --log-dir "$LOG_DIR" \
  --run-name "$RUN_NAME" \
  --on-missing generate \
  --on-generate delete || train_status=$?

if (( train_status != 0 )); then
  echo "HashGram training exited with status $train_status" >&2
  exit "$train_status"
fi

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
