#!/usr/bin/env bash
# Parallel Token-Latent Feedback (方案设计 v1.2) online training recipe.
# The feedback stage is block-parallel: one packed projection, one strict-prefix
# latent matmul, and one up-projection are added after the DFlash backbone.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
WS="${WS:-$(dirname -- "$REPO")}"
ENV_REPO="${ENV_REPO:-$WS/speculators}"
MODEL="${MODEL:-$WS/model_weights/Qwen--Qwen3.6-35B-A3B}"
DATA_DIR="${DATA_DIR:-$WS/datasets/qwen3.6-35b-a3b/qwen3.6-35b-a3b_train_spec_800k_len4096_fullvocab}"
RUN_DIR="${RUN_DIR:-$WS/model_weights/token_latent_feedback_qwen3_6_35b_a3b_5swa}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_DIR/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_DIR}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-/tmp/token_latent_feedback_qwen3_6_35b_a3b_hidden_states}"
VLLM_PORT="${VLLM_PORT:-8600}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:${VLLM_PORT}/v1}"
VLLM_HEALTH_ENDPOINT="${VLLM_HEALTH_ENDPOINT:-http://localhost:${VLLM_PORT}/health}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen3.6-35b-a3b-5swa}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$WS/.secrets/wandb_key}"

SPEC_PYTHON="${SPEC_PYTHON:-$ENV_REPO/speculators_venv/bin/python}"
TORCHRUN="${TORCHRUN:-$ENV_REPO/speculators_venv/bin/torchrun}"
VLLM_PYTHON="${VLLM_PYTHON:-$ENV_REPO/vllm_venv/bin/python}"
LAUNCH_VLLM="${LAUNCH_VLLM:-$REPO/scripts/launch_vllm.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO/scripts/train.py}"
PYTHONPATH_LOCAL="${PYTHONPATH_LOCAL:-$REPO/src:$REPO/hs_connectors/src}"

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$LOG_DIR" "$HIDDEN_STATES_DIR"
for executable in "$SPEC_PYTHON" "$TORCHRUN" "$VLLM_PYTHON"; do
    [[ -x "$executable" ]] || { echo "Missing executable: $executable" >&2; exit 1; }
done
[[ -f "$MODEL/config.json" ]] || { echo "Missing model config: $MODEL/config.json" >&2; exit 1; }
[[ -f "$DATA_DIR/state.json" ]] || { echo "Missing data state: $DATA_DIR/state.json" >&2; exit 1; }
[[ -f "$DATA_DIR/dataset_info.json" ]] || { echo "Missing data info: $DATA_DIR/dataset_info.json" >&2; exit 1; }

ALLOCATED_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "$ALLOCATED_GPUS"
(( ${#GPU_LIST[@]} == 8 )) || { echo "Expected 8 GPUs, got $ALLOCATED_GPUS" >&2; exit 1; }
VLLM_GPUS="$(IFS=,; printf '%s' "${GPU_LIST[*]:0:2}")"
TRAIN_GPUS="$(IFS=,; printf '%s' "${GPU_LIST[*]:2:6}")"

VLLM_PID=""
TRAIN_PID=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    [[ -z "$TRAIN_PID" ]] || kill -TERM -- "-$TRAIN_PID" 2>/dev/null || true
    [[ -z "$VLLM_PID" ]] || kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
    rm -rf "$HIDDEN_STATES_DIR"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -f "$WANDB_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi

echo "Model: $MODEL"
echo "Data: $DATA_DIR"
echo "Run: $RUN_DIR"
echo "vLLM GPUs: $VLLM_GPUS; training GPUs: $TRAIN_GPUS"
echo "Hidden states: $HIDDEN_STATES_DIR"

setsid env \
    CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
    PYTHONPATH="$PYTHONPATH_LOCAL" \
    PYTHONUNBUFFERED=1 \
    "$VLLM_PYTHON" \
    "$LAUNCH_VLLM" \
    "$MODEL" \
    --target-layer-ids 2 11 20 29 38 \
    --include-last-layer \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    -- \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.92 \
    --port "$VLLM_PORT" \
    >"$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

deadline=$((SECONDS + 1800))
until curl -sf "$VLLM_HEALTH_ENDPOINT" >/dev/null 2>&1; do
    kill -0 "$VLLM_PID" 2>/dev/null || { tail -n 100 "$RUN_DIR/vllm.log" >&2; exit 1; }
    (( SECONDS < deadline )) || { echo "Timed out waiting for vLLM" >&2; exit 1; }
    sleep 2
done

setsid env \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    PYTHONPATH="$PYTHONPATH_LOCAL" \
    PYTHONUNBUFFERED=1 \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_MODE="$WANDB_MODE" \
    "$TORCHRUN" \
    --standalone \
    --nproc_per_node 6 \
    "$TRAIN_SCRIPT" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --save-path "$CHECKPOINT_DIR" \
    --epochs 1 \
    --train-data-ratio 0.98 \
    --optimizer muon \
    --muon-lr 2e-4 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --noise-std 0 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.04 \
    --total-seq-len 4096 \
    --hidden-states-dtype bfloat16 \
    --speculator-type token_latent_feedback \
    --draft-attn-impl simple_flex_attention \
    --block-size 8 \
    --no-sample-from-anchor \
    --max-anchors 512 \
    --num-layers 5 \
    --sliding-window 2048 \
    --target-layer-ids 2 11 20 29 38 \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' \
    --per-position-loss-weight fixed-exp-decay \
    --latent-dim 128 \
    --feedback-stages 1 \
    --prefix-mixer-mode full \
    --latent-loss-alpha 0.1 \
    --hidden-states-backend file \
    --vllm-endpoint "$VLLM_ENDPOINT" \
    --request-timeout 300 \
    --max-retries 5 \
    --on-missing generate \
    --on-generate delete \
    --fail-on-hidden-state-error \
    --logger wandb \
    --log-dir "$LOG_DIR" \
    --checkpoint-freq 0.1 \
    --run-name token_latent_feedback_qwen3_6_35b_a3b &
TRAIN_PID=$!
wait "$TRAIN_PID"
