#!/usr/bin/env bash
# v1.1 Token-Latent SSM online training recipe for Qwen3.6-35B-A3B.
# Two GPUs serve verifier hidden states and six GPUs train the five-layer draft.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
WS="${WS:-$(dirname -- "$REPO")}"
ENV_REPO="${ENV_REPO:-$WS/speculators}"

MODEL="${MODEL:-$WS/model_weights/Qwen--Qwen3.6-35B-A3B}"
DATA_DIR="${DATA_DIR:-$WS/datasets/qwen3.6-35b-a3b/qwen3.6-35b-a3b_train_spec_800k_len4096_fullvocab}"
RUN_DIR="${RUN_DIR:-$WS/model_weights/token_latent_ssm_qwen3_6_35b_a3b_5swa}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_DIR/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_DIR}"

VLLM_PORT="${VLLM_PORT:-8500}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:${VLLM_PORT}/v1}"
VLLM_HEALTH_ENDPOINT="${VLLM_HEALTH_ENDPOINT:-http://localhost:${VLLM_PORT}/health}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

BLOCK_SIZE="${BLOCK_SIZE:-8}"
MAX_ANCHORS="${MAX_ANCHORS:-512}"
TOKEN_CODE_DIM="${TOKEN_CODE_DIM:-128}"
SSM_STATE_DIM="${SSM_STATE_DIM:-256}"
HIDDEN_CANDIDATE_COUNT="${HIDDEN_CANDIDATE_COUNT:-32}"
TRANSITION_CANDIDATE_COUNT="${TRANSITION_CANDIDATE_COUNT:-32}"
TRAINING_NEGATIVE_COUNT="${TRAINING_NEGATIVE_COUNT:-128}"
RETRIEVAL_LOSS_WEIGHT="${RETRIEVAL_LOSS_WEIGHT:-0.5}"
CONDITIONAL_LOSS_WEIGHT="${CONDITIONAL_LOSS_WEIGHT:-1.0}"
DPACE_ALPHA="${DPACE_ALPHA:-0.5}"

WANDB_PROJECT="${WANDB_PROJECT:-qwen3.6-35b-a3b-5swa}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$WS/.secrets/wandb_key}"
JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-$$}}"
VLLM_LOG="${VLLM_LOG:-$RUN_DIR/vllm_${JOB_TAG}.log}"

SPEC_PYTHON="${SPEC_PYTHON:-$ENV_REPO/speculators_venv/bin/python}"
TORCHRUN="${TORCHRUN:-$ENV_REPO/speculators_venv/bin/torchrun}"
VLLM_PYTHON="${VLLM_PYTHON:-$ENV_REPO/vllm_venv/bin/python}"
LAUNCH_VLLM="${LAUNCH_VLLM:-$REPO/scripts/launch_vllm.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO/scripts/train.py}"
PYTHONPATH_LOCAL="${PYTHONPATH_LOCAL:-$REPO/src:$REPO/hs_connectors/src}"

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-}"
HIDDEN_STATES_DIR_OWNED=0

for executable in "$SPEC_PYTHON" "$TORCHRUN" "$VLLM_PYTHON"; do
    [[ -x "$executable" ]] || {
        echo "Missing executable: $executable" >&2
        exit 1
    }
done

for path in \
    "$MODEL/config.json" \
    "$DATA_DIR/state.json" \
    "$DATA_DIR/dataset_info.json" \
    "$LAUNCH_VLLM" \
    "$TRAIN_SCRIPT"; do
    [[ -f "$path" ]] || {
        echo "Missing required file: $path" >&2
        exit 1
    }
done

for command in setsid curl; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "Missing command: $command" >&2
        exit 1
    }
done

exec 9>"$RUN_DIR/training.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
    echo "Another training job already holds $RUN_DIR/training.lock" >&2
    exit 1
fi

if [[ -f "$WANDB_KEY_FILE" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
    [[ -z "$WANDB_API_KEY" ]] || export WANDB_API_KEY
fi

ALLOCATED_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "$ALLOCATED_GPUS"
if (( ${#GPU_LIST[@]} != 8 )); then
    echo "Expected exactly 8 GPUs, got: $ALLOCATED_GPUS" >&2
    exit 1
fi
VLLM_GPUS="$(IFS=,; printf '%s' "${GPU_LIST[*]:0:2}")"
TRAIN_GPUS="$(IFS=,; printf '%s' "${GPU_LIST[*]:2:6}")"

VLLM_PID=""
TRAIN_PID=""

terminate_group() {
    local pgid="$1"
    [[ -n "$pgid" ]] || return 0
    kill -0 -- "-$pgid" 2>/dev/null || return 0
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in {1..30}; do
        kill -0 -- "-$pgid" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL -- "-$pgid" 2>/dev/null || true
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    terminate_group "$TRAIN_PID"
    terminate_group "$VLLM_PID"
    if (( HIDDEN_STATES_DIR_OWNED == 1 )) && [[ "$HIDDEN_STATES_DIR" == /tmp/token_latent_ssm_hidden_states.* ]]; then
        rm -rf -- "$HIDDEN_STATES_DIR"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -z "$HIDDEN_STATES_DIR" ]]; then
    HIDDEN_STATES_DIR="$(mktemp -d /tmp/token_latent_ssm_hidden_states.XXXXXX)"
    HIDDEN_STATES_DIR_OWNED=1
else
    if [[ "$HIDDEN_STATES_DIR" != /tmp/* ]]; then
        echo "HIDDEN_STATES_DIR must be under /tmp, got: $HIDDEN_STATES_DIR" >&2
        exit 1
    fi
    mkdir -p "$HIDDEN_STATES_DIR"
fi

echo "=== Token-Latent SSM v1.1 training configuration ==="
echo "Model:                  $MODEL"
echo "Data:                   $DATA_DIR"
echo "Run directory:          $RUN_DIR"
echo "vLLM GPUs:              $VLLM_GPUS"
echo "Training GPUs:          $TRAIN_GPUS"
echo "Hidden states:          $HIDDEN_STATES_DIR"
echo "Token code / SSM state: $TOKEN_CODE_DIM / $SSM_STATE_DIM"
echo "Candidates:             hidden=$HIDDEN_CANDIDATE_COUNT transition=$TRANSITION_CANDIDATE_COUNT"
echo "Training negatives:     $TRAINING_NEGATIVE_COUNT"
echo "Loss weights:           retrieval=$RETRIEVAL_LOSS_WEIGHT conditional=$CONDITIONAL_LOSS_WEIGHT"
echo "D-PACE alpha:           $DPACE_ALPHA"
echo "HF endpoint:            $HF_ENDPOINT"
echo "vLLM log:               $VLLM_LOG"

"$SPEC_PYTHON" - "$VLLM_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1)
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        raise SystemExit(f"Port {port} is already in use")
PY

echo "=== Launching verifier hidden-state server ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
    HF_ENDPOINT="$HF_ENDPOINT" \
    PYTHONPATH="$PYTHONPATH_LOCAL" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
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
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM on port $VLLM_PORT (PID/PGID $VLLM_PID)..."
deadline=$((SECONDS + 1800))
until curl -sf "$VLLM_HEALTH_ENDPOINT" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming healthy. Last log lines:" >&2
        tail -n 100 "$VLLM_LOG" >&2 || true
        wait "$VLLM_PID" || true
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for vLLM. See $VLLM_LOG" >&2
        exit 1
    fi
    sleep 2
done

echo "vLLM is healthy."
echo "=== Launching Token-Latent SSM v1.1 training ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    HF_ENDPOINT="$HF_ENDPOINT" \
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
    --speculator-type token_latent_ssm \
    --draft-attn-impl simple_flex_attention \
    --block-size "$BLOCK_SIZE" \
    --no-sample-from-anchor \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers 5 \
    --sliding-window 2048 \
    --sliding-window-non-causal \
    --target-layer-ids 2 11 20 29 38 \
    --loss-fn ce \
    --per-position-loss-weight dpace \
    --dpace-alpha "$DPACE_ALPHA" \
    --token-code-dim "$TOKEN_CODE_DIM" \
    --ssm-state-dim "$SSM_STATE_DIM" \
    --hidden-candidate-count "$HIDDEN_CANDIDATE_COUNT" \
    --transition-candidate-count "$TRANSITION_CANDIDATE_COUNT" \
    --training-negative-count "$TRAINING_NEGATIVE_COUNT" \
    --retrieval-loss-weight "$RETRIEVAL_LOSS_WEIGHT" \
    --conditional-loss-weight "$CONDITIONAL_LOSS_WEIGHT" \
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
    --run-name token_latent_ssm_qwen3_6_35b_a3b_v1_1 &
TRAIN_PID=$!

wait "$TRAIN_PID"
