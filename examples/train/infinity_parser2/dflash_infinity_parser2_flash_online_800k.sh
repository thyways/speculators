#!/usr/bin/env bash
# Online DFlash training for Infinity-Parser2.1-Flash on the prepared v2.1 corpus
# (1443776 records -> 1803882 assistant-turn rows, full verifier vocab).
# --total-seq-len must stay >= the prepare --seq-length (16384): the packing
# sampler cannot batch a row longer than its budget.

set -Eeuo pipefail

ROOT="/inspire/sfs/project/inf-multimodal/public/wumengke"
REPO="$ROOT/speculators"
MODEL="/home/ma-user/work/data_mllm/publish_models/Infinity-Parser2.1-Flash-2608"
DATA_DIR="$ROOT/datasets/infinity_parsers2_v2_1/dflash_data/full"
RUN_DIR="$REPO/output/dflash_infinity_parser2_flash_online_800k"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
LOG_DIR="${LOG_DIR:-$RUN_DIR}"
WANDB_PROJECT="${WANDB_PROJECT:-infinity-parser2-flash}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$ROOT/.secrets/wandb_key}"
MEDIA_ROOT="/inspire/sfs/project/inf-multimodal/public"
VLLM_PORT="${VLLM_PORT:-8100}"
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
VLLM_GPUS=$(IFS=,; printf '%s' "${GPU_LIST[*]:0:4}")
TRAIN_GPUS=$(IFS=,; printf '%s' "${GPU_LIST[*]:4:4}")

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
        */parser2_flash_*) rm -rf -- "$HIDDEN_STATES_DIR" ;;
    esac
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

HIDDEN_STATES_DIR="$(mktemp -d "${TMPDIR:-/tmp}/parser2_flash_${JOB_TAG}.XXXXXX")"

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

setsid env \
    CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
    PYTHONUNBUFFERED=1 \
    "$VLLM_PYTHON" "$REPO/scripts/launch_vllm.py" "$MODEL" \
    --target-layer-ids 2 7 12 17 22 \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max_model_len 131072 \
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
    --nproc_per_node 4 \
    "$REPO/scripts/train.py" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_DIR" \
    --save-path "$CHECKPOINT_DIR" \
    --speculator-type dflash \
    --draft-arch qwen3 \
    --draft-hidden-act silu \
    --num-layers 5 \
    --mask-token-id 248077 \
    --block-size 16 \
    --max-anchors 3072 \
    --target-layer-ids 2 7 12 17 22 \
    --draft-mrope-full-head-hack \
    --sliding-window 2048 \
    --sliding-window-non-causal \
    --draft-attn-impl simple_flex_attention \
    --no-sample-from-anchor \
    --loss-fn kl_div \
    --dflash-decay-gamma 4.0 \
    --optimizer adamw \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.04 \
    --epochs 2 \
    --total-seq-len 16384 \
    --train-data-ratio 0.99 \
    --noise-std 0 \
    --hidden-states-dtype bfloat16 \
    --prefetch-factor 2 \
    --hidden-states-backend file \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --vllm-endpoint "http://127.0.0.1:${VLLM_PORT}/v1" \
    --on-missing generate \
    --on-generate delete \
    --request-timeout 600 \
    --max-retries 5 \
    --fail-on-hidden-state-error \
    --seed 42 \
    --logger wandb \
    --log-dir "$LOG_DIR" \
    --checkpoint-freq 0.1 \
    --run-name "$JOB_TAG" &
TRAIN_PID=$!

wait "$TRAIN_PID"
