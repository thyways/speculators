#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

export REPO="${REPO:-$DEFAULT_REPO}"
export ROOT="${ROOT:-$(dirname -- "$REPO")}"
export ENV_REPO="${ENV_REPO:-$ROOT/speculators}"

MODEL="${MODEL:-$ROOT/model_weights/Qwen/Qwen3.6-35B-A3B}"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/qwen3_6_35b_500k}"
export RUN_DIR="${RUN_DIR:-$ROOT/model_weights/kv_native_dflash_qwen3_6_35b_a3b_dual_stream_raw_kv_final_5full}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_DIR/checkpoints}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$RUN_DIR/tensorboard}"

VLLM_PORT="${VLLM_PORT:-8200}"
VLLM_GPU_COUNT="${VLLM_GPU_COUNT:-2}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
TOTAL_SEQ_LEN="${TOTAL_SEQ_LEN:-4096}"
MAX_ANCHORS="${MAX_ANCHORS:-1024}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-10000}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:${VLLM_PORT}/v1}"
# The online verifier is launched with --speculative_config method
# extract_hidden_states, which vLLM's V2 model runner rejects outright (it
# whitelists eagle/eagle3/mtp/dflash/dspark only). Datagen therefore has to run
# on the V1 runner; only the serving/eval scripts can use V2.
VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"

JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-$$}}"
RUN_NAME="${RUN_NAME:-dual_stream_raw_kv_final_5full_b${BLOCK_SIZE}_a${MAX_ANCHORS}}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-/tmp/dual_stream_raw_kv_final_qwen3_6_35b_hidden_states}"
VLLM_LOG="${VLLM_LOG:-$RUN_DIR/vllm_${JOB_TAG}.log}"

SPEC_PYTHON="${SPEC_PYTHON:-$ENV_REPO/speculators_venv/bin/python}"
TORCHRUN="${TORCHRUN:-$ENV_REPO/speculators_venv/bin/torchrun}"
VLLM_PYTHON="${VLLM_PYTHON:-$ENV_REPO/vllm_venv/bin/python}"
LAUNCH_VLLM="${LAUNCH_VLLM:-$REPO/scripts/launch_vllm.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO/scripts/train.py}"
LOCAL_PYTHONPATH="${LOCAL_PYTHONPATH:-$REPO/src:$REPO/hs_connectors/src}"

for executable in "$SPEC_PYTHON" "$TORCHRUN" "$VLLM_PYTHON"; do
    if [[ ! -x "$executable" ]]; then
        echo "Missing executable: $executable" >&2
        exit 1
    fi
done

for path in \
    "$MODEL/config.json" \
    "$DATA_DIR/state.json" \
    "$DATA_DIR/dataset_info.json" \
    "$DATA_DIR/token_freq.pt" \
    "$DATA_DIR/d2t.npy" \
    "$DATA_DIR/t2d.npy"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required artifact: $path" >&2
        exit 1
    fi
done

for command in setsid curl flock; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "The runtime image must provide '$command'." >&2
        exit 1
    fi
done

if (( BLOCK_SIZE < 2 )); then
    echo "BLOCK_SIZE must be at least 2" >&2
    exit 1
fi
if (( NUM_SPECULATIVE_TOKENS != BLOCK_SIZE - 1 )); then
    echo "NUM_SPECULATIVE_TOKENS must equal BLOCK_SIZE-1" >&2
    exit 1
fi
if [[ "$VLLM_USE_V2_MODEL_RUNNER" != "0" ]]; then
    echo "VLLM_USE_V2_MODEL_RUNNER must be 0: the datagen verifier runs with" \
        "speculative method extract_hidden_states, which the V2 model runner" \
        "does not support." >&2
    exit 1
fi
if (( VLLM_MAX_MODEL_LEN < TOTAL_SEQ_LEN )); then
    echo "VLLM_MAX_MODEL_LEN must be at least TOTAL_SEQ_LEN" >&2
    exit 1
fi
ALLOCATED_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "$ALLOCATED_GPUS"
if (( ${#GPU_LIST[@]} != 8 )); then
    echo "Expected exactly 8 allocated GPUs, got: $ALLOCATED_GPUS" >&2
    exit 1
fi
if (( VLLM_GPU_COUNT < 1 || VLLM_GPU_COUNT >= ${#GPU_LIST[@]} )); then
    echo "VLLM_GPU_COUNT must be in [1, 7], got $VLLM_GPU_COUNT" >&2
    exit 1
fi
TRAIN_GPU_COUNT=$(( ${#GPU_LIST[@]} - VLLM_GPU_COUNT ))
VLLM_GPUS=$(IFS=,; printf '%s' "${GPU_LIST[*]:0:VLLM_GPU_COUNT}")
TRAIN_GPUS=$(IFS=,; printf '%s' "${GPU_LIST[*]:VLLM_GPU_COUNT}")

VLLM_CMD=(
    "$VLLM_PYTHON"
    "$LAUNCH_VLLM"
    "$MODEL"
    --hidden-states-backend file
    --hidden-states-path "$HIDDEN_STATES_DIR"
    --verifier-kv-layer-ids 3 11 19 27 35
    --
    --host 127.0.0.1
    --port "$VLLM_PORT"
    --tensor-parallel-size 1
    --data-parallel-size "$VLLM_GPU_COUNT"
    --dtype bfloat16
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --gpu-memory-utilization 0.92
    --enable-chunked-prefill
)

TRAIN_ARGS=(
    --verifier-name-or-path "$MODEL"
    --data-path "$DATA_DIR"
    --hidden-states-backend file
    --hidden-states-path "$HIDDEN_STATES_DIR"
    --save-path "$CHECKPOINT_DIR"
    --speculator-type kv_native_dflash
    --verifier-kv-layer-ids 3 11 19 27 35
    --verifier-kv-layer-mapping 3 11 19 27 35
    --verifier-num-key-value-heads 2
    --verifier-head-dim 256
    --num-speculative-tokens "$NUM_SPECULATIVE_TOKENS"
    --draft-vocab-size 32000
    --num-layers 5
    --draft-arch qwen3
    --draft-hidden-act silu
    --draft-attn-impl simple_flex_attention
    --block-size "$BLOCK_SIZE"
    --no-sample-from-anchor
    --full-attention-indices 0 1 2 3 4
    --loss-fn ce
    --per-position-loss-weight dpace
    --optimizer adamw
    --lr 6e-4
    --weight-decay 0.01
    --scheduler-type cosine
    --scheduler-warmup-ratio 0.04
    --epochs 1
    --checkpoint-freq 0.1
    --train-data-ratio 0.98
    --total-seq-len "$TOTAL_SEQ_LEN"
    --max-anchors "$MAX_ANCHORS"
    --noise-std 0
    --hidden-states-dtype bfloat16
    --vllm-endpoint "$VLLM_ENDPOINT"
    --request-timeout 300
    --max-retries 3
    --on-missing generate
    --on-generate delete
    --logger tensorboard
    --log-dir "$TENSORBOARD_DIR"
    --run-name "$RUN_NAME"
    --seed 42
)

TRAIN_CMD=(
    "$TORCHRUN"
    --standalone
    --nproc_per_node "$TRAIN_GPU_COUNT"
    "$TRAIN_SCRIPT"
    "${TRAIN_ARGS[@]}"
)

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    echo "=== vLLM command (TP=1 per DP replica) ==="
    printf 'env CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q PYTHONUNBUFFERED=1 VLLM_USE_V2_MODEL_RUNNER=%q ' \
        "$VLLM_GPUS" "$LOCAL_PYTHONPATH" "$VLLM_USE_V2_MODEL_RUNNER"
    print_command "${VLLM_CMD[@]}"
    echo "=== training command ==="
    printf 'env CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q PYTHONUNBUFFERED=1 ' \
        "$TRAIN_GPUS" "$LOCAL_PYTHONPATH"
    print_command "${TRAIN_CMD[@]}"
    exit 0
fi

env PYTHONPATH="$LOCAL_PYTHONPATH" "$SPEC_PYTHON" - "${TRAIN_ARGS[@]}" <<'PY'
import sys

from speculators.train.config import TrainConfig

config = TrainConfig.resolve(sys.argv[1:])
expected_full = list(range(5))
if config.speculator_type != "kv_native_dflash":
    raise SystemExit("Expected --speculator-type kv_native_dflash")
if config.draft.num_layers != 5:
    raise SystemExit("Expected exactly five draft layers")
if config.draft.full_attention_indices != expected_full:
    raise SystemExit("All five draft layers must use full attention")
if config.dflash.sample_from_anchor is not False:
    raise SystemExit("KV-native DFlash must use --no-sample-from-anchor")
if config.loss.loss_fn != "ce":
    raise SystemExit("KV-native DFlash recipe requires CE")
if config.dflash.per_position_loss_weight != "dpace":
    raise SystemExit("KV-native DFlash recipe requires D-PACE")
print(
    "Config validation passed:",
    f"type={config.speculator_type}",
    f"layers={config.draft.num_layers}",
    f"full={config.draft.full_attention_indices}",
    f"seq={config.data.total_seq_len}",
    f"anchors={config.data.max_anchors}",
    f"block={config.dflash.block_size}",
    f"spec_tokens={config.kv_native_dflash.num_speculative_tokens}",
    "architecture=dual_stream_raw_kv",
    f"per_position_loss_weight={config.dflash.per_position_loss_weight}",
)
PY

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$TENSORBOARD_DIR"

exec 9>"$RUN_DIR/training.lock"
if ! flock -n 9; then
    echo "Another job already holds $RUN_DIR/training.lock" >&2
    exit 1
fi

if [[ -d "$HIDDEN_STATES_DIR" ]] \
    && find "$HIDDEN_STATES_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "Hidden-state directory is not empty: $HIDDEN_STATES_DIR" >&2
    exit 1
fi
mkdir -p "$HIDDEN_STATES_DIR"

"$SPEC_PYTHON" - "$VLLM_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1)
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        raise SystemExit(f"Port {port} is already in use")
PY

if [[ "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    env CUDA_VISIBLE_DEVICES="$ALLOCATED_GPUS" \
        "$SPEC_PYTHON" - "${MIN_FREE_GPU_GIB:-70}" <<'PY'
import sys

import torch

minimum = float(sys.argv[1]) * 1024**3
failures = []
for index in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(index)
    props = torch.cuda.get_device_properties(index)
    print(
        f"visible GPU {index}: {props.name}, "
        f"free={free / 1024**3:.1f} GiB/{total / 1024**3:.1f} GiB"
    )
    if free < minimum:
        failures.append(index)
if failures:
    raise SystemExit(f"Allocated GPUs are still busy: {failures}")
PY
fi

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
    case "$HIDDEN_STATES_DIR" in
        */dual_stream_raw_kv_final_qwen3_6_35b_hidden_states)
            if [[ -d "$HIDDEN_STATES_DIR" ]]; then
                find "$HIDDEN_STATES_DIR" -mindepth 1 -delete
                rmdir "$HIDDEN_STATES_DIR" 2>/dev/null || true
            fi
            ;;
    esac
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Repository:          $REPO"
echo "Verifier:            $MODEL"
echo "Data:                $DATA_DIR"
echo "Draft:               Dual-stream raw-KV DFlash, 5 full-attention layers"
echo "Sequence/anchors:    $TOTAL_SEQ_LEN / $MAX_ANCHORS"
echo "Block/spec tokens:   $BLOCK_SIZE / $NUM_SPECULATIVE_TOKENS"
echo "Verifier max length: $VLLM_MAX_MODEL_LEN"
echo "Verifier KV layers:  3 11 19 27 35"
echo "Checkpoints:         $CHECKPOINT_DIR"
echo "Step budget:         one full epoch"
echo "vLLM GPUs (DP/TP):   $VLLM_GPUS (${VLLM_GPU_COUNT}/1)"
echo "vLLM model runner:    V$((VLLM_USE_V2_MODEL_RUNNER + 1))"
echo "Training GPUs:       $TRAIN_GPUS ($TRAIN_GPU_COUNT ranks)"
echo "vLLM log:            $VLLM_LOG"

echo "=== Launching online verifier ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
    PYTHONPATH="$LOCAL_PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    VLLM_USE_V2_MODEL_RUNNER="$VLLM_USE_V2_MODEL_RUNNER" \
    "${VLLM_CMD[@]}" \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM on port $VLLM_PORT (PID/PGID $VLLM_PID)..."
deadline=$((SECONDS + 1800))
until curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming healthy. Last log lines:" >&2
        tail -n 120 "$VLLM_LOG" >&2 || true
        wait "$VLLM_PID" || true
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for vLLM. See $VLLM_LOG" >&2
        exit 1
    fi
    sleep 2
done

echo "vLLM is healthy. Starting from-scratch KV-native DFlash training."
setsid env \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    PYTHONPATH="$LOCAL_PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    "${TRAIN_CMD[@]}" &
TRAIN_PID=$!

wait "$TRAIN_PID"
