#!/usr/bin/env bash
# KV-native DSpark trained FROM SCRATCH on Qwen3.6-35B-A3B (R0: 5 SWA + 1 Full).
#
# The verifier exports only its final hidden state plus six selected full-attention
# K/V layers. The draft and its identity-initialized KV-space adapters are trained
# jointly for one full pass over the 500K corpus.
#
# Run this as the cluster job command; do not wrap it in nohup. It does not
# manage or terminate pre-existing GPU jobs -- stop the GPU-hold script first.
# PRINT_ONLY=1 validates the config and prints both commands without launching.

set -Eeuo pipefail

ROOT="/inspire/sfs/project/inf-multimodal/public/wumengke"
REPO="$ROOT/speculators"
MODEL="$ROOT/model_weights/Qwen/Qwen3.6-35B-A3B"
DATA_DIR="$ROOT/datasets/qwen3_6_35b_500k"

RUN_DIR="${RUN_DIR:-$ROOT/model_weights/kv_native_dspark_qwen3_6_35b_a3b_perfectblend_online_500k_scratch}"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
TENSORBOARD_DIR="$RUN_DIR/tensorboard"

VLLM_PORT="${VLLM_PORT:-8200}"
VLLM_GPU_COUNT="${VLLM_GPU_COUNT:-2}"

# Empty MAX_STEPS means "one full pass over the corpus"; the cosine schedule
# then spans the real epoch length instead of a guessed horizon.
MAX_STEPS="${MAX_STEPS:-}"
DEFAULT_LOSS_FN='{"ce": 0.1, "tv": 0.9}'

IFS=' ' read -r -a VERIFIER_KV_LAYER_IDS \
    <<< "${VERIFIER_KV_LAYER_IDS:-3 11 19 27 35 39}"
IFS=' ' read -r -a VERIFIER_KV_LAYER_MAPPING \
    <<< "${VERIFIER_KV_LAYER_MAPPING:-${VERIFIER_KV_LAYER_IDS[*]}}"

JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-$$}}"
RUN_NAME="${RUN_NAME:-kv_native_scratch_r0_${JOB_TAG}}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-${TMPDIR:-/tmp}/\
kv_native_scratch_qwen3_6_35b_${JOB_TAG}_hidden_states}"
VLLM_LOG="$RUN_DIR/vllm_${JOB_TAG}.log"

SPEC_PYTHON="$REPO/speculators_venv/bin/python"
TORCHRUN="$REPO/speculators_venv/bin/torchrun"
VLLM_PYTHON="$REPO/vllm_venv/bin/python"
LOCAL_PYTHONPATH="$REPO/src:$REPO/hs_connectors/src"

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

if (( ${#VERIFIER_KV_LAYER_MAPPING[@]} != 6 )); then
    echo "VERIFIER_KV_LAYER_MAPPING must contain one ID per draft layer (6)" >&2
    exit 1
fi

# Preserve scheduler-provided device identifiers, including UUID lists. On a
# bare 8-GPU node, default to local indices 0..7. The first two visible devices
# host two DP replicas, each with TP=1; the remaining six train the draft.
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
    "$REPO/scripts/launch_vllm.py"
    "$MODEL"
    --hidden-states-backend file
    --hidden-states-path "$HIDDEN_STATES_DIR"
    --verifier-kv-layer-ids "${VERIFIER_KV_LAYER_IDS[@]}"
    --
    --host 127.0.0.1
    --port "$VLLM_PORT"
    --tensor-parallel-size 1
    --data-parallel-size "$VLLM_GPU_COUNT"
    --dtype bfloat16
    --max-model-len "${VLLM_MAX_MODEL_LEN:-4104}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
    --enable-chunked-prefill
)

TRAIN_ARGS=(
    --verifier-name-or-path "$MODEL"
    --data-path "$DATA_DIR"
    --hidden-states-backend file
    --hidden-states-path "$HIDDEN_STATES_DIR"
    --save-path "$CHECKPOINT_DIR"
    --speculator-type kv_native_dspark
    --verifier-kv-layer-ids "${VERIFIER_KV_LAYER_IDS[@]}"
    --verifier-kv-layer-mapping "${VERIFIER_KV_LAYER_MAPPING[@]}"
    --verifier-num-key-value-heads 2
    --verifier-head-dim 256
    --verifier-partial-rotary-factor 0.25
    --verifier-rope-theta 10000000
    --verifier-mrope-section 11 11 10
    --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS:-7}"
    --draft-vocab-size "${DRAFT_VOCAB_SIZE:-32000}"
    --num-layers 6
    --draft-arch qwen3
    --draft-hidden-act silu
    --draft-mrope-full-head-hack
    --draft-attn-impl simple_flex_attention
    --block-size "${BLOCK_SIZE:-8}"
    --sample-from-anchor
    --sliding-window 2048
    --full-attention-indices 5
    --sliding-window-non-causal
    --markov-rank "${MARKOV_RANK:-256}"
    --markov-head-type "${MARKOV_HEAD_TYPE:-vanilla}"
    --enable-confidence-head
    --confidence-head-with-markov
    --confidence-head-alpha "${CONFIDENCE_HEAD_ALPHA:-1.0}"
    --loss-fn "${LOSS_FN:-$DEFAULT_LOSS_FN}"
    --local-kv-loss-alpha "${LOCAL_KV_LOSS_ALPHA:-0.1}"
    --query-key-loss-alpha "${QUERY_KEY_LOSS_ALPHA:-1.0}"
    --attention-value-loss-alpha "${ATTENTION_VALUE_LOSS_ALPHA:-1.0}"
    --optimizer adamw
    --lr "${LR:-6e-4}"
    --weight-decay "${WEIGHT_DECAY:-0.01}"
    --scheduler-type cosine
    --scheduler-warmup-ratio "${WARMUP_RATIO:-0.04}"
    --epochs 1
    # In epochs. 0.1 keeps ten checkpoints during this single-epoch run.
    --checkpoint-freq "${CHECKPOINT_FREQ:-0.1}"
    --train-data-ratio "${TRAIN_DATA_RATIO:-0.98}"
    --total-seq-len "${TOTAL_SEQ_LEN:-4096}"
    --max-anchors "${MAX_ANCHORS:-1024}"
    --noise-std 0
    --hidden-states-dtype bfloat16
    --num-workers "${NUM_WORKERS:-8}"
    --prefetch-factor "${PREFETCH_FACTOR:-4}"
    --vllm-endpoint "http://127.0.0.1:${VLLM_PORT}/v1"
    --request-timeout "${REQUEST_TIMEOUT:-300}"
    --max-retries "${MAX_RETRIES:-3}"
    --on-missing generate
    --on-generate delete
    --logger tensorboard
    --log-dir "$TENSORBOARD_DIR"
    --run-name "$RUN_NAME"
    --log-freq "${LOG_FREQ:-10}"
    --seed "${SEED:-42}"
)
if [[ -n "$MAX_STEPS" ]]; then
    TRAIN_ARGS+=(
        --max-steps "$MAX_STEPS"
        --scheduler-total-steps "$MAX_STEPS"
    )
fi

TRAIN_CMD=(
    "$TORCHRUN"
    --standalone
    --nproc_per_node "$TRAIN_GPU_COUNT"
    "$REPO/scripts/train.py"
    "${TRAIN_ARGS[@]}"
)

# Parse the exact argument array with the real typed schema. This is read-only:
# it does not construct a model, contact vLLM, or start training.
env PYTHONPATH="$LOCAL_PYTHONPATH" "$SPEC_PYTHON" - "${TRAIN_ARGS[@]}" <<'PY'
import sys

from speculators.train.config import TrainConfig

config = TrainConfig.resolve(sys.argv[1:])
if config.draft.from_pretrained:
    raise SystemExit("This script must initialize KV-native DSpark from scratch")
print(
    "Config validation passed:",
    f"type={config.speculator_type}",
    f"mapping={config.kv_native_dspark.verifier_kv_layer_mapping}",
    f"max_steps={config.trainer.max_steps}",
)
PY

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    echo "=== vLLM command (TP=1 per DP replica) ==="
    printf 'env CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q PYTHONUNBUFFERED=1 ' \
        "$VLLM_GPUS" "$LOCAL_PYTHONPATH"
    print_command "${VLLM_CMD[@]}"
    echo "=== training command ==="
    printf 'env CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q PYTHONUNBUFFERED=1 ' \
        "$TRAIN_GPUS" "$LOCAL_PYTHONPATH"
    print_command "${TRAIN_CMD[@]}"
    exit 0
fi

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$TENSORBOARD_DIR"

# Prevent concurrent writers to the same checkpoint directory.
exec 9>"$RUN_DIR/training.lock"
if ! flock -n 9; then
    echo "Another job already holds $RUN_DIR/training.lock" >&2
    exit 1
fi

# Refuse to mix transient payloads from an earlier run. Cleanup below removes
# only this exact per-job directory.
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
    raise SystemExit(
        "Allocated GPUs are still busy: "
        f"{failures}. Release the GPU-hold job before starting training."
    )
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
    if [[ -d "$HIDDEN_STATES_DIR" ]]; then
        find "$HIDDEN_STATES_DIR" -mindepth 1 -delete
        rmdir "$HIDDEN_STATES_DIR" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Repository:          $REPO"
echo "Verifier:            $MODEL"
echo "Data:                $DATA_DIR"
echo "Init:                from scratch"
echo "KV layer mapping:    ${VERIFIER_KV_LAYER_MAPPING[*]}"
echo "Hidden state export: final layer only (teacher logits)"
echo "Checkpoints:         $CHECKPOINT_DIR (every ${CHECKPOINT_FREQ:-0.1} epoch)"
echo "TensorBoard:         $TENSORBOARD_DIR/$RUN_NAME"
echo "Step budget:         ${MAX_STEPS:-one full epoch}"
echo "vLLM GPUs (DP/TP):   $VLLM_GPUS (${VLLM_GPU_COUNT}/1)"
echo "Training GPUs:       $TRAIN_GPUS ($TRAIN_GPU_COUNT ranks)"
echo "Transient payloads:  $HIDDEN_STATES_DIR"
echo "vLLM log:            $VLLM_LOG"

echo "=== Launching online verifier ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
    PYTHONPATH="$LOCAL_PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
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

echo "vLLM is healthy. Starting from-scratch KV-native training."
setsid env \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    PYTHONPATH="$LOCAL_PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    "${TRAIN_CMD[@]}" &
TRAIN_PID=$!

# Keep the cluster job attached to torchrun. The EXIT trap stops vLLM and
# deletes all transient verifier hidden/K/V payloads on success or failure.
wait "$TRAIN_PID"
