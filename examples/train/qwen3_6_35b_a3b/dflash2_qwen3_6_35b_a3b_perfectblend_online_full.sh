#!/usr/bin/env bash
# DFlash2 on Qwen3.6-35B-A3B, PerfectBlend 500k, online hidden states.
#
# Same cluster shape and the same backbone recipe as
# dflash_qwen3_6_35b_a3b_perfectblend_online_full.sh -- 2 GPUs serving vLLM, 6
# training, 5 full-attention draft layers, block 16, D-PACE with cross-entropy --
# so a run of this script is comparable against that one. DFlash2 adds the local
# convolution and the candidate selector from
# https://github.com/vllm-project/vllm/pull/52816.
#
# Two things differ from the DFlash script, both forced by DFlash2's full-vocabulary
# requirement (the candidate selector emits the draft head's top-K ids directly as
# draft tokens, and the inference side applies no d2t remap to them):
#
#   * The data directory is re-exposed as a symlink view without d2t.npy / t2d.npy /
#     token_freq.pt. Those artifacts describe a 32000-entry pruned draft vocabulary
#     and train.py picks them up ahead of any flag, so with them present a DFlash2
#     run dies at startup on a draft-vocab mismatch. Without them train.py falls
#     back to the verifier's full vocabulary, which is what DFlash2 needs -- so this
#     script passes no --draft-vocab-size at all.
#   * --max-anchors is halved. The forward holds four
#     [max_anchors * block_size, vocab_size] tensors at peak (targets, the unary
#     logits, the selector bias, their sum). At Qwen3.6's 248320-entry vocabulary
#     that is 7.8x wider per tensor than the DFlash run's pruned 32000, so 1024
#     anchors would put ~30 GiB of logit activations on each training GPU.
#
# See docs/user_guide/algorithms/dflash2.md for the algorithm and for what the
# serving side still needs.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

export REPO="${REPO:-$DEFAULT_REPO}"
export ROOT="${ROOT:-$(dirname -- "$REPO")}"
export ENV_REPO="${ENV_REPO:-$ROOT/speculators}"

MODEL="${MODEL:-$ROOT/model_weights/Qwen/Qwen3.6-35B-A3B}"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/qwen3_6_35b_500k}"
export RUN_DIR="${RUN_DIR:-$ROOT/model_weights/dflash2_qwen3_6_35b_a3b_5full}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_DIR/checkpoints}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$RUN_DIR/tensorboard}"

# Full-vocabulary view of DATA_DIR (see the header). Symlinks, so it costs nothing.
FULL_VOCAB_DATA_DIR="${FULL_VOCAB_DATA_DIR:-$RUN_DIR/data_full_vocab}"

# Anchors per step. The four full-vocabulary logit tensors cost
#   max_anchors * block_size * vocab_size * 2 bytes * 4
# which is 4 x 3.8 GiB = 15.2 GiB at 512 anchors, block 16 and vocab 248320. The
# script prints the figure below before it launches anything. Scale this -- not the
# hyperparameters -- if you change the verifier or the block size.
MAX_ANCHORS="${MAX_ANCHORS:-512}"

# DFlash2 modules. conv_group_size must divide the draft hidden size (2048 here);
# selector_top_k is the candidate count the inference path walks.
CONV_KERNEL_SIZE="${CONV_KERNEL_SIZE:-3}"
CONV_GROUP_SIZE="${CONV_GROUP_SIZE:-64}"
SELECTOR_RANK="${SELECTOR_RANK:-256}"
SELECTOR_TOP_K="${SELECTOR_TOP_K:-16}"

BLOCK_SIZE="${BLOCK_SIZE:-16}"

VLLM_PORT="${VLLM_PORT:-8101}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:${VLLM_PORT}/v1}"
VLLM_HEALTH_ENDPOINT="${VLLM_HEALTH_ENDPOINT:-http://localhost:${VLLM_PORT}/health}"
JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-$$}}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-/tmp/dflash2_qwen3_6_35b_a3b_hidden_states}"
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

# d2t.npy / t2d.npy / token_freq.pt are deliberately NOT required here: DFlash2
# trains on the full vocabulary and the symlink view below excludes them.
for path in \
    "$DATA_DIR/state.json" \
    "$DATA_DIR/dataset_info.json"; do
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

# Rebuild the full-vocabulary view from scratch each run so a stale d2t.npy left by
# an earlier experiment can never leak in.
rm -rf "$FULL_VOCAB_DATA_DIR"
mkdir -p "$FULL_VOCAB_DATA_DIR"
shopt -s nullglob
for path in "$DATA_DIR"/*; do
    case "$(basename -- "$path")" in
        d2t.npy|t2d.npy|token_freq.pt) continue ;;
    esac
    ln -s -- "$path" "$FULL_VOCAB_DATA_DIR/"
done
shopt -u nullglob
if [[ ! -f "$FULL_VOCAB_DATA_DIR/state.json" ]]; then
    echo "Failed to build the full-vocabulary data view at $FULL_VOCAB_DATA_DIR" >&2
    exit 1
fi

# Report the vocabulary DFlash2 will actually train on, and the logit footprint it
# implies, before anything expensive starts.
VOCAB_SIZE="$("$SPEC_PYTHON" - "$MODEL" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
text_config = config.get("text_config", config)
print(text_config["vocab_size"])
PY
)"
LOGIT_GIB="$("$SPEC_PYTHON" -c "
print(round(4 * $MAX_ANCHORS * $BLOCK_SIZE * $VOCAB_SIZE * 2 / 1024 ** 3, 1))
")"

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
echo "Data:           $FULL_VOCAB_DATA_DIR (full-vocab view of $DATA_DIR)"
echo "Checkpoints:    $CHECKPOINT_DIR"
echo "TensorBoard:    $TENSORBOARD_DIR/$JOB_TAG"
echo "vLLM GPUs:      $VLLM_GPUS"
echo "Training GPUs:  $TRAIN_GPUS"
echo "vLLM log:       $VLLM_LOG"
echo "Draft vocab:    $VOCAB_SIZE (full verifier vocabulary; DFlash2 cannot prune it)"
echo "Anchors/step:   $MAX_ANCHORS  ->  ~${LOGIT_GIB} GiB of logit activations per training GPU"

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
    --target-layer-ids 2 7 12 17 23 28 33 38 \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
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
echo "=== Launching DFlash2 training ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    PYTHONPATH="$LOCAL_PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    "$TORCHRUN" \
    --standalone \
    --nproc_per_node 6 \
    "$TRAIN_SCRIPT" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$FULL_VOCAB_DATA_DIR" \
    --save-path "$CHECKPOINT_DIR" \
    --epochs 1 \
    --train-data-ratio 0.98 \
    --optimizer adamw \
    --lr 1e-4 \
    --noise-std 0 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.04 \
    --total-seq-len 4096 \
    --speculator-type dflash2 \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers 5 \
    --loss-fn ce \
    --per-position-loss-weight dpace \
    --conv-kernel-size "$CONV_KERNEL_SIZE" \
    --conv-group-size "$CONV_GROUP_SIZE" \
    --selector-rank "$SELECTOR_RANK" \
    --selector-top-k "$SELECTOR_TOP_K" \
    --full-attention-indices 0 1 2 3 4 \
    --target-layer-ids 2 7 12 17 23 28 33 38 \
    --vllm-endpoint "$VLLM_ENDPOINT" \
    --request-timeout 300 \
    --on-missing generate \
    --on-generate delete \
    --logger tensorboard \
    --log-dir "$TENSORBOARD_DIR" \
    --run-name dflash2_qwen3_6_35b_a3b_5full &
TRAIN_PID=$!

# Keep the cluster job attached to training. Its stdout/stderr is therefore
# visible in the cluster job log. On completion or cancellation, cleanup stops
# the full vLLM process group and removes transient hidden-state files.
#
# Watch selector_accept_len against unary_accept_len in TensorBoard: the latter is
# the DFlash baseline computed inside the same run, so their difference is what the
# candidate selector is earning. candidate_recall bounds it.
wait "$TRAIN_PID"
