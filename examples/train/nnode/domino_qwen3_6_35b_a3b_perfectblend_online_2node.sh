#!/usr/bin/env bash
# Online Domino Training Script (Qwen3.6-35B-A3B + perfectblend, full attention)
# 两节点各 2 卡推理 + 6 卡训练，12 个 DDP rank 每步 all-reduce。
set -Eeuo pipefail

COMMENT=""
while getopts "c:" opt; do
    case $opt in
        c) COMMENT="$OPTARG" ;;
        *) echo "Usage: $0 [-c comment]" >&2; exit 1 ;;
    esac
done

# ============ Configuration ============
WS="${ROOT:-/inspire/sfs/project/inf-multimodal/public/wumengke}"
REPO="${REPO:-$WS/speculators}"
ENV_REPO="${ENV_REPO:-$WS/speculators}"
MODEL="${MODEL:-$WS/model_weights/Qwen--Qwen3.6-35B-A3B}"
DATA_DIR="${DATA_DIR:-$WS/datasets/qwen3.6-35b-a3b/qwen3.6-35b-a3b_train_spec_len3072_fullvocab}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR:-$WS/model_weights/domino_qwen3_6-35b-a3b-perfectblend_2node}}"
# 平台每节点执行一次本脚本，8 张分配卡中仅后 6 张加入训练进程组。
NNODES="${PET_NNODES:?需要 PET_NNODES=2}"
NODE_RANK="${PET_NODE_RANK:?需要 PET_NODE_RANK=0 或 1}"
[[ "$NNODES" == 2 && "${PET_NPROC_PER_NODE:-}" == 8 ]] || {
    echo "需要 2 节点、每节点 8 卡（2 推理 + 6 训练）" >&2; exit 1;
}
[[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]] || {
    echo "PET_NODE_RANK 必须为 0 或 1" >&2; exit 1;
}
DIST_MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:?需要 MASTER_ADDR 或 PET_MASTER_ADDR}}"
DIST_MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:?需要 MASTER_PORT 或 PET_MASTER_PORT}}"
# 继承平台所有 NCCL/GLOO 参数；后 6 卡训练沿用同集群的单 rail 缺省设置。
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
if [[ -z "${GLOO_SOCKET_IFNAME:-}" && -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
    export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
fi
JOB_TAG="${DIST_MASTER_ADDR}_${DIST_MASTER_PORT}"
JOB_TAG="${JOB_TAG//[^[:alnum:]._-]/_}"
RUN_NAME="${RUN_NAME:-domino-${COMMENT:-base}-fullattn-2node-${JOB_TAG}}"
SAVE_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/runs/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO/logs/train/qwen3_6_35b_a3b/domino_2node/$RUN_NAME}"
VLLM_LOG="$LOG_DIR/vllm_node${NODE_RANK}.log"
TRAIN_LOG="$LOG_DIR/train_node${NODE_RANK}.log"
SEQ_LENGTH=3072
PACK_SEQ_LEN=8192
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
MUON_LR=2e-4
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_ENDPOINT="http://127.0.0.1:${VLLM_PORT}/v1"

# 每节点前 2 张可见卡推理（TP=1 / DP=2），后 6 张训练。
IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[[ ${#GPU_LIST[@]} == 8 ]] || { echo "每节点需要 8 张可见 GPU" >&2; exit 1; }
VLLM_GPUS=$(IFS=,; printf '%s' "${GPU_LIST[*]:0:2}")
TRAIN_GPUS=$(IFS=,; printf '%s' "${GPU_LIST[*]:2:6}")
NUM_TRAIN_GPUS=6

# Domino-specific parameters
SPECULATOR_TYPE="domino"
NUM_LAYERS=5
FULL_ATTENTION_INDICES="0 1 2 3 4"
TARGET_LAYER_IDS="1 10 19 28 37"
BLOCK_SIZE="${BLOCK_SIZE:-7}"
MAX_ANCHORS="${MAX_ANCHORS:-2048}"
DECAY_GAMMA=7
GRU_HIDDEN_DIM="${GRU_HIDDEN_DIM:-1024}"
LOGITS_CORRECTION_EMB_DIM="${LOGITS_CORRECTION_EMB_DIM:-256}"
PURE_DRAFT_PREFIX_LEN="${PURE_DRAFT_PREFIX_LEN:-1}"
LOSS_FN="${LOSS_FN:-ce}"
LAMBDA_BASE_START="${LAMBDA_BASE_START:-1.0}"
LAMBDA_BASE_DECAY_RATIO="${LAMBDA_BASE_DECAY_RATIO:-1.0}"

SPEC_PYTHON="${SPEC_PYTHON:-$ENV_REPO/speculators_venv/bin/python}"
TORCHRUN="${TORCHRUN:-$ENV_REPO/speculators_venv/bin/torchrun}"
VLLM_PYTHON="${VLLM_PYTHON:-$ENV_REPO/vllm_venv/bin/python}"
LAUNCH_VLLM="${LAUNCH_VLLM:-$REPO/scripts/launch_vllm.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO/scripts/train.py}"
export PYTHONPATH="$REPO/src:$REPO/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$WS/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$WS/datasets/.cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WS/.cache}"
export TORCH_HOME="${TORCH_HOME:-$WS/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WS/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WS/.cache/torchinductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$WS/.cache/vllm}"
export WANDB_PROJECT="${WANDB_PROJECT:-speculators-qwen3_6-35b-a3b-perfectblend}"
export WANDB_MODE="${WANDB_MODE:-online}"
if [[ "$WANDB_MODE" == online && -z "${WANDB_API_KEY:-}" ]]; then
    WANDB_KEY_FILE="${WANDB_KEY_FILE:-$WS/.secrets/wandb_key}"
    [[ -s "$WANDB_KEY_FILE" ]] || { echo "缺少 W&B key：$WANDB_KEY_FILE" >&2; exit 1; }
    export WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi

# ============ Step 1: Prepared data and output ============
for path in "$MODEL/config.json" "$DATA_DIR/state.json" "$DATA_DIR/dataset_info.json"; do
    [[ -f "$path" ]] || { echo "缺少文件：$path" >&2; exit 1; }
done

mkdir -p "$SAVE_DIR" "$LOG_DIR"
exec 9>"$SAVE_DIR/training.lock.node${NODE_RANK}"
flock -n 9 || { echo "本节点已有任务使用 $SAVE_DIR" >&2; exit 1; }
HS_PATH="$(mktemp -d "/tmp/domino_node${NODE_RANK}.XXXXXX")"
VLLM_PID=""
TRAIN_PID=""

terminate_group() {
    local pid="$1"
    [[ -n "$pid" ]] || return 0
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in {1..30}; do
        kill -0 -- "-$pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM HUP
    terminate_group "$TRAIN_PID"
    terminate_group "$VLLM_PID"
    rm -r -- "$HS_PATH"  # 只删除本次 mktemp 创建的 hidden-state 目录。
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

"$SPEC_PYTHON" - "$VLLM_PORT" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(1)
    if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0:
        raise SystemExit(f"Port {sys.argv[1]} is already in use")
PY

# ============ Step 2: Local vLLM server ============
echo "Node $NODE_RANK: vLLM GPUs=$VLLM_GPUS, training GPUs=$TRAIN_GPUS"
echo "Checkpoints: $SAVE_DIR"
echo "vLLM log: $VLLM_LOG"
echo "Training log: $TRAIN_LOG"
setsid env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u MASTER_ADDR -u MASTER_PORT \
    CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
    "$VLLM_PYTHON" "$LAUNCH_VLLM" "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --hidden-states-backend file --hidden-states-path "$HS_PATH" \
    -- \
    --tensor-parallel-size 1 --data-parallel-size 2 --data-parallel-backend mp \
    --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 \
    --data-parallel-address 127.0.0.1 \
    --max-model-len $((SEQ_LENGTH + 16)) \
    --gpu-memory-utilization 0.92 --host 127.0.0.1 --port "$VLLM_PORT" \
    >>"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for local vLLM..."
deadline=$((SECONDS + 1800))
until curl --noproxy '*' -fsS --connect-timeout 2 --max-time 5 \
    "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        tail -n 100 "$VLLM_LOG" >&2
        echo "本机 vLLM 在就绪前退出" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "等待 vLLM 超过 1800 秒，见 $VLLM_LOG" >&2
        exit 1
    fi
    sleep 2
done

# ============ Step 3: Two-node DDP training (global world size = 12) ============
setsid env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    "$TORCHRUN" \
    --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NUM_TRAIN_GPUS" \
    --master_addr "$DIST_MASTER_ADDR" --master_port "$DIST_MASTER_PORT" \
    --rdzv_backend static --rdzv_conf timeout=3600 \
    "$TRAIN_SCRIPT" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_DIR" \
    --vllm-endpoint "$VLLM_ENDPOINT" \
    --save-path "$SAVE_DIR" \
    --hidden-states-path "$HS_PATH" \
    --epochs "$EPOCHS" \
    --checkpoint-freq 0.5 \
    --lr "$LR" \
    --noise-std 0 \
    --muon-lr "$MUON_LR" \
    --scheduler-type cosine \
    --total-seq-len "$PACK_SEQ_LEN" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --dflash-decay-gamma "$DECAY_GAMMA" \
    --gru-hidden-dim "$GRU_HIDDEN_DIM" \
    --logits-correction-emb-dim "$LOGITS_CORRECTION_EMB_DIM" \
    --pure-draft-prefix-len "$PURE_DRAFT_PREFIX_LEN" \
    --loss-fn "$LOSS_FN" \
    --lambda-base-start "$LAMBDA_BASE_START" \
    --lambda-base-decay-ratio "$LAMBDA_BASE_DECAY_RATIO" \
    --train-data-ratio 0.98 \
    --logger wandb \
    --log-dir "$LOG_DIR" \
    --run-name "$RUN_NAME" \
    --on-missing generate \
    --on-generate delete \
    --optimizer muon \
    --scheduler-warmup-ratio 0.01 \
    --hidden-states-backend file \
    --full-attention-indices $FULL_ATTENTION_INDICES \
    >>"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

status=0
wait -n -p finished_pid "$TRAIN_PID" "$VLLM_PID" || status=$?
if [[ "$finished_pid" == "$VLLM_PID" ]]; then
    tail -n 100 "$VLLM_LOG" >&2
    echo "训练期间本机 vLLM 退出" >&2
    exit 1
fi
if (( status != 0 )); then
    tail -n 100 "$TRAIN_LOG" >&2
    exit "$status"
fi
echo "Done. Checkpoints saved to $SAVE_DIR"
