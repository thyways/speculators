#!/usr/bin/env bash
# 分离式（disaggregated）mooncake 在线 DSpark 训练：Infinity-Parser2-Flash。
#
# 拓扑（2 节点，平台按 PET_NODE_RANK 分角色）：
#   node0 (rank 0) = 生成节点：mooncake_master + 8 卡 vLLM（producer）。起来后
#                    「陪跑」到 node1 训练结束（靠共享 SFS 上的完成标志感知）。
#   node1 (rank 1) = 训练节点：8 卡单机训练（consumer），HTTP 触发 node0 的 vLLM
#                    生成，hidden states 经 mooncake（TCP/RDMA over IB）跨机取回。
#
# 地址全部来自平台注入：MASTER_ADDR = rank0(node0) 地址，复用为 vLLM/mooncake 的
# host；MASTER_PORT/PET_* 由平台给。本机 IP 作 MOONCAKE_LOCAL_HOSTNAME。
#
# 注意：对称硬件下 8:8 分离的训练并行度(8)与 co-located 相同，只是 hidden states
# 改走跨机。分离式的价值在于生成:训练配比可脱离 1:1；此脚本给的是干净的 8:8。

set -Eeuo pipefail

ROOT="/inspire/sfs/project/inf-multimodal/public/wumengke"
REPO="$ROOT/speculators"
MODEL="/home/ma-user/work/data_mllm/publish_models/Infinity-Parser2-2B-2604"
DATA_DIR="$ROOT/datasets/infinity_parser2_v1_12_dflash_data/full"
RUN_DIR="${RUN_DIR:-$REPO/output/dspark_infinity_parser2_flash_online_full}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-3}"
FETCH_THREADS="${FETCH_THREADS:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/model_weights/dspark_infinity_parser2_2b_v1_12}"
LOG_DIR="${LOG_DIR:-$RUN_DIR}"
WANDB_PROJECT="${WANDB_PROJECT:-infinity-parser2-flash}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$ROOT/.secrets/wandb_key}"
MEDIA_ROOT="/inspire/sfs/project/inf-multimodal/public"
VLLM_PORT="${VLLM_PORT:-8200}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MARKOV_RANK="${MARKOV_RANK:-256}"
MARKOV_HEAD_TYPE="${MARKOV_HEAD_TYPE:-vanilla}"
CONFIDENCE_HEAD_ALPHA="${CONFIDENCE_HEAD_ALPHA:-1.0}"
DEFAULT_LOSS_FN='{"ce": 0.1, "tv": 0.9}'
LOSS_FN="${LOSS_FN:-$DEFAULT_LOSS_FN}"
TARGET_LAYER_IDS=(2 7 12 17 22)
JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-$$}}"
HIDDEN_STATES_DIR=""

# ---- 分离式拓扑（平台注入 PET_*/MASTER_*）----
NNODES="${PET_NNODES:-2}"
NODE_RANK="${PET_NODE_RANK:-0}"
# 分离式必须多机：node0 生成、node1 训练，地址靠平台注入
: "${MASTER_ADDR:?分离式需要平台注入 MASTER_ADDR（rank0/生成节点地址）}"
: "${MASTER_PORT:?分离式需要平台注入 MASTER_PORT}"
if (( NNODES != 2 )); then
    echo "本分离式脚本按 2 节点设计（1 生成 + 1 训练），当前 PET_NNODES=$NNODES" >&2
    exit 1
fi

# 本机可路由 IP：mooncake 各 client 用它做 P2P 握手/注册
LOCAL_IP="${MOONCAKE_LOCAL_HOSTNAME:-$(hostname -i 2>/dev/null | awk '{print $1}')}"
if [[ -z "$LOCAL_IP" ]]; then
    echo "无法确定本机 IP，请显式设置 MOONCAKE_LOCAL_HOSTNAME" >&2
    exit 1
fi
export MOONCAKE_LOCAL_HOSTNAME="$LOCAL_IP"

# mooncake / 远程 vLLM 地址（生成侧都在 node0 = MASTER_ADDR）
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-50051}"
MOONCAKE_MASTER="${MASTER_ADDR}:${MOONCAKE_MASTER_PORT}"
MOONCAKE_METADATA="${MOONCAKE_METADATA:-P2PHANDSHAKE}"
# tcp 最稳（走 IPoIB/普通网络）；想吃满 IB 带宽设 MOONCAKE_PROTOCOL=rdma
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-rdma}"
VLLM_ENDPOINT="http://${MASTER_ADDR}:${VLLM_PORT}/v1"

# 跨节点协调标志：两节点须一致，用 master 地址派生（不依赖各自的 $$）
RUN_SIG="$(printf '%s_%s' "$MASTER_ADDR" "$MASTER_PORT" | tr ':.-' '___')"
DONE_FLAG="$RUN_DIR/.sep_${RUN_SIG}.done"
FAILED_FLAG="$RUN_DIR/.sep_${RUN_SIG}.failed"
# node1 训练是单机 8 卡；仍为 NCCL/GLOO 兜底网卡（平台一般注入 NCCL_SOCKET_IFNAME）
if [[ -n "${NCCL_SOCKET_IFNAME:-}" && -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
    export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
fi

VLLM_LOG="$RUN_DIR/vllm_${JOB_TAG}_node${NODE_RANK}.log"
MASTER_LOG="$RUN_DIR/mooncake_master_${JOB_TAG}.log"

SPEC_PYTHON="$REPO/speculators_venv/bin/python"
TORCHRUN="$REPO/speculators_venv/bin/torchrun"
VLLM_PYTHON="$REPO/vllm_venv/bin/python"
MOONCAKE_MASTER_BIN="$REPO/vllm_venv/bin/mooncake_master"

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR"

for executable in "$SPEC_PYTHON" "$TORCHRUN" "$VLLM_PYTHON" "$MOONCAKE_MASTER_BIN"; do
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
# 分离式：整节点一个角色。node0 全部卡跑 vLLM，node1 全部卡训练。
ALL_GPUS=$(IFS=,; printf '%s' "${GPU_LIST[*]}")
NGPU="${#GPU_LIST[@]}"

# 每节点各自加锁：防同一节点重复起同一角色，不误伤对端节点。
exec 9>"$RUN_DIR/training.lock.node${NODE_RANK}"
if ! flock -n 9; then
    echo "Another job is using $RUN_DIR on node $NODE_RANK" >&2
    exit 1
fi

VLLM_PID=""
TRAIN_PID=""
MASTER_PID=""

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
    # 训练节点异常退出时落个失败标志，避免 node0 干等到超时
    if (( NODE_RANK == 1 )) && [[ -n "$TRAIN_PID" && ! -e "$DONE_FLAG" ]]; then
        : >"$FAILED_FLAG" 2>/dev/null || true
    fi
    terminate_job "$TRAIN_PID"
    terminate_job "$VLLM_PID"
    terminate_job "$MASTER_PID"
    [[ -n "$TRAIN_PID" ]] && wait "$TRAIN_PID" 2>/dev/null || true
    [[ -n "$VLLM_PID" ]] && wait "$VLLM_PID" 2>/dev/null || true
    [[ -n "$MASTER_PID" ]] && wait "$MASTER_PID" 2>/dev/null || true
    case "$HIDDEN_STATES_DIR" in
        */dspark_parser2_flash_*) rm -rf -- "$HIDDEN_STATES_DIR" ;;
    esac
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

echo "Role:          node $NODE_RANK / $NNODES  ($([[ $NODE_RANK == 0 ]] && echo 生成节点 || echo 训练节点))"
echo "Local IP:      $LOCAL_IP"
echo "Mooncake:      master=$MOONCAKE_MASTER  metadata=$MOONCAKE_METADATA  protocol=$MOONCAKE_PROTOCOL"
echo "vLLM endpoint: $VLLM_ENDPOINT"
echo "Model:         $MODEL"
echo "Data:          $DATA_DIR"
echo "Checkpoints:   $CHECKPOINT_DIR"

# ===========================================================================
if (( NODE_RANK == 0 )); then
    # ---------------------- 生成节点：master + 8 卡 vLLM ----------------------
    HIDDEN_STATES_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dspark_parser2_flash_${JOB_TAG}.XXXXXX")"

    # 本机 vLLM 端口与 mooncake master 端口不能被占用
    "$SPEC_PYTHON" - "$VLLM_PORT" "$MOONCAKE_MASTER_PORT" <<'PY'
import socket, sys
for port in (int(sys.argv[1]), int(sys.argv[2])):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"Port {port} is already in use")
PY

    # node0 起服务前清掉上一轮残留的协调标志
    rm -f "$DONE_FLAG" "$FAILED_FLAG"

    # 1) mooncake master（rpc_address 默认 0.0.0.0，node1 可用 MASTER_ADDR:PORT 连）
    setsid "$MOONCAKE_MASTER_BIN" --port "$MOONCAKE_MASTER_PORT" \
        >"$MASTER_LOG" 2>&1 &
    MASTER_PID=$!
    sleep 3
    if ! kill -0 "$MASTER_PID" 2>/dev/null; then
        tail -n 50 "$MASTER_LOG" >&2 || true
        echo "mooncake_master 启动失败" >&2
        exit 1
    fi
    echo "mooncake_master 已就绪 ($MOONCAKE_MASTER)"

    # 2) 8 卡 vLLM（mooncake producer），监听 0.0.0.0 供 node1 触发生成
    setsid env \
        CUDA_VISIBLE_DEVICES="$ALL_GPUS" \
        PYTHONUNBUFFERED=1 \
        MOONCAKE_LOCAL_HOSTNAME="$LOCAL_IP" \
        "$VLLM_PYTHON" "$REPO/scripts/launch_vllm.py" "$MODEL" \
        --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
        --hidden-states-backend mooncake \
        --mooncake-master "$MOONCAKE_MASTER" \
        --mooncake-metadata-server "$MOONCAKE_METADATA" \
        --mooncake-protocol "$MOONCAKE_PROTOCOL" \
        --hidden-states-path "$HIDDEN_STATES_DIR" \
        --tensor-parallel-size 1 \
        --data-parallel-size "$NGPU" \
        --gpu-memory-utilization 0.9 \
        --max_model_len 65536 \
        --api-server-count 8 \
        --served-model-name "$MODEL" \
        --allowed-local-media-path "$MEDIA_ROOT" \
        --limit-mm-per-prompt '{"image":16}' \
        --host 0.0.0.0 \
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
    echo "vLLM 已就绪，等待 node1 训练完成…"

    # 3) 陪跑：直到训练完成/失败标志出现，或本地服务挂掉，或总超时
    MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-$((5 * 24 * 3600))}"
    gen_deadline=$((SECONDS + MAX_TRAIN_SECONDS))
    while [[ ! -e "$DONE_FLAG" && ! -e "$FAILED_FLAG" ]]; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "vLLM 提前退出" >&2; tail -n 100 "$VLLM_LOG" >&2 || true; exit 1
        fi
        if ! kill -0 "$MASTER_PID" 2>/dev/null; then
            echo "mooncake_master 提前退出" >&2; tail -n 50 "$MASTER_LOG" >&2 || true; exit 1
        fi
        if (( SECONDS >= gen_deadline )); then
            echo "陪跑超时（MAX_TRAIN_SECONDS=$MAX_TRAIN_SECONDS）" >&2; exit 1
        fi
        sleep 15
    done

    if [[ -e "$FAILED_FLAG" ]]; then
        echo "node1 训练失败，生成节点退出" >&2
        exit 1
    fi
    echo "node1 训练完成，生成节点收尾退出"
    exit 0
fi

# ===========================================================================
# ---------------------- 训练节点：8 卡单机训练（consumer）--------------------
echo "等待 node0 vLLM 就绪：$VLLM_ENDPOINT"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-2400}"
deadline=$((SECONDS + VLLM_WAIT_SECONDS))
until curl -fsS --max-time 10 "http://${MASTER_ADDR}:${VLLM_PORT}/health" >/dev/null 2>&1; do
    if [[ -e "$FAILED_FLAG" ]]; then
        echo "生成节点已标记失败，训练节点退出" >&2; exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "等待 node0 vLLM 超时（VLLM_WAIT_SECONDS=$VLLM_WAIT_SECONDS）" >&2
        exit 1
    fi
    sleep 5
done
echo "node0 vLLM 就绪，开始训练（单机 $NGPU 卡）"

setsid env \
    CUDA_VISIBLE_DEVICES="$ALL_GPUS" \
    PYTHONUNBUFFERED=1 \
    MOONCAKE_LOCAL_HOSTNAME="$LOCAL_IP" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_MODE="$WANDB_MODE" \
    "$TORCHRUN" \
    --standalone \
    --nproc_per_node "$NGPU" \
    "$REPO/scripts/train.py" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_DIR" \
    --save-path "$CHECKPOINT_DIR" \
    --speculator-type dspark \
    --draft-arch qwen3 \
    --draft-hidden-act silu \
    --num-layers 5 \
    --mask-token-id 248077 \
    --block-size "$BLOCK_SIZE" \
    --sample-from-anchor \
    --max-anchors 3072 \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --draft-mrope-full-head-hack \
    --sliding-window 2048 \
    --sliding-window-non-causal \
    --draft-attn-impl simple_flex_attention \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --loss-fn "$LOSS_FN" \
    --dflash-decay-gamma 4.0 \
    --optimizer muon \
    --muon-lr 2e-4 \
    --lr 1e-4 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.04 \
    --epochs 3 \
    --checkpoint-freq 0.1 \
    --total-seq-len 16384 \
    --train-data-ratio 0.99 \
    --noise-std 0 \
    --hidden-states-dtype bfloat16 \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --fetch-threads "$FETCH_THREADS" \
    --hidden-states-backend mooncake \
    --mooncake-master "$MOONCAKE_MASTER" \
    --mooncake-metadata-server "$MOONCAKE_METADATA" \
    --mooncake-protocol "$MOONCAKE_PROTOCOL" \
    --vllm-endpoint "$VLLM_ENDPOINT" \
    --on-missing generate \
    --on-generate delete \
    --request-timeout "$REQUEST_TIMEOUT" \
    --max-retries 5 \
    --fail-on-hidden-state-error \
    --seed 42 \
    --logger wandb \
    --log-dir "$LOG_DIR" \
    --run-name "$JOB_TAG" &
TRAIN_PID=$!

if wait "$TRAIN_PID"; then
    : >"$DONE_FLAG"
    echo "训练完成，已写完成标志"
else
    status=$?
    : >"$FAILED_FLAG"
    echo "训练失败（exit $status），已写失败标志" >&2
    exit "$status"
fi
