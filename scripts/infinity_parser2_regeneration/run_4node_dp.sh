#!/usr/bin/env bash
# 四节点 vLLM data-parallel regeneration：Infinity-Parser2.1-Flash v2.1。
#
# 平台会在每个节点各执行一次本脚本，并注入 PET_NNODES、PET_NODE_RANK、
# PET_NPROC_PER_NODE、MASTER_*、PET_MASTER_* 以及 NCCL/GLOO 网络变量。
# 0 号节点启动唯一的 OpenAI API、执行采样/生成，并在停服后做 16K prepare
# 和 32K draft-vocab mapping；其余节点只运行 headless DP engine。
# TP=1，每卡并发 32；唯一 API 入口的总并发为 32 × 全局 DP 数，4×8 卡时为 1024。
#
# 默认直接跑完整流程，也可显式指定：
#   bash run_4node_dp.sh full       # 生成 + prepare + 32K vocab（默认）
#   bash run_4node_dp.sh generate   # 只生成
#   bash run_4node_dp.sh prepare    # 只在 0 号节点 prepare + 32K vocab
#   bash run_4node_dp.sh status     # 只在 0 号节点查看生成进度
# 修正旧配置后续跑（在四节点任务的启动命令中设置）：
#   PARSER2_RETRY_ERRORS=1 PARSER2_ALLOW_CONFIG_CHANGE=1 bash run_4node_dp.sh full

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

readonly RUN_ALL="${SCRIPT_DIR}/run_all.sh"
readonly VOCAB_SCRIPT="${REPO_ROOT}/scripts/build_vocab_mapping.py"

PYTHON_BIN="${PARSER2_PYTHON_BIN:-${REPO_ROOT}/speculators_venv/bin/python}"
VLLM_BIN="${PARSER2_VLLM_BIN:-${REPO_ROOT}/vllm_venv/bin/vllm}"

SOURCE_JSONL="${PARSER2_SOURCE_JSONL:-/home/ma-user/work/data_mllm/new_datasets/swift_merged_datasets/version_v2.1/train_v2.1.jsonl}"
MODEL="${PARSER2_MODEL:-/home/ma-user/work/data_mllm/publish_models/Infinity-Parser2.1-Flash-2608}"
MEDIA_ROOT="${PARSER2_MEDIA_ROOT:-/inspire/sfs/project/inf-multimodal/public}"

# 保留已有任务的目录，复用已完成的 1.5M 采样及失败记录。
# 目录名沿用首次启动时的命名；实际输出长度以 MAX_TOKENS 为准。
DATA_ROOT="${PARSER2_DATA_ROOT:-/inspire/sfs/project/inf-multimodal/public/wumengke/datasets/infinity_parsers2_v2_1_max32768_vocab32k}"
OUTPUT_ROOT="${PARSER2_REGEN_ROOT:-${DATA_ROOT}/regen}"
FINAL_DIR="${PARSER2_FINAL_DIR:-${DATA_ROOT}/target_answers}"
PREPARED_ROOT="${PARSER2_PREPARED_ROOT:-${DATA_ROOT}/dflash_data}"

# 与 run_all.sh 一致：总上下文 65536，输出最多 16384，prepare 长度 16384。
# max-model-len 包含输入（含图片展开）和输出，必须给输入保留预算。
MAX_MODEL_LEN="${PARSER2_TEACHER_MAX_MODEL_LEN:-65536}"
MAX_TOKENS="${PARSER2_MAX_TOKENS:-16384}"
SEQ_LENGTH="${PARSER2_SEQ_LENGTH:-16384}"
DRAFT_VOCAB_SIZE="${PARSER2_DRAFT_VOCAB_SIZE:-32000}"
CONCURRENCY_PER_GPU="${PARSER2_CONCURRENCY_PER_GPU:-32}"
MAX_NUM_SEQS="${PARSER2_TEACHER_MAX_NUM_SEQS:-${CONCURRENCY_PER_GPU}}"
MAX_IMAGES="${PARSER2_TEACHER_MAX_IMAGES:-16}"
API_HOST="${PARSER2_TEACHER_HOST:-0.0.0.0}"
API_PORT="${PARSER2_TEACHER_BASE_PORT:-8000}"
START_TIMEOUT="${PARSER2_TEACHER_START_TIMEOUT:-1800}"

NNODES="${PET_NNODES:-}"
NODE_RANK="${PET_NODE_RANK:-}"
LOCAL_DP_SIZE="${PET_NPROC_PER_NODE:-}"
GLOBAL_DP_SIZE=""
GLOBAL_CONCURRENCY=""
DIST_MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
DIST_MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-}}"
DP_ADDRESS="${PARSER2_DP_ADDRESS:-${PET_MASTER_ADDR:-${DIST_MASTER_ADDR}}}"
DP_RPC_PORT="${PARSER2_DP_RPC_PORT:-${PET_MASTER_PORT:-}}"

VLLM_PID=""
COORD_ACTIVE=0
COORD_DIR=""
STOP_FILE=""
STARTED_FILE=""
ACK_FILE=""
FAILED_FILE=""
LOG_FILE=""

die() {
    echo "Error: $*" >&2
    exit 1
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

require_file() {
    [[ -f "$1" ]] || die "missing file: $1"
}

require_executable() {
    [[ -x "$1" ]] || die "missing executable: $1"
}

validate_common_paths() {
    require_executable "$PYTHON_BIN"
    require_file "$RUN_ALL"
    require_file "$VOCAB_SCRIPT"
    require_file "$SOURCE_JSONL"
    require_file "${MODEL}/config.json"
}

validate_cluster_topology() {
    is_positive_integer "$NNODES" || die "PET_NNODES must be a positive integer"
    [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || die "PET_NODE_RANK must be a non-negative integer"
    is_positive_integer "$LOCAL_DP_SIZE" || \
        die "PET_NPROC_PER_NODE must be a positive integer"
    (( NNODES == 4 )) || die "this launcher requires PET_NNODES=4, got ${NNODES}"
    (( NODE_RANK < NNODES )) || \
        die "PET_NODE_RANK must be in [0, $((NNODES - 1))], got ${NODE_RANK}"
    [[ -n "$DIST_MASTER_ADDR" ]] || \
        die "MASTER_ADDR or PET_MASTER_ADDR must be set"
    is_positive_integer "$DIST_MASTER_PORT" || \
        die "MASTER_PORT or PET_MASTER_PORT must be a positive integer"

    GLOBAL_DP_SIZE=$((NNODES * LOCAL_DP_SIZE))

    # data-parallel RPC 和 torch.distributed rendezvous 不能监听同一端口。
    if ! is_positive_integer "$DP_RPC_PORT" || (( DP_RPC_PORT == DIST_MASTER_PORT )); then
        DP_RPC_PORT=$((DIST_MASTER_PORT + 1))
    fi
    (( DIST_MASTER_PORT <= 65535 && DP_RPC_PORT <= 65535 )) || \
        die "invalid rendezvous ports: master=${DIST_MASTER_PORT}, dp_rpc=${DP_RPC_PORT}"
    is_positive_integer "$API_PORT" && (( API_PORT <= 65535 )) || \
        die "PARSER2_TEACHER_BASE_PORT must be in [1, 65535]"

    for value_name in MAX_MODEL_LEN MAX_TOKENS SEQ_LENGTH DRAFT_VOCAB_SIZE \
        CONCURRENCY_PER_GPU MAX_NUM_SEQS MAX_IMAGES START_TIMEOUT; do
        is_positive_integer "${!value_name}" || \
            die "${value_name} must be a positive integer, got ${!value_name}"
    done
    GLOBAL_CONCURRENCY=$((GLOBAL_DP_SIZE * CONCURRENCY_PER_GPU))
    (( MAX_TOKENS < MAX_MODEL_LEN )) || \
        die "PARSER2_MAX_TOKENS=${MAX_TOKENS} leaves no prompt budget below max-model-len=${MAX_MODEL_LEN}; reduce output tokens or increase context length"
}

configure_runtime_paths() {
    local coord_tag
    coord_tag="${DIST_MASTER_ADDR//[^[:alnum:]._-]/_}_${DIST_MASTER_PORT}_${DP_RPC_PORT}"
    COORD_DIR="${REPO_ROOT}/tmp/infinity_parser2_regeneration_4node_${coord_tag}"
    STOP_FILE="${COORD_DIR}/stop"
    STARTED_FILE="${COORD_DIR}/started.node${NODE_RANK}"
    ACK_FILE="${COORD_DIR}/stopped.node${NODE_RANK}"
    FAILED_FILE="${COORD_DIR}/failed.node${NODE_RANK}"
    LOG_FILE="${REPO_ROOT}/logs/infinity_parser2_regeneration/4node_${coord_tag}/vllm_node${NODE_RANK}.log"
    mkdir -p "$COORD_DIR" "$(dirname -- "$LOG_FILE")"
}

terminate_process_group() {
    local pid="$1"
    local alive

    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 0
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..30}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
    alive=0
    kill -0 "$pid" 2>/dev/null && alive=1
    (( alive == 0 )) || echo "Warning: process ${pid} is still alive" >&2
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM HUP

    if (( COORD_ACTIVE == 1 )); then
        if (( NODE_RANK == 0 )); then
            touch "$STOP_FILE"
        fi
        terminate_process_group "$VLLM_PID"
        touch "$ACK_FILE"
        if (( status != 0 )); then
            printf 'node=%s status=%s\n' "$NODE_RANK" "$status" >"$FAILED_FILE"
        fi
    else
        terminate_process_group "$VLLM_PID"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

initialize_coordination() {
    local rank

    configure_runtime_paths
    if (( NODE_RANK == 0 )); then
        rm -f -- "$STOP_FILE"
        for ((rank = 0; rank < NNODES; rank++)); do
            rm -f -- \
                "${COORD_DIR}/stopped.node${rank}" \
                "${COORD_DIR}/failed.node${rank}"
        done
    fi
    COORD_ACTIVE=1
}

start_vllm() {
    local -a args=(
        "$VLLM_BIN" serve "$MODEL"
        --served-model-name "$MODEL"
        --tensor-parallel-size 1
        --data-parallel-size "$GLOBAL_DP_SIZE"
        --data-parallel-backend mp
        --nnodes "$NNODES"
        --node-rank "$NODE_RANK"
        --master-addr "$DIST_MASTER_ADDR"
        --master-port "$DIST_MASTER_PORT"
        --data-parallel-address "$DP_ADDRESS"
        --data-parallel-rpc-port "$DP_RPC_PORT"
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-seqs "$MAX_NUM_SEQS"
        --allowed-local-media-path "$MEDIA_ROOT"
        --limit-mm-per-prompt "{\"image\":${MAX_IMAGES}}"
    )
    local -a extra_args=()

    if (( NODE_RANK == 0 )); then
        args+=(
            --host "$API_HOST"
            --port "$API_PORT"
            --api-server-count 1
        )
    else
        args+=(--headless)
    fi
    if [[ -n "${PARSER2_VLLM_EXTRA_ARGS:-}" ]]; then
        read -r -a extra_args <<<"$PARSER2_VLLM_EXTRA_ARGS"
        args+=("${extra_args[@]}")
    fi

    echo "Starting node ${NODE_RANK}/${NNODES} with local DP ${LOCAL_DP_SIZE}, global DP ${GLOBAL_DP_SIZE}"
    echo "Concurrency: ${CONCURRENCY_PER_GPU}/GPU x ${GLOBAL_DP_SIZE} DP replicas = ${GLOBAL_CONCURRENCY} total; max-num-seqs=${MAX_NUM_SEQS}/replica"
    echo "vLLM log: ${LOG_FILE}"
    # 平台的 NCCL_*、GLOO_SOCKET_IFNAME、NCCL_SOCKET_IFNAME 等网络变量原样
    # 继承。外层 WORLD_SIZE/RANK 描述的是平台任务，不是 vLLM 自己创建的
    # engine rank；只对这个子进程清掉，避免被误认为 external launcher。
    setsid env \
        -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
        VLLM_PLUGINS="" \
        HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
        HF_HOME="${HF_HOME:-/inspire/sfs/project/inf-multimodal/public/wumengke/.cache/huggingface}" \
        PYTHONPATH="${REPO_ROOT}/hs_connectors/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${args[@]}" >"$LOG_FILE" 2>&1 &
    VLLM_PID=$!
    touch "$STARTED_FILE"
}

new_failure_file() {
    local failure
    for failure in "${COORD_DIR}"/failed.node*; do
        [[ -e "$failure" ]] || continue
        if [[ "$failure" -nt "$STARTED_FILE" ]]; then
            printf '%s\n' "$failure"
            return 0
        fi
    done
    return 1
}

wait_for_api() {
    local deadline=$((SECONDS + START_TIMEOUT))
    local failure

    command -v curl >/dev/null || die "curl is required"
    while true; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            tail -n 120 "$LOG_FILE" >&2 || true
            die "vLLM API process exited during startup"
        fi
        if failure="$(new_failure_file)"; then
            cat "$failure" >&2 || true
            die "a remote vLLM node exited during startup"
        fi
        if curl -fsS --max-time 5 "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
            echo "vLLM API ready: http://127.0.0.1:${API_PORT}"
            return 0
        fi
        (( SECONDS < deadline )) || {
            tail -n 120 "$LOG_FILE" >&2 || true
            die "vLLM startup timed out after ${START_TIMEOUT}s"
        }
        sleep 5
    done
}

monitor_headless_node() {
    while kill -0 "$VLLM_PID" 2>/dev/null; do
        if [[ -e "$STOP_FILE" && "$STOP_FILE" -nt "$STARTED_FILE" ]]; then
            terminate_process_group "$VLLM_PID"
            VLLM_PID=""
            touch "$ACK_FILE"
            COORD_ACTIVE=0
            echo "Node ${NODE_RANK} stopped after coordinator signal"
            return 0
        fi
        sleep 3
    done

    wait "$VLLM_PID" || true
    VLLM_PID=""
    # 主节点先写 stop 再终止 API；远端 engine 可能因连接断开而抢先退出。
    # 只要本次启动之后已经收到 stop，这仍是正常的集群停服。
    if [[ -e "$STOP_FILE" && "$STOP_FILE" -nt "$STARTED_FILE" ]]; then
        touch "$ACK_FILE"
        COORD_ACTIVE=0
        echo "Node ${NODE_RANK} stopped after coordinator signal"
        return 0
    fi
    tail -n 120 "$LOG_FILE" >&2 || true
    die "headless vLLM process exited before the coordinator stopped it"
}

run_generation() {
    PARSER2_PYTHON_BIN="$PYTHON_BIN" \
    PARSER2_SOURCE_JSONL="$SOURCE_JSONL" \
    PARSER2_MODEL="$MODEL" \
    PARSER2_MEDIA_ROOT="$MEDIA_ROOT" \
    PARSER2_DATA_ROOT="$DATA_ROOT" \
    PARSER2_REGEN_ROOT="$OUTPUT_ROOT" \
    PARSER2_FINAL_DIR="$FINAL_DIR" \
    PARSER2_PREPARED_ROOT="$PREPARED_ROOT" \
    PARSER2_ENDPOINTS="http://127.0.0.1:${API_PORT}/v1/chat/completions" \
    PARSER2_MAX_TOKENS="$MAX_TOKENS" \
    PARSER2_CONCURRENCY_PER_ENDPOINT="$GLOBAL_CONCURRENCY" \
    PARSER2_SEQ_LENGTH="$SEQ_LENGTH" \
        bash "$RUN_ALL" generate full
}

wait_for_cluster_stop() {
    local deadline=$((SECONDS + 180))
    local rank
    local all_stopped

    touch "$STOP_FILE"
    terminate_process_group "$VLLM_PID"
    VLLM_PID=""
    touch "$ACK_FILE"

    while true; do
        all_stopped=1
        for ((rank = 0; rank < NNODES; rank++)); do
            if [[ ! -e "${COORD_DIR}/stopped.node${rank}" || \
                  ! "${COORD_DIR}/stopped.node${rank}" -nt "$STOP_FILE" ]]; then
                all_stopped=0
                break
            fi
        done
        (( all_stopped == 1 )) && break
        (( SECONDS < deadline )) || \
            die "timed out waiting for all headless nodes to stop; keeping ${COORD_DIR} for late workers"
        sleep 3
    done

    for ((rank = 0; rank < NNODES; rank++)); do
        rm -f -- \
            "${COORD_DIR}/started.node${rank}" \
            "${COORD_DIR}/stopped.node${rank}" \
            "${COORD_DIR}/failed.node${rank}"
    done
    rm -f -- "$STOP_FILE"
    rmdir "$COORD_DIR" 2>/dev/null || true
    COORD_ACTIVE=0
}

run_prepare() {
    PARSER2_PYTHON_BIN="$PYTHON_BIN" \
    PARSER2_SOURCE_JSONL="$SOURCE_JSONL" \
    PARSER2_MODEL="$MODEL" \
    PARSER2_MEDIA_ROOT="$MEDIA_ROOT" \
    PARSER2_DATA_ROOT="$DATA_ROOT" \
    PARSER2_REGEN_ROOT="$OUTPUT_ROOT" \
    PARSER2_FINAL_DIR="$FINAL_DIR" \
    PARSER2_PREPARED_ROOT="$PREPARED_ROOT" \
    PARSER2_PREPARE_ONLY=1 \
    PARSER2_PREPARE_MODE=fast \
    PARSER2_SEQ_LENGTH="$SEQ_LENGTH" \
    PARSER2_HOLD_GPUS="${PARSER2_HOLD_GPUS:-0}" \
        bash "$RUN_ALL" full
}

build_draft_vocab() {
    local prepared_dir="${PREPARED_ROOT}/full"
    local token_freq="${prepared_dir}/token_freq.pt"

    require_file "$token_freq"
    "$PYTHON_BIN" "$VOCAB_SCRIPT" \
        --token-freq-path "$token_freq" \
        --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
        --target-model-path "$MODEL" \
        --output-path "$prepared_dir"

    "$PYTHON_BIN" - \
        "${prepared_dir}/d2t.npy" \
        "${prepared_dir}/t2d.npy" \
        "$DRAFT_VOCAB_SIZE" <<'PY'
import sys
from pathlib import Path

import numpy as np

d2t_path, t2d_path = map(Path, sys.argv[1:3])
draft_vocab_size = int(sys.argv[3])
d2t = np.load(d2t_path, mmap_mode="r")
t2d = np.load(t2d_path, mmap_mode="r")
if d2t.shape != (draft_vocab_size,):
    raise SystemExit(f"unexpected d2t shape: {d2t.shape}")
if t2d.ndim != 1:
    raise SystemExit(f"unexpected t2d shape: {t2d.shape}")
print(f"32K vocab mapping ready: d2t={d2t.shape}, t2d={t2d.shape}")
PY
}

show_status() {
    PARSER2_PYTHON_BIN="$PYTHON_BIN" \
    PARSER2_MODEL="$MODEL" \
    PARSER2_DATA_ROOT="$DATA_ROOT" \
    PARSER2_REGEN_ROOT="$OUTPUT_ROOT" \
        bash "$RUN_ALL" status full
}

action="${1:-full}"
case "$action" in
    full|generate)
        validate_common_paths
        require_executable "$VLLM_BIN"
        command -v setsid >/dev/null || die "setsid is required"
        validate_cluster_topology
        initialize_coordination
        start_vllm
        if (( NODE_RANK == 0 )); then
            wait_for_api
            run_generation
            wait_for_cluster_stop
            if [[ "$action" == "full" ]]; then
                run_prepare
                build_draft_vocab
            fi
        else
            monitor_headless_node
        fi
        ;;
    prepare)
        validate_common_paths
        if [[ "${PET_NODE_RANK:-0}" == "0" ]]; then
            run_prepare
            build_draft_vocab
        else
            echo "Node ${PET_NODE_RANK}: prepare only runs on node 0"
        fi
        ;;
    status)
        validate_common_paths
        if [[ "${PET_NODE_RANK:-0}" == "0" ]]; then
            show_status
        else
            echo "Node ${PET_NODE_RANK}: status only runs on node 0"
        fi
        ;;
    *)
        echo "Usage: $(basename "$0") [full|generate|prepare|status]" >&2
        exit 2
        ;;
esac
