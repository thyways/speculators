#!/usr/bin/env bash
# 四节点单卡副本 regeneration：Infinity-Parser2.1-Flash v2.1。
#
# 平台会在每个节点各执行一次本脚本，并注入 PET_NNODES、PET_NODE_RANK、
# PET_NPROC_PER_NODE、MASTER_*、PET_MASTER_* 以及 NCCL/GLOO 网络变量。
# 与 run_all.sh 一样，每张卡启动独立的 vLLM API 和 engine，使用独立端口。
# 每个节点本地处理图片；0 号节点汇总所有端口，执行采样/生成，并在停服后
# 做 16K prepare 和 32K draft-vocab mapping。
# TP=1，服务端使用 vLLM 默认调度上限；客户端总并发为每卡 32 × 全局 DP 数，
# 4×8 卡时为 1024。
# 每个端口固定 32 个客户端 worker，避免共享端口的长连接集中到少数 API 进程。
# 请求并发包含前处理和等待时间，不代表每卡始终 Running=32。
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
MAX_NUM_SEQS="${PARSER2_TEACHER_MAX_NUM_SEQS:-}"
MAX_IMAGES="${PARSER2_TEACHER_MAX_IMAGES:-16}"
API_HOST="${PARSER2_TEACHER_HOST:-0.0.0.0}"
API_PORT="${PARSER2_TEACHER_BASE_PORT:-8000}"
NODE_ADDRESS="${PARSER2_TEACHER_ADVERTISE_HOST:-}"
START_TIMEOUT="${PARSER2_TEACHER_START_TIMEOUT:-1800}"

NNODES="${PET_NNODES:-}"
NODE_RANK="${PET_NODE_RANK:-}"
LOCAL_DP_SIZE="${PET_NPROC_PER_NODE:-}"
GLOBAL_DP_SIZE=""
GLOBAL_CONCURRENCY=""
DIST_MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
DIST_MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-}}"

declare -a VLLM_PIDS=()
declare -a ENDPOINTS=()
COORD_ACTIVE=0
COORD_DIR=""
STOP_FILE=""
STARTED_FILE=""
ACK_FILE=""
FAILED_FILE=""
READY_FILE=""
LOG_DIR=""

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

    is_positive_integer "$API_PORT" && (( API_PORT + LOCAL_DP_SIZE - 1 <= 65535 )) || \
        die "API ports ${API_PORT}..$((API_PORT + LOCAL_DP_SIZE - 1)) must be in [1, 65535]"

    for value_name in MAX_MODEL_LEN MAX_TOKENS SEQ_LENGTH DRAFT_VOCAB_SIZE \
        CONCURRENCY_PER_GPU MAX_IMAGES START_TIMEOUT; do
        is_positive_integer "${!value_name}" || \
            die "${value_name} must be a positive integer, got ${!value_name}"
    done
    if [[ -n "$MAX_NUM_SEQS" ]]; then
        is_positive_integer "$MAX_NUM_SEQS" || \
            die "MAX_NUM_SEQS must be a positive integer, got ${MAX_NUM_SEQS}"
    fi
    GLOBAL_CONCURRENCY=$((GLOBAL_DP_SIZE * CONCURRENCY_PER_GPU))
    (( MAX_TOKENS < MAX_MODEL_LEN )) || \
        die "PARSER2_MAX_TOKENS=${MAX_TOKENS} leaves no prompt budget below max-model-len=${MAX_MODEL_LEN}; reduce output tokens or increase context length"
}

configure_runtime_paths() {
    local coord_tag
    coord_tag="${DIST_MASTER_ADDR//[^[:alnum:]._-]/_}_${DIST_MASTER_PORT}"
    COORD_DIR="${REPO_ROOT}/tmp/infinity_parser2_regeneration_4node_${coord_tag}"
    STOP_FILE="${COORD_DIR}/stop"
    STARTED_FILE="${COORD_DIR}/started.node${NODE_RANK}"
    ACK_FILE="${COORD_DIR}/stopped.node${NODE_RANK}"
    FAILED_FILE="${COORD_DIR}/failed.node${NODE_RANK}"
    READY_FILE="${COORD_DIR}/endpoints.node${NODE_RANK}"
    LOG_DIR="${REPO_ROOT}/logs/infinity_parser2_regeneration/4node_${coord_tag}"
    mkdir -p "$COORD_DIR" "$LOG_DIR"
}

stop_vllm() {
    local pid
    local alive

    # 同时通知所有单卡服务，再等待退出，避免逐卡串行等待 30 秒。
    for pid in "${VLLM_PIDS[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    for _ in {1..30}; do
        alive=0
        for pid in "${VLLM_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive=1
        done
        (( alive == 0 )) && break
        sleep 1
    done
    for pid in "${VLLM_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    done
    VLLM_PIDS=()
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM HUP

    if (( COORD_ACTIVE == 1 )); then
        if (( NODE_RANK == 0 )); then
            touch "$STOP_FILE"
        fi
        stop_vllm
        touch "$ACK_FILE"
        if (( status != 0 )); then
            printf 'node=%s status=%s\n' "$NODE_RANK" "$status" >"$FAILED_FILE"
        fi
    else
        stop_vllm
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
    rm -f -- "$READY_FILE"
    touch "$STARTED_FILE"
    COORD_ACTIVE=1
}

start_vllm() {
    local index gpu port
    local -a gpus=()
    local -a args=(
        "$VLLM_BIN" serve "$MODEL"
        --served-model-name "$MODEL"
        --tensor-parallel-size 1
        --host "$API_HOST"
        --max-model-len "$MAX_MODEL_LEN"
        --allowed-local-media-path "$MEDIA_ROOT"
        --limit-mm-per-prompt "{\"image\":${MAX_IMAGES}}"
    )
    local -a extra_args=()

    if [[ -n "$MAX_NUM_SEQS" ]]; then
        args+=(--max-num-seqs "$MAX_NUM_SEQS")
    fi
    if [[ -n "${PARSER2_TEACHER_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}" ]]; then
        IFS=',' read -r -a gpus <<<"${PARSER2_TEACHER_GPU_IDS:-${CUDA_VISIBLE_DEVICES}}"
    else
        for ((index = 0; index < LOCAL_DP_SIZE; index++)); do
            gpus+=("$index")
        done
    fi
    (( ${#gpus[@]} >= LOCAL_DP_SIZE )) || \
        die "PET_NPROC_PER_NODE=${LOCAL_DP_SIZE}, but only ${#gpus[@]} GPUs are configured"
    if [[ -n "${PARSER2_VLLM_EXTRA_ARGS:-}" ]]; then
        read -r -a extra_args <<<"$PARSER2_VLLM_EXTRA_ARGS"
        args+=("${extra_args[@]}")
    fi

    echo "Starting node ${NODE_RANK}/${NNODES}: ${LOCAL_DP_SIZE} independent single-GPU servers on ports ${API_PORT}..$((API_PORT + LOCAL_DP_SIZE - 1))"
    echo "Concurrency: ${CONCURRENCY_PER_GPU}/endpoint x ${GLOBAL_DP_SIZE} GPU endpoints = ${GLOBAL_CONCURRENCY} total; max-num-seqs=${MAX_NUM_SEQS:-vLLM default}/GPU"
    # 平台的 NCCL_*、GLOO_SOCKET_IFNAME、NCCL_SOCKET_IFNAME 等网络变量原样
    # 继承。外层 WORLD_SIZE/RANK 描述的是平台任务，不是 vLLM 自己创建的
    # engine rank；只对这个子进程清掉，避免被误认为 external launcher。
    for ((index = 0; index < LOCAL_DP_SIZE; index++)); do
        gpu="${gpus[$index]//[[:space:]]/}"
        port=$((API_PORT + index))
        echo "GPU ${gpu}: ${LOG_DIR}/vllm_node${NODE_RANK}_gpu${index}.log"
        setsid env \
            -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
            -u MASTER_ADDR -u MASTER_PORT \
            CUDA_VISIBLE_DEVICES="$gpu" \
            VLLM_PLUGINS="" \
            HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
            HF_HOME="${HF_HOME:-/inspire/sfs/project/inf-multimodal/public/wumengke/.cache/huggingface}" \
            PYTHONPATH="${REPO_ROOT}/hs_connectors/src${PYTHONPATH:+:${PYTHONPATH}}" \
            "${args[@]}" --port "$port" \
            >"${LOG_DIR}/vllm_node${NODE_RANK}_gpu${index}.log" 2>&1 &
        VLLM_PIDS+=("$!")
    done
}

check_local_servers() {
    local index
    for index in "${!VLLM_PIDS[@]}"; do
        if ! kill -0 "${VLLM_PIDS[$index]}" 2>/dev/null; then
            tail -n 120 "${LOG_DIR}/vllm_node${NODE_RANK}_gpu${index}.log" >&2 || true
            die "node ${NODE_RANK} GPU ${index} vLLM process exited"
        fi
    done
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
    local index ready

    command -v curl >/dev/null || die "curl is required"
    while true; do
        check_local_servers
        if failure="$(new_failure_file)"; then
            cat "$failure" >&2 || true
            die "a remote vLLM node exited during startup"
        fi
        ready=0
        for ((index = 0; index < LOCAL_DP_SIZE; index++)); do
            if curl -fsS --max-time 5 "http://127.0.0.1:$((API_PORT + index))/health" >/dev/null 2>&1; then
                ready=$((ready + 1))
            fi
        done
        (( ready == LOCAL_DP_SIZE )) && break
        (( SECONDS < deadline )) || \
            die "vLLM startup timed out after ${START_TIMEOUT}s"
        echo "Node ${NODE_RANK}: waiting for local APIs, ${ready}/${LOCAL_DP_SIZE} ready"
        sleep 5
    done

    if [[ -z "$NODE_ADDRESS" ]]; then
        # 平台为每个 worker 提供可解析的主机名；公布节点地址供 0 号节点访问。
        NODE_ADDRESS="$(hostname -f)"
    fi
    for ((index = 0; index < LOCAL_DP_SIZE; index++)); do
        printf 'http://%s:%s/v1/chat/completions\n' "$NODE_ADDRESS" "$((API_PORT + index))"
    done >"${READY_FILE}.tmp"
    mv -- "${READY_FILE}.tmp" "$READY_FILE"
    echo "Node ${NODE_RANK}: ${LOCAL_DP_SIZE} APIs ready on ${NODE_ADDRESS}"
}

wait_for_cluster_apis() {
    local deadline=$((SECONDS + START_TIMEOUT))
    local rank failure all_ready
    local -a node_endpoints=()

    while true; do
        check_local_servers
        if failure="$(new_failure_file)"; then
            cat "$failure" >&2 || true
            die "a remote vLLM node exited during startup"
        fi
        all_ready=1
        for ((rank = 0; rank < NNODES; rank++)); do
            [[ -s "${COORD_DIR}/endpoints.node${rank}" ]] || all_ready=0
        done
        (( all_ready == 1 )) && break
        (( SECONDS < deadline )) || die "timed out waiting for all node APIs"
        sleep 3
    done
    ENDPOINTS=()
    for ((rank = 0; rank < NNODES; rank++)); do
        mapfile -t node_endpoints <"${COORD_DIR}/endpoints.node${rank}"
        (( ${#node_endpoints[@]} == LOCAL_DP_SIZE )) || \
            die "node ${rank} published ${#node_endpoints[@]} endpoints, expected ${LOCAL_DP_SIZE}"
        ENDPOINTS+=("${node_endpoints[@]}")
    done
    echo "Cluster ready: ${#ENDPOINTS[@]} GPU endpoints, ${CONCURRENCY_PER_GPU} concurrent requests each"
}

monitor_worker_node() {
    while true; do
        if [[ -e "$STOP_FILE" && "$STOP_FILE" -nt "$STARTED_FILE" ]]; then
            stop_vllm
            touch "$ACK_FILE"
            COORD_ACTIVE=0
            echo "Node ${NODE_RANK} stopped after coordinator signal"
            return 0
        fi
        check_local_servers
        sleep 3
    done
}

run_generation() {
    local endpoint_list
    endpoint_list="$(IFS=','; printf '%s' "${ENDPOINTS[*]}")"
    PARSER2_PYTHON_BIN="$PYTHON_BIN" \
    PARSER2_SOURCE_JSONL="$SOURCE_JSONL" \
    PARSER2_MODEL="$MODEL" \
    PARSER2_MEDIA_ROOT="$MEDIA_ROOT" \
    PARSER2_DATA_ROOT="$DATA_ROOT" \
    PARSER2_REGEN_ROOT="$OUTPUT_ROOT" \
    PARSER2_FINAL_DIR="$FINAL_DIR" \
    PARSER2_PREPARED_ROOT="$PREPARED_ROOT" \
    PARSER2_ENDPOINTS="$endpoint_list" \
    PARSER2_MAX_TOKENS="$MAX_TOKENS" \
    PARSER2_CONCURRENCY_PER_ENDPOINT="$CONCURRENCY_PER_GPU" \
    PARSER2_SEQ_LENGTH="$SEQ_LENGTH" \
        bash "$RUN_ALL" generate full
}

wait_for_cluster_stop() {
    local deadline=$((SECONDS + 180))
    local rank
    local all_stopped

    touch "$STOP_FILE"
    stop_vllm
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
            die "timed out waiting for all nodes to stop; keeping ${COORD_DIR} for late workers"
        sleep 3
    done

    for ((rank = 0; rank < NNODES; rank++)); do
        rm -f -- \
            "${COORD_DIR}/started.node${rank}" \
            "${COORD_DIR}/stopped.node${rank}" \
            "${COORD_DIR}/failed.node${rank}" \
            "${COORD_DIR}/endpoints.node${rank}"
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
        wait_for_api
        if (( NODE_RANK == 0 )); then
            wait_for_cluster_apis
            run_generation
            wait_for_cluster_stop
            if [[ "$action" == "full" ]]; then
                run_prepare
                build_draft_vocab
            fi
        else
            monitor_worker_node
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
