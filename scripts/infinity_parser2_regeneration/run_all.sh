#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

PYTHON_BIN="${PARSER2_PYTHON_BIN:-${REPO_ROOT}/speculators_venv/bin/python}"
VLLM_BIN="${PARSER2_VLLM_BIN:-${REPO_ROOT}/vllm_venv/bin/vllm}"
PIPELINE="${SCRIPT_DIR}/script.py"
PREPARE_SCRIPT="${REPO_ROOT}/scripts/infinity_parser2_prepare_data.py"
VOCAB_SCRIPT="${REPO_ROOT}/scripts/build_vocab_mapping.py"

SOURCE_JSONL="${PARSER2_SOURCE_JSONL:-/home/ma-user/work/data_mllm/new_datasets/swift_merged_datasets/version_v1.12/train_v1.12.jsonl}"
MEDIA_ROOT="${PARSER2_MEDIA_ROOT:-/inspire/sfs/project/inf-multimodal/public}"
MODEL="${PARSER2_MODEL:-/home/ma-user/work/data_mllm/publish_models/Infinity-Parser2-2B-2604}"
OUTPUT_ROOT="${PARSER2_REGEN_ROOT:-/inspire/sfs/project/inf-multimodal/public/wumengke/datasets/infinity_parser2_v1_12_regen_1p5m}"
FINAL_DIR="${PARSER2_FINAL_DIR:-/inspire/sfs/project/inf-multimodal/public/wumengke/datasets/infinity_parser2_v1_12_target_answers}"
PREPARED_ROOT="${PARSER2_PREPARED_ROOT:-/inspire/sfs/project/inf-multimodal/public/wumengke/datasets/infinity_parser2_v1_12_dflash_data}"

POPULATION_SIZE="${PARSER2_POPULATION_SIZE:-5275950}"
FULL_SIZE="${PARSER2_FULL_SIZE:-800000}"
# 0 = no intermediate pilot stage; sample straight to FULL_SIZE rows.
PILOT_SIZE="${PARSER2_PILOT_SIZE:-0}"
RESERVE_SIZE="${PARSER2_RESERVE_SIZE:-20000}"
SEED="${PARSER2_SEED:-42}"
CONVERT_WORKERS="${PARSER2_CONVERT_WORKERS:-64}"
# Cap on generated tokens; 0 would let the teacher run to its context limit,
# which lets greedy repetition loops hold a scheduler slot for over an hour.
# 16384 is well past the longest answer observed in a 13k-record sample (13443
# tokens), so it only ever truncates degenerate output.
MAX_TOKENS="${PARSER2_MAX_TOKENS:-16384}"
# Deeper than TEACHER_MAX_NUM_SEQS on purpose: the engine batch must stay full
# while the client parses a response and queues the next turn.
CONCURRENCY="${PARSER2_CONCURRENCY_PER_ENDPOINT:-128}"
REQUEST_TIMEOUT="${PARSER2_REQUEST_TIMEOUT:-3600}"
SEQ_LENGTH="${PARSER2_SEQ_LENGTH:-20480}"
PREPROCESSING_WORKERS="${PARSER2_PREPROCESSING_WORKERS:-16}"
PREPROCESSING_BATCH_SIZE="${PARSER2_PREPROCESSING_BATCH_SIZE:-64}"
MINIMUM_VALID_TOKENS="${PARSER2_MINIMUM_VALID_TOKENS:-1}"
DRAFT_VOCAB_SIZE="${PARSER2_DRAFT_VOCAB_SIZE:-32000}"
TRAIN_DATA_RATIO="${PARSER2_TRAIN_DATA_RATIO:-0.99}"
SMOKE_TRAIN_DATA_RATIO="${PARSER2_SMOKE_TRAIN_DATA_RATIO:-0.90}"
PREPARE_ONLY="${PARSER2_PREPARE_ONLY:-0}"

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

TEACHER_GPU_IDS="${PARSER2_TEACHER_GPU_IDS:-0,1,2,3,4,5,6,7}"
TEACHER_BASE_PORT="${PARSER2_TEACHER_BASE_PORT:-8000}"
TEACHER_HOST="${PARSER2_TEACHER_HOST:-127.0.0.1}"
TEACHER_MAX_MODEL_LEN="${PARSER2_TEACHER_MAX_MODEL_LEN:-262144}"
TEACHER_MAX_NUM_SEQS="${PARSER2_TEACHER_MAX_NUM_SEQS:-64}"
TEACHER_MAX_IMAGES="${PARSER2_TEACHER_MAX_IMAGES:-16}"
TEACHER_START_TIMEOUT="${PARSER2_TEACHER_START_TIMEOUT:-1800}"
TEACHER_ENDPOINTS="${PARSER2_ENDPOINTS:-}"

declare -a TEACHER_PIDS=()
declare -a ENDPOINTS=()
RENDER_ENDPOINT=""

usage() {
    cat <<EOF
Usage: $(basename "$0") sample|generate|smoke|pilot|full|status [stage]

  sample    Build the deterministic sample database.
  generate  Regenerate one stage only, without the dflash preparation
            (default stage: full).
  smoke     Regenerate and prepare 100 rows.
  pilot     Regenerate and prepare the pilot; needs PARSER2_PILOT_SIZE > 0.
  full      Regenerate and prepare the ${FULL_SIZE}-row dataset.
  status    Show local progress (default stage: full).

Set PARSER2_ENDPOINTS to comma-separated external endpoints to skip local
teacher startup. Set PARSER2_PREPARE_ONLY=1 to use completed generations only.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_executable() {
    [[ -x "$1" ]] || die "missing executable: $1"
}

stop_teacher_pids() {
    local -a pids=("$@")
    local pid
    local alive
    local attempt

    for pid in "${pids[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    for attempt in {1..30}; do
        alive=0
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive=1
        done
        (( alive == 0 )) && break
        sleep 1
    done
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    done
}

stop_teachers() {
    stop_teacher_pids "${TEACHER_PIDS[@]}"
    TEACHER_PIDS=()
}

retain_first_local_teacher() {
    local -a extra_pids=()
    (( ${#TEACHER_PIDS[@]} <= 1 )) && return 0
    extra_pids=("${TEACHER_PIDS[@]:1}")
    stop_teacher_pids "${extra_pids[@]}"
    TEACHER_PIDS=("${TEACHER_PIDS[0]}")
    ENDPOINTS=("${ENDPOINTS[0]}")
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM HUP
    stop_teachers
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

sample_data() {
    local -a args=(
        "$PYTHON_BIN" "$PIPELINE" sample
        --source "$SOURCE_JSONL"
        --output-root "$OUTPUT_ROOT"
        --population-size "$POPULATION_SIZE"
        --sample-size "$FULL_SIZE"
        --reserve-size "$RESERVE_SIZE"
        --seed "$SEED"
        --convert-workers "$CONVERT_WORKERS"
        --source-version v1.12
        --allowed-media-root "$MEDIA_ROOT"
        --path-map "/home/ma-user/work/=${MEDIA_ROOT}/"
    )
    (( PILOT_SIZE > 0 )) && args+=(--pilot-size "$PILOT_SIZE")
    [[ "${PARSER2_OVERWRITE_SAMPLE:-0}" == "1" ]] && args+=(--overwrite)
    "${args[@]}"
}

start_teachers() {
    local teacher_count="${1:-0}"
    local -a gpus=()
    local -a extra_args=()
    local gpu
    local index
    local port
    local pid
    local log_dir="${OUTPUT_ROOT}/teacher_logs"

    require_executable "$VLLM_BIN"
    command -v curl >/dev/null || die "curl is required"
    command -v setsid >/dev/null || die "setsid is required"
    IFS=',' read -r -a gpus <<< "$TEACHER_GPU_IDS"
    [[ ${#gpus[@]} -gt 0 ]] || die "PARSER2_TEACHER_GPU_IDS is empty"
    if (( teacher_count > 0 )); then
        (( teacher_count <= ${#gpus[@]} )) || \
            die "requested ${teacher_count} teachers from only ${#gpus[@]} GPUs"
        gpus=("${gpus[@]:0:teacher_count}")
    fi
    [[ -n "${PARSER2_VLLM_EXTRA_ARGS:-}" ]] && \
        read -r -a extra_args <<< "$PARSER2_VLLM_EXTRA_ARGS"
    mkdir -p "$log_dir"

    for index in "${!gpus[@]}"; do
        gpu="${gpus[$index]//[[:space:]]/}"
        port=$((TEACHER_BASE_PORT + index))
        # hs_connectors lives in the workspace, not in vllm_venv. Without it on
        # the path the speculators vLLM plugins fail to import and dump a
        # traceback per engine process at startup; harmless for a plain teacher,
        # but it buries the real log.
        setsid env CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="${REPO_ROOT}/hs_connectors/src${PYTHONPATH:+:${PYTHONPATH}}" \
            "$VLLM_BIN" serve "$MODEL" \
            --host "$TEACHER_HOST" \
            --port "$port" \
            --served-model-name "$MODEL" \
            --tensor-parallel-size 1 \
            --max-model-len "$TEACHER_MAX_MODEL_LEN" \
            --max-num-seqs "$TEACHER_MAX_NUM_SEQS" \
            --allowed-local-media-path "$MEDIA_ROOT" \
            --limit-mm-per-prompt "{\"image\":${TEACHER_MAX_IMAGES}}" \
            "${extra_args[@]}" \
            >"${log_dir}/teacher_${index}.log" 2>&1 &
        pid=$!
        TEACHER_PIDS+=("$pid")
        ENDPOINTS+=("http://${TEACHER_HOST}:${port}/v1/chat/completions")
    done

    local deadline=$((SECONDS + TEACHER_START_TIMEOUT))
    local ready
    while true; do
        ready=0
        for index in "${!TEACHER_PIDS[@]}"; do
            pid="${TEACHER_PIDS[$index]}"
            port=$((TEACHER_BASE_PORT + index))
            if ! kill -0 "$pid" 2>/dev/null; then
                tail -n 80 "${log_dir}/teacher_${index}.log" >&2 || true
                die "teacher ${index} exited during startup"
            fi
            if curl -fsS --max-time 5 \
                "http://${TEACHER_HOST}:${port}/health" >/dev/null 2>&1; then
                ready=$((ready + 1))
            fi
        done
        (( ready == ${#TEACHER_PIDS[@]} )) && break
        (( SECONDS < deadline )) || die "teacher startup timed out"
        echo "Waiting for teachers: ${ready}/${#TEACHER_PIDS[@]} ready"
        sleep 5
    done
}

resolve_endpoints() {
    local local_teacher_count="${1:-0}"
    local endpoint
    if [[ -n "$TEACHER_ENDPOINTS" ]]; then
        IFS=',' read -r -a ENDPOINTS <<< "$TEACHER_ENDPOINTS"
        for endpoint in "${!ENDPOINTS[@]}"; do
            ENDPOINTS[$endpoint]="${ENDPOINTS[$endpoint]//[[:space:]]/}"
        done
    else
        start_teachers "$local_teacher_count"
    fi
}

render_endpoint_from_chat_endpoint() {
    local endpoint="${1%/}"
    case "$endpoint" in
        */v1/chat/completions/render)
            endpoint="${endpoint%/v1/chat/completions/render}"
            ;;
        */v1/chat/completions)
            endpoint="${endpoint%/v1/chat/completions}"
            ;;
        */v1)
            endpoint="${endpoint%/v1}"
            ;;
    esac
    [[ -n "$endpoint" ]] || die "cannot derive render endpoint from $1"
    printf '%s\n' "$endpoint"
}

generate_stage() {
    local stage="$1"
    local endpoint
    local -a args=(
        "$PYTHON_BIN" "$PIPELINE" generate
        --output-root "$OUTPUT_ROOT"
        --model-path "$MODEL"
        --model "$MODEL"
        --stage "$stage"
        --max-tokens "$MAX_TOKENS"
        --temperature 0
        --top-p 1
        --seed "$SEED"
        --concurrency-per-endpoint "$CONCURRENCY"
        --timeout "$REQUEST_TIMEOUT"
        --connect-timeout 30
        --max-retries 5
    )
    for endpoint in "${ENDPOINTS[@]}"; do
        args+=(--endpoint "$endpoint")
    done
    [[ "${PARSER2_RETRY_ERRORS:-0}" == "1" ]] && args+=(--retry-errors)
    [[ "${PARSER2_ALLOW_CONFIG_CHANGE:-0}" == "1" ]] && args+=(--allow-config-change)
    "${args[@]}"
}

prepare_stage() {
    local stage="$1"
    local target_records
    local token_freq_ratio
    local pool_path="${FINAL_DIR}/${stage}.pool.jsonl"
    local final_path="${FINAL_DIR}/${stage}.jsonl"
    local prepared_dir="${PREPARED_ROOT}/${stage}"
    local -a prepare_args

    case "$stage" in
        smoke)
            target_records=100
            token_freq_ratio="$SMOKE_TRAIN_DATA_RATIO"
            ;;
        pilot)
            (( PILOT_SIZE > 0 )) || die "stage pilot needs PARSER2_PILOT_SIZE > 0"
            target_records="$PILOT_SIZE"
            token_freq_ratio="$TRAIN_DATA_RATIO"
            ;;
        full)
            target_records="$FULL_SIZE"
            token_freq_ratio="$TRAIN_DATA_RATIO"
            ;;
        *) die "unknown stage: $stage" ;;
    esac

    mkdir -p "$FINAL_DIR" "$PREPARED_ROOT"
    "$PYTHON_BIN" "$PIPELINE" export \
        --output-root "$OUTPUT_ROOT" \
        --stage "$stage" \
        --output "$pool_path" \
        --minimum-count "$target_records" \
        --report "${pool_path%.jsonl}.export.json"

    prepare_args=(
        "$PYTHON_BIN" "$PREPARE_SCRIPT"
        --model "$MODEL"
        --data "$pool_path"
        --output "$prepared_dir"
        --seq-length "$SEQ_LENGTH"
        --ranked-target-samples "$target_records"
        --token-freq-train-ratio "$token_freq_ratio"
        --render-endpoint "$RENDER_ENDPOINT"
        --minimum-valid-tokens "$MINIMUM_VALID_TOKENS"
        --num-preprocessing-workers "$PREPROCESSING_WORKERS"
        --preprocessing-batch-size "$PREPROCESSING_BATCH_SIZE"
    )
    [[ "${PARSER2_OVERWRITE_PREPARED:-0}" == "1" ]] && \
        prepare_args+=(--overwrite)
    "${prepare_args[@]}"

    "$PYTHON_BIN" "$VOCAB_SCRIPT" \
        --token-freq-path "${prepared_dir}/token_freq.pt" \
        --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
        --target-model-path "$MODEL" \
        --output-path "$prepared_dir"

    "$PYTHON_BIN" "$PIPELINE" export \
        --output-root "$OUTPUT_ROOT" \
        --stage "$stage" \
        --output "$final_path" \
        --selection-manifest "${prepared_dir}/ranked_selection.json" \
        --report "${final_path%.jsonl}.export.json"
}

require_executable "$PYTHON_BIN"
[[ "$PREPARE_ONLY" == "0" || "$PREPARE_ONLY" == "1" ]] || \
    die "PARSER2_PREPARE_ONLY must be 0 or 1"

action="${1:-}"
case "$action" in
    sample)
        sample_data
        ;;
    generate)
        stage="${2:-full}"
        sample_data
        if ! "$PYTHON_BIN" "$PIPELINE" status \
            --output-root "$OUTPUT_ROOT" \
            --stage "$stage" \
            --require-complete >/dev/null; then
            resolve_endpoints
            generate_stage "$stage"
        fi
        "$PYTHON_BIN" "$PIPELINE" status \
            --output-root "$OUTPUT_ROOT" \
            --stage "$stage"
        ;;
    status)
        "$PYTHON_BIN" "$PIPELINE" status \
            --output-root "$OUTPUT_ROOT" \
            --stage "${2:-full}"
        ;;
    smoke|pilot|full)
        sample_data
        if [[ "$PREPARE_ONLY" == "0" ]]; then
            if ! "$PYTHON_BIN" "$PIPELINE" status \
                --output-root "$OUTPUT_ROOT" \
                --stage "$action" \
                --require-complete >/dev/null; then
                resolve_endpoints
                generate_stage "$action"
            fi
        else
            "$PYTHON_BIN" "$PIPELINE" status \
                --output-root "$OUTPUT_ROOT" \
                --stage "$action" \
                --require-complete
        fi
        "$PYTHON_BIN" "$PIPELINE" status \
            --output-root "$OUTPUT_ROOT" \
            --stage "$action"
        if (( ${#ENDPOINTS[@]} == 0 )); then
            # Completed generations still need one target-model server for /render.
            resolve_endpoints 1
        fi
        retain_first_local_teacher
        RENDER_ENDPOINT="$(render_endpoint_from_chat_endpoint "${ENDPOINTS[0]}")"
        prepare_stage "$action"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
