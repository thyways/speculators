#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

PYTHON_BIN="${PARSER2_PYTHON_BIN:-${REPO_ROOT}/speculators_venv/bin/python}"
VLLM_BIN="${PARSER2_VLLM_BIN:-${REPO_ROOT}/vllm_venv/bin/vllm}"
PIPELINE="${SCRIPT_DIR}/script.py"
PREPARE_SCRIPT="${REPO_ROOT}/scripts/infinity_parser2_prepare_data.py"
PREPARE_FAST_SCRIPT="${SCRIPT_DIR}/prepare_fast.sh"

SOURCE_JSONL="${PARSER2_SOURCE_JSONL:-/home/ma-user/work/data_mllm/new_datasets/swift_merged_datasets/version_v2.1/train_v2.1.jsonl}"
# Only tags the stable record ids ("<version>:<line index>"), so it has to track
# SOURCE_JSONL or ids from two corpora become indistinguishable.
SOURCE_VERSION="${PARSER2_SOURCE_VERSION:-v2.1}"
MEDIA_ROOT="${PARSER2_MEDIA_ROOT:-/inspire/sfs/project/inf-multimodal/public}"
MODEL="${PARSER2_MODEL:-/home/ma-user/work/data_mllm/publish_models/Infinity-Parser2.1-Flash-2608}"
# Everything this pipeline produces lives under one directory:
#   regen/          sample database, generations, teacher logs
#   target_answers/ the regenerated answers as jsonl
#   dflash_data/    the prepared dataset the training recipes point at
DATA_ROOT="${PARSER2_DATA_ROOT:-/inspire/sfs/project/inf-multimodal/public/wumengke/datasets/infinity_parsers2_v2_1}"
OUTPUT_ROOT="${PARSER2_REGEN_ROOT:-${DATA_ROOT}/regen}"
FINAL_DIR="${PARSER2_FINAL_DIR:-${DATA_ROOT}/target_answers}"
PREPARED_ROOT="${PARSER2_PREPARED_ROOT:-${DATA_ROOT}/dflash_data}"

# script.py hard-checks this against the source's real line count and aborts on a
# mismatch, so it must be the exact `wc -l` of SOURCE_JSONL: 5795953 for v2.1
# (v1.12 measured 5275950).
POPULATION_SIZE="${PARSER2_POPULATION_SIZE:-5795953}"
FULL_SIZE="${PARSER2_FULL_SIZE:-1500000}"
# 0 = no intermediate pilot stage; sample straight to FULL_SIZE rows.
PILOT_SIZE="${PARSER2_PILOT_SIZE:-0}"
RESERVE_SIZE="${PARSER2_RESERVE_SIZE:-20000}"
SEED="${PARSER2_SEED:-42}"
CONVERT_WORKERS="${PARSER2_CONVERT_WORKERS:-64}"
# Cap on generated tokens. Never set this to 0: the teacher would run to its
# context limit and greedy repetition loops would hold a scheduler slot for
# hours. It has to stay at or below SEQ_LENGTH too, because prepare drops any row
# whose rendered ids exceed that -- generating past it buys nothing. Measured at
# 32768 the degenerate tail ate most of the token budget: record throughput fell
# 34.3 -> 28.8/s while token throughput held at 34.8k/s, with the finish_reason
# error rate climbing past 4%. The longest real answer in a 13k-record sample was
# 13443 tokens, so 16384 only ever truncates output that prepare would discard.
MAX_TOKENS="${PARSER2_MAX_TOKENS:-16384}"
# Half of TEACHER_MAX_NUM_SEQS, so the engine batch runs at most half full. That
# is deliberate: it caps how much work a degenerate sequence can hold hostage.
# Raise it towards TEACHER_MAX_NUM_SEQS to trade that back for throughput -- at
# 128 the batch stayed pinned at 64 running with KV at only 20%.
CONCURRENCY="${PARSER2_CONCURRENCY_PER_ENDPOINT:-32}"
REQUEST_TIMEOUT="${PARSER2_REQUEST_TIMEOUT:-3600}"
# Must match --total-seq-len in the training script: that is the per-rank token
# budget the packing sampler works with, so a longer row can never be packed.
# 16384 is what examples/train/infinity_parser2/dspark_infinity_parser2_flash_online.sh
# uses. In fast mode rows above it are dropped outright, not truncated.
SEQ_LENGTH="${PARSER2_SEQ_LENGTH:-16384}"
# The render path is bound by the endpoint, so extra clients only queue; fast
# renders in-process, where the work is pure CPU and scales with cores.
PREPROCESSING_WORKERS="${PARSER2_PREPROCESSING_WORKERS:-16}"
FAST_PREPROCESSING_WORKERS="${PARSER2_FAST_PREPROCESSING_WORKERS:-64}"
PREPROCESSING_BATCH_SIZE="${PARSER2_PREPROCESSING_BATCH_SIZE:-64}"
MINIMUM_VALID_TOKENS="${PARSER2_MINIMUM_VALID_TOKENS:-1}"
# No draft vocab mapping: this pipeline trains the draft on the verifier's full
# vocab, so d2t/t2d are unnecessary (identity no-ops the framework ignores).
# prepare no longer builds them; leave --draft-vocab-size off the training side.
TRAIN_DATA_RATIO="${PARSER2_TRAIN_DATA_RATIO:-0.99}"
SMOKE_TRAIN_DATA_RATIO="${PARSER2_SMOKE_TRAIN_DATA_RATIO:-0.90}"
PREPARE_ONLY="${PARSER2_PREPARE_ONLY:-0}"
# fast delegates to prepare_fast.sh, which tokenizes in-process and needs no
# server at all; render drives a vLLM /render endpoint. At this corpus size the
# endpoint is days of work for identical ids -- see prepare_fast.sh's header.
PREPARE_MODE="${PARSER2_PREPARE_MODE:-fast}"

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

TEACHER_GPU_IDS="${PARSER2_TEACHER_GPU_IDS:-0,1,2,3,4,5,6,7}"
TEACHER_BASE_PORT="${PARSER2_TEACHER_BASE_PORT:-8000}"
TEACHER_HOST="${PARSER2_TEACHER_HOST:-127.0.0.1}"
TEACHER_MAX_MODEL_LEN="${PARSER2_TEACHER_MAX_MODEL_LEN:-65536}"
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
  smoke     Sample, regenerate and prepare 100 rows.
  pilot     Sample, regenerate and prepare the pilot; needs
            PARSER2_PILOT_SIZE > 0.
  full      Sample, regenerate and prepare the ${FULL_SIZE}-row dataset.
            This is the one-shot entry point: teachers come up, generation
            resumes where it left off, they shut down, then prepare runs.
  status    Show local progress (default stage: full).

Model:   ${MODEL}
Source:  ${SOURCE_JSONL}
Prepare: ${PREPARE_MODE}

Set PARSER2_ENDPOINTS to comma-separated external endpoints to skip local
teacher startup. Set PARSER2_PREPARE_ONLY=1 to use completed generations only.
Set PARSER2_PREPARE_MODE=render to prepare through a vLLM /render endpoint
instead of tokenizing in-process.
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
        --source-version "$SOURCE_VERSION"
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

stage_token_freq_ratio() {
    case "$1" in
        smoke) printf '%s\n' "$SMOKE_TRAIN_DATA_RATIO" ;;
        pilot)
            (( PILOT_SIZE > 0 )) || die "stage pilot needs PARSER2_PILOT_SIZE > 0"
            printf '%s\n' "$TRAIN_DATA_RATIO"
            ;;
        full) printf '%s\n' "$TRAIN_DATA_RATIO" ;;
        *) die "unknown stage: $1" ;;
    esac
}

# The target answers that actually made it into the prepared dataset, as jsonl.
# Both prepare paths end here so their artifact sets stay identical.
export_selected() {
    local stage="$1"
    local final_path="${FINAL_DIR}/${stage}.jsonl"

    "$PYTHON_BIN" "$PIPELINE" export \
        --output-root "$OUTPUT_ROOT" \
        --stage "$stage" \
        --output "$final_path" \
        --selection-manifest "${PREPARED_ROOT}/${stage}/ranked_selection.json" \
        --report "${final_path%.jsonl}.export.json"
}

# prepare_fast.sh reads the same PARSER2_* names, so pass every shared setting
# explicitly: the two entry points must never disagree on model, paths or
# seq length. Anything it owns alone (target margin, GPU hold) it reads from the
# inherited environment.
prepare_stage_fast() {
    local stage="$1"
    local token_freq_ratio

    [[ -f "$PREPARE_FAST_SCRIPT" ]] || die "missing script: $PREPARE_FAST_SCRIPT"
    token_freq_ratio="$(stage_token_freq_ratio "$stage")"
    mkdir -p "$FINAL_DIR" "$PREPARED_ROOT"

    PARSER2_PYTHON_BIN="$PYTHON_BIN" \
    PARSER2_MODEL="$MODEL" \
    PARSER2_REGEN_ROOT="$OUTPUT_ROOT" \
    PARSER2_FINAL_DIR="$FINAL_DIR" \
    PARSER2_PREPARED_ROOT="$PREPARED_ROOT" \
    PARSER2_STAGE="$stage" \
    PARSER2_SEQ_LENGTH="$SEQ_LENGTH" \
    PARSER2_TRAIN_DATA_RATIO="$token_freq_ratio" \
    PARSER2_MINIMUM_VALID_TOKENS="$MINIMUM_VALID_TOKENS" \
    PARSER2_PREPROCESSING_WORKERS="$FAST_PREPROCESSING_WORKERS" \
    PARSER2_PREPROCESSING_BATCH_SIZE="$PREPROCESSING_BATCH_SIZE" \
        bash "$PREPARE_FAST_SCRIPT"

    export_selected "$stage"
}

prepare_stage() {
    local stage="$1"
    local target_records
    local token_freq_ratio
    local pool_path="${FINAL_DIR}/${stage}.pool.jsonl"
    local prepared_dir="${PREPARED_ROOT}/${stage}"
    local -a prepare_args

    token_freq_ratio="$(stage_token_freq_ratio "$stage")"
    case "$stage" in
        smoke) target_records=100 ;;
        pilot) target_records="$PILOT_SIZE" ;;
        full) target_records="$FULL_SIZE" ;;
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

    export_selected "$stage"
}

require_executable "$PYTHON_BIN"
[[ "$PREPARE_ONLY" == "0" || "$PREPARE_ONLY" == "1" ]] || \
    die "PARSER2_PREPARE_ONLY must be 0 or 1"
[[ "$PREPARE_MODE" == "fast" || "$PREPARE_MODE" == "render" ]] || \
    die "PARSER2_PREPARE_MODE must be fast or render"

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
        if [[ "$PREPARE_MODE" == "fast" ]]; then
            # Fast prepare needs no server, and it wants every GPU for its hold
            # script, so the teachers have to be gone before it starts.
            stop_teachers
            prepare_stage_fast "$action"
        else
            if (( ${#ENDPOINTS[@]} == 0 )); then
                # Completed generations still need one target-model server for /render.
                resolve_endpoints 1
            fi
            retain_first_local_teacher
            RENDER_ENDPOINT="$(render_endpoint_from_chat_endpoint "${ENDPOINTS[0]}")"
            prepare_stage "$action"
        fi
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
