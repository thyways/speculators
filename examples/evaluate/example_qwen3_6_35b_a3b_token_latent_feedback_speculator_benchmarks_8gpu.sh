#!/usr/bin/env bash
# Run the five-layer Qwen3.6-35B-A3B token-latent-feedback checkpoint on eight
# single-GPU vLLM replicas. Nine benchmark subsets are sharded across the eight
# workers, then acceptance and latency CSVs are merged.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE="${WORKSPACE:-$(dirname -- "$REPO")}"
RUNTIME_REPO="${RUNTIME_REPO:-$WORKSPACE/speculators}"
SINGLE_GPU_SCRIPT="$REPO/examples/evaluate/example_qwen3_6_35b_a3b_token_latent_feedback_speculator_benchmarks.sh"
MODEL="${MODEL:-$WORKSPACE/model_weights/token_latent_feedback_qwen3_6_35b_a3b_5swa/checkpoints/0}"
VERIFIER_MODEL="${VERIFIER_MODEL:-$WORKSPACE/model_weights/Qwen--Qwen3.6-35B-A3B}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-8108}"
BASE_INTERNAL_PORT="${BASE_INTERNAL_PORT:-20000}"
MAX_REQUESTS="${MAX_REQUESTS:-200}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0}"
PREFIX_CACHING="${PREFIX_CACHING:-default}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-2}"

if [[ -z "${GEN_KWARGS+x}" ]]; then
    GEN_KWARGS="{\"temperature\":${TEMPERATURE}}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKSPACE/evaluation_results/token_latent_feedback_qwen3_6_35b_a3b_5swa_ckpt0_spec${NUM_SPECULATIVE_TOKENS}_8gpu}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
WORKLOADS=(
    "HumanEval"
    "math_reasoning"
    "qa,translation"
    "question"
    "rag"
    "summarization"
    "tool_call"
    "writing"
)

if [[ "${#GPU_ARRAY[@]}" -ne "${#WORKLOADS[@]}" ]]; then
    echo "Expected 8 comma-separated GPU IDs, got: $GPU_IDS" >&2
    exit 1
fi
if [[ ! "$NUM_SPECULATIVE_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SPECULATIVE_TOKENS must be a positive integer, got: $NUM_SPECULATIVE_TOKENS" >&2
    exit 1
fi
if [[ ! -x "$SINGLE_GPU_SCRIPT" ]]; then
    echo "Missing executable script: $SINGLE_GPU_SCRIPT" >&2
    exit 1
fi
for path in \
    "$MODEL/config.json" \
    "$MODEL/model.safetensors" \
    "$VERIFIER_MODEL/config.json"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

for index in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$index]}"
    port=$((BASE_PORT + index))
    internal_port=$((BASE_INTERNAL_PORT + index * 100))
    memory_used="$(
        nvidia-smi --id="$gpu" --query-gpu=memory.used \
            --format=csv,noheader,nounits | tr -d '[:space:]'
    )"
    if [[ ! "$memory_used" =~ ^[0-9]+$ ]] || ((memory_used > 1024)); then
        echo "GPU $gpu is not idle (memory used: ${memory_used:-unknown} MiB)." >&2
        exit 1
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
        echo "Port $port is already in use." >&2
        exit 1
    fi
    for offset in {0..9}; do
        candidate_port=$((internal_port + offset))
        if (exec 3<>"/dev/tcp/127.0.0.1/$candidate_port") 2>/dev/null; then
            echo "Internal port $candidate_port is already in use." >&2
            exit 1
        fi
    done
done

mkdir -p "$OUTPUT_ROOT"

declare -a PIDS=()
declare -a LABELS=()

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM -- "-$pid" 2>/dev/null || true
        fi
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "=== Eight-GPU token-latent feedback evaluation ==="
echo "Draft model:         $MODEL"
echo "Verifier model:      $VERIFIER_MODEL"
echo "Speculative tokens:  $NUM_SPECULATIVE_TOKENS"
echo "GPUs:                $GPU_IDS"
echo "Per-GPU concurrency: $MAX_CONCURRENCY"
echo "Global concurrency:  ${#GPU_ARRAY[@]}"
echo "Max model length:    $MAX_MODEL_LEN"
echo "Max output tokens:   $MAX_OUTPUT_TOKENS"
echo "Generation kwargs:   $GEN_KWARGS"
echo "Prefix caching:      $PREFIX_CACHING"
echo "Request endpoint:    /v1/chat/completions"
echo "HF endpoint:         $HF_ENDPOINT"
echo "Output root:         $OUTPUT_ROOT"

for index in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$index]}"
    subsets="${WORKLOADS[$index]}"
    port=$((BASE_PORT + index))
    internal_port=$((BASE_INTERNAL_PORT + index * 100))
    safe_subsets="${subsets//,/_}"
    worker_dir="$OUTPUT_ROOT/gpu${gpu}_${safe_subsets}"
    runner_log="$worker_dir/runner.log"
    mkdir -p "$worker_dir"

    echo "Launching GPU $gpu on API port $port (internal $internal_port) for: $subsets"
    setsid env \
        REPO="$REPO" \
        WORKSPACE="$WORKSPACE" \
        RUNTIME_REPO="$RUNTIME_REPO" \
        MODEL="$MODEL" \
        VERIFIER_MODEL="$VERIFIER_MODEL" \
        NUM_SPECULATIVE_TOKENS="$NUM_SPECULATIVE_TOKENS" \
        CUDA_VISIBLE_DEVICES="$gpu" \
        VLLM_PORT="$port" \
        VLLM_INTERNAL_PORT="$internal_port" \
        SUBSETS="$subsets" \
        OUTPUT_DIR="$worker_dir" \
        MAX_REQUESTS="$MAX_REQUESTS" \
        MAX_CONCURRENCY="$MAX_CONCURRENCY" \
        MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
        MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" \
        TEMPERATURE="$TEMPERATURE" \
        PREFIX_CACHING="$PREFIX_CACHING" \
        GEN_KWARGS="$GEN_KWARGS" \
        HF_ENDPOINT="$HF_ENDPOINT" \
        MODE=throughput \
        bash "$SINGLE_GPU_SCRIPT" >"$runner_log" 2>&1 &
    PIDS+=("$!")
    LABELS+=("GPU $gpu ($subsets)")

    if [[ "$STARTUP_STAGGER_SECONDS" != "0" && "$index" -lt 7 ]]; then
        sleep "$STARTUP_STAGGER_SECONDS"
    fi
done

failed=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "Completed: ${LABELS[$index]}"
    else
        status=$?
        echo "Failed (exit $status): ${LABELS[$index]}" >&2
        failed=1
    fi
done

mapfile -t acceptance_files < <(
    find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -name acceptance.csv -type f | sort
)
if [[ "${#acceptance_files[@]}" -gt 0 ]]; then
    awk 'FNR == 1 && header_seen++ { next } { print }' \
        "${acceptance_files[@]}" > "$OUTPUT_ROOT/acceptance.csv"
    echo "Merged acceptance results: $OUTPUT_ROOT/acceptance.csv"
fi

mapfile -t performance_files < <(
    find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -name perf_results.csv -type f | sort
)
if [[ "${#performance_files[@]}" -gt 0 ]]; then
    awk 'FNR == 1 && header_seen++ { next } { print }' \
        "${performance_files[@]}" > "$OUTPUT_ROOT/perf_results.csv"
    echo "Merged latency results: $OUTPUT_ROOT/perf_results.csv"
fi

if [[ "$failed" -ne 0 ]]; then
    echo "One or more workers failed. Inspect */runner.log under $OUTPUT_ROOT." >&2
    exit 1
fi

echo "All subsets complete: $OUTPUT_ROOT"
