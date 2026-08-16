#!/usr/bin/env bash
# Evaluate the final dual-stream raw-KV DFlash checkpoint with
# RedHatAI/speculator_benchmarks. This serves one TP=1/PP=1 replica because the
# current raw-KV implementation reads both verifier KV heads on one rank.
#
# Smoke test:
#
#   SUBSETS=HumanEval MAX_REQUESTS=4 \
#     bash examples/evaluate/example_qwen3_6_35b_a3b_kv_native_dflash_speculator_benchmarks.sh
#
# Set MODE=sweep for output-length estimation and a multi-rate sweep.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

REPO="${REPO:-$DEFAULT_REPO}"
WORKSPACE="${WORKSPACE:-$(dirname -- "$REPO")}"
RUNTIME_REPO="${RUNTIME_REPO:-$WORKSPACE/speculators}"
MODEL="${MODEL:-$WORKSPACE/model_weights/kv_native_dflash_qwen3_6_35b_a3b_dual_stream_raw_kv_final_5full/checkpoints/0}"
VERIFIER_MODEL="${VERIFIER_MODEL:-$WORKSPACE/model_weights/Qwen/Qwen3.6-35B-A3B}"
DATASET="RedHatAI/speculator_benchmarks"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

VLLM_PORT="${VLLM_PORT:-8108}"
VLLM_INTERNAL_PORT="${VLLM_INTERNAL_PORT:-29500}"
SERVER_URL="http://127.0.0.1:${VLLM_PORT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODE="${MODE:-throughput}"
SUBSETS="${SUBSETS:-HumanEval,math_reasoning,qa,question,rag,summarization,tool_call,translation,writing}"
MAX_REQUESTS="${MAX_REQUESTS:-200}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0}"
if [[ -z "${GEN_KWARGS+x}" ]]; then
    GEN_KWARGS="{\"temperature\":${TEMPERATURE}}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/evaluation_results/kv_native_dflash_qwen3_6_35b_a3b_ckpt0_spec${NUM_SPECULATIVE_TOKENS}_${TIMESTAMP}}"
VLLM_LOG="$OUTPUT_DIR/vllm.log"

VLLM_PYTHON="$RUNTIME_REPO/vllm_venv/bin/python"
EVAL_PYTHON="$RUNTIME_REPO/speculators_venv/bin/python"
GUIDELLM="$RUNTIME_REPO/speculators_venv/bin/guidellm"
EVALUATE_PY="$RUNTIME_REPO/scripts/evaluate/evaluate.py"

VLLM_PID=""
PLUGIN_METADATA_ROOT=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM

    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server..."
        kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi

    if [[ -n "$PLUGIN_METADATA_ROOT" && -d "$PLUGIN_METADATA_ROOT" ]]; then
        find "$PLUGIN_METADATA_ROOT" -mindepth 1 -delete
        rmdir "$PLUGIN_METADATA_ROOT" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for executable in "$VLLM_PYTHON" "$EVAL_PYTHON" "$GUIDELLM"; do
    if [[ ! -x "$executable" ]]; then
        echo "Missing executable: $executable" >&2
        exit 1
    fi
done

for path in \
    "$MODEL/config.json" \
    "$MODEL/model.safetensors" \
    "$VERIFIER_MODEL/config.json" \
    "$REPO/src/speculators/vllm/kv_native_dflash.py" \
    "$REPO/src/speculators/vllm/kv_native_dflash_model.py" \
    "$EVALUATE_PY"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

if [[ "$MODE" != "throughput" && "$MODE" != "sweep" ]]; then
    echo "MODE must be 'throughput' or 'sweep', got: $MODE" >&2
    exit 1
fi
if [[ "$NUM_SPECULATIVE_TOKENS" != "15" ]]; then
    echo "Final block-16 KV-native DFlash requires NUM_SPECULATIVE_TOKENS=15" >&2
    exit 1
fi
if [[ "$VLLM_USE_V2_MODEL_RUNNER" != "0" && "$VLLM_USE_V2_MODEL_RUNNER" != "1" ]]; then
    echo "VLLM_USE_V2_MODEL_RUNNER must be 0 or 1" >&2
    exit 1
fi

# The shared vLLM environment is editable-installed from another worktree, so
# its dist-info may predate this new plugin entry point. Add a tiny temporary
# distribution metadata shim when needed; PYTHONPATH still points at this
# worktree's real source, and the environment itself is left untouched.
if ! "$VLLM_PYTHON" - <<'PY'
from importlib.metadata import entry_points

plugins = entry_points(group="vllm.general_plugins")
raise SystemExit(
    not any(item.name == "speculators_kv_native_dflash" for item in plugins)
)
PY
then
    PLUGIN_METADATA_ROOT="$(mktemp -d /tmp/speculators-kv-native-plugin.XXXXXX)"
    DIST_INFO="$PLUGIN_METADATA_ROOT/speculators_kv_native_runtime-0.0.0.dist-info"
    mkdir -p "$DIST_INFO"
    printf '%s\n' \
        'Metadata-Version: 2.1' \
        'Name: speculators-kv-native-runtime' \
        'Version: 0.0.0' > "$DIST_INFO/METADATA"
    printf '%s\n' \
        '[vllm.general_plugins]' \
        'speculators_kv_native_dflash = speculators.vllm.kv_native_dflash:register' \
        > "$DIST_INFO/entry_points.txt"
fi

PLUGIN_PYTHONPATH="$REPO/src:$REPO/hs_connectors/src"
if [[ -n "$PLUGIN_METADATA_ROOT" ]]; then
    PLUGIN_PYTHONPATH="$PLUGIN_PYTHONPATH:$PLUGIN_METADATA_ROOT"
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Configuration ==="
echo "Draft model:           $MODEL"
echo "Verifier model:        $VERIFIER_MODEL"
echo "Speculative tokens:    $NUM_SPECULATIVE_TOKENS"
echo "vLLM model runner:      V$((VLLM_USE_V2_MODEL_RUNNER + 1))"
echo "Plugin source:         $REPO"
echo "Dataset:               $DATASET"
echo "HF endpoint:           $HF_ENDPOINT"
echo "Subsets:               $SUBSETS"
echo "Mode:                  $MODE"
echo "Max requests:          $MAX_REQUESTS"
echo "Max concurrency:       $MAX_CONCURRENCY"
echo "Max output tokens:     $MAX_OUTPUT_TOKENS"
echo "Generation kwargs:     $GEN_KWARGS"
echo "CUDA devices:          $CUDA_VISIBLE_DEVICES"
echo "API port:              $VLLM_PORT"
echo "vLLM internal port:    $VLLM_INTERNAL_PORT"
echo "Output:                $OUTPUT_DIR"

echo "=== Step 1: Launching vLLM server ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    HF_ENDPOINT="$HF_ENDPOINT" \
    PYTHONPATH="$PLUGIN_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
    VLLM_PLUGINS=speculators_kv_native_dflash \
    VLLM_USE_V2_MODEL_RUNNER="$VLLM_USE_V2_MODEL_RUNNER" \
    VLLM_PORT="$VLLM_INTERNAL_PORT" \
    TOKENIZERS_PARALLELISM=false \
    "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$VERIFIER_MODEL" \
        --host 127.0.0.1 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 1 \
        --data-parallel-size 1 \
        --spec-model "$MODEL" \
        --spec-method dflash \
        --spec-tokens "$NUM_SPECULATIVE_TOKENS" \
        --dtype bfloat16 \
        --kv-cache-dtype auto \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --trust-remote-code \
        --language-model-only \
        --enable-per-request-metrics \
        >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server (log: $VLLM_LOG)..."
until curl -sf "${SERVER_URL}/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming healthy. Last log lines:" >&2
        tail -n 160 "$VLLM_LOG" >&2
        wait "$VLLM_PID" || true
        exit 1
    fi
    sleep 2
done
echo "vLLM server ready."

echo "=== Step 2: Running $MODE evaluation ==="
env HF_ENDPOINT="$HF_ENDPOINT" PATH="$RUNTIME_REPO/speculators_venv/bin:$PATH" \
    "$EVAL_PYTHON" "$EVALUATE_PY" \
        --target "${SERVER_URL}/v1" \
        --dataset "$DATASET" \
        --data-column-mapper \
            "kind=generative_column_mapper,column_mappings.text_column=prompt" \
        --subsets "$SUBSETS" \
        --output-dir "$OUTPUT_DIR" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --max-requests "$MAX_REQUESTS" \
        --max-output-tokens "$MAX_OUTPUT_TOKENS" \
        --gen-kwargs "$GEN_KWARGS" \
        "$MODE"

echo "Done. Results: $OUTPUT_DIR"
