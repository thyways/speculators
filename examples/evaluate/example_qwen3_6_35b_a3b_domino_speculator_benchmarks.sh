#!/usr/bin/env bash
# Evaluate the local Qwen3.6-35B-A3B Domino checkpoint with
# RedHatAI/speculator_benchmarks.
#
# The default run mirrors example_qwen3_8b_dflash_humaneval.sh's
# acceptance-only evaluation, but covers all nine benchmark subsets. Override
# SUBSETS or MAX_REQUESTS for a smaller smoke test, for example:
#
#   SUBSETS=HumanEval MAX_REQUESTS=4 \
#     bash examples/evaluate/example_qwen3_6_35b_a3b_domino_speculator_benchmarks.sh
#
# Set MODE=sweep for the full output-length estimation and rate sweep.

set -Eeuo pipefail

WORKSPACE="/inspire/sfs/project/inf-multimodal/public/wumengke"
REPO="$WORKSPACE/speculators"
MODEL="$WORKSPACE/model_weights/domino_qwen3_6_35b_a3b_perfectblend_online_500k/checkpoints/0"
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
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-1}"
if [[ -z "${GEN_KWARGS+x}" ]]; then
    GEN_KWARGS="{\"temperature\":${TEMPERATURE}}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/evaluation_results/domino_qwen3_6_35b_a3b_ckpt0_${TIMESTAMP}}"
VLLM_LOG="$OUTPUT_DIR/vllm.log"

VLLM_PYTHON="$REPO/vllm_venv/bin/python"
EVAL_PYTHON="$REPO/speculators_venv/bin/python"
GUIDELLM="$REPO/speculators_venv/bin/guidellm"
EVALUATE_PY="$REPO/scripts/evaluate/evaluate.py"

# The checkpoint was trained from the Domino integration branch. The main
# worktree may be on another branch, so materialize the matching serving source
# without switching or modifying that worktree.
DOMINO_SOURCE_COMMIT="${DOMINO_SOURCE_COMMIT:-231dd13fbac522d4a675fe822f726d74eb6c659c}"
DOMINO_SOURCE_DIR=""
VLLM_PID=""

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

    if [[ -n "$DOMINO_SOURCE_DIR" && -d "$DOMINO_SOURCE_DIR" ]]; then
        find "$DOMINO_SOURCE_DIR" -mindepth 1 -delete
        rmdir "$DOMINO_SOURCE_DIR" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for executable in "$VLLM_PYTHON" "$EVAL_PYTHON" "$GUIDELLM"; do
    if [[ ! -x "$executable" ]]; then
        echo "Missing executable: $executable" >&2
        echo "Install evaluation dependencies with:" >&2
        echo "  uv --system-certs pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \\" >&2
        echo "    --python $REPO/speculators_venv/bin/python guidellm==0.7.3" >&2
        exit 1
    fi
done

for path in "$MODEL/config.json" "$MODEL/model.safetensors" "$EVALUATE_PY"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

if [[ "$MODE" != "throughput" && "$MODE" != "sweep" ]]; then
    echo "MODE must be 'throughput' or 'sweep', got: $MODE" >&2
    exit 1
fi

if ! git -C "$REPO" cat-file -e "${DOMINO_SOURCE_COMMIT}^{commit}"; then
    echo "Domino source commit is unavailable: $DOMINO_SOURCE_COMMIT" >&2
    exit 1
fi

DOMINO_SOURCE_DIR="$(mktemp -d /tmp/speculators-domino-eval.XXXXXX)"
git -C "$REPO" archive "$DOMINO_SOURCE_COMMIT" \
    src/speculators hs_connectors/src/hs_connectors \
    | tar -x -C "$DOMINO_SOURCE_DIR"
PLUGIN_PYTHONPATH="$DOMINO_SOURCE_DIR/src:$DOMINO_SOURCE_DIR/hs_connectors/src"

mkdir -p "$OUTPUT_DIR"

echo "=== Configuration ==="
echo "Model:             $MODEL"
echo "Dataset:           $DATASET"
echo "HF endpoint:       $HF_ENDPOINT"
echo "Subsets:           $SUBSETS"
echo "Mode:              $MODE"
echo "Max requests:      $MAX_REQUESTS"
echo "Max concurrency:   $MAX_CONCURRENCY"
echo "Max output tokens: $MAX_OUTPUT_TOKENS"
echo "Generation kwargs: $GEN_KWARGS"
echo "CUDA devices:      $CUDA_VISIBLE_DEVICES"
echo "API port:          $VLLM_PORT"
echo "vLLM internal port:$VLLM_INTERNAL_PORT"
echo "Output:            $OUTPUT_DIR"

echo "=== Step 1: Launching vLLM server ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    HF_ENDPOINT="$HF_ENDPOINT" \
    PYTHONPATH="$PLUGIN_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
    VLLM_PLUGINS=speculators_domino \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_PORT="$VLLM_INTERNAL_PORT" \
    TOKENIZERS_PARALLELISM=false \
    "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
        --host 127.0.0.1 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --data-parallel-size 1 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --trust-remote-code \
        --enable-per-request-metrics \
        >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server (log: $VLLM_LOG)..."
until curl -sf "${SERVER_URL}/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming healthy. Last log lines:" >&2
        tail -n 120 "$VLLM_LOG" >&2
        wait "$VLLM_PID" || true
        exit 1
    fi
    sleep 2
done
echo "vLLM server ready."

echo "=== Step 2: Running $MODE evaluation ==="
env HF_ENDPOINT="$HF_ENDPOINT" PATH="$REPO/speculators_venv/bin:$PATH" \
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
