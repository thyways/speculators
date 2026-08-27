#!/usr/bin/env bash
# Evaluate the local five-layer Qwen3.6-35B-A3B DFlash2 checkpoint with
# RedHatAI/speculator_benchmarks. Throughput mode records speculative
# acceptance metrics together with latency, TTFT, ITL, and throughput metrics.
#
# Same shape as example_qwen3_6_35b_a3b_dflash_speculator_benchmarks.sh, with
# three DFlash2-specific differences:
#
#   * NUM_SPECULATIVE_TOKENS is block_size - 1, not a free knob. DFlash2's
#     convolution and selector are trained over a block of `block_size` query
#     slots and at inference that block is 1 + num_speculative_tokens, so serving
#     a different width convolves over a block the checkpoint never saw. This
#     script defaults to the checkpoint's own value and re-checks it below.
#   * vLLM 0.28+ carries the DFlash2 model, selector and proposal runtime. New
#     Speculators checkpoints also write the native DFlash2 config fields, so no
#     external vLLM plugin is loaded.
#   * VLLM_USE_V2_MODEL_RUNNER=1 is mandatory, not an optimization: the V1
#     DFlashProposer has no candidate selector, so a DFlash2 checkpoint reaching
#     it drafts as DFlash1.
#
# Smoke test:
#
#   SUBSETS=HumanEval MAX_REQUESTS=4 \
#     bash examples/evaluate/example_qwen3_6_35b_a3b_dflash2_speculator_benchmarks.sh
#
# This repo's scripts/evaluate/evaluate.py fixes throughput-mode output length
# at 4096 tokens and has no --max-output-tokens flag, so there is no knob for it
# here. Set MODE=sweep to have it estimate the per-subset length instead.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE="${WORKSPACE:-$(dirname -- "$REPO")}"
RUNTIME_REPO="${RUNTIME_REPO:-$WORKSPACE/speculators}"

MODEL="${MODEL:-$WORKSPACE/model_weights/dflash2_qwen3_6_35b_a3b_5full/checkpoints/0}"
VERIFIER_MODEL="${VERIFIER_MODEL:-$WORKSPACE/model_weights/Qwen/Qwen3.6-35B-A3B}"
DATASET="${DATASET:-RedHatAI/speculator_benchmarks}"
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
# block_size 8 -> 7 drafted tokens (the anchor is the bonus token).
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TEMPERATURE="${TEMPERATURE:-1}"
VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
if [[ -z "${GEN_KWARGS+x}" ]]; then
    GEN_KWARGS="{\"temperature\":${TEMPERATURE}}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/evaluation_results/dflash2_qwen3_6_35b_a3b_5full_ckpt0_spec${NUM_SPECULATIVE_TOKENS}_${TIMESTAMP}}"
VLLM_LOG="$OUTPUT_DIR/vllm.log"

VLLM_PYTHON="${VLLM_PYTHON:-$RUNTIME_REPO/vllm_venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-$RUNTIME_REPO/speculators_venv/bin/python}"
GUIDELLM="${GUIDELLM:-$RUNTIME_REPO/speculators_venv/bin/guidellm}"
EVALUATE_PY="${EVALUATE_PY:-$REPO/scripts/evaluate/evaluate.py}"

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
        echo "    --python $RUNTIME_REPO/speculators_venv/bin/python guidellm==0.7.3" >&2
        exit 1
    fi
done

for path in \
    "$MODEL/config.json" \
    "$MODEL/model.safetensors" \
    "$VERIFIER_MODEL/config.json" \
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
if [[ ! "$NUM_SPECULATIVE_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SPECULATIVE_TOKENS must be a positive integer, got: $NUM_SPECULATIVE_TOKENS" >&2
    exit 1
fi

# The draft config decides the convolution's block boundary, so a mismatch here
# is a silently different model rather than a shorter draft. Fail before the
# server spends minutes loading the verifier.
CHECKPOINT_BLOCK_SIZE="$(
    "$VLLM_PYTHON" - "$MODEL/config.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    config = json.load(handle)
print(config.get("block_size", ""))
PY
)"
if [[ -n "$CHECKPOINT_BLOCK_SIZE" ]] \
    && ((NUM_SPECULATIVE_TOKENS + 1 != CHECKPOINT_BLOCK_SIZE)); then
    echo "DFlash2 was trained with block_size=$CHECKPOINT_BLOCK_SIZE, so it must" >&2
    echo "be served with NUM_SPECULATIVE_TOKENS=$((CHECKPOINT_BLOCK_SIZE - 1));" >&2
    echo "got $NUM_SPECULATIVE_TOKENS." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== DFlash2 evaluation configuration ==="
echo "Draft model:           $MODEL"
echo "Verifier model:        $VERIFIER_MODEL"
echo "Speculative tokens:    $NUM_SPECULATIVE_TOKENS (block_size ${CHECKPOINT_BLOCK_SIZE:-unknown})"
echo "Source repository:     $REPO"
echo "Runtime environments:  $RUNTIME_REPO"
echo "Dataset:               $DATASET"
echo "HF endpoint:           $HF_ENDPOINT"
echo "Subsets:               $SUBSETS"
echo "Mode:                  $MODE"
echo "Max requests:          $MAX_REQUESTS"
echo "Max concurrency:       $MAX_CONCURRENCY"
echo "Max model length:      $MAX_MODEL_LEN"
echo "Generation kwargs:     $GEN_KWARGS"
echo "CUDA devices:          $CUDA_VISIBLE_DEVICES"
echo "API port:              $VLLM_PORT"
echo "vLLM internal port:    $VLLM_INTERNAL_PORT"
echo "vLLM V2 runner:        $VLLM_USE_V2_MODEL_RUNNER"
echo "Output:                $OUTPUT_DIR"

echo "=== Step 1: Launching vLLM DFlash2 server ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    HF_ENDPOINT="$HF_ENDPOINT" \
    VLLM_PLUGINS="" \
    VLLM_USE_V2_MODEL_RUNNER="$VLLM_USE_V2_MODEL_RUNNER" \
    VLLM_PORT="$VLLM_INTERNAL_PORT" \
    TOKENIZERS_PARALLELISM=false \
    "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$VERIFIER_MODEL" \
        --host 127.0.0.1 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --data-parallel-size 1 \
        --spec-model "$MODEL" \
        --spec-method dflash \
        --spec-tokens "$NUM_SPECULATIVE_TOKENS" \
        --dtype bfloat16 \
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
        tail -n 160 "$VLLM_LOG" >&2
        wait "$VLLM_PID" || true
        exit 1
    fi
    sleep 2
done
echo "vLLM server ready."

# DFlash2 and DFlash share method=dflash, so pin the resolved architecture in the
# log to guard against silently benchmarking the base DFlash proposer.
if ! grep -q "Resolved architecture: DFlash2DraftModel" "$VLLM_LOG"; then
    echo "DFlash2 did not load natively. Inspect $VLLM_LOG." >&2
    exit 1
fi

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
        --gen-kwargs "$GEN_KWARGS" \
        "$MODE"

echo "Done. Results: $OUTPUT_DIR"
