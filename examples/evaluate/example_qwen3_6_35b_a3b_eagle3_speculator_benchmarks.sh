#!/usr/bin/env bash
# Evaluate the local one-layer Qwen3.6-35B-A3B EAGLE-3 checkpoint with
# RedHatAI/speculator_benchmarks. Throughput mode records speculative
# acceptance metrics together with latency, TTFT, ITL, and throughput metrics.
#
# Smoke test:
#
#   SUBSETS=HumanEval MAX_REQUESTS=4 \
#     bash examples/evaluate/example_qwen3_6_35b_a3b_eagle3_speculator_benchmarks.sh
#
# Like P-EAGLE, EAGLE-3 has no dedicated vLLM `--spec-method`: the speculators
# translator maps `speculators_model_type=eagle3` onto the EAGLE-3 runtime
# (vllm/transformers_utils/configs/speculators/algos.py::update_eagle3). That
# translation keys off the *draft* config, which is what `--speculative-config`
# loads, so the method is spelled out there rather than via --spec-method.
#
# Unlike P-EAGLE this needs no V1 pin. P-EAGLE sets
# `parallel_drafting=true`, which makes
# `net_num_new_slots_per_request = num_speculative_tokens - 1 > 0`, and that is
# what trips `LlmBaseProposer._raise_if_mrope`. Plain EAGLE-3 keeps
# `extra_slots_per_request = 1` and passes hidden states, so the net is
# `1 - 1 = 0`, `needs_extra_input_slots` is False, and the M-RoPE guard never
# fires (vllm/v1/spec_decode/llm_base_proposer.py).
#
# CAVEAT: the guard not firing is not the same as M-RoPE being handled. This
# drafter was trained with `--draft-mrope-full-head-hack`, i.e. against linear
# positions, but its config still advertises `mrope_section`, so vLLM runs it
# with `uses_mrope=True`. `update_eagle3` does not strip that field the way the
# P-EAGLE's checkpoint serializer now does. Use the pos0 acceptance rate as the
# tell: peers land near 0.79-0.84, so a markedly lower value means the positions
# are wrong and the drafter needs the same normalization.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE="${WORKSPACE:-$(dirname -- "$REPO")}"
RUNTIME_REPO="${RUNTIME_REPO:-$WORKSPACE/speculators}"

MODEL="${MODEL:-$WORKSPACE/model_weights/eagle3_1_qwen3_6_35b_a3b_1full/checkpoints/0}"
VERIFIER_MODEL="${VERIFIER_MODEL:-$WORKSPACE/model_weights/Qwen/Qwen3.6-35B-A3B}"
DATASET="${DATASET:-RedHatAI/speculator_benchmarks}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

VLLM_PORT="${VLLM_PORT:-8110}"
VLLM_INTERNAL_PORT="${VLLM_INTERNAL_PORT:-29502}"
SERVER_URL="http://127.0.0.1:${VLLM_PORT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODE="${MODE:-throughput}"
SUBSETS="${SUBSETS:-HumanEval,math_reasoning,qa,question,rag,summarization,tool_call,translation,writing}"
MAX_REQUESTS="${MAX_REQUESTS:-200}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0}"
VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
if [[ -z "${GEN_KWARGS+x}" ]]; then
    GEN_KWARGS="{\"temperature\":${TEMPERATURE}}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/evaluation_results/eagle3_1_qwen3_6_35b_a3b_1full_ckpt0_spec${NUM_SPECULATIVE_TOKENS}_${TIMESTAMP}}"
VLLM_LOG="$OUTPUT_DIR/vllm.log"

VLLM_PYTHON="${VLLM_PYTHON:-$RUNTIME_REPO/vllm_venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-$RUNTIME_REPO/speculators_venv/bin/python}"
GUIDELLM="${GUIDELLM:-$RUNTIME_REPO/speculators_venv/bin/guidellm}"
EVALUATE_PY="${EVALUATE_PY:-$REPO/scripts/evaluate/evaluate.py}"
PLUGIN_PYTHONPATH="$REPO/src:$REPO/hs_connectors/src"

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

SPECULATIVE_CONFIG="$(
    printf '{"model":"%s","method":"eagle3","num_speculative_tokens":%s}' \
        "$MODEL" "$NUM_SPECULATIVE_TOKENS"
)"

mkdir -p "$OUTPUT_DIR"

echo "=== EAGLE-3 evaluation configuration ==="
echo "Draft model:           $MODEL"
echo "Verifier model:        $VERIFIER_MODEL"
echo "Speculative tokens:    $NUM_SPECULATIVE_TOKENS"
echo "Speculative config:    $SPECULATIVE_CONFIG"
echo "Dataset:               $DATASET"
echo "Subsets:               $SUBSETS"
echo "Mode:                  $MODE"
echo "Max requests:          $MAX_REQUESTS"
echo "Max concurrency:       $MAX_CONCURRENCY"
echo "Generation kwargs:     $GEN_KWARGS"
echo "CUDA devices:          $CUDA_VISIBLE_DEVICES"
echo "API port:              $VLLM_PORT"
echo "vLLM V2 runner:        $VLLM_USE_V2_MODEL_RUNNER"
echo "Output:                $OUTPUT_DIR"

echo "=== Step 1: Launching vLLM EAGLE-3 server ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    HF_ENDPOINT="$HF_ENDPOINT" \
    PYTHONPATH="$PLUGIN_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
    VLLM_USE_V2_MODEL_RUNNER="$VLLM_USE_V2_MODEL_RUNNER" \
    VLLM_PORT="$VLLM_INTERNAL_PORT" \
    TOKENIZERS_PARALLELISM=false \
    "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$VERIFIER_MODEL" \
        --host 127.0.0.1 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --data-parallel-size 1 \
        --speculative-config "$SPECULATIVE_CONFIG" \
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
