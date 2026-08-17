#!/usr/bin/env bash
# Evaluate the local five-layer Qwen3.6-35B-A3B P-EAGLE checkpoint with
# RedHatAI/speculator_benchmarks. Throughput mode records speculative
# acceptance metrics together with latency, TTFT, ITL, and throughput metrics.
#
# Smoke test:
#
#   SUBSETS=HumanEval MAX_REQUESTS=4 \
#     bash examples/evaluate/example_qwen3_6_35b_a3b_peagle_speculator_benchmarks.sh
#
# Unlike DFlash/DSpark, P-EAGLE has no dedicated vLLM `--spec-method`: vLLM's
# speculators translator maps `speculators_model_type=peagle` onto the EAGLE-3
# runtime with `parallel_drafting=True` (see
# vllm/transformers_utils/configs/speculators/base.py). That translation only
# fires when the *served* model is the speculator, and here we serve the
# verifier, so the mapping is spelled out via --speculative-config instead.
#
# Parallel drafting on the EAGLE-3 runtime is not implemented by the V2 model
# runner (vllm/config/vllm.py rejects it outright), so this script pins
# VLLM_USE_V2_MODEL_RUNNER=0. The V1 EagleProposer in turn refuses a draft whose
# config still advertises `mrope_section`, which every drafter trained against
# the M-RoPE Qwen3.6 verifier does; the `speculators_peagle` plugin normalizes
# that to linear positions. See src/speculators/vllm/peagle.py.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE="${WORKSPACE:-$(dirname -- "$REPO")}"
RUNTIME_REPO="${RUNTIME_REPO:-$WORKSPACE/speculators}"

DRAFT_ROOT="$WORKSPACE/model_weights/Qwen3_6_35B_A3B_draft"
MODEL="${MODEL:-$DRAFT_ROOT/peagle_qwen3_6_35b_a3b_5full/checkpoints/0}"
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
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
# 0.90 (the DFlash/DSpark default) leaves only ~0.7 GiB for the KV cache here:
# the 35B verifier's weights alone are ~70 GiB, and the V1 runner additionally
# reserves a multimodal encoder cache that the V2 runner path does not.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0}"
VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
if [[ -z "${GEN_KWARGS+x}" ]]; then
    GEN_KWARGS="{\"temperature\":${TEMPERATURE}}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/evaluation_results/peagle_qwen3_6_35b_a3b_5full_ckpt0_spec${NUM_SPECULATIVE_TOKENS}_${TIMESTAMP}}"
VLLM_LOG="$OUTPUT_DIR/vllm.log"

VLLM_PYTHON="${VLLM_PYTHON:-$RUNTIME_REPO/vllm_venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-$RUNTIME_REPO/speculators_venv/bin/python}"
GUIDELLM="${GUIDELLM:-$RUNTIME_REPO/speculators_venv/bin/guidellm}"
EVALUATE_PY="${EVALUATE_PY:-$REPO/scripts/evaluate/evaluate.py}"
PLUGIN_PYTHONPATH="$REPO/src:$REPO/hs_connectors/src"

VLLM_PID=""
PLUGIN_METADATA_ROOT=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM

    if [[ -n "$PLUGIN_METADATA_ROOT" && -d "$PLUGIN_METADATA_ROOT" ]]; then
        rm -rf -- "$PLUGIN_METADATA_ROOT"
    fi

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

if [[ ! -d "$REPO/src/speculators" || ! -d "$REPO/hs_connectors/src" ]]; then
    echo "Missing Speculators source or hs_connectors under: $REPO" >&2
    exit 1
fi

if [[ "$MODE" != "throughput" && "$MODE" != "sweep" ]]; then
    echo "MODE must be 'throughput' or 'sweep', got: $MODE" >&2
    exit 1
fi
if [[ ! "$NUM_SPECULATIVE_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SPECULATIVE_TOKENS must be a positive integer, got: $NUM_SPECULATIVE_TOKENS" >&2
    exit 1
fi
if [[ "$VLLM_USE_V2_MODEL_RUNNER" != "0" ]]; then
    echo "P-EAGLE parallel drafting is unsupported by the V2 model runner;" >&2
    echo "VLLM_USE_V2_MODEL_RUNNER must be 0, got: $VLLM_USE_V2_MODEL_RUNNER" >&2
    exit 1
fi

SPECULATIVE_CONFIG="$(
    printf '{"model":"%s","method":"eagle3","num_speculative_tokens":%s,"parallel_drafting":true}' \
        "$MODEL" "$NUM_SPECULATIVE_TOKENS"
)"

# The shared vLLM environment is editable-installed from another worktree, so
# its dist-info may predate this new plugin entry point. Add a tiny temporary
# distribution metadata shim when needed; PYTHONPATH still points at this
# worktree's real source, and the environment itself is left untouched.
if ! "$VLLM_PYTHON" - <<'PY'
from importlib.metadata import entry_points

plugins = entry_points(group="vllm.general_plugins")
raise SystemExit(not any(item.name == "speculators_peagle" for item in plugins))
PY
then
    PLUGIN_METADATA_ROOT="$(mktemp -d /tmp/speculators-peagle-plugin.XXXXXX)"
    DIST_INFO="$PLUGIN_METADATA_ROOT/speculators_peagle_runtime-0.0.0.dist-info"
    mkdir -p "$DIST_INFO"
    printf '%s\n' \
        'Metadata-Version: 2.1' \
        'Name: speculators-peagle-runtime' \
        'Version: 0.0.0' > "$DIST_INFO/METADATA"
    printf '%s\n' \
        '[vllm.general_plugins]' \
        'speculators_peagle = speculators.vllm.peagle:register' \
        > "$DIST_INFO/entry_points.txt"
fi

if [[ -n "$PLUGIN_METADATA_ROOT" ]]; then
    PLUGIN_PYTHONPATH="$PLUGIN_PYTHONPATH:$PLUGIN_METADATA_ROOT"
fi

mkdir -p "$OUTPUT_DIR"

echo "=== P-EAGLE evaluation configuration ==="
echo "Draft model:           $MODEL"
echo "Verifier model:        $VERIFIER_MODEL"
echo "Speculative tokens:    $NUM_SPECULATIVE_TOKENS"
echo "Speculative config:    $SPECULATIVE_CONFIG"
echo "Source repository:     $REPO"
echo "Runtime environments:  $RUNTIME_REPO"
echo "Dataset:               $DATASET"
echo "HF endpoint:           $HF_ENDPOINT"
echo "Subsets:               $SUBSETS"
echo "Mode:                  $MODE"
echo "Max requests:          $MAX_REQUESTS"
echo "Max concurrency:       $MAX_CONCURRENCY"
echo "Max model length:      $MAX_MODEL_LEN"
echo "GPU memory util:       $GPU_MEMORY_UTILIZATION"
echo "Max output tokens:     $MAX_OUTPUT_TOKENS"
echo "Generation kwargs:     $GEN_KWARGS"
echo "CUDA devices:          $CUDA_VISIBLE_DEVICES"
echo "API port:              $VLLM_PORT"
echo "vLLM internal port:    $VLLM_INTERNAL_PORT"
echo "vLLM V2 runner:        $VLLM_USE_V2_MODEL_RUNNER"
echo "Output:                $OUTPUT_DIR"

echo "=== Step 1: Launching vLLM P-EAGLE server ==="
setsid env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    HF_ENDPOINT="$HF_ENDPOINT" \
    PYTHONPATH="$PLUGIN_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
    VLLM_PLUGINS=speculators_peagle \
    VLLM_USE_V2_MODEL_RUNNER="$VLLM_USE_V2_MODEL_RUNNER" \
    VLLM_PORT="$VLLM_INTERNAL_PORT" \
    TOKENIZERS_PARALLELISM=false \
    "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$VERIFIER_MODEL" \
        --host 127.0.0.1 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --data-parallel-size 1 \
        --speculative-config "$SPECULATIVE_CONFIG" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --dtype bfloat16 \
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
