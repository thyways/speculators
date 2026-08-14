#!/usr/bin/env bash
# B2: bounded all-layer target-K/V bridge trained only by the original DSpark
# logit/confidence objective. No KV-space auxiliary loss participates in backward.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

export REPO="${REPO:-$DEFAULT_REPO}"
export ROOT="${ROOT:-$(dirname -- "$REPO")}"
export ENV_REPO="${ENV_REPO:-$ROOT/speculators}"
export RUN_DIR="${RUN_DIR:-$ROOT/model_weights/kv_native_dspark_qwen3_6_35b_a3b_kv_bridge_b2_pure_dspark}"
export RUN_NAME="${RUN_NAME:-kv_native_kv_bridge_b2_pure_dspark}"

# 1-based verifier full-attention layers 4/12/20/28/32/40.
export VERIFIER_KV_LAYER_IDS="${VERIFIER_KV_LAYER_IDS:-3 11 19 27 31 39}"
export KV_BRIDGE_ENABLED=1
export KV_BRIDGE_RANK="${KV_BRIDGE_RANK:-32}"
export KV_BRIDGE_RESIDUAL_SCALE="${KV_BRIDGE_RESIDUAL_SCALE:-0.1}"
export KV_BRIDGE_MAX_CORRECTION_RATIO="${KV_BRIDGE_MAX_CORRECTION_RATIO:-0.5}"
export KV_BRIDGE_NORMALIZE_KEYS=1
export KV_BRIDGE_LR="${KV_BRIDGE_LR:-6e-5}"

export LR="${LR:-6e-4}"

# Run the complete epoch with the original DSpark CE/TV/confidence objective.
export MAX_STEPS=""

exec bash "$SCRIPT_DIR/kv_native_dspark_qwen3_6_35b_a3b_perfectblend_online_full_scratch.sh"
