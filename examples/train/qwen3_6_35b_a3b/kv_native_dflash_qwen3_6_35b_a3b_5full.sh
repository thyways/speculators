#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Compatibility entry point. The active recipe is the final dual-stream raw-KV
# architecture; the historical kv_bridge filename is retained for existing jobs.
exec "$SCRIPT_DIR/kv_native_dflash_qwen3_6_35b_a3b_kv_bridge_5full.sh" "$@"
