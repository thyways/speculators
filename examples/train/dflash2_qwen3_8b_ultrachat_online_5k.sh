#!/bin/bash
# Online DFlash2 Training Script
#
# DFlash2 is the DFlash drafter plus the two modules from
# https://github.com/vllm-project/vllm/pull/52816:
#
#   * a grouped dynamic depthwise convolution inside each block, so a proposal
#     position sees the ones before it without another backbone pass
#     (--conv-kernel-size, --conv-group-size), and
#   * a candidate selector that scores adjacent transitions between the target
#     head's top-K candidates and walks the best path from the verified anchor
#     (--selector-rank, --selector-top-k).
#
# Same pipeline as dflash_qwen3_8b_ultrachat_online_5k_bestpractices.sh: data
# preparation, vLLM server launch, online training. This uses main's experimental
# DFlash2 defaults: five layers, block_size=8, fixed exponential position weighting,
# KL unary loss, and a two-tap/group-16 local convolution.
#
# Two constraints DFlash2 does not share with DFlash:
#
#   * --draft-vocab-size must be the verifier's full vocabulary. The candidate
#     selector emits the draft head's top-K ids directly as draft tokens, and the
#     inference side applies no d2t remap to them, so a pruned draft vocabulary
#     would draft the wrong tokens. Training raises rather than let that ship.
#     If $OUTPUT_DIR already holds t2d.npy/d2t.npy from a pruned-vocab DFlash
#     run, delete them (or use a fresh directory) -- they are picked up ahead of
#     this flag.
#   * sample_from_anchor must stay False (the default). The convolution's block
#     boundary is the inference query block, 1 + num_speculative_tokens, which
#     equals block_size only when the anchor is the bonus token.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dflash2_qwen3_8b_ultrachat_online_5k.sh
#
# With 5k samples the drafter will not be good; there is enough signal to verify
# the pipeline runs and the model learns. Watch these metrics, which the DFlash2
# model reports alongside DFlash's:
#
#   candidate_recall        fraction of slots whose target token is in the top-K;
#                           the ceiling on what the selector can reach, and what
#                           the unary_loss term buys
#   unary_accept_len        mean accepted run using the per-slot top-1, i.e. the
#                           DFlash baseline inside the same run
#   selector_accept_len     the same run using the selector's path walk
#   unary_loss/selector_loss  the two terms of the total loss: DFlash's
#                           full-vocabulary loss on the unary logits, and the
#                           cross-entropy of the selector's top-K decision
#
# The selector loss is controlled by --selector-loss-alpha. Both unary and selector
# objectives use the configured position weighting; the selector is trained with the
# target injected into the weakest candidate when it is absent from unary Top-K.

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="ultrachat"               # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="./output/dflash2_qwen3_8b_ultrachat"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=5
LR=3e-4

# DFlash-family backbone (best-practices recipe from RFC #979)
SPECULATOR_TYPE="dflash2"
BLOCK_SIZE=8
NUM_LAYERS=5
PER_POSITION_LOSS_WEIGHT="fixed-exp-decay"
LOSS_FN="kl_div"
# Full verifier vocabulary -- required by DFlash2 (see the header). 151936 is
# Qwen3-8B's vocab_size; change it with the model.
DRAFT_VOCAB_SIZE=151936
# The DFlash script this mirrors uses 3072 anchors, but it pairs them with a pruned
# 32k draft vocabulary. The forward holds two
# [MAX_ANCHORS * BLOCK_SIZE, DRAFT_VOCAB_SIZE] tensors at peak -- the targets and
# the unary logits -- so at 151936 the same footprint (~3 GiB per tensor in bf16)
# means about a fifth as many anchors. Scale this with the verifier's vocab_size,
# not by copying it from a DFlash recipe.
MAX_ANCHORS=640
TARGET_LAYER_IDS="2 18 33"  # Must match vLLM's eagle_aux_hidden_state_layer_ids

# DFlash2 modules. Defaults; the reference checkpoint in the PR uses K=16.
CONV_KERNEL_SIZE=2      # taps per sublayer; must be <= BLOCK_SIZE
CONV_GROUP_SIZE=16      # channels per dynamic coefficient; must divide hidden_size
SELECTOR_RANK=256
SELECTOR_TOP_K=16
SELECTOR_LOSS_ALPHA=1.0

# GPU assignments (online training needs separate GPUs for vLLM and training)
VLLM_GPUS="0,1"
TRAIN_GPUS="2,3"
NUM_TRAIN_GPUS=2
# =======================================

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size 2 --port "$VLLM_PORT" &
VLLM_PID=$!

# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 3: Train against the live vLLM server
echo "=== Step 3: Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --per-position-loss-weight "$PER_POSITION_LOSS_WEIGHT" \
    --loss-fn "$LOSS_FN" \
    --conv-kernel-size "$CONV_KERNEL_SIZE" \
    --conv-group-size "$CONV_GROUP_SIZE" \
    --selector-rank "$SELECTOR_RANK" \
    --selector-top-k "$SELECTOR_TOP_K" \
    --selector-loss-alpha "$SELECTOR_LOSS_ALPHA" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
echo
echo "To serve the checkpoint, vLLM needs a 'dflash2' entry in"
echo "vllm/transformers_utils/configs/speculators/algos.py that emits"
echo "architectures=['DFlash2DraftModel'] and the conv/selector keys into"
echo "dflash_config -- see docs/user_guide/algorithms/dflash2.md#serving."
