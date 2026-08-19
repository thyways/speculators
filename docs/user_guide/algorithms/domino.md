# Domino

Domino extends [DFlash](dflash.md) with a recurrent logit-correction head on top of the same block-parallel draft backbone. Pure block-parallel drafting predicts every position in a block independently, so acceptance decays toward the end of the block. Domino restores the dependency with a GRU that runs over the block's token embeddings: its state at each position, concatenated with that position's draft hidden state, is projected to an additive logit correction. Unlike [DSpark](dspark.md)'s Markov head, which conditions on the single previous token, Domino's correction carries state across the whole block prefix. The draft model subclasses DFlash, so the architecture and training pipeline are otherwise unchanged, and it can be paired with any supported verifier. Serving reuses vLLM's `dspark` sequential in-block sampler.

## How It Works

### Recurrent Logit Correction

For each anchored block, a single-layer bias-free GRU consumes the block's token embeddings. At slot `k` the head computes

```
final_logits[k] = base_logits[k] + embed_proj([hidden[k]; gru_state[k]])
```

where `embed_proj` is a two-layer SiLU MLP (`hidden_size + gru_hidden_dim -> logits_correction_emb_dim -> draft_vocab_size`). The state at slot `k` has consumed exactly the tokens preceding slot `k`'s label, so the correction is causal and matches what is available at serving time.

The GRU uses `nn.GRU`'s standard initialization, while both projection layers use the verifier config's `initializer_range`, matching SpecForge's Qwen initialization. This lets gradients reach the whole correction head as soon as the corrected objective has non-zero weight.

### Uncorrected Prefix

The first `--pure-draft-prefix-len` predicted slots in each block keep the uncorrected DFlash logits. Those are the positions DFlash already predicts most reliably, and leaving them alone keeps the head focused on the block tail where acceptance actually decays. With `sample_from_anchor: False` the anchor slot is not a prediction, so the correction starts one slot later.

### Blended Base/Final Objective

Domino trains two objectives at once and blends them:

```
loss = (1 - lambda_base) * L(final_logits) + lambda_base * L(base_logits)
```

`L` is the shared DFlash objective, so every configured loss term and the per-position decay apply unchanged. The base term's gradient cannot reach the correction head (the draft LM head is frozen), so it anchors early training on the backbone alone. `lambda_base` starts at `--lambda-base-start` and decays linearly to 0 over the first `--lambda-base-decay-ratio` of the run, after which only the corrected -- i.e. deployed -- logits are optimized.

Two consequences worth knowing:

- Once `lambda_base` reaches 0 the base term is skipped entirely rather than multiplied by zero, so its extra pass over the vocabulary stops costing memory. The `base_loss` / `base_full_acc` / `base_eal` metrics are therefore only logged while the base term is live; `lambda_base` is always logged.
- Validation always scores the corrected objective (`lambda_base = 0`), so val loss stays comparable across epochs and `--save-best` remains meaningful.

`--per-position-loss-weight dpace` is rejected for Domino: D-PACE derives its per-position weights from the loss of the logits being scored, which would weight the two terms differently and make the blend meaningless.

### Packed Document Boundaries

Anchors are sampled from supervised positions without checking that the whole block stays inside one packed document, so a block anchored near a document boundary can have trailing slots whose labels belong to the next document. This is inherited DFlash behavior -- attention is document-aware for the context, but those trailing slots are still supervised -- and Domino's recurrence additionally carries state across the boundary. With `--total-seq-len` far larger than `--block-size` the affected fraction of blocks is small, but it is worth knowing when interpreting the last positions' per-position accuracy.

### Sample From Anchor

Domino defaults to `sample_from_anchor: True` -- the anchor and all mask positions predict future tokens, producing `block_size` speculative tokens. This matches upstream Domino's `shift_label: true`. See [DFlash](dflash.md#sample-from-anchor) for details.

## Key Parameters

| Parameter                     | Default | Description                                                                 |
| ----------------------------- | ------- | --------------------------------------------------------------------------- |
| `--gru-hidden-dim`            | 1024    | Hidden width of the intra-block GRU                                         |
| `--logits-correction-emb-dim` | 256     | Bottleneck width of the correction MLP (upstream calls this `emb_dim`)      |
| `--pure-draft-prefix-len`     | 1       | Leading predicted slots per block that keep the uncorrected DFlash logits   |
| `--lambda-base-start`         | 1.0     | Initial weight of the uncorrected loss term; 0 disables base anchoring      |
| `--lambda-base-decay-ratio`   | 0.5     | Fraction of total training steps over which `--lambda-base-start` reaches 0 |

All DFlash parameters (`--block-size`, `--max-anchors`, `--num-layers`, ...) apply unchanged.

The `lambda_base` schedule needs a step horizon. It is taken from `--scheduler-total-steps` when set, otherwise `--epochs * steps_per_epoch`, clamped by `--max-steps`. With no horizon at all the base term stays disabled.

### Matching the Upstream Recipe

The head's math is identical to upstream Domino, but several *training* knobs default differently in this repo. To follow upstream's published recipe, set these explicitly:

| Flag                        | This repo's default | Upstream Domino |
| --------------------------- | ------------------- | --------------- |
| `--block-size`              | 8                   | 16              |
| `--dflash-decay-gamma`      | 4.0                 | 7.0             |
| `--lambda-base-decay-ratio` | 0.5                 | 1.0             |
| `--max-anchors`             | 3072                | 256             |

`--lambda-base-decay-ratio 1.0` is the one worth thinking about: upstream keeps the base term alive for the *whole* run rather than retiring it halfway, so the backbone stays anchored throughout and the correction head's weight ramps in more slowly.

Two differences cannot be closed by flags, both inherited from how this repo's whole DFlash family is trained:

- The objective distills the verifier's logits (`--loss-fn kl_div` by default) where upstream computes cross-entropy against ground-truth token ids. Note `--loss-fn ce` is *not* the upstream objective either -- it is cross-entropy against `argmax` of the verifier logits.
- The loss is normalized by the *undecayed* count of supervised positions, where upstream divides by the decayed weight sum. With decay on, that makes this repo's loss magnitude smaller by the mean decay weight (roughly 0.29x at `block_size=16`, `gamma=4`), which matters when transplanting a learning rate or a gradient-clipping threshold.

## Memory

Domino scores the block twice while `lambda_base > 0`, and there is no chunked reduction in the loss, so the `[1, max_anchors * block_size, draft_vocab_size]` logits are materialized in full. The extra cost is about one additional bf16 logits tensor, not a doubling of the head.

Measured peak activation bytes for one forward+backward at `max_anchors * block_size = 4096` and `draft_vocab_size = 32000` (one such tensor is 250 MiB in bf16):

| `--loss-fn`              | base term live | final term only |
| ------------------------ | -------------- | --------------- |
| `tv` / `nla`             | 1534 MiB       | 1286 MiB        |
| `kl_div` (default)       | 1809 MiB       | 1526 MiB        |
| `ce`                     | 2034 MiB       | 1785 MiB        |
| `{"ce": 0.1, "tv": 0.9}` | 2287 MiB       | 1527 MiB        |

`tv` and `nla` run fused kernels that never materialize the distributions, which is why they are cheapest; a compound loss pays for every term it lists. So if a Domino run OOMs where the matching DFlash run fits, switch to `--loss-fn tv` (or `nla`) and lower `--max-anchors` before reducing `--block-size` -- and note that a `ce`/`tv` compound costs *more* than the `kl_div` default, not less. Memory also drops back to DFlash levels once `lambda_base` reaches zero.

## Training

```bash
torchrun --standalone --nproc_per_node 8 scripts/train.py \
    --verifier-name-or-path Qwen/Qwen3-8B \
    --data-path ./output/domino_qwen3_8b/data \
    --save-path ./output/domino_qwen3_8b/checkpoints \
    --speculator-type domino \
    --block-size 16 \
    --max-anchors 1024 \
    --num-layers 6 \
    --draft-vocab-size 32000
```

A full-data cluster recipe is in `examples/train/qwen3_6_35b_a3b/domino_qwen3_6_35b_a3b_perfectblend_online_full.sh`.

## Converting an Upstream Checkpoint

Domino checkpoints trained outside Speculators (the upstream `DominoDraftModel` layout) convert with:

```python
from speculators.convert import convert_model

convert_model(
    model="./upstream/domino-checkpoint",
    verifier="Qwen/Qwen3-8B",
    algorithm="domino",
    output_path="./converted",
)
```

The GRU's parameter names and math are identical to `nn.GRU(bias=False)`, so conversion only renames the head onto the `logits_correction` container -- both the flat upstream layout and the older `logit_head` container are handled. The head's output layer spans the full verifier vocabulary, so the converted checkpoint requires `draft_vocab_size == vocab_size`.

## See Also

- [DFlash](dflash.md) -- The base algorithm Domino extends
- [DSpark](dspark.md) -- The same idea with a memoryless previous-token bias plus a confidence head
- [Train a Speculator](../tutorials/train.md) -- Step-by-step training guide (select Domino, then online, offline, or hybrid)
