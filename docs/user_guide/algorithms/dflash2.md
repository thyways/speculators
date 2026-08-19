# DFlash2

DFlash2 extends [DFlash](dflash.md) with two modules on the same block-parallel draft backbone: a grouped dynamic depthwise convolution inside each decoder block, and a candidate selector that scores the transitions between adjacent draft slots. Both target the same weakness -- a DFlash block drafts every slot from mask tokens in one pass, so slot `k` cannot see what landed in slot `k-1`, and acceptance decays toward the end of the block. The convolution restores that dependency at the level of hidden states; the selector restores it at the level of sampled tokens. The draft model subclasses DFlash, so the training pipeline is otherwise unchanged.

Both modules come from [vllm-project/vllm#52816](https://github.com/vllm-project/vllm/pull/52816), which carries them on a separate `DFlash2DraftModel` architecture so existing DFlash checkpoints resolve to the class they resolve to today. The training code here is the counterpart to that inference implementation: parameter names, shapes and arithmetic are the ones it loads.

## How It Works

### Grouped Dynamic Convolution

Each decoder layer wraps both its attention sublayer and its MLP in a convolution over the draft block's query positions:

```
out[i, c] = sum_t (base[t, c] + delta[i, t, g(c)]) * x[i - t, c]
```

`base` is a static per-channel kernel; `delta` is produced from the sublayer's input, so the taps are input-dependent. Channels share a dynamic coefficient within a group of `conv_group_size`. Taps are zeroed across the block boundary, so a slot only ever mixes in slots from its own draft block -- the target context reaches the layer through attention, and is never convolved. One projection of the sublayer's input produces the coefficients for both the pre-sublayer (`prepare`) and the post-sublayer (`finish`) convolution.

The convolution is initialized to the identity (tap 0 static coefficient 1, everything else 0), so a fresh DFlash2 block computes exactly what a DFlash block computes. Together with the zero-initialized selector below, that makes DFlash weights a usable starting point -- see [Starting from DFlash weights](#starting-from-dflash-weights).

### Candidate Selector

The selector adds a low-rank, predecessor-conditioned correction to the logits:

```
bias[p, c] = <A[p] * project(h), B[c]>
```

`A` is the predecessor codebook indexed by the token in the previous slot, `B` the successor codebook, and `project(h)` a rank-space gate from the backbone hidden state. It is the same shape of correction as [DSpark](dspark.md)'s Markov head, with the hidden state entering multiplicatively rather than through a sigmoid gate.

At inference the selector keeps the target head's top-`selector_top_k` per slot, scores every adjacent transition, and walks the best path from the verified anchor token -- so the token chosen in slot `k-1` decides which distribution slot `k` is read from. Training uses the ground-truth previous token as the predecessor and scores the whole vocabulary, which lets the existing DFlash losses apply unchanged.

Training the full vocabulary asks for strictly more than the walk needs, not for something different: making the target token the full-vocabulary argmax of `unary + bias` makes it win inside any candidate set that contains it, and the same gradient reaches the hidden states that produce `unary`, which is what puts it in the set. It is more than the walk needs because the candidate set is `topK(unary)`, chosen before the bias -- a correction that would only pay off outside the top-K is wasted, and neither the loss nor DFlash's `eal` / `position_i_acc` metrics (which argmax over the full vocabulary) can see that. The selector diagnostics below replay the walk under the real top-K restriction, and are the numbers to read.

The selector is zero-initialized on the successor side, so the correction is exactly 0 at step 0. As with LoRA's `B = 0`, the predecessor codebook and the projection get no gradient until the successor side is nonzero, which costs one optimizer step.

### Training Metrics

DFlash2 reports DFlash's metrics plus the inference-shaped walk, replayed with the selector on and off, so a single run shows the selector's contribution:

| Metric                | Meaning                                                                               |
| --------------------- | ------------------------------------------------------------------------------------- |
| `candidate_recall`    | Fraction of slots whose target token is inside the top-K; the ceiling on the selector |
| `unary_accept_len`    | Mean accepted run using the per-slot top-1, i.e. the DFlash baseline in the same run  |
| `selector_accept_len` | The same run using the selector's path walk                                           |
| `unary_eal`           | DFlash's EAL estimator with the selector off                                          |
| `selector_eal`        | DFlash's EAL estimator with the selector on                                           |

`*_accept_len` counts drafted tokens only; add 1 for the bonus token to compare with the acceptance length vLLM reports. Both walks start out identical because the selector is zero-initialized.

The diagnostics run every step and cannot be turned off -- a selector that is not earning anything should be visible in the metrics, not at eval time. The cost is the vocabulary top-K, the same operation the inference side calls the selector's largest single cost, and it lands in the single-digit percent of a training step at production shapes.

## Key Parameters

| Parameter                   | Default | Description                                                                              |
| --------------------------- | ------- | ---------------------------------------------------------------------------------------- |
| `--conv-kernel-size`        | 3       | Convolution taps per sublayer; tap `t` reaches back `t` slots. Must be `<= --block-size` |
| `--conv-group-size`         | 64      | Channels sharing one dynamic coefficient. Must divide the draft hidden size              |
| `--selector-rank`           | 256     | Rank of the predecessor/successor codebooks                                              |
| `--selector-top-k`          | 16      | Candidates kept per slot at inference; sizes the path walk and the diagnostics           |
| `--input-embedding-scale`   | 1.0     | Multiplier on the draft's input embeddings                                               |
| `--output-multiplier`       | 1.0     | Multiplier on the draft logits, before the selector bias                                 |
| `--final-logit-softcapping` | unset   | Softcap the multiplied logits as `tanh(x / cap) * cap`, before the selector bias         |

All DFlash parameters (`--block-size`, `--max-anchors`, `--num-layers`, ...) apply unchanged, and `--speculator-type dflash2` takes DFlash's [RFC #979](https://github.com/vllm-project/speculators/issues/979) defaults: 5 layers, D-PACE with cross-entropy, `block_size=16`.

## Two Constraints DFlash Does Not Have

**The draft vocabulary must be the verifier's full vocabulary.** The selector emits the draft head's top-K ids directly as draft tokens and the inference side applies no `d2t` remap to them, so a pruned draft vocabulary would draft the wrong tokens. Pass `--draft-vocab-size <verifier vocab_size>` (or omit it, when the data directory has no cached `t2d.npy`/`d2t.npy`). Training raises rather than produce a checkpoint that loads and drafts badly.

**`sample_from_anchor` must stay `False`** (the DFlash default). The convolution's block boundary at inference is the query block, `1 + num_speculative_tokens`, which equals `block_size` only when the anchor is the bonus token.

### What the full vocabulary costs

The full-vocabulary requirement is also the memory constraint. The forward holds four `[max_anchors * block_size, draft_vocab_size]` tensors at peak -- the targets, the unary logits, the selector bias, and their sum -- so with a pruned 32k draft vocabulary DFlash gets away with `--max-anchors 3072`, and DFlash2 at Qwen3's 151936 needs roughly a fifth of that for the same footprint. Scale `--max-anchors` by `dflash_draft_vocab / verifier_vocab`, or read the number off the arithmetic: `max_anchors * block_size * vocab_size * 2 bytes * 4`.

### Starting from DFlash weights

`--from-pretrained` does not cross algorithms -- each speculator config pins its own `speculators_model_type`, so a DFlash checkpoint's `config.json` will not validate as a DFlash2 one. Build the DFlash2 checkpoint first, then train from it:

```python
dflash = SpeculatorModel.from_pretrained(dflash_checkpoint)
dflash2 = DFlash2DraftModel.from_training_args(...)          # or from a saved config
dflash2.load_state_dict(dflash.state_dict(), strict=False)    # skips conv + selector
dflash2.save_pretrained(warm_start_dir)
```

The convolution and the selector are the only keys `strict=False` skips, and their initialization means the result computes exactly what the DFlash checkpoint did -- `tests/integration/models/test_dflash2_cuda.py` pins that bit for bit.

## Serving

The trained checkpoint's weight names are already the ones vLLM's DFlash2 model loads. `DFlashQwen3ForCausalLM.load_weights` prefixes every key except `lm_head` and `d2t` with `model.`, which puts them at:

```
layers.{i}.attention_conv.base_kernel                -> model.layers.{i}.attention_conv.base_kernel
layers.{i}.attention_conv.kernel_projection.weight   -> ...
layers.{i}.mlp_conv.base_kernel                      -> ...
layers.{i}.mlp_conv.kernel_projection.weight         -> ...
candidate_selector.predecessor_codebook              -> model.candidate_selector.predecessor_codebook
candidate_selector.successor_codebook                -> ...
candidate_selector.hidden_projection.weight          -> ...
```

`tests/unit/models/test_dflash2.py::TestWeightContract` pins that name set, and `tests/unit/models/test_dflash2_model_definitions.py` runs the PR's own reference tests against this repo's convolution and edge scorer.

`save_pretrained` writes `architectures: ["DFlash2DraftModel"]` into the checkpoint's `config.json` (it takes the model class name), which is the string #52816 selects DFlash2 on.

The config translation still needs two additions on the vLLM side that #52816 does not carry, because it targets a native z-lab-format checkpoint rather than a speculators one. `vllm/transformers_utils/configs/speculators/` dispatches on `speculators_model_type`: `base.py` rejects a value `algos.py` has no entry for, and it derives `speculative_config.method` from that same value -- while #52816 requires `method == "dflash"` in both `_is_dflash2_draft` and `init_speculator`.

First, a `dflash2` updater in `algos.py`, alongside the existing `dflash` and `dspark` ones, to carry the module sizes into `dflash_config`:

```python
@register_speculator("dflash2")
def update_dflash2(config_dict: dict, pre_trained_config: dict) -> None:
    update_dflash(config_dict, pre_trained_config)
    pre_trained_config["architectures"] = ["DFlash2DraftModel"]
    for key in (
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
        "input_embedding_scale",
        "output_multiplier",
        "final_logit_softcapping",
    ):
        if config_dict.get(key) is not None:
            pre_trained_config["dflash_config"][key] = config_dict[key]
```

Second, `extract_vllm_speculative_config` in `base.py` has to map the method back onto `dflash`, the way it already maps `peagle` onto `eagle3`:

```python
if result["method"] == "dflash2":
    result["method"] = "dflash"
```

Until both land, a checkpoint can be served by writing the native-format config by hand: the draft's `transformer_layer_config` at the top level, `architectures: ["DFlash2DraftModel"]`, and the keys above under `dflash_config`.

## See Also

- [DFlash](dflash.md) -- The base algorithm DFlash2 extends
- [DSpark](dspark.md) -- The other DFlash extension; its Markov head is the selector's closest relative
- [Train a Speculator](../tutorials/train.md) -- Step-by-step training guide
- `examples/train/dflash2_qwen3_8b_ultrachat_online_5k.sh` -- End-to-end online training run
