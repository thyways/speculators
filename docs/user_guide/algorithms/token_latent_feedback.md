# Parallel Token-Latent Feedback

`token_latent_feedback` implements the latest Parallel Token-Latent Feedback design in `方案设计.md`. It keeps DFlash's block-parallel backbone and adds one constant-depth feedback stage after the backbone:

1. A packed projection predicts a normalized token-intent latent and a scalar reliability gate for every block slot.
2. A strictly lower-triangular Toeplitz matrix mixes all earlier latents in one low-dimensional matmul.
3. A zero-initialized latent-to-hidden projection writes the prefix signal back; the existing full-vocabulary LM head then runs once for the whole block.

The zero initialization makes a fresh model's logits equal to the ordinary DFlash path. The latent cosine target is used only during training. It is obtained from the frozen verifier LM-head rows with a frozen projection initialized by the usual global training seed and is removed by the vLLM loader.

## Defaults

| Parameter          |                  Default |
| ------------------ | -----------------------: |
| Draft layers       |                        5 |
| Block size         |                        8 |
| Latent dimension   |                      128 |
| Prefix mixer       |                   `full` |
| Reliability gate   |                  enabled |
| Main loss          | `{"ce": 0.1, "tv": 0.9}` |
| Position weighting |        `fixed-exp-decay` |
| Latent loss weight |                      0.1 |

Use `--prefix-mixer-mode shifted` for the first-order shifted-latent ablation, or `--prefix-mixer-mode none` to measure the auxiliary-loss-only baseline. The feedback projection remains one block operation; no per-position sampler or CUDA-graph node chain is added.

## Training

See `examples/train/qwen3_6_35b_a3b/token_latent_feedback_qwen3_6_35b_a3b_online_full.sh`. The checkpoint can be loaded with `SpeculatorModel.from_pretrained` and served through the `speculators_token_latent_feedback` vLLM general plugin.

## Relationship to other DFlash extensions

Domino, DSpark, and DFly condition on the actually sampled predecessor and use a serial in-block path. DFlash2 uses local convolution and a discrete selector. Token-Latent Feedback communicates continuous token intent before discrete selection, so it can be evaluated independently or composed with those heads.
