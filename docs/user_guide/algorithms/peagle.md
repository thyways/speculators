# P-EAGLE

P-EAGLE (Parallel EAGLE) extends Eagle-3 with parallel multi-token prediction. Instead of drafting tokens autoregressively one at a time, P-EAGLE predicts multiple tokens in parallel. It uses Conditional Drop-token (COD) sampling during training for memory efficiency, and Llama-style transformer layers inherited from Eagle-3. It can be paired with any supported verifier model.

## How It Works

### Architecture

P-EAGLE builds on the Eagle-3 architecture: the target model produces hidden states from three layers (`2`, `L/2`, and `L-1` in the paper), which are concatenated, projected, and passed through decoder layers. Depth 0 uses those verifier hidden states, while every deeper position uses the same learnable shared hidden state and the learned mask-token embedding. The token embedding remains trainable; freezing it removes the model's ability to learn a useful mask-token representation.

### COD Sampling

![COD Sampling](../../assets/peagle_cod_sampling.png)

Training a parallel multi-depth model naively would require memory proportional to `num_depths × sequence_length`. P-EAGLE uses **Conditional Drop-token (COD) sampling** to reduce this cost:

- Depth 0 retains all n positions
- Depth d retains approximately n × r^d positions, where r is the `down-sample-ratio`
- Each sampled depth-d position keeps its depth-(d-1) predecessor, preserving the conditional rollout dependency

This geometric decay means deeper predictions train on fewer positions per batch, keeping memory usage manageable while still learning to predict multiple tokens ahead.

### Sequence Partitioning

`--sequence-partitions S` enables the dependency-aware partitioning from Algorithm 1 of the paper. Depths 0 and 1 are assigned by sequence position; depths 2 and deeper inherit the segment of their predecessor at `(depth - 1, position - 1)`. Each segment receives the cumulative depth-0 causal prefix it needs. The trainer performs a separate forward/backward pass for each segment and updates the optimizer only after all segments, reducing peak activation memory without changing the COD objective.

### Inference Process

1. P-EAGLE drafts multiple tokens in parallel across all depths in a single pass
2. Target model verifies all draft tokens in one forward pass
3. The longest correct prefix is accepted
4. Repeat from the last accepted token

vLLM 0.28 and later provide the P-EAGLE model and parallel-drafting runtime natively. Speculators checkpoints serialize their text-only rotary configuration in the form expected by that runtime, so no external vLLM plugin is required. P-EAGLE parallel drafting currently uses vLLM's V1 model runner; set `VLLM_USE_V2_MODEL_RUNNER=0` when serving it.

## Key Parameters

| Parameter                 | Default | Description                                                   |
| ------------------------- | ------- | ------------------------------------------------------------- |
| `--num-depths`            | 8       | Number of parallel prediction depths                          |
| `--down-sample-ratio`     | 0.7     | Geometric decay ratio for COD sampling                        |
| `--down-sample-ratio-min` | 0.0     | Optional non-paper retention floor                            |
| `--sequence-partitions`   | 1       | Dependency-aware within-sequence accumulation segments        |
| `--embed-requires-grad`   | true    | Keep the embedding trainable, as required by the paper recipe |
| `--target-layer-ids`      | —       | Pass exactly three verifier layers; paper uses `2 L/2 L-1`    |

## Pretrained Models

There are currently no pretrained P-EAGLE models available. You can train your own using the tutorials linked below.

## Research & Citation

P-EAGLE is based on research from AWS AI Labs: [arXiv Paper](https://arxiv.org/abs/2602.01469)

```bibtex
@article{hui2026peagle,
  title={P-EAGLE: Parallel-Drafting EAGLE with Scalable Training},
  author={Hui, Mude and Huang, Xin and Salas, Jaime Campos and Sun, Yue and Pemberton, Nathan and Song, Xiang and Khetan, Ashish and Karypis, George},
  journal={arXiv preprint arXiv:2602.01469},
  year={2026}
}
```

## See Also

- [Train a Speculator](../tutorials/train.md) -- Step-by-step training guide (select P-EAGLE, then online, offline, or hybrid)
