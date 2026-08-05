# Infinity-Parser2 DFlash smoke analysis

## Decision

The locked 100-record, 10-step smoke gate **passes**. H2 is supported at smoke scale, so the deterministic 800k pilot is authorized. This result establishes pipeline correctness and short-run optimization stability; it is not yet an acceleration claim.

## Confirmatory run

- Run ID: `20260805T023908Z-425140`
- Code: protocol commit `e44c3de`; max-step fix commit `27037e3`
- Teacher/verifier: `/home/ma-user/work/data_mllm/publish_models/Infinity-Parser2-2B-2604`
- Data: 100 preprocessing-eligible records, 90 train / 10 validation, sequence cap 20,480
- Real multimodal preflight: 3,961 exact token IDs; four hidden-state layers; hidden width 2,048
- Training: six-rank DDP, five-layer DFlash, block size 8, bidirectional sliding window 2,048, AdamW `1e-4`, weight decay `0.01`, cosine schedule
- Optimizer steps: exactly global steps 0--9; the trainer stopped at `global_step=10` before epoch 6
- Training losses: `[5.820, 5.749, 8.318, 4.009, 5.224, 4.355, 3.824, 3.439, 3.434, 3.407]`
- Validation loss by epoch: `8.608062 -> 5.406 -> 3.947 -> 3.877 -> 3.809215`
- Checkpoints: epochs 0--4, with `checkpoint_best -> 4`
- Final checkpoint: 1,656,266,704 bytes, 62 tensors, SHA-256 `9004c715969c0514db87c5dce5d6a5e3d348733276c015efa04859c911a77b26`
- vLLM requests: 501 successful hidden-state chat-completion requests; no HTTP or fatal-log failures

All losses were finite. No token mismatch, hidden-state failure, NaN/Inf, OOM, NCCL fatal, or traceback appeared. The temporary hidden-state directory was removed after completion, and the eight-GPU keepalive was restored as PID 517984.

## Invalid exploratory attempt

Run `20260805T022001Z-264688` is excluded. It exposed an infrastructure bug: `train_epoch()` respected `max_steps`, but `run_training()` continued into subsequent epochs and performed one extra update per epoch. The run logged global steps 0--14 rather than 0--9. It is preserved at `output/infinity_parser2_v1_12_dflash/smoke_100_seq20480_failed_max_steps_20260805T022001Z-264688`.

The fix adds a run-level stop before starting another epoch and after the final checkpoint/validation sequence. A focused 60-test suite, including a new cross-epoch regression test, passed before the clean rerun.

## Interpretation and next gate

The smoke resolves the basic compatibility questions: this vLLM build can serve Parser2 multimodal prompts and export the requested target layers; strict prepared-token alignment works; the requested optimizer/schedule/attention recipe trains without immediate numerical failure; and checkpoint publication works.

The result does not establish acceptance length or speedup. The next experiment is the nested 800k pilot. It must first publish exactly 800,000 preprocessing-eligible records, then complete stable training. Only afterward should the draft be compared with the autoregressive target on held-out prompts for output equivalence, accepted tokens per verification, TTFT, decode latency, throughput, and memory.
