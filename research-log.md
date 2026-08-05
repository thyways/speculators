# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-08-05 | bootstrap | Resumed the Parser2 DFlash effort. The source is a 12.2 GB ShareGPT-style multimodal JSONL, target/verifier is `Infinity-Parser2-2B-2604`, pilot is 800k, and final validation is a nested 1.5M sample. Existing uncommitted code already implements deterministic ranked sampling and strict preprocessing, but its regeneration default incorrectly points to `Infinity-Parser2-Flash`, while its training launcher uses Muon + linear and causal sliding attention. H1/H2 lock the required corrections before GPU execution. |
| 2 | 2026-08-05 | invalid-run | The first real smoke passed target serving and exact multimodal token alignment, but revealed that `max_steps=10` only broke the inner epoch loop. Training continued with one update in each later epoch and reached 15 updates. The run was stopped, archived, and excluded from evidence. |
| 3 | 2026-08-05 | implementation | Added a run-level `max_steps` termination guard plus a cross-epoch regression test in commit `27037e3`; 60 focused unit tests passed while all eight H100s remained occupied by the keepalive process. |
| 4 | 2026-08-05 | confirmatory-result | Clean smoke `20260805T023908Z-425140` passed all locked H2 criteria: exact 3,961-token multimodal round-trip, four hidden-state layers of width 2,048, six-rank DDP steps 0--9 only, finite losses, no mismatch/OOM/NaN, and a readable checkpoint. Final validation loss was 3.809215 and checkpoint SHA-256 was `9004c715969c0514db87c5dce5d6a5e3d348733276c015efa04859c911a77b26`. Direction is DEEPEN to the nested 800k pilot. |
