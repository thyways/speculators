# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-08-05 | bootstrap | Resumed the Parser2 DFlash effort. The source is a 12.2 GB ShareGPT-style multimodal JSONL, target/verifier is `Infinity-Parser2-2B-2604`, pilot is 800k, and final validation is a nested 1.5M sample. Existing uncommitted code already implements deterministic ranked sampling and strict preprocessing, but its regeneration default incorrectly points to `Infinity-Parser2-Flash`, while its training launcher uses Muon + linear and causal sliding attention. H1/H2 lock the required corrections before GPU execution. |

