# Infinity-Parser2 regeneration concurrency protocol

Status: LOCKED before changing the scaled generation concurrency  
Date: 2026-08-05  
Classification: exploratory operational optimization

## Objective

Choose the per-replica request concurrency that minimizes wall time for the immutable 1.5M target-response regeneration without changing its semantic generation configuration.

## Fixed semantics

- teacher and served model: `/home/ma-user/work/data_mllm/publish_models/Infinity-Parser2-2B-2604`
- temperature 0, top-p 1, seed 42, thinking disabled
- max completion tokens 32,768
- eight independent one-H100 vLLM replicas
- same frozen sample manifest, generation fingerprint, and resumable segment journal

Concurrency and endpoint scheduling are operational parameters only. Previously committed successes are reused and never regenerated.

## Candidates and measurement

- Baseline: 4 concurrent requests per endpoint (32 total), measured from the initial steady-state full-generation segments.
- Candidate A: 16 per endpoint (128 total).
- Candidate B: 32 per endpoint (256 total).

For each candidate, exclude process fingerprint/startup time and measure steady-state committed-record throughput across at least 4,096 newly published records when practical. Record GPU utilization, endpoint health, immutable error-event count, and fatal-log matches.

## Selection rule

Reject a candidate on any CUDA OOM, fatal traceback, unhealthy replica, journal-integrity failure, or material increase in immutable generation errors. Otherwise select the highest committed records/second. If two candidates differ by no more than 5%, select the lower concurrency to reduce tail latency and operational risk. The selected concurrency is then used for the remaining full generation.
