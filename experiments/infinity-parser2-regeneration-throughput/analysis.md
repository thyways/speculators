# Infinity-Parser2 regeneration concurrency analysis

## Decision

Use 32 concurrent requests per one-GPU endpoint (256 requests across eight replicas), with the vLLM multimodal processor cache disabled via `--mm-processor-cache-gb 0`.

## Measurements

Each steady-state window uses immutable 128-event segments and excludes the first completed segment as the timing marker.

| Per-endpoint concurrency | MM processor cache | Records | Seconds | Records/s | Window errors | Server health |
|---:|---|---:|---:|---:|---:|---|
| 4 | enabled | 3,072 | 275 | 11.17 | not used for selection | healthy |
| 16 | enabled | 4,096 | 134 | 30.57 | 2 | rejected: MM-cache HTTP 500/traceback |
| 16 | disabled | 4,096 | 146 | 28.05 | 3 empty responses | 8/8 healthy; no fatal log |
| 32 | disabled | 4,096 | 94 | 43.57 | 0 | 8/8 healthy; no fatal log |

Concurrency 32 is 55.3% faster than the clean concurrency-16 run and about 3.90x faster than the concurrency-4 baseline. It therefore wins the locked selection rule by much more than the 5% tie threshold.

## Cache failure and mitigation

With the default 4 GiB multimodal processor cache, higher concurrency triggered a vLLM 0.26 receiver-cache race on repeated image hashes:

`AssertionError: Expected a cached item for mm_hash=...`

One record exhausted six retries and published an HTTP-500 error event. Restarting every replica with `--mm-processor-cache-gb 0` removed the sender/receiver cache path. Because prefix caching is also disabled, vLLM assigns request-local multimodal UUIDs. No traceback, HTTP 500, OOM, or unhealthy replica appeared in either cache-disabled measurement.

The three errors in the clean concurrency-16 window were deterministic empty model responses, not transport/server failures. The selected concurrency-32 window had zero error events.

## Runtime implication

At 43.57 committed records/s, processing the full 1,519,877-record candidate pool takes about 9.7 hours in a constant-rate projection. The operational estimate is 10--13 hours to allow for longer later samples and shard-boundary variance. Merge, target-tokenization, mapping, and training are additional stages and are not included in this generation ETA.
