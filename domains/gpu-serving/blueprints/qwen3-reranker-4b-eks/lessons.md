# Qwen3-Reranker-4B on EKS g6e — Lessons

## Iteration 1 (2026-05-14)

### blocking: `--task classify` is not a vLLM 0.19.1 flag
<!-- captured: 2026-05-14 | stage: 5 -->

Initial manifest used `--task classify` (carried over from pre-0.19 conventions). vLLM 0.19.1 crashes immediately with:

```
vllm: error: unrecognized arguments: --task classify
```

**Fix**: vLLM 0.19 split the single `--task` flag into `--runner` (`generate`/`pooling`/`draft`) and `--convert` (`auto`/`classify`/`embed`/`none`). Use `--runner pooling --convert classify`.

### blocking: `score.weight` missing from Qwen3-Reranker-4B checkpoint
<!-- captured: 2026-05-14 | stage: 5 -->

Even with `--runner pooling --convert classify`, engine core failed with:

```
ValueError: Following weights were not initialized from checkpoint: {'score.weight'}
```

Qwen3-Reranker ships as a plain `Qwen3ForCausalLM` checkpoint with no classification head. Its reranking mechanism is to prompt the model to emit "yes"/"no" and read the logit differential — there is no dedicated `score.weight` tensor.

**Fix**: use `--hf-overrides` to tell vLLM to build a binary seq-cls head from the lm_head rows for the "no"/"yes" tokens:

```yaml
- --hf-overrides
- '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
- --runner
- pooling
- --convert
- classify
```

After this the engine loads cleanly, resolves architecture as `Qwen3ForSequenceClassification`, and exposes `/classify`, `/score`, `/v1/score`, `/rerank`, `/v1/rerank`, `/v2/rerank` routes.

### api-shape: /v1/score works in request-order; /v1/rerank returns sorted
<!-- captured: 2026-05-14 | stage: 5 -->

Both `/v1/score` and `/v1/rerank` return 200 on Qwen3-Reranker-4B with correct semantic ordering (relevant candidates > distractor):

- `/v1/score` — body `{"model": ..., "text_1": "<query>", "text_2": [candidates]}`; returns `data: [{"index": 0, "score": 0.98}, ...]` in request order. Best for benchmark plumbing because scores align with request indices.
- `/v1/rerank` — body `{"model": ..., "query": ..., "documents": [...]}`; returns `results: [...]` SORTED by `relevance_score` desc. Best for production integration.

Chosen `/v1/score` for Stage 6 runners. `smoke-test.sh` validates both shapes + semantic ordering (Paris > Berlin on "capital of France" probe).

### behavior: L40S saturates at c=4 for this workload
<!-- captured: 2026-05-14 | stage: 6 -->

Iteration 1 sweep (k=50, pair_length=1024):

| Concurrency | rps | pairs/s | e2e p50 | e2e p99 |
|-------------|-----|---------|---------|---------|
| 1           | 3.17 | 158   | 299 ms  | 476 ms  |
| 4           | 7.72 | 386   | 500 ms  | 684 ms  |
| 16          | 7.66 | 383   | 1974 ms | 2407 ms |
| 64          | 7.66 | 383   | 3409 ms | 6472 ms |

Peak throughput at c=4; higher concurrency just queues. 50 ms p99-per-pair SLO target from spec is hit ONLY at c=4 (p99 684 ms / 50 pairs = 13.7 ms/pair) — well under budget. At c=1, per-pair is 9.5 ms/pair. Head-room for batch-size optimization (larger k per request) on L40S. 256-concurrency row from spec will be pure latency blow-up without batching changes.

### substrate: L40S, not L4
<!-- captured: 2026-05-14 | stage: 0 -->

Ran on the existing g6e.2xlarge node (L40S 48GB, sm_89) rather than provisioning the spec-preferred g6.xlarge (L4 24GB). Artifact's `infrastructure.substrate_deviation` flags this explicitly. Cost-per-pair figures derived from this artifact are NOT valid for the L4 row and should be recomputed against L40S hourly pricing only.

### housekeeping: old ReplicaSets hold GPUs through CrashLoopBackOff
<!-- captured: 2026-05-14 | stage: 5 -->

While iterating on launch flags, the old CrashLoopBackOff pod kept the GPU reserved, so the new RS stayed Pending ("Insufficient nvidia.com/gpu"). `kubectl rollout` also waited for the sleep-patched RS to terminate. Fastest resolution: `kubectl scale rs <old-rs> --replicas=0` + `kubectl delete pod --force --grace-period=0` on the crashing pod. Plain `kubectl apply` alone is not enough when the image entrypoint fails fast.

## Iteration 2 (2026-05-14)

### behavior: pair-length scales pairs/s inverse-linearly; L=512 is peak
<!-- captured: 2026-05-14 | stage: 6 -->

Pair-length sweep at c=4, k=50 (iteration 2):

| pair_length | rps   | pairs/s | e2e p50 | e2e p99 | prompt toks/req |
|-------------|-------|---------|---------|---------|-----------------|
| 512         | 13.99 | 700     | 270 ms  | 379 ms  | ~822            |
| 1024        |  7.86 | 393     | 495 ms  | 673 ms  | ~1693           |
| 2048        |  4.09 | 204     | 953 ms  | 1275 ms | ~3434           |
| 4096        |  0.00 | 0 (fail)| —       | —       | 4097 > ctx 4096 |

pairs/s scales cleanly as ~1/pair_length (halving length ≈ doubles throughput). Per-pair latency ∝ pair_length, so pair-level tokens/s stays roughly flat — the GPU is compute-bound on attention across the full pair sequence, not bottlenecked by request overhead. **L=512 is the headline length** for throughput claims; it's 78% faster than L=1024 and 3.4× faster than L=2048 in pairs/s.

### failure: L=4096 overshoots max_model_len by 1 token
<!-- captured: 2026-05-14 | stage: 6 -->

All 50 requests at pair_length=4096 failed with HTTP 400:

```
This model's maximum context length is 4096 tokens. However, you requested 4097 tokens in the input for score.
```

Our `build_corpus()` sizes candidates to approximately `pair_length` tokens, but the Qwen3-Reranker chat template adds ~1 token of overhead on top of the raw text. Setting `pair_length=4096` when `max_model_len=4096` is guaranteed to be off-by-one.

**Fix (future)**: either bump `max_model_len` to 4352 (next rope multiple) at deploy time, or cap sweep to 3840 to leave template headroom. For iteration 2 we deliberately captured the failure as a data point — `metrics.failed=50` and the error is recorded in `saturation_notes`.

### behavior: c=256 does not break vLLM on this workload
<!-- captured: 2026-05-14 | stage: 6 -->

Extended concurrency sweep added c=256 (pair_length=1024, k=50). Expected latency blow-up + queue rejection; observed **neither**:

| c   | rps  | pairs/s | e2e p50 | e2e p99 | errors |
|-----|------|---------|---------|---------|--------|
| 1   | 3.33 | 167     | 292 ms  | 452 ms  | 0      |
| 4   | 7.78 | 389     | 497 ms  | 669 ms  | 0      |
| 16  | 7.60 | 380     | 1978 ms | 2465 ms | 0      |
| 64  | 7.31 | 366     | 3660 ms | 6772 ms | 0      |
| 256 | 7.78 | 389     | 3248 ms | 6355 ms | 0      |

c=256 **matches c=4 on pairs/s** (389) and is actually slightly better than c=64 on both throughput and p99 latency. Why this doesn't implode:

- The runner only submits `min(n_steady=50, c=256)` = 50 in-flight requests before the steady window closes, so the effective concurrency ceiling is the request count, not the semaphore limit.
- vLLM's `--max-num-seqs=64` caps true batch size; excess requests wait in vLLM's internal queue, which is well-behaved under a 50-request burst.
- No context overflow: k=50 candidates at pair_length=1024 fits under 4096 ctx (measured: ~1693 prompt tokens/req).

**Interpretation**: for this workload+config, the throughput knee is at c=4; c≥16 is pure queueing; c=256 does not add failure modes because `n_steady=50` caps the burst. To actually stress vLLM's queue we'd need `steady >> 256` or a sustained arrival rate higher than service rate.

### surprise: non-monotonic p99 at high concurrency
<!-- captured: 2026-05-14 | stage: 6 -->

p99 latency is **lower at c=256 (6355 ms) than at c=64 (6772 ms)** in iteration 2. At first glance counter-intuitive. Likely cause: at higher admission concurrency, vLLM's scheduler can form larger continuous-batching groups on each step (max-num-seqs=64 always full), giving tail requests more predictable service. At c=64, the semaphore delivers requests just barely faster than service, so the queue oscillates. Small effect (~6%), well inside run-to-run noise, but worth flagging — it means "more concurrency = worse latency" is NOT always monotonic once the server queue is saturated.
