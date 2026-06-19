# Tier Stack Comparison — Qwen3-Embedding-8B on A10G

**Workload:** `rag-qa` (2-10K char mixed contexts, 8 requests per concurrency level)
**Model:** Qwen/Qwen3-Embedding-8B (BF16, max_model_len=8192)
**Hardware:** ml.g5.4xlarge (1× A10G 24GB) on SageMaker HyperPod
**Engine:** vLLM v0.19.1
**Date:** 2026-05-13

## Tier configurations tested

| Tier | Attention backend | torch.compile | Prefix cache | CUDA graphs |
|------|-------------------|---------------|--------------|-------------|
| **T0 baseline** | FLASH_ATTN | ❌ `--enforce-eager` | ❌ `--no-enable-prefix-caching` | ❌ disabled by eager |
| **T5 optimized** | FLASH_ATTN | ✅ Inductor | ✅ (default on) | ✅ (piecewise) |

Skipped tiers (not applicable to this deployment):
- T1 Quantization — A10G has no FP8 tensor cores; INT4/INT8 rarely helps embeddings
- T2 KV/prefix cache — embeddings are single-pass; prefix cache gives minor effect, bundled into T5
- T3 Spec decode — embeddings don't generate tokens
- T4 Parallelism — single-GPU deployment, no TP/DP/PP to explore

## Results

| Concurrency | T0 req/s | T5 req/s | T5/T0 | T0 p50 (ms) | T5 p50 (ms) | T0 p99 (ms) | T5 p99 (ms) |
|:-----------:|:--------:|:--------:|:-----:|:-----------:|:-----------:|:-----------:|:-----------:|
| 1 | 4.77 | 7.16 | 1.5× | 204 | 106 | 360 | 224 |
| 2 | 6.66 | 17.92 | 2.7× | 282 | 105 | 569 | 142 |
| 4 | 5.58 | 29.21 | 5.2× | 693 | 138 | 978 | 168 |
| 8 | 6.01 | 50.28 | 8.4× | 1,165 | 149 | 2,026 | 197 |
| 16 | 5.66 | 77.56 | 13.7× | 2,877 | 191 | 3,405 | 309 |
| 32 | 5.78 | 122.96 | **21.3×** | 5,356 | 243 | 6,320 | 345 |

## Observations

### The T0→T5 gap widens with concurrency
- At **c=1** the delta is a modest **1.5×** — eager-mode overhead shows but isn't catastrophic.
- At **c=32** the delta is **21.3×** — T0 can't batch past ~6 req/s because every request triggers a full Python-side interpreter loop without CUDA graph capture.
- The T0 curve **saturates at ~6 req/s** regardless of concurrency — classic "compute-starved by scheduling overhead" shape.
- T5 scales near-linearly up to c=32.

### Latency tells the same story
- T0 p99 at c=32: **6.3 seconds** — unusable for production
- T5 p99 at c=32: **345 ms** — production-viable
- The gap is not "T5 is faster"; it's "T0 collapses under load"

### Tier delta attribution

For an embedding-on-A10G deployment, the **T5 kernel tier is the entire story**. With no FP8, no KV cache, no spec decode, and single-GPU, there are no other knobs. But that one tier delivers the **21× throughput** at production concurrency — which aligns with `docs/optimization-stack.md`'s claim that T5 is the "last 15%" for LLMs but is disproportionately larger for compute-dense, single-pass workloads like embeddings.

### Cost impact

- At c=32, T5 delivers 122.96 req/s; T0 delivers 5.78 req/s
- Same $2.03/hr for ml.g5.4xlarge
- **T5 cost per million embeddings**: ~$3.44
- **T0 cost per million embeddings**: ~$73.14
- T5 is 21× cheaper per embedding, directly reflecting the throughput ratio.

## Tier stack table (required by CTO engagement spec)

| Tier | Config landed | req/s @ c=32 | Δ vs T0 | Notes |
|------|---------------|--------------|---------|-------|
| T0 | Eager mode, no prefix cache | 5.78 | 1.0× (ref) | Establishes baseline |
| T1 | (skipped) | — | — | A10G no FP8; INT4/8 not useful for embeddings |
| T2 | (skipped) | — | — | Embeddings have no KV cache |
| T3 | (skipped) | — | — | Embeddings don't generate |
| T4 | (skipped) | — | — | Single-GPU |
| T5 | FLASH_ATTN + torch.compile + CUDA graphs | 122.96 | **21.3×** | The entire optimization story for this deployment |

## Raw data

- `workload-rag-qa-t0-baseline.json`
- `workload-rag-qa-t5-optimized.json`
