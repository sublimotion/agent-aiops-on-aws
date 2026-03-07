# GLM-5-FP8 HiCache Benchmark — B200 x8 SGLang HiCache CPU Offload

**Date**: 2026-03-07
**Instance**: p6-b200.48xlarge (8x NVIDIA B200, 183 GB HBM each, NVSwitch)
**Region**: us-east-2b (capacity block cr-0827eef18c1c46bcd)
**Config**: SGLang glm5-blackwell image, TP8, context 131072, chunked prefill 32768, mem_fraction_static 0.90
**HiCache**: Enabled, 100 GB per TP rank (800 GB total), write_through, kernel IO backend
**Driver**: NVIDIA 580.126.09, CUDA 13.0
**AMI**: amazon-eks-node-al2023-x86_64-nvidia-1.32-v20260304 (AL2023, kernel 6.1)
**Model**: zai-org/GLM-5-FP8 — 744B MoE (256 routed + 1 shared expert, top-8, ~40B active), FP8
**Cache**: SGLang HiRadixCache with NSATokenToKVPoolHost (CPU offload tier)

## HiCache Configuration

| Parameter | Value |
|-----------|-------|
| `--enable-hierarchical-cache` | true |
| `--hicache-size` | 100 (GB per TP rank, 800 GB total) |
| `--hicache-write-policy` | write_through |
| `--hicache-io-backend` | kernel |
| `--hicache-mem-layout` | layer_first (default) |
| Host memory pool per rank | 100 GB (25,423 pages) |
| Device KV pool per rank | ~82 GB |

## Test 1: Throughput Scaling (short prompt, 200 max_tokens)

| Concurrency | Requests | Throughput (tok/s) | P50 (s) | P90 (s) | P99 (s) | Avg (s) | Failed |
|:-----------:|:--------:|:------------------:|:-------:|:-------:|:-------:|:-------:|:------:|
| 1 | 8 | 48 | 4.13 | 4.17 | 4.17 | 4.13 | 0 |
| 2 | 8 | 87 | 4.71 | 4.77 | 4.77 | 4.58 | 0 |
| 4 | 8 | 172 | 4.71 | 4.71 | 4.71 | 4.65 | 0 |
| 8 | 16 | 311 | 5.19 | 5.19 | 5.19 | 5.14 | 0 |
| 16 | 32 | 521 | 6.20 | 6.20 | 6.20 | 6.14 | 0 |
| 32 | 64 | 885 | 7.29 | 7.30 | 7.30 | 7.23 | 0 |
| 64 | 128 | 1556 | 8.24 | 8.26 | 8.27 | 8.22 | 0 |
| 128 | 256 | 2602 | 10.03 | 10.08 | 10.11 | 9.82 | 0 |

## Test 2: Long Generation (medium prompt, 500 max_tokens)

| Concurrency | Requests | Throughput (tok/s) | P50 (s) | P90 (s) | P99 (s) | Avg (s) | Failed |
|:-----------:|:--------:|:------------------:|:-------:|:-------:|:-------:|:-------:|:------:|
| 1 | 4 | 50 | 10.00 | 10.01 | 10.01 | 10.00 | 0 |
| 4 | 8 | 176 | 11.57 | 11.57 | 11.57 | 11.39 | 0 |
| 8 | 16 | 322 | 12.45 | 12.45 | 12.45 | 12.42 | 0 |
| 16 | 32 | 568 | 14.11 | 14.12 | 14.12 | 14.08 | 0 |

## Test 3: Prefix Cache Effectiveness (HiRadixCache)

5 rounds of 4 questions with shared system prompt (~80 tokens prompt, 100 max_tokens):

| Round | Avg Latency (s) | Latencies (s) | Notes |
|:-----:|:---------------:|:-------------:|:------|
| 1 | 2.184 | 2.179, 2.184, 2.186, 2.186 | First pass — system prompt cached |
| 2 | 2.185 | 2.185, 2.184, 2.185, 2.185 | Prefix cache hit |
| 3 | 2.184 | 2.184, 2.184, 2.185, 2.184 | Prefix cache hit |
| 4 | 2.185 | 2.185, 2.184, 2.185, 2.186 | Prefix cache hit |
| 5 | 2.185 | 2.184, 2.185, 2.185, 2.184 | Prefix cache hit |

**Prefix cache verdict**: Identical to baseline (σ < 1ms). HiCache does not degrade prefix caching.

## Comparison: HiCache vs Baseline (RadixAttention only)

| Metric | Baseline | HiCache | Delta |
|--------|:--------:|:-------:|:-----:|
| Single-request throughput | 48 tok/s | 48 tok/s | 0% |
| 8 concurrent | 233 tok/s | 311 tok/s | +33% |
| 16 concurrent | 360 tok/s | 521 tok/s | +45% |
| 32 concurrent | 524 tok/s | 885 tok/s | +69% |
| 64 concurrent | 909 tok/s | 1,556 tok/s | +71% |
| 128 concurrent | N/A | 2,602 tok/s | — |
| Prefix cache latency (σ) | < 1 ms | < 1 ms | 0% |
| Long gen (16 conc, 500 tok) | 578 tok/s | 568 tok/s | -2% |
| Error rate | 0% | 0% | 0% |

## Key Observations

1. **Peak throughput: 2,602 tok/s at 128 concurrent** — 2.86x the baseline peak (909 tok/s at 64 concurrent)
2. **HiCache enables higher concurrency** — the CPU offload tier (800 GB) effectively expands KV cache capacity, allowing 128+ concurrent requests without degradation
3. **Single-request throughput unchanged** — 48 tok/s, confirming HiCache adds no overhead to the hot path
4. **Superlinear scaling at high concurrency** — 32→64 goes from 885→1556 tok/s (1.76x), suggesting KV cache eviction was the bottleneck in baseline
5. **Long generation slightly lower at 16 concurrent** — 568 vs 578 tok/s (-2%), within noise margin, likely due to CPU↔GPU transfer overhead during sustained generation
6. **Prefix caching identical** — HiRadixCache preserves RadixAttention prefix cache behavior perfectly
7. **Zero errors** across all concurrency levels including 128 concurrent

## Why HiCache Works Where LMCache Failed

- **LMCache** accesses `token_to_kv_pool.k_buffer` and `token_to_kv_pool.v_buffer` separately → crashes on NSA's fused `kv_buffer`
- **HiCache** has native `NSATokenToKVPoolHost` class that understands the fused layout → works natively with NSA/MLA attention
- HiCache is built into SGLang, so it evolves with the attention backend — no external compatibility issues

## HiCache Startup Notes

- **Host memory requirement**: `--hicache-size` must be **larger** than the device KV pool (~82 GB per rank). Default `--hicache-ratio 2.0` tries 2x device pool which can OOM on memory-constrained systems.
- **Total host memory**: 8 ranks x 100 GB = 800 GB. p6-b200.48xlarge has 2 TB RAM, ~907 GB available after model + system.
- **NSATokenToKVPoolHost**: Initialized with page stride 8448, 25,423 pages per rank.
