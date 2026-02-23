# Mooncake KV Cache Assessment Plan

**Date**: February 14, 2026 (v1.0) / February 20, 2026 (v2.0)
**Objective**: Evaluate Mooncake as an L3 storage backend for SGLang HiCache on AWS
**Status**: Updated with LMCache and Dynamo KVBM benchmark results

---

## Executive Summary

Mooncake is a KVCache-centric disaggregated architecture developed by Moonshot AI (FAST 2025 Best Paper). It implements **hierarchical tiered caching**: GPU VRAM → Host DRAM → Local NVMe → Remote Store.

**Updated assessment (v2.0)**: After completing LMCache and Dynamo KVBM benchmarks, the recommended path is **SGLang + HiCache** with Mooncake as an optional L3 storage backend. Mooncake is not a standalone framework to deploy — it integrates as a storage tier within SGLang's HiCache system via `--hicache-storage-backend mooncake`.

### Key Differentiators (Updated with Actual Data)

| Capability | LMCache | Dynamo KVBM | SGLang HiCache | HiCache + Mooncake |
|------------|---------|-------------|----------------|-------------------|
| **Tiered caching** | 2 tiers | 4 tiers (architecture) | 3 tiers (GPU/CPU/Storage) | 3 tiers + RDMA |
| **Scheduler behavior** | vLLM gates admission | vLLM gates admission | Cascading eviction | Cascading eviction |
| **Tiered offloading tested** | Never triggered | Never triggered | TBD | TBD |
| **Prefix TTFT speedup** | 1.07-1.31x | 1.41-1.82x | TBD | TBD |
| **Memory pressure** | 13x worse than baseline | 19.7x worse than baseline | TBD | TBD |
| **Transport** | File I/O, GDS | NIXL (GDS, RDMA) | cudaMemcpy, kernel | RDMA, TCP |
| **Kimi K2.5 support** | Yes (vLLM) | Yes (vLLM, patched) | Yes (native MLA pools) | Yes |

---

## What Changed Since v1.0

1. **LMCache benchmarks complete**: Synchronous FSx write-back causes 13x worse TTFT under moderate pressure. Prefix speedups are modest (1.07-1.31x). See `results/BENCHMARK_REPORT.md`.
2. **Dynamo KVBM benchmarks complete**: Better prefix caching (1.41-1.82x) but vLLM's scheduler prevented tiered offloading from triggering. See `results/DYNAMO_BENCHMARK_REPORT.md`.
3. **SGLang HiCache identified as the right architecture**: Cascading eviction instead of admission gating. Mooncake is one of five supported L3 backends for HiCache.
4. **Mooncake is NOT a standalone deployment**: It integrates into SGLang via `--hicache-storage-backend mooncake`. The original plan assumed standalone deployment — this is updated.

## Revised Assessment Phases

### Phase 1: SGLang + HiCache Baseline (No Mooncake)

**Objective**: Deploy SGLang with HiCache GPU→CPU tiering and validate Kimi K2.5 serving.

```bash
python3 -m sglang.launch_server \
  --model-path /mnt/nvme/models/Kimi-K2.5 \
  --tp 8 \
  --trust-remote-code \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --page-size 64 \
  --port 8000
```

Run the same benchmark suite (`scripts/run_kimi_benchmarks.py --config sglang`).

**Success Criteria**:
- [ ] SGLang serves Kimi K2.5 with MLA (uses `MLATokenToKVPool`)
- [ ] HiCache L1→L2 eviction triggers under memory pressure
- [ ] Prefix cache hit/miss metrics captured via `/metrics`

### Phase 2: SGLang + HiCache + NVMe Storage Backend

**Objective**: Add NVMe as L3 storage tier to test GPU→CPU→NVMe chain.

```bash
python3 -m sglang.launch_server \
  ... \
  --hicache-storage-backend file \
  --hicache-storage-backend-extra-config '{"path": "/mnt/nvme/kv-cache"}'
```

### Phase 3: SGLang + HiCache + Mooncake L3 Backend

**Objective**: Replace file backend with Mooncake for RDMA-based L3 tier.

```bash
export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_PROTOCOL="rdma"

python3 -m sglang.launch_server \
  ... \
  --hicache-storage-backend mooncake \
  --hicache-storage-prefetch-policy timeout
```

**Success Criteria**:
- [ ] Mooncake Transfer Engine initializes with EFA RDMA
- [ ] L3 data shared across instances (if multi-node)
- [ ] Measurable TTFT benefit from L3 prefetch vs recompute

---

### Comparison Matrix (Updated with Actual Data)

| Benchmark | Baseline vLLM | LMCache+GDS | Dynamo KVBM | SGLang HiCache | HiCache+Mooncake |
|-----------|--------------|-------------|-------------|----------------|-----------------|
| Multi-turn r20 TTFT | ~277ms* | 349ms | **191ms** | TBD | TBD |
| API gateway speedup | 1.00x | 1.31x | **1.82x** | TBD | TBD |
| Doc RAG speedup | 1.00x | 1.07x | **1.41x** | TBD | TBD |
| Conv resumption | 1.00x | 1.04x | **0.99x** | TBD | TBD |
| Moderate pressure fg TTFT | **403ms** | 5,330ms | 7,959ms | TBD | TBD |
| Aggressive pressure bg time | **165s** | 183s | 741s | TBD | TBD |
| Tiered offload triggered | N/A | No (FSx only) | No (scheduler gated) | TBD | TBD |
| Cache hit/miss captured | No | No | No | Must capture | Must capture |

*Baseline multi-turn TTFT estimated from LMCache report context.

### Metrics to Collect (Updated)

**Critical gap from previous rounds**: No cache hit/miss data was captured. For SGLang + HiCache:

1. **Per-tier cache metrics** (from `/metrics`):
   - `sglang:hicache_l1_hits/misses` (GPU)
   - `sglang:hicache_l2_hits/misses` (CPU)
   - `sglang:hicache_l3_hits/misses` (Storage)
   - vLLM: `prefix_cache_hit_total`, `prefix_cache_miss_total`

2. **Latency by tier**: TTFT when serving from L1 vs L2 vs L3

3. **Tier occupancy over time**: Capture before/after each test

4. **Write-back volume**: Bytes written per tier per test

---

### Phase 4: Tiered Caching Validation

**Objective**: Verify that HiCache tiers actually get exercised under memory pressure.

**Key lesson from Dynamo KVBM round**: vLLM's scheduler prevented tiered offloading from triggering even at 2.6x oversubscription. SGLang's cascading eviction should avoid this, but verify:

1. **Reduce GPU cache artificially** (`--mem-fraction-static 0.5`) to force earlier eviction
2. **Monitor per-tier occupancy** during memory pressure tests
3. **Verify L2→L3 writes** actually occur (check FSx/NVMe for data)

```bash
# Memory pressure with constrained GPU cache
python scripts/run_kimi_benchmarks.py \
  --config sglang --endpoint http://localhost:8000 \
  --mode memory-pressure --level moderate
```

### Phase 5: Transfer Protocol Comparison (Mooncake only)

**Objective**: Compare RDMA (EFA) vs TCP for L2↔L3 transfers.

**P5e EFA Configuration**: 3200 Gbps, EFA v2, RDMA-capable.

| Protocol | Expected Throughput | Expected Latency |
|----------|-------------------|-----------------|
| TCP | ~10 GB/s | ~100μs |
| RDMA (EFA) | ~50+ GB/s | ~10μs |

---

## Evaluation Criteria (Updated with Actual Baselines)

### Primary Metrics

| Metric | Baseline vLLM | LMCache | Dynamo KVBM | SGLang HiCache Target |
|--------|--------------|---------|-------------|----------------------|
| Multi-turn r20 TTFT | ~277ms | 349ms | 191ms | <191ms |
| API gateway cold→warm | 1.00x | 1.31x | 1.82x | >1.82x |
| Moderate pressure fg TTFT | **403ms** | 5,330ms | 7,959ms | <1,000ms |
| Tiered offload triggered | N/A | No | No | **Yes** (critical) |
| Cache hit rate captured | No | No | No | **Yes** (critical) |
| Per-tier hit breakdown | N/A | N/A | N/A | L1/L2/L3 rates |

### Success Criteria (Revised)

**SGLang + HiCache is RECOMMENDED if**:
1. Tiered offloading actually triggers (L1→L2 or L2→L3 writes observed)
2. Moderate pressure fg TTFT < 1,000ms (vs baseline 403ms, LMCache 5,330ms, Dynamo 7,959ms)
3. Prefix TTFT matches or beats Dynamo's 1.41-1.82x speedup
4. Per-tier cache metrics are captured and show meaningful L2/L3 utilization

**Mooncake L3 backend is RECOMMENDED if**:
1. File-based L3 backend works but RDMA provides >2x throughput
2. Multi-node scenarios benefit from shared L3 cache
3. EFA RDMA transport is stable on p5e

---

## Success Criteria

### SGLang + HiCache is NOT RECOMMENDED if:

1. SGLang cannot serve Kimi K2.5 reliably (MLA/MoE issues)
2. HiCache tiering still doesn't trigger on H200 even with constrained GPU cache
3. Performance regression vs vLLM on standard prefix workloads
4. Operational complexity doesn't justify gains over baseline vLLM

---

## Deliverables

1. **Benchmark Report**: `results/SGLANG_HICACHE_BENCHMARK_REPORT.md`
2. **Four-way Comparison**: Baseline vs LMCache vs Dynamo vs SGLang HiCache
3. **Per-tier Cache Metrics**: L1/L2/L3 hit rates, occupancy, eviction rates
4. **Recommendation**: Deploy/Not Deploy decision with justification
5. **Mooncake Assessment**: If HiCache baseline is promising, evaluate Mooncake L3 backend

---

## Resource Requirements

| Resource | Specification | Cost |
|----------|---------------|------|
| p5e.48xlarge | 8x H200, EFA | ~$98/hr |
| FSx Lustre | 2.15 TiB PERSISTENT_2 | ~$1/hr |
| Assessment Duration | ~2-3 days (phases 1-3) | ~$5,000-7,000 |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Mooncake vLLM integration issues | Medium | High | Test on separate cluster first |
| RDMA driver incompatibility | Low | Medium | Fallback to TCP |
| FSx performance variance | Low | Low | Multiple test runs |
| Model loading conflicts | Medium | Medium | Scale down before testing |

---

## Schedule (Revised)

| Day | Phase | Tasks |
|-----|-------|-------|
| 1 | SGLang + HiCache baseline | Deploy SGLang, verify Kimi K2.5 MLA serving, run benchmark suite |
| 2 | Tiered validation + NVMe L3 | Constrain GPU cache, verify tier cascading, add NVMe storage backend |
| 3 | Mooncake L3 + RDMA | Install Mooncake, configure EFA RDMA, compare vs file L3 backend |

---

## References

- [Mooncake Paper (FAST 2025)](https://arxiv.org/abs/2407.00079)
- [Mooncake GitHub](https://github.com/kvcache-ai/Mooncake)
- [SGLang HiCache PR #2693](https://github.com/sgl-project/sglang/pull/2693)
- [LMCache Benchmark Report](../results/BENCHMARK_REPORT.md)
- [Dynamo KVBM Benchmark Report](../results/DYNAMO_BENCHMARK_REPORT.md)
- [Lessons Learned](../lessons.md) (31 lessons across all rounds)
- [Dynamo Plan (Phase 4: SGLang)](dynamo_gds_fsx_plan.md)

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-14 | 1.0 | Initial Mooncake assessment plan |
| 2026-02-20 | 2.0 | Major update: reframed as SGLang+HiCache evaluation with Mooncake as optional L3 backend. Updated comparison tables with actual LMCache and Dynamo data. Revised phases from standalone Mooncake deployment to SGLang-first approach. Added metrics collection requirements based on gap discovered in rounds 1-3. |
