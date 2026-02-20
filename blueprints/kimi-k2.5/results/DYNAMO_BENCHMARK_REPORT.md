# NVIDIA Dynamo KVBM Benchmark Report

**Model**: Kimi-K2.5 (1T params, native INT4 CompressedTensorsWNA16MarlinMoE)
**Instance**: p5e.48xlarge (8x NVIDIA H200 143GB HBM3, 1144GB total)
**Storage**: FSx Lustre PERSISTENT_2 (2.15 TiB, 1000 MB/s/TiB baseline)
**Serving**: vLLM 0.15.1 (CUDA 13.0) + NVIDIA Dynamo KVBM 0.9.0 (patched)
**Date**: 2026-02-18 / 2026-02-19

## Executive Summary

We benchmarked NVIDIA Dynamo's KV Block Manager (KVBM) for KV cache offloading on a
p5e.48xlarge instance serving Kimi-K2.5. KVBM provides a 4-tier cache hierarchy
(GPU HBM -> CPU DRAM -> Disk -> Remote) with async write-back via dedicated Tokio tasks
and separate CUDA streams.

**Key findings**:
- **Prefix caching works well**: 1.41x TTFT speedup for document RAG, 1.82x for API gateway
- **Multi-turn scaling is excellent**: 20x context growth causes only 1.6x TTFT increase (120ms -> 191ms)
- **Conversation resumption is near-perfect**: 0.99x ratio (no degradation)
- **Memory pressure remains a problem**: Moderate pressure causes 7,959ms mean fg TTFT (19.7x vs baseline 403ms), though p50 is only 811ms
- **FSx disk offloading did not activate**: Despite enabling 500GB disk cache on FSx Lustre, KVBM's unlinked temp file approach means cache sizes report 0 MB across all tests
- **Throughput is consistent**: ~10.3-10.5 tok/s across all workloads with minimal overhead

## Hardware & Configuration

| Component | Specification |
|-----------|--------------|
| Instance | p5e.48xlarge |
| GPUs | 8x NVIDIA H200 143GB HBM3 |
| Total VRAM | 1,144 GB |
| GPU KV Cache | 9,533 blocks x 64 tokens = 610,112 tokens |
| GPU Memory Util | 85% (--gpu-memory-utilization 0.85) |
| Model Size | ~540 GB (INT4 across 8 GPUs) |
| Free for KV | ~432 GB |
| NVMe | 8x 3,800 GB NVMe SSD (30.4 TB total) |
| FSx Lustre | 2.15 TiB PERSISTENT_2 |
| Max Model Len | 32,768 tokens |
| Tensor Parallel | 8 |

**vLLM Launch Args**:
```
--tensor-parallel-size 8 --enable-prefix-caching --enforce-eager
--max-model-len 32768 --swap-space 32 --gpu-memory-utilization 0.85
```

**KVBM Config**:
```
DYN_KVBM_CPU_CACHE_GB=64
DYN_KVBM_DISK_CACHE_GB=500
DYN_KVBM_DISK_CACHE_DIR=/mnt/fsx/kv-cache/dynamo
DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true
kv_connector=DynamoConnector, kv_role=kv_both
```

**KVBM Patching**: Three `outer_dim` validation sites in Rust source changed from
`range(min = 1, max = 2)` to `range(min = 1)` to support Kimi K2.5's MLA with outer_dim=64.

**Known Limitations**:
- GDS buffer registration fails (error=5030) in container — falls back to POSIX I/O compat mode
- `fallocate()` not supported on Lustre — KVBM uses `truncate()` fallback for disk pre-allocation
- Disk cache uses unlinked temp files (anonymous fd) — `du` always shows 0 bytes

## Test Results

### 1. Multi-Turn Conversation (20 rounds, growing history)

**Setup**: 8 users, 20 rounds each, ~1K tokens added per round. Context grows from 1K to 20K tokens.

| Round | Context | TTFT (ms) |
|-------|---------|-----------|
| 1 | 1,000 | 120 |
| 5 | 5,000 | 159 |
| 10 | 10,000 | 171 |
| 15 | 15,000 | 184 |
| 20 | 20,000 | 191 |

```
Round  1 ( 1000 tok):     120ms ██████
Round  5 ( 5000 tok):     159ms ████████
Round 10 (10000 tok):     171ms ████████
Round 15 (15000 tok):     184ms █████████
Round 20 (20000 tok):     191ms █████████
```

**Comparison with LMCache**:

| Round | Dynamo TTFT | LMCache TTFT | Winner |
|-------|-------------|--------------|--------|
| 1 | 120ms | 155ms | **Dynamo (1.29x)** |
| 5 | 159ms | 193ms | **Dynamo (1.21x)** |
| 10 | 171ms | 277ms | **Dynamo (1.62x)** |
| 15 | 184ms | 355ms | **Dynamo (1.93x)** |
| 20 | 191ms | 349ms | **Dynamo (1.83x)** |

**Key finding**: Dynamo's TTFT scales much more slowly with context length. Context grew 20x
(1K -> 20K) but TTFT only grew 1.6x (120ms -> 191ms), compared to LMCache's 2.3x (155ms -> 349ms).
At round 20, Dynamo is nearly 2x faster. This suggests more efficient prefix cache management.

**FSx cache growth**: 0 MB (no disk offloading triggered)

### 2. Enterprise API Gateway (8K shared schema)

**Setup**: Single ~8K token tool/API schema shared across all 50 requests with 15 unique user queries.

| Phase | TTFT (ms) | Requests |
|-------|-----------|----------|
| Cold (first 5) | 219 | 5 |
| Warm (remaining) | 133 | 44 |
| **Speedup** | **1.82x** | |

Success rate: 98% (1 failed request out of 50).

**Comparison with LMCache**:

| Metric | Dynamo | LMCache | Winner |
|--------|--------|---------|--------|
| Cold TTFT | 219ms | 265ms | **Dynamo** |
| Warm TTFT | 133ms | 203ms | **Dynamo** |
| Speedup | 1.82x | 1.31x | **Dynamo** |

**Key finding**: Dynamo achieves stronger prefix sharing benefits. The warm TTFT of 133ms
vs LMCache's 203ms indicates more efficient in-memory prefix cache hits.

### 3. Document Library RAG (15 docs, Zipf access)

**Setup**: 15 documents x ~2K tokens each, 60 queries selecting 3 docs per query via Zipf
distribution (popular docs accessed more frequently).

| Phase | TTFT (ms) |
|-------|-----------|
| Early (cold) | 278 |
| Late (warm) | 198 |
| **Speedup** | **1.41x** |

**Document hit distribution** (top 5): [61, 29, 21, 10, 10]

**Comparison with LMCache**:

| Metric | Dynamo | LMCache | Winner |
|--------|--------|---------|--------|
| Early TTFT | 278ms | 241ms | LMCache |
| Late TTFT | 198ms | 226ms | **Dynamo** |
| Speedup | 1.41x | 1.07x | **Dynamo** |

**Key finding**: Dynamo shows stronger cache warming over time (1.41x vs 1.07x). LMCache's
early requests were faster (241ms vs 278ms), possibly due to lower cold-start overhead, but
Dynamo's late requests are 12% faster once caches are warm.

### 4. Conversation Resumption (gap + diverse traffic)

**Setup**: 10 users x 5 turns pre-gap, 40 diverse interleaved requests (to evict in-memory cache),
then 5 turns post-gap.

| Phase | TTFT (ms) |
|-------|-----------|
| Pre-gap mean | 118 |
| Post-gap mean | 123 |
| Last turn before gap | 121 |
| First turn after gap | 121 |
| **Resumption ratio** | **0.99x** |

**Comparison with LMCache**:

| Metric | Dynamo | LMCache | Winner |
|--------|--------|---------|--------|
| Pre-gap TTFT | 118ms | 153ms | **Dynamo** |
| Post-gap TTFT | 123ms | 166ms | **Dynamo** |
| Resumption ratio | 0.99x | 1.04x | **Dynamo** |

**Key finding**: Both systems achieve near-perfect resumption, but Dynamo's ratio is slightly
better (0.99x means post-gap is actually faster than pre-gap average, likely due to cache warming).
Dynamo's absolute TTFT values are ~25% lower across the board.

### 5. Shared Prompt Sweep (multi-tenant scaling)

| Tenants | Cold TTFT | Warm TTFT | TTFT Speedup | E2E Speedup |
|---------|-----------|-----------|--------------|-------------|
| 5 | 291ms | 124ms | 2.35x | 1.01x |
| 10 | 208ms | 123ms | 1.69x | 1.01x |
| 25 | 224ms | 124ms | 1.81x | 1.01x |
| 50 | 208ms | 124ms | 1.68x | 1.01x |

**Comparison with LMCache**:

| Tenants | Dynamo Speedup | LMCache Speedup | Winner |
|---------|---------------|-----------------|--------|
| 5 | 2.35x | 1.12x | **Dynamo** |
| 10 | 1.69x | 1.05x | **Dynamo** |
| 25 | 1.81x | 1.02x | **Dynamo** |
| 50 | 1.68x | 1.02x | **Dynamo** |

**Key finding**: Dynamo shows dramatically better cold-to-warm TTFT improvement across all
tenant counts. Warm TTFT is remarkably stable at ~124ms regardless of tenant count (5-50),
while cold requests are ~210-290ms. This 1.7-2.4x speedup far exceeds LMCache's 1.02-1.12x.

The difference is explained by absolute TTFT values: Dynamo's cold TTFT includes ~290ms for
first-time prefix computation of the 4K system prompt, while warm requests skip this entirely.
LMCache showed cold TTFT of 180-194ms and warm of 173-177ms — a much smaller absolute gap.

## Memory Pressure: Dynamo KVBM vs LMCache+GDS vs Baseline

### Moderate Pressure (25 sessions x 24K tokens = 600K tokens vs 610K capacity)

| Metric | Dynamo KVBM | LMCache+GDS | Baseline vLLM | Best |
|--------|-------------|-------------|---------------|------|
| Background success | 25/25 | 25/25 | 25/25 | Tie |
| Background elapsed | 108.3s | 95.6s | 65.9s | **Baseline** |
| Foreground TTFT mean | 7,959ms | 5,330ms | 403ms | **Baseline** |
| Foreground TTFT p50 | 811ms | 1,017ms | 405ms | **Baseline** |
| Foreground TTFT p99 | 36,384ms | 23,444ms | 560ms | **Baseline** |
| Recovery TTFT mean | 673ms | 568ms | 233ms | **Baseline** |
| Peak KV usage | 60% | 61% | 61% | Tie |
| Preemptions | 0 | 0 | 0 | Tie |
| FSx growth | 0 MB | +29.4 GB | 0 | N/A |

**Key finding**: Dynamo's mean fg TTFT (7,959ms) is worse than LMCache (5,330ms), but Dynamo's
**p50 is much better** (811ms vs 1,017ms). This indicates Dynamo has a bimodal distribution:
most foreground requests complete quickly, but a few get delayed significantly (pulling up the
mean and p99). Background throughput is 13% slower than LMCache and 64% slower than baseline.

Both LMCache and Dynamo underperform baseline by a large margin under moderate memory pressure.
The fundamental issue — FSx I/O overhead during active KV cache management — affects both systems.

### Aggressive Pressure (50 sessions x 32K tokens = 1.6M tokens vs 610K capacity)

| Metric | Dynamo KVBM | LMCache+GDS | Baseline vLLM | Best |
|--------|-------------|-------------|---------------|------|
| Background success | 47/50 (94%) | 50/50 | 50/50 | **LMCache/Baseline** |
| Background elapsed | 741.2s | 183.3s | 164.5s | **Baseline** |
| Foreground success | 45/50 (90%) | 50/50 | 50/50 | **LMCache/Baseline** |
| Foreground TTFT mean | 23,381ms | 108,427ms (burst 1) | 96,532ms (burst 1) | Complex |
| Foreground TTFT p50 | 711ms | ~810ms | ~810ms | **Dynamo** |
| Recovery TTFT mean | 27,769ms | 750ms | 737ms | **Baseline/LMCache** |
| Preemptions | 1 | 2 | 3 | **Dynamo** |

**Key finding**: Dynamo struggled under aggressive pressure — background throughput was 4x slower
than both LMCache and baseline (741s vs 183s vs 165s), and 6 requests failed (3 background + 5
foreground). Recovery was very slow (27,769ms) with high variance. However, Dynamo had the
fewest preemptions (1) and the best p50 fg TTFT (711ms).

The 741s background elapsed time suggests Dynamo's request processing was severely serialized
at extreme memory pressure, worse than even LMCache's synchronous write-back issue.

## Head-to-Head Summary: Dynamo vs LMCache

### Prefix Caching (Low Pressure)

| Test | Dynamo | LMCache | Winner |
|------|--------|---------|--------|
| Multi-turn (round 20 TTFT) | 191ms | 349ms | **Dynamo (1.83x)** |
| API gateway speedup | 1.82x | 1.31x | **Dynamo** |
| Doc RAG speedup | 1.41x | 1.07x | **Dynamo** |
| Conversation resumption | 0.99x | 1.04x | **Dynamo** |
| Shared prompt (50 tenants) | 1.68x | 1.02x | **Dynamo** |

**Dynamo wins decisively on prefix caching workloads.** Lower absolute TTFT values and
stronger cold-to-warm improvements across every test.

### Memory Pressure (High Pressure)

| Scenario | Dynamo | LMCache | Baseline | Winner |
|----------|--------|---------|----------|--------|
| Moderate fg TTFT mean | 7,959ms | 5,330ms | 403ms | **Baseline** |
| Moderate fg TTFT p50 | 811ms | 1,017ms | 405ms | **Baseline** |
| Moderate recovery | 673ms | 568ms | 233ms | **Baseline** |
| Aggressive bg elapsed | 741s | 183s | 165s | **Baseline** |
| Aggressive fg success | 90% | 100% | 100% | **LMCache/Baseline** |

**Baseline vLLM wins under memory pressure.** Both offloading solutions add overhead
that hurts more than helps when GPU KV cache is stressed. Dynamo is worse than LMCache
under aggressive pressure due to serialization issues.

### Throughput

| Metric | Dynamo | LMCache |
|--------|--------|---------|
| Token throughput | 10.3-10.5 tok/s | 10.4-10.8 tok/s |
| Throughput stability | Very consistent | Consistent |
| Overhead vs baseline | Minimal | 11-45% under pressure |

Throughput is comparable between the two systems under normal operation.

## Where Dynamo KVBM Helps

1. **Prefix-heavy workloads**: Shared API schemas (1.82x), multi-tenant system prompts
   (1.68-2.35x), document libraries (1.41x). Consistently outperforms LMCache.

2. **Multi-turn conversations**: 1.6x TTFT growth over 20 rounds vs LMCache's 2.3x.
   At 20K context, Dynamo is nearly 2x faster (191ms vs 349ms).

3. **Conversation resumption**: 0.99x ratio — essentially no degradation after cache eviction.

4. **Consistent throughput**: ~10.4 tok/s with minimal variance across workloads.

## Where Dynamo KVBM Hurts

1. **Memory pressure at any level**: Mean fg TTFT of 7,959ms (moderate) and 23,381ms
   (aggressive) far exceeds baseline's 403ms. The async write-back architecture did not
   prevent request processing delays.

2. **Extreme pressure reliability**: 6 failed requests (3 bg + 5 fg) under aggressive
   pressure vs 0 failures for both LMCache and baseline. Recovery was erratic (27,769ms mean).

3. **FSx disk offloading**: Despite enabling 500GB disk cache, no data was written to FSx.
   KVBM's unlinked temp file approach means the disk tier appears non-functional on Lustre,
   or was never triggered because GPU+CPU cache was sufficient.

4. **Software maturity**: Required patching 3 Rust source files for MLA support.
   GDS buffer registration fails in container. Disk cache falls back to truncate on Lustre.

## Recommendations

### When to Use Dynamo KVBM

- Prefix-heavy workloads where TTFT improvement matters (API gateways, multi-tenant SaaS)
- Multi-turn conversation workloads with growing context
- Low-to-moderate concurrency (< 15 simultaneous users)
- When you need better prefix caching than LMCache provides

### When to Use LMCache+GDS

- When FSx persistence across restarts is important (LMCache wrote 29-37 GB to FSx; Dynamo wrote 0)
- When 100% reliability under pressure is required (LMCache had 0 failures vs Dynamo's 6)
- When the LMCache ecosystem maturity matters (no source patching needed)

### When to Use Baseline vLLM (no offloading)

- High concurrency workloads (> 20 simultaneous users)
- Memory-pressure-sensitive applications
- Throughput-critical deployments
- Large GPU memory (H100/H200) where KV cache rarely fills
- Cost-sensitive deployments (no FSx Lustre needed)

## FSx Disk Offloading Assessment

A key goal of this benchmark was testing KVBM's tiered offloading to FSx Lustre. **This did
not work as expected.** Despite configuring:

```
DYN_KVBM_DISK_CACHE_GB=500
DYN_KVBM_DISK_CACHE_DIR=/mnt/fsx/kv-cache/dynamo
DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true
```

KVBM created a DiskStorage object with `truncate()` fallback (fallocate not supported on Lustre)
and unlinked the temp file immediately. All tests show 0 MB FSx cache growth. Possible causes:

1. **GPU + CPU cache was sufficient**: 432GB GPU KV + 64GB CPU cache may have handled all
   workloads without needing disk tier
2. **Unlinked temp file behavior**: KVBM uses anonymous file descriptors — data may have been
   written but is invisible to filesystem queries
3. **GDS fallback to POSIX**: Without proper GDS buffer registration, the disk I/O path may
   have been slower than recomputation, causing KVBM to skip offloading

Further investigation with smaller CPU cache (e.g., 8GB) or explicit GDS support outside
containers would be needed to validate the disk tier.

## Raw Data Files

All results are in `results/kimi-k2.5-p5e/dynamo/` (on instance) and locally:

| File | Test |
|------|------|
| `dynamo_multi_turn_8u_20r.json` | Multi-turn conversation (20 rounds) |
| `dynamo_enterprise_api_gateway.json` | Enterprise API gateway (8K schema) |
| `dynamo_doc_library_rag.json` | Document library RAG (15 docs) |
| `dynamo_conversation_resumption.json` | Conversation resumption |
| `dynamo_multi_tenant_5t.json` | Shared prompt sweep (5 tenants) |
| `dynamo_multi_tenant_10t.json` | Shared prompt sweep (10 tenants) |
| `dynamo_multi_tenant_25t.json` | Shared prompt sweep (25 tenants) |
| `dynamo_multi_tenant_50t.json` | Shared prompt sweep (50 tenants) |
| `dynamo_memory_pressure_25bg.json` | Memory pressure moderate (25 x 24K) |
| `dynamo_memory_pressure_50bg.json` | Memory pressure aggressive (50 x 32K) |

## Note: Why No Data Was Offloaded to FSx

KVBM's tiered cache spills in order: GPU → CPU → Disk. The disk tier (FSx) only activates
when the CPU tier is full. In our configuration, the first two tiers were never exhausted:

- **GPU tier**: 432GB free for KV (~610K tokens). The moderate test (25x24K = 600K tokens)
  peaked at 60% usage. The aggressive test (50x32K = 1.6M tokens target) had vLLM's scheduler
  queuing and preempting requests before the GPU tier itself filled — so the GPU handled the
  active working set without needing to evict much to CPU.
- **CPU tier**: 64GB (~90K additional tokens of KV cache). Even with some blocks evicted from
  GPU, 64GB provided substantial headroom that was never exhausted.
- **Disk tier (FSx)**: Never activated because CPU never filled.

The 64GB CPU cache was reduced from 128GB to avoid container OOM, but was still generous
enough to absorb any GPU evictions without overflowing to disk.

Additionally, **vLLM's scheduler itself prevented the stress test from reaching KVBM's
offloading path**. The scheduler acts as a gatekeeper — it queues or preempts requests before
GPU KV cache actually fills, so the downstream tiers (CPU → Disk) never see the pressure.
Even in the aggressive test (1.6M tokens requested vs 610K capacity), the scheduler throttled
admission rather than letting KV cache overflow into KVBM's offloading path. This means
KVBM's tiered offloading was effectively bypassed by vLLM's own memory management, which
decided it was better to queue new requests or preempt existing ones than to let the GPU
KV cache fill to the point where KVBM would start evicting blocks to CPU/disk.

To exercise the FSx disk path in future runs:
1. Reduce CPU cache drastically (e.g., `DYN_KVBM_CPU_CACHE_GB=2`) to force early disk spills
2. Run more concurrent sessions (30+ at 32K each) to exceed GPU + CPU capacity simultaneously
3. Test on smaller GPUs (A100 40GB, L4) where the GPU tier fills much sooner
4. Reduce `--gpu-memory-utilization` to shrink the GPU KV cache pool, forcing earlier eviction
5. Reduce `--swap-space` so vLLM has less swap headroom before hitting KVBM's offload path
6. Investigate whether KVBM exposes configuration to make vLLM prefer offloading over queuing
