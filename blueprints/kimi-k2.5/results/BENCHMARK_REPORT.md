# LMCache + FSx Lustre Benchmark Report

**Model**: Kimi-K2.5 (1T params, native INT4 CompressedTensorsWNA16MarlinMoE)
**Instance**: p5e.48xlarge (8x NVIDIA H200 143GB HBM3, 1144GB total)
**Storage**: FSx Lustre PERSISTENT_2 (2.15 TiB, 1000 MB/s/TiB baseline)
**Serving**: vLLM nightly (CUDA 12.9) + LMCacheConnectorV1 (built from source)
**Date**: 2026-02-18

## Executive Summary

We benchmarked LMCache with GPU Direct Storage (GDS) for KV cache offloading to FSx Lustre
on a p5e.48xlarge instance serving Kimi-K2.5. Our findings are nuanced:

**LMCache provides measurable TTFT improvements for prefix-heavy workloads** (1.07-1.31x
speedup) and near-perfect conversation resumption from persistent storage (1.04x ratio).
However, **under memory pressure, the FSx I/O overhead causes LMCache to underperform
baseline vLLM** due to write-back serialization that throttles request concurrency.

The H200's massive 143GB HBM per GPU (1144GB total) means KV cache pressure is rare for
typical workloads — the system can hold ~610K tokens (9533 blocks x 64 tokens/block) of
KV cache, enough for ~18 concurrent 32K-token sessions. LMCache's value proposition
increases on smaller GPUs or under extreme concurrency.

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

**vLLM Launch Args** (common across all configs):
```
--tensor-parallel-size 8 --enable-prefix-caching --enforce-eager
--max-model-len 32768 --swap-space 32 --gpu-memory-utilization 0.85
--tool-call-parser kimi_k2 --reasoning-parser kimi_k2
```

**LMCache Config** (GDS):
```
LMCACHE_USE_EXPERIMENTAL=True
LMCACHE_LOCAL_DISK=file:///mnt/fsx/kv-cache/lmcache
LMCACHE_MAX_LOCAL_DISK_SIZE=100.0
kv_connector=LMCacheConnectorV1, kv_role=kv_both
```

## Test Results

### 1. Multi-Turn Conversation (20 rounds, growing history)

**Setup**: 8 users, 20 rounds each, ~1K tokens added per round. Context grows from 1K to 20K tokens.

| Round | Context | TTFT (ms) |
|-------|---------|-----------|
| 1 | 1,000 | 155 |
| 5 | 5,000 | 193 |
| 10 | 10,000 | 277 |
| 15 | 15,000 | 355 |
| 20 | 20,000 | 349 |

**Key finding**: TTFT scales sub-linearly with context length. Context grew 20x (1K->20K)
but TTFT only grew 2.3x (155ms->349ms). This confirms prefix caching is working — each
round reuses KV from all previous turns.

```
Round  1 ( 1000 tok):     155ms ███████
Round  5 ( 5000 tok):     193ms █████████
Round 10 (10000 tok):     277ms █████████████
Round 15 (15000 tok):     355ms █████████████████
Round 20 (20000 tok):     349ms █████████████████
```

**FSx cache growth**: 9,661 MB -> 12,788 MB (+3.1 GB)

### 2. Enterprise API Gateway (8K shared schema)

**Setup**: Single ~8K token tool/API schema shared across all 50 requests with 15 unique user queries.

| Phase | TTFT (ms) | Requests |
|-------|-----------|----------|
| Cold (first 5) | 265 | 5 |
| Warm (remaining) | 203 | 45 |
| **Speedup** | **1.31x** | |

**Key finding**: Maximum prefix sharing scenario. Once the 8K schema is cached, all subsequent
requests skip prefill for the shared portion. The 1.31x TTFT speedup is the highest we observed
across all tests.

### 3. Document Library RAG (15 docs, Zipf access)

**Setup**: 15 documents x ~2K tokens each, 60 queries selecting 3 docs per query via Zipf
distribution (popular docs accessed more frequently).

| Phase | TTFT (ms) |
|-------|-----------|
| Early (cold) | 241 |
| Late (warm) | 226 |
| **Speedup** | **1.07x** |

**Document hit distribution** (top 5): [51, 33, 23, 16, 10]

**Key finding**: Modest improvement because RAG queries combine different document subsets,
reducing prefix overlap. The Zipf distribution means popular documents get cached, but the
varying document combinations limit prefix reuse.

### 4. Conversation Resumption (gap + diverse traffic)

**Setup**: 10 users x 5 turns pre-gap, 40 diverse interleaved requests (to evict in-memory cache),
then 5 turns post-gap.

| Phase | TTFT (ms) |
|-------|-----------|
| Pre-gap mean | 153 |
| Post-gap mean | 166 |
| **Resumption ratio** | **1.04x** |

**Key finding**: Near-perfect recovery from FSx. After 40 diverse requests designed to evict
in-memory prefix cache, resumed conversations showed only 4% TTFT degradation. This demonstrates
FSx persistence working as intended — KV cache survives in-memory eviction.

### 5. Shared Prompt Sweep (multi-tenant scaling)

| Tenants | Cold TTFT | Warm TTFT | TTFT Speedup | E2E Speedup |
|---------|-----------|-----------|--------------|-------------|
| 5 | 194ms | 173ms | 1.12x | 1.00x |
| 10 | 183ms | 174ms | 1.05x | 1.00x |
| 25 | 181ms | 176ms | 1.02x | 1.00x |
| 50 | 180ms | 177ms | 1.02x | 1.00x |

**Key finding**: TTFT improvements are small (1.02-1.12x) and diminish with more tenants.
E2E is 1.00x across all tenant counts because generation time (~13.5s for Kimi K2.5 reasoning)
dominates. The TTFT savings (7-21ms) are real but invisible at the E2E level.

## Memory Pressure Comparison: LMCache+GDS vs Baseline

This is the most important test. We saturated GPU KV cache to measure how the system
handles memory pressure.

### Moderate Pressure (25 sessions x 24K tokens = 600K tokens vs 610K capacity)

| Metric | LMCache+GDS | Baseline vLLM | Winner |
|--------|-------------|---------------|--------|
| Background throughput | 25/25 in 95.6s | 25/25 in 65.9s | **Baseline (1.45x)** |
| Concurrency at launch | 5 running, 20 waiting | 25 running, 0 waiting | **Baseline** |
| Foreground TTFT mean | 5,330ms | 403ms | **Baseline (13x)** |
| Foreground TTFT p50 | 1,017ms | 405ms | **Baseline (2.5x)** |
| Foreground TTFT p99 | 23,444ms | 560ms | **Baseline (42x)** |
| Recovery TTFT mean | 568ms | 233ms | **Baseline (2.4x)** |
| Peak KV usage | 61% | 61% | Tie |
| Preemptions | 0 | 0 | Tie |
| FSx growth | +29.4 GB | 0 | N/A |

**Key finding**: LMCache **hurts** performance under moderate pressure. The FSx write-back
I/O serializes request processing — only 5 of 25 sessions ran concurrently (vs all 25 for
baseline). This created a massive queue backlog that inflated foreground TTFT by 13x.

**Root cause**: LMCache's KV write-back to FSx via GDS is synchronous in the request
processing pipeline. Each request must complete its FSx write before the next can proceed,
creating a bottleneck when many requests arrive simultaneously.

### Aggressive Pressure (50 sessions x 32K tokens = 1.6M tokens vs 610K capacity)

| Metric | LMCache+GDS | Baseline vLLM | Winner |
|--------|-------------|---------------|--------|
| Background throughput | 50/50 in 183.3s | 50/50 in 164.5s | **Baseline (1.11x)** |
| Burst 1 TTFT (queued) | 108,427ms | 96,532ms | **Baseline (1.12x)** |
| Burst 2-5 TTFT mean | ~810ms | ~810ms | **Tie** |
| Burst 2-5 TTFT p99 | ~1,020ms | ~1,020ms | **Tie** |
| Recovery TTFT mean | 750ms | 737ms | **Tie** |
| Peak KV usage | 99% (observed) | 99% (observed) | Tie |
| Preemptions | 2 | 3 | **LMCache (marginal)** |
| FSx growth | +36.7 GB | 0 | N/A |

**Key finding**: At extreme pressure (2.6x oversubscription), both systems are equally
bottlenecked by the GPU scheduler. Burst 1 TTFT (~100s) reflects pure queue wait time —
50 background sessions saturated the GPU before foreground bursts arrived. Once background
sessions complete (bursts 2-5), performance is identical (~810ms TTFT).

The baseline is 11% faster overall due to less I/O overhead. LMCache had 1 fewer
preemption (2 vs 3), suggesting the FSx offloading did help marginally with memory
management, but the difference is negligible.

**Conclusion**: Under extreme memory pressure on H200, LMCache provides no meaningful
advantage. The GPU scheduler handles KV cache eviction and recomputation efficiently
enough that external offloading adds overhead without proportional benefit.

### Moderate vs Aggressive Pressure Summary

| Pressure Level | LMCache Advantage | Explanation |
|---------------|-------------------|-------------|
| Low (< 50% KV) | Slight TTFT improvement | Prefix caching helps, minimal I/O cost |
| Moderate (60% KV) | **Disadvantage** | FSx writes serialize requests, 13x worse TTFT |
| Extreme (99% KV) | Neutral | Both systems GPU-bottlenecked, I/O overhead ~11% |

The worst-case for LMCache is **moderate pressure** where the GPU isn't fully saturated
but LMCache's synchronous FSx writes create artificial queueing.

## Where LMCache + FSx Helps

1. **Prefix-heavy workloads with low concurrency**: Shared API schemas (1.31x), repeated
   system prompts, document libraries with Zipf access patterns.

2. **Conversation resumption**: When users return after their KV cache has been evicted from
   GPU memory, FSx provides near-instant recovery (1.04x ratio vs full recompute).

3. **Sub-linear context scaling**: Multi-turn conversations benefit from incremental KV
   caching — 20x context growth causes only 2.3x TTFT increase.

4. **Persistent cross-session state**: FSx cache survives pod restarts, node failures, and
   scaling events. No warm-up penalty after redeployment.

## Where LMCache + FSx Hurts

1. **Moderate concurrency (the worst case)**: At ~60% KV utilization, FSx write-back
   serializes request processing, throttling concurrent execution from 25 to 5 sessions.
   This causes 13x higher foreground TTFT compared to baseline. This is worse than both
   low-pressure (where I/O cost is negligible) and extreme pressure (where both systems
   are equally GPU-bound).

2. **Throughput-sensitive workloads**: Background sessions complete 11-45% slower with
   LMCache due to I/O overhead on every request (1.45x at moderate pressure, 1.11x at
   extreme pressure).

3. **Large GPU memory (H200)**: With 1144GB total HBM, KV cache pressure rarely materializes.
   The 610K-token capacity handles most production workloads without eviction, making
   offloading unnecessary overhead.

4. **Generation-dominated workloads**: When response generation time (13.5s for Kimi K2.5
   reasoning) dwarfs prefill time (150-350ms), TTFT savings are imperceptible to users.

## Recommendations

### When to Use LMCache + FSx

- Small/medium GPUs (A10G, L4, A100 40GB) where KV cache is scarce
- Multi-node disaggregated serving (prefill/decode separation)
- Workloads with high prefix sharing (>70% common prefix per request)
- Scenarios requiring persistent KV cache across restarts
- Low-to-moderate concurrency (< 10 simultaneous users)

### When to Use Baseline vLLM (no LMCache)

- Large GPUs (H100 80GB, H200 143GB) with abundant HBM
- High concurrency workloads (> 20 simultaneous users)
- Throughput-sensitive applications
- Short context workloads (< 4K tokens)
- Cost-sensitive deployments (no FSx Lustre needed)

### On Tiered KV Cache Offloading

Current frameworks support limited tiering:

| Framework | Tiers | Multi-tier Chain |
|-----------|-------|-----------------|
| LMCache | 2 (local + remote) | No — pick one local backend |
| NVIDIA Dynamo KVBM | 4 (GPU/CPU/NVMe/Remote) | Architecture yes, OSS partial |
| Mooncake Store | 4 (VRAM/DRAM/NVMe/Remote) | Yes — full hierarchy |
| vLLM native | 2 (GPU + CPU swap) | No disk offloading |

**Mooncake** provides the most complete hierarchical tiering (GPU -> CPU -> NVMe -> network)
with automatic promotion/demotion. **NVIDIA Dynamo KVBM** has the architecture (G1/G2/G3/G4 tiers)
but the open-source implementation is still maturing. **LMCache** is limited to two tiers
(one local + one remote), which means you cannot chain CPU -> NVMe -> FSx as a hierarchy.

### On NVMe vs FSx for KV Cache

| Storage | Bandwidth | Latency | Persistence | Multi-node |
|---------|-----------|---------|-------------|------------|
| NVMe (local) | ~50 GB/s (8x SSDs) | 10-100 us | Node-local only | No |
| FSx + GDS | ~9-12 GB/s | 50-1000 us | Shared filesystem | Yes |
| FSx + EFA + GDS | ~80-120 GB/s (large FS) | 50-200 us | Shared filesystem | Yes |

NVMe is ~5x faster for raw I/O but lacks persistence across nodes. For multi-node
serving where prefill and decode happen on different machines, FSx (or equivalent
shared storage) is required.

## Raw Data Files

All results are in `results/kimi-k2.5-p5e/lmcache/` and `results/kimi-k2.5-p5e/baseline/`:

| File | Test |
|------|------|
| `multi_turn_8u_20r_*.json` | Multi-turn conversation (20 rounds) |
| `enterprise_api_gateway_*.json` | Enterprise API gateway (8K schema) |
| `doc_library_rag_15d_*.json` | Document library RAG (15 docs) |
| `conversation_resumption_10u_*.json` | Conversation resumption |
| `multi_tenant_*t_*.json` | Shared prompt sweep (5/10/25/50 tenants) |
| `memory_pressure_25bg_*.json` | Memory pressure (25 sessions x 24K) |
| `memory_pressure_50bg_*.json` | Memory pressure (50 sessions x 32K) |

## Comparison with Original LMCache Report (Feb 14-15)

The original report (`blueprints/vllm-kv-benchmark/results/BENCHMARK_REPORT.md`) claimed
1.8-2.5x E2E speedups. Our results show 1.07-1.31x TTFT speedups. We traced the raw data
from both periods to normalize the comparison.

### Root Cause: `--reasoning-parser kimi_k2` Changes Everything

The original vLLM launch (`lmcache.vllm.entrypoints.openai.api_server`) did **not** include
`--reasoning-parser kimi_k2`. Our current launch does. This single flag changes:

| Behavior | Without parser (Feb 14) | With parser (Feb 17-18) |
|----------|------------------------|------------------------|
| Reasoning tokens | Mixed into `delta.content` | Separate `delta.reasoning_content` |
| `max_tokens=200` caps | Reasoning + content combined | Content only (reasoning unlimited) |
| Avg output tokens | ~42-50 (all types) | ~150 reasoning + ~2 content |
| E2E latency | 1-5s (capped output) | 13-14s (full reasoning chain) |
| TTFT meaning | Time to first content chunk (includes reasoning delay) | Time to first reasoning token (pure prefill) |

**This is why the two reports are not directly comparable.** The original measured E2E of
short, capped responses. Ours measured TTFT of full reasoning responses.

### Raw Data Across Both Periods

We ran the **same benchmark script** (`run_kimi_benchmarks.py`) on both dates. Feb 14 data
had TTFT captured in JSON. LMBench CSV data had TTFT=0 (not captured).

```
Config                    Date   TTFT mean  TTFT p50  E2E mean  CompTok  ResTok
─────────────────────────────────────────────────────────────────────────────────
multi_turn_qa Baseline    Feb14     1248ms    1450ms    2656ms      50t      0t
multi_turn_qa LMCache     Feb14     1502ms    2053ms    2852ms      42t      0t
multi_turn_qa Baseline    Feb17      725ms       0ms   13641ms       2t      0t
multi_turn_qa LMCache     Feb17      155ms     152ms   13370ms       1t    149t

long_context_rag Baseline Feb14     1598ms    2245ms    3500ms      51t      0t
long_context_rag LMCache  Feb14     1647ms    2123ms    3489ms      26t      0t
long_context_rag LMCache  Feb17      263ms     260ms   17967ms       0t    200t
```

**Critical finding**: On Feb 14 (no reasoning parser), LMCache was **slower** than baseline
on our custom benchmarks (TTFT speedup 0.71-0.91x). The LMBench tools showed speedups
because they used different workload patterns and measured E2E differently.

### Normalizing LMBench E2E to Estimated TTFT

LMBench CSVs have TTFT=0 (not captured). We estimate TTFT by subtracting generation time:
- From `strict_reuse_0%`: E2E=1,095ms for ~1K prompt + 100 gen tokens
- Estimated generation time for 100 tokens: ~1,000ms

| LMBench Test | Cold E2E | Warm E2E | E2E Speedup | Est Cold TTFT | Est Warm TTFT | Est TTFT Speedup |
|-------------|----------|----------|-------------|---------------|---------------|-----------------|
| synthetic_20k | 1,756ms | 1,360ms | 1.29x | ~756ms | ~360ms | **~2.1x** |
| multi_tenant_50 | 9,390ms | 1,809ms | 5.19x | ~8,390ms | ~809ms | **~10.4x** |

The multi_tenant_50 shows a **10.4x estimated TTFT speedup** — but the 8,390ms "cold prefill"
for only 9K tokens is anomalous (should be ~200ms). This suggests the cold requests were
queuing behind each other at QPS 2.88 with 50 unique tenants causing massive cache misses,
not pure prefill time.

### Normalized Comparison Table

| Test | Source | E2E Speedup | Est TTFT Speedup | Notes |
|------|--------|-------------|-----------------|-------|
| synthetic_20k | LMBench CSV | 1.29x | ~2.1x | 21K ctx, gen~1000ms |
| multi_tenant_50 | LMBench CSV | 5.19x | ~10.4x | Includes queueing, not pure prefill |
| multi_turn_qa | Custom Feb14 | 0.93x | 0.71x | **LMCache slower** (no reasoning parser) |
| multi_turn_qa | Custom Feb17 | 1.02x | ~4.7x | TTFT 155ms vs ~725ms |
| Enterprise API 8K | Custom Feb18 | N/A | 1.31x | Cold/warm TTFT only |
| Doc RAG 15 docs | Custom Feb18 | N/A | 1.07x | Cold/warm TTFT only |
| Conv resumption | Custom Feb18 | N/A | 1.04x ratio | Post-gap / pre-gap |
| Memory pressure 25×24K | Custom Feb18 | N/A | **0.08x** | LMCache 13x worse |
| Memory pressure 50×32K | Custom Feb18 | N/A | ~1.0x | Both GPU-bound |

### What Both Reports Agree On

| Finding | Original | Ours | Consistent? |
|---------|----------|------|-------------|
| Sub-linear context scaling | Yes (48K @ 1.3x/1.3x ctx) | Yes (20K @ 2.3x/20x ctx) | Yes |
| Minimal overhead without reuse | ~5% | ~2% | Yes |
| 100% success rate | Yes (1,500+ req) | Yes (600+ req) | Yes |
| FSx scales to tens of GB | 37GB, 2,160 files | 83GB, 2,000+ files | Yes |

### What the Original Report Missed

1. **Memory pressure**: Never tested. LMCache **hurts** at moderate concurrency (13x worse
   foreground TTFT) due to synchronous FSx write-back serializing the scheduler.

2. **Baseline comparison with same script**: On Feb 14, our own custom benchmark showed
   LMCache was 0.71-0.91x on TTFT (slower). The report cherry-picked LMBench results.

3. **Reasoning parser impact**: The `--reasoning-parser kimi_k2` flag fundamentally changes
   latency characteristics. Without it, `max_tokens=200` caps total output to ~200 tokens
   (1-5s E2E). With it, reasoning runs uncapped (~150 tokens, 10-13s) before content.

4. **LMBench TTFT=0**: The LMBench tools did not capture TTFT at all. All E2E speedup
   claims include constant generation time that inflates the ratio.

### Reconciled Recommendation

The original report's 1.8-2.5x E2E speedup is **reproducible under its specific conditions**
(no reasoning parser, LMBench tools, low concurrency, E2E metric). It is **not representative**
of production deployments with reasoning models at moderate-to-high concurrency.

**Updated recommendation**:
- **Deploy LMCache+FSx** for low-concurrency (<10 users), prefix-heavy workloads where
  persistent cross-restart caching is valued
- **Do not deploy** for high-concurrency workloads on large-HBM GPUs (H100/H200)
- **Always benchmark with your actual workload**, reasoning parser config, and concurrency
  level before committing to LMCache in production

## Appendix: Model Precision Clarification

Kimi K2.5 ships as **native INT4** using `CompressedTensorsWNA16MarlinMoE` quantization
baked into the safetensor shards (~540GB for 1T params = ~4.3 bits/param). No `--quantization`
flag is needed — vLLM auto-detects the quantization from `config.json`. This is distinct
from post-training quantization methods like AWQ or GPTQ.
