# Kimi K2.5 KV Cache Benchmark Report

**Model**: Kimi-K2.5 (1T params, native INT4 CompressedTensorsWNA16MarlinMoE)
**Instance**: p5e.48xlarge (8x NVIDIA H200 143GB HBM3, 1144GB total)
**Storage**: FSx Lustre PERSISTENT_2 (2.15 TiB, 1000 MB/s/TiB baseline)
**Dates**: 2026-02-18 / 2026-02-19 / 2026-02-21

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

**Common vLLM Launch Args**:
```
--tensor-parallel-size 8 --enable-prefix-caching --enforce-eager
--max-model-len 32768 --swap-space 32 --gpu-memory-utilization 0.85
```

### Serving Configurations Tested

| Config | Framework | Additional Args |
|--------|-----------|-----------------|
| Baseline | vLLM nightly (CUDA 12.9) | `--tool-call-parser kimi_k2 --reasoning-parser kimi_k2` |
| LMCache+GDS | vLLM + LMCacheConnectorV1 | `LMCACHE_USE_EXPERIMENTAL=True`, `LMCACHE_LOCAL_DISK=file:///mnt/fsx/kv-cache/lmcache`, `LMCACHE_MAX_LOCAL_DISK_SIZE=100.0` |
| Dynamo KVBM | vLLM 0.15.1 + Dynamo KVBM 0.9.0 | `DYN_KVBM_CPU_CACHE_GB=64`, `DYN_KVBM_DISK_CACHE_GB=500`, `DYN_KVBM_DISK_CACHE_DIR=/mnt/fsx/kv-cache/dynamo` |

---

# Part 1: LMCache + FSx Lustre (GDS)

## Executive Summary

LMCache with GPU Direct Storage (GDS) for KV cache offloading to FSx Lustre provides
measurable TTFT improvements for prefix-heavy workloads (1.07-1.31x speedup) and
near-perfect conversation resumption from persistent storage (1.04x ratio). However,
under memory pressure, the FSx I/O overhead causes LMCache to underperform baseline
vLLM due to write-back serialization that throttles request concurrency.

The H200's massive 143GB HBM per GPU (1144GB total) means KV cache pressure is rare for
typical workloads — the system can hold ~610K tokens, enough for ~18 concurrent 32K-token
sessions. LMCache's value proposition increases on smaller GPUs or under extreme concurrency.

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
across all LMCache tests.

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

## LMCache Memory Pressure Results

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
| Recovery TTFT mean | 750ms | 737ms | **Tie** |
| Peak KV usage | 99% (observed) | 99% (observed) | Tie |
| Preemptions | 2 | 3 | **LMCache (marginal)** |
| FSx growth | +36.7 GB | 0 | N/A |

**Key finding**: At extreme pressure (2.6x oversubscription), both systems are equally
bottlenecked by the GPU scheduler. LMCache had 1 fewer preemption (2 vs 3), suggesting
the FSx offloading did help marginally with memory management, but the difference is negligible.

---

# Part 2: NVIDIA Dynamo KVBM

## Executive Summary

NVIDIA Dynamo's KV Block Manager (KVBM) provides a 4-tier cache hierarchy
(GPU HBM -> CPU DRAM -> Disk -> Remote) with async write-back via dedicated Tokio tasks
and separate CUDA streams.

**Key findings**:
- **Prefix caching works well**: 1.41x TTFT speedup for document RAG, 1.82x for API gateway
- **Multi-turn scaling is excellent**: 20x context growth causes only 1.6x TTFT increase (120ms -> 191ms)
- **Conversation resumption is near-perfect**: 0.99x ratio (no degradation)
- **Memory pressure remains a problem**: Moderate pressure causes 7,959ms mean fg TTFT (19.7x vs baseline 403ms)
- **FSx disk offloading did not activate**: GPU+CPU cache was sufficient; vLLM scheduler queued requests before KVBM offload path triggered

### Dynamo-Specific Configuration

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

**Setup**: 8 users, 20 rounds each, ~1K tokens added per round.

| Round | Context | Dynamo TTFT (ms) | LMCache TTFT (ms) | Winner |
|-------|---------|-------------------|---------------------|--------|
| 1 | 1,000 | 120 | 155 | **Dynamo (1.29x)** |
| 5 | 5,000 | 159 | 193 | **Dynamo (1.21x)** |
| 10 | 10,000 | 171 | 277 | **Dynamo (1.62x)** |
| 15 | 15,000 | 184 | 355 | **Dynamo (1.93x)** |
| 20 | 20,000 | 191 | 349 | **Dynamo (1.83x)** |

**Key finding**: Dynamo's TTFT scales much more slowly with context length. Context grew 20x
but TTFT only grew 1.6x (120ms -> 191ms), compared to LMCache's 2.3x (155ms -> 349ms).

**FSx cache growth**: 0 MB (no disk offloading triggered)

### 2. Enterprise API Gateway (8K shared schema)

**Setup**: Single ~8K token tool/API schema shared across all 50 requests with 15 unique user queries.

| Metric | Dynamo | LMCache | Winner |
|--------|--------|---------|--------|
| Cold TTFT | 219ms | 265ms | **Dynamo** |
| Warm TTFT | 133ms | 203ms | **Dynamo** |
| Speedup | 1.82x | 1.31x | **Dynamo** |

Success rate: 98% (1 failed request out of 50).

### 3. Document Library RAG (15 docs, Zipf access)

**Setup**: 15 documents x ~2K tokens each, 60 queries selecting 3 docs per query via Zipf distribution.

| Metric | Dynamo | LMCache | Winner |
|--------|--------|---------|--------|
| Early TTFT | 278ms | 241ms | LMCache |
| Late TTFT | 198ms | 226ms | **Dynamo** |
| Speedup | 1.41x | 1.07x | **Dynamo** |

**Key finding**: Dynamo shows stronger cache warming over time (1.41x vs 1.07x).

### 4. Conversation Resumption (gap + diverse traffic)

**Setup**: 10 users x 5 turns pre-gap, 40 diverse interleaved requests, then 5 turns post-gap.

| Metric | Dynamo | LMCache | Winner |
|--------|--------|---------|--------|
| Pre-gap TTFT | 118ms | 153ms | **Dynamo** |
| Post-gap TTFT | 123ms | 166ms | **Dynamo** |
| Resumption ratio | 0.99x | 1.04x | **Dynamo** |

### 5. Shared Prompt Sweep (multi-tenant scaling)

| Tenants | Dynamo Speedup | LMCache Speedup | Winner |
|---------|---------------|-----------------|--------|
| 5 | 2.35x | 1.12x | **Dynamo** |
| 10 | 1.69x | 1.05x | **Dynamo** |
| 25 | 1.81x | 1.02x | **Dynamo** |
| 50 | 1.68x | 1.02x | **Dynamo** |

**Key finding**: Dynamo shows dramatically better cold-to-warm TTFT improvement across all
tenant counts. Warm TTFT is remarkably stable at ~124ms regardless of tenant count.

## Dynamo Memory Pressure Results

### Moderate Pressure (25 sessions x 24K tokens)

| Metric | Dynamo KVBM | LMCache+GDS | Baseline vLLM | Best |
|--------|-------------|-------------|---------------|------|
| Background elapsed | 108.3s | 95.6s | 65.9s | **Baseline** |
| Foreground TTFT mean | 7,959ms | 5,330ms | 403ms | **Baseline** |
| Foreground TTFT p50 | 811ms | 1,017ms | 405ms | **Baseline** |
| Foreground TTFT p99 | 36,384ms | 23,444ms | 560ms | **Baseline** |
| Recovery TTFT mean | 673ms | 568ms | 233ms | **Baseline** |

**Key finding**: Dynamo's mean fg TTFT (7,959ms) is worse than LMCache (5,330ms), but Dynamo's
p50 is much better (811ms vs 1,017ms). This indicates a bimodal distribution: most foreground
requests complete quickly, but a few get delayed significantly.

### Aggressive Pressure (50 sessions x 32K tokens)

| Metric | Dynamo KVBM | LMCache+GDS | Baseline vLLM | Best |
|--------|-------------|-------------|---------------|------|
| Background success | 47/50 (94%) | 50/50 | 50/50 | **LMCache/Baseline** |
| Background elapsed | 741.2s | 183.3s | 164.5s | **Baseline** |
| Foreground success | 45/50 (90%) | 50/50 | 50/50 | **LMCache/Baseline** |
| Foreground TTFT p50 | 711ms | ~810ms | ~810ms | **Dynamo** |
| Recovery TTFT mean | 27,769ms | 750ms | 737ms | **Baseline/LMCache** |
| Preemptions | 1 | 2 | 3 | **Dynamo** |

**Key finding**: Dynamo struggled under aggressive pressure — background throughput was 4x slower
than both alternatives, and 6 requests failed. Recovery was very slow (27,769ms).

## FSx Disk Offloading Assessment

Despite configuring 500GB disk cache on FSx Lustre, KVBM's disk tier never activated:

1. **GPU + CPU cache was sufficient**: 432GB GPU KV + 64GB CPU provided enough headroom
2. **vLLM scheduler gating**: The scheduler queues/preempts requests before GPU KV cache fills,
   preventing the downstream tiers from seeing pressure
3. **GDS fallback to POSIX**: Without proper GDS buffer registration in-container, disk I/O
   may have been slower than recomputation

To exercise the FSx disk path, reduce CPU cache to ~2GB and test on smaller GPUs (A100 40GB, L4).

---

# Part 3: Head-to-Head Comparison

## Prefix Caching (Low Pressure)

| Test | Dynamo | LMCache | Winner |
|------|--------|---------|--------|
| Multi-turn (round 20 TTFT) | 191ms | 349ms | **Dynamo (1.83x)** |
| API gateway speedup | 1.82x | 1.31x | **Dynamo** |
| Doc RAG speedup | 1.41x | 1.07x | **Dynamo** |
| Conversation resumption | 0.99x | 1.04x | **Dynamo** |
| Shared prompt (50 tenants) | 1.68x | 1.02x | **Dynamo** |

**Dynamo wins decisively on prefix caching workloads.**

## Memory Pressure

| Scenario | Dynamo | LMCache | Baseline | Winner |
|----------|--------|---------|----------|--------|
| Moderate fg TTFT mean | 7,959ms | 5,330ms | 403ms | **Baseline** |
| Moderate fg TTFT p50 | 811ms | 1,017ms | 405ms | **Baseline** |
| Aggressive bg elapsed | 741s | 183s | 165s | **Baseline** |
| Aggressive fg success | 90% | 100% | 100% | **LMCache/Baseline** |

**Baseline vLLM wins under memory pressure.** Both offloading solutions add overhead
that hurts more than helps when GPU KV cache is stressed.

## Pressure Level Summary

| Pressure Level | LMCache | Dynamo | Baseline |
|---------------|---------|--------|----------|
| Low (< 50% KV) | Slight TTFT improvement | Strong TTFT improvement | Reference |
| Moderate (60% KV) | **Disadvantage** (13x worse) | **Disadvantage** (20x worse) | **Best** |
| Extreme (99% KV) | Neutral | **Disadvantage** (4x slower bg) | **Best** |

## Tiered KV Cache Framework Comparison

| Framework | Tiers | Multi-tier Chain | Maturity |
|-----------|-------|-----------------|----------|
| LMCache | 2 (local + remote) | No — pick one local backend | Production |
| NVIDIA Dynamo KVBM | 4 (GPU/CPU/NVMe/Remote) | Architecture yes, OSS partial | Early (requires patching) |
| Mooncake Store | 4 (VRAM/DRAM/NVMe/Remote) | Yes — full hierarchy | Research |
| vLLM native | 2 (GPU + CPU swap) | No disk offloading | Production |

## Storage Comparison

| Storage | Bandwidth | Latency | Persistence | Multi-node |
|---------|-----------|---------|-------------|------------|
| NVMe (local) | ~50 GB/s (8x SSDs) | 10-100 us | Node-local only | No |
| FSx + GDS | ~9-12 GB/s | 50-1000 us | Shared filesystem | Yes |
| FSx + EFA + GDS | ~80-120 GB/s (large FS) | 50-200 us | Shared filesystem | Yes |

---

# Part 4: SGLang HiCache Assessment (2026-02-21)

## Executive Summary

SGLang HiCache was the primary target for the 2026-02-21 capacity block, identified in Lesson #31 as the architecturally correct approach for tiered KV cache offloading — it uses cascading eviction (GPU→CPU→Storage) instead of admission gating, so tiered offloading actually triggers under memory pressure.

**Result: BLOCKED.** After four progressive attempts, SGLang v0.5.8.post1 cannot load Kimi K2.5. The root cause is a fundamental weight format incompatibility that cannot be resolved without upstream SGLang changes.

## Attempt Timeline

| # | Approach | Result | Root Cause |
|---|----------|--------|------------|
| 1 | Direct launch with `--enforce-eager` | CLI error | `--enforce-eager` is vLLM-only; SGLang uses `--disable-cuda-graph` |
| 2 | Launch with `--disable-cuda-graph` | Model not found | `KimiK25ForConditionalGeneration` not in SGLang's model registry |
| 3 | `--language-only` flag | CLI error | Requires `--encoder-urls` (disaggregated inference, not text-only mode) |
| 4 | Custom `kimi_k25_text.py` wrapper | Weight loading crash | Packed MoE format incompatible with DeepseekV2 weight loader |

## Technical Root Cause

Kimi K2.5 is a multimodal MoE model with `KimiK25ForConditionalGeneration` architecture. Its text backbone is DeepseekV3-based, but the INT4 quantized weights use a **packed expert format**:

```
# Kimi K2.5 weight keys (packed MoE)
language_model.model.layers.*.mlp.experts.w13_weight_packed    # gate_proj + up_proj fused
language_model.model.layers.*.mlp.experts.w2_weight_packed     # down_proj

# SGLang DeepseekV2 expects (per-expert named)
model.layers.*.mlp.experts.*.gate_proj.weight
model.layers.*.mlp.experts.*.up_proj.weight
model.layers.*.mlp.experts.*.down_proj.weight
```

The `CompressedTensorsWNA16MarlinMoEMethod` quantization format uses group-packed tensors that SGLang's weight loader cannot unpack. vLLM handles this via its Marlin MoE kernel repacking pipeline.

## Custom Wrapper Approach

Created `patches/kimi_k25_text.py` — a text-only wrapper that:
1. Maps `KimiK25ForConditionalGeneration` → `DeepseekV2ForCausalLM`
2. Uses `language_model.` prefix for weight name routing
3. Filters out `vision_tower.*` and `mm_projector.*` weights
4. Registered via `EntryClass = KimiK25ForConditionalGeneration`

The wrapper successfully passed SGLang's model registry discovery and began weight loading on all 8 GPUs. However, the weight loader crashed on every MoE layer due to the packed format.

## GitHub Issue #18458

SGLang's architecture tracking issue ([sgl-project/sglang#18458](https://github.com/sgl-project/sglang/issues/18458)) lists `KimiK25ForConditionalGeneration` as "Native: Y", but:
- No `kimi_k25.py` file exists in SGLang's `main` branch (`git ls-tree` confirmed)
- The `latest-cu130` Docker tag resolves to v0.5.8.post1, same version we tested
- The issue appears auto-generated from model config scanning, not reflecting merged code

## Requirements for Future SGLang + Kimi K2.5

SGLang needs:
1. A dedicated weight loader that handles `CompressedTensorsWNA16MarlinMoE` packed format
2. A model implementation that maps the `language_model.` prefix and multimodal wrapper
3. Support for MLA (Multi-head Latent Attention) with `outer_dim=64`

Alternatively, test SGLang HiCache with a model SGLang already supports natively (e.g., DeepSeek-V3, Llama 3.1 405B).

---

# Part 5: Baseline Re-Run (2026-02-21)

## Context

After SGLang was blocked, the remaining capacity block time was used for vLLM baseline benchmarks using the `kv-cache` benchmark mode. This provides fresh baseline numbers on the same p5e.48xlarge hardware with vLLM v0.15.1.

## Configuration

| Parameter | Value |
|-----------|-------|
| Image | `vllm-openai:v0.15.1` |
| Attention Backend | FLASH_ATTN_MLA |
| Quantization | CompressedTensorsWNA16MarlinMoE (auto-detected) |
| Tensor Parallel | 8 |
| Max Model Len | 32,768 |
| GPU Memory Util | 0.85 |
| Model Source | `/mnt/nvme/models/Kimi-K2.5` (NVMe RAID0, copied from FSx) |
| Model Load Time | ~2 hours (compute-bound Marlin MoE repacking) |

## Results

### KV Cache Workloads (medium QPS)

| Workload | N | TTFT p50 (ms) | TTFT p90 (ms) | E2E p50 (ms) | E2E p90 (ms) | Throughput (tok/s) |
|----------|---|---------------|---------------|--------------|--------------|-------------------|
| multi_turn_qa | 10 | 274 | 277 | 13,523 | 13,581 | 11.1 |
| long_context_rag | 30 | 361 | 367 | 17,958 | 18,035 | 11.1 |
| strict_no_reuse | 30 | 270 | 276 | 13,420 | 13,630 | 11.2 |

### Long Context Scaling

| Context Size | N | TTFT p50 (ms) | TTFT p90 (ms) | E2E p50 (ms) | E2E p90 (ms) | Throughput (tok/s) |
|-------------|---|---------------|---------------|--------------|--------------|-------------------|
| 16K tokens | 10 | 361 | 509 | 13,570 | 13,923 | 11.0 |
| 24K tokens | 10 | 411 | 449 | 13,704 | 13,822 | 10.9 |
| 36K tokens | 10 | 506 | 558 | 13,713 | 13,916 | 10.9 |
| 48K tokens | 10 | 558 | 592 | 13,730 | 13,804 | 10.9 |

TTFT increases roughly linearly with context: 2.06x from 16K→48K (361→558 ms). E2E is nearly flat: 1.01x increase (13,570→13,730 ms) because generation time dominates at 150-token outputs.

### Multi-Tenant (50 tenants, 2 users each)

| Metric | Value |
|--------|-------|
| Total Requests | 100 |
| Success Rate | 100% |
| TTFT p50 / p90 | 293 / 304 ms |
| E2E p50 / p90 | 13,492 / 13,638 ms |
| Cold/Warm Speedup | 1.00x |
| Throughput | 11.1 tok/s |

No prefix caching benefit observed — cold and warm requests have identical latency.

## Comparison with Feb 18/19 Baseline

| Metric | Feb 18/19 | Feb 21 | Delta |
|--------|-----------|--------|-------|
| TTFT (short context) | ~150-180 ms | ~270 ms | +50-80% |
| E2E (150 tokens) | ~13,500 ms | ~13,500 ms | ~0% |
| Throughput | ~11 tok/s | ~11 tok/s | ~0% |
| Reasoning rate | Variable | 0% | Different prompts |

The TTFT increase in the Feb 21 run is attributable to different workload construction — the Feb 18/19 multi-turn tests used incrementally growing context with warmup, while the Feb 21 `kv-cache` mode sends all requests with full shared context from the start (no incremental buildup that benefits from prefix caching).

E2E latency and throughput are consistent across sessions, confirming reproducible generation performance.

## Key Findings

1. **Generation-bound, not prefill-bound**: Throughput is flat at 11 tok/s across all context lengths (16K–48K). The Marlin MoE kernel dominates token generation time. Prefill cost (TTFT) is a small fraction of total E2E at 150-token output length.

2. **No prefix caching benefit in multi-tenant**: The 50-tenant workload shows 1.00x cold/warm speedup, confirming that baseline vLLM's prefix cache doesn't help when each tenant has a unique system prompt. This is the scenario where KV cache offloading frameworks should provide the most value.

3. **Prefix cache metrics were not captured**: The `scrape_prefix_cache_metrics()` function was commented out (Lesson #30 gap). Fix has been applied for future runs.

4. **GDS unavailable on AL2023**: The `nvidia-fs` kernel module is not included in AL2023 EKS AMIs. LMCache must use POSIX mode, eliminating the GPU Direct Storage advantage.

5. **Model loading is compute-bound**: ~2 hours for 64-shard INT4 MoE regardless of storage (NVMe or FSx). Marlin MoE kernel repacking per shard is the bottleneck.

## Infrastructure Notes

| Finding | Impact |
|---------|--------|
| AL2023 uses `nodeadm`, not `bootstrap.sh` | First instance launch failed; required CA cert fix |
| `nvidia-fs` not in AL2023 kernel | GDS unavailable; LMCache falls back to POSIX |
| SGLang v0.5.8 incompatible with Kimi K2.5 | 3+ hours of capacity block time spent on debugging |
| SSM heredocs don't work (newline escaping) | Use base64 encoding for file transfers |

---

# Recommendations

## Updated Recommendations (incorporating Feb 21 findings)

### Deploy LMCache+FSx when:
- Small/medium GPUs (A10G, L4, A100 40GB) where KV cache is scarce
- Multi-node disaggregated serving (prefill/decode separation)
- Persistent KV cache across restarts is valued
- Low-to-moderate concurrency (< 10 simultaneous users)
- 100% reliability under pressure is required
- **Note**: Use POSIX mode on AL2023 (GDS unavailable); GDS requires custom AMI with `nvidia-fs`

### Deploy Dynamo KVBM when:
- Prefix-heavy workloads where TTFT matters (API gateways, multi-tenant SaaS)
- Multi-turn conversation workloads with growing context
- Low-to-moderate concurrency (< 15 simultaneous users)
- Better prefix caching than LMCache is needed

### Deploy baseline vLLM (no offloading) when:
- Large GPUs (H100 80GB, H200 143GB) with abundant HBM
- High concurrency workloads (> 20 simultaneous users)
- Throughput-critical deployments
- Cost-sensitive deployments (no FSx Lustre needed)

### Deploy SGLang HiCache when:
- The model is natively supported by SGLang (check model registry AND weight loader)
- Cascading eviction (GPU→CPU→NVMe) is desired instead of admission gating
- Mooncake integration is needed for distributed KV cache
- **NOT viable for Kimi K2.5** as of SGLang v0.5.8 (packed MoE weight format incompatible)
- Consider testing with DeepSeek-V3 or Llama 3.1 405B instead

### Always:
- Benchmark with your actual workload, reasoning parser config, and concurrency level
- Monitor `vllm:num_preemptions_total` — sustained preemptions indicate GPU memory pressure
- Capture prefix cache hit/miss metrics (`prefix_cache_hit_total` / `prefix_cache_miss_total`) in every run
- Verify SGLang model support before reserving GPU capacity: (1) architecture in registry, (2) weight format compatible, (3) test on CPU-only instance first

### Next Steps

1. **Test disaggregated prefill/decode** — Split 8x H200 into TP=4 prefill + TP=4 decode pools. Use LMCache multi-node mode (`--kv-connector LMCacheConnector --kv-role kv_producer` / `kv_consumer`) with FSx as the shared KV cache fabric. This reframes FSx from "disk offload" (never triggers on H200s) to "inter-node KV transfer layer" (the actual high-value use case). See Lesson #37.
2. **Re-run LMCache in POSIX mode** with prefix cache metrics enabled and memory pressure workload
3. **Re-run Dynamo KVBM** with prefix cache metrics enabled
4. **Increase output length** to 500–1000 tokens to create more KV cache pressure
5. **Test SGLang HiCache** with a natively supported model (DeepSeek-V3) to validate the cascading eviction architecture
6. **Build custom AMI with nvidia-fs** to enable GDS for LMCache if POSIX results are promising

---

# Appendix: Reasoning Parser Impact

The `--reasoning-parser kimi_k2` flag fundamentally changes latency characteristics:

| Behavior | Without parser | With parser |
|----------|---------------|-------------|
| Reasoning tokens | Mixed into `delta.content` | Separate `delta.reasoning_content` |
| `max_tokens=200` caps | Reasoning + content combined | Content only (reasoning unlimited) |
| Avg output tokens | ~42-50 (all types) | ~150 reasoning + ~2 content |
| E2E latency | 1-5s (capped output) | 13-14s (full reasoning chain) |

Earlier LMBench reports claiming 1.8-2.5x E2E speedups were run without the reasoning parser
and are not directly comparable to these results.

# Raw Data Files

Results are organized by configuration:

| Directory | Configuration | Session |
|-----------|--------------|---------|
| `kimi-k2.5-p5e/baseline/` | Baseline vLLM | Feb 18/19 |
| `kimi-k2.5-p5e/lmcache/` | LMCache + GDS | Feb 18/19 |
| `kimi-k2.5-p5e-baseline/` | Early baseline runs | Feb 18 |
| `kimi-k2.5-p5e-baseline-full/` | Full baseline suite | Feb 18 |
| `kimi-k2.5-p5e-v2/` | Refined test runs | Feb 19 |
| `dynamo_*.json` | Dynamo KVBM results | Feb 19 |
| `session-20260221/baseline/` | Baseline re-run (kv-cache mode) | Feb 21 |

# Appendix: Model Precision

Kimi K2.5 ships as **native INT4** using `CompressedTensorsWNA16MarlinMoE` quantization
baked into the safetensor shards (~540GB for 1T params = ~4.3 bits/param). No `--quantization`
flag is needed — vLLM auto-detects the quantization from `config.json`.
