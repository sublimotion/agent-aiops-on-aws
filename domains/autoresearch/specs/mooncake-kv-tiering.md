# Autoresearch Spec: Mooncake KV Cache Tiering on Kimi K2.6

## Status: DRAFT

## Overview

Evaluate Mooncake as an L3 KV cache backend for SGLang HiCache on Kimi K2.6 (B300), measuring whether tiered caching (GPU VRAM → Host DRAM → NVMe/RDMA) closes the 3.1x throughput gap between SGLang and vLLM observed in the K2.6 benchmark.

**Core hypothesis**: SGLang + HiCache + Mooncake L3 with RDMA can match or exceed vLLM's 10,437 tok/s throughput at c=512 by enabling cascading KV eviction to NVMe/remote storage, freeing GPU memory for larger batch sizes.

**Why this matters**:
- K2.6 benchmark showed vLLM 3.1x over SGLang at scale (10,437 vs 3,400 tok/s at c=512). SGLang saturated at c=128. The bottleneck is KV cache memory pressure under high concurrency — exactly what tiered caching should solve.
- HiCache alone gave +58% single-stream throughput but **no benefit under load** (compute-bound, not KV-bound at c=16+). Mooncake's L3 tier (NVMe + RDMA) extends the capacity further.
- K2.5 assessment (v2.0) identified SGLang + HiCache + Mooncake as the architecturally superior approach (cascading eviction > admission gating), but Phases 3-5 were never executed.
- Mooncake is Moonshot's own serving infrastructure (FAST 2025 Best Paper, 525% throughput gain in production). Testing it on their model with our hardware validates the partnership path.
- B300's 30.4TB local NVMe provides massive L3 capacity (vs typical cloud instances with 1-2TB).

**Evolution from K2.5 Mooncake assessment**:

| Dimension | K2.5 Assessment | This Spec |
|---|---|---|
| Hardware | p5e (8x H100 80GB, 640GB VRAM) | p6-b300 (8x B300 268GB, 2.15TB VRAM) |
| SGLang version | v0.4.x (HiCache experimental) | v0.5.10+ (HiCache stable) |
| Mooncake integration | Theoretical (configs written, not run) | Execute Phases 3-5 from assessment |
| Baseline comparison | LMCache + Dynamo KVBM | vLLM FLASHINFER_MLA (10,437 tok/s) |
| Context tested | 32K (memory constrained) | Up to 128K (B300 has headroom) |
| NVMe capacity | Limited | 30.4 TB (8x 3.8TB) |

---

## Components

### 1. Compute

- **Platform**: EKS on EC2 (spot or capacity block)
- **Instance Type**: p6-b300.48xlarge (8x B300 268GB HBM3e, NVLink 5 / NVSwitch)
- **EKS Cluster**: `qn-sglang-eks-cluster` (v1.32, us-west-2)
- **NVMe**: 8x 3.8TB local NVMe SSDs (30.4TB total) — critical for Mooncake L3
- **System RAM**: 4 TB — HiCache L2 tier
- **EFA**: Yes — required for RDMA tier (cross-node Mooncake, if multi-node tested)

### 2. Codebase

- **Source**:
  - `github.com/kvcache-ai/Mooncake` — Mooncake transfer engine
  - `github.com/sgl-project/sglang` — SGLang with HiCache + Mooncake backend
  - Existing configs: `domains/gpu-serving/blueprints/kimi-k2.5/configs/sglang-mooncake.sh`

- **Fixed files** (define the metric):
  - Benchmark workloads from K2.6 spec (W1-W6, pressure test)
  - vLLM baseline results (10,437 tok/s @ c=512, TTFT/TPOT from K2.6 benchmark)
  - `scripts/benchmark-serving.py`

- **Agent-editable files**:
  - SGLang + Mooncake launch configs
  - HiCache sizing parameters (`--hicache-size`, `--hicache-storage-backend`)
  - Mooncake configuration (tiers, prefetch policy, transfer engine settings)
  - Benchmark orchestration scripts

- **Agent instructions**: `domains/autoresearch/blueprints/mooncake-kv-tiering/program.md`

### 3. Experiment Protocol

#### Metric (higher is better)
- **Primary**: Aggregate throughput (tok/s) at c=128 and c=512
- **Secondary**: TTFT p50/p99, TPOT p50/p99, KV cache hit rate, tier utilization (GPU/DRAM/NVMe), eviction rate
- **Baseline**: vLLM v0.19.1 FLASHINFER_MLA = 10,437 tok/s @ c=512

#### Time budget
- **Per configuration**: ~30 min (cold start + 6 workloads + pressure test)
- **Total**: 1 capacity block session (~8 hours)

#### Loop structure

```
PHASE 1: SGLang HiCache Baseline on K2.6 (re-establish)
  Config: --enable-hierarchical-cache --hicache-size 200
  Run: W1-W6 workloads + pressure test (c=1,4,16,64,128,256,512)
  Compare: vs K2.6 vLLM baseline and K2.6 SGLang baseline (already have data)
  Measure: tier utilization (how much spills from GPU to DRAM)

PHASE 2: HiCache + NVMe L3
  Config: --hicache-storage-backend local --hicache-local-path /mnt/nvme/kv-cache
  Sweep: hicache-size 100/200/300 GB/rank
  Run: Same workloads
  Measure: NVMe tier hit rate, eviction frequency, throughput delta

PHASE 3: HiCache + Mooncake L3
  Config: --hicache-storage-backend mooncake
  Setup: Mooncake metadata server, transfer engine configuration
  Sweep: prefetch policy (none, timeout, predictive)
  Run: Same workloads
  Measure: Mooncake transfer latency, RDMA vs TCP, throughput delta

PHASE 4: Pressure Test — Forced Tiered Eviction
  Goal: Verify eviction actually triggers under realistic load
  Method: Run with reduced hicache-size (50 GB/rank) to force eviction,
          then scale concurrency from 1 to 512
  Measure: eviction rate, re-fetch latency, throughput degradation curve
  Key question: Does cascading eviction maintain throughput under pressure,
                or does it cliff like admission gating?

PHASE 5: Long Context Stress
  Goal: Test tiered caching with K2.6's full 128K context
  Workload: W5 (RAG long context) with 128K input, high concurrency
  Measure: KV cache per-request size, tier pressure, throughput vs 4K context
  Key question: Does Mooncake L3 enable high concurrency at 128K
                where baseline SGLang would OOM?
```

#### Termination
- **Success**: SGLang + Mooncake achieves ≥80% of vLLM throughput at c=512 (≥8,350 tok/s)
- **Partial success**: Mooncake enables ≥2x SGLang baseline at c=512 (≥6,800 tok/s)
- **Failure**: Mooncake adds <10% over HiCache-only at all concurrency levels
- **Hard stop**: 1 session (this is a benchmark, not iterative optimization)

#### Logging
- All configs saved to blueprint `configs/`
- Benchmark results in `results/experiments.jsonl`
- Tier utilization metrics in `results/tier_metrics/`
- Mooncake transfer engine logs in `results/mooncake_logs/`

### 4. Networking

- **Access**: EKS kubectl exec
- **Mooncake metadata server**: Pod-local or sidecar (single-node, no cross-node transfer)
- **RDMA**: EFA if testing cross-node (Phase 5 stretch goal)

### 5. Storage

- **Model weights**: `/mnt/nvme/models/Kimi-K2.6/` (~594GB)
- **KV cache L3**: `/mnt/nvme/kv-cache/` (NVMe, up to 25TB available after model)
- **Results**: Blueprint `results/` directory

---

## Research Questions

### RQ1: Does tiered KV caching close the vLLM gap at high concurrency?
SGLang saturates at c=128 (3,400 tok/s). Is this because KV cache fills GPU memory and the scheduler stops admitting requests? If so, Mooncake's cascading eviction should allow higher concurrency.

### RQ2: Does eviction actually trigger under realistic workloads?
K2.5 lessons showed LMCache and Dynamo KVBM **never triggered tiered offload** because vLLM's admission gating prevented memory overflow. SGLang's cascading eviction should behave differently — verify this empirically.

### RQ3: What is the optimal tier configuration for B300?
B300 has 268GB GPU VRAM, ~475GB DRAM/rank available for HiCache, and 3.8TB NVMe/rank. What is the right split? Does NVMe L3 add value when DRAM is already 475GB?

### RQ4: Can Mooncake enable high-concurrency long-context serving?
At 128K context, each request's KV cache is ~100MB+ (MLA compressed). At c=64, that's 6.4GB+ just for KV. Does Mooncake's NVMe tier prevent OOM at high concurrency + long context?

### RQ5: What is the transfer latency overhead?
Mooncake adds a transfer step when evicting/fetching KV blocks. What is the latency penalty per tier hop (GPU→DRAM, DRAM→NVMe, NVMe→DRAM→GPU)? Is it hidden by prefetching?

---

## Configuration Matrix

| Config | HiCache | L3 Backend | hicache-size | Expected Behavior |
|--------|---------|------------|--------------|-------------------|
| C1: SGLang baseline | No | N/A | N/A | 3,400 tok/s @ c=512 (from K2.6 data) |
| C2: HiCache-only | Yes | None | 200 GB/rank | K2.6 HiCache data: +58% single-stream, no load benefit |
| C3: HiCache + NVMe | Yes | local | 200 GB/rank | NVMe as overflow tier |
| C4: HiCache + Mooncake | Yes | mooncake | 200 GB/rank | Mooncake manages NVMe + prefetching |
| C5: HiCache-small + Mooncake | Yes | mooncake | 50 GB/rank | Force eviction to test cascading |
| C6: HiCache + Mooncake (128K) | Yes | mooncake | 200 GB/rank | Long context stress test |

**Baseline**: vLLM FLASHINFER_MLA = 10,437 tok/s @ c=512 (not re-run, use K2.6 data)

---

## Success Criteria

1. **Throughput**: SGLang + Mooncake achieves ≥8,350 tok/s at c=512 (80% of vLLM baseline)
2. **Tier validation**: Empirically observe KV cache eviction from GPU → DRAM → NVMe under load, with metrics showing tier utilization
3. **Eviction works**: Throughput degrades gracefully (not cliff) as concurrency increases beyond KV cache capacity
4. **Long context**: Enable c=64 at 128K context where baseline SGLang would OOM
5. **TTFT**: Mooncake config TTFT p99 ≤ 2x vLLM TTFT p99 at same concurrency (KV transfer latency tax is bounded)

## Non-Requirements

- **Not testing vLLM + Mooncake** — Mooncake integrates via SGLang HiCache, not vLLM. vLLM is the comparison baseline only.
- **Not testing cross-node RDMA** — Single-node only. Multi-node Mooncake is a follow-on if single-node shows promise.
- **Not testing LMCache or Dynamo KVBM** — Already benchmarked on K2.5 with known limitations (admission gating, no tiered offload). SGLang HiCache is the winner.
- **Not optimizing kernels** — That's the kernel-optimization-agent spec. This is scheduling/caching layer.
- **Not testing vision workloads** — Text-only, matching K2.6 baseline.

## Known Limitations

1. **Mooncake installation**: May require building from source. The `sglang-mooncake.sh` config from K2.5 used `--hicache-storage-backend mooncake` which assumes Mooncake is pre-installed in the SGLang image. Need to verify K2.6-era SGLang images include Mooncake.
2. **Mooncake metadata server**: Requires a separate process. For single-node, this is lightweight but adds operational complexity.
3. **HiCache-size interaction with B300 RAM**: At 200 GB/rank × 8 = 1.6TB for HiCache L2. The B300 has ~3.8TB free RAM after model loading. This leaves ~2.2TB for system + Mooncake metadata. Should be sufficient but monitor OOM.
4. **NVMe bandwidth**: B300 NVMe is ~7 GB/s per drive. With 8 drives, peak is ~56 GB/s. At 128K context with heavy eviction, NVMe could become the bottleneck.
5. **Single-node limitation**: Mooncake's biggest advantage (cross-node KV sharing) is not tested here. Single-node tests only measure the tiered storage benefit, not the disaggregated P/D benefit.

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Mooncake not in SGLang v0.5.10 image | Medium | High | Build custom image with Mooncake. Fallback: test NVMe-only L3 without Mooncake. |
| Eviction still doesn't trigger (B300 has too much VRAM) | Medium | Medium | Use C5 config (50 GB/rank) to force eviction. If it triggers there, the mechanism works. |
| SGLang's 3.1x gap is in the scheduler, not KV cache | Medium | High | Profile SGLang at c=128 and c=512 to confirm the bottleneck is memory, not compute/scheduling. If it's scheduling, this spec produces a negative result (valuable data). |
| Mooncake transfer latency adds >100ms TTFT | Low | Medium | Test prefetch policies (timeout, predictive). If latency is unavoidable, document the throughput-latency tradeoff curve. |
| HiCache + Mooncake OOMs on B300 (like 500GB/rank did) | Low | Low | Start with 100 GB/rank, increase gradually. Max 250 GB/rank (proven safe in K2.6). |

## Estimated Cost

| Phase | Duration | Instance Cost | Total |
|-------|----------|---------------|-------|
| Phases 1-5 | ~8 hours | ~$400 (B300 spot) | ~$400 |

This is a single-session benchmark, not a multi-session iterative experiment. Total cost ~$400.

## Relationship to Other Specs

| Spec | Relationship |
|------|-------------|
| `kimi-k2.6.md` | Provides all baselines (vLLM 10,437 tok/s, SGLang 3,400 tok/s, HiCache +58% single-stream) |
| `kimi-k2.5.md` | Prior Mooncake assessment, LMCache/Dynamo results, 37 lessons |
| `kernel-optimization-agent.md` | Complementary — this optimizes caching/scheduling, that optimizes compute kernels |
| `glm5-lmcache.md` | HiCache + NSA reference (2.86x throughput on GLM-5 with HiCache) |
| `glm5.md` | HiCache sizing reference (hicache-size must exceed device KV pool) |

---

> **Note**: Operational artifacts (lessons learned, experiment results, tier metrics)
> belong in the blueprint directory, not in this spec.
