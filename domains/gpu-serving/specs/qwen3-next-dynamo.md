# Qwen3-Next Disaggregated Serving with NVIDIA Dynamo

## Status: DRAFT (2026-02-25)

## Parent Specs

- [`qwen3-next.md`](./qwen3-next.md) — Model details, parallelism, benchmark methodology
- [`qwen3-next-g7e.md`](./qwen3-next-g7e.md) — Blackwell g7e hardware validation, baseline benchmarks

## Overview

Evaluate **NVIDIA Dynamo** disaggregated inference on g7e Blackwell instances for Qwen3-Next-80B-A3B. The hypothesis: disaggregated prefill/decode with NVMe-backed KV cache offloading can overcome the PCIe interconnect bottleneck observed in baseline benchmarks, enabling g7e to match or exceed H200 NVLink throughput at high concurrency while maintaining 4.6x cost advantage.

### Motivation

Baseline g7e.24xlarge benchmarks (2026-02-25) revealed:

| Finding | Impact |
|---------|--------|
| Per-GPU throughput matches H200 (455 vs 452 tok/s) | Cost efficiency proven |
| MTP degrades 2-41% on PCIe | Speculative decoding unusable without NVLink |
| 1000 concurrent 10K requests → 229s TTFT | KV cache capacity (2.76M tokens) is the bottleneck |
| PCIe all-reduce limits decode step throughput | NVLink-style optimizations don't apply |

Dynamo's disaggregated architecture addresses all four by:
1. Separating prefill (compute-bound) from decode (memory-bound) onto different GPU groups
2. Offloading cold KV cache to NVMe (7 TB available) or CPU RAM
3. Reducing cross-GPU communication by keeping decode workers independent
4. Routing requests intelligently to maximize KV cache locality

### Optimization Objective

```
Primary:   Maximize concurrent requests at acceptable latency
           (TTFT < 5s, TPOT < 50ms for 10K/1K workloads)
Secondary: Minimize $/M output tokens vs baseline g7e and H200
Metric:    Max sustainable concurrency at SLO × cost efficiency
```

---

## Components

### 1. Compute

- **Platform**: Bare EC2 (not EKS — faster iteration, proven in baseline benchmarks)
- **Prefill Workers**: g7e.24xlarge (4x RTX PRO 6000 Blackwell, 96 GB GDDR7 each)
- **Decode Workers**: g7e.24xlarge (same instance type, separate role)
- **Router**: g7e.12xlarge or c7i.4xlarge (CPU-only, Dynamo planner + KV router)
- **Region**: us-west-2 (g7e capacity confirmed in 2c/2d)
- **Minimum viable**: 2x g7e.24xlarge (1 prefill + 1 decode) + 1 router

#### Scaling Configurations

| Config | Prefill | Decode | Total GPUs | Est. $/hr | Purpose |
|--------|---------|--------|-----------|-----------|---------|
| `d1` | 1x g7e.24xlarge | 1x g7e.24xlarge | 8 | $16.54 | Disaggregation baseline |
| `d2` | 1x g7e.24xlarge | 2x g7e.24xlarge | 12 | $24.81 | Decode-heavy (high concurrency) |
| `d4` | 2x g7e.24xlarge | 2x g7e.24xlarge | 16 | $33.08 | Balanced (match p5en GPU count) |

### 2. Model

Same as parent spec:
- **Model**: Qwen3-Next-80B-A3B-Instruct (MoE, 80B total / 3B active, FP8)
- **Architecture**: Hybrid attention + Gated DeltaNet (mamba-like linear attention)
- **Serving**: vLLM 0.15.0+ with Dynamo integration, or Dynamo native runtime
- **Quantization**: FP8 (confirmed working on Blackwell sm_120)
- **TP=4 per node** (confirmed optimal for PCIe)

### 3. NVIDIA Dynamo

| Component | Role |
|-----------|------|
| **Dynamo Planner** | Request routing, prefill/decode scheduling |
| **Dynamo Worker (prefill)** | Runs prefill phase, produces KV cache |
| **Dynamo Worker (decode)** | Runs decode phase, consumes KV cache |
| **KV Cache Router** | Transfers KV state between prefill and decode workers |
| **NIXL** | Low-level KV cache transfer library (GPU→GPU, GPU→NVMe, GPU→CPU) |

#### Dynamo Architecture

```
Client Request
      │
      ▼
┌─────────────┐
│   Dynamo    │  Request scheduling + KV routing
│   Planner   │  Decides: which prefill worker, which decode worker
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌──────┐ ┌──────┐
│Prefill│ │Prefill│  g7e.24xlarge #1, #2
│Worker │ │Worker │  TP=4, FP8, compute-optimized
│(4 GPU)│ │(4 GPU)│  Processes input tokens → produces KV cache
└──┬───┘ └──┬───┘
   │         │
   │  NIXL KV Transfer (PCIe → NVMe → PCIe, or GPU Direct)
   │         │
   ▼         ▼
┌──────┐ ┌──────┐
│Decode │ │Decode │  g7e.24xlarge #3, #4
│Worker │ │Worker │  TP=4, FP8, memory-bandwidth-optimized
│(4 GPU)│ │(4 GPU)│  Generates output tokens using transferred KV cache
└──────┘ └──────┘
   │         │
   ▼         ▼
  NVMe (7 TB per node)
  Cold KV cache overflow tier
```

### 4. Storage Tiers

#### GDS Findings (Validated 2026-02-25)

- **GDS installed**: `nvidia-gds 13.1.1`, `nvidia_fs` kernel module v2.25.7 loaded, 16 `/dev/nvidia-fs*` channels
- **NVMe P2PDMA: Unsupported** — Amazon EC2 NVMe controllers (Amazon Elastic Block Store) do not support PCIe peer-to-peer DMA. GDS falls back to **compat mode** (CPU bounce buffer). Still faster than standard POSIX I/O due to pinned GPU memory and async DMA, but not true zero-copy.
- **FSx GDS via EFA: Viable** — g7e.24xlarge and g7e.48xlarge both support EFA (confirmed via `describe-instance-types`). FSx Lustre PERSISTENT_2 with EFA enables GPUDirect Storage over RDMA, bypassing both CPU and NVMe entirely for a true GPU↔network zero-copy path.

| Tier | Medium | Bandwidth | Capacity | Latency | I/O Mode | Purpose |
|------|--------|-----------|----------|---------|----------|---------|
| 0 (Hot) | GPU VRAM (GDDR7) | 1.5 TB/s | 63 GiB KV/node | <1 us | Native | Active decode sequences |
| 1 (Warm) | CPU RAM (DDR5) | ~64 GB/s (PCIe Gen5) | ~300 GiB usable/node | ~1 us | CPU offload | Recently evicted KV |
| 2 (Cold) | NVMe RAID0 | ~5.3 GiB/s read (validated) | 7 TB/node | ~735 us | Standard POSIX (GDS compat useless) | Per-node overflow |
| 3 (Shared) | FSx Lustre via GDS+EFA | ~4.8 GB/s (RDMA, projected) | 4.8 TiB+ | ~100 us (projected) | GDS true P2P (RDMA) | Cross-node KV, shared prefix cache |

**NVMe GDS compat mode** (validated 2026-02-25 via gdsio on g7e.24xlarge):

| Operation | GDS Compat (GPUD) | CPU-Only | Delta |
|-----------|-------------------|----------|-------|
| Read | 5.08 GiB/s, 769 μs | 5.31 GiB/s, 735 μs | -4.3% (GDS slower) |
| Write | 5.91 GiB/s, 661 μs | 5.87 GiB/s, 666 μs | +0.7% (negligible) |

**Finding**: GDS compat mode provides **no benefit** over standard CPU-mediated I/O on EC2 NVMe. For reads, it's actually 4% slower due to extra indirection overhead without P2PDMA. For NVMe-tier KV offloading, use standard POSIX I/O (or vLLM's native `--cpu-offload-gb` path) rather than GDS.

**FSx GDS via EFA**: The only true zero-copy GPU storage path on g7e. FSx PERSISTENT_2 with EFA support enables RDMA-based GDS where data flows directly from FSx storage servers to GPU VRAM over the EFA network fabric, bypassing CPU entirely. This makes Tier 3 potentially faster than Tier 2 for large sequential reads despite being network-attached.

**KV cache capacity projection** (per decode node):

| Tier | Tokens (FP8 KV) | Concurrent @ 10K | Notes |
|------|-----------------|-----------------|-------|
| VRAM only | 2.76M | ~276 | Current baseline |
| VRAM + CPU | ~8M | ~800 | `--cpu-offload-gb` |
| VRAM + CPU + NVMe (GDS compat) | ~50M+ | ~5000 | Per-node, no network |
| VRAM + CPU + FSx (GDS+EFA) | Virtually unlimited | ~10000+ | Shared across nodes, true P2P |

With multi-tier offloading, the g7e's KV cache bottleneck at high concurrency is eliminated. The key question is whether the latency penalty of lower tiers (especially during decode, where KV blocks must be fetched back to VRAM) is acceptable.

### 5. Networking

- **Inter-node**: VPC placement group (cluster) for lowest latency
- **EFA**: Enabled on all g7e instances (confirmed supported on g7e.24xlarge/g7e.48xlarge). Required for FSx GDS via RDMA.
- **KV transfer**: TCP/RDMA via NIXL (Dynamo's native transfer library). EFA enables RDMA path for inter-node KV transfers.
- **FSx GDS path**: GPU VRAM ↔ EFA NIC ↔ FSx Lustre storage servers (bypasses CPU entirely)
- **Client access**: ALB or NLB in front of Dynamo planner
- **Security**: VPC-internal only, security group allows EFA traffic (all protocols) within placement group + FSx security group rules

---

## Benchmark Phases

### Phase D0: Dynamo Smoke Test

| What | Details |
|------|---------|
| Config | `d1` (1 prefill + 1 decode, 8 GPUs total) |
| Workload | QPS 0.5, 1024/512 random |
| Runs | 3 |
| Purpose | Validate Dynamo worker startup, NIXL KV transfer, end-to-end inference |
| Success | Model serves inference, KV transfers complete, no errors |

### Phase D1: Disaggregated vs Monolithic Comparison

| What | Details |
|------|---------|
| Configs | `d1` (disaggregated) vs baseline g7e.24xlarge (monolithic, same 8 GPUs) |
| Workload | QPS sweep 1, 2, 4, 8 at 1024/512 |
| Runs | 3 each |
| Purpose | Quantify disaggregation overhead/benefit at equal GPU count |
| Success | Disaggregated throughput within 10% of monolithic at low QPS; better at high QPS |

### Phase D2a: KV Cache Offloading — NVMe (Standard POSIX I/O)

| What | Details |
|------|---------|
| Config | Single g7e.24xlarge (monolithic, 4 GPUs) — isolate offloading effect |
| Tiers | T0 (VRAM-only) → T0+T1 (VRAM+CPU) → T0+T1+T2 (VRAM+CPU+NVMe) |
| Workload | 100 concurrent, 10K/1K (C4-like but scaled to 4 GPUs) |
| Runs | 3 per tier |
| Metrics | TTFT, TPOT, output tok/s, NVMe I/O bandwidth, CPU utilization delta |
| Purpose | Quantify NVMe offloading benefit vs pure VRAM and CPU offloading |
| Success | T0+T1+T2 achieves <10s TTFT at 100 concurrent 10K requests |

**Note**: GDS compat mode validated as useless on EC2 NVMe (no P2PDMA). Use standard POSIX I/O or vLLM's native `--cpu-offload-gb` + NVMe swap. Raw NVMe RAID0 bandwidth: 5.3 GiB/s read, 5.9 GiB/s write (validated via gdsio).

### Phase D2b: KV Cache Offloading — FSx (GDS + EFA)

| What | Details |
|------|---------|
| Config | Single g7e.24xlarge + FSx PERSISTENT_2 (4.8 TiB, 1000 MB/s/TiB, EFA) |
| Tiers | T0+T3 (VRAM + FSx GDS) — skip CPU/NVMe to isolate FSx GDS path |
| Workload | 100 concurrent, 10K/1K |
| Runs | 3 |
| Metrics | TTFT, TPOT, FSx GDS bandwidth (via `nvidia-smi` GDS counters), EFA utilization |
| Purpose | Validate true zero-copy GPU↔FSx path; measure if RDMA GDS outperforms NVMe compat |
| Success | FSx GDS bandwidth >2 GB/s sustained; TTFT comparable to NVMe tier |
| Prerequisite | FSx PERSISTENT_2 provisioned with EFA, GDS driver loaded, Lustre client mounted |

**Pre-test**: Run `gdsio` against FSx mount point to establish raw GDS+EFA throughput baseline.

### Phase D2c: Multi-Tier KV Cache (All Tiers)

| What | Details |
|------|---------|
| Config | Single g7e.24xlarge + FSx PERSISTENT_2 |
| Tiers | T0+T1+T2+T3 (VRAM → CPU → NVMe GDS → FSx GDS) |
| Workload | 200, 500 concurrent at 10K/1K |
| Runs | 3 per concurrency level |
| Metrics | TTFT, TPOT, per-tier hit rate, eviction rate, tier promotion/demotion latency |
| Purpose | Test full 4-tier hierarchy under increasing pressure; find where each tier activates |
| Success | 500 concurrent: TTFT < 15s, TPOT < 50ms; clear tier waterfall visible in metrics |

**Note**: This phase depends on whether Dynamo/vLLM supports multi-tier configuration. If only 2-tier is supported (VRAM + one offload), test the best single offload tier from D2a/D2b.

### Phase D3: High Concurrency Scaling (Disaggregated)

| What | Details |
|------|---------|
| Config | `d2` (1 prefill + 2 decode, 12 GPUs) + FSx PERSISTENT_2 |
| Offloading | Best tier combination from D2a-D2c |
| Workload | 500, 1000 concurrent at 10K/1K |
| Runs | 3 each |
| Purpose | Match customer's 1000-concurrent benchmark with disaggregated serving + KV offloading |
| Success | 1000 concurrent: TTFT < 5s, throughput > 3000 tok/s, E2E < 30s |

### Phase D4: Cost-Efficiency Comparison

| What | Details |
|------|---------|
| Configs | `d2` and `d4` with optimal offloading tier |
| Workload | Same as customer benchmark (1000 concurrent, 10K/1K) |
| Comparison | g7e Dynamo vs H200 monolithic (customer baseline) |
| Include | FSx cost amortized over benchmark hours |
| Success | $/M output tokens < $2.00 (vs customer's ~$4.88 on H200) |

---

## Dynamo-Specific Configuration

### NIXL KV Transfer Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| Transfer protocol | EFA RDMA (fallback: TCP; shared memory for same-node) | EFA supported on g7e, enables RDMA for inter-node + FSx GDS |
| Chunk size | 256 KB | Balance between transfer overhead and granularity |
| NVMe cache path | `/mnt/nvme/kv-cache/` | 7 TB RAID0, GDS compat mode (~6-12 GB/s) |
| FSx cache path | `/mnt/fsx/kv-cache/` | 4.8 TiB PERSISTENT_2, GDS+EFA true P2P (~4.8 GB/s) |
| CPU cache size | 128 GiB | Half of system RAM, leaves headroom for OS |
| Eviction policy | LRU with prefix-aware pinning | Keep shared prefixes hot; evict to NVMe before FSx |
| Tier priority | VRAM → CPU → NVMe (GDS compat) → FSx (GDS+EFA) | Latency-ordered; NVMe for per-node overflow, FSx for cross-node sharing |

### Prefill Worker Config

```
--tensor-parallel-size 4
--quantization fp8
--gpu-memory-utilization 0.95   # Higher util OK — no long-lived KV
--max-model-len 32768
--kv-cache-dtype fp8
--role prefill                   # Dynamo role assignment
```

Prefill workers don't hold KV cache long-term — they compute KV and transfer it to decode workers. Higher `gpu-memory-utilization` is safe because KV is ephemeral.

### Decode Worker Config

```
--tensor-parallel-size 4
--quantization fp8
--gpu-memory-utilization 0.85   # Lower util — reserve VRAM for KV cache churn
--max-model-len 32768
--kv-cache-dtype fp8
--enable-prefix-caching
--role decode                    # Dynamo role assignment
--kv-offload-tiers gpu,cpu,nvme,fsx  # Multi-tier offloading (test each tier independently first)
--nvme-cache-path /mnt/nvme/kv-cache/
--fsx-cache-path /mnt/fsx/kv-cache/
--cpu-kv-cache-gb 128
--gds-enabled true               # Enable GDS for NVMe (compat mode) and FSx (EFA P2P)
```

**Tier testing order** (Phase D2a → D2b → D2c):
1. `--kv-offload-tiers gpu,cpu,nvme` — NVMe GDS compat only
2. `--kv-offload-tiers gpu,fsx` — FSx GDS+EFA only (skip CPU/NVMe to isolate)
3. `--kv-offload-tiers gpu,cpu,nvme,fsx` — Full multi-tier

### Dynamo Planner Config

```
--prefill-workers <prefill_ips>
--decode-workers <decode_ips>
--scheduling-policy locality-aware   # Route to decode worker with most cached KV
--max-prefill-batch 32               # Batched prefill for compute efficiency
--kv-transfer-timeout 5000           # 5s max for KV transfer
```

---

## Mamba Hybrid Considerations

Qwen3-Next's hybrid architecture (Gated DeltaNet + Gated Attention) creates unique challenges for disaggregated serving:

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| Mamba state is not pure KV cache | NIXL may not handle DeltaNet state transfer | Verify Dynamo supports hybrid state serialization |
| Mamba 'align' cache mode | Block size padding (544 tokens) affects transfer granularity | Align NIXL chunk size to mamba page size |
| Prefix caching experimental for mamba | May interact poorly with cross-node KV migration | Test with and without prefix caching |

**Risk**: If Dynamo's NIXL doesn't support hybrid mamba+attention state transfer, disaggregated serving may not work for this model. Fallback: use Dynamo for pure request routing without disaggregation (still benefits from multi-node scaling and KV offloading on individual nodes).

---

## Infrastructure

### Per-Node Setup (Bare EC2)

Reuse the proven bare EC2 approach from g7e baseline benchmarks, with EFA and GDS additions:

1. Launch g7e.24xlarge in us-west-2 (placement group: cluster, **EFA enabled**)
2. NVMe RAID0 setup (4x 1.75 TB → 7 TB at `/mnt/nvme`)
3. Install EFA driver: `curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz && tar xzf aws-efa-installer-latest.tar.gz && cd aws-efa-installer && sudo ./efa_installer.sh -y`
4. Verify GDS: `nvidia_fs` module loaded, `/dev/nvidia-fs*` devices present
5. Mount FSx Lustre at `/mnt/fsx` with GDS-compatible mount options (`flock`)
6. Create KV cache directories: `mkdir -p /mnt/nvme/kv-cache /mnt/fsx/kv-cache`
7. Model stage from S3 to NVMe: `aws s3 sync s3://qwen3-next-bench-models.../qwen3-next-fp8/ /mnt/nvme/models/qwen3-next-fp8/`
8. Start containerd: `sudo systemctl start containerd`
9. Run gdsio pre-tests on NVMe and FSx to validate bandwidth
10. Pull Dynamo container image (TBD — NVIDIA NGC or custom build)
11. Launch Dynamo worker with role assignment

### Container Image

| Option | Pros | Cons |
|--------|------|------|
| `nvcr.io/nvidia/dynamo:latest` | Official, optimized | May not have Blackwell sm_120 support yet |
| Custom build: Dynamo + vLLM 0.15.0 | Proven vLLM version, add Dynamo layer | Build complexity |
| vLLM 0.16+ with native disaggregation | vLLM's own disaggregated prefill/decode | May not have Dynamo's KV routing sophistication |

**Recommendation**: Start with vLLM 0.16+ native disaggregated serving (if available) as it has the lowest integration risk, then evaluate full Dynamo if vLLM's implementation is insufficient.

### FSx Lustre (GDS-Optimized — Multi-Node KV + Shared Prefix Cache)

Configuration derived from kimi-k2.5 blueprint (validated for GDS):

| Setting | Value | Rationale |
|---------|-------|-----------|
| Deployment type | `PERSISTENT_2` | Required for GDS + high throughput |
| Per-unit throughput | `1000` MB/s/TiB | Maximum throughput tier |
| Storage capacity | `4800` GiB | Minimum for PERSISTENT_2 at 1000 MB/s |
| File system version | `2.15` | Required for GDS support |
| EFA enabled | `true` | Enables RDMA-based GDS (true zero-copy GPU↔FSx) |
| Data compression | `LZ4` | Reduces network transfer for KV blocks |
| Metadata config | `AUTOMATIC` | Optimized for mixed workloads |
| Mount path | `/mnt/fsx` | |
| KV cache subpath | `/mnt/fsx/kv-cache/` | Dedicated directory for KV offload |

**Aggregate bandwidth**: 4.8 GB/s (4800 GiB × 1000 MB/s/TiB). With GDS+EFA, this bandwidth flows directly to GPU VRAM via RDMA.

**Cost**: ~$720/month for 4.8 TiB PERSISTENT_2 at 1000 MB/s/TiB. Acceptable for benchmark duration; can be destroyed after testing.

FSx serves dual purposes:
1. **Cross-node KV migration**: When decode load shifts between nodes, KV state migrates via shared filesystem rather than point-to-point
2. **Shared prefix cache**: Common prefixes (e.g., system prompts, shared documents) are stored once on FSx and loaded by any decode worker via GDS, avoiding redundant prefill computation

---

## Cost Analysis (Projected)

### Compute

| Config | Instances | Compute $/hr | Projected tok/s | Compute $/M tokens |
|--------|-----------|------|----------------|-----------|
| g7e baseline (monolithic) | 1x g7e.24xlarge | $8.27 | 2,172 | $1.06 |
| `d1` (disaggregated) | 2x g7e.24xlarge | $16.54 | ~3,000 (est.) | $1.53 |
| `d2` (decode-heavy) | 3x g7e.24xlarge | $24.81 | ~4,500 (est.) | $1.53 |
| `d4` (balanced) | 4x g7e.24xlarge | $33.08 | ~6,000 (est.) | $1.53 |
| Customer H200 | 1x p5en.48xlarge | $63.30 | 3,612 | $4.88 |

### Storage (FSx Lustre PERSISTENT_2)

| Component | Spec | Monthly Cost | Hourly Cost |
|-----------|------|-------------|-------------|
| FSx 4.8 TiB @ 1000 MB/s/TiB | PERSISTENT_2, EFA, Lustre 2.15 | ~$720 | ~$1.00 |

### Total Cost (Compute + Storage)

| Config | Total $/hr | Projected tok/s | Total $/M tokens |
|--------|-----------|----------------|-----------------|
| `d1` + FSx | $17.54 | ~3,000 (est.) | $1.62 |
| `d2` + FSx | $25.81 | ~4,500 (est.) | $1.59 |
| `d4` + FSx | $34.08 | ~6,000 (est.) | $1.58 |
| Customer H200 (no FSx) | $63.30 | 3,612 | $4.88 |

**Break-even**: Even with FSx costs included, `d4` at $34.08/hr (54% of p5en cost) is 3x more cost-efficient per output token. The real win is **concurrency**: `d2` with NVMe+FSx offloading should handle 1000+ concurrent requests where the monolithic g7e.24xlarge cannot.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dynamo doesn't support Qwen3-Next mamba hybrid | High | Blocks disaggregation entirely | Fall back to vLLM native disaggregated prefill/decode |
| NIXL KV transfer latency on PCIe is too high | Medium | Disaggregation overhead exceeds benefit | Measure transfer time in D0; abort if >500ms for 10K context |
| Dynamo not yet optimized for Blackwell sm_120 | Medium | Suboptimal kernel performance | Use vLLM backend within Dynamo workers |
| g7e capacity insufficient for multi-node | Medium | Can't provision enough instances | Use spot instances; have fallback to single-node with NVMe offload only |
| NVMe GDS compat useless (validated) | Confirmed | GDS compat provides 0% benefit on EC2 NVMe | Use standard POSIX I/O for NVMe tier; FSx GDS+EFA is the true zero-copy path |
| FSx GDS+EFA setup complexity | Medium | Driver/mount/RDMA misconfiguration delays testing | Follow kimi-k2.5 blueprint pattern; pre-validate gdsio on FSx before benchmarks |
| Dynamo/vLLM doesn't support multi-tier offloading | Medium | Can only test 2 tiers (VRAM + one offload) | Test each tier independently; use best single offload tier for D3/D4 |
| FSx PERSISTENT_2 provisioning delay | Low | 15-30 min provision time | Create FSx before starting benchmark phases |
| Inter-node network bandwidth insufficient | Low | KV transfer bottleneck | Placement group + EFA; compress KV for transfer |

---

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| D0 smoke test passes | Inference works end-to-end | Correct model output, no errors |
| D1 disaggregated overhead | < 15% throughput loss vs monolithic at QPS 2 | tok/s comparison |
| D2a NVMe offload TTFT | < 10s at 100 concurrent 10K requests | C4-style benchmark |
| D2a NVMe I/O utilization | NVMe bandwidth >2 GiB/s during KV offload | iostat + benchmark metrics |
| D2b FSx GDS+EFA bandwidth | > 2 GB/s sustained KV read | gdsio on FSx mount + benchmark metrics |
| D2b FSx offload TTFT | < 15s at 100 concurrent 10K requests | C4-style benchmark |
| D2c Multi-tier waterfall | Clear tier activation visible in metrics | Per-tier hit rate, eviction stats |
| D3 high concurrency | 1000 concurrent: TTFT < 5s, E2E < 30s | Match customer SLO |
| D4 cost efficiency | < $2.00/M output tokens at 1000 concurrent | Cost calculation (include FSx) |
| KV transfer latency | < 500ms for 10K context KV state | NIXL metrics |

---

## Non-Requirements

- Production autoscaling (benchmark only)
- Multi-region deployment
- Authentication / rate limiting
- Long-running stability tests (> 2 hours)
- NVMe P2P DMA (unsupported on EC2 NVMe controllers; GDS compat mode used instead)

---

## Prerequisites

1. Baseline g7e benchmarks complete (done — 2026-02-25)
2. GDS installed and validated on g7e (done — nvidia_fs v2.25.7, NVMe compat mode confirmed)
3. Dynamo container image available for Blackwell (verify on NGC)
4. g7e.24xlarge capacity in us-west-2 for 2-4 instances (EFA-enabled launch)
5. Placement group created in target AZ (cluster type)
6. VPC with placement group subnet, EFA security group rules (all protocols within SG)
7. FSx Lustre PERSISTENT_2 provisioned (4.8 TiB, 1000 MB/s/TiB, EFA, Lustre 2.15)
8. gdsio baseline tests on both NVMe RAID0 and FSx mount point
9. EFA driver installed on all instances (amazon-efa-installer)

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes)
> belong in the blueprint directory at `blueprints/qwen3-next-dynamo/`.
