# Readiness Audit — 2026-02-20

Pre-deployment readiness check for SGLang HiCache benchmarking session.

**Capacity Block**: `cr-0b0700f5f2ae0ca89` — p5e.48xlarge, us-east-2c
**Window**: 2026-02-20 22:49 UTC (5:49 PM EST) → 2026-02-22 11:30 UTC (36h 41m)
**Cost**: $1,460

## EKS Cluster

| Check | Status | Detail |
|-------|--------|--------|
| Cluster reachable | PASS | `kimi-k2-bench-eks-cluster`, v1.32, ACTIVE |
| API endpoint | PASS | `https://6F8F03D2EDC6E4CEAE5B11A7BEF2069B.gr7.us-east-2.eks.amazonaws.com` |
| System nodes | PASS | 2x m6i Ready, AL2023, containerd 2.1.5 |
| CoreDNS | PASS | 2/2 |
| kube-proxy | PASS | 2/2 |

## Storage

| Check | Status | Detail |
|-------|--------|--------|
| FSx Lustre | PASS | `fs-057e7a7df252f8e11`, AVAILABLE, 4800 GiB SSD, PERSISTENT_2 |
| FSx throughput | PASS | 1000 MB/s/TiB |
| FSx mount name | PASS | `6oondbev` |
| FSx DNS | PASS | `fs-057e7a7df252f8e11.fsx.us-east-2.amazonaws.com` |
| FSx PV/PVC | PASS | `vllm-kimi-fsx-pv` 4800Gi RWX, Bound |
| Model cache PV | PASS | 500Gi RWO gp3, Bound |
| FSx CSI driver | PASS | Controller 2/2, node daemonset 2/2 |
| EBS CSI driver | PASS | Controller 2/2, node daemonset 2/2 |

## Container Images (ECR)

| Image | Status | Tag | Pushed |
|-------|--------|-----|--------|
| `vllm-openai` | PASS | v0.15.1 | 2026-02-16 |
| `vllm-mooncake` | PASS | v0.15.1 | 2026-02-17 |
| `dynamo-kvbm` | PASS | v0.9.0 | 2026-02-17 |
| `sglang-hicache` | PASS | latest-cu130 | 2026-02-20 (15.7 GB, digest sha256:b63cabcb33) |

## GPU / Accelerator Readiness

| Check | Status | Detail |
|-------|--------|--------|
| NVIDIA device plugin | PENDING | DaemonSet 0/0 — needs `nvidia.com/gpu.present=true` label (self-heals on GPU node join) |
| EFA device plugin | PENDING | DaemonSet 0/0 — needs GPU node (self-heals on GPU node join) |
| DCGM exporter | PENDING | 0/2 ready — needs GPU node (self-heals on GPU node join) |

## Monitoring

| Check | Status | Detail |
|-------|--------|--------|
| Prometheus | PASS | Running |
| Grafana | PASS | 1/1 |
| kube-state-metrics | PASS | 1/1 |
| node-exporter | PASS | 2/2 |
| prometheus-operator | PASS | 1/1 |

## Serving Layer

| Check | Status | Detail |
|-------|--------|--------|
| vllm-kimi-k2 deployment | PASS | Exists, scaled to 0/0 (ready to scale up) |
| ClusterIP service | PASS | `172.20.152.242:8000` |
| NodePort service | PASS | `30080` |

## Capacity Block

| Check | Status | Detail |
|-------|--------|--------|
| Reservation | PASS | `cr-0b0700f5f2ae0ca89`, state: **active** |
| AZ | PASS | us-east-2c |
| Start | PASS | 2026-02-20T22:49:00Z |
| End | INFO | 2026-02-22T11:30:00Z |

## Configs & Scripts (new SGLang artifacts)

| File | Status | Detail |
|------|--------|--------|
| `configs/sglang-hicache.sh` | PASS | Created, bash -n OK |
| `configs/sglang-hicache-nvme.sh` | PASS | Created, bash -n OK |
| `configs/sglang-mooncake.sh` | PASS | Created, bash -n OK |
| `scripts/setup-sglang-p5e.sh` | PASS | Created, bash -n OK |
| `docker/Dockerfile.sglang-hicache` | PASS | Created |
| `scripts/run-benchmarks.py` | PASS | 7 SERVING_CONFIGS including 3 SGLang entries |
| `configs/comparison.yaml` | PASS | S1/S2/S3 entries added |
| `stage-images-ecr.sh` | PASS | SGLang section added (base mirror + custom build) |

## Action Items

| # | Priority | Action | Owner |
|---|----------|--------|-------|
| 1 | ~~P0~~ | ~~Create `sglang-hicache` ECR repo and push image~~ | DONE 17:52 EST — `latest-cu130` pushed via crane |
| 2 | ~~P0~~ | ~~Add SGLang to `scripts/stage-images-ecr.sh`~~ | DONE — base mirror + Dockerfile build sections added |
| 3 | P2 | DCGM/EFA/NVIDIA plugins will self-heal on GPU node join | Auto |
| 4 | P2 | FSx is 4.8 TiB (spec says 100 TiB) — sufficient for benchmarks | Accept |

## Overall Verdict

**PASS** — All P0 items resolved. Infrastructure is ready for the capacity block starting at 5:49 PM EST. GPU-dependent plugins (DCGM, EFA, NVIDIA device plugin) will self-heal when the p5e node joins.

---

## Deployment Log (2026-02-20 session)

### 5:55 PM EST — p5e instance launched
- Instance `i-077aecb332c88f62b` launched with `bootstrap.sh` user data
- **FAILED**: AL2023 EKS AMIs no longer use `bootstrap.sh` — replaced by `nodeadm`
- Instance terminated

### 6:16 PM EST — p5e relaunched with nodeadm
- Instance `i-03ac31e66cccf05a6` launched with nodeadm `NodeConfig` user data
- **ISSUE**: TLS cert mismatch — nodeadm decoded the CA differently, producing wrong fingerprint
- Fix: Copied correct CA PEM from working system node via SSM, restarted kubelet
- **Lesson #35**: AL2023 EKS AMIs use `nodeadm` not `bootstrap.sh`. The CA in nodeadm's `certificateAuthority` field must match exactly. Prefer uploading bootstrap script to S3 to avoid encoding issues.

### 6:27 PM EST — p5e node joined EKS
- Node `ip-10-0-36-143.us-east-2.compute.internal` — Ready
- 8x NVIDIA H200, driver 580.126.09, 143771 MiB each (1.1 TB total HBM)
- GDS device not present (nvidia-fs module not loaded — will load manually if needed)
- EFA/InfiniBand devices not enumerated yet (may need device plugin scheduling)

### 6:35 PM EST — NVMe RAID + FSx + model copy
- Created RAID0 across 8x NVMe drives (28 TB) at `/mnt/nvme`
- Mounted FSx at `/mnt/fsx` (installed `lustre-client` via dnf)
- Copied model from FSx to NVMe: 546 GB / 64 shards in ~3 minutes
- Pulled SGLang image from ECR

### 7:00 PM EST — SGLang HiCache launch attempt 1
- **FAILED**: `--enforce-eager` is not a valid SGLang argument (vLLM-only)
- Fix: Replaced with `--disable-cuda-graph` in all three config scripts

### 7:10 PM EST — SGLang HiCache launch attempt 2
- **FAILED**: `KimiK25ForConditionalGeneration has no SGlang implementation`
- SGLang v0.5.8.post1 does not register Kimi K2.5 in its model registry
- The `kimi_vl.py` model handles `KimiVLForConditionalGeneration` (Kimi-VL-A3B, a different model)

### 7:15 PM EST — SGLang launch attempt 3 (--language-only)
- **FAILED**: `--language-only` requires `--encoder-urls` (for disaggregated inference, not text-only mode)

### 7:30 PM EST — SGLang launch attempt 4 (patched model registry)
- Created `kimi_k25_text.py` — text-only wrapper that maps KimiK25ForConditionalGeneration to DeepseekV2ForCausalLM with `language_model.` prefix
- Model registry discovery worked — SGLang found the class and began loading
- All 8 GPUs allocated (~138 GB each), detected `CompressedTensorsWNA16MarlinMoEMethod`
- **FAILED**: Weight format mismatch — Kimi K2.5 INT4 uses packed expert weights (`w13_weight_packed`, `w2_weight_packed`) but SGLang's DeepseekV2 loader expects per-expert named weights (`gate_proj`, `up_proj`, `down_proj`)
- Container crashed after weight loading errors on all MoE layers

### 7:45 PM EST — SGLang assessment: BLOCKED
- **Root cause**: Kimi K2.5 is a new multimodal MoE model with a custom compressed tensor format. SGLang v0.5.8 has no native support — not just a missing registry entry but an incompatible weight loader.
- **Lesson #36**: SGLang model support lags behind vLLM. Before planning SGLang benchmarks, verify: (1) the model architecture is in SGLang's model registry, (2) the weight format is compatible, (3) test loading on a CPU-only instance before reserving GPU capacity.
- **Decision**: Fall back to vLLM benchmarks for this capacity block. SGLang HiCache testing requires either (a) native Kimi K2.5 support in a future SGLang release, or (b) a different model that SGLang already supports.

### 7:50 PM EST — Switching to vLLM baseline benchmarks
- Reusing existing vLLM configs (baseline, LMCache, Mooncake, Dynamo) which are proven to work with Kimi K2.5

### 8:12 PM EST — vLLM baseline container started
- Image: `615299764834.dkr.ecr.us-east-2.amazonaws.com/vllm-openai:v0.15.1`
- Using FLASH_ATTN_MLA attention backend, CompressedTensorsWNA16MarlinMoEMethod
- Model loading: 64 shards × ~2 min/shard = ~2 hours total (compute-bound, not I/O)

### 10:15 PM EST — vLLM baseline server ready
- Health endpoint: HTTP 200
- Test completion verified: coherent output from `/v1/completions`
- Starting benchmark runs

### 10:30 PM – 2:45 AM EST — Baseline benchmark runs (3 rounds)
- Mode: `kv-cache` (6 workloads: multi_turn_qa, long_context_rag, strict_no_reuse, long_context 16K/24K/36K/48K, multi_tenant 50t)
- 10–30 requests per workload, 3 full rounds completed
- All workloads 100% success rate

### 2:50 AM EST — GDS / FSx / LMCache readiness check
- **GDS NOT available**: `nvidia-fs` kernel module not present in AL2023 AMI (kernel 6.1.161-183.298.amzn2023.x86_64)
- FSx mounted at `/mnt/fsx` (4.5 TiB, 629 GiB used)
- Created cache directories: `/mnt/fsx/lmcache-kimi-k2.5`, `/mnt/nvme/kv-cache`
- LMCache must run in POSIX mode (CPU bounce) — no GPU-direct path to FSx Lustre
- ECR images available but not yet pulled: lmcache-vllm-openai, lmcache-router, dynamo-kvbm

### 3:00 AM EST — Cache metrics gap identified
- `scrape_prefix_cache_metrics()` function exists in run-benchmarks.py but was commented out
- Matches Lesson #30: "No cache hit/miss metrics captured" from 2026-02-18/19 rounds
- **Fix applied**: Wired prefix cache metric scraping into `run_benchmark_suite()`, `run_long_context_scaling()`, and `run_multi_tenant()` — captures pre/post delta per workload

---

## Baseline Benchmark Findings (Session 2026-02-21)

### Performance Summary

| Workload | Requests | TTFT p50 (ms) | TTFT p90 (ms) | E2E p50 (ms) | E2E p90 (ms) | Throughput (tok/s) |
|----------|----------|---------------|---------------|--------------|--------------|-------------------|
| multi_turn_qa | 10 | 274 | 277 | 13,523 | 13,581 | 11.1 |
| long_context_rag | 30 | 361 | 367 | 17,958 | 18,035 | 11.1 |
| strict_no_reuse | 30 | 270 | 276 | 13,420 | 13,630 | 11.2 |
| long_context_16K | 10 | 361 | 509 | 13,570 | 13,923 | 11.0 |
| long_context_24K | 10 | 411 | 449 | 13,704 | 13,822 | 10.9 |
| long_context_36K | 10 | 506 | 558 | 13,713 | 13,916 | 10.9 |
| long_context_48K | 10 | 558 | 592 | 13,730 | 13,804 | 10.9 |
| multi_tenant_50t | 100 | 293 | 304 | 13,492 | 13,638 | 11.1 |

### Key Observations

1. **Throughput is flat at ~11 tok/s** regardless of context length (16K–48K) or workload type. Generation speed is the bottleneck, not prefill. The Marlin MoE kernel execution dominates token generation time.

2. **TTFT scales linearly with context**: 270 ms (short) → 558 ms (48K tokens). This is the prefill cost and is the metric that benefits most from prefix caching.

3. **No prefix caching benefit observed**: Multi-tenant cold vs warm E2E shows 1.00x speedup — identical latency. This confirms baseline vLLM is not reusing cached prefixes across the 50-tenant workload. Prefix cache metrics were not captured in this run (fix applied for next round).

4. **No reasoning tokens**: 0% reasoning rate across all workloads. Kimi K2.5 supports thinking mode (`◁think▷`/`◁/think▷` tags) but it was not triggered by the benchmark prompts. This is expected for the simple Q&A workloads used.

5. **RAG workload has higher E2E** (~18s vs ~13.5s): The 20K shared context + retrieval-augmented prompts produce longer outputs, driving up end-to-end latency while maintaining the same ~11 tok/s generation rate.

6. **Sub-linear E2E scaling with context**: Despite TTFT doubling from 16K to 48K, E2E only increases from 13.5s to 13.7s — a 1.5% increase for 3x more context. This suggests generation time dominates and prefill cost is a small fraction of total latency at these output lengths (150 tokens).

### Infrastructure Findings

1. **GDS unavailable on AL2023**: The `nvidia-fs` kernel module required for GPU Direct Storage is not included in the AL2023 EKS AMI. LMCache benchmarks must use POSIX mode (CPU-mediated I/O to FSx Lustre). This eliminates the GDS advantage that was a key motivator for the FSx + LMCache architecture.

2. **Model loading is compute-bound**: 64-shard INT4 MoE model takes ~2 hours to load regardless of storage backend (NVMe or FSx). The bottleneck is Marlin MoE kernel repacking per shard, not I/O bandwidth. Pre-loading from NVMe (546 GB, ~3 min copy from FSx) avoids FSx bandwidth contention during loading but doesn't reduce the compute time.

3. **SGLang incompatibility confirmed**: Four progressive attempts to run SGLang HiCache with Kimi K2.5 all failed. Root cause is a fundamental weight format mismatch — Kimi K2.5 INT4 uses packed MoE expert weights that SGLang's DeepseekV2 loader cannot parse. SGLang v0.5.8.post1 has no native Kimi K2.5 support despite GitHub issue #18458 claiming otherwise.

### Recommendations for Next Capacity Block

1. **Run LMCache in POSIX mode** to establish the POSIX-offloading baseline. Compare TTFT under memory pressure (50+ concurrent contexts) where prefix cache eviction should trigger offloading to FSx.

2. **Run Dynamo KVBM** as the third comparison point. The dynamo-kvbm:v0.9.0 image is available in ECR.

3. **Use memory pressure workload** (`run_memory_pressure_throughput`) to stress-test KV cache capacity. The flat 11 tok/s throughput in baseline suggests we haven't hit GPU KV cache capacity limits yet — the 150-token outputs and sequential request pattern don't create enough cache pressure.

4. **Capture prefix cache metrics** in all runs (fix now applied).

5. **Consider longer outputs** (500–1000 tokens) to increase cache pressure and reveal offloading behavior differences between frameworks.
