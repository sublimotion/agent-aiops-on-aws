# Lessons Learned - qwen3-next-g7e Blueprint

## Deployment Challenges

### 1. IAM Role Name Length Limit
**Issue**: Default project name "qwen3-next-g7e-bench" caused IAM role name prefix to exceed 38 character limit.
**Resolution**: Shortened project name to "qwen3-g7e" in terraform.tfvars.
**Lesson**: Keep project names short to avoid AWS naming constraints.

### 2. Kubernetes Provider Dependencies
**Issue**: Kubernetes and Helm providers fail during initial terraform apply because they depend on EKS cluster outputs that don't exist yet.
**Workaround**: Apply infrastructure in stages or handle provider errors gracefully.
**Lesson**: Consider splitting Kubernetes resources into a separate Terraform configuration.

### 3. g7e Capacity Crisis — Pivot to Bare EC2
**Issue**: g7e.48xlarge and g7e.24xlarge completely unavailable in us-east-2 (all AZs). Capacity Blocks not supported for g7e instance types. EC2 dry-run returns "would succeed" but actual launch fails (dry-run validates permissions/quotas only, not physical capacity).
**Resolution**: Shotgunned `aws ec2 run-instances` across us-east-1, us-east-2, us-west-2 — found capacity in us-west-2c and us-west-2d. Launched bare EC2 g7e.24xlarge directly (bypassing EKS).
**Lesson**: For scarce GPU instances, try multiple regions simultaneously. Bare EC2 is faster for benchmarks than EKS when capacity is unpredictable. EC2 dry-run is useless for capacity validation.

### 4. `bootstrap_self_managed_addons` State Mismatch
**Issue**: EKS cluster imported with `bootstrap_self_managed_addons=false` but Terraform wanted `true`, forcing cluster destruction.
**Resolution**: Edited Terraform state JSON directly with Python to change the attribute value.
**Lesson**: When importing existing resources, verify all attributes match before planning. Use `terraform state pull/push` for surgical state fixes.

### 5. nerdctl vs Docker on EKS AMI
**Issue**: EKS-optimized AL2023 AMI uses nerdctl/containerd, not Docker. containerd service not running by default on bare EC2. nerdctl doesn't support `-d` with `--rm` together. GPU `--gpus` flag uses count syntax, not device IDs.
**Resolution**: `sudo systemctl start containerd`, use `--gpus 4` instead of `--gpus '"device=0,1,2,3"'`, remove `--rm` when using `-d`.
**Lesson**: Test container runtime quirks before deploying workloads.

### 6. Terraform State Lock from Background Tasks
**Issue**: Multiple background terraform processes held state locks, blocking new operations.
**Resolution**: `pkill -9 terraform` and remove lock file.
**Lesson**: Don't run terraform in parallel background tasks. Monitor with `ps aux | grep terraform`.

## Benchmark Findings

### 7. Blackwell (sm_120) Works with vLLM 0.15.0
**Finding**: CUDA 13.0 (cu130) image works on Blackwell compute capability 12.0. No custom build needed.
**Caveats**: SymmMemCommunicator not supported, custom allreduce disabled (PCIe-only >2 GPUs). Default TRITON Fp8 MoE backend used (no tuned config for Blackwell).

### 8. MTP Speculative Decoding Hurts on PCIe GPUs
**Finding**: MTP degrades throughput by 2-41% across QPS levels on g7e.24xlarge (PCIe interconnect).
**Root cause**: Speculative head computation + inter-GPU verification overhead exceeds speculative decoding gains without NVLink fast interconnect.
**Additional issue**: MTP incompatible with mamba cache mode 'align' in v0.15.0 — requires `--no-enable-prefix-caching`, further reducing performance.
**Lesson**: Only use MTP on NVLink-connected GPUs (H200, A100). Skip on PCIe platforms.

### 9. Per-GPU Throughput Parity Between Blackwell and H200
**Finding**: g7e Blackwell delivers 455.2 tok/s per GPU vs customer H200 at 451.6 tok/s — nearly identical per-GPU output throughput despite GDDR7 vs HBM3 memory.
**Implication**: Cost per token is 4.6x lower on g7e ($1.06/M vs $4.88/M output tokens at QPS 8).
**Caveat**: 4-GPU PCIe setup can't match 8-GPU NVLink at extreme concurrency (1000 concurrent 10K-token requests).

### 10. Qwen3-Next Is a Hybrid Mamba Model
**Finding**: Qwen3-Next uses `Qwen3NextForCausalLM` architecture with mamba cache mode. This is a hybrid attention+mamba model, not pure transformer.
**Impact**: Prefix caching triggers experimental mamba 'align' mode. MTP conflicts with mamba cache. Attention block size padded to 544 tokens for mamba alignment.
**Lesson**: Check model architecture before assuming transformer-only behavior. Mamba hybrids have different caching and speculative decoding constraints.

### 11. g7e.24xlarge Sweet Spot: QPS 4-8
**Finding**: At QPS 4-8 with 1024/512 tokens, the g7e.24xlarge delivers 1,500-2,200 tok/s with TPOT under 27ms. QPS 16 pushes to 3,389 tok/s but TPOT rises to 43ms.
**Context scaling**: 32K context (24576/512) works at QPS 2 with 717 tok/s, TTFT 745ms.

## Infrastructure Observations

### NVMe RAID0 on g7e.24xlarge
- 4x 1.75 TB NVMe drives = 7 TB RAID0 at /mnt/nvme
- Model load from NVMe: <2 min for 77 GB FP8 model
- Generic `lsblk` + `grep nvme` detection works across instance types

### GDDR7 Thermals
- Idle: 28-29C, ~82W per GPU
- Under load: stays cool (good thermal headroom)
- 96 GB VRAM per GPU comfortably fits Qwen3-Next FP8 with 63 GB KV cache remaining

### KV Cache Capacity
- 2,758,080 tokens total (FP8 KV cache)
- Max 302.73x concurrent requests at 32K context
- Sufficient for moderate concurrency; not enough for 1000 concurrent 10K-token requests

## Future Work: Multi-Tier KV Cache Offloading via Dynamo + GDS

### Problem

The g7e.24xlarge has 384 GB total VRAM across 4 GPUs, yielding 63 GiB usable KV cache (2.76M tokens at FP8). At 1000 concurrent requests × 10K input tokens (10M total), the cache is 3.6x oversubscribed — the C4 benchmark showed 229s mean TTFT from queuing. CPU offloading (`--cpu-offload-gb`) helps but the g7e has only ~384 GB host DRAM, limiting the second tier.

The g7e's strength is its 7 TB NVMe RAID0, which is currently only used for model weight storage.

### Proposed Tiered Architecture

vLLM Dynamo's NIXL transfer layer supports multiple KV cache backends:

| Tier | Medium | Capacity (g7e.24xlarge) | Bandwidth (per GPU) | Latency | Backend |
|------|--------|------------------------|--------------------|---------|---------|
| 0 | GPU VRAM (GDDR7) | 63 GiB | ~1.5 TB/s | ~ns | Native |
| 1 | CPU DRAM | ~300 GiB usable | ~64 GB/s (PCIe Gen5) | ~1 μs | CPU offload |
| 2 | NVMe via GDS | 7 TB | ~6-12 GB/s (PCIe Gen5) | ~10-20 μs | GPUDirect Storage |
| 3 | FSx Lustre via POSIX | 4.8 TiB+ | ~4.8 GB/s (network) | ~100+ μs | POSIX filesystem |

**GPUDirect Storage (GDS)** enables direct DMA between GPU memory and NVMe, bypassing CPU entirely. This eliminates the CPU copy bottleneck and makes NVMe a viable KV cache tier.

### Why This Matters for g7e

The g7e cost-efficiency story is strong (4.6x cheaper per output token vs p5en) but breaks down at extreme concurrency due to limited VRAM. Multi-tier offloading closes this gap:

- **Tier 0+1 (VRAM + DRAM)**: Expands effective KV capacity from 2.76M → ~8M tokens. Handles ~800 concurrent × 10K requests.
- **Tier 0+1+2 (+ NVMe via GDS)**: Expands to ~300M+ tokens. Handles virtually any concurrency level. The 10-20μs NVMe latency adds ~1-3ms per decode step (fetching KV blocks back to VRAM), which is far less than the hundreds of seconds of queue wait time under 3.6x oversubscription.
- **Tier 3 (FSx POSIX)**: Useful for batch/async workloads with higher latency tolerance, or as a persistent prefix cache store shared across instances.

### For prefill-heavy workloads (customer's 10K input):

The KV cache write during prefill is a one-time cost per request. Once prefill completes and KV is written to NVMe via GDS, subsequent decode tokens read from hot cache in VRAM (tier 0). The NVMe tier primarily absorbs overflow from requests that don't fit in VRAM+DRAM, preventing the catastrophic queuing seen in C4.

### Prerequisites

1. **vLLM Dynamo** with NIXL KV cache transfer (disaggregated prefill/decode)
2. **GDS driver** on g7e (NVIDIA 580.x driver supports GDS on Blackwell sm_120)
3. **NVMe formatted for GDS** — needs O_DIRECT support, ext4/xfs on the RAID0 volume
4. Benchmark: compare same 1000-concurrent workload with Tier 0 only vs Tier 0+1 vs Tier 0+1+2

### Expected Outcome

If GDS-backed NVMe eliminates the KV cache bottleneck at 1000 concurrent, the g7e becomes viable for the customer's exact workload at 4.6x lower cost than p5en — transforming it from a "good for moderate concurrency" option to a full replacement.

### 12. CPU Swap-Space Has Zero Effect on High-Concurrency TTFT
**Finding**: Increasing vLLM's `--swap-space` from 4 GiB to 64 GiB had **zero measurable effect** on throughput or latency at both 100 and 500 concurrent 10K-input-token requests.
**Validated via benchmarks** (g7e.24xlarge, 10240/1024, inf QPS):

| Concurrency | swap=4 TTFT | swap=64 TTFT | swap=4 tok/s | swap=64 tok/s |
|-------------|------------|-------------|-------------|--------------|
| 100 | 60,125 ms | 60,082 ms | 961.8 | 956.4 |
| 500 | 142,949 ms | 142,601 ms | 1,596.9 | 1,597.1 |

**Root cause**: The bottleneck at high concurrency is **prefill compute queuing**, not KV cache capacity. 500 requests × 10K tokens = 5M tokens of prefill, which takes ~143s to process serially on 4 GPUs regardless of how much KV cache overflow space is available. The vLLM V1 engine may also not fully utilize CPU swap for the mamba hybrid KV cache.
**Lesson**: For high-concurrency workloads, add more GPUs (disaggregated prefill) rather than more KV cache memory. CPU/NVMe KV offloading is primarily useful for long-lived session contexts and prefix cache persistence, not for raw concurrent throughput.

### 14. GDS Compat Mode Useless on EC2 NVMe
**Finding**: GPUDirect Storage compat mode provides **zero benefit** over standard CPU-mediated I/O on EC2 NVMe controllers (Amazon Elastic Block Store). NVMe P2PDMA is unsupported.
**Validated via gdsio** (g7e.24xlarge, 4-thread, 1M I/O, NVMe RAID0):

| Operation | GDS Compat (GPUD) | CPU-Only | Delta |
|-----------|-------------------|----------|-------|
| Read | 5.08 GiB/s, 769 μs | 5.31 GiB/s, 735 μs | -4.3% (GDS slower) |
| Write | 5.91 GiB/s, 661 μs | 5.87 GiB/s, 666 μs | +0.7% (negligible) |

**Lesson**: On EC2, skip GDS compat for NVMe. Use standard POSIX I/O. The real GDS opportunity is FSx Lustre via EFA (RDMA), where true zero-copy GPU↔storage is possible.

### 15. EFA Supported on g7e — Enables FSx GDS via RDMA
**Finding**: Both g7e.24xlarge and g7e.48xlarge support EFA (Elastic Fabric Adapter). This enables FSx Lustre PERSISTENT_2 with GDS over RDMA — true zero-copy GPU↔FSx path bypassing CPU entirely.
**Implication**: FSx GDS+EFA becomes the primary KV cache offloading strategy for g7e, not NVMe GDS. FSx also provides cross-node KV sharing for disaggregated serving.
**Config**: PERSISTENT_2, 1000 MB/s/TiB, EFA enabled, Lustre 2.15, minimum 4800 GiB.

## Best Practices

1. For scarce GPU types, try bare EC2 across multiple regions before EKS
2. Keep project names short to avoid AWS naming limits
3. Test MTP only on NVLink platforms — it hurts PCIe GPUs
4. Check model architecture (mamba hybrid vs transformer) before configuring speculative decoding
5. Use `--gpus <count>` syntax with nerdctl, not Docker's `--gpus '"device=..."'`
6. EC2 dry-run validates permissions only, not capacity — don't trust it
7. Monitor terraform background tasks to avoid state lock conflicts
8. Use NVMe RAID0 for model serving — much faster than FSx/S3
9. Don't bother with GDS compat mode on EC2 NVMe — use standard POSIX I/O
10. For GPU-direct storage on EC2, use FSx Lustre PERSISTENT_2 + EFA (true RDMA path)
