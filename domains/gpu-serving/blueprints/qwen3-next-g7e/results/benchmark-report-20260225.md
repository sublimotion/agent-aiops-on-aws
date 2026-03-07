# Qwen3-Next G7E Benchmark Report

**Model**: Qwen3-Next-80B-A3B-Instruct (MoE, 80B total / 3B active, FP8)
**Instance**: g7e.24xlarge (4x RTX PRO 6000 Blackwell Server Edition, 96 GB GDDR7 each, sm_120)
**Region**: us-west-2d (on-demand, bare EC2)
**Date**: 2026-02-25
**vLLM**: v0.15.0 (cu130, TRITON Fp8 MoE backend)
**Driver**: NVIDIA 580.126.09, CUDA compute capability 12.0

---

## Hardware Summary

| Component | Spec |
|-----------|------|
| GPUs | 4x NVIDIA RTX PRO 6000 Blackwell Server Edition |
| VRAM per GPU | 96 GB GDDR7 (97,887 MiB) |
| Total VRAM | 384 GB |
| Interconnect | PCIe (no NVLink) |
| vCPUs | 96 |
| NVMe Storage | 7 TB RAID0 (4x 1.75 TB) |
| KV Cache (FP8) | 63.17 GiB available, 2,758,080 tokens capacity |
| Max concurrent @ 32K ctx | 302.73x |

## vLLM Configuration (Baseline)

```
--tensor-parallel-size 4
--quantization fp8
--gpu-memory-utilization 0.90
--max-model-len 32768
--kv-cache-dtype fp8
--max-num-seqs 512
--enable-prefix-caching
```

Notes:
- Custom allreduce disabled (PCIe-only, >2 GPUs)
- SymmMemCommunicator unavailable (sm_120 not yet supported)
- Mamba cache mode 'align' auto-enabled with prefix caching (hybrid attention+mamba architecture)
- No MoE config tuned for Blackwell — using default TRITON backend

---

## Benchmark Results

### C0: Smoke Test (QPS 0.5, 1024/512, 20 requests)

| Metric | Value |
|--------|-------|
| Output throughput | 230.87 tok/s |
| Peak throughput | 465.00 tok/s |
| Total throughput | 692.62 tok/s |
| Mean TTFT | 196.32 ms |
| P50 TTFT | 156.96 ms |
| Mean TPOT | 10.04 ms |
| Mean E2E | 5,327 ms |
| Failed requests | 0 |

### C1: QPS Sweep (1024/512, 100 requests each)

| QPS | Output tok/s | Peak tok/s | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Mean E2E (s) | Peak Conc |
|-----|-------------|-----------|-------------|---------------|---------------|-------------|-----------|
| 0.5 | 230.9 | 465 | 692.6 | 196.3 | 10.0 | 5.3 | 6 |
| 1.0 | 484.7 | 1,211 | 1,454.1 | 148.7 | 12.3 | 6.4 | 16 |
| 2.0 | 911.3 | 1,446 | 2,734.0 | 70.5 | 13.2 | 6.8 | 26 |
| 4.0 | 1,532.6 | 3,029 | 4,597.8 | 138.5 | 21.1 | 10.9 | 64 |
| 8.0 | 2,172.4 | 3,843 | 6,517.1 | 161.0 | 26.5 | 13.7 | 100 |
| 16.0 | 3,389.0 | 5,800 | 10,167.0 | 304.4 | 42.8 | 22.2 | 200 |

**Sweet spot**: QPS 4-8 delivers 1,500-2,200 tok/s with acceptable latency (<27ms TPOT).

### C3: Context Scaling (QPS 2.0, 50 requests each)

| Context | Input/Output | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Mean E2E (s) |
|---------|-------------|-------------|-------------|---------------|---------------|-------------|
| 4K | 4096/256 | 451.8 | 7,680.2 | 250.7 | 15.4 | 4.2 |
| 16K | 16384/512 | 676.6 | 22,327.2 | 1,335.8 | 40.1 | 21.8 |
| 32K | 24576/512 | 717.9 | 35,175.6 | 744.6 | 31.3 | 16.7 |

TTFT scales with context length as expected. 32K context works well within the 96 GB GDDR7 budget.

### C4: Customer-Comparable (1000 concurrent, 10K/1K, inf QPS)

| Metric | g7e.24xlarge (4 GPU) | Customer H200 (8 GPU) |
|--------|---------------------|----------------------|
| Output tok/s | 1,820.7 | 3,612.8 |
| **Per-GPU tok/s** | **455.2** | **451.6** |
| Peak tok/s | 6,344 | — |
| Mean TTFT | 229,154 ms | 1,470 ms |
| Mean E2E | 446,269 ms | 19,070 ms |
| Requests/min | 109.2 | 182.7 |
| Failed | 0 | — |

**Key insight**: Per-GPU output throughput is nearly identical (455 vs 452 tok/s). The massive TTFT/E2E difference is due to queuing — 1000 concurrent requests at 10K tokens each (10M total input tokens) far exceeds the 4-GPU KV cache capacity (2.76M tokens), causing requests to queue. The customer's 8-GPU setup has ~2x the capacity, allowing much better concurrency handling.

---

## MTP Speculative Decoding Comparison

Tested with `--speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}'` and `--no-enable-prefix-caching` (required: mamba 'align' mode incompatible with MTP in v0.15.0).

| QPS | Baseline tok/s | MTP tok/s | Delta | Baseline TPOT | MTP TPOT |
|-----|---------------|-----------|-------|---------------|----------|
| 1.0 | 484.7 | 476.4 | -1.7% | 12.3 ms | 15.9 ms |
| 2.0 | 911.3 | 856.3 | -6.0% | 13.2 ms | 22.0 ms |
| 4.0 | 1,532.6 | 1,329.9 | -13.2% | 21.1 ms | 32.5 ms |
| 8.0 | 2,172.4 | 1,708.3 | -21.4% | 26.5 ms | 37.7 ms |
| 16.0 | 3,389.0 | 2,009.6 | -40.7% | 42.8 ms | 39.6 ms |

**Finding**: MTP degrades throughput on g7e.24xlarge by 2-41% across QPS levels. The speculative head computation + inter-GPU verification overhead on PCIe interconnect outweighs the speculative decoding benefit. MTP may be better suited to NVLink-connected GPUs (H200, A100) where inter-GPU communication is faster.

Note: The customer uses `--no-enable-chunked-prefill` with MTP, but vLLM 0.15.0 requires chunked prefill for mamba cache mode. They may be on a custom/patched build.

---

## Cost-Efficiency Analysis

| Metric | g7e.24xlarge (4 GPU) | p5en.48xlarge (8x H200) |
|--------|---------------------|------------------------|
| On-demand $/hr | $8.27 | $63.30 |
| GPUs | 4 | 8 |
| $/hr/GPU | $2.07 | $7.91 |
| Output tok/s @ QPS 8 | 2,172 | ~3,600 (estimated) |
| $/M output tokens | $1.06 | $4.88 |
| **Cost efficiency ratio** | **4.6x cheaper** | baseline |

| Metric | g7e.48xlarge (8 GPU) | p5en.48xlarge (8x H200) |
|--------|---------------------|------------------------|
| On-demand $/hr | $16.54 | $63.30 |
| GPUs | 8 | 8 |
| Projected tok/s @ QPS 8 | ~4,300 (linear scale) | ~3,600 |
| $/M output tokens | $1.07 | $4.88 |
| **Cost efficiency ratio** | **4.6x cheaper** | baseline |

**The g7e Blackwell platform delivers equivalent per-GPU throughput at 3.8x lower $/GPU/hr.** Total cost per output token is approximately 4.6x lower than H200-based p5en instances.

Caveats:
- g7e.48xlarge not tested (capacity unavailable); 8-GPU scaling may not be perfectly linear due to PCIe limitations
- NVLink on p5en enables better scaling with MTP and high-concurrency workloads
- g7e uses GDDR7 (1.5 TB/s bandwidth) vs H200 HBM3 (4.8 TB/s); memory-bound workloads may differ

---

## KV Cache Offloading Test

Tested whether increasing vLLM's CPU swap space for KV cache offloading improves high-concurrency performance.

### Configuration

- `--swap-space 4` (default) vs `--swap-space 64` (64 GiB CPU RAM for KV overflow)
- NVMe-backed Linux swap (256 GiB) also configured but not activated by vLLM
- vLLM V1 engine with `--kv-offloading-backend native`

### Results (10240 input / 1024 output, inf QPS)

| Concurrency | Swap Size | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Duration (s) |
|-------------|-----------|-------------|---------------|---------------|-------------|
| 100 | 4 GiB | 961.8 | 60,125 | 44.3 | 106.5 |
| 100 | 64 GiB | 956.4 | 60,082 | 44.9 | 107.1 |
| 500 | 4 GiB | 1,596.9 | 142,949 | 160.9 | 320.6 |
| 500 | 64 GiB | 1,597.1 | 142,601 | 161.1 | 320.6 |

### Finding

**CPU swap-space has zero effect on performance.** At 100 concurrent (1M total tokens < 2.76M KV capacity), KV fits entirely in VRAM. At 500 concurrent (5M tokens > 2.76M capacity), the bottleneck is **prefill compute queuing** — 5M tokens of prefill computation takes ~143s regardless of KV cache overflow capacity.

### GDS Bandwidth (NVMe RAID0)

| Operation | GDS Compat (GPUD) | CPU-Only | Delta |
|-----------|-------------------|----------|-------|
| Read | 5.08 GiB/s, 769 μs | 5.31 GiB/s, 735 μs | -4.3% (GDS slower) |
| Write | 5.91 GiB/s, 661 μs | 5.87 GiB/s, 666 μs | +0.7% (negligible) |

GDS compat mode provides no benefit on EC2 NVMe (no P2PDMA support). FSx GDS via EFA (RDMA) is the viable zero-copy path but requires EFA-enabled instances.

### Implication

For this workload (high-concurrency, long-input), the solution is **more prefill compute** (disaggregated prefill/decode or more GPUs), not more KV cache memory. KV offloading to NVMe/FSx is primarily useful for:
- Long-lived session contexts (keeping historical KV warm)
- Prefix cache persistence (reusing shared system prompts)
- Asymmetric workloads (short prefill, very long decode with many concurrent sessions)

---

## Compatibility Notes

1. **Blackwell sm_120 support**: vLLM 0.15.0 (cu130) works on Blackwell. SymmMemCommunicator and custom allreduce not yet optimized for sm_120.
2. **MoE kernel tuning**: No pre-tuned MoE config for `NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition`. Using default TRITON Fp8 MoE backend — performance may improve with tuned configs.
3. **Mamba cache**: Qwen3-Next hybrid architecture uses mamba cache mode 'align'. Prefix caching works (experimental). MTP incompatible with mamba 'align' in v0.15.0.
4. **GDDR7 thermals**: Idle ~28-29C, ~82W per GPU. Good thermal headroom.
5. **NVMe performance**: 7 TB RAID0 from 4x NVMe, model loaded in <2 min from local storage.

## Recommendations

1. **For cost-optimized inference**: Use g7e.24xlarge with baseline config (no MTP) at QPS 4-8. This delivers the best cost-efficiency.
2. **For latency-sensitive workloads**: Use g7e.48xlarge (when available) with 8 GPUs for 2x concurrency headroom.
3. **Skip MTP on g7e**: Speculative decoding hurts throughput on PCIe GPUs. Keep it for NVLink platforms.
4. **MoE tuning opportunity**: Creating a tuned MoE config for Blackwell could unlock additional throughput.
5. **g7e.48xlarge testing**: When capacity becomes available, test 8-GPU scaling to validate linear throughput projection.

## Instance Details

- **Instance ID**: i-03955d59a22d67ad1
- **Region/AZ**: us-west-2d
- **AMI**: AL2023 x86_64 NVIDIA (EKS-optimized)
- **Key pair**: g7e-bench (us-west-2)
- **IAM role**: g7e-bench-role (S3 read + SSM)
