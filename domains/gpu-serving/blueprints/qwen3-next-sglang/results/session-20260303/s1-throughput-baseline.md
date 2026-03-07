# S1 Throughput Baseline — Qwen3-Next FP8 on SGLang (g7e.24xlarge)

**Date**: 2026-03-03
**Hardware**: g7e.24xlarge (2x RTX PRO 6000 Blackwell Server Edition, sm_120)
**Config**: SGLang nightly-dev-20260221-b2573fe4, TP=2, FP8, all-Triton backends
**Model**: Qwen3-Next FP8 (77GB checkpoint, hybrid DeltaNet+GQA, 512 MoE experts)
**Workload**: Random tokens, input=1024, output=512

## Key Flags

```
--tp 2 --dtype bfloat16 --context-length 65536 --chunked-prefill-size 32768
--max-running-requests 128 --mem-fraction-static 0.80
--attention-backend triton --fp8-gemm-backend triton --moe-runner-backend triton
--disable-cuda-graph --mamba-scheduler-strategy no_buffer --mamba-ssm-dtype bfloat16
```

## Results Summary

| Metric | S1a (QPS 0.5) | S1b (QPS 2.0) | S1c (QPS 4.0) | S1d (QPS 8.0) |
|--------|---------------|---------------|---------------|---------------|
| **Num Prompts** | 100 | 200 | 400 | 400 |
| **Request Rate (target)** | 0.5 | 2.0 | 4.0 | 8.0 |
| **Request Throughput (req/s)** | 0.45 | 1.53 | 2.75 | 3.24 |
| **Output Throughput (tok/s)** | **118.23** | **400.82** | **723.85** | **850.81** |
| **Peak Output (tok/s)** | 221 | 806 | 1,662 | 1,664 |
| **Total Token Throughput (tok/s)** | 335.21 | 1,176.69 | 2,149.27 | 2,526.26 |
| **Avg Concurrency** | 10.36 | 38.00 | 87.05 | 134.63 |
| **Peak Concurrency** | 17 | 72 | 159 | 286 |
| **Mean TTFT (ms)** | 226 | 177 | 2,014 | 12,731 |
| **Median TTFT (ms)** | 206 | 175 | 226 | 10,367 |
| **Mean TPOT (ms)** | 87 | 95 | 116 | 114 |
| **Median TPOT (ms)** | 87 | 94 | 116 | 117 |
| **P99 TPOT (ms)** | 97 | 112 | 162 | 152 |
| **Mean E2E Latency (ms)** | 23,250 | 24,823 | 31,598 | 41,579 |
| **P99 E2E Latency (ms)** | 44,630 | 49,222 | 62,145 | 72,635 |
| **Mean ITL (ms)** | 87 | 95 | 113 | 110 |
| **P99 ITL (ms)** | 245 | 342 | 405 | 331 |
| **Duration (s)** | 224.5 | 130.6 | 145.2 | 123.5 |

## Analysis

### Target Assessment
- **Target**: >= 150 tok/s output throughput
- **Result**: **PASS** at QPS >= 2.0 (400+ tok/s)
- S1a at QPS 0.5 yields 118 tok/s — below target but this is expected at very low load since the server is underutilized

### Throughput Scaling
- Output throughput scales well from QPS 0.5 to 8.0: 118 -> 401 -> 724 -> 851 tok/s
- Peak output throughput plateaus at ~1,664 tok/s (seen in both S1c and S1d)
- Server saturates around QPS 4.0-8.0 (actual throughput: 2.75-3.24 req/s vs target 4.0-8.0)

### Latency Behavior
- TTFT is excellent at low load (175-226ms) but degrades significantly under saturation (12.7s at QPS 8.0)
- TPOT remains stable across all loads (87-117ms), indicating consistent decode performance
- The ~87ms TPOT translates to ~11.5 tok/s per-request decode speed

### Comparison to vLLM on p5en (TP=4)
- vLLM on p5en target: 230 tok/s at TP=4
- SGLang on g7e at TP=2 delivers 724-851 tok/s at moderate-high load
- **3.1-3.7x the vLLM baseline** at batch saturation (apples-to-oranges: different hardware + engine)

### Known Performance Limiters
1. **No CUDA graphs** — disabled due to DeepGEMM sm_120 incompatibility
2. **Triton fallback for all kernels** — no optimized FlashInfer/DeepGEMM/Flash Attention
3. **No FP8 kernel tuning configs** — using default W8A8 Block FP8 kernel config (no device-specific optimization)
4. **No MoE kernel tuning configs** — using default Triton MoE config (512 experts)
5. **TP=2 instead of TP=4** — only 2 of 4 GPUs utilized due to memory constraints

### Optimization Opportunities
- Generate device-specific FP8 GEMM and MoE kernel configs for RTX PRO 6000
- Test with CUDA graphs once DeepGEMM adds sm_120 support
- Evaluate TP=4 with reduced context length or mem-fraction-static
- Test with HiCache (Phase S3) for tiered KV cache on NVMe
