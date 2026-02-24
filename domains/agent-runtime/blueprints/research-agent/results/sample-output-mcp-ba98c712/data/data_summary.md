# SGLang Performance Data Summary

**Generated:** 2026-02-22
**Sources:** Research notes from `sglang_architecture.md`, `sglang_features.md`, `sglang_benchmarks.md`

---

## 1. Throughput Comparison: SGLang vs Other Frameworks

All throughput multipliers are relative to the stated baseline system.

| Benchmark Scenario | SGLang Speedup | Baseline System | Source |
|--------------------|---------------|-----------------|--------|
| Llama-8B, 1xA100, bf16 | ~1.52x | vLLM (~3,300 tok/s vs ~5,000 tok/s) | Llama 3 benchmarks, Jul 2024 |
| Llama-70B, 8xA100, bf16 | **3.1x** | vLLM | Llama 3 benchmarks, Jul 2024 |
| Llama-70B, 8xH100, fp8 (short input) | ~3.0x | vLLM | Llama 3 benchmarks, Jul 2024 |
| DeepSeek MLA, H100 | **3x--7x** | Baseline systems (BF16/FP8) | v0.3 release, Sep 2024 |
| JSON Decoding (xgrammar) | **Up to 10x** | Competing open-source solutions | v0.4 release, Dec 2024 |
| LLaVA-OneVision | **4.5x** | HuggingFace Transformers | v0.3 release, Sep 2024 |
| Overall (paper, diverse workloads) | **Up to 6.4x** | State-of-the-art systems (vLLM, Guidance, TGI) | Original paper, Jan 2024 |

### SGLang vs TensorRT-LLM (Llama-70B)

| Configuration | Result |
|---------------|--------|
| 8xA100, bf16 | SGLang significantly faster throughput; TRT-LLM ~1.5x over vLLM |
| 8xH100, fp8, short input | SGLang wins |
| 8xH100, fp8, long input | TRT-LLM wins |
| Online latency (Llama-70B, A100) | TRT-LLM slightly better |

---

## 2. Key Optimization Speedup Factors

Ranked by magnitude (descending):

| Optimization | Speedup | Context |
|-------------|---------|---------|
| xgrammar JSON Decoding | **10.0x** | vs previous FSM backend and alternatives |
| DeepSeek MLA Kernels | **3x--7.0x** | vs baseline on H100 GPUs |
| Overall Peak (paper) | **6.4x** | Across diverse workloads vs SoTA |
| RadixAttention Prefix Caching | **Up to 5.0x** | vs systems without prefix caching |
| GB200 NVL72 Decode Throughput | **4.8x** | Decode throughput on NVL72 |
| LLaVA-OneVision | **4.5x** | vs HuggingFace Transformers |
| Prefill-Decode Disaggregation | **2.7x--3.8x** | On specialized hardware configs |
| Llama-70B vs vLLM | **3.1x** | 8xA100, bf16 |
| Compressed FSM (original) | **2.0x--2.5x** | vs Outlines + vLLM |
| EAGLE-3 Speculative Decoding | **2.36x** | 373.25 vs 158.34 tok/s on Llama-8B |
| Cache-Aware Load Balancer | **1.9x** | Throughput; 3.8x cache hit rate improvement |
| DP Attention for DeepSeek | **1.9x** | Decoding throughput on 8xH100 |
| torch.compile | **Up to 1.5x** | Batch sizes 1-32 |
| Zero-Overhead Scheduler | **1.3x** | vs competing schedulers (SoTA) |

---

## 3. Speculative Decoding Performance (Llama-8B)

| Method | Throughput (tok/s) | Speedup vs Baseline |
|--------|--------------------|---------------------|
| Baseline (no speculation) | 158.34 | 1.00x |
| EAGLE-2 | 244.10 | 1.54x |
| EAGLE-3 | 373.25 | 2.36x |

---

## 4. Latency Benchmarks

### Qwen3-235B on AMD MI300X (Feb 2026)

| Metric | Baseline | Optimized | Speedup |
|--------|----------|-----------|---------|
| TTFT | 756.54 ms | 450.59 ms | 1.67x |
| TPOT | 26.44 ms | 12.44 ms | 2.12x |

### Qwen3-VL-235B on AMD MI300X (Feb 2026)

| Metric | Baseline | Optimized | Speedup |
|--------|----------|-----------|---------|
| TTFT | 1,764 ms | 1,084.59 ms | 1.62x |
| TPOT | 23.7 ms | 12.48 ms | 1.90x |

### Kernel-Level Micro-Optimizations

| Optimization | Before | After | Speedup |
|-------------|--------|-------|---------|
| QKNorm + RoPE fusion | 11.6 us | 5.1 us | 127% (2.27x) |
| AllReduce + AddRMSNorm + Quantization | 35 us | 21 us | 67% (1.67x) |
| Image decoding (rocJPEG) | 27 ms | 4 ms | ~7x |

---

## 5. DeepSeek on GB300 NVL72 (Feb 2026)

| Metric | GB300 | GB200 | Speedup |
|--------|-------|-------|---------|
| Peak TPS/GPU (no MTP) | 226.2 | 147.9 | 1.53x |
| Peak TPS/GPU (with MTP) | 224.2 | 169.1 | 1.33x |
| Per-user throughput (MTP) | 43 tok/s | 23 tok/s | +87% |
| Effective decode batch size | 40 req/GPU | 24 req/GPU | 1.6x |
| FMHA kernel latency | 205 ms | 277 ms | 1.35x |
| TTFT for 128K prefill | 8.6s | ~10s | 1.07--1.23x |

---

## 6. Cache-Aware Load Balancer (v0.4)

| Metric | Without | With | Improvement |
|--------|---------|------|-------------|
| Throughput (tok/s) | 82,665 | 158,596 | 1.9x |
| Cache hit rate | 20% | 75% | 3.8x |

---

## 7. Scale and Adoption Metrics

| Metric | Value |
|--------|-------|
| GPUs powered globally | 400,000+ |
| GitHub stars | ~23,600 |
| Contributors | 1,182 |
| Commits | 9,893+ |
| Supported model configs | ~166 |
| Current version | v0.5.8 (Jan 2026) |
| Hardware platforms | 5 (NVIDIA, AMD, Intel, Google TPU, Huawei Ascend) |

---

## Charts Generated

1. **`/app/files/charts/sglang_throughput_comparison.png`** -- Bar chart showing SGLang throughput advantage (1.52x to 10x) across six benchmark scenarios vs respective baselines.

2. **`/app/files/charts/sglang_speedup_factors.png`** -- Horizontal bar chart ranking 15 documented SGLang optimizations from 1.3x (zero-overhead scheduler) to 10x (xgrammar JSON decoding).

3. **`/app/files/charts/sglang_detailed_benchmarks.png`** -- Two-panel chart: (left) speculative decoding throughput on Llama-8B (158--373 tok/s), (right) three-way framework comparison for Llama-70B across hardware configs.
