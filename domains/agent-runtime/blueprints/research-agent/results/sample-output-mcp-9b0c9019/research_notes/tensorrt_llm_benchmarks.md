# NVIDIA TensorRT-LLM Benchmark Performance Research

> **Research compiled:** February 2026
> **Sources:** NVIDIA official documentation, MLPerf submissions, third-party benchmark studies (BentoML, vLLM blog, HuggingFace Optimum), NVIDIA developer blogs

---

## Table of Contents

1. [Overview](#overview)
2. [Throughput Benchmarks (Tokens/Second)](#throughput-benchmarks-tokenssecond)
3. [Performance Comparisons vs Other Frameworks](#performance-comparisons-vs-other-frameworks)
4. [MLPerf Inference Benchmark Results](#mlperf-inference-benchmark-results)
5. [Quantization Performance Gains](#quantization-performance-gains)
6. [GPU Generation Performance Comparisons](#gpu-generation-performance-comparisons)
7. [Published Speedup Claims](#published-speedup-claims)
8. [Latency Benchmarks](#latency-benchmarks)
9. [Key Optimizations Driving Performance](#key-optimizations-driving-performance)
10. [Caveats and Methodology Notes](#caveats-and-methodology-notes)

---

## Overview

TensorRT-LLM is NVIDIA's open-source library for optimizing Large Language Model (LLM) inference on NVIDIA GPUs. It leverages kernel fusion, quantization (FP8, FP4, INT8, INT4), in-flight batching, paged KV caching, tensor/pipeline parallelism, and custom attention kernels (FlashAttention, XQA) to maximize throughput and minimize latency.

TensorRT-LLM supports multi-GPU and multi-node deployments with built-in parallelism strategies. As of early 2026, it supports a broad set of model architectures including LLaMA 2/3/3.1/3.3/4, DeepSeek, Mixtral, Falcon, GPT-J, Mistral, Qwen, and many others.

---

## Throughput Benchmarks (Tokens/Second)

### Official NVIDIA Reference Benchmarks (TensorRT-LLM, Offline Maximum Throughput)

All results below use an offline maximum throughput scenario where all requests are queued in rapid succession. Starting with TensorRT-LLM v0.19, the PyTorch backend is used, eliminating engine build requirements.

#### LLaMA 3.1 8B (FP8, Tensor Parallelism = 1)

| ISL/OSL     | GH200 96GB | H100 SXM 80GB | H200 SXM 141GB |
|-------------|------------|----------------|-----------------|
| 128/128     | 26,402     | ~26,500        | 27,305          |
| 128/2048    | 21,413     | ~22,000        | 24,046          |
| 2048/128    | ~22,000    | ~22,500        | ~24,000         |
| 2048/2048   | ~18,000    | ~18,500        | ~20,000         |
| 20000/2000  | 1,341      | ~1,500         | 1,706           |

*Peak: ~27,305 tokens/sec on H200 at ISL=128, OSL=128*

#### LLaMA 3.3 70B (FP8, Tensor Parallelism = 2)

| ISL/OSL     | H100 SXM 80GB | H200 SXM 141GB |
|-------------|----------------|-----------------|
| 128/128     | ~7,500         | ~9,000          |
| 128/2048    | ~6,500         | ~8,500          |
| 2048/2048   | ~5,500         | ~7,000          |
| 20000/2000  | ~800           | ~1,100          |

#### LLaMA 3.3 70B (FP4, Tensor Parallelism = 1) -- Blackwell Only

| ISL/OSL     | B200 180GB | GB200 192GB |
|-------------|------------|-------------|
| 128/128     | 10,614     | 11,101      |
| 128/2048    | 9,446      | 10,276      |
| 2048/2048   | ~8,000     | ~8,500      |
| 20000/2000  | 636        | 732         |

*Notable: 70B model on a SINGLE GPU with FP4 quantization on Blackwell achieves >10K tok/s*

#### LLaMA 3.1 405B (FP8, Tensor Parallelism = 8)

| ISL/OSL     | H100 SXM 80GB | H200 SXM 141GB |
|-------------|----------------|-----------------|
| 128/128     | ~3,500         | ~5,000          |
| 128/2048    | ~3,800         | ~5,200          |
| 2048/2048   | ~3,200         | ~4,500          |

#### LLaMA 3.1 405B (FP4, Tensor Parallelism = 4) -- Blackwell Only

| ISL/OSL     | B200 180GB | GB200 192GB |
|-------------|------------|-------------|
| 128/128     | 6,219      | 6,599       |
| 128/2048    | 7,178      | 7,497       |

*Notable: 405B model on only 4 GPUs with FP4, exceeding 7K tok/s*

#### LLaMA 4 Maverick 17Bx128E (FP8, Tensor Parallelism = 8)

Tested on H100 and H200 systems. Throughput numbers available in official documentation.

#### Standalone Benchmark Tool Results (trtllm-bench)

- **LLaMA 3.1 8B** (ISL=128, OSL=128): **28,390 tokens/sec** token throughput, **222 requests/sec** request throughput (3,000 requests benchmark)
- **Qwen2-VL-2B** (multimodal): 780 tokens/sec, 1.44 requests/sec, average latency 3,672ms

### Headline Throughput Claims from NVIDIA

| Model | GPU | Throughput | Notes |
|-------|-----|-----------|-------|
| LLaMA 4 | B200 | >40,000 tokens/sec | NVIDIA headline claim |
| LLaMA 3 | (not specified) | 24,000 tokens/sec | Aggregate throughput |
| LLaMA 3.1 405B | Multi-node | 400 tok/s per node, 37 tok/s per user | Production serving scenario |
| LLaMA 2 13B | H200 | ~12,000 tokens/sec | Near-peak throughput |

---

## Performance Comparisons vs Other Frameworks

### TensorRT-LLM vs HuggingFace Transformers

Source: HuggingFace Optimum-NVIDIA blog (official HuggingFace partnership with NVIDIA)

| Metric | Improvement | Details |
|--------|-------------|---------|
| **Throughput** | **Up to 28x faster** | Compared to stock HuggingFace Transformers |
| **First Token Latency** | **Up to 3.3x faster** | Time from prompt to first token |
| **Peak Throughput** | **1,200 tokens/sec** | LLaMA 2 7B/13B on H100 with FP8 |

- Tested on LLaMA 2 (7B and 13B) on H100
- H200 showed up to **2x additional throughput boost** over H100
- FP8 quantization was a key enabler of the 28x speedup claim

### TensorRT-LLM vs vLLM

Source: vLLM blog (v0.6.0 release, September 2024) and BentoML benchmarks

**Key findings from the vLLM team's own benchmarks:**

- **LLaMA 3 8B on A100:** TensorRT-LLM had the **highest throughput** among all engines. vLLM achieved second-highest on ShareGPT and decode-heavy workloads.
- **LLaMA 3 70B on A100:** vLLM achieved highest throughput on ShareGPT dataset; comparable to TensorRT-LLM on other metrics.
- **LLaMA 3 8B on H100:** vLLM claimed "state-of-the-art throughput on ShareGPT and decode-heavy datasets" but lower throughput on prefill-heavy workloads (where TensorRT-LLM excels).
- **LLaMA 3 70B on H100:** vLLM throughput described as "marginally higher than TensorRT-LLM" on ShareGPT/decode-heavy, but TensorRT-LLM was faster on prefill-heavy scenarios.
- **TTFT and TPOT:** Both systems showed comparable time-to-first-token and time-per-output-token across most scenarios.

**Summary:** The two frameworks are broadly competitive in 2024-2025, with TensorRT-LLM generally stronger on prefill-heavy and batch throughput workloads, and vLLM competitive or marginally ahead on decode-heavy/ShareGPT-style conversational workloads.

### TensorRT-LLM vs vLLM vs LMDeploy vs Others (BentoML Benchmark)

Source: BentoML blog, tested on A100 80GB

**LLaMA 3 8B (FP16):**

| Backend | Throughput at 100 users | TTFT | Notes |
|---------|------------------------|------|-------|
| LMDeploy | ~4,000 tok/s | Best-in-class at 10 users | Overall winner |
| vLLM | 2,300-2,500 tok/s | Best-in-class across all user counts | Most consistent latency |
| TensorRT-LLM | 2,300-2,500 tok/s | Comparable | Similar to vLLM |
| TGI | ~2,300-2,500 tok/s | Comparable | HuggingFace ecosystem |
| MLC-LLM | Similar to LMDeploy at 10 users | Degrades under load | Cross-platform |

**LLaMA 3 70B (4-bit Quantization):**

| Backend | Throughput at 100 users | TTFT at 100 users | Notes |
|---------|------------------------|---------------------|-------|
| LMDeploy | ~700 tok/s | Lowest across all levels | Best overall |
| TensorRT-LLM | Similar to LMDeploy initially | Exceeded 6 seconds | Degraded at high concurrency |
| vLLM | Lower | Consistently low | Lacks optimized quantized kernels |

**Key takeaway:** For quantized models, TensorRT-LLM and LMDeploy outperformed vLLM. However, TensorRT-LLM's TTFT degraded at high concurrency (100 users) in this benchmark.

---

## MLPerf Inference Benchmark Results

MLPerf is the industry-standard benchmark suite run by MLCommons. NVIDIA consistently uses TensorRT-LLM as its inference backend for LLM submissions.

### MLPerf Inference v5.0 (April 2025) -- Latest Major Round

#### LLaMA 2 70B Results

| GPU System | Server (tok/s) | Offline (tok/s) | vs H200 Speedup |
|------------|-----------------|------------------|------------------|
| B200 NVL8 | **98,443** | **98,858** | **3.0x (server), 2.8x (offline)** |
| H200 8-GPU | 33,072 | 34,988 | Baseline |

#### Mixtral 8x7B Results

| GPU System | Server (tok/s) | Offline (tok/s) | vs H200 Speedup |
|------------|-----------------|------------------|------------------|
| B200 | **126,845** | **128,148** | **2.1x** |
| H200 | 61,802 | 62,630 | Baseline |

#### LLaMA 3.1 405B (Server Mode)

- **GB200 NVL72** delivered **up to 3.4x higher per-GPU performance** compared to H200 8-GPU systems
- At system level: **up to 30x throughput increase** (combining 3.4x per-GPU gains with 9x more GPUs in NVL72)

#### LLaMA 2 70B Interactive

- **B200 NVL8** achieved **3.1x higher throughput** compared to H200 submissions

#### Historical Software Optimization Gains (Same Hardware)

- **H100 on LLaMA 2 70B:** Throughput increased **1.5x in one year** purely through TensorRT-LLM software optimizations (kernel fusions, pipeline parallelism improvements)
- **GPT-J on Hopper:** Cumulative improvement since introduction: **2.9x (offline)** and **3.8x (server)** -- all from software optimization alone

### MLPerf Inference v4.1 (August-September 2024)

#### Blackwell First Submissions (B200 single GPU)

| Benchmark | Server (tok/s) | Offline (tok/s) | vs H100 per-GPU |
|-----------|-----------------|------------------|------------------|
| LLaMA 2 70B | 10,756 | 11,264 | **Up to 4x** |

*First Blackwell submission, using FP4 quantization with TensorRT Model Optimizer.*

#### H200 8-GPU System Results

| Benchmark | Server Throughput |
|-----------|------------------|
| LLaMA 2 70B | 32,790 tok/s |
| Mixtral 8x7B | 57,177 tok/s |

#### GH200 Grace Hopper Superchip (per-accelerator vs H100)

| Benchmark | GH200 vs H100 |
|-----------|----------------|
| LLaMA 2 70B | **1.4x** faster |
| Mixtral 8x7B | **1.2x** faster |

- Single GH200 delivered **up to 22x higher throughput** on GPT-J compared to best two-socket x86 CPU-only submissions
- CPU-only x86 showed 55% performance degradation in server vs offline scenario on LLaMA 2 70B; GH200 maintained within 5%

#### TensorRT-LLM Software Gains (v4.0 to v4.1)

- XQA kernel optimizations and layer fusions yielded **up to 14% performance improvements** on existing hardware

### MLPerf Inference v4.0 (March 2024)

- **H100 + TensorRT-LLM** achieved speedups on GPT-J of **2.4x (offline)** and **2.9x (server)** -- nearly triple performance compared to prior MLPerf round
- **H200 at 700W:** 13.8 queries/sec on LLaMA 2 70B (server), 13.7 samples/sec (offline)
- **H200 at 700W vs H100:** Up to **28% better** on LLaMA 2 70B
- **H200 at 1000W vs H100:** **43-45% better** on LLaMA 2 70B

### MLPerf Benchmark Models Tracked

As of v5.0, MLPerf Inference Datacenter tracks these LLM benchmarks:
- LLaMA 2 70B (Q&A) -- OpenOrca dataset
- LLaMA 3.1 8B (Summarization) -- CNN-DailyMail dataset
- DeepSeek-R1 (Reasoning) -- Custom dataset
- Mixtral 8x7B (Text Generation) -- OpenOrca/GSM8K/MBXP
- LLaMA 3.1 405B (Text Generation) -- LongBench subset

---

## Quantization Performance Gains

### Supported Quantization Formats

| Format | Precision | GPU Requirement | Key Characteristics |
|--------|-----------|-----------------|---------------------|
| FP32 | 32-bit | All | Baseline, highest accuracy |
| FP16/BF16 | 16-bit | All | Standard training/inference precision |
| FP8 (E4M3/E5M2) | 8-bit float | Hopper+ (H100, H200) | Near-FP16 accuracy, ~2x speedup |
| INT8 SmoothQuant | 8-bit int (W8A8) | Ampere+ | Weights AND activations quantized |
| INT8 Weight-Only | 8-bit int (W8A16) | Ampere+ | Only weights quantized |
| INT4 AWQ/GPTQ | 4-bit int (W4A16) | Ampere+ | Aggressive compression |
| FP4 (NVFP4) | 4-bit float | Blackwell only (B200, GB200) | New in Blackwell, near-INT8 accuracy |

### Quantization Performance Impact

**FP8 vs FP16 (on Hopper GPUs):**
- FP8 is the primary driver behind the "up to 28x faster" claim vs HuggingFace Transformers (combined with other TensorRT-LLM optimizations)
- NVIDIA states FP8 "retains higher accuracy compared to other data formats like INT8 or INT4 while achieving the fastest performance" on Hopper
- The H100 Transformer Engine enables automatic FP8 format conversion

**FP4 (Blackwell):**
- FP4 quantization enabled the first single-GPU inference of LLaMA 3.3 70B on B200 at >10K tok/s
- At MLPerf v4.1, FP4 on B200 delivered **4x per-GPU performance** over H100 on LLaMA 2 70B
- FP4 reduces the tensor parallelism requirement: 405B model runs on 4 Blackwell GPUs (vs 8 on Hopper)

**INT4 AWQ (on A100, third-party benchmarks):**
- BentoML benchmarks showed TensorRT-LLM with 4-bit quantized LLaMA 3 70B achieving throughput comparable to LMDeploy (~700 tok/s at 100 concurrent users)
- vLLM underperformed on quantized models due to "lack of inference optimization for quantized models" at the time

**General Quantization Scaling:**
- Scaling factors can be per-tensor, per-token (M factors for M tokens), or per-channel (N factors for N channels)
- AWQ and GPTQ use per-group scaling factors with zero-offsetting

---

## GPU Generation Performance Comparisons

### Cross-Generation Comparison Summary

| GPU | Architecture | Memory | Bandwidth | LLaMA 2 70B Relative Perf |
|-----|-------------|--------|-----------|---------------------------|
| A100 80GB | Ampere | 80GB HBM2e | 2.0 TB/s | 1.0x (baseline) |
| H100 SXM 80GB | Hopper | 80GB HBM3 | 3.35 TB/s | ~4.0-4.6x vs A100 |
| GH200 96GB | Grace Hopper | 96GB HBM3 | 4.0 TB/s | ~1.4x vs H100 |
| H200 SXM 141GB | Hopper | 141GB HBM3e | 4.8 TB/s | ~1.3-1.45x vs H100 |
| B200 180GB | Blackwell | 180GB HBM3e | 8.0 TB/s | ~3.0x vs H200, ~4x vs H100 |
| GB200 192GB | Blackwell | 192GB HBM3e | 8.0 TB/s | Highest per-GPU |

### Detailed GPU Comparisons

#### A100 vs H100

Source: NVIDIA H100 TensorRT-LLM launch blog

| Model | Speedup (H100 vs A100) | Notes |
|-------|------------------------|-------|
| GPT-J 6B | **8x total** (4x GPU + 2x TRT-LLM+batching) | CNN/DailyMail summarization |
| LLaMA 2 70B | **4.6x** | Same workload |

- H100 alone (hardware only): **4x faster** than A100
- Combined with TensorRT-LLM in-flight batching: **8x total** for GPT-J 6B

#### H100 vs H200

Source: MLPerf v4.0 submissions

| Configuration | Improvement |
|---------------|-------------|
| H200 at 700W TDP vs H100 | Up to **28%** on LLaMA 2 70B |
| H200 at 1000W TDP vs H100 | **43-45%** on LLaMA 2 70B |
| H200 per-accelerator vs H100 (GH200 form) | **1.4x** on LLaMA 2 70B |

Key driver: H200's 141GB HBM3e (vs 80GB HBM3) and 4.8 TB/s bandwidth (1.4x higher than H100).

#### H200 vs B200/GB200 (Blackwell)

Source: MLPerf v5.0

| Benchmark | B200 vs H200 |
|-----------|--------------|
| LLaMA 2 70B (server) | **3.0x** |
| LLaMA 2 70B (offline) | **2.8x** |
| Mixtral 8x7B | **2.1x** |
| LLaMA 3.1 405B (per-GPU, GB200 NVL72) | **3.4x** |

#### Total Cost of Ownership (TCO) Impact

Source: NVIDIA blog (H100 + TensorRT-LLM vs A100 + PyTorch)

| Model | TCO Reduction | Energy Reduction |
|-------|---------------|------------------|
| GPT-J 6B | **5.3x** | **5.6x** |
| LLaMA 2 70B | **3.0x** | **3.2x** |

---

## Published Speedup Claims

### Official NVIDIA Claims

| Claim | Context | Source |
|-------|---------|--------|
| **"Up to 8x faster"** | GPT-J 6B: H100+TRT-LLM+batching vs A100+PyTorch | NVIDIA H100 TRT-LLM blog |
| **"Up to 28x faster"** | LLaMA 2 on H100 vs stock HuggingFace Transformers | HuggingFace Optimum-NVIDIA |
| **"Up to 4.6x faster"** | LLaMA 2 70B: H100 vs A100 | NVIDIA H100 TRT-LLM blog |
| **"Up to 4x per-GPU"** | B200 vs H100 on LLaMA 2 70B | MLPerf v4.1 |
| **"Up to 3.4x per-GPU"** | GB200 NVL72 vs H200 on LLaMA 3.1 405B | MLPerf v5.0 |
| **"Up to 3x throughput"** | Speculative decoding on LLaMA 3.3 70B | TensorRT-LLM GitHub |
| **">3x throughput"** | Multiblock attention for long sequences on H200 | TensorRT-LLM GitHub |
| **"Up to 30x system-level"** | GB200 NVL72 vs H200 8-GPU on LLaMA 3.1 405B | MLPerf v5.0 |
| **"Over 40,000 tok/s"** | LLaMA 4 on B200 GPUs | TensorRT-LLM GitHub |
| **"Up to 50x better performance"** | Blackwell Ultra for agentic AI | NVIDIA Newsroom (Feb 2026) |
| **"10x cost-per-token reduction"** | Blackwell at inference providers | NVIDIA Newsroom (Baseten, DeepInfra, etc.) |

### Third-Party Validated Claims

| Claim | Context | Source |
|-------|---------|--------|
| Comparable to vLLM on decode-heavy | LLaMA 3 8B/70B on H100 | vLLM blog v0.6.0 |
| Highest throughput on prefill-heavy | LLaMA 3 8B on A100 | vLLM blog v0.6.0 |
| Comparable to LMDeploy on quantized | LLaMA 3 70B INT4 on A100 | BentoML benchmark |

---

## Latency Benchmarks

### Time to First Token (TTFT)

- **TensorRT-LLM vs HuggingFace Transformers:** Up to **3.3x lower** TTFT (official Optimum-NVIDIA claim on LLaMA 2)
- **BentoML benchmark (LLaMA 3 8B, A100):** TensorRT-LLM TTFT comparable to vLLM and TGI at low concurrency; all backends similar
- **BentoML benchmark (LLaMA 3 70B INT4, A100):** TensorRT-LLM TTFT degraded to >6 seconds at 100 concurrent users (LMDeploy stayed lower)
- **MLPerf Server scenario:** GH200 maintained latency within 5% between server and offline scenarios, demonstrating consistent low-latency serving

### Inter-Token Latency / Time Per Output Token (TPOT)

- **vLLM blog comparison:** TTFT and TPOT "comparable" between vLLM and TensorRT-LLM across most LLaMA 3 configurations on both A100 and H100
- **Speculative decoding:** TensorRT-LLM's speculative decoding provides up to **3.6x** throughput increase, which directly reduces effective inter-token latency for generation

### Reference Latency Points (LLaMA 2 70B, 2xA100 80GB)

From Cursor.com blog (general LLaMA 2 70B inference characteristics):
- Prompt processing (512 tokens): ~170ms
- Prompt processing (1536 tokens): ~530ms
- Single-batch generation: ~18.6 tok/s (~54ms inter-token)
- Batch size 8 generation: Cost-efficient at $0.00825/1K tokens

---

## Key Optimizations Driving Performance

| Optimization | Impact | Description |
|-------------|--------|-------------|
| **In-flight Batching** | Major throughput gain | Continuously batches new requests while others are still generating |
| **Paged KV Cache** | Memory efficiency | Enables larger batch sizes by reducing KV cache memory waste |
| **FP8/FP4 Quantization** | 2-4x throughput | Leverages Hopper/Blackwell Transformer Engine hardware |
| **Tensor Parallelism** | Linear scaling | Distributes model across GPUs with NVLink interconnect |
| **Pipeline Parallelism** | Multi-node scaling | Distributes layers across nodes |
| **XQA Kernels** | 14%+ improvement | Optimized cross-attention for grouped-query attention models |
| **Speculative Decoding** | Up to 3.6x | Generates multiple candidate tokens per step |
| **Multiblock Attention** | >3x for long sequences | Optimized attention for very long contexts on H200 |
| **Kernel Fusion** | Reduced memory traffic | Fuses multiple operations into single GPU kernels |
| **FlashAttention** | Reduced memory bandwidth | Memory-efficient attention computation |

### Software-Only Gains Over Time

One of the most notable aspects of TensorRT-LLM is the continuous software optimization cadence:
- **GPT-J on Hopper:** 2.9x (offline) to 3.8x (server) improvement through software alone since initial Hopper support
- **LLaMA 2 70B on H100:** 1.5x throughput increase in one year from software optimizations
- **v4.0 to v4.1 MLPerf:** 14% improvement from XQA kernel optimizations and layer fusions

---

## Caveats and Methodology Notes

1. **NVIDIA's own benchmarks** use "offline maximum throughput" scenarios where all requests are queued simultaneously. This represents peak batch throughput, not interactive serving latency. NVIDIA explicitly states these "should not be considered as the peak performance that can be delivered by TensorRT-LLM."

2. **The "up to 28x" claim** vs HuggingFace Transformers combines multiple optimizations (TensorRT-LLM compilation, FP8 quantization, batching) against an unoptimized baseline. The comparison is between a fully-optimized production stack and a research-oriented library.

3. **vLLM comparisons** are workload-dependent. TensorRT-LLM excels on prefill-heavy and high-batch-throughput scenarios; vLLM is competitive or marginally ahead on decode-heavy/conversational workloads. The gap has narrowed significantly through 2024-2025.

4. **MLPerf results** represent highly tuned submissions with extensive configuration optimization. Production deployments typically achieve lower throughput.

5. **TensorRT-LLM requires CUDA-only NVIDIA GPUs** and model compilation (though the PyTorch backend in v0.19+ reduces this friction). This is a deployment consideration vs more portable frameworks.

6. **Quantization accuracy trade-offs** are not fully captured by throughput benchmarks. FP8 generally maintains near-FP16 quality, but INT4 may show measurable accuracy degradation depending on the model and task.

7. **Third-party benchmarks** (BentoML, vLLM blog) showed TensorRT-LLM's TTFT can degrade at very high concurrency (100 concurrent users), a consideration for latency-sensitive production serving.

8. **Blackwell FP4 results** are significant because they enable single-GPU inference for 70B-class models, fundamentally changing the deployment economics.

---

## Summary

TensorRT-LLM is consistently among the fastest LLM inference engines available, with particular strengths in:

- **Maximum batch throughput** on NVIDIA hardware (often the fastest in head-to-head comparisons)
- **Quantization support** with hardware-accelerated FP8 (Hopper) and FP4 (Blackwell)
- **Multi-GPU scaling** with optimized tensor/pipeline parallelism
- **Continuous improvement** through software optimizations (1.5-3.8x gains on same hardware over time)

The competitive landscape is nuanced: vLLM and LMDeploy are within striking distance on many workloads, and may be preferred for ease of deployment or specific workload patterns. However, for maximum throughput on NVIDIA GPUs -- especially on newer Hopper and Blackwell architectures with quantization -- TensorRT-LLM remains the performance leader in most benchmarks.
