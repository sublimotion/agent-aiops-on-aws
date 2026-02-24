# TensorRT-LLM Data Summary

**Compiled:** February 2026
**Sources:** NVIDIA official documentation, MLPerf submissions, BentoML benchmarks, vLLM blog, HuggingFace Optimum-NVIDIA

---

## 1. Model Ecosystem Statistics

| Metric | Value |
|--------|-------|
| Total distinct model architectures | ~93+ |
| PyTorch backend language models | ~23 architecture classes |
| PyTorch backend multimodal models | ~13 architecture classes |
| TensorRT backend LLM models | ~48 architectures |
| TensorRT backend multimodal models | ~16 architectures |
| MoE model architectures supported | 13+ |
| Supported GPU generations | 5 (Ampere, Ada Lovelace, Hopper, Grace Hopper, Blackwell) |
| Quantization methods available | 10+ (FP8, FP4, INT8 SmoothQuant, INT4 AWQ, INT4 GPTQ, W8A16, W4A16, W4A8, etc.) |
| Source framework conversion paths | 5 (HuggingFace, NeMo, DeepSpeed, JAX, ModelOpt) |
| Speculative decoding methods | 8 (Draft-Target, N-Gram, Medusa, EAGLE v1/v2, EAGLE3, ReDrafter, Lookahead, MTP) |
| Current version (early 2026) | v1.3.x |

---

## 2. Throughput Benchmarks -- LLaMA Family on H100 SXM 80GB (tokens/sec)

All figures are offline maximum throughput, FP8 quantization.

| Model | ISL=128/OSL=128 | ISL=128/OSL=2048 | ISL=2048/OSL=2048 | ISL=20000/OSL=2000 | Tensor Parallelism |
|-------|-----------------|-------------------|--------------------|--------------------|-------------------|
| LLaMA 3.1 8B | ~26,500 | ~22,000 | ~18,500 | ~1,500 | 1 |
| LLaMA 3.3 70B | ~7,500 | ~6,500 | ~5,500 | ~800 | 2 |
| LLaMA 3.1 405B | ~3,500 | ~3,800 | ~3,200 | N/A | 8 |

**Peak recorded:** 28,390 tokens/sec (LLaMA 3.1 8B, ISL=128/OSL=128, trtllm-bench)

---

## 3. Throughput Benchmarks -- H200 SXM 141GB (tokens/sec)

| Model | ISL=128/OSL=128 | ISL=128/OSL=2048 | ISL=2048/OSL=2048 | Tensor Parallelism |
|-------|-----------------|-------------------|--------------------|--------------------|
| LLaMA 3.1 8B | 27,305 | 24,046 | ~20,000 | 1 |
| LLaMA 3.3 70B | ~9,000 | ~8,500 | ~7,000 | 2 |
| LLaMA 3.1 405B | ~5,000 | ~5,200 | ~4,500 | 8 |

---

## 4. Throughput Benchmarks -- B200 Blackwell (tokens/sec, FP4)

| Model | ISL=128/OSL=128 | ISL=128/OSL=2048 | Tensor Parallelism |
|-------|-----------------|-------------------|--------------------|
| LLaMA 3.3 70B | 10,614 | 9,446 | 1 (single GPU!) |
| LLaMA 3.1 405B | 6,219 | 7,178 | 4 |
| LLaMA 4 (headline) | >40,000 | -- | 8 |

---

## 5. Framework Comparison (BentoML, A100 80GB, 100 Concurrent Users)

### LLaMA 3 8B (FP16)

| Framework | Throughput (tok/s) |
|-----------|--------------------|
| LMDeploy | ~4,000 |
| TensorRT-LLM | ~2,400 |
| vLLM | ~2,400 |
| TGI (HuggingFace) | ~2,400 |

### LLaMA 3 70B (INT4 Quantized)

| Framework | Throughput (tok/s) |
|-----------|--------------------|
| TensorRT-LLM | ~700 |
| LMDeploy | ~700 |
| vLLM | ~450 (lower, lacked optimized quantized kernels) |

### TensorRT-LLM vs HuggingFace Transformers (H100, LLaMA 2)

| Metric | Improvement |
|--------|-------------|
| Throughput | Up to 28x faster |
| First token latency | Up to 3.3x faster |
| Peak throughput (LLaMA 2 7B/13B FP8) | 1,200 tokens/sec |

---

## 6. GPU Generation Performance Scaling

### Relative Performance on LLaMA 2 70B

| GPU | Architecture | Memory | Bandwidth | Relative Perf vs A100 |
|-----|-------------|--------|-----------|----------------------|
| A100 80GB | Ampere | 80GB HBM2e | 2.0 TB/s | 1.0x (baseline) |
| H100 SXM 80GB | Hopper | 80GB HBM3 | 3.35 TB/s | 4.0-4.6x |
| GH200 96GB | Grace Hopper | 96GB HBM3 | 4.0 TB/s | ~5.6x (1.4x vs H100) |
| H200 SXM 141GB | Hopper | 141GB HBM3e | 4.8 TB/s | ~6.0x (1.3-1.45x vs H100) |
| B200 180GB | Blackwell | 180GB HBM3e | 8.0 TB/s | ~17.8x (3.0x vs H200) |
| GB200 192GB | Blackwell | 192GB HBM3e | 8.0 TB/s | Highest per-GPU |

### MLPerf v5.0 Absolute Numbers (LLaMA 2 70B, 8-GPU Server Mode)

| System | Server (tok/s) | Offline (tok/s) |
|--------|----------------|-----------------|
| H200 8-GPU | 33,072 | 34,988 |
| B200 NVL8 | 98,443 | 98,858 |

### MLPerf v5.0 -- Mixtral 8x7B

| System | Server (tok/s) | Offline (tok/s) |
|--------|----------------|-----------------|
| H200 | 61,802 | 62,630 |
| B200 | 126,845 | 128,148 |

---

## 7. Key Speedup Claims (Published)

| Claim | Context |
|-------|---------|
| Up to 8x faster | GPT-J 6B: H100+TRT-LLM vs A100+PyTorch |
| Up to 28x faster | LLaMA 2 on H100 vs stock HuggingFace Transformers |
| Up to 4.6x faster | LLaMA 2 70B: H100 vs A100 |
| Up to 4x per-GPU | B200 vs H100 on LLaMA 2 70B (MLPerf v4.1) |
| Up to 3.4x per-GPU | GB200 NVL72 vs H200 on LLaMA 3.1 405B |
| Up to 3.0x | B200 vs H200 on LLaMA 2 70B (MLPerf v5.0) |
| Up to 30x system-level | GB200 NVL72 vs H200 8-GPU on LLaMA 3.1 405B |
| >40,000 tok/s | LLaMA 4 on B200 GPUs |
| Up to 3.6x | Speculative decoding throughput improvement |
| >2x throughput | In-flight batching vs static batching |
| Up to 22% | CUDA Graphs end-to-end throughput increase |

---

## 8. Total Cost of Ownership (TCO) Impact

| Model | TCO Reduction (H100+TRT-LLM vs A100+PyTorch) | Energy Reduction |
|-------|-----------------------------------------------|------------------|
| GPT-J 6B | 5.3x | 5.6x |
| LLaMA 2 70B | 3.0x | 3.2x |

---

## 9. Software-Only Optimization Gains (Same Hardware)

| Benchmark | Improvement | Timeframe |
|-----------|-------------|-----------|
| GPT-J on Hopper (offline) | 2.9x | Since initial Hopper support |
| GPT-J on Hopper (server) | 3.8x | Since initial Hopper support |
| LLaMA 2 70B on H100 | 1.5x | One year of software updates |
| MLPerf v4.0 to v4.1 | 14% | XQA kernel + layer fusion optimizations |

---

## 10. Quantization Memory Reduction

| Format | Approximate Memory Reduction |
|--------|------------------------------|
| FP16/BF16 | 2x vs FP32 |
| INT8 / FP8 | 4x vs FP32 (2x vs FP16) |
| INT4 / FP4 | 8x vs FP32 (4x vs FP16) |

Notable: FP4 on Blackwell enables LLaMA 3.3 70B on a single B200 GPU at >10,000 tok/s, and LLaMA 3.1 405B on just 4 GPUs (vs 8 on Hopper).

---

## Charts Generated

| File | Description |
|------|-------------|
| `/app/files/charts/chart1_llama_throughput_h100.png` | Bar chart: TensorRT-LLM throughput (tokens/sec) across LLaMA 3.1 8B, 3.3 70B, and 3.1 405B on H100, at three ISL/OSL configurations |
| `/app/files/charts/chart2_framework_comparison.png` | Side-by-side bar charts: TensorRT-LLM vs vLLM vs LMDeploy vs TGI on LLaMA 3 8B (FP16) and LLaMA 3 70B (INT4) |
| `/app/files/charts/chart3_gpu_scaling.png` | Bar chart: GPU generation performance scaling from A100 through B200 on LLaMA 2 70B (MLPerf data) |
