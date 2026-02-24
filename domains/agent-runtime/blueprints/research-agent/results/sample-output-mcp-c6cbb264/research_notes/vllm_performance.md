# vLLM Performance Benchmarks and Comparative Analysis

**Research Date:** February 22, 2026
**Sources:** Official vLLM blog, academic paper (arXiv:2309.06180), Anyscale benchmarks, BentoML benchmarks, vLLM GitHub repository, vllm.ai

---

## 1. Overview

vLLM is an open-source, high-throughput LLM inference and serving engine originally developed at UC Berkeley's Sky Computing Lab. Its core innovation is **PagedAttention**, an attention algorithm inspired by virtual memory and paging in operating systems. The project is licensed under Apache 2.0 and has become one of the most widely adopted LLM serving frameworks in the industry.

---

## 2. Throughput Benchmarks

### 2.1 vLLM vs. HuggingFace Transformers

| Scenario | Throughput Improvement |
|---|---|
| Single output completion | **14x - 24x** higher throughput |
| Three parallel output completions | **8.5x - 15x** higher throughput |
| Real-world deployment (LMSYS Chatbot Arena) | Up to **30x** higher throughput vs. initial HF backend |

*Source: vLLM launch blog (June 2023)*

### 2.2 vLLM vs. HuggingFace Text Generation Inference (TGI)

| Scenario | Throughput Improvement |
|---|---|
| Single output completion | **2.2x - 2.5x** higher throughput |
| Three parallel output completions | **3.3x - 3.5x** higher throughput |

*Source: vLLM launch blog (June 2023)*

### 2.3 vLLM vs. DeepSpeed-FastGen

- vLLM is **up to 1.8x faster** than DeepSpeed-FastGen across most workloads (longer outputs, balanced prompt/output ratios).
- DeepSpeed-FastGen shows advantages only in narrow use cases with long prompts and short outputs, though the vLLM team notes "the performance gain we observe isn't as significant as 2x" despite DeepSpeed's marketing claims.

*Source: vLLM blog (November 2023)*

### 2.4 vLLM vs. TensorRT-LLM, SGLang, and LMDeploy (v0.6.0 Benchmarks)

Tested on Llama 3 models with ShareGPT, Prefill-heavy, and Decode-heavy datasets:

**Llama 3 8B on 1x H100:**
- vLLM v0.6.0 achieved **state-of-the-art throughput** on ShareGPT and Decode-heavy datasets.
- Lower throughput on Prefill-heavy workloads compared to TensorRT-LLM.
- **2.7x throughput improvement** and **5x faster TPOT** vs. vLLM v0.5.3.

**Llama 3 70B on 4x H100:**
- vLLM achieved **highest throughput** on ShareGPT and Decode-heavy datasets.
- Only **marginally higher** throughput than TensorRT-LLM r24.07.
- **1.8x throughput improvement** and **2x less TPOT** vs. vLLM v0.5.3.

**Llama 3 70B on 4x A100:**
- vLLM achieved **highest throughput** on ShareGPT dataset with comparable latency to competitors.

*Source: vLLM v0.6.0 performance update (September 2024)*

### 2.5 Continuous Batching Throughput Study (Anyscale)

| Framework / Technique | Throughput vs. Naive Batching |
|---|---|
| HuggingFace Pipelines (baseline) | 1x |
| NVIDIA FasterTransformer | **4x** (optimized model implementation) |
| TGI / Ray Serve (continuous batching) | **8x** |
| vLLM (continuous batching + PagedAttention) | **Up to 23x** |

At high output length variance (max 1536 tokens): static batching dropped to 81 tokens/sec, while vLLM more than doubled naive continuous batching performance.

*Source: Anyscale continuous batching blog*

### 2.6 BentoML Multi-Backend Benchmark (Llama 3, A100 80GB)

**Llama 3 8B - Token Generation Rate:**

| Backend | Tokens/sec (100 concurrent users) |
|---|---|
| LMDeploy | ~4,000 (best) |
| vLLM | ~2,300-2,500 |
| TGI | ~2,300-2,500 (similar to vLLM) |
| TensorRT-LLM | ~2,300-2,500 (similar to vLLM) |

**Llama 3 70B Q4 - Token Generation Rate:**

| Backend | Tokens/sec (100 concurrent users) |
|---|---|
| LMDeploy | ~700 (best) |
| TensorRT-LLM | Similar to LMDeploy |
| vLLM | Notably lower (under-optimized for quantized models at that time) |

**Key finding:** vLLM achieved **best-in-class TTFT (time-to-first-token)** across all concurrency levels for 8B models, even when raw token generation throughput was not the highest.

*Source: BentoML inference backend benchmark*

---

## 3. Latency Benchmarks

### 3.1 Time-to-First-Token (TTFT)

- vLLM consistently demonstrates **best-in-class or near-best TTFT** across multiple independent benchmarks.
- In the BentoML benchmark: vLLM had the best TTFT across all concurrency levels (10, 50, 100 users) for Llama 3 8B on A100.

### 3.2 Time-Per-Output-Token (TPOT)

- vLLM v0.6.0 achieved **5x faster TPOT** on Llama 3 8B compared to v0.5.3.
- vLLM v0.6.0 achieved **2x faster TPOT** on Llama 3 70B compared to v0.5.3.

### 3.3 Load Testing (QPS Scaling)

- At QPS=1: All continuous batching systems performed equally.
- At QPS=4: vLLM outperformed TGI and Ray Serve, with saturation occurring around QPS=8 at approximately **1,900 tokens/second** throughput.

### 3.4 Speculative Decoding Latency Improvements

| Method | Model | Dataset | Speedup |
|---|---|---|---|
| Draft model (Qwama-0.5B) | Llama 3 70B (4xH100) | ShareGPT | **Up to 1.5x** at low QPS |
| Prompt lookup (n-gram) | Llama 3 70B (4xH100) | CNN/DailyMail | **Up to 2.8x** at low QPS |

Note: Performance degrades under heavy load (1.4x-1.8x slowdown at high QPS).

*Source: vLLM speculative decoding blog (October 2024)*

---

## 4. Memory Efficiency (PagedAttention)

### 4.1 The KV Cache Problem

- The KV cache for a single sequence in LLaMA-13B can consume **up to 1.7 GB** of memory.
- Traditional systems waste **60-80%** of KV cache memory due to fragmentation and over-allocation.

### 4.2 PagedAttention Solution

PagedAttention partitions the KV cache into fixed-size blocks (analogous to OS memory pages), enabling:

| Metric | Traditional Systems | vLLM (PagedAttention) |
|---|---|---|
| KV cache memory waste | 60-80% | **Under 4%** |
| Memory sharing (parallel sampling) | Not supported | **Up to 55% memory reduction** |
| Throughput from memory sharing | Baseline | **Up to 2.2x improvement** |

### 4.3 Academic Paper Results (arXiv:2309.06180)

- **2-4x throughput improvements** compared to FasterTransformer and Orca at equivalent latency levels.
- Gains are more pronounced with longer sequences, larger models, and complex decoding algorithms (beam search, parallel sampling).
- Achieves **near-zero waste** in KV cache memory allocation.

### 4.4 FP8 Quantization (Llama 3.1 405B)

On a single 8xH100 or 8xA100 setup with Meta's official FP8 model:
- **Throughput:** 2.82 requests/sec (avg 1024 input + 128 output tokens)
- **Token throughput:** 2,884.86 input tokens/sec, 291.53 output tokens/sec
- **Accuracy:** GSM8K benchmark achieved 95.38% (+/- 0.56% stddev) exact match vs. 96.8% BF16 baseline (minimal accuracy loss).

---

## 5. vLLM V1 Architecture Improvements (January 2025)

The V1 rewrite introduced significant architectural changes:

| Improvement | Detail |
|---|---|
| Throughput vs. V0 | **Up to 1.7x higher** for text models |
| Prefix caching | **Near-zero performance degradation** even at 0% cache hit rate (enabled by default) |
| Multiprocessing | Isolated EngineCore execution loop with better CPU/GPU overlap |
| Persistent Batch | Caches input tensors with differential updates to minimize CPU overhead |
| Tensor parallelism | Symmetric architecture with incremental state updates, minimizing IPC |

Larger speedups reported for vision-language models (e.g., Qwen2-VL).

---

## 6. Pipeline Parallelism Performance

On 16x H100 GPUs (Llama 3.1 405B FP8):
- 2-way pipeline + 8-way tensor parallelism vs. 16-way tensor parallelism alone.
- **6.6x performance improvement** when nodes lack InfiniBand interconnect.
- Both configurations performed similarly when InfiniBand was available.

---

## 7. Framework Comparison Summary

### 7.1 vLLM vs. TensorRT-LLM

| Dimension | vLLM | TensorRT-LLM |
|---|---|---|
| Throughput | Comparable or slightly higher on decode-heavy workloads | Better on prefill-heavy workloads |
| Ease of use | Drop-in Python library, OpenAI-compatible API | Requires model compilation step, more complex setup |
| Hardware support | NVIDIA, AMD, Intel, TPU, ARM, PowerPC | NVIDIA-only |
| Model support | Broad HuggingFace ecosystem | Requires explicit model support/conversion |
| Quantization | GPTQ, AWQ, INT4/INT8, FP8 | FP8, INT8, INT4, extensive quantization toolkit |
| Open source | Apache 2.0 | Apache 2.0 |

### 7.2 vLLM vs. TGI (Text Generation Inference)

| Dimension | vLLM | TGI |
|---|---|---|
| Throughput | **2.2x - 3.5x higher** (2023 benchmarks) | Lower, but continuously improving |
| Memory efficiency | PagedAttention (< 4% waste) | Flash Attention, less aggressive memory optimization |
| API compatibility | OpenAI-compatible | Custom API + OpenAI-compatible |
| Deployment | Python-native, Docker | Rust-based server, Docker |
| Ecosystem | Broader hardware support | Tight HuggingFace integration |

### 7.3 vLLM vs. llama.cpp

| Dimension | vLLM | llama.cpp |
|---|---|---|
| Target use case | Server-side, high-throughput, multi-GPU | Edge/desktop, single-GPU, CPU inference |
| Optimization focus | GPU batching, PagedAttention | Quantization (GGUF), Apple Silicon, CPU |
| Throughput (batched) | Significantly higher at batch sizes > 1 | Optimized for low-batch regime |
| Hardware | Multi-GPU, data center | Single GPU, CPU, Apple Metal |
| Language | Python + CUDA | C++ |

### 7.4 vLLM vs. SGLang

| Dimension | vLLM | SGLang |
|---|---|---|
| Throughput | Comparable (v0.6.0+) | Competitive, especially on structured generation |
| Unique feature | PagedAttention, broad ecosystem | RadixAttention, constrained decoding |
| Maturity | More mature, larger community | Newer, rapidly evolving |

### 7.5 vLLM vs. LMDeploy

| Dimension | vLLM | LMDeploy |
|---|---|---|
| Raw throughput | Lower on some quantized models | Higher in BentoML benchmarks (Llama 3 8B) |
| TTFT | Best-in-class | Competitive |
| Community | Much larger | Smaller, primarily Chinese ecosystem |

---

## 8. Real-World Deployment Use Cases

### 8.1 LMSYS Chatbot Arena

- Handles **30,000 daily requests** on average, with peaks of **60,000 requests/day**.
- vLLM powers **more than half** of Chatbot Arena requests.
- **Reduced GPU requirements by 50%** after switching to vLLM.

### 8.2 Production Deployment Stack

The vLLM ecosystem includes purpose-built tools for production:
- **AIBrix** - Kubernetes-native deployment controller
- **Production Stack** - Reference architecture for scaling
- **GuideLLM** - Performance evaluation and optimization toolkit
- **LLM Compressor** - Model quantization and optimization

### 8.3 Large-Scale Serving (2025-2026)

Recent blog posts reference:
- **DeepSeek-V3 serving at 2.2k tokens/sec/H200** using Wide Expert Parallelism.
- **DeepSeek-V3.2 on NVIDIA GB300** performance breakthroughs.
- **GPT-OSS optimizations on NVIDIA Blackwell** architecture.
- SemiAnalysis InferenceMAX benchmark collaboration with NVIDIA.

### 8.4 RLHF Acceleration

vLLM is used as the inference backend for **OpenRLHF**, enabling efficient rollout generation during reinforcement learning from human feedback training pipelines.

### 8.5 Docker Integration

**Docker Model Runner** integrated vLLM for high-throughput inferencing (November 2025), bringing vLLM to Docker's developer ecosystem.

---

## 9. Community Adoption Statistics

### 9.1 GitHub Metrics (as of February 2026)

| Metric | Value |
|---|---|
| GitHub Stars | **~70,900** |
| Forks | **~13,600** |
| Contributors | **2,205** |
| Used by (downstream projects) | **7,600+** |
| Latest release | v0.15.1 (February 4, 2026) |
| License | Apache 2.0 |

### 9.2 Financial Backing

- Backed by **a16z** and **Sequoia Capital**.
- Compute sponsorship from **AWS, Google Cloud, NVIDIA**, and others.

### 9.3 Community Channels

- Active Slack workspace, discussion forum (discuss.vllm.ai), and GitHub Issues.
- **45+ official blog posts** covering features, benchmarks, and ecosystem updates (as of February 2026).

### 9.4 Hardware Ecosystem Support

Supported hardware backends (reflecting broad industry adoption):
- NVIDIA CUDA GPUs
- AMD ROCm GPUs
- Intel CPUs and GPUs (via Intel Gaudi plugin)
- Google Cloud TPUs
- AWS Neuron (Inferentia/Trainium)
- IBM Spyre
- Huawei Ascend
- Apple Silicon
- PowerPC CPUs
- ARM CPUs

### 9.5 Model Ecosystem

Supports trending open-source model families:
- **DeepSeek** (V3, V3.2, R1)
- **Meta Llama** (3, 3.1, 3.2, 3.3)
- **Google Gemma**
- **Mistral / Mixtral**
- **Qwen** (including Qwen2-VL multimodal)
- Mixture-of-Experts (MoE) models
- Embedding models
- Multi-modal LLMs (LLaVA, etc.)

### 9.6 Integration Ecosystem

- **Llama Stack** - Official inference provider
- **LangChain / LlamaIndex** - LLM framework integrations
- **Ray Serve** - Distributed serving
- **BentoML** - ML model serving platform
- **OpenAI-compatible API** - Drop-in replacement for OpenAI clients

---

## 10. Key Technical Features Contributing to Performance

| Feature | Performance Impact |
|---|---|
| **PagedAttention** | Near-zero KV cache waste (< 4%), enables 2-4x throughput vs. prior art |
| **Continuous batching** | Up to 23x throughput vs. naive static batching |
| **CUDA/HIP graph execution** | Reduces kernel launch overhead |
| **FlashAttention integration** | Faster attention computation, reduced memory |
| **Speculative decoding** | Up to 2.8x latency reduction for specific workloads |
| **Prefix caching** | Avoids redundant computation for shared prefixes |
| **Chunked prefill** | Overlaps prefill and decode for better GPU utilization |
| **Tensor parallelism** | Efficient multi-GPU scaling |
| **Pipeline parallelism** | Up to 6.6x improvement on non-InfiniBand clusters |
| **FP8 quantization** | Near-lossless compression with minimal accuracy impact |
| **torch.compile** | JIT compilation for optimized execution |

---

## 11. Summary and Key Takeaways

1. **Throughput leader**: vLLM consistently ranks among the top 1-2 inference engines for decode-heavy and general-purpose LLM serving, with 14-24x improvements over naive HuggingFace and 2-3.5x over TGI.

2. **Memory efficiency breakthrough**: PagedAttention reduces KV cache waste from 60-80% to under 4%, a foundational improvement that enables higher batch sizes and thus higher throughput.

3. **Rapid improvement cadence**: From v0.5.3 to v0.6.0, vLLM achieved 2.7x throughput gains and 5x TPOT reduction. The V1 architecture added another 1.7x on top of that.

4. **Broadest hardware support**: No other inference engine supports as many hardware backends (NVIDIA, AMD, Intel, TPU, Neuron, Apple Silicon, ARM, PowerPC).

5. **Largest open-source community**: With ~70,900 GitHub stars, 2,205 contributors, and 7,600+ downstream projects, vLLM has the largest community among dedicated LLM serving frameworks.

6. **Trade-offs**: vLLM is not always the absolute fastest. LMDeploy can outperform it on raw token generation rate, TensorRT-LLM can be faster on prefill-heavy workloads, and llama.cpp is better optimized for single-user/edge scenarios. vLLM's strength is its balance of performance, ease of use, and ecosystem breadth.

7. **Production-proven**: Used at scale by LMSYS Chatbot Arena (60k daily peak requests), integrated into Docker, Llama Stack, and backed by a16z and Sequoia Capital.

---

*Note: Benchmark numbers should be interpreted with caution as they depend heavily on hardware, model size, input/output distribution, concurrency level, and software version. Numbers cited here come from various dates (2023-2026) and may not reflect the absolute latest performance of each framework.*
