# SGLang Performance Benchmarks and Ecosystem Research

**Research date:** February 22, 2026

---

## 1. Project Overview

**SGLang** (Structured Generation Language) is a high-performance serving framework for large language models and multimodal models, developed originally at UC Berkeley/LMSYS. It provides low-latency, high-throughput inference from single GPUs to distributed clusters.

- **GitHub:** https://github.com/sgl-project/sglang
- **Stars:** ~23,600
- **Forks:** ~4,500
- **Contributors:** 1,182
- **License:** Apache-2.0
- **Current version:** v0.5.8 (January 23, 2026); PyPI latest: 0.5.8.post1 (February 5, 2026)
- **Documentation:** https://docs.sglang.io/
- **Academic paper:** https://arxiv.org/abs/2312.07104

---

## 2. Core Technical Innovations

### RadixAttention (KV Cache Reuse)
- Uses a radix tree data structure to manage the KV cache across multiple requests
- Enables automatic prefix caching so overlapping prompt prefixes reuse cached computations
- Claimed "up to 5x faster inference" from KV cache reuse alone
- Ablation studies show negligible overhead even without cache hits

### Compressed Finite State Machine (Constrained Decoding)
- "Jump-forward" decoding skips tokens whose next-token is deterministic under a grammar constraint
- Makes constrained decoding **faster than normal unconstrained decoding** -- a reversal of the typical performance penalty
- Uses a compressed FSM representation for efficient structured output generation

### Zero-Overhead Batch Scheduler (v0.4+)
- Prepares the next batch's metadata while the GPU processes the current batch
- Uses CUDA event synchronization and "future tokens" to resolve dependencies
- Entire scheduler implemented in ~4,000 lines of Python, yet matches or beats C++ alternatives

### Additional Optimizations
- Prefill-decode disaggregation
- Speculative decoding and continuous batching
- Tensor, pipeline, expert, and data parallelism
- Multi-LoRA batching
- Quantization: FP4, FP8, INT4, AWQ, GPTQ
- Flash Attention 4 decoding kernel support (v0.5.8)

---

## 3. Throughput Benchmarks

### 3.1 General Throughput (Paper Results, Jan 2024)

**Source:** https://lmsys.org/blog/2024-01-17-sglang/ and https://arxiv.org/abs/2312.07104

- **Claim:** Up to **6.4x higher throughput** compared to state-of-the-art inference systems
- **Baselines tested:** vLLM v0.2.5, Guidance v0.1.8, Hugging Face TGI v1.3.0
- **Models:** Llama-7B (single A10G), Mixtral-8x7B (8 GPUs with TP)
- **Precision:** FP16

**Workloads tested:**
| Workload | Description |
|----------|-------------|
| MMLU | 5-shot multi-choice |
| HellaSwag | 20-shot sentence completion |
| ReAct Agent | Agent control tasks |
| Tree-of-Thought | Multi-step reasoning |
| JSON extraction | Structured data extraction |
| Chat (short/long) | Multi-turn conversations |
| DSPy RAG | Retrieval-augmented generation |
| LLaVA | Vision-language benchmarks |

Across all workloads, SGLang achieved **up to 5x higher throughput** compared to baseline systems. Benefits were most pronounced for workloads with significant prefix overlap and structured output requirements.

### 3.2 Llama 3 Serving Benchmarks (July 2024)

**Source:** https://lmsys.org/blog/2024-07-25-sglang-llama3/

**Key headline:** Up to **3.1x higher throughput on Llama-70B** vs. vLLM; "often matches or sometimes outperforms TensorRT-LLM."

| Configuration | SGLang vs vLLM | SGLang vs TensorRT-LLM |
|---------------|----------------|------------------------|
| Llama-8B, 1xA100, bf16 | Significantly faster | Approximately equal (~5,000 tok/s both) |
| Llama-70B, 8xA100, bf16 | ~3.1x higher throughput | Comparable; TRT-LLM better latency in online |
| Llama-70B, 8xH100, fp8 | Much higher | SGLang wins on short inputs; TRT-LLM wins on long inputs |
| Llama-405B, 8xH100, fp8 | Outperforms vLLM | TRT-LLM not benchmarked (missing optimizations) |

Notable: SGLang's batch scheduler is ~4K lines of Python yet achieves competitive performance with TensorRT-LLM's C++ engine.

### 3.3 DeepSeek MLA Optimizations (v0.3, September 2024)

**Source:** https://lmsys.org/blog/2024-09-04-sglang-v0-3/

- **3x to 7x higher throughput** than baseline for DeepSeek models on H100 GPUs
- Tested with BF16 and FP8 precision on ShareGPT datasets
- Key techniques: weight absorption, grouped decoding kernels, FP8 batched MatMul, quantized KV cache

### 3.4 v0.4 Scheduler and Load Balancer (December 2024)

**Source:** https://lmsys.org/blog/2024-12-04-sglang-v0-4/

**Zero-overhead batch scheduler:**
- 1.1x throughput increase over v0.3
- 1.3x speedup compared to state-of-the-art baselines
- Largest gains on small models and large tensor parallelism setups

**Cache-aware load balancer:**
- Up to **1.9x throughput increase**
- **3.8x improvement in cache hit rates** (20% to 75%)
- Absolute throughput: 82,665 --> 158,596 tokens/second
- Implemented in Rust for 2x speedup over Python-based alternatives

**DeepSeek data parallelism attention:**
- **1.9x decoding throughput improvement** on 8xH100 80GB GPUs

### 3.5 DeepSeek on GB300 NVL72 (February 2026)

**Source:** https://lmsys.org/blog/2026-02-19-gb300-longctx/ (joint NVIDIA and SGLang team)

Testing DeepSeek on NVIDIA GB300 vs GB200 with 128K/8K workloads:

| Metric | GB300 | GB200 | Speedup |
|--------|-------|-------|---------|
| Peak TPS/GPU (no MTP) | 226.2 | 147.9 | 1.53x |
| Peak TPS/GPU (with MTP) | 224.2 | 169.1 | 1.33x |
| Per-user throughput (MTP) | 43 | 23 | +87% |
| Effective decode batch size | 40 req/GPU | 24 req/GPU | 1.6x |
| FMHA kernel latency | 205ms | 277ms | 1.35x |
| TTFT for 128K prefill | 8.6s | ~10s | 1.07-1.23x |

Supports up to 576 concurrent requests at Expert Parallelism level 16.

---

## 4. Latency Benchmarks

### 4.1 Time to First Token (TTFT)

- SGLang "excelled in terms of latency, particularly for the first token latency, where a prefix cache hit can be significantly beneficial" (original paper blog post)
- On Llama 3 70B across A100 configurations, vLLM, SGLang, and TensorRT-LLM showed "similar TTFT and TPOT" (per vLLM's own benchmarks, September 2024 -- Source: https://blog.vllm.ai/2024/09/05/perf-update.html)

### 4.2 Qwen3-235B on AMD MI300X (February 2026)

**Source:** https://lmsys.org/blog/2026-02-11-Qwen-latency/

| Metric | Baseline | Optimized | Speedup |
|--------|----------|-----------|---------|
| **Qwen3-235B TTFT** | 756.54 ms | 450.59 ms | 1.67x |
| **Qwen3-235B TPOT** | 26.44 ms | 12.44 ms | 2.12x |
| **Qwen3-VL-235B TTFT** | 1,764 ms | 1,084.59 ms | 1.62x |
| **Qwen3-VL-235B TPOT** | 23.7 ms | 12.48 ms | 1.90x |

Test config: Single request, 8K input tokens, 500 output tokens.

Kernel-level optimizations:
- QKNorm + RoPE fusion: 127% speedup (11.6 us --> 5.1 us)
- AllReduce + AddRMSNorm + Quantization: 67% speedup (35 us --> 21 us)
- Image decoding (rocJPEG): ~7x speedup (27 ms --> 4 ms per 720p image)

### 4.3 torch.compile Integration (v0.3)

- Up to **1.5x speedup** on small batch sizes (1-32)
- Surpasses gpt-fast performance at batch size 1
- Targets linear/norm/activation layers

---

## 5. Structured Output Generation Performance

### 5.1 Compressed FSM / Jump-Forward Decoding (February 2024)

**Source:** https://lmsys.org/blog/2024-02-05-compressed-fsm/

**Key result:** Reduces latency by up to **2x** and boosts throughput by up to **2.5x** for constrained decoding compared to state-of-the-art.

**Baselines:** vLLM v0.2.7, Guidance v0.1.0, Outlines v0.2.5, Llama.cpp v0.2.38

- Model: Llama-7B on NVIDIA A10 GPU (24GB)
- Tasks: JSON character generation, structured city information extraction
- Key finding: SGLang *without* jump-forward already outperformed Outlines+vLLM
- Jump-forward decoding makes constrained decoding **faster than normal unconstrained decoding**

### 5.2 xgrammar Integration (v0.4, December 2024)

- Up to **10x faster JSON decoding** than competing open-source solutions
- Integrated xgrammar grammar backend for structured outputs
- Combined with the zero-overhead scheduler for end-to-end improvements

### 5.3 Overall Structured Output Claim

SGLang documentation claims **"3x faster JSON decoding"** as a headline feature across typical use cases.

---

## 6. Multimodal and Diffusion Model Performance

### 6.1 LLaVA-OneVision (v0.3)

- Up to **4.5x speedup** compared to HuggingFace/transformers implementation
- Supports interleaved text, multi-image, and video processing

### 6.2 SGLang-Diffusion (February 2026)

**Source:** https://lmsys.org/blog/2026-02-16-sglang-diffusion-advanced-optimizations/

Advanced optimizations for video generation models (Wan2.2):
- Token-level sequence sharding: reduces padding from 14.3% to 0%
- Parallel folding: decouples text encoder and DiT parallelism
- Distributed VAE with halo exchange
- LayerNorm fusion with custom JIT kernels
- Tested on 8xH100 GPUs, model input shape B=1, T=21, H=90, W=160
- Communication reduction: 12.5% lower communication volume vs. frame-level sharding

### 6.3 v0.5.8 Diffusion Performance

- "Up to 1.5x faster across the board for all major diffusion models"
- Layerwise offload reducing VRAM by 30GB
- Performance improvements reaching 58% in some configurations

---

## 7. Head-to-Head Comparison Summary

### SGLang vs. vLLM

| Aspect | SGLang Advantage | Notes |
|--------|------------------|-------|
| Throughput (general) | Up to 3-6x | Most pronounced with prefix-heavy workloads |
| Llama 3 70B throughput | Up to 3.1x | July 2024 benchmarks |
| DeepSeek MLA | 3-7x | Specialized MLA kernel optimizations |
| Structured output | 3-10x | Compressed FSM + xgrammar |
| Latency (Llama 70B) | ~Similar | vLLM's own blog confirms parity on A100 |
| Scheduler overhead | Lower | Zero-overhead batch scheduler |
| Prefix caching | Significantly better | RadixAttention with automatic reuse |

### SGLang vs. TensorRT-LLM

| Aspect | Result | Notes |
|--------|--------|-------|
| Llama-8B throughput | ~Equal | Both ~5,000 tok/s on A10G |
| Llama-70B (short input) | SGLang faster | On H100 with FP8 |
| Llama-70B (long input) | TRT-LLM faster | Better long-context handling |
| Online latency | TRT-LLM slightly better | For Llama-70B on A100 |
| Ease of use | SGLang (Python) | TRT-LLM requires C++ compilation |
| Model support breadth | SGLang | Faster new model onboarding |

### SGLang vs. HuggingFace TGI

| Aspect | SGLang Advantage | Notes |
|--------|------------------|-------|
| Throughput | Up to 5-6x | Original paper benchmarks |
| Multimodal (LLaVA) | Up to 4.5x | v0.3 benchmarks |

---

## 8. Real-World Adoption and Use Cases

### Major Adopters (listed on GitHub README)

| Organization | Category |
|-------------|----------|
| **xAI** | AI company (Grok models) |
| **NVIDIA** | GPU/AI hardware |
| **AMD** | GPU/AI hardware |
| **Intel** | CPU/AI hardware |
| **LinkedIn** | Social platform |
| **Cursor** | AI code editor |
| **Oracle Cloud** | Cloud provider |
| **Google Cloud** | Cloud provider |
| **Microsoft Azure** | Cloud provider |
| **AWS** | Cloud provider |
| **Stanford University** | Academic |
| **UC Berkeley** | Academic (origin) |
| **MIT** | Academic |

### Scale of Deployment

- Powers **over 400,000 GPUs worldwide** in production (per official documentation)
- Used as rollout backend for reinforcement learning systems (AReaL, Miles, slime, Tunix, verl)
- LMSYS Chatbot Arena uses SGLang for serving models

### Hardware Ecosystem Support

- NVIDIA: GB200, GB300, H100, A100, Spark, Jetson Orin
- AMD: MI355, MI300 series
- Intel: Xeon CPUs
- Google: TPUs
- Huawei: Ascend NPUs

---

## 9. Release History and Development Velocity

| Version | Date | Key Highlights |
|---------|------|---------------|
| v0.1.x | Jan-Jul 2024 | Initial release; RadixAttention, compressed FSM |
| v0.2.x | Jul-Sep 2024 | Llama 3 optimization; up to 3.1x vs vLLM |
| v0.3.x | Sep-Nov 2024 | 7x faster DeepSeek MLA; 1.5x torch.compile; 4.5x LLaVA speedup |
| v0.4.x | Dec 2024-Aug 2025 | Zero-overhead scheduler; 10x faster JSON; cache-aware load balancer |
| v0.5.x | Aug 2025-present | Diffusion model support; Flash Attention 4; GB300 support |
| **v0.5.8** | **Jan 23, 2026** | **1.5x faster diffusion; near-linear scaling for million-token contexts; DeepSeek V3.2 NVFP4** |

**Development pace:** Multiple releases per month during active development cycles. 1,182 contributors as of February 2026. The project maintains a rapid iteration cadence with 10+ patch releases per minor version.

**Gateway releases:** SGLang also ships a separate Model Gateway component (latest: Gateway v0.3.1, January 9, 2026) providing enterprise routing with JWT/OIDC auth, 10-12x faster cache-aware routing with 99% memory reduction.

---

## 10. Community and Ecosystem

- **GitHub stars:** ~23,600 (as of February 2026)
- **Forks:** ~4,500
- **Contributors:** 1,182
- **PyPI package:** `sglang` (latest 0.5.8.post1)
- **Python requirement:** >=3.10

### Integrations with Post-Training Frameworks
- AReaL (reinforcement learning)
- Miles
- slime
- Tunix
- verl

### Partnerships/Collaborations
- Joint NVIDIA + SGLang blog posts (GB300 benchmarks)
- Joint AMD + Qwen + SGLang latency optimizations
- Official backend for LMSYS Chatbot Arena

---

## 11. Key Takeaways

1. **Throughput leadership:** SGLang consistently demonstrates 2-7x throughput advantages over vLLM across most benchmarks, with the gap being largest for DeepSeek MLA models (3-7x) and structured output generation (up to 10x).

2. **Competitive with TensorRT-LLM:** Despite being implemented in Python, SGLang matches or exceeds TensorRT-LLM on many configurations, though TRT-LLM retains advantages for long-input workloads and certain latency-sensitive scenarios.

3. **Structured output is a standout feature:** The compressed FSM + xgrammar combination makes constrained decoding faster than unconstrained decoding -- a unique achievement among inference frameworks.

4. **Rapid adoption:** From its 2024 launch to powering 400K+ GPUs by early 2026, SGLang has achieved remarkable industry adoption across cloud providers, AI companies, and hardware vendors.

5. **Broad hardware support:** Unlike TensorRT-LLM (NVIDIA-only), SGLang supports AMD, Intel, Google TPU, and Ascend NPUs, making it more portable across deployment environments.

6. **Active development:** With 1,182 contributors and frequent releases, the project maintains one of the most active development communities in the LLM infrastructure space.

---

## Sources

1. SGLang GitHub Repository: https://github.com/sgl-project/sglang
2. Original SGLang Paper (arXiv): https://arxiv.org/abs/2312.07104
3. SGLang Launch Blog Post (Jan 2024): https://lmsys.org/blog/2024-01-17-sglang/
4. Compressed FSM Blog Post (Feb 2024): https://lmsys.org/blog/2024-02-05-compressed-fsm/
5. Llama 3 Serving Benchmarks (Jul 2024): https://lmsys.org/blog/2024-07-25-sglang-llama3/
6. SGLang v0.3 Release (Sep 2024): https://lmsys.org/blog/2024-09-04-sglang-v0-3/
7. vLLM Performance Update (Sep 2024): https://blog.vllm.ai/2024/09/05/perf-update.html
8. SGLang v0.4 Release (Dec 2024): https://lmsys.org/blog/2024-12-04-sglang-v0-4/
9. Qwen3 Latency on MI300X (Feb 2026): https://lmsys.org/blog/2026-02-11-Qwen-latency/
10. SGLang-Diffusion Optimizations (Feb 2026): https://lmsys.org/blog/2026-02-16-sglang-diffusion-advanced-optimizations/
11. DeepSeek on GB300 (Feb 2026): https://lmsys.org/blog/2026-02-19-gb300-longctx/
12. SGLang Documentation: https://docs.sglang.io/
13. PyPI Package: https://pypi.org/project/sglang/
