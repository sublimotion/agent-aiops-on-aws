# vLLM Quantitative Data Summary

**Compiled:** February 22, 2026
**Sources:** vLLM research notes (architecture, features, performance)

---

## 1. Throughput Benchmarks (vLLM vs. Competing Frameworks)

| Comparison Framework        | Throughput Multiplier (Range) | Midpoint | Scenario / Baseline                     |
|-----------------------------|-------------------------------|----------|-----------------------------------------|
| HuggingFace Transformers    | 14x - 24x                    | 19.0x    | Single output completion                |
| HuggingFace TGI             | 2.2x - 3.5x                  | 2.85x    | Single and parallel completions         |
| NVIDIA FasterTransformer    | 2x - 4x                      | 3.0x     | Equivalent latency (PagedAttention paper)|

**Additional throughput data points:**

| Comparison                          | Value           | Context                                     |
|-------------------------------------|-----------------|---------------------------------------------|
| vs. HF Transformers (3 parallel)    | 8.5x - 15x     | Three parallel output completions            |
| vs. DeepSpeed-FastGen               | Up to 1.8x     | Most workloads (longer outputs)              |
| Continuous batching vs. static      | Up to 23x       | Anyscale benchmark, high output variance     |
| LMSYS real-world deployment         | Up to 30x       | vs. initial HuggingFace backend              |
| v0.6.0 vs. v0.5.3 (Llama 3 8B)     | 2.7x            | Internal version-over-version improvement    |
| v0.6.0 vs. v0.5.3 (Llama 3 70B)    | 1.8x            | Internal version-over-version improvement    |
| V1 architecture vs. V0             | Up to 1.7x      | Text models                                  |

**Chart:** `../charts/throughput_comparison.png`

---

## 2. Memory Efficiency (PagedAttention vs. Traditional Systems)

| Metric                              | Traditional Systems   | vLLM (PagedAttention)  | Improvement       |
|-------------------------------------|-----------------------|------------------------|--------------------|
| KV cache memory waste               | 60-80%                | Under 4%               | ~95% reduction     |
| Memory sharing (parallel sampling)  | Not supported         | Up to 55% reduction    | New capability     |
| Throughput from memory sharing      | Baseline              | Up to 2.2x             | Enabled by sharing |

**Root causes of traditional waste:**
1. Internal fragmentation -- pre-allocated buffers larger than needed
2. External fragmentation -- freed memory gaps cannot be reused
3. Redundant duplication -- parallel sequences maintain separate KV copies

**vLLM solution:** PagedAttention partitions KV cache into fixed-size blocks with on-demand allocation. Waste occurs only in the last block of each sequence.

**Real-world example:** KV cache for a single sequence in LLaMA-13B can consume up to 1.7 GB. For Llama-2 7B with 10,000 tokens, KV cache requires approximately 5 GB.

**Chart:** `../charts/memory_waste_comparison.png`

---

## 3. Community and Adoption Metrics (as of February 2026)

| Metric              | Value     |
|---------------------|-----------|
| GitHub Stars        | ~70,900   |
| Forks               | ~13,600   |
| Contributors        | 2,205     |
| Downstream Projects | 7,600+    |
| Latest Release      | v0.15.1 (February 4, 2026) |
| License             | Apache 2.0 |
| Blog Posts           | 45+       |

**Financial backing:** a16z, Sequoia Capital; compute sponsorship from AWS, Google Cloud, NVIDIA.

**Chart:** `../charts/community_adoption_metrics.png`

---

## 4. Latency Benchmarks

| Metric                                  | Value                | Context                              |
|-----------------------------------------|----------------------|--------------------------------------|
| TPOT improvement (v0.6.0 vs v0.5.3)    | 5x faster            | Llama 3 8B on H100                   |
| TPOT improvement (v0.6.0 vs v0.5.3)    | 2x faster            | Llama 3 70B on 4xH100               |
| Speculative decoding (draft model)      | Up to 1.5x speedup   | Llama 3 70B, ShareGPT, low QPS      |
| Speculative decoding (n-gram)           | Up to 2.8x speedup   | Llama 3 70B, CNN/DailyMail, low QPS |
| Saturation throughput                   | ~1,900 tokens/sec     | At QPS=8                             |

---

## 5. Architecture Performance Gains

| Architectural Feature           | Performance Impact                              |
|---------------------------------|-------------------------------------------------|
| Process separation (GIL fix)    | 2.7x throughput (8B), 1.8x (70B)               |
| Multi-step scheduling           | 28% throughput improvement (Llama 70B, 4xH100)  |
| Pipeline parallelism            | 6.6x on non-InfiniBand 16xH100 clusters         |
| Continuous batching             | Up to 23x vs. naive static batching              |

---

## 6. Real-World Deployment (LMSYS Chatbot Arena)

| Metric                | Value                     |
|-----------------------|---------------------------|
| Daily requests        | 30,000 average            |
| Peak daily requests   | 60,000                    |
| Share of Arena traffic| More than 50%             |
| GPU reduction         | 50% fewer GPUs needed     |

---

## 7. FP8 Quantization Performance (Llama 3.1 405B, 8xH100)

| Metric                | Value                     |
|-----------------------|---------------------------|
| Request throughput    | 2.82 requests/sec         |
| Input token throughput| 2,884.86 tokens/sec       |
| Output token throughput| 291.53 tokens/sec        |
| GSM8K accuracy (FP8)  | 95.38% (+/- 0.56%)       |
| GSM8K accuracy (BF16) | 96.8% (baseline)         |
| Accuracy loss         | ~1.4 percentage points    |

---

## 8. Hardware and Model Coverage

- **Hardware backends:** 10+ (NVIDIA CUDA, AMD ROCm, Intel XPU, Intel Gaudi, Google TPU, AWS Neuron, IBM Spyre, Huawei Ascend, Apple Silicon, ARM, PowerPC)
- **Model families supported:** 30+ architectures (Llama, Mistral, Mixtral, Qwen, DeepSeek, Phi, GPT variants, Falcon, Bloom, Gemma, and more)
- **Multimodal support:** Vision-language (LLaVA, Qwen-VL), audio-language, video-language models
- **Quantization methods:** 12+ (AutoAWQ, GPTQModel, FP8, INT8, INT4, BitsAndBytes, GGUF, AutoRound, TorchAO, LLM Compressor, NVIDIA Model Optimizer, AMD Quark, Quantized KV Cache)

---

## Charts Index

| Chart | File | Description |
|-------|------|-------------|
| 1 | `../charts/throughput_comparison.png` | Bar chart of vLLM throughput multiplier vs. HuggingFace Transformers, TGI, and FasterTransformer |
| 2 | `../charts/memory_waste_comparison.png` | Bar chart comparing KV cache memory waste between traditional systems (60-80%) and vLLM (<4%) |
| 3 | `../charts/community_adoption_metrics.png` | Horizontal bar chart of GitHub stars, forks, downstream projects, and contributors |
