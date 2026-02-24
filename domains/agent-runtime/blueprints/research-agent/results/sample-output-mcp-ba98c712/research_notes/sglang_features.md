# SGLang: Key Performance Features and Optimizations for LLM Inference

SGLang is a high-performance serving framework for large language models (LLMs) and multimodal models, developed by the LMSYS organization under the Apache-2.0 license. It is deployed across 400,000+ GPUs worldwide, processing trillions of tokens daily for production workloads at enterprises including xAI, AMD, NVIDIA, and LinkedIn.

This document covers its key performance features and architectural optimizations.

---

## 1. RadixAttention for KV Cache Reuse and Prefix Caching

### Overview

RadixAttention is SGLang's core innovation for automatic and efficient KV cache reuse across multiple LLM generation calls. Traditional serving systems discard KV cache tensors after each request completes; RadixAttention retains them in a persistent data structure for reuse by subsequent requests that share common prefixes.

### Data Structure: Radix Tree

- The KV cache is organized as a **radix tree** (also called a Patricia trie), a space-efficient alternative to traditional prefix trees.
- **Edges are labeled with token sequences** rather than single tokens, compressing the tree and reducing traversal overhead.
- The radix tree maps token sequences to GPU-stored KV cache tensors organized in a **paged format** (one token per page).
- The tree structure itself resides on the **CPU**, keeping maintenance overhead minimal while the actual KV tensors remain on GPU memory.

### Eviction Policy

- An **LRU (Least Recently Used) eviction policy** manages memory constraints.
- When GPU memory capacity is reached, the system **recursively removes leaf nodes** from the radix tree, freeing their associated KV cache pages.
- This approach automatically retains the most frequently and recently accessed prefixes.

### Cache-Aware Scheduling

- SGLang's scheduler is **cache-aware**: it can prioritize requests that have a higher cache hit rate in the radix tree, increasing the likelihood of prefix reuse.
- The default scheduling policy is **FCFS (first-come-first-served)**, but cache-aware scheduling can be used for workloads where prefix reuse is critical.
- Ablation studies showed **negligible overhead even when there are no cache hits**, so the radix tree structure imposes essentially zero cost in the worst case.

### Use Cases

RadixAttention automatically accelerates common patterns that benefit from prefix sharing:
- **Few-shot learning**: The shared few-shot examples prefix is cached.
- **Multi-turn chat**: Previous turns' KV cache is reused.
- **Self-consistency / majority-vote sampling**: Multiple completions of the same prompt share the prefix.
- **Tree-of-Thought search**: Branching search paths share common ancestors.
- **Retrieval-Augmented Generation (RAG)**: Shared system prompts and document contexts are cached.

### Performance

- RadixAttention enables **up to 5x higher throughput** compared to systems without prefix caching (benchmarked against Guidance and vLLM at the time of introduction).
- The feature can be disabled via the `--disable-radix-cache` server argument if not needed.

### Sources

- https://lmsys.org/blog/2024-01-17-sglang/
- https://arxiv.org/abs/2312.07104
- https://github.com/sgl-project/sglang

---

## 2. Compressed Finite State Machine for Faster Structured Output

### Overview

SGLang introduced a **Compressed Finite State Machine (FSM)** approach for constrained decoding, enabling structured outputs like JSON to be generated significantly faster than previous methods (e.g., Outlines).

Note: As of SGLang v0.4, the default structured output backend was upgraded to **xgrammar**, which achieves up to 10x faster JSON decoding. The compressed FSM approach laid the architectural groundwork.

### How It Works

1. **Schema-to-Regex Conversion**: JSON schemas (or other structural constraints) are transformed into regular expressions.
2. **FSM Construction**: The regex is compiled into a finite state machine that guides LLM generation token-by-token.
3. **Jump-Forward Decoding**: The key innovation -- the system identifies **singular transition paths** in the FSM (edges with only one outgoing transition) and **compresses consecutive deterministic transitions together**. Rather than decoding token-by-token through deterministic sequences, the system **directly prefills the compressed paths**, jumping forward to the next branching point.

### The Compression Algorithm

- The FSM is analyzed to locate edges where there is only one possible outgoing transition.
- Consecutive single-transition edges are concatenated into singular jump-forward paths.
- At each step, if the current FSM state has a single deterministic next string, that string is inserted directly without requiring the LLM to generate it.
- The system terminates the current request and enqueues a new one with the jumped-forward prefix. **RadixAttention automatically reuses the previous KV cache**, eliminating redundant computation.

### Tokenization Handling

- A critical challenge is tokenization boundary mismatches when inserting deterministic strings.
- The system implements **re-tokenization** during the jump-forward phase: it appends the deterministic string as raw text, then re-tokenizes the full text.
- This adds approximately **4% computational overhead** but resolves most boundary issues.

### Performance

- **Up to 2x lower latency** and **up to 2.5x higher throughput** compared to Outlines + vLLM (benchmarked on Llama-7B).
- The optimization can make **constrained decoding even faster than unconstrained normal decoding**, since deterministic portions are prefilled rather than generated autoregressively.
- With the xgrammar backend (v0.4+): **up to 10x faster JSON decoding**.

### Sources

- https://lmsys.org/blog/2024-02-05-compressed-fsm/
- https://lmsys.org/blog/2024-12-04-sglang-v0-4/

---

## 3. Continuous Batching and Scheduling Strategies

### Continuous Batching

SGLang implements **continuous batching** (also called iteration-level scheduling), where:
- New requests can be added to a running batch at each iteration step, rather than waiting for all requests in a batch to complete.
- Completed requests are immediately removed and new ones inserted, maximizing GPU utilization.
- This is combined with **paged attention** for efficient GPU memory management of variable-length KV caches.

### Zero-Overhead Batch Scheduler (v0.4+)

The scheduler in SGLang v0.4 achieves **zero overhead** by:
- **Overlapping CPU scheduling with GPU computation**: The scheduler runs **one batch ahead**, preparing all metadata required for the next batch while the current batch is executing on the GPU.
- This eliminates the scheduling gap between batches that other systems experience.
- Delivers a **1.1x throughput increase** over the previous SGLang scheduler and **1.3x speedup** compared to competing solutions.
- Gains are particularly strong on **smaller models** and configurations using **extensive tensor parallelism** (where GPU batch processing is fast relative to scheduling overhead).
- Activates automatically -- no configuration changes needed.

### Scheduling Policies

- **FCFS (First-Come-First-Served)**: The default scheduling policy (`--schedule-policy fcfs`).
- **Priority scheduling**: Enabled via `--enable-priority-scheduling`, allowing requests with higher priority to be scheduled first.
- **Schedule conservativeness**: A multiplier (`--schedule-conservativeness`, default 1.0) that controls how aggressively the scheduler fills batches, trading latency for throughput.

### Cache-Aware Load Balancer

- A standalone routing component (`sglang-router`) distributes requests across multiple SGLang worker instances.
- Maintains an **approximate radix tree** mirroring each worker's actual cache state.
- Uses a **communication-free design** implemented in Rust.
- Achieves up to **1.9x throughput improvement** and **3.8x higher cache hit rate**.
- Supports multi-node deployments with both Python bindings and CLI tools.

### Chunked Prefill

- Long prompt prefills are broken into **chunks** to avoid blocking the decode phase of other requests.
- Default chunk sizes are automatically tuned based on GPU tier: **2048 to 16384 tokens** depending on GPU memory capacity.
- This prevents long-prompt requests from causing latency spikes in concurrent shorter requests.

### Sources

- https://lmsys.org/blog/2024-12-04-sglang-v0-4/
- https://github.com/sgl-project/sglang

---

## 4. Tensor Parallelism and Model Parallelism Support

SGLang supports four forms of parallelism, configurable via server arguments:

### Tensor Parallelism (TP)

- **`--tp-size N`**: Shards model weights across N GPUs, with each GPU holding a slice of each layer.
- The primary parallelism mode for single-node multi-GPU setups.
- Used with FlashInfer's AllReduce fusion (`--enable-flashinfer-allreduce-fusion`) for optimized communication.

### Pipeline Parallelism (PP)

- **`--pp-size N`**: Distributes model layers across N pipeline stages (each stage on a different GPU or set of GPUs).
- Configurable micro-batch size via `--pp-max-micro-batch-size`.
- Useful for very large models that do not fit in a single GPU's memory even with tensor parallelism.

### Expert Parallelism (EP)

- **`--ep-size N`**: For Mixture-of-Experts (MoE) models, distributes experts across GPUs.
- Demonstrated scaling up to **96 H100 GPUs** for large-scale expert parallelism.
- Multiple MoE All-to-All backends: `deepep`, `mooncake`, `mori`, `ascend_fuseep`, `flashinfer`.

### Data Parallelism (DP)

- **`--dp-size N`**: Runs N independent copies of the model, distributing requests across them.
- For DeepSeek models, SGLang applies **data parallelism to the attention mechanism** (multi-head latent attention) rather than traditional tensor parallelism. This significantly reduces redundant KV cache storage and enables larger batch processing, delivering **1.9x decoding throughput improvement** on 8xH100 configurations.

### Context Parallelism (CP)

- **`--attn-cp-size N`**: Distributes attention computation across GPUs for very long sequences.

### Sources

- https://github.com/sgl-project/sglang
- https://lmsys.org/blog/2024-12-04-sglang-v0-4/

---

## 5. Supported Model Architectures

SGLang supports approximately **166 model configurations** spanning diverse architectures:

### Large Language Models

Llama (1/2/3/3.1/4), Qwen (1/1.5/2/2.5/3), DeepSeek (V1/V2/V3/R1), Mistral, Mixtral, Gemma (1/2/3), GPT-2, GPT-NeoX, Phi (1/2/3/4), ChatGLM/GLM-4, InternLM (1/2), OLMo, Falcon, StableLM, DBRX, Command-R, Baichuan, Solar, Grok, Nemotron, Kimi, MiMo, Persimmon, OPT.

### Mixture-of-Experts (MoE) Models

Mixtral, DeepSeek-V2/V3 MoE, Qwen MoE variants, GLM-4 MoE, DBRX.

### Multimodal / Vision-Language Models

LLaVA, LLaVA-ViD, LLaVA-OneVision, Qwen2-VL, DeepSeek-VL2, GLM-4V, InternVL, NVILA.

### Audio Models

Qwen2-Audio, Phi4MM-Audio, Gemma3N-Audio.

### Embedding Models

e5-mistral, gte, mcdse, Llama-Embedding.

### Reward / Classification Models

Skywork Reward, Llama-Reward, Gemma2-Reward, Llama-Classification, Qwen2-Classification.

### Diffusion Models

WAN, Qwen-Image, LLaDA.

### OCR Models

DeepSeek-OCR, GLM-OCR, LightonOCR.

### Vision Encoders

CLIP, SigLIP (used as components in VLM architectures).

### General Compatibility

- Most Hugging Face Transformers-compatible models are supported.
- OpenAI-compatible API for easy integration.
- `--trust-remote-code` flag for custom model architectures.

### Sources

- https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models

---

## 6. Unique Optimizations

### 6.1 FlashInfer Integration

SGLang deeply integrates with **FlashInfer**, a specialized GPU kernel library for LLM inference:

- **Attention backends**: FlashInfer provides high-performance PagedAttention kernels for both prefill and decode phases. SGLang auto-selects the best backend from: `fa3` (FlashAttention-3), `trtllm_mha`, `flashinfer`, or `triton`.
- **Sampling kernels**: FlashInfer provides GPU-optimized sampling (top-k, top-p, min-p) that SGLang uses as its default sampling backend.
- **Quantized GEMM**: FP4, FP8, and mixed-precision (FP8xFP4) matrix multiplication kernels.
- **Fused MoE kernels**: Specialized expert computation with block-scaled quantization.
- **Communication primitives**: AllReduce fusion and multi-node NVLink (MNNVL) utilities for distributed inference.
- **RoPE, RMSNorm, activation fusion**: Kernel-fused operations that reduce memory bandwidth bottlenecks.

Configurable via `--attention-backend`, `--sampling-backend`, `--prefill-attention-backend`, `--decode-attention-backend`.

### 6.2 torch.compile Integration

- SGLang integrates PyTorch 2.0's `torch.compile` for JIT-compiling **linear layers, normalization layers, and activation functions**.
- Enabled via `--enable-torch-compile`.
- Compilation is applied for **batch sizes 1-32**, yielding **up to 1.5x speedup**.
- Compatible with FlashInfer attention kernels, continuous batching, and RadixAttention prefix caching.
- CUDA graphs (`--cuda-graph-max-bs`) are used alongside torch.compile for further kernel launch overhead reduction, with piecewise CUDA graph support (`--enable-piecewise-cuda-graph`) for more flexible graph capture.

### 6.3 Chunked Prefill

- Long prompts are split into chunks to prevent blocking the decode phase of concurrent requests.
- Chunk sizes are auto-tuned per GPU tier (2048-16384 tokens).
- Configured via `--chunked-prefill-size`.

### 6.4 Prefill-Decode Disaggregation

- Separates the prefill (prompt processing) and decode (token generation) phases, allowing them to be scheduled and optimized independently.
- This is critical for production deployments where prefill latency and decode throughput have different optimization targets.

### 6.5 Speculative Decoding

SGLang supports multiple speculative decoding algorithms:
- **EAGLE**: Uses a separate draft model to propose multiple candidate tokens, which are then verified in parallel by the main model.
- **NGRAM**: Uses n-gram patterns from the prompt for speculative proposals (no separate model needed).
- **STANDALONE**: Custom standalone draft model.

Configuration:
- `--speculative-algorithm`: EAGLE, NGRAM, or STANDALONE
- `--speculative-num-steps`: Number of draft steps
- `--speculative-draft-model-path`: Path to draft model
- `--speculative-accept-threshold-single` / `--speculative-accept-threshold-acc`: Acceptance criteria

### 6.6 DeepSeek MLA Optimizations

Special optimizations for DeepSeek's Multi-Head Latent Attention (MLA) architecture:
- **Weight absorption**: Eliminates redundant projections.
- **Grouped decoding kernels**: Efficient batched decoding.
- **FP8 batched MatMul and FP8 KV cache quantization**: Reduces memory and compute for the attention mechanism.
- Achieves **3x to 7x higher throughput** than baseline systems on H100 GPUs.

### 6.7 CUDA Graph Capture

- CUDA graphs eliminate kernel launch overhead by recording and replaying GPU operations.
- Auto-sized based on GPU memory: `--cuda-graph-max-bs` ranges from 8 to 512 depending on GPU tier.
- Can be disabled via `--disable-cuda-graph` for debugging.

### 6.8 Multi-LoRA Batching

- Serves multiple LoRA adapters concurrently within a single model instance.
- Requests targeting different LoRA adapters can be batched together, with the base model weights shared and adapter-specific computations applied per-request.

### Sources

- https://lmsys.org/blog/2024-09-04-sglang-v0-3/
- https://lmsys.org/blog/2024-12-04-sglang-v0-4/
- https://flashinfer.ai/
- https://docs.flashinfer.ai/
- https://github.com/sgl-project/sglang

---

## 7. Quantization Support

SGLang supports multiple quantization methods for reduced memory footprint and faster inference:

| Method | Precision | Description |
|--------|-----------|-------------|
| **FP8** | 8-bit float | E4M3 format; supported for both weights and KV cache (`--kv-cache-dtype fp8_e4m3`) |
| **FP4** | 4-bit float | Ultra-low precision with FlashInfer FP4 GEMM kernels |
| **INT4** | 4-bit integer | General 4-bit integer quantization |
| **AWQ** | 4-bit | Activation-aware Weight Quantization; pre-quantized models |
| **GPTQ** | 4-bit | Post-training quantization with calibration data |
| **BitsAndBytes** | 4/8-bit | HuggingFace bitsandbytes integration for dynamic quantization |
| **GGUF** | Various | Supports GGUF format models (popularized by llama.cpp) |

Configuration:
- `--quantization {awq, fp8, gptq, bitsandbytes, gguf, ...}`
- `--kv-cache-dtype {auto, fp8_e4m3, bfloat16}`
- `--dtype {auto, float16, bfloat16, float32}`

### Sources

- https://github.com/sgl-project/sglang

---

## 8. Hardware Support

| Platform | Specific Hardware |
|----------|-------------------|
| **NVIDIA** | GB200, B300, H100, A100, A10G, and others |
| **AMD** | MI355, MI300 |
| **Intel** | Xeon CPUs |
| **Google** | TPUs (via SGLang-Jax backend) |
| **Huawei** | Ascend NPUs |

Notable benchmarks:
- **GB200 NVL72**: 3.8x prefill throughput and 4.8x decode throughput.
- **8xA10G**: Mixtral-8x7B with tensor parallelism (FP16).
- **8xH100**: DeepSeek models with data parallelism for attention.
- **96xH100**: Large-scale expert parallelism for MoE models.

### Sources

- https://github.com/sgl-project/sglang

---

## 9. Frontend Language Features

SGLang also provides a Python-embedded **domain-specific language (DSL)** for programming LLM interactions:

- **`gen()`**: Non-blocking LLM generation call.
- **`fork()`**: Creates parallel copies of a prompt for branching generation.
- **`choices()`**: Constrains generation to a set of options.
- **Interpreter mode**: Eager execution of SGLang programs.
- **Compiler mode**: Builds a dataflow graph with optimization opportunities including code movement and auto-tuning.

This frontend is what gives SGLang its name ("Structured Generation Language") and enables automatic exploitation of RadixAttention caching patterns in complex multi-call programs.

### Sources

- https://lmsys.org/blog/2024-01-17-sglang/
- https://arxiv.org/abs/2312.07104

---

## 10. Performance Summary

| Benchmark / Feature | Speedup | Baseline |
|---------------------|---------|----------|
| RadixAttention prefix caching | Up to 5x throughput | Systems without prefix caching |
| Compressed FSM structured output | 2-2.5x throughput | Outlines + vLLM |
| xgrammar structured output (v0.4) | Up to 10x faster JSON | Previous FSM backend |
| torch.compile | Up to 1.5x speedup | Without compilation |
| DeepSeek MLA optimizations | 3-7x throughput | Baseline systems |
| Zero-overhead scheduler | 1.3x throughput | Competing schedulers |
| Cache-aware load balancer | 1.9x throughput | Without cache-aware routing |
| LLaVA-OneVision | 4.5x speedup | HuggingFace Transformers |
| GB200 NVL72 | 3.8x prefill, 4.8x decode | Previous generation |
| Overall (paper, diverse workloads) | Up to 6.4x throughput | State-of-the-art systems |

---

## Key Sources

1. **SGLang GitHub Repository**: https://github.com/sgl-project/sglang
2. **SGLang Paper (arXiv)**: https://arxiv.org/abs/2312.07104
3. **LMSYS Blog - SGLang Introduction**: https://lmsys.org/blog/2024-01-17-sglang/
4. **LMSYS Blog - Compressed FSM**: https://lmsys.org/blog/2024-02-05-compressed-fsm/
5. **LMSYS Blog - SGLang v0.3**: https://lmsys.org/blog/2024-09-04-sglang-v0-3/
6. **LMSYS Blog - SGLang v0.4**: https://lmsys.org/blog/2024-12-04-sglang-v0-4/
7. **SGLang Documentation**: https://docs.sglang.io/
8. **FlashInfer**: https://flashinfer.ai/ and https://docs.flashinfer.ai/
