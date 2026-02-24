# NVIDIA TensorRT-LLM: Core Features and Architecture

## 1. Overview

NVIDIA TensorRT-LLM is an open-source library designed to optimize Large Language Model (LLM) inference on NVIDIA GPUs. It provides a high-level Python API for defining, optimizing, and executing LLMs with state-of-the-art performance. The library is architected on PyTorch and wraps NVIDIA's TensorRT deep learning compiler, incorporating optimized kernels---including FlashAttention implementations and masked multi-head attention kernels---specifically engineered for LLM workloads.

TensorRT-LLM supports deployment scenarios ranging from single-GPU setups to multi-GPU and multi-node configurations. It integrates into the broader NVIDIA inference ecosystem, including NVIDIA Dynamo and NVIDIA Triton Inference Server.

**Current Version (as of early 2026):** 1.3.x series, supporting Python 3.10--3.12, CUDA 13.1, and PyTorch 2.9.

**Supported GPU Architectures:** NVIDIA Ampere (A100), Ada Lovelace (L40S), Hopper (H100/H200), and Blackwell (B200).

---

## 2. Core Architecture

### 2.1 High-Level Design

TensorRT-LLM's architecture centers on three major stages:

1. **Model Definition**: Models are defined using a PyTorch-native, modular Python API. Pre-defined models for 40+ architectures are available (Llama, DeepSeek, Mixtral, Falcon, GPT-J, Qwen, Mistral, Baichuan, StarCoder, and many more), or users can define custom models using native PyTorch code.

2. **Engine Compilation**: The TensorRT compiler performs graph-level optimizations---selecting the best kernel for each operation and available GPU, identifying fusion opportunities to reduce memory movement and kernel launch overhead, and compiling operations into optimized CUDA execution plans. Explicit plugins handle complex fusions that cannot be automatically discovered (e.g., the `gpt_attention` plugin for FlashAttention-like fused attention, the `gemm` plugin for FP32-accumulation matrix multiplication).

3. **Runtime Execution**: The compiled TensorRT engine is executed through the runtime system, which manages request scheduling, KV cache allocation, token sampling, and result delivery.

### 2.2 The LLM Class Entry Point

The primary user-facing abstraction is the `LLM` class, which provides a simplified `generate()` interface. This class handles:

- Tokenization (encoding input prompts into numerical representations)
- Model loading and optimization
- Inference execution
- Detokenization (converting output tokens back to text)

```python
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
sampling_params = SamplingParams(temperature=0.7, top_p=0.9)

prompts = ["What is deep learning?", "Explain quantum computing."]
for output in llm.generate(prompts, sampling_params):
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

### 2.3 PyExecutor Worker Architecture

Internally, TensorRT-LLM creates a dedicated `PyExecutor(Worker)` process per rank that runs in a continuous background loop for asynchronous inference processing. The PyExecutor contains four primary components:

| Component | Responsibility |
|---|---|
| **Scheduler** | Determines which active requests proceed at each processing step |
| **KVCacheManager** | Manages allocation and maintenance of the Key-Value cache, storing previously computed attention keys and values for autoregressive generation |
| **ModelEngine** | Loads and executes the language model on GPU hardware |
| **Sampler** | Applies sampling strategies (greedy, top-k, top-p, beam search) to convert logits into output tokens |

**Execution Loop Sequence:**
1. Fetch requests from internal queue
2. Schedule ready requests via the Scheduler
3. Allocate KV cache resources via KVCacheManager
4. Execute forward model pass for next-token prediction via ModelEngine
5. Apply sampling strategy via Sampler
6. Finalize and return outputs for completed requests

---

## 3. Quantization Support

TensorRT-LLM provides extensive quantization support to reduce model memory footprint and accelerate inference. Quantization methods fall into several categories:

### 3.1 Floating-Point Precision Modes

| Format | Description | Hardware Requirement |
|---|---|---|
| **FP32** | Full precision; baseline reference | All GPUs |
| **FP16** | Half precision; 2x memory reduction | All supported GPUs |
| **BF16** | Brain floating point; better dynamic range than FP16 | Ampere and later |

### 3.2 Weight-Only Quantization (W4A16 / W8A16)

- **INT4 Weight-Only**: Quantizes weights to 4-bit integers; activations remain in FP16/BF16. Achieves approximately 4x memory reduction for weights.
- **INT8 Weight-Only**: Quantizes weights to 8-bit integers; activations remain in FP16/BF16. Achieves approximately 2x memory reduction for weights.
- Weights are dequantized on-the-fly during linear layer execution.

### 3.3 Advanced Quantization Techniques

| Method | Precision | Description |
|---|---|---|
| **SmoothQuant** | W8A8 (INT8) | Quantizes both weights and activations to INT8 while maintaining accuracy by mathematically "smoothing" activation outliers into weights before quantization |
| **AWQ (Activation-Aware Weight Quantization)** | W4A16 | Per-group scaling factors with zero-offsetting for improved 4-bit weight compression; preserves salient weights identified through activation analysis |
| **GPTQ** | W4A16 | Per-group quantization with learned scaling factors; uses second-order optimization (approximate Hessian) for calibration |
| **FP8 (E4M3)** | W8A8 | Native 8-bit floating-point format on Hopper GPUs; retains higher accuracy than INT8/INT4 while achieving fastest performance through hardware Transformer Engine support |
| **NVFP4** | W4A4 | Blackwell-specific 4-bit floating-point datatype; supported for Llama, Mixtral, and other models |

### 3.4 Scaling Modes

TensorRT-LLM implements three quantization scaling approaches:

- **Per-tensor**: A single scaling factor for the entire tensor
- **Per-token**: M scaling factors for M tokens (row-wise)
- **Per-channel**: N scaling factors for N output channels (column-wise)

Per-token and per-channel modes can be combined (e.g., for SmoothQuant W8A8) to achieve finer-grained quantization with better accuracy preservation.

### 3.5 FP8 Automatic Quantization

On Hopper and later GPUs, TensorRT-LLM can automatically convert model weights to FP8 format and compile models to use optimized FP8 kernels without modifying model code. This leverages the hardware Transformer Engine for maximum throughput.

---

## 4. In-Flight Batching (Continuous Batching)

### 4.1 The Problem with Static Batching

Traditional static batching groups a fixed number of requests into a batch and processes them together. Because LLM outputs vary widely in length, some sequences finish much earlier than others. Under static batching, completed sequences remain idle (wasting GPU cycles) until the longest sequence in the batch finishes. This leads to poor GPU utilization and throughput degradation.

### 4.2 How In-Flight Batching Works

TensorRT-LLM implements **in-flight batching** (also called continuous batching or iteration-level scheduling):

1. **Immediate Eviction**: As soon as a sequence completes generation, it is immediately evicted from the active batch.
2. **Dynamic Insertion**: New requests from the waiting queue are inserted into the batch at any iteration, filling the slot vacated by the completed sequence.
3. **Iteration-Level Scheduling**: The scheduler makes decisions at every forward-pass iteration rather than at the batch level.

This approach ensures the GPU is always processing the maximum number of active sequences, eliminating idle compute cycles caused by variable-length outputs.

### 4.3 Performance Impact

In-flight batching has been shown to **at minimum double the throughput** on benchmarks of real-world LLM requests on NVIDIA H100 GPUs compared to static batching, with even greater improvements at higher concurrency levels.

### 4.4 Context and Generation Phases

TensorRT-LLM distinguishes between two phases of request processing:

- **Context phase (prefill)**: Processing the input prompt tokens. This is compute-intensive and processes all input tokens in parallel.
- **Generation phase (decode)**: Autoregressive token generation. This is memory-bandwidth-bound and generates one token per iteration.

The scheduler can interleave context and generation phases from different requests (chunked context processing), further improving utilization.

---

## 5. Paged KV Caching

### 5.1 The KV Cache Challenge

During autoregressive generation, Transformer models compute attention over all previously generated tokens. To avoid recomputation, the Key and Value tensors from each attention layer are cached (the "KV cache"). For large models with long sequences, this cache consumes substantial GPU memory---often exceeding the memory required by the model weights themselves.

### 5.2 How Paged KV Caching Works

TensorRT-LLM implements paged KV caching, inspired by virtual memory paging in operating systems:

- The KV cache memory is divided into fixed-size **blocks** (pages).
- Each request's KV cache is allocated in non-contiguous blocks as needed.
- Blocks are allocated on demand as sequences grow and freed immediately when sequences complete.
- A block table maps logical token positions to physical memory blocks.

This approach eliminates the need to pre-allocate contiguous memory for the maximum possible sequence length, dramatically reducing memory waste from fragmentation.

### 5.3 KV Cache Configuration

The KV cache is configured through `KVCacheConfig`:

- **`maxTokens`**: Explicit upper limit on the number of tokens the cache can hold.
- **`freeGpuMemoryFraction`**: Fraction of available GPU memory to allocate for KV cache (default: 90%).
- When both parameters are specified, the minimum computed value is used.

### 5.4 Advanced KV Cache Features

| Feature | Description |
|---|---|
| **KV Cache Reuse Across Requests** | Shared prompt prefixes (e.g., system prompts) can share KV cache blocks across multiple requests, avoiding redundant computation |
| **Limited Attention Window** | Support for sliding-window attention models that only attend to recent tokens, automatically freeing old KV cache blocks |
| **KV Cache Offloading** | Ability to offload KV cache blocks to CPU memory when GPU memory is constrained, swapping them back as needed |

---

## 6. Memory Optimization Techniques

### 6.1 GPU Memory Components

TensorRT-LLM GPU memory consumption consists of three primary components:

1. **Model Weights**: Fixed size determined by model dimensions, precision level, and parallelization strategy. Lower-precision formats (INT4, INT8, FP8) directly reduce this footprint.
2. **Activation Tensors**: Pre-computed at engine build time by TensorRT using optimized graph analysis. The compiler determines the optimal memory layout and reuse strategy.
3. **I/O Tensors (KV Cache)**: The largest variable component, driven by batch size, sequence lengths, and number of active requests.

### 6.2 Specific Optimization Techniques

**Reducing Activation Memory:**
- **Decrease `max_num_tokens`**: Most transformer activation tensors scale linearly with input token count. Reducing this parameter lowers peak activation memory.
- **Enable Context FMHA (Fused Multi-Head Attention)**: Enabling `context_fmha_type` significantly reduces memory footprint by fusing attention operations and avoiding materialization of the full attention matrix.
- **Packed Tensors Format**: Uses variable-length tensor packing instead of padding to a fixed maximum length, conserving both memory and compute.

**Reducing Weight Memory:**
- Apply quantization (FP8, INT8, INT4, AWQ, GPTQ) to reduce weight storage by 2--8x.
- Use tensor parallelism to distribute weights across GPUs.

**Reducing KV Cache Memory:**
- Paged allocation eliminates fragmentation waste.
- Group-Query Attention (GQA) and Multi-Query Attention (MQA) architectures inherently require less KV cache than Multi-Head Attention (MHA).
- KV cache offloading to CPU memory extends effective capacity.

### 6.3 Memory Pool Management

TensorRT-LLM uses a **stream-ordered memory allocator** for buffer management. Memory is allocated from and released back to a CUDA memory pool. Note that `nvidia-smi` may still display high memory occupation after deallocation---this reflects CUDA driver memory pool behavior (memory is held in the pool for fast reallocation rather than returned to the OS) and is expected.

### 6.4 Runtime Optimizations

**CUDA Graphs**: Reduce CPU overhead by capturing GPU kernel sequences as single executable graphs. CUDA Graph padding matches batch sizes to cached graphs, demonstrating up to a **22% end-to-end throughput increase** by eliminating per-iteration kernel launch overhead.

**Overlap Scheduler**: Hides CPU scheduling latency behind GPU computation by launching the next iteration's GPU work immediately without waiting for the current iteration's CPU post-processing to complete.

---

## 7. Tensor Parallelism and Pipeline Parallelism

### 7.1 Tensor Parallelism (TP)

Tensor parallelism splits individual weight matrices across multiple GPUs. Each GPU holds a slice of every layer's weights and computes a portion of each operation. Results are combined through collective communication operations (all-reduce or all-gather) at synchronization points.

**Key Characteristics:**
- Every GPU participates in every layer's computation.
- Requires high-bandwidth interconnect (NVLink preferred) due to frequent inter-GPU communication.
- Provides excellent memory efficiency: weight memory, activation memory, and KV cache are all distributed.
- Ideal for scaling within a single node (e.g., 2, 4, or 8 GPUs connected via NVLink).
- TensorRT-LLM typically offers superior memory efficiency with tensor parallelism vs. pipeline parallelism because layers execute sequentially, allowing memory reuse across layers.

**Usage:**
```python
llm = LLM(model="meta-llama/Llama-3.1-70B-Instruct", tensor_parallel_size=4)
```

### 7.2 Pipeline Parallelism (PP)

Pipeline parallelism assigns different layers of the model to different GPUs. Each GPU processes a contiguous block of transformer layers, passing intermediate activations to the next GPU in the pipeline.

**Key Characteristics:**
- Each GPU only stores and computes its assigned layers.
- Less inter-GPU communication than TP (only at pipeline stage boundaries).
- Can introduce pipeline bubbles (idle time) unless micro-batching is used.
- Useful for scaling across nodes where inter-node bandwidth is lower.
- Can be combined with tensor parallelism (e.g., TP=4, PP=2 across 8 GPUs on 2 nodes).

### 7.3 Expert Parallelism (EP)

For Mixture-of-Experts (MoE) models (e.g., Mixtral, DeepSeek), TensorRT-LLM supports expert parallelism, which distributes different experts across different GPUs. This includes "Wide Expert Parallelism" for configurations where the number of GPUs exceeds the number of experts. Expert parallelism can be combined with tensor and pipeline parallelism.

### 7.4 Helix Parallelism

A more advanced parallelism strategy documented in TensorRT-LLM for distributed execution across heterogeneous or complex topologies.

### 7.5 Multi-GPU and Multi-Node Configuration

TensorRT-LLM provides built-in support for multi-GPU, multi-node (MGMN) inference:

- **Single-node multi-GPU**: Configure via `tensor_parallel_size` and/or `pipeline_parallel_size` parameters. No `mpirun` prefix needed.
- **Multi-node**: Uses MPI for inter-node communication. On Slurm systems: `mpirun -n 1 --oversubscribe --allow-run-as-root python script.py`.
- The framework handles weight distribution, communication patterns, and synchronization automatically.

---

## 8. The Python API

### 8.1 Core API Components

**`LLM` Class**: The primary interface for all inference workflows.

```python
from tensorrt_llm import LLM, SamplingParams

# Initialize from HuggingFace Hub (automatic download)
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Or from a local checkpoint path
llm = LLM(model="/path/to/local/model")

# Or from pre-quantized NVIDIA checkpoints (FP4/FP8)
llm = LLM(model="nvidia/Llama-3.1-8B-Instruct-FP8")
```

**`SamplingParams` Class**: Controls text generation behavior.

```python
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    max_tokens=256,
)
```

### 8.2 Synchronous Generation

```python
prompts = ["Explain relativity", "What is a neural network?"]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

### 8.3 Asynchronous and Streaming Generation

The API supports async generation and streaming for real-time applications:

```python
# Async generation
async for output in llm.generate_async(prompts, sampling_params):
    print(output.outputs[0].text)

# Streaming
for output in llm.generate(prompts, sampling_params, streaming=True):
    print(output.outputs[0].text, end="", flush=True)
```

### 8.4 Distributed Inference via API

```python
# Multi-GPU tensor parallelism
llm = LLM(model="meta-llama/Llama-3.1-70B-Instruct",
           tensor_parallel_size=4)

# Combined parallelism
llm = LLM(model="meta-llama/Llama-3.1-405B-Instruct",
           tensor_parallel_size=4,
           pipeline_parallel_size=2)
```

### 8.5 Additional API Capabilities

- **LoRA Adapter Support**: Load and switch between LoRA adapters at runtime.
- **Multimodal Inputs**: Process text + image inputs for vision-language models.
- **Speculative Decoding**: Enable draft-model or other speculative decoding methods for lower latency.

### 8.6 Serving via OpenAI-Compatible API

TensorRT-LLM includes `trtllm-serve`, a built-in server that exposes OpenAI-compatible endpoints:

```bash
trtllm-serve "meta-llama/Llama-3.1-8B-Instruct"
```

This launches a server with `v1/chat/completions` and other standard OpenAI endpoints for HTTP-based inference.

### 8.7 CLI Tools

| Tool | Purpose |
|---|---|
| `trtllm-serve` | Launch an OpenAI-compatible serving endpoint |
| `trtllm-bench` | Benchmark inference performance |
| `trtllm-eval` | Evaluate model quality on standard benchmarks |

---

## 9. Speculative Decoding

TensorRT-LLM supports multiple speculative decoding methods to reduce per-token latency, especially effective when GPU utilization is low due to small batch sizes. The core idea: predict multiple draft tokens using an efficient method, then validate them through the target LLM in a single forward pass.

### 9.1 Supported Methods

| Method | Approach | Best For |
|---|---|---|
| **Draft-Target Model** | Separate smaller draft model generates candidates; target model validates | General-purpose; any task |
| **N-Gram** | Copies tokens from input prompt/previous output as draft tokens | Summarization, document QA, multi-turn chat, code editing (high n-gram overlap) |
| **Medusa** | Multiple language model heads predict future tokens in a tree structure | Tasks with predictable continuations |
| **EAGLE (v1 & v2)** | Single-layer transformer predicts drafts from hidden states and decoded tokens | General-purpose with moderate overhead |
| **EAGLE3** | Newest EAGLE variant with improved acceptance rates | Latest iteration of EAGLE approach |
| **ReDrafter** | Recurrent predictor with beam search inside the TensorRT engine | Diverse draft generation |
| **Lookahead Decoding** | Parallel lookahead and verification branches; no additional training needed | Zero-cost setup (no draft model training) |
| **MTP (Multi-Token Prediction)** | Multiple prediction heads trained jointly | Models specifically trained with MTP objectives |

---

## 10. Attention Mechanism Support

TensorRT-LLM provides optimized implementations for all major attention variants:

- **Multi-Head Attention (MHA)**: Standard Transformer attention with separate Q, K, V projections per head.
- **Multi-Query Attention (MQA)**: All heads share a single set of K, V projections (e.g., Falcon, PaLM).
- **Group-Query Attention (GQA)**: K, V projections are shared within groups of heads (e.g., Llama 2 70B, Mixtral). Reduces KV cache size proportionally.

### 10.1 Attention Backends

TensorRT-LLM includes multiple attention backends:

- **FlashAttention-like fused kernels** via the `gpt_attention` plugin: Fuses Q*K, scaling, masking, softmax, and attention*V into a single kernel pass, avoiding materialization of the O(N^2) attention matrix.
- **XQA (Cross-Query Attention) kernels**: Specialized high-performance attention kernels for specific attention patterns.
- **Custom attention backends**: Configurable per deployment for optimal performance on different GPU architectures.

---

## 11. Integration with NVIDIA Triton Inference Server

### 11.1 Architecture

The TensorRT-LLM backend for Triton Inference Server enables production deployment of optimized LLMs. It implements a C++ backend and chains three model components:

1. **Preprocessing Model**: Tokenizes text inputs (strings to token IDs).
2. **TensorRT-LLM Model**: Executes inference on compiled TensorRT-LLM engines.
3. **Postprocessing Model**: Detokenizes outputs (token IDs back to text).

These components are orchestrated via either **ensemble models** (declarative pipeline) or **Business Logic Scripting (BLS)** (programmatic control).

### 11.2 Key Features of the Triton Backend

- In-flight batching with paged KV cache
- Speculative decoding (Top-k, Top-p, beam search, Medusa, EAGLE)
- Tensor, pipeline, and expert parallelism
- LoRA adapter support at runtime
- Chunked context processing
- Quantization (INT8, FP8, INT4)
- Multi-node deployments
- MIG (Multi-Instance GPU) support

### 11.3 Multi-GPU Execution Modes

| Mode | Description |
|---|---|
| **Leader Mode** | One Triton process per GPU; rank-0 handles requests while others remain dormant waiting for work |
| **Orchestrator Mode** | Spawns worker processes via MPI for more flexible GPU coordination |

### 11.4 Deployment Workflow

1. **Build TensorRT-LLM Engines**: Use `trtllm-build` to compile model weights into optimized TensorRT engines.
2. **Prepare Model Repository**: Copy model configurations from `all_models/inflight_batcher_llm` template directory.
3. **Configure Models**: Use `fill_template.py` to populate model settings with engine paths, tokenizer directories, and batching parameters.
4. **Launch Server**: Execute `launch_triton_server.py` with the desired GPU count (`--world_size`).

```bash
# Example deployment
python3 scripts/launch_triton_server.py --world_size 4 --model_repo ./model_repo
```

### 11.5 Client Interfaces

- **HTTP Generate Endpoint**: RESTful API accepting text input, token limits, and stop/bad word lists.
- **Python Client SDK**: `inflight_batcher_llm_client.py` for programmatic access.
- **Batch Support**: Handles multi-request batches with indexed responses via `batch_index` output tensor.

### 11.6 Tested Model Architectures

The Triton backend is officially tested with: Llama, Gemma, Mistral, Mixtral, and multimodal architectures (BLIP2-OPT, LLaVA). Encoder-decoder models (e.g., T5) and speech models (Whisper) are also supported.

---

## 12. Performance Benchmarks and Results

### 12.1 Reported Performance Gains

| Benchmark | Hardware | Result |
|---|---|---|
| GPT-J-6B text summarization | H100 vs A100 | **8x speedup** (4x from hardware + 2x from TensorRT-LLM optimizations) |
| GPT-J-6B TCO reduction | H100 + TensorRT-LLM | **5.3x cost reduction** |
| GPT-J-6B energy efficiency | H100 + TensorRT-LLM | **5.6x energy reduction** |
| Llama 2 70B | H100 vs A100 | **4.6x performance gain** |
| Llama 2 70B TCO reduction | H100 + TensorRT-LLM | **3x cost reduction** |
| In-flight batching throughput | H100 | **>2x throughput** vs static batching |
| CUDA Graphs optimization | General | **Up to 22%** end-to-end throughput increase |
| Peak throughput (advanced hardware) | Latest GPUs | **>40,000 tokens/second** |

### 12.2 Benchmarking Tool

TensorRT-LLM includes `trtllm-bench` for measuring:
- Throughput (tokens/second)
- Latency (time-to-first-token, inter-token latency)
- Memory utilization
- Scaling efficiency across GPUs

---

## 13. Supported Model Architectures

TensorRT-LLM supports 40+ model architectures including (non-exhaustive):

- **Meta**: Llama 1/2/3, Llama 3.1, Code Llama
- **Mistral AI**: Mistral, Mixtral (MoE)
- **Alibaba**: Qwen, Qwen2
- **Google**: Gemma
- **DeepSeek**: DeepSeek, DeepSeek-V2/V3 (MoE)
- **Microsoft**: Phi-1/2/3
- **BigCode**: StarCoder
- **Falcon**: Falcon-7B/40B/180B
- **Others**: GPT-2, GPT-J, GPT-NeoX, MPT, Baichuan, ChatGLM, BLOOM, OPT, Mamba, DBRX, Arctic, and more
- **Multimodal**: BLIP2-OPT, LLaVA, and other vision-language models
- **Encoder-Decoder**: T5, BART
- **Speech**: Whisper

---

## 14. Summary of Key Technical Differentiators

| Feature | Benefit |
|---|---|
| TensorRT compiler integration | Automatic kernel selection, fusion, and graph optimization |
| In-flight batching | >2x throughput improvement through continuous request scheduling |
| Paged KV caching | Eliminates memory fragmentation; enables larger batch sizes |
| Comprehensive quantization (FP8/FP4/INT8/INT4/AWQ/GPTQ/SmoothQuant) | 2--8x memory reduction with minimal accuracy loss |
| Tensor + Pipeline + Expert parallelism | Scales from single GPU to multi-node clusters |
| CUDA Graphs | Up to 22% throughput gain from reduced CPU overhead |
| Overlap Scheduler | Hides CPU latency behind GPU computation |
| Speculative decoding (7+ methods) | Reduced per-token latency for latency-sensitive workloads |
| Triton Inference Server backend | Production-grade serving with monitoring, scaling, and enterprise features |
| PyTorch-native design | Familiar API; easy customization and experimentation |
| OpenAI-compatible serving | Drop-in replacement for existing OpenAI API integrations |

---

*Research compiled on 2026-02-22. Sources: NVIDIA TensorRT-LLM GitHub repository, official documentation (nvidia.github.io/TensorRT-LLM), NVIDIA Developer Blog, and Triton Inference Server TensorRT-LLM backend repository.*
