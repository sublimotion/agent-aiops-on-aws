# SGLang: Architecture and Design Research Notes

## 1. What Is SGLang

### Overview

SGLang (Structured Generation Language) is a high-performance open-source serving framework for large language models (LLMs) and multimodal models. It is designed for low-latency, high-throughput inference across deployments ranging from single GPUs to large distributed clusters. The project's full name reflects its dual emphasis: efficient *serving* of LLMs and a *language* for expressing structured generation programs.

### Origin and Creators

SGLang originated from academic research at UC Berkeley and is hosted under the **LMSYS** (Large Model Systems Organization) umbrella, a non-profit open-source organization known for maintaining Chatbot Arena. The foundational paper is:

> Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al. "SGLang: Efficient Execution of Structured Language Model Programs." arXiv:2312.07104 (2023).

Key authors include **Lianmin Zheng** (also a creator of vLLM), **Liangsheng Yin**, and **Zhiqiang Xie**, with contributions from researchers at UC Berkeley, Stanford, and other major institutions. The project has grown to over **9,893 commits** and **1,182+ contributors** as of early 2025.

### Open-Source Status

SGLang is fully open source, available on GitHub at `sgl-project/sglang` and installable via PyPI as the `sglang` package. It also maintains a separate `sgl-kernel` package for low-level CUDA/hardware kernel optimizations. Documentation is hosted at `docs.sglang.io`.

### Industry Adoption

SGLang powers over **400,000 GPUs globally**, generating trillions of tokens daily. Notable adopters include xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, MIT, Stanford, UC Berkeley, and Tsinghua University.

---

## 2. Core Architecture

SGLang is a **co-designed system** with two major components:

1. **SGLang Runtime (SRT)** -- the backend inference engine
2. **SGLang Frontend Language** -- a Python-embedded domain-specific language for structured LLM programs

This co-design is the central thesis of the project: that optimizing the runtime and the programming interface *together* produces better results than optimizing either in isolation.

### High-Level Data Flow

```
User Request
    |
    v
[TokenizerManager]  (main process)
    |  tokenize input, validate, route
    v  (ZMQ IPC)
[Scheduler]          (subprocess)
    |  batch scheduling, KV cache mgmt, RadixAttention
    |  forward pass dispatch
    v
[ModelRunner]        (GPU execution)
    |  model forward, attention, sampling
    v  (ZMQ IPC)
[DetokenizerManager] (subprocess)
    |  incremental token-to-text decoding
    v
User Response
```

### Three-Process Architecture

The SGLang Runtime engine (`Engine`) launches three core process types that communicate via **ZMQ sockets** for RPC and **multiprocessing pipes** for initialization handshakes:

1. **TokenizerManager** -- Runs in the main process. Receives raw text requests, tokenizes them (with support for async dynamic batching of tokenization itself), validates inputs (e.g., LoRA configs, max length), and routes tokenized requests to the scheduler via ZMQ. Maintains `ReqState` objects tracking completion status, timing metrics, and incremental output.

2. **Scheduler** -- Runs as a subprocess. The brain of the system. Manages a waiting queue and running batch for continuous batching. Performs prefill/decode scheduling, KV cache allocation via RadixCache, memory-pressure-driven request retraction, and dispatches forward passes to the ModelRunner. Supports speculative decoding with draft model workers.

3. **DetokenizerManager** -- Runs as a subprocess. Receives batches of token IDs from the scheduler, performs incremental grouped decoding (batching by tokenizer configuration), handles stop-sequence trimming, and sends decoded text back through the TokenizerManager to the user. Maintains a `LimitedCapacityDict` (default capacity 65,536) for active decode states.

---

## 3. The SGLang Runtime (SRT) Backend

### Scheduler Design

The scheduler is the most architecturally significant component. Key design elements:

**Continuous Batching**: Requests do not wait for an entire batch to complete. New requests can join a running batch at any decode step, and completed requests are immediately removed, maximizing GPU utilization.

**Two-Phase Execution**:
- **Prefill (EXTEND mode)**: Processes all input tokens for new requests. A `PrefillAdder` determines which queued requests fit within the current token budget and available KV cache space.
- **Decode mode**: Generates one token per step for all active sequences in the batch.

**Priority-Based Scheduling**: The `SchedulePolicy` calculates priorities for queued requests with configurable preemption thresholds. Under memory pressure, the scheduler can retract decode requests back to the waiting queue, dynamically adjusting `token_ratio` with a decay function: `new_token_ratio = max(new_token_ratio - decay, minimum)`.

**Zero-Overhead CPU Scheduling**: Batch management logic runs on CPU without introducing scheduling bottlenecks. When overlap mode is enabled, separate CUDA streams (forward, copy, default) allow CPU scheduling to overlap with GPU computation. A `FutureMap` manages asynchronous tensor resolution for this pipeline.

**Chunked Prefill**: Long input sequences are processed in manageable chunks rather than all at once, reducing latency spikes and allowing decode steps to interleave with prefill processing.

### RadixAttention and KV Cache Management

RadixAttention is SGLang's signature innovation -- an automatic KV cache reuse system based on a **radix tree** (compressed prefix tree) data structure.

**Core Concept**: Instead of discarding KV cache tensors after each request completes (the standard approach), SGLang retains them in a radix tree where edges represent token sequences. When a new request shares a prefix with a cached sequence, the corresponding KV tensors are reused without recomputation.

**Data Structure Details**:
- `TreeNode` objects store `token_ids`, KV `value` tensors, `lock_ref` (reference counter), `priority`, and SHA256 `hash_value` for integrity.
- Nodes maintain parent-child relationships for hierarchical traversal.
- Page-aligned matching for hardware efficiency (`page_size` configurable).
- Node splitting occurs when a match ends mid-node, creating precise prefix boundaries.

**LRU Eviction**: Multiple strategies are supported (LRU, LFU, FIFO, MRU, FILO, priority-based). For LRU:
- `last_access_time` timestamps update on every node access.
- A heap orders `evictable_leaves` by timestamp.
- Cascading removal: when leaf nodes are freed, parent nodes become evictable if childless and unlocked.

**Lock Reference Counting**: Active requests increment `lock_ref` up the tree; nodes with `lock_ref > 0` are protected from eviction. `inc_lock_ref()` / `dec_lock_ref()` manage this.

**Integration with Memory Pool**: The cache interfaces with a `token_to_kv_pool_allocator` for physical GPU memory management and a `req_to_token_pool` for mapping request token positions to KV indices. Optional two-tier storage supports host (CPU) memory backup via `host_value` tensors.

**Use Cases Automatically Optimized**:
- Few-shot learning (shared examples prefix)
- Self-consistency sampling (same prompt, multiple completions)
- Multi-turn chat (conversation history prefix)
- Tree-of-thought / beam search (shared reasoning prefixes)

**Performance**: The original paper reports up to **5x higher throughput** compared to systems without prefix caching. An ablation study found **no noticeable overhead even when cache hits do not occur**, making RadixAttention universally beneficial.

### Model Runner

The `ModelRunner` handles GPU-side model execution:

- **Model Loading**: Uses `LoadConfig` with a pluggable loader system (`DefaultModelLoader` and others). Supports FP8 KV cache with scaling factors, multiple quantization formats.
- **Forward Modes**: DECODE (single token per request), EXTEND/Prefill (multiple tokens per request), IDLE (padded batches for distributed synchronization).
- **CUDA Graph Capture**: Supports full and piecewise CUDA graph replay for optimized execution of repeated computational patterns.
- **Parallelism**: Manages tensor parallelism (TP), pipeline parallelism (PP), expert parallelism (EP), and data parallelism through NCCL (GPU) and GLOO (CPU) backends.
- **Interface**: Receives `ForwardBatch` objects from the scheduler and returns logits/hidden states.

### Prefill-Decode Disaggregation

SGLang supports physically separating the prefill and decode phases onto different hardware, allowing each to be independently optimized. This architecture achieves **2.7x--3.8x throughput gains** on specialized hardware configurations because:
- Prefill is compute-bound (benefits from high FLOPS)
- Decode is memory-bandwidth-bound (benefits from high bandwidth)

### Speculative Decoding

Multiple speculative decoding methods are supported:

| Method | Description | Throughput (Llama 8B) |
|--------|-------------|----------------------|
| Baseline | No speculation | 158.34 tok/s |
| EAGLE-2 | Feature-based draft model | 244.10 tok/s |
| EAGLE-3 | Low+mid layer feature drafting | 373.25 tok/s |
| MTP | Built-in multi-token prediction heads | Model-dependent |
| Ngram | Cache-based, no draft model needed | CUDA-only |

EAGLE drafting predicts next feature vectors through a draft model, then expands a draft tree with configurable branching factors (`speculative-eagle-topk`) and depth (`speculative-num-steps`).

### Additional Optimizations

- **Paged Attention**: Memory-efficient attention with non-contiguous KV cache pages.
- **Quantization**: FP4, FP8, INT4, AWQ, GPTQ formats supported.
- **Multi-LoRA Batching**: Multiple LoRA adapters served simultaneously with batched execution.
- **Cache-Aware Load Balancing**: Routes requests to maximize cache hit rates.
- **Expert Parallelism**: Scales to 96+ GPUs for Mixture-of-Experts models.

---

## 4. Frontend Language for Structured Generation

### Design Philosophy

The SGLang frontend is a **Python-embedded domain-specific language (DSL)** for writing LLM programs that involve multiple generation calls, control flow, structured outputs, and parallelism. It draws inspiration from Microsoft's Guidance library but adds primitives for intra-program parallelism and batching.

### Core Primitives

The language is built on an expression-based intermediate representation (IR) where all constructs inherit from `SglExpr`:

- **`gen()`** (`SglGen`): Non-blocking LLM generation call with sampling parameters. Stores results in named variables for later retrieval.
- **`fork()`** (`SglFork`): Creates parallel branches of execution from the current state, enabling tree-like exploration patterns.
- **`choices()` / `select()`** (`SglSelect`): Constrained choice selection between predefined options using likelihood-based decoding.
- **Role markers** (`SglRoleBegin` / `SglRoleEnd`): Denote system/user/assistant conversation roles.
- **`SglConstantText`**: Literal string content injected into the prompt.
- **`SglVariable`**: References to previously generated or extracted values.
- **`SglSeparateReasoning`**: Handles reasoning tokens separately (for chain-of-thought models).
- **`SglCommitLazy`**: Controls lazy evaluation for optimization.

### Program Model

Programs are defined as Python functions decorated with SGLang, where the first argument `s` is the state object:

```python
@sgl.function
def multi_turn_qa(s, question1, question2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question1)
    s += sgl.assistant(sgl.gen("answer1", max_tokens=256))
    s += sgl.user(question2)
    s += sgl.assistant(sgl.gen("answer2", max_tokens=256))
```

**Execution Model**: User-written functions are traced into IR directed acyclic graphs (DAGs) tracked through `node_id` and `prev_node` references. Execution can occur in interpreter mode (immediate) or compiler mode (optimized). `SglFunction` supports batch execution via `run_batch()` with threading.

**Multi-Backend Support**: The IR is backend-agnostic. `SglSamplingParams` provides conversion methods: `to_openai_kwargs()`, `to_anthropic_kwargs()`, `to_srt_kwargs()`, and `to_vertexai_kwargs()`, enabling the same program to run against OpenAI, Anthropic, Google, or local SGLang Runtime backends.

### Structured Output / Constrained Decoding

SGLang provides three levels of constrained generation:

1. **JSON Schema**: Output constrained to match a specified JSON schema (via Pydantic models or direct schema definitions).
2. **Regular Expressions**: Output constrained to match regex patterns (e.g., `(Paris|London)`).
3. **EBNF Grammars**: Full context-free grammar constraints using Extended Backus-Naur Form.

**Compressed Finite State Machines (FSMs)**: A key innovation from the paper. Traditional constrained decoding applies token-level masks at each step, which is slow. SGLang compresses the FSM by precomputing multi-token transitions, achieving **3x faster JSON decoding** compared to token-by-token approaches.

**Backend Options for Constrained Decoding**:
- **XGrammar** (default): Supports JSON schema, regex, and EBNF (GGML BNF format).
- **Outlines**: Supports JSON schema and regex.
- **Llguidance**: Supports all three constraint types.

### Frontend-Backend Communication

The `RuntimeEndpoint` class connects the frontend to the SRT backend via HTTP:
- Initialization queries `/get_model_info` for model metadata and chat template.
- Generation sends POST requests to `/generate` with JSON payloads.
- Streaming uses `"stream": True` with line-delimited JSON response chunks.
- The `Runtime` wrapper manages the full backend lifecycle: server launch, health polling (`/health_generate`), and `atexit` shutdown.

---

## 5. Position in the LLM Inference Ecosystem

### Comparison with Other Frameworks

| Feature | SGLang | vLLM | TensorRT-LLM | Text Generation Inference (TGI) |
|---------|--------|------|---------------|--------------------------------|
| Automatic KV cache reuse | RadixAttention (radix tree) | Block-level prefix caching | Manual configuration | Limited |
| Structured generation | Native (compressed FSM) | Via plugins | Limited | Via plugins |
| Frontend DSL | Yes (Python-embedded) | No | No | No |
| Speculative decoding | EAGLE-2/3, MTP, Ngram | Yes | Yes | Yes |
| Prefill-decode disaggregation | Yes | Experimental | Yes | No |
| Multi-LoRA batching | Yes | Yes | Limited | Yes |
| Hardware support | NVIDIA, AMD, Intel, TPU, NPU | NVIDIA, AMD, TPU | NVIDIA only | NVIDIA, AMD |

### Key Differentiators

1. **Co-designed frontend + backend**: SGLang is unique in providing both a serving engine *and* a programming language for LLM applications. The frontend language enables optimizations (like automatic parallelism and prefix sharing) that are impossible with bare API calls.

2. **RadixAttention**: Automatic, zero-overhead KV cache sharing across requests using a radix tree is SGLang's signature contribution. Other systems require manual configuration or provide only block-level caching.

3. **Compressed FSMs for structured output**: Rather than applying token-level masks (slow), SGLang precomputes multi-token state transitions for constrained decoding grammars, yielding significant speedups.

4. **Breadth of optimization techniques**: SGLang integrates continuous batching, paged attention, RadixAttention, chunked prefill, speculative decoding (multiple methods), prefill-decode disaggregation, and quantization into a single cohesive system.

### Design Philosophy

The SGLang project adheres to several core principles:

- **Every line of code is on the critical path**: The contribution guide emphasizes that "most of your code runs on the critical path for every request," demanding optimization of even minor overheads. CPU-GPU synchronization must be minimized; repeated runtime checks should be cached as booleans.

- **Co-design over specialization**: Rather than building a standalone serving engine or a standalone programming framework, SGLang unifies both, enabling cross-layer optimizations.

- **Automatic over manual**: RadixAttention automatically detects and exploits prefix sharing without requiring users to manually manage caches or annotate shared prefixes.

- **Zero-overhead abstractions**: RadixAttention adds no overhead when cache misses occur. The scheduler's CPU-based batch management avoids becoming a bottleneck.

- **Production-grade modularity**: Files stay under 2,000 lines, hardware-specific code lives in dedicated files (not conditionals), and the kernel package (`sgl-kernel`) is versioned and released independently.

---

## 6. Supported Models and Hardware

### Model Support

- **Language models**: Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, and most Hugging Face Transformers-compatible models
- **Multimodal**: LLaVA and other vision-language models
- **Embedding models**: For retrieval and ranking tasks
- **Reward models**: For RLHF pipelines
- **Diffusion models**: WAN, Qwen-Image (dedicated `SGLang Diffusion` subsystem)

### Hardware Platforms

- NVIDIA: GB200, B300, H100, A100
- AMD: MI355, MI300
- Intel: Xeon CPUs
- Google: TPUs
- Huawei: Ascend NPUs

### API Compatibility

SGLang exposes an **OpenAI-compatible API**, enabling drop-in replacement for applications using the OpenAI client library. It also supports Ollama API compatibility.

---

## 7. Performance Highlights

From official benchmarks and release notes:

- **Up to 6.4x higher throughput** vs. state-of-the-art inference systems (paper, across diverse benchmarks)
- **Up to 5x speedup** from RadixAttention prefix caching alone
- **7x faster DeepSeek MLA inference** (v0.3 release)
- **3x faster JSON decoding** with compressed finite state machines
- **2.7x--3.8x throughput gains** from prefill-decode disaggregation
- **4.8x decode throughput** on GB200 NVL72 systems
- **373.25 tok/s** with EAGLE-3 speculative decoding on Llama 8B (vs. 158.34 baseline)

---

## Sources

- GitHub Repository: https://github.com/sgl-project/sglang
- Documentation: https://docs.sglang.io
- Original Paper: https://arxiv.org/abs/2312.07104
- LMSYS Blog Post (Jan 2024): https://lmsys.org/blog/2024-01-17-sglang/
- Contribution Guide: https://docs.sglang.io/developer_guide/contribution_guide.html
- Source code files examined:
  - `python/sglang/srt/entrypoints/engine.py` (Engine architecture)
  - `python/sglang/srt/managers/scheduler.py` (Scheduler design)
  - `python/sglang/srt/managers/tokenizer_manager.py` (Tokenizer pipeline)
  - `python/sglang/srt/managers/detokenizer_manager.py` (Detokenizer pipeline)
  - `python/sglang/srt/model_executor/model_runner.py` (Model execution)
  - `python/sglang/srt/mem_cache/radix_cache.py` (RadixAttention implementation)
  - `python/sglang/lang/ir.py` (Frontend IR)
  - `python/sglang/lang/backend/runtime_endpoint.py` (Frontend-backend communication)
