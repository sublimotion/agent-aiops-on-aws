# vLLM: Core Architecture and Technology Foundations

*Research compiled: 2026-02-22*

---

## 1. What is vLLM?

### Origin and Creators

vLLM (Virtual Large Language Model) is a high-throughput, memory-efficient inference and serving engine for large language models. It was created at **UC Berkeley** by a team of researchers:

- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Yu, Joey Gonzalez, Hao Zhang, and Ion Stoica.

The project was first announced in June 2023 and the accompanying research paper, *"Efficient Memory Management for Large Language Model Serving with PagedAttention,"* was published at **SOSP 2023** (the ACM Symposium on Operating Systems Principles). The paper introduced PagedAttention as its central algorithmic innovation.

### Open-Source Status

- **License**: Apache 2.0 (fully open-source, permissive license)
- **Repository**: github.com/vllm-project/vllm
- **Stars**: ~70,900 (as of February 2026)
- **Contributors**: 2,205+
- **Forks**: 13,600+
- **Downstream dependents**: 7,600+ projects
- **Current version**: v0.15.1 (released February 4, 2026)
- **Language composition**: Python (87.6%), CUDA (6.8%), C++ (4.1%)

### Real-World Deployment

vLLM has been deployed in production at LMSYS (the organization behind Chatbot Arena and Vicuna), where it:
- Handles **30,000 daily requests** with peaks of **60,000**
- Processes more than half of all Chatbot Arena requests
- Achieved a **50% GPU reduction** for equivalent traffic levels compared to prior serving solutions

---

## 2. The PagedAttention Algorithm

### The Problem It Solves

In autoregressive LLM inference, each generated token requires access to the key-value (KV) pairs of all previously generated tokens. These KV pairs are stored in GPU memory as the **KV cache**. The KV cache presents a critical memory management challenge:

- **Size**: For a Llama-2 7B model with a 10,000-token context, the KV cache alone requires approximately **5 GB** of memory -- roughly one-third of the model's own parameter memory in half-precision.
- **Scaling**: KV cache memory scales linearly with sequence length, batch size, and the model's architectural dimensions (`num_layers x num_key_value_heads x head_dim`).
- **Per-token cost**: Each token's KV cache entry requires `2 x 2 x num_layers x num_key_value_heads x head_dim` bytes (in FP16).
- **Dynamic growth**: The cache grows with every generated token and varies unpredictably across requests, since output lengths are not known in advance.

Before vLLM, existing serving systems (e.g., HuggingFace Transformers, FasterTransformer) allocated KV cache memory in large contiguous blocks, pre-reserving the maximum possible sequence length for each request. This led to **60-80% of KV cache memory being wasted** due to:

1. **Internal fragmentation**: Pre-allocated buffers are larger than actually needed.
2. **External fragmentation**: Freed memory blocks leave gaps that cannot be reused.
3. **Redundant duplication**: Parallel sequences (beam search, parallel sampling) each maintain separate copies of shared KV data.

### How PagedAttention Works

PagedAttention is an attention algorithm directly inspired by **virtual memory and paging** in operating systems. It applies the OS concept of non-contiguous memory allocation to KV cache management.

#### Core Concepts (OS Analogy)

| OS Concept | PagedAttention Equivalent |
|---|---|
| Process | Sequence (request) |
| Page | KV cache block |
| Byte | Token |
| Virtual address space | Logical block space |
| Physical memory | GPU KV cache memory |
| Page table | Block table |

#### Block-Based KV Cache Partitioning

Instead of allocating one large contiguous buffer per sequence, PagedAttention:

1. **Divides the KV cache into fixed-size blocks**, where each block holds the keys and values for a fixed number of tokens (the block size).
2. **Maintains a block table** per sequence that maps **logical blocks** (contiguous in the sequence's view) to **physical blocks** (potentially non-contiguous in GPU memory).
3. **Allocates physical blocks on demand** -- new blocks are only allocated as new tokens are generated, not pre-reserved for the maximum possible length.

This means that:
- Blocks for the same sequence do **not** need to be physically contiguous in GPU memory.
- Memory waste occurs **only in the last block** of a sequence (since all other blocks are fully utilized).
- The resulting memory waste is **under 4%**, compared to 60-80% in traditional systems.

#### Memory Sharing via Reference Counting

For parallel decoding strategies (beam search, parallel sampling), multiple sequences often share a common prefix of tokens. PagedAttention enables **physical block sharing**:

- Multiple sequences can map their logical blocks to the **same physical block**.
- The system maintains **reference counts** on physical blocks to track how many sequences share each block.
- When a sequence needs to modify a shared block (e.g., diverging in beam search), PagedAttention uses **copy-on-write (CoW)**: it creates a new physical copy of the block only at the moment of modification, not before.
- This reduces memory overhead for parallel sampling by **up to 55%**.

---

## 3. Memory Management Approach and Why It Matters

### Why Memory Management Is the Key Bottleneck

LLM serving throughput is fundamentally limited by GPU memory, not compute, for the following reasons:

1. **Model parameters** consume a fixed portion of GPU memory (e.g., ~14 GB for a 7B model in FP16).
2. **KV cache** consumes the remaining memory and determines how many requests can be batched simultaneously.
3. **Batch size directly controls throughput**: larger batches amortize the cost of loading model weights from GPU HBM, increasing arithmetic intensity and GPU utilization.
4. **Memory waste reduces effective batch size**: if 60-80% of KV cache memory is wasted, the system can serve far fewer concurrent requests.

Therefore, efficient KV cache memory management translates directly to higher batch sizes, higher GPU utilization, and higher throughput.

### vLLM's Memory Management in Practice

vLLM's memory manager operates as follows:

1. **Pre-allocation of a KV cache pool**: At startup, vLLM pre-allocates a large pool of physical KV cache blocks in GPU memory based on available memory after loading the model.
2. **Dynamic block allocation**: As requests arrive and tokens are generated, physical blocks are allocated from the pool to each sequence on demand via the block table.
3. **Block reclamation**: When a sequence completes, all its physical blocks are returned to the pool immediately for reuse.
4. **Sharing and CoW**: For parallel decoding, blocks are shared and only copied when modified.

This approach achieves **near-zero memory waste** and enables vLLM to batch significantly more requests simultaneously than systems using contiguous allocation.

### Quantitative Impact

| Metric | vLLM Improvement |
|---|---|
| Memory waste | Under 4% (vs. 60-80% in prior systems) |
| Throughput vs. HuggingFace Transformers | Up to **24x** higher |
| Throughput vs. HuggingFace TGI | Up to **3.5x** higher |
| Throughput vs. FasterTransformer / Orca | **2-4x** higher at equivalent latency |
| Parallel sampling memory reduction | Up to **55%** |
| Parallel sampling throughput gain | Up to **2.2x** |

Performance improvements are **more pronounced** with:
- Longer sequences (more KV cache pressure)
- Larger models (higher per-token KV cache cost)
- More complex decoding algorithms (beam search, high-n sampling)

---

## 4. Core Architectural Design Decisions

### 4.1 Continuous Batching (Iteration-Level Scheduling)

Traditional **static batching** groups requests into fixed-size batches and waits for all sequences in the batch to complete before admitting new requests. This wastes GPU compute because:
- Short sequences finish early but the GPU idles until the longest sequence completes.
- Batch slots remain occupied by completed sequences.

vLLM implements **continuous batching** (also called iteration-level scheduling):
- After **each forward pass** (each generated token), the scheduler can:
  - Remove completed sequences from the batch.
  - Add new pending requests into freed slots.
- This maximizes GPU utilization by keeping the batch as full as possible at all times.

Continuous batching combined with PagedAttention enables up to **23x throughput improvement** over naive static batching.

### 4.2 Multi-Process Architecture (V1)

vLLM's V1 architecture separates concerns into distinct processes:

1. **API Server Process**: Handles incoming HTTP requests (OpenAI-compatible API). Communicates with the engine core via ZMQ (ZeroMQ) sockets.
2. **Engine Core Process**: Orchestrates the inference pipeline -- runs the scheduler, manages the block table, and coordinates execution.
3. **GPU Worker Processes**: Execute the actual model computation (forward passes) on GPU hardware.
4. **DP Coordinator Process**: Manages data parallelism when enabled across multiple devices.

**Why process separation matters**: Profiling of earlier versions (pre-v0.6.0) revealed that on a single H100 GPU running Llama 3 8B:
- The HTTP API server consumed **33%** of total execution time.
- Scheduling and data preparation consumed **29%**.
- Actual GPU computation was only **38%**.

By separating these into different OS processes communicating via ZMQ, vLLM eliminates **Python GIL contention** and allows the CPU-bound API serving and scheduling to run in parallel with GPU computation. This yielded:
- **2.7x throughput improvement** for Llama 8B
- **1.8x throughput improvement** for Llama 70B
- **5x faster** time-per-output-token for Llama 8B

### 4.3 Engine and Execution Pipeline

The inference pipeline flows through several layers:

```
Request --> API Server --> LLMEngine/AsyncLLMEngine --> Scheduler --> Worker --> ModelRunner --> Model
```

- **LLMEngine**: Synchronous inference coordinator. Manages request queues, invokes the scheduler, and dispatches execution to workers.
- **AsyncLLMEngine**: Asynchronous wrapper for non-blocking concurrent request handling (used by the API server).
- **Scheduler**: Determines which requests to include in the next batch, manages preemption when memory is exhausted, and interfaces with the block manager for KV cache allocation.
- **Worker**: Interface between the engine and GPU hardware. Manages GPU resources.
- **ModelRunner**: Handles the actual forward pass execution, including input preparation, CUDA graph management, and output processing.

### 4.4 Multi-Step Scheduling

Rather than scheduling one token generation step at a time (which causes GPU idle time while the CPU prepares the next step), vLLM schedules **multiple steps ahead**. The scheduler plans several forward passes before returning control to the CPU for output processing. This:
- Amortizes CPU scheduling overhead across multiple GPU steps.
- Yields a **28% throughput improvement** for Llama 70B on 4xH100s.
- Trades slightly increased latency for significantly higher throughput.

### 4.5 Asynchronous Output Processing

Output data handling (detokenization, response formatting) is **overlapped with GPU computation**. By deferring output processing until the next execution cycle begins, the GPU is never idle waiting for CPU-side output work.

### 4.6 Distributed Inference Support

vLLM supports multiple parallelism strategies for models that exceed single-GPU memory:

- **Tensor Parallelism (TP)**: Splits individual layers across GPUs.
- **Pipeline Parallelism (PP)**: Distributes different layers across GPUs.
- **Data Parallelism (DP)**: Runs replicas of the model across GPUs with request distribution.
- **Expert Parallelism (EP)**: For Mixture-of-Experts models, distributes experts across GPUs.

### 4.7 Advanced Features

vLLM incorporates several additional optimizations:

- **Prefix Caching (Automatic Prefix Caching / APC)**: Caches KV blocks for common prefixes (system prompts, shared context) across requests, avoiding redundant computation.
- **Speculative Decoding**: Uses a smaller draft model to propose multiple tokens, then verifies them in a single forward pass of the target model, reducing the number of expensive forward passes.
- **Chunked Prefill**: Splits long prompt processing into chunks to allow interleaving with decode steps, reducing time-to-first-token for concurrent requests.
- **CUDA/HIP Graph Optimization**: Captures and replays GPU execution graphs to reduce kernel launch overhead.
- **Quantization Support**: GPTQ, AWQ, AutoRound, INT4, INT8, FP8 -- reduces model memory footprint to serve larger models or increase batch sizes.

### 4.8 Hardware Support

vLLM is designed for broad hardware compatibility:
- NVIDIA GPUs (primary target, CUDA)
- AMD GPUs (ROCm/HIP)
- Intel CPUs and GPUs
- Google TPUs
- Intel Gaudi accelerators
- Huawei Ascend NPUs
- PowerPC and ARM CPUs

---

## 5. Summary of Key Technical Insights

1. **The central insight**: LLM serving throughput is memory-bound, and KV cache management is the critical bottleneck. By borrowing virtual memory techniques from operating systems, vLLM nearly eliminates memory waste in KV cache allocation.

2. **PagedAttention** partitions KV cache into fixed-size blocks mapped via block tables, enabling non-contiguous allocation, on-demand growth, and copy-on-write sharing. This reduces waste from 60-80% to under 4%.

3. **Continuous batching** ensures GPU utilization stays high by dynamically adding and removing requests from the active batch at every iteration.

4. **Process separation** eliminates Python GIL contention between API serving, scheduling, and GPU execution -- a pragmatic engineering decision that nearly doubled throughput.

5. **Multi-step scheduling and async output processing** further reduce CPU-GPU synchronization overhead.

6. The combination of these techniques delivers **2-24x throughput improvements** over prior systems, with gains increasing for longer sequences, larger models, and more complex decoding strategies.

---

## Sources

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023 (arXiv:2309.06180)
- vLLM Blog: "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention" (June 2023)
- vLLM Blog: "vLLM v0.6.0 Performance Benchmarks" (September 2024)
- vLLM Official Documentation: docs.vllm.ai
- vLLM GitHub Repository: github.com/vllm-project/vllm
- Anyscale Blog: "Continuous Batching in LLM Inference"
- Hugging Face Blog: "KV Cache Quantization"
