# EAGLE Implementation: vLLM vs SGLang

A technical comparison of EAGLE speculative decoding implementations.

---

## What is EAGLE?

**EAGLE** (Extrapolation Algorithm for Greater Language-model Efficiency) is a speculative decoding technique that accelerates LLM inference by:

1. Using a lightweight draft model to predict multiple future tokens
2. Verifying all predictions in parallel with the target model
3. Accepting correct predictions, rejecting incorrect ones

### Key Insight
EAGLE operates at the **feature level** (second-to-top layer) rather than token level, which is more predictable. It resolves uncertainty by incorporating a token sequence advanced by one time step.

### EAGLE Evolution

| Version | Key Innovation | Speedup |
|---------|----------------|---------|
| EAGLE-1 | Feature-level autoregression | 2.7x-3.5x |
| EAGLE-2 | Context-aware dynamic draft trees | 3.05x-4.26x (20-40% over v1) |
| EAGLE-3 | Tree attention kernel + overlap scheduling | 2x-3x + additional 10-20% |

---

## SGLang EAGLE Implementation

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SGLang EAGLE Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Draft   │───▶│  Build   │───▶│  Verify  │              │
│  │ Forward  │    │   Tree   │    │  (Target)│              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │                               │                     │
│       └───────────────────────────────┘                     │
│              Zero-Overhead Overlap                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components (from `eagle_worker.py`)

**1. EAGLEWorker Class**
- Extends `TpModelWorker` for tensor parallelism support
- Orchestrates draft → verify → extend cycle

**2. Draft Phase**
```python
# Generates K candidate tokens per step
topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
```
- Uses efficient top-k sampling
- Produces multiple candidate branches

**3. Tree Construction**
```python
build_tree_kernel_efficient()
```
- Organizes draft tokens into hierarchical tree structure
- Enables efficient batch verification
- Reduces redundant computation through shared prefixes

**4. Tree Attention Kernel**
- Custom CUDA kernel for EAGLE's draft tree structure
- Parallel verification of all branches in single forward pass
- Critical optimization mentioned in EAGLE-3 blog

**5. Zero-Overhead Overlap Scheduler**
```
Draft ──────────────────────────────────┐
                                        │ Overlap
Verify ─────────────────────────────────┤
                                        │
CPU Prep (FutureMap) ──────────────────┘
```
- While GPU executes verification, CPU prepares next kernels
- Uses `FutureMap` data structure for async preparation
- Delivers **additional 10-20% speedup**

**6. Attention Backend**
```python
self.draft_attn_backend = draft_backend_factory.create_decode_backend()
```
- Separate backends for decode vs extend phases
- Multi-step attention with intermediate state caching

**7. KV Cache Management**
```
KV Cache Layout:
├── Prefix tokens (shared)
├── Speculative tokens (per branch)
└── Padding tokens
```
- Strategic slot allocation via `alloc_paged_token_slots_extend()`
- Duplicates partial pages for large page sizes with multiple branches

**8. CUDA Graph Optimization**
- Separate graphs for draft and extend phases
- Eliminates CPU-GPU sync overhead between iterations

---

## vLLM EAGLE Implementation

### Architecture

vLLM takes a more modular approach:

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Spec Decode System                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Speculative Config Module               │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│       ┌───────────────────┼───────────────────┐             │
│       ▼                   ▼                   ▼             │
│  ┌─────────┐       ┌─────────────┐      ┌─────────┐        │
│  │ Draft   │       │ EAGLE Draft │      │ N-Gram  │        │
│  │ Models  │       │   Models    │      │ Spec    │        │
│  └─────────┘       └─────────────┘      └─────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

**1. Modular Strategy Selection**
- Pluggable speculation backends
- EAGLE is one of several options (Draft, MLP, N-Gram, Suffix)
- Strategy selected via configuration

**2. Integration Points**
- Connects to `LLMEngine` for inference pipeline
- Uses standard `sampling_params` for configuration
- Compatible with quantization, parallelism, LoRA

**3. Draft Model Architecture**
- Separate lightweight model for token prediction
- Loaded through standard model loading infrastructure
- Supports distributed execution

### What's Less Clear About vLLM

- Specific tree attention optimizations
- Overlap scheduling implementation
- CUDA graph capture strategy for EAGLE specifically

---

## Head-to-Head Comparison

| Aspect | SGLang | vLLM |
|--------|--------|------|
| **Architecture** | Integrated, EAGLE-specific optimizations | Modular, pluggable strategies |
| **Tree Attention** | Custom CUDA kernel | Standard attention (unclear if optimized) |
| **Overlap Scheduling** | Yes (10-20% gain) | Not explicitly documented |
| **CUDA Graphs** | Separate draft/extend graphs | General CUDA graph support |
| **KV Cache** | EAGLE-aware slot allocation | Standard paged attention |
| **EAGLE-3 Support** | Yes (documented) | Unclear |
| **Configuration** | EAGLE-focused flags | Generic spec decode config |
| **Documented Speedup** | 2x-3x + overlap gains | Not EAGLE-specific |

---

## Performance (From EAGLE-3 Blog - SGLang)

**Test Setup**: Llama 4 Scout 17B Instruct

| Metric | Improvement |
|--------|-------------|
| Decoding speedup | 2x-3x |
| Additional overlap gains | 10-20% |
| Consistency across concurrency | Yes |

**Caveats**:
- Task-dependent variations
- Potential TTFT (Time-to-First-Token) increase
- Best gains on longer generations

---

## When to Use Which

### Choose SGLang EAGLE When:
- Maximum speculative decoding performance is critical
- Using EAGLE-3 specifically
- Willing to use EAGLE-specific configuration
- Need overlap scheduling optimizations
- Latency-sensitive applications

### Choose vLLM When:
- Need flexibility between speculation strategies
- Using other vLLM features heavily (LoRA, quantization, etc.)
- Prefer modular, pluggable architecture
- May switch speculation methods based on workload
- Team more familiar with vLLM ecosystem

---

## Implementation Complexity

### SGLang Approach
```
Pros:
+ Deep EAGLE-specific optimizations
+ Custom kernels for tree attention
+ Overlap scheduling built-in
+ Better documented for EAGLE use case

Cons:
- More EAGLE-coupled code
- Less flexibility for other methods
- Steeper learning curve for internals
```

### vLLM Approach
```
Pros:
+ Cleaner abstraction boundaries
+ Easier to swap speculation strategies
+ More general-purpose
+ Larger community

Cons:
- May miss EAGLE-specific optimizations
- Less documented EAGLE performance
- Overlap scheduling unclear
```

---

## Key Takeaways

1. **SGLang has deeper EAGLE integration** - Custom tree attention kernel, overlap scheduling, EAGLE-specific KV cache management

2. **vLLM is more modular** - EAGLE is one strategy among many, easier to swap but potentially less optimized

3. **EAGLE-3 innovations** (tree attention + overlap) appear better documented in SGLang

4. **Both support basic EAGLE** - The performance gap comes from implementation details

5. **Benchmark yourself** - Published numbers vary by model/task; test on your specific workload

---

## References

- [EAGLE Paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) - Original EAGLE
- [EAGLE-2 Paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) - Context-aware trees
- [EAGLE-3 on SGLang (LMSYS Blog)](https://lmsys.org/blog/2025-12-01-eagle3-vertex/) - Tree attention + overlap
- [SGLang GitHub](https://github.com/sgl-project/sglang) - `eagle_worker.py`
- [vLLM GitHub](https://github.com/vllm-project/vllm) - `spec_decode/` module
