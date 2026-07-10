# Pragmatic Inference Optimization Guide

**Audience**: Engineers deploying open-weight LLMs on AWS GPU instances (g7e, p5e, p6-b200, p6-b300)
**Last updated**: 2026-07-08
**Sources**: InferenceX v2 (SemiAnalysis), our own B200/B300/g7e benchmarks, field deployments

---

## Decision Framework

Before optimizing, identify your position on the **throughput-interactivity curve**:

| Use case | Interactivity target | Optimization priority |
|----------|---------------------|----------------------|
| Agentic coding (Claude Code-like) | 100-150 tok/s/user | Latency: TP, MTP, low batch |
| Interactive chat | 40-75 tok/s/user | Balanced: hybrid TP+EP, disagg |
| Batch processing / eval | 10-30 tok/s/user | Throughput: wide EP, high batch, FP4 |
| Reasoning (long CoT) | 50-100 tok/s/user | Decode-heavy: more decode nodes, MTP |

The fundamental law: **higher interactivity = fewer tokens amortized over fixed GPU cost = higher $/token**. Fast mode (2.5x speed) costs 6-12x more. This is physics, not pricing strategy.

---

## 1. Parallelism Strategy Selection

### The Batch Size Rule

The optimal parallelism strategy depends almost entirely on batch size (concurrent sequences):

| Batch size | Best strategy | Why |
|-----------|---------------|-----|
| 1-16 | Pure TP | EP load imbalance kills throughput; <30% experts activate per layer |
| 16-64 | Hybrid TP+EP (TEP) | TP for attention, EP for MoE layers. 50-60% expert activation |
| 64+ | Pure DEP (DP+EP) | Full expert activation, weight amortization across more GPUs |

**Our validated configs:**
- GLM-5 744B on B200 TP8: Single-stream 50 tok/s, 2,602 tok/s @ c=128
- Qwen3-235B FP8 on B300 TP4: 110 tok/s single, 11,820 tok/s @ c=512
- Kimi K2.6 1T on B300 TP8: 10,437 tok/s @ c=512 (vLLM), 3,400 tok/s (SGLang)

### Expert Parallelism Sizing

For MoE models (DeepSeek, Qwen3-MoE, GLM-5, Kimi):

```
Experts per GPU = total_experts / EP_size
Tokens per expert per step = batch_size * top_k / total_experts

Rule of thumb: want >= 4 tokens/expert/step for EP to beat TP
  - DeepSeek R1: 256 experts, top_k=8 → need batch >= 128 for EP64
  - Qwen3-235B: 128 experts, top_k=8 → need batch >= 64 for EP32
  - GLM-5: 256 experts, top_k=8 → same as DeepSeek
```

### Wide EP: When NVLink Domain Matters

Wide EP spreads experts across more GPUs than a single node. Benefits:
1. **Less expert weight per GPU** → more KV cache headroom
2. **More tokens funneled per expert** → higher arithmetic intensity
3. **Aggregate HBM bandwidth scales linearly** with GPU count

**When wide EP helps:**
- NVL72/NVSwitch (p5e, p6): EP across all 8 GPUs at 900 GB/s — no IB penalty
- Multi-node over IB: EP16/32/64 across nodes — all-to-all tolerates ~50-100 GB/s/GPU

**When wide EP does NOT help:**
- Low batch sizes (< 32) — expert imbalance dominates
- Very high interactivity (>130 tok/s/user) — latency-bound, NVLink BW not saturated
- PCIe-only topology (g7e) — no NVSwitch for cross-GPU EP; use DP replicas instead

### Critical: FP8 MoE TP Divisibility

Before deploying ANY FP8 MoE model:
```
moe_intermediate_size / TP_SIZE % 128 must == 0
```
Fine-grained FP8 uses block_n=128. If this fails, inference silently produces garbage.

Examples:
- Qwen3-235B: `1536 / 8 = 192` (FAIL), `1536 / 4 = 384` (OK) → use TP4
- DeepSeek R1: `2048 / 8 = 256` (OK) → TP8 fine

---

## 2. Prefill/Decode Disaggregation

### When to Disaggregate

Disagg eliminates prefill/decode interference. Worth it when:
- Mixed context lengths (short + long requests simultaneously)
- Tight p99 TPOT SLOs (decode must not be interrupted by prefill bursts)
- Moderate-to-high QPS (crossover vs chunked prefill)
- Workload is prefill-heavy (long input, short output — RAG, summarization)

**NOT worth it when:**
- Low QPS with uniform context lengths
- Single-node with TP>1 (NVLink contention for KV transfer — see below)
- Very short contexts (<1K tokens) where prefill is trivial
- Prefix caching is active and hit rates are high (>60%) — prefill is already skipped for most requests
- MoE models on single node — EP dispatch overhead (0.6ms/step NVSwitch, 6-12ms/step EFA) exceeds any benefit

### When NOT to Disaggregate (Anti-Patterns)

Disaggregation is the most overapplied optimization in LLM serving today. Our field data shows it's actively harmful in common scenarios:

**Anti-pattern 1: Disagg with prefix caching active**

If prefix caching already eliminates 60-90% of prefill work, disagg adds KV transfer latency for no benefit. Our K2.6 benchmark: prefix caching gave 103x TTFT improvement on cold→warm. After that, most requests have ~0ms effective prefill — there's nothing to disaggregate.

```
Without prefix cache: prefill 5,928ms → disagg helps (offloads heavy prefill)
With prefix cache:    prefill 57ms     → disagg adds 1-10ms KV transfer for 0 gain
```

**Anti-pattern 2: Disagg on single-node MoE with TP>1**

With TP8 (required for 1T models like K2.6), every decode step already uses NVLink for AllReduce. Adding KV transfer over the same NVLink fabric creates contention. Our measured result: single-node TP8 with vLLM reached 10,437 tok/s — no disagg architecture matched this on the same hardware.

**Anti-pattern 3: Disagg for agentic workloads (short turns, high interactivity)**

Agentic coding has short inputs (tool results ~1-4K tokens) and prefix-heavy patterns (system prompt + tool defs reuse across turns). The prefill is either cached (free) or trivial (<50ms). Disagg adds architectural complexity and a fixed TTFT penalty (KV transfer) on every request.

**Anti-pattern 4: Disagg without InfiniBand (EFA/TCP)**

On AWS EFA (p5en, g7e), KV transfer requires CPU bounce (GPU→cudaMemcpy→CPU→EFA→CPU→cudaMemcpy→GPU). At 100-200μs per hop with 60 layers of KV data, this adds 6-12ms to TTFT per request — wiping out any scheduling benefit.

**When disagg IS the right call:**
- Multi-node deployment where model doesn't fit on one node
- Genuine prefill-heavy workloads (RAG with 32K+ documents, first-turn only)
- Mixed-priority traffic (batch + interactive sharing the same fleet)
- InfiniBand or NVSwitch with TP=1 (zero-contention cuda_ipc)

### P:D Ratio Selection

| Workload pattern | Input:Output | P:D ratio | Example |
|-----------------|-------------|-----------|---------|
| RAG / summarization | 8K:1K | 4P:1D to 7P:2D | Long prefill, short decode |
| Interactive chat | 1K:1K | 1P:1D to 1P:2D | Balanced |
| Reasoning / CoT | 1K:8K | 1P:3D to 1P:4D | Short prefill, long decode |
| Agentic coding | 4K:2K | 2P:3D | Mixed, decode-heavy |

### Single-Node Disagg (TP=1 Models)

For models fitting on one GPU (8B, 14B, 32B-FP8) on an 8-GPU NVSwitch node:

```
GPU0-3: prefill workers (TP=1 each)
GPU4-7: decode workers (TP=1 each)
KV transfer: cuda_ipc over NVLink — 900 GB/s, zero contention
```

**Key requirement**: TP=1 means no NCCL collectives → cuda_ipc is safe.
With TP>1, NCCL allreduce and NIXL KV transfer compete for NVLink bandwidth.

Pod config for cuda_ipc:
```yaml
spec:
  hostIPC: true
  containers:
  - env:
    - name: UCX_TLS
      value: "cuda_copy,cuda_ipc"
    securityContext:
      capabilities:
        add: ["IPC_LOCK"]
```

### KV Cache Transfer Cost

For DeepSeek R1 (61 layers, FP8 KV):
- 8K context → ~500 MB KV data
- Over NVLink (900 GB/s): <1ms
- Over IB 400G (50 GB/s): ~10ms
- Over TCP loopback: ~200ms (40x slower — NIXL fallback without cuda_ipc)

**Rule**: KV transfer latency adds directly to TTFT. Budget it.

---

## 3. Multi-Token Prediction (MTP) / Speculative Decoding

### When MTP Helps Most

MTP trades compute for fewer memory-bound decode steps. Effectiveness depends on:

| Condition | MTP benefit |
|-----------|-------------|
| Low batch, memory-bandwidth-bound | **High** — compute slack available |
| High interactivity target (>100 tok/s/user) | **Critical** — often the only way to hit target economically |
| Large batch, compute-bound | **Low** — no slack for verification |
| FP4 quantized | **High** — weights smaller, decode more BW-bound |

### Measured Impact

From InferenceX data at fixed 50 tok/s/user:
- GB300 NVL72 FP4: MTP reduces cost from $0.251 → $0.057/M tokens (**4.4x cheaper**)
- At 150 tok/s/user: $2.35 → $0.11/M tokens (**21x cheaper**)
- B200 Disagg FP4: MTP provides 30-50% throughput uplift across the frontier

**Our results (B300, SGLang EAGLE3 with stock off-the-shelf draft models):**
- **GLM-5 on B200 with MTP** (`--speculative-config.method mtp --speculative-config.num_speculative_tokens 1`): Measurable throughput gain at decode (built-in MTP heads, distribution-aligned).
- **Kimi K2.6 + `lightseekorg/kimi-k2.6-eagle3`** (`s4_d4_k1` after tuning):
  - Single-stream: **128 → 302 tok/s (+136%)** at c=1.
  - c=128: 6,410 tok/s — net positive.
  - c=512 fullstack (HiCache 200 GB/rank): 7,759 tok/s vs **10,437 tok/s no-spec baseline (−26%)**. EAGLE3 is net-negative past c≈256.
  - SGLang defaults (`s3_d4_k1`) collapse to **3,657 tok/s @ c=64 (−65%)** — defaults are mistuned for K2.6, do not ship them.
- **Qwen3-235B + `lmsys/Qwen3-235B-A22B-EAGLE3`** (FP8 TP8, SGLang):
  - Advertised draft accept length 3.0–3.5 (vs Kimi's 5.0) — lower theoretical headroom.
  - ShareGPT @ c=16: **63.8 tok/s/req**, statistically tied with vLLM TP4 NVFP4 no-spec (63.3) and **~2× behind CoreWeave's tuned stack (128.2)**. The gap is attributed to custom NVFP4 kernels + a *custom-trained* draft, not stock EAGLE3.
  - DP+EP+EAGLE3 combo crashes (SGLang 0.5.10); single-node EP+EAGLE3 regresses 14–39%.
- **Kimi K2.6 vLLM**: MTP not yet supported in vLLM for this architecture; EAGLE3 requires custom image.

### Potential vs Actual: the Stock-Draft Trap

> **The dominant factor in real-world spec-decode performance is whether the draft model has been fine-tuned on the target model's serving distribution.** Off-the-shelf EAGLE3 drafts (`lightseekorg/kimi-k2.6-eagle3`, `lmsys/Qwen3-235B-A22B-EAGLE3`) are trained on generic mixtures and do not match production traffic.

Synthetic benchmarks systematically overstate stock-draft EAGLE3 by 3–5× (Kimi K2.6 measured):

| Metric | Synthetic (`vllm bench --dataset-name random`) | Real ShareGPT |
|---|---|---|
| Accept rate | **1.00** | **0.156** |
| Accept length | **5.0** | **1.62** |
| Per-req tok/s | 325 | **54** |

**Why this matters for reporting**: Most published EAGLE3 / MTP throughput numbers (including some of our internal sweeps) use synthetic prompts. They represent the *upper bound* of what a perfectly-aligned draft could deliver, not what the stock draft delivers in production. Always:

1. Quote both synthetic and ShareGPT/production-distribution numbers.
2. Treat synthetic figures as **theoretical potential** that requires draft fine-tuning to realize.
3. Assume the **2× CoreWeave gap on Qwen3** and the **−26% regression at c=512 on Kimi** both close substantially with a draft fine-tuned on actual traffic — neither is a property of EAGLE3 itself.
4. For new model rollouts, budget for draft fine-tuning (or eviction of spec-decode from the rollout plan) rather than relying on stock weights.

### MTP vs External Draft Model

| Approach | Pros | Cons |
|----------|------|------|
| MTP (built-in heads) | No extra model, high acceptance rate, aligned distribution | Requires model was trained with MTP heads |
| External draft model | Works with any model | Misaligned distributions, operational complexity, extra GPU memory |

**Models with MTP heads**: DeepSeek R1/V3/V4, GLM-5
**Models requiring external draft**: Qwen3, Kimi K2.x (EAGLE3: `lightseekorg/kimi-k2.6-eagle3`), Llama

### EAGLE3 for Kimi K2.6

EAGLE3 uses a lightweight draft model trained to predict the target model's next tokens. For K2.6:

```bash
# SGLang with EAGLE3
SGLANG_ENABLE_SPEC_V2=1 sglang serve \
  --model-path moonshotai/Kimi-K2.6 \
  --tp 8 \
  --speculative-algorithm=EAGLE3 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-draft-model-path lightseekorg/kimi-k2.6-eagle3 \
  --speculative-draft-attention-backend trtllm_mha \
  --trust-remote-code

# vLLM with EAGLE3 (requires custom image for Blackwell)
# Image: voipmonitor/vllm:cu130-mtp-tuned-v3-20260423
# Draft model: lightseekorg/kimi-k2.5-eagle3-mla
```

**Critical**: SGLang requires `--speculative-draft-attention-backend trtllm_mha` — without it, the draft model inherits the target's `trtllm_mla` backend and produces garbage.

**Known issues** (as of May 2026):
- EAGLE3 + DCP (Decode Context Parallelism) has workspace allocation failures (SGLang #40791)
- vLLM MTP deadlocks at concurrency > 1 with TP > 1 (vLLM #41404)
- Best at low concurrency (bs=1-4); at high batch, verification overhead may negate gains

### Accuracy Validation

MTP does NOT degrade accuracy for models trained with it (DeepSeek V3 paper confirms). But always validate:
- Run GSM8k / MATH-500 on the exact serving config
- Compare accepted token ratio between synthetic and real data (should be similar)
- At high interactivity, acceptance rate matters more (fewer recovery decode steps)

---

## 4. KV Cache Optimization

### Hierarchy: GPU HBM → CPU DRAM → NVMe

| Tier | Bandwidth | Capacity (B300) | Use case |
|------|-----------|-----------------|----------|
| GPU HBM | 8 TB/s aggregate | 275 GB/GPU | Active decode sequences |
| CPU DRAM (HiCache) | 900 GB/s (GB200), 128 GB/s (PCIe 5.0) | 2 TB+ | Warm cache, prefix reuse |
| NVMe | 7 GB/s | 30 TB | Cold cache, background swap |

### Prefix Caching

**Critical for multi-turn and agentic workloads.** Shared system prompts, conversation history, and tool definitions skip prefill entirely.

Our measured results:
- Kimi K2.6 on B300 (vLLM): Cold→warm TTFT improvement **103x** (5928ms → 57ms)
- SGLang RadixAttention: No benefit observed in our benchmark patterns (single-turn synthetic)

**Lesson**: Prefix caching only helps with real multi-turn data. Synthetic random-token benchmarks (like InferenceX baseline) show 0% hit rate by design. Always test with representative workloads.

### HiCache (Hierarchical KV Offloading)

Offloads inactive KV cache blocks to CPU DRAM. Measured on GLM-5 B200:
- `--enable-hierarchical-cache --hicache-size 100` (100 GB/rank)
- **71% throughput gain at 64 concurrent**, 2.86x peak vs baseline
- `hicache-size` must exceed device KV pool (e.g., ~82 GB/rank for GLM-5)

**Sizing rule:**
```
hicache_size_gb >= device_kv_pool_gb * 1.2
total_cpu_ram_needed = hicache_size_gb * num_gpus
```

**Failure mode**: HiCache 500GB/rank on B300 = 4 TB total → exceeds system RAM → OOM.
Safe: 200GB/rank on B300 (1.6 TB total).

### KV Cache Memory Budget

```
available_kv_memory = total_hbm - model_weights - activation_memory - cuda_overhead
max_concurrent_seqs = available_kv_memory / (kv_per_token * max_seq_len * num_layers)
```

| Model | Weights (FP8) | KV/token/layer | Available KV (B300 TP8) |
|-------|--------------|----------------|------------------------|
| DeepSeek R1 671B | ~670 GB | 1 KB (FP8, MLA) | ~1,500 GB |
| GLM-5 744B | ~740 GB | ~2 KB | ~1,400 GB |
| Qwen3-235B | ~235 GB | ~1.5 KB | ~1,800 GB (TP4, 4 GPUs) |
| Kimi K2.6 1T | ~555 GB (INT4) | ~2 KB | ~1,600 GB |

More KV headroom = higher max concurrency = lower $/token at throughput-optimized configs.

---

## 5. Quantization: FP8 vs FP4

### Decision Matrix

| Factor | FP8 | FP4 |
|--------|-----|-----|
| Accuracy | Near-lossless | Slight degradation (QAT recovers most) |
| Throughput (single node) | Baseline | 1.5-2x (smaller weights, faster load) |
| Throughput (disagg+wideEP) | Baseline | Up to 3x on NVL72 |
| Composability | Mature everywhere | Nvidia-only production-ready |
| Model availability | All major models | DeepSeek, some Qwen3 |

### Practical Rules

1. **Start with FP8** — universal support, near-lossless, simpler deployment
2. **Move to FP4 only when**: (a) model has QAT/PAT weights, (b) serving on Blackwell, (c) need max throughput at scale
3. **GPTQ-Int4 on vLLM is unreliable** for some models (Qwen3.5 MoE produces garbage). Always validate output quality.
4. **FP4 + disagg + wide EP composability**: Only works well on NVIDIA today. AMD MI355X has severe performance regression when combining all three.

---

## 6. Engine Selection: vLLM vs SGLang vs TRT-LLM

### Quick Selection Guide

| Model / Use case | Recommended engine | Why |
|-----------------|--------------------|-----|
| DeepSeek R1 (production scale) | TRT-LLM + Dynamo | 2x+ throughput vs SGLang at disagg+wideEP |
| DeepSeek R1 (single node) | SGLang or vLLM | Simpler, competitive on FP8 single-node |
| GLM-5 | SGLang (`lmsysorg/sglang:glm5-blackwell`) | Only engine with full MoE+MTP+NSA support |
| GLM-5 | vLLM (`vllm/vllm-openai:glm5`) | Good alternative, tool calling works |
| Qwen3/3.5 | vLLM | Best tool calling support (`--tool-call-parser hermes`) |
| Kimi K2.x | vLLM | 3.1x better throughput scaling vs SGLang in our tests |
| Latency-critical chat | vLLM (prefix caching) | FLASHINFER_MLA + aggressive caching |
| Maximum throughput | TRT-LLM (if supported) | Consistently highest peak tok/s |

### Engine Maturity Matrix

| Feature | vLLM | SGLang | TRT-LLM |
|---------|------|--------|---------|
| FP8 single-node | Mature | Mature | Mature |
| FP4 | Good (Blackwell) | Good (Blackwell) | Best |
| Disagg prefill | Via Dynamo/NIXL | Native | Via Dynamo |
| Wide EP | Via Dynamo | Native EP scaling | Via Dynamo |
| MTP (DeepSeek) | Supported | Supported | Best |
| Prefix caching | Excellent | RadixAttention | Good |
| HiCache / KV offload | SGLang only | Native | No |
| Tool calling | Best ecosystem | Limited | Limited |
| Multi-modal | Good | Limited | No |

### Cold Start Considerations

| Engine | Cold start (GLM-5 744B, B200) | Notes |
|--------|-------------------------------|-------|
| SGLang | ~15 min | DeepGEMM JIT compilation |
| vLLM | ~16 min | torch.compile + CUDA graphs |
| TRT-LLM | ~20-30 min (first time) | Engine build; cached thereafter |

For spot instances or autoscaling: SGLang cold start 2.8x faster than vLLM on Kimi K2.6 (3 min vs 8.3 min). Matters when instances reclaim.

---

## 7. Hardware-Specific Optimization

### g7e (4x RTX PRO 6000 Blackwell, PCIe)

- **Topology**: PCIe only, no NVSwitch — DP replicas, NOT TP across GPUs
- **NCCL broken**: NCCL 2.25.1 fails on sm_120 PCIe. Fixed in 2.26.2+ (NGC 25.03+)
- **Sweet spot**: 4 independent TP1 replicas for models <=24B FP8 (96 GB/GPU)
- **Network**: `--network host` required (no CNI plugin on bare metal)
- **Container runtime**: `nerdctl` (not docker). Use `sudo nerdctl`
- **EFA**: 2 interfaces on 24xl. Enables NIXL LIBFABRIC for multi-node disagg, NOT GPUDirect RDMA

### p6-b200.48xlarge (8x B200, NVSwitch)

- **AMI**: Must use AL2023 NVIDIA AMI. AL2 lacks `ib_umad` for Fabric Manager
- **Scale-up**: 8 GPUs at 900 GB/s NVLink — full EP8, TP8
- **IB scale-out**: 800 Gbit/s for multi-node disagg/wideEP
- **TP8**: Good for dense models and MoE at low batch
- **EP8 + DP (multi-node)**: For throughput at high concurrency
- **Capacity blocks**: Instance termination takes ~10 min

### p6-b300.48xlarge (8x B300 Ultra, NVSwitch)

- **VRAM**: 275 GB/GPU (2,200 GB total) — largest single-node memory
- **FP4**: 1.5x theoretical vs B200, but only 1.1x measured (software immature)
- **FP8**: Up to 1.5x better than B200 (measured, despite same spec)
- **Container tags**: Must use `-cu130` for sm_103
- **Sweet spot for MoE**: Massive VRAM allows TP4 with huge KV cache headroom
- **CLI renamed**: `hf download` not `huggingface-cli download` (huggingface_hub v1.11+)

### NVL72 (GB200/GB300)

- **72 GPUs on NVLink** — 900 GB/s/GPU, 9x bandwidth vs IB multi-node
- **EP across all 72**: Only 4 experts/GPU at EP64 → massive KV headroom
- **Throughput**: 3x per-GPU vs B200 at 60 tok/s/user interactivity
- **When NVL72 is NOT better**: >130 tok/s/user (low batch, latency-bound, IB not saturated)
- **Cost**: Higher per-GPU TCO, but 8x better bandwidth/$ — dominates at throughput

---

## 8. Workload-Specific Recipes

### Agentic Coding (Claude Code-like)

```yaml
target: 100-150 tok/s/user, tight TPOT p99
parallelism: TP8 (or TP4 for smaller models)
speculative_decode: EAGLE3 draft model if available (critical for single-stream latency)
prefix_caching: enabled (system prompt + tool defs reuse 80-95% — the dominant optimization)
kv_offload: hicache only if context accumulates beyond GPU KV budget
disagg: NO for single-node (prefix caching eliminates prefill; disagg adds latency)
disagg: YES only for multi-node fleets with IB and mixed-priority traffic
quantization: FP8 (accuracy matters for code)
dynamic_mla_routing: enabled if available (3x TTFT for <1K prefill — matches agentic turn pattern)
```

**Why no disagg**: Agentic turns are short (1-4K input after first turn), prefix-cached (80-95% hit rate), and latency-sensitive. Disagg adds KV transfer cost on every request for workloads where prefill is already ~0ms effective. Our K2.6 data: 10,437 tok/s on single-node TP8 with prefix caching — no disagg needed.

### RAG / Long-Context QA

```yaml
target: 40-75 tok/s/user
parallelism: EP8 or TEP (hybrid)
mtp: enabled
prefix_caching: critical (shared document chunks)
kv_offload: hicache for document cache retention
disagg: yes, 4P:1D ratio (prefill-heavy, 8K+ inputs)
quantization: FP8
```

### Batch Evaluation / Offline Processing

```yaml
target: maximize tok/s (10-30 tok/s/user acceptable)
parallelism: wide EP (EP32/64 if multi-node), pure DEP
mtp: enabled (smaller benefit at high batch, but still positive)
prefix_caching: enabled if dataset has shared prefixes
kv_offload: not needed (batch completes, KV freed)
disagg: yes, balanced P:D
quantization: FP4 if available (throughput >> accuracy at margins)
```

### Interactive Chat (Multi-turn)

```yaml
target: 50-75 tok/s/user
parallelism: TEP (TP4+EP2 within node, DP across nodes)
mtp: enabled
prefix_caching: critical (conversation history, system prompt)
kv_offload: hicache (retain conversation KV between turns)
disagg: yes, 1P:2D (short prefill after first turn due to cache hits)
quantization: FP8
```

---

## 9. Cost Optimization Decision Tree

```
START: What's your interactivity target?

├── >100 tok/s/user (agentic, fast mode)
│   ├── MTP enabled? → if NO, enable it (up to 21x cost reduction)
│   ├── Small batch → use TP, NOT EP
│   └── Accept higher $/token — this is the race car, not the bus
│
├── 40-100 tok/s/user (interactive chat)
│   ├── Disagg + hybrid TP/EP → best $/tok in this range
│   ├── Prefix caching → critical for multi-turn (103x TTFT improvement)
│   └── FP8 sufficient, FP4 if throughput-limited
│
└── <40 tok/s/user (batch, throughput)
    ├── Wide EP (EP32+) on NVL72 → lowest $/token possible
    ├── FP4 if model supports it → 2-3x cheaper
    ├── Maximize batch size → amortize GPU cost over more tokens
    └── This is the bus — pack it full
```

### Spot vs On-Demand

| Instance | Spot discount | Cold start penalty | Recommendation |
|----------|--------------|-------------------|----------------|
| g7e.24xl | ~65% | 2-3 min | Always spot for dev/eval |
| p6-b200 | ~60% | 15-16 min | Spot for batch; on-demand for serving |
| p6-b300 | ~70% | 15-16 min | Spot for throughput workloads |

At 15 min cold start (GLM-5), a spot reclaim costs 15 min of downtime. For latency-SLO serving, use capacity blocks or on-demand with reserve.

---

## 10. Benchmarking Methodology

> **Standard workload catalog**: Use the 7 workload cards in `standards/benchmark-commons/workloads/` for comparable benchmarks. Each card defines dataset, load pattern, and SLO targets. Results are stored as **enriched artifacts** per the schema in `standards/benchmark-commons/PROPOSAL.md`.

### What to Measure

| Metric | What it tells you | When it matters |
|--------|-------------------|-----------------|
| TTFT p50/p99 | Time to first token | Interactive users, agentic loops |
| TPOT p50/p99 | Inter-token latency | Reading speed, streaming UX |
| Throughput (tok/s) | Total system output | Batch, cost optimization |
| Tok/s/user | Per-user speed | User experience |
| $/M tokens | Unit economics | Business viability |
| Error rate | Reliability | Production readiness |

### Common Pitfalls

1. **Synthetic data disables prefix caching** — InferenceX uses random tokens, 0% cache hit. Real multi-turn workloads see 60-90% hit rates. Always benchmark both.
2. **Single-stream != production** — A model doing 110 tok/s at batch=1 tells you nothing about throughput at batch=256.
3. **Cold vs warm** — First request after deploy includes JIT/compile overhead. Always discard warmup.
4. **Output length variance** — Reasoning models (R1, QwQ) produce 10-100x output variation. Use real reasoning prompts, not fixed output length.
5. **Concurrency sweep is mandatory** — One QPS point is meaningless. Sweep from 1 to saturation to find your Pareto frontier.

### Storing Results: Enriched Artifacts

Every benchmark run should produce an **enriched artifact** per `standards/benchmark-commons/PROPOSAL.md`:

```
blueprints/<name>/results/
├── qwen3-235b_eks_p6-b300_vllm_concurrency-sweep_20260422.json
├── qwen3-235b_eks_p6-b300_vllm_coding-agent_20260422.json
└── benchmark.yaml   ← sidecar: portable deployment context
```

The artifact contains: envelope (schema, timestamp) + portable context (model, engine, infra, workload) + core metrics (TTFT/TPOT/throughput at each concurrency) + SLO evaluation (pass/fail) + extensions (KV cache stats, GPU telemetry). This format enables cross-blueprint and cross-team comparison.

### Reference: Our Measured Baselines

| Model | Instance | Engine | Config | Peak tok/s | $/M tok |
|-------|----------|--------|--------|-----------|---------|
| Qwen3-235B FP8 | p6-b300 | vLLM | TP4 | 11,820 @ c=512 | $0.39 |
| Kimi K2.6 INT4 | p6-b300 | vLLM | TP8 | 10,437 @ c=512 | $0.43 |
| GLM-5 FP8 | p6-b200 | SGLang | TP8+HiCache | 2,602 @ c=128 | ~$1.50 |
| GLM-5 FP8 | p6-b200 | vLLM | TP8+MTP | 2,375 @ c=128 | ~$1.60 |

---

## 11. KV Cache Benchmarking

The InferenceX baseline uses random single-turn data with prefix caching disabled — this **undersells real-world performance** by up to 103x on TTFT (our measured Kimi K2.6 prefix caching result). A complete benchmark must characterize KV cache behavior under production-like workloads.

### What to Measure

| Metric | Tool / Source | Why it matters |
|--------|--------------|----------------|
| **Prefix cache hit rate** | Engine `/metrics` (vLLM: `vllm:cache_hit_rate`) | Determines effective TTFT — 90% hit rate means 90% of requests skip prefill |
| **KV cache utilization %** | Engine `/metrics` (vLLM: `vllm:gpu_cache_usage_perc`) | Shows when you're capacity-constrained vs compute-constrained |
| **Block eviction rate** | Engine `/metrics` + custom scrape | High eviction = thrashing, need more HBM or HiCache |
| **HiCache tier latency** | Custom probe (time CPU→GPU block load) | Determines warm-cache TTFT penalty vs full-cache hit |
| **NVMe offload latency** | Custom probe (time NVMe→GPU) | Determines cold-cache TTFT; ~7 GB/s NVMe = ~70ms for 500MB KV |
| **Memory pressure curve** | Sweep concurrent sequences at fixed QPS | Find the concurrency cliff where evictions spike |
| **Prefix reuse ratio** | Count unique vs shared prefixes in dataset | Predicts caching benefit for workload |

### Benchmark Design for KV Cache

```yaml
# Multi-turn chat (high cache hit potential)
dataset:
  type: multi_turn
  source: allenai/WildChat-4.8M  # Real conversations
  turns_per_session: 3-8
  system_prompt: shared_across_sessions  # Prefix reuse

measurements:
  - cold_start_ttft: first request, empty cache
  - warm_ttft: same system prompt, new user turn
  - hot_ttft: repeated conversation continuation
  - eviction_onset: concurrency where cache_usage > 95%
  - degradation_curve: throughput vs concurrency past eviction_onset

# Agentic coding (extreme prefix reuse)
dataset:
  type: agentic
  system_prompt_tokens: 12000  # Claude Code-like (~22K tokens)
  tool_definitions: 20
  turns_per_session: 10-30
  inter_turn_delay: 5-30s  # Simulates human + tool execution time

measurements:
  - all of the above
  - kv_retention_across_turns: does KV survive between tool calls?
  - prefix_dedup_efficiency: N agents sharing same system prompt
```

### KV Cache Sizing Validation Protocol

Before deploying, validate your KV budget actually supports target concurrency:

```
1. Deploy model at target TP/config
2. Monitor: vllm:gpu_cache_usage_perc, vllm:num_requests_running
3. Ramp concurrency: 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 → 256 → 512
4. At each level, hold for 60s, record:
   - Cache utilization %
   - Eviction rate (if available)
   - TPOT p99 degradation
   - Request rejection rate (503s)
5. Find the "knee": where cache utilization > 90% AND TPOT p99 starts climbing

The knee is your true max_concurrent_seqs — NOT the theoretical calculation.
```

### HiCache / CPU Offload Benchmarking

When using hierarchical caching:

| Test | What it reveals |
|------|-----------------|
| Disable HiCache, run to saturation | Baseline: pure HBM capacity limit |
| Enable HiCache, same workload | How much concurrency headroom offloading adds |
| Multi-turn with pauses | Whether HiCache retains KV across idle periods |
| Burst after idle | CPU→GPU reload latency under production patterns |

**Key gotcha from our GLM-5 work**: `hicache-size` must exceed the device KV pool size, otherwise the offload tier adds latency without adding capacity. Benchmark BOTH with and without to validate the tier is net-positive.

### Reference: Cache Hit Rates by Workload Type

| Workload | Expected hit rate | TTFT impact |
|----------|------------------|-------------|
| Single-turn random (InferenceX baseline) | 0% | None (baseline) |
| Multi-turn chat (shared system prompt) | 60-80% | 3-5x TTFT reduction |
| Agentic coding (same tools, repeated context) | 80-95% | 10-100x TTFT reduction |
| RAG with shared corpus chunks | 40-70% | 2-4x TTFT reduction |
| Batch eval (unique prompts) | 5-15% | Minimal |

**Bottom line**: If you're benchmarking with random data and reporting TTFT, your numbers are 10-100x worse than production reality for agentic/chat workloads. Always benchmark BOTH to establish the range.

---

## 12. Kernel-Level Optimization (From Our Profiling)

The highest-leverage software optimizations happen at the kernel level. Our kernel optimization agent work on Kimi K2.6 (384-expert MoE + MLA, H200) produced concrete data on where time actually goes.

> **⚠️ Every "near-optimal" / "no headroom" claim is scoped to a regime — never inherit it across regimes.** A kernel verdict is only valid for the exact tuple **hardware × kernel × concurrency × quantization × phase (prefill vs decode)**. The K2.6 findings below were measured on **H200 (sm_90), FP8, at decode (bs=1–128)**. They say nothing about the same model on Blackwell, in NVFP4, or during prefill. Concrete example of why this matters: the agent found K2.6's **FP8 MoE dispatch is near-optimal at decode** (4.6% BW util is architectural, and MoE is only 3% of c=128 compute → <1% e2e headroom). This does **not** contradict the pending **CuTe-DSL NVFP4 MoE-GEMM** "up to 2×" claim — that is a *different kernel* (NVFP4 GEMM, not FP8 dispatch) on *different hardware* (B200/B300 sm_100/103) in a *partly compute-bound* regime, and it is **unverified because the upstream merge has not landed** (FlashInfer #3645 open, SGLang #28354 draft — see `domains/gpu-serving/blueprints/kimi-k2.6-cutedsl-moe/`). Same model name, otherwise nothing in common. Before quoting a kernel verdict, restate its regime; before acting on one, re-profile in *your* regime.

### Where Time Goes (CUDA Graphs Enabled)

**At decode bs=1 (latency optimization):**

| Component | % of TPOT | Optimization lever |
|-----------|-----------|-------------------|
| RMSNorm (x183) | 36% | Fuse with adjacent ops |
| MLA Decode (x61) | 21% | Near BW-optimal — limited headroom |
| MoE FFN (x61) | 15% | Expert prefetch, TMA |
| Q/KV Projection (x61) | 13% | Fuse with RMSNorm |
| Router top-8/384 (x60) | 12% | Flat routing makes this cheap |

**At decode bs=128 (throughput optimization):**

| Component | % of TPOT | Optimization lever |
|-----------|-----------|-------------------|
| **MLA Decode (x61)** | **81%** | Memory-bound at ceiling — more GPUs or offload |
| RMSNorm (x183) | 9% | Marginal |
| MoE FFN (x61) | 3% | Compute-bound, already efficient |
| Q/KV Projection (x61) | 3% | Marginal |
| Router (x60) | 3% | Already efficient |

**Key insight**: At production concurrency, MLA attention dominates. At low batch (agentic single-stream), overhead is spread more evenly.

### CUDA Graphs: The Single Biggest Optimization

| Metric | Without CUDA Graphs | With CUDA Graphs | Impact |
|--------|--------------------|--------------------|--------|
| TPOT @ bs=1 | 112 ms | 8.9 ms | **12.6x** |
| Throughput @ c=1 | 7.4 tok/s | 110 tok/s | **14.9x** |
| Throughput @ c=128 | 733 tok/s | 3,844 tok/s | **5.2x** |

73% of the time without CUDA graphs was **scheduling/launch overhead**, not actual compute. CUDA graphs eliminate this by replaying the entire forward pass as a single GPU submission.

**Practical implications**:
- vLLM: CUDA graphs enabled by default. `--enforce-eager` disables them (useful for debugging, terrible for production)
- SGLang: Also enabled by default
- If you see unexpectedly low single-stream tok/s, check that CUDA graphs are active

### MoE Dispatch: Bandwidth Underutilization

At decode (bs=1-8), MoE expert dispatch only uses **4.6% of HBM bandwidth**:

```
384 experts × 2048 intermediate × 7168 hidden × 2 (up+down) × 1 byte (FP8)
= ~11.2 GB per layer, but only 8/384 experts loaded per token
= ~233 MB actually loaded per layer per token
at 0.289 ms/layer → 153 GB/s effective (vs 3,350 GB/s H200 peak)
```

This is because expert weights are scattered in memory — no spatial locality, no prefetch overlap. Opportunities:
1. **TMA async prefetch**: Start loading next-layer experts during current-layer compute
2. **Expert weight reordering**: Group frequently-co-activated experts contiguously
3. **Persistent kernels**: Keep expert data in registers/shared memory across tokens (FlashMoE approach)

### MLA Decode: Near Hardware Ceiling

At high batch (c≥32), MLA attention reaches **102-122% BW utilization** (>100% = L2 cache amplification helping). This means:

- **Cannot be further optimized on the same hardware** at high batch
- Only escape hatches: more GPUs (EP/TP to distribute KV), KV compression, or architectural changes (MLA already IS the compression — kv_lora_rank=512 instead of full kv_dim=7168)
- At low batch (c=1), only 21.8% BW util → headroom exists for better memory access patterns

### Practical Implications for Benchmarking

1. **Always enable CUDA graphs in benchmarks** — `--enforce-eager` makes results 5-15x worse than production
2. **MoE kernel configs matter** — a missing autotuning config (our K2.6 finding) causes +23% regression with no visible error
3. **Profile before optimizing** — intuition says "MoE dispatch is the bottleneck for MoE models" but data shows MLA attention dominates at production batch sizes
4. **Batch size changes the bottleneck** — memory-bound at bs=1 (optimize scheduling, launch overhead, BW) vs compute-bound at bs=512 (optimize arithmetic intensity, fusion)
5. **Engine overhead is 73% at low batch** — for agentic workloads (bs=1-4), the scheduling framework matters more than the kernel

### What This Means for Your Benchmark Methodology

| If your workload is... | The bottleneck is... | Optimize... |
|------------------------|---------------------|-------------|
| Agentic (low concurrency) | Scheduling overhead + MoE dispatch | CUDA graphs, launch latency, expert prefetch |
| Chat (moderate concurrency) | MLA decode (BW-bound) | More GPUs (wider TP/EP), HiCache, prefix caching |
| Batch (high concurrency) | MLA decode (at HW ceiling) | More GPUs, FP4 (smaller KV), architectural batching |
| Mixed (variable concurrency) | Different per-request | Dynamic batching, disagg P/D (isolate regimes) |

---

## 13. Optimization Composition Matrix

Not all optimizations compose well. Some conflict, some are redundant, some multiply.

### Composes Well (Multiplicative or Additive)

| A | B | Interaction |
|---|---|-------------|
| Prefix caching | TP8 single-node | Independent — caching reduces prefill, TP handles decode |
| CUDA graphs | Prefix caching | Independent — graphs accelerate decode, caching skips prefill |
| EAGLE3/MTP | Prefix caching | Multiplicative — spec decode accelerates remaining decode after cache hits |
| HiCache | Long-context workloads | Additive — extends KV capacity for retained multi-turn context |
| FP8 quantization | Any parallelism | Independent — smaller weights help everywhere |

### Conflicts or Redundant

| A | B | Problem |
|---|---|---------|
| **Disagg P/D** | **Prefix caching** | Redundant — prefix caching eliminates the prefill disagg was meant to offload |
| **EAGLE3** | **DCP (Decode Context Parallel)** | Conflicts — workspace allocation failures, deadlocks (active bugs) |
| **EAGLE3/MTP** | **High concurrency (c>64)** | Diminishing — verification overhead grows, acceptance rate drops |
| **EP** | **Single-node TP8** | Conflicts — EP dispatch overhead exceeds GEMM gains within one node |
| **HiCache** | **High concurrency** | Redundant — bottleneck shifts to compute, not KV capacity |
| **Disagg P/D** | **Single-node TP>1** | Conflicts — NVLink contention between AllReduce and KV transfer |
| **Dynamic MLA routing** | **Long prefill (>4K)** | Redundant — only helps short prefills where MHA is faster than MLA |

### Optimization Priority Order (Agentic Coding on Single Node)

Apply in this order — each subsequent optimization has decreasing marginal value:

```
1. CUDA graphs (14.9x) — always on, free
2. Correct MoE tile config (+23%) — check config exists for your GPU
3. Prefix caching (103x TTFT for repeated prefixes) — always on
4. EAGLE3 speculative decode — TESTED on Kimi K2.6 + Qwen3-235B (B300). +136% single-stream on K2.6 (`s4_d4_k1`); net-negative past c≈256 with stock draft. Real ShareGPT accept rate 0.156 vs 1.00 synthetic — draft fine-tuning on production traffic is the unlock, not the algorithm. See §3.
5. Dynamic MLA/MHA routing (expected 2-3x TTFT for <1K) — HIGH PRIORITY UNTESTED
6. HiCache (only if KV pressure at target concurrency)
7. Disagg P/D (only if multi-node or prefix cache miss rate >40%)
```

Item 5 is the highest-leverage untested optimization for K2.6 coding agents. Item 4 is realized only with a workload-tuned draft.

---

## 14. Scale Thresholds: When Multi-Node Optimizations Become Relevant

Single-node optimizations (TP, prefix caching, EAGLE3) cover most deployments. Multi-node architectures (disagg, EP across nodes, fleet routing) add complexity and only pay off at specific scale thresholds.

### Quick Decision: Do You Need Multi-Node?

```
Single node serves:
  K2.6 B300:  ~20 req/s sustained (512 output avg) = ~10,000 tok/s
  Qwen3-235B B300 TP4: ~23 req/s sustained = ~11,800 tok/s
  GLM-5 B200: ~5 req/s sustained = ~2,600 tok/s

If your peak QPS < single-node capacity with 30% headroom → stay single-node.
```

### Concrete Thresholds

| Optimization | Becomes Relevant At | Why Not Before | AWS Example |
|---|---|---|---|
| **Horizontal replicas** (simplest) | >70% single-node saturation (~14 req/s for K2.6) | Below this, single node handles load with latency headroom | 2-4 B300 nodes behind ALB |
| **Disagg P/D** (separate prefill/decode pools) | >50 req/s AND prefix cache miss rate >40% AND prefills >4K tokens | Below this, prefix caching handles prefill; disagg adds latency for no gain | Prefill: 2x p5e (compute-optimized), Decode: 4x p6-b300 (BW-optimized) |
| **Wide EP** (experts across nodes) | Batch >128 sustained AND model has >128 experts AND IB/NVLink available | Below batch 128, expert imbalance dominates (too few tokens per expert) | 4-node p5e NVL72 with EP32 |
| **Fleet routing** (multi-tier, priority queues) | >100 req/s mixed priority OR >3 model variants served | Below this, single-queue scheduling is simpler and equally fast | EKS + Karpenter + llm-d gateway |
| **Cross-model multiplexing** (model swapping) | >5 models, bursty traffic, <30% avg utilization per model | Below this, dedicated instances per model are simpler and avoid swap latency | GPU time-sharing with MIG or MPS |
| **Mooncake/NIXL remote KV** | >500 concurrent sessions at 128K context OR cross-region DR | Below this, local HiCache (NVMe) handles KV overflow | ElastiCache + NIXL LIBFABRIC |

### Scale Math (Work Through Your Numbers)

```
Step 1: Single-node capacity
  peak_tok_s = benchmark result (e.g., 10,437 for K2.6 B300)
  avg_output_tokens = your workload average (e.g., 512 for coding agent)
  max_qps = peak_tok_s / avg_output_tokens = 10,437 / 512 ≈ 20 req/s

Step 2: Required QPS
  concurrent_users × requests_per_user_per_minute / 60
  Example: 50 coding agents × 4 requests/min / 60 = 3.3 req/s → single node

Step 3: Headroom
  Leave 30% headroom for latency (p99 degrades as utilization → 100%)
  Usable capacity = max_qps × 0.7 = 14 req/s

Step 4: Nodes needed
  nodes = ceil(required_qps / usable_capacity)
  Example: 60 req/s / 14 = 5 nodes (simple replicas, no disagg needed)
```

### At What User Count Do You Need What?

| Users (coding agents) | Reqs/min (4 req/user/min) | Required QPS | Architecture |
|---|---|---|---|
| 1-10 | 4-40 | 0.07-0.67 | Single B300, EAGLE3 |
| 10-50 | 40-200 | 0.67-3.3 | Single B300, EAGLE3 (plenty of headroom) |
| 50-200 | 200-800 | 3.3-13.3 | Single B300 at limit; 2 replicas for safety |
| 200-500 | 800-2000 | 13.3-33 | 2-3 B300 replicas behind ALB |
| 500-2000 | 2000-8000 | 33-133 | 5-10 replicas; consider fleet routing for priority |
| 2000+ | 8000+ | 133+ | Fleet with disagg, priority queues, potentially EP |

### The Baseten/Fireworks Scale

Inference providers operate at >10,000 req/s across many models. At that scale:

| Technique | Threshold they hit | Our equivalent |
|---|---|---|
| Multi-tier serving (hot/warm/cold) | >10 models with bursty demand | Not needed — we serve 1-3 models |
| Custom draft models per traffic pattern | >1M req/day on single model | Not needed — standard EAGLE3 suffices |
| Disagg with heterogeneous SKU pools | >100 req/s per model AND mixed prefill lengths | Potentially relevant at 500+ users |
| Cross-request batching (share compute across users) | >50 req/s sustained | Already handled by continuous batching |
| Model weight caching across requests | Serving 50+ model variants | Not relevant for dedicated deployments |

**Key insight**: Most enterprise deployments (1-200 coding agents) are well within single-node capacity with EAGLE3. Multi-node becomes relevant only at 200+ concurrent users or when SLA requirements demand redundancy.

### Right-Sizing at Each Scale

| Scale | Cheapest config that meets SLA | Monthly cost (spot) |
|---|---|---|
| 1-50 users | 1x B300 spot + EAGLE3 | ~$11K |
| 50-200 users | 2x B300 spot + ALB | ~$22K |
| 200-500 users | 3x B300 spot + llm-d routing | ~$33K |
| 500+ users | Consider Bedrock/managed service (operational overhead > savings) |  API pricing |

At each tier, exhaust single-node optimizations (EAGLE3, prefix caching, dynamic MLA routing) before adding nodes. Adding 60% throughput via EAGLE3 is equivalent to avoiding an entire extra node ($11K/month).

---

## 15. Attention Family Taxonomy

**Read this first when a new model lands.** The attention family a model uses determines its serving behavior more than any flag: whether KV is prefix-cacheable, whether HiCache/LMCache/disagg work at all, where the decode bottleneck sits, and which kernels even exist on your GPU. This section maps the families we serve, which models use each, and the serving consequence. It exists because the mechanism is easy to confuse with the vendor — e.g. "MiniMax Sparse Attention (MSA)" is a MiniMax *project*, but the MiniMax-M2 we serve is dense GQA and does not use it.

`SERVING_COMPAT_MATRIX.md` (§Attention Backends) covers the *kernel implementations* (FlashAttention, FlashInfer, FlashMLA, Triton) per GPU. This section covers the *architectural families* those kernels serve. Read them together: family here → backend there.

### The Families

| Family | Mechanism | KV / state | Prefix-cacheable? | HiCache / LMCache / disagg | Decode bottleneck | Models we serve |
|---|---|---|:---:|:---:|---|---|
| **Dense MHA / GQA** (± qk_norm) | Full softmax over all past tokens; GQA shares KV across head groups | Full KV, grows with context | Yes | Yes | KV bandwidth; for MoE models the MoE dispatch is the cost center | **MiniMax-M2** (GQA + per-layer qk_norm), **Qwen3-235B** (GQA) |
| **MLA** (Multi-head Latent Attention) | KV compressed to a low-rank latent (`kv_lora_rank`) then projected up | Small latent KV (20-28× smaller) | Yes | Yes | Memory-bound; **at HW ceiling at high batch** (§12) — MLA already IS the compression | **Kimi K2.x**, **DeepSeek V3/R1**, **GLM-5** (MLA + NSA) |
| **Sparse attention** (top-k / NSA / DSA) | Score a cheap proxy, attend to only the top-k KV blocks | Full KV stored, sparse *read* | Yes | Yes (KV is standard) | Query-dependent block selection overhead | GLM-5 (NSA path). **MSA target: no served model uses it yet** |
| **Linear / gated-delta / DeltaNet** | Recurrent fixed-size state replaces softmax KV; delta-rule erase/write | **Fixed-size recurrent state — NOT KV** | **No (state layers)** | **No — incompatible with all KV-transfer connectors** | Prefill slower; no KV to be BW-bound on | **Qwen3-Next** (hybrid gated-delta + full attention) |
| **SSM / Mamba-2 hybrid** | State-space recurrence interleaved with a few full-attention layers | Fixed-size SSM state + KV for attention layers only | **Partial (attention layers only)** | **No — recurrent state blocks HiCache/LMCache/NIXL** | Sub-quadratic long-context prefill is the win | **Nemotron-3-Super / -Ultra** (Mamba-2 + LatentMoE + Select-Attention) |
| **Sliding-window + full hybrid** | Local window on most layers, full attention on a few | Bounded local KV + full on global layers | Yes | Yes | Long-context prefill penalty on global layers | Gemma-4 (local + global) |

### The Serving Rules That Fall Out of This

1. **Recurrent-state families (linear/gated-delta, Mamba-2 hybrid) break KV offload and disaggregation.** The fixed-size recurrent state is not a KV cache — HiCache, LMCache, NIXL, and disagg P/D all assume transferable KV pages and silently fail or degrade. This is why the Nemotron specs forbid HiCache and disagg (nemotron-super lessons #1/#2/#17/#21). Prefix caching helps only the interleaved attention layers, so expect a cold/warm TTFT ratio between 1× and 2×, not the ~100× you get on a pure transformer.
2. **MLA decode is at the hardware ceiling at production batch (§12).** No kernel rewrite helps; the only escape hatches are more GPUs (distribute KV via TP/EP), FP4 (smaller KV), or accepting it. Don't spend a kernel-optimization budget here.
3. **Dense-GQA MoE models are MoE-bound, not attention-bound.** For MiniMax-M2 and Qwen3-235B the attention is ordinary; the levers are MoE backend (FlashInfer vs Triton), qk-norm fusion, and speculative decode — not attention sparsity. A sparse-attention kernel (MSA) has nothing to accelerate on a dense model.
4. **Sparse-attention kernels need sparse-native weights.** MSA-style top-k attention only applies to a model whose config/weights invoke the sparse path. You cannot bolt it onto a dense checkpoint. As of this writing no model we serve uses MSA.
5. **A family is a training-time choice, not a swap-in.** Gated DeltaNet-2 (NVlabs, [arXiv 2605.22791](https://arxiv.org/abs/2605.22791)) beats Mamba-2 on RULER long-context retrieval at matched state size, making it a plausible *successor* architecture for a future Nemotron generation — but the weights are not interchangeable, and it inherits the same recurrent-state serving constraints as rule 1. Track it as an architecture watch-item, not a kernel to merge.

### Watch-Items (architectures not yet servable but worth tracking)

| Candidate | Family | Why track it | Status |
|---|---|---|---|
| **MSA** (MiniMax Sparse Attention) | Sparse top-k | Sparse-attn + FP4/NVFP4 indexer kernels; sm_120 (g7e) port exists (chutesai fork) | No released sparse MiniMax checkpoint confirmed; kernel library + microbenchmarks only |
| **Gated DeltaNet-2** | Linear / gated-delta | Mamba-2 successor; best-in-class RULER retrieval at matched state; relevant to the Nemotron Mamba-2 hybrid family | Research/training repo (NVlabs; chutesai added sm_120 training kernels). No servable checkpoint |

---

## Appendix A: Quick Reference — Engine Launch Flags

### vLLM (Qwen3-235B, B300 TP4)
```bash
vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --tool-call-parser hermes
```

### SGLang (GLM-5, B200 TP8 + HiCache)
```bash
python -m sglang.launch_server \
  --model THUDM/glm-5-0520 \
  --tp 8 \
  --enable-hierarchical-cache \
  --hicache-size 100 \
  --host 0.0.0.0
```

### vLLM (GLM-5, B200 TP8 + MTP)
```bash
vllm serve THUDM/glm-5-0520 \
  --tensor-parallel-size 8 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 1
```

---

## Appendix B: The Throughput-Interactivity Curve (Mental Model)

```
Throughput (tok/s/GPU)
     │
     │    ╭── Wide EP + Disagg + FP4 + MTP (production frontier)
     │   ╱
     │  ╱   ╭── Disagg + EP8 + FP8 + MTP
     │ ╱   ╱
     │╱   ╱   ╭── Single-node EP8 + FP8
     │   ╱   ╱
     │  ╱   ╱   ╭── Single-node TP8
     │ ╱   ╱   ╱
     │╱   ╱   ╱
     ├───┴───┴───┴──────────────────────→ Interactivity (tok/s/user)
     0   20  40  60  80  100  120  150

Each optimization pushes the Pareto frontier outward.
You pick your point on the curve based on use case.
```

---

## Appendix C: What's NOT Worth Optimizing

| Optimization | When to skip |
|--------------|-------------|
| Wide EP | Single-node, low batch, PCIe topology |
| FP4 | Model lacks QAT weights, accuracy-critical, AMD hardware |
| Disagg prefill | Low QPS, short contexts, single-node TP>1 without IB |
| KV offload to NVMe | Workloads completing in <30s (KV freed anyway) |
| CUDA graphs | Already enabled by default in vLLM/SGLang |
| Custom NCCL tuning | Unless you see actual collective bottlenecks in profiling |
| Chunked prefill tuning | Default chunk sizes are fine for 95% of workloads |

---

## Appendix D: Intelligence-Adjusted Cost (IAC) Framework

**Reference**: [Artificial Analysis Coding Agents](https://artificialanalysis.ai/agents/coding-agents) — the industry's first Pareto chart for model selection by cost/task vs pass rate.

### The Problem with $/M Tokens

Raw $/M tokens is misleading for model selection:
- A cheap model that fails 50% of the time costs MORE per successful outcome
- A verbose model that uses 3x tokens but succeeds on first attempt may be cheaper overall
- Prefix caching dramatically changes effective cost (cached tokens are 75-90% cheaper)

### The Right Metric: Cost per Successful Outcome

```
Cost per task = (input_tokens × $/input_token) + (output_tokens × $/output_token)
              - (cached_tokens × cache_discount)

Cost per success = cost_per_task / pass_rate

True cost (with human fallback) = cost_per_task + (1 - pass_rate) × human_intervention_cost
```

### Measured Data (Self-Hosted, Our Benchmarks)

| Model | SKU | $/M out tok | Tokens/task | Cost/task | Pass rate | **$/success** | **True cost** |
|---|---|---|---|---|---|---|---|
| K2.6 + EAGLE3 | B200 spot | $0.18 | ~50K | $0.009 | 80% | **$0.011** | **$5.01** |
| K2.6 + EAGLE3 | B300 spot | $0.28 | ~50K | $0.014 | 80% | **$0.018** | **$5.01** |
| K2.6 baseline | B300 spot | $0.43 | ~50K | $0.022 | 80% | **$0.027** | **$5.02** |
| Qwen3.5 122B | g7e TP4 | $0.15 | ~45K | $0.007 | 66% | **$0.010** | **$8.51** |
| Devstral 24B | g7e TP1 | $0.04 | ~10K | $0.0004 | 50% | **$0.0008** | **$12.50** |
| SERA-32B | g7e TP1 | $0.04 | ~15K | $0.0006 | 64% | **$0.0009** | **$9.00** |

**API comparison** (for context):

| Model | $/M out tok (API) | Tokens/task | Cost/task | Pass rate | **$/success** | **True cost** |
|---|---|---|---|---|---|---|
| Claude Opus 4.6 | $75 | ~30K | $2.25 | 79% | **$2.85** | **$7.50** |
| Claude Sonnet 4.6 | $15 | ~35K | $0.53 | 72% | **$0.73** | **$7.53** |
| GPT-5.5 | $30 | ~40K | $1.20 | 75% | **$1.60** | **$7.45** |
| Gemini 3.1 Pro | $10 | ~45K | $0.45 | 68% | **$0.66** | **$8.45** |

*True cost assumes human_intervention_cost = $25 (15 min developer review at $100/hr)*

### Key Insights

1. **$/success spans 3,500x** (Devstral $0.0008 → Claude $2.85) but **true cost only spans 2.5x** ($5.01 → $12.50). Human fallback cost dominates when pass rate < 70%.

2. **The Pareto frontier for model selection**:
   - If you optimize for **$/success** (automated pipeline, no human review): Devstral wins
   - If you optimize for **true cost** (human-in-the-loop): K2.6 wins (highest pass rate at low infra cost)
   - If you optimize for **developer experience** (latency + quality): Claude API wins (no infrastructure)

3. **Self-hosted is 100-250x cheaper per success than API** for equivalent-quality models. The value proposition of self-hosting is NOT cheaper tokens — it's that you can run thousands of tasks without budget constraints.

4. **Pass rate dominates cost at production scale**: Going from 50% → 80% pass rate saves more than any infra optimization (halves human intervention cost, which is 99%+ of true cost).

### The Pareto Chart (Model Selection)

```
Pass Rate (%)
100 │
    │                              ● Claude Opus ($2.25/task)
 80 │       ● K2.6+EAGLE3          ● GPT-5.5
    │         ($0.009/task)
    │                    ● Sonnet ($0.53/task)
 70 │            ● Qwen3.5         ● Gemini 3.1
    │              ($0.007)
    │
 60 │  ● SERA-32B ($0.0006)
    │
 50 │● Devstral ($0.0004)
    │
    └───────┬──────────┬──────────┬──────────┬───→ $/task
         $0.001      $0.01      $0.1        $1

Pareto frontier: Devstral → SERA-32B → Qwen3.5 → K2.6 → Claude Opus
(can't get higher pass rate without paying more)
```

**Models ON the frontier** (optimal): Devstral, SERA-32B, Qwen3.5, K2.6, Claude Opus
**Models BELOW the frontier** (suboptimal): Any model where a cheaper alternative has equal or better pass rate

### How Hardware Optimization Shifts the Frontier

Every infrastructure optimization (EAGLE3, prefix caching, B200 right-sizing) shifts self-hosted models LEFT on the chart (cheaper per task) without changing pass rate:

```
K2.6 B300 baseline:     $0.022/task, 80% → Pareto position: good
K2.6 B300 + EAGLE3:     $0.014/task, 80% → shifted LEFT (better)
K2.6 B200 + EAGLE3:     $0.009/task, 80% → shifted LEFT again (best self-hosted)
K2.6 B200 + full stack: $0.006/task, 80% → pushes frontier further left
```

This is why infrastructure optimization matters: it makes self-hosted models dominate more of the Pareto frontier, pushing the "break-even with API" point to lower and lower task volumes.

### Break-Even Analysis: Self-Hosted vs API

```
Monthly infra cost (B200 + EAGLE3): ~$8,760 (spot)
API cost for equivalent (Claude Opus): $2.25/task

Break-even: $8,760 / $2.25 = 3,893 tasks/month
            = ~130 tasks/day
            = ~16 coding agents running 8 tasks/day

Above 16 active coding agents: self-hosted wins.
Below 16: API is cheaper (no infra overhead).
```

For most teams with >20 developers using AI coding assistants, self-hosted K2.6 on B200 is economically dominant.

### Adding IAC to Benchmark Artifacts

The benchmark artifact schema (`standards/benchmark-commons/PROPOSAL.md`) should include cost metrics:

```yaml
# In the sidecar benchmark.yaml:
cost:
  instance_type: p5e.48xlarge
  spot_price_per_hr: 48.00
  utilization_target: 0.70  # 30% headroom for latency

# Derived in artifact (computed by adapter):
metrics:
  cost_per_m_output_tokens: 0.18        # $/M output tokens at target utilization
  cost_per_task:                          # Requires task profile:
    coding_agent:                         #   specific to workload type
      avg_tokens_per_task: 50000
      cost_usd: 0.009
  intelligence_adjusted:                  # Requires external pass rate data:
    pass_rate: 0.802                      #   from SWE-bench or harness eval
    cost_per_success: 0.011
    true_cost_with_human_fallback: 5.01   #   assumes $25 intervention cost
```

This lets `compare.py` compute IAC automatically when pass rate data is available, enabling direct comparison between hardware configurations AND between self-hosted vs API options.
