# The Inference Optimization Stack

A six-tier reference for how to take an LLM serving deployment from baseline to production-ceiling. Each tier composes on top of the previous; each has a canonical configuration, a typical delta range (from this repo's measured blueprints), and a list of known conflicts.

Every blueprint's Stage 6 benchmark report should fill out the **Tier Stack Table** (see template below) stating which tiers landed, which were blocked, and the delta each tier delivered.

> **Companion docs**: [inference-optimization-guide.md](inference-optimization-guide.md) has the theory (roofline, Pareto, cost-per-success). This file is the operator's checklist.

## Where this doc sits in the loop

This catalog is the **generalized-guideline** rung of the optimization knowledge ladder — read it at config-selection time, feed measured results back into it after benchmarking:

- **Start of loop (spec Stage 0b):** predict the regime from [`.claude/steering/inference-first-principles.md`](../.claude/steering/inference-first-principles.md), then use the tier priorities + conflicts here to fill the spec's lever ledger. Account for every tier; defer with a reason, never skip silently.
- **End of loop (`compound-learner`):** the Stage 6 Tier Stack Table's measured deltas are fed back into this doc's typical-Δ cells, and any high-priority tier skipped without justification is flagged. See `.claude/agents/compound-learner.md` § "Optimization coverage refresh".

**Which rung absorbs a lesson** (invariance test — "still true after a framework version bump?"):

| Lesson kind | Home |
|-------------|------|
| Physics, rederivable from the roofline | `.claude/steering/inference-first-principles.md` (T1) |
| Technique-class, true across ≥2 models/frameworks | **this doc** (generalized lever) |
| Specific model/engine/version/instance | blueprint `lessons.md` + `mdc`/`gpu-infra` cards (T2/T3) |

A single datapoint is a card fact; promote to this doc only on the *second* occurrence across models.

## The six tiers

| T | Name | Character | Default recommendation (2026-05) |
|---|------|-----------|----------------------------------|
| **T0** | Baseline | Reference point; BF16, minimum-to-fit TP, no caching, no spec decode, eager mode | BF16, TP=min-to-fit, L0 cache |
| **T1** | Quantization | Shrink weights + activations + KV bytes to free compute and memory | FP8 (E4M3) weights + KV; INT4/NVFP4 if quality passes |
| **T2** | KV / prefix cache | Cut prefill work on repeated prefixes; spill KV when it becomes the bottleneck | L1 prefix cache ON by default; HiCache/LMCache when KV hit rate > 20% or context > 16K |
| **T3** | Speculative decode | Produce multiple tokens per BW-bound step when decode is the bottleneck | EAGLE3 (when draft available); MTP (when native); none otherwise |
| **T4** | Parallelism | Shape the replica and fleet for the workload | TP-to-fit + DP replicas; not pure TP=max |
| **T5** | Kernel / compile | Squeeze scheduling + kernel overhead out of the remaining margin | torch.compile + FLASHINFER_MLA (MLA) or FA3 (others) |

## Composition order

Tiers must be enabled in order. Each tier's memory and throughput effects feed into the next tier's feasibility.

```
T0 → T1 → T2 → T3 → T4 → T5
```

Why the order matters:

- **T1 before T2**: quantization halves KV bytes — cache sizing and spill thresholds depend on post-quant memory.
- **T2 before T3**: prefix cache changes prefill cost profile; spec decode only helps decode. Measure T2 delta before enabling T3 or you conflate the two.
- **T3 before T4**: spec decode runs per replica; sweep TP/DP after T3 is in place or you'll re-tune parallelism.
- **T4 before T5**: kernel tuning is replica-local — need the final replica shape to tune against.

## Canonical configuration per tier

### T0 — Baseline

**Goal**: honest reference. Most specs underspecify this and every "X.Y× speedup" claim becomes slippery.

```yaml
engine:
  precision: bf16
  kv_cache_dtype: bf16
  tensor_parallel: <minimum to fit the model in VRAM>
  pipeline_parallel: 1
  data_parallel: 1
  enable_prefix_caching: false
  attention_backend: <engine default>
  compile: false
  cuda_graphs: false
  speculative_decode: null
```

**Measure**: TTFT/ITL at c=1, c=32, c=saturation; tokens/s; VRAM/GPU; cold start.

---

### T1 — Quantization

**Goal**: free up memory for KV and enable higher batch sizes.

| Format | When to use | Typical throughput Δ | Typical memory Δ | Quality risk |
|--------|-------------|-----------------------|-------------------|---------------|
| **FP8 E4M3** (weights + KV) | Default. Every Hopper/Blackwell deployment | 1.5–2.0× | ~2× headroom | Low (< 1pp MMLU drop typical) |
| **INT8** | Older hardware without FP8 tensor cores | 1.3–1.7× | ~2× headroom | Low–medium |
| **INT4-GPTQ / AWQ** | When FP8 doesn't fit and quality allows | 2.0–2.5× | ~4× headroom | Medium–high; run quality gate |
| **INT4-QAT** | Pre-quantized checkpoints (Kimi K2.6) | 2.5–3.0× | ~4× headroom | Low (trained to this) |
| **NVFP4** | Blackwell-only; latest Tensor Cores | 1.15–1.25× on top of FP8 | same as FP8 | Low, but tooling immature |

**Canonical config**: FP8 E4M3 weights + FP8 KV cache. Quality-gate before accepting any non-BF16 precision (see `standards/benchmark-commons/container/run-quality-eval.py`).

**Blueprint evidence**: GLM-5 FP8 (744B fits TP=8 B200 at 175GB/GPU); Kimi K2.6 INT4 QAT (1T on TP=4 B300); Qwen3-235B FP8 (TP=4 on B300).

**Conflicts**:
- **Qwen3.5 MoE GPTQ-Int4**: broken on vLLM 0.18 — produces garbage even with gptq_marlin. FP8 works.
- **FP8 MoE TP divisibility**: `moe_intermediate_size / TP % 128 must == 0`. Fails on Qwen3-Next + TP=8.

---

### T2 — KV / prefix cache

**Goal**: cut prefill when prefixes repeat; spill KV to CPU/NVMe when working set > VRAM.

| Strategy | When to use | Typical Δ | Blueprint evidence |
|----------|-------------|-----------|--------------------|
| **L1 prefix cache** (in-GPU) | Always on. Default for every engine. | 1.2–1.8× on chat/agent traces | Qwen3-Next custbench: 82–94% TTFT gain |
| **HiCache** (SGLang CPU offload) | Long-context or multi-tenant workloads | 1.7–2.9× at c≥64 | GLM-5 SGLang: 2.86× peak |
| **LMCache** (vLLM CPU/NVMe offload) | Same workloads, vLLM engine | 1.5–2.5× (where supported) | **Currently blocked on MLA/NSA models** — check LMCache dev branch status |
| **Disagg P/D (NIXL)** | Prefix-heavy workloads at high concurrency, multi-node | 1.3–2× aggregate | **Not yet benchmarked in repo** |
| **Mooncake-style KV tiering** | Wide-EP, multi-node | — | Spec exists (`mooncake-kv-tiering.md`), no data |

**Canonical config**: enable L1 prefix cache unconditionally. Add HiCache or LMCache when a baseline run shows KV hit rate > 20% **or** context P95 > 16K tokens.

**Conflicts** — *every blocker below is a point-in-time upstream claim and decays. Before deferring a lever on one, re-verify against the live tracker (`gh pr view <N> --repo <repo>`, `gh issue list`). A merged PR lifts the blocker silently — nothing here will error.*
- **Hybrid KV (Qwen3-Next, Nemotron Mamba)**: HMA auto-disables when KV transfer connectors engage → HiCache/LMCache/disagg all blocked. <!-- stack: vllm=0.16 | validated: 2026-05-16 — RE-CHECK before use -->
- **MLA + LMCache** — *re-checked against LMCache tracker 2026-06-17; the old "blocked until #2951/#2629 land" framing was wrong:* <!-- stack: lmcache=dev | validated: 2026-06-17 -->
  - **vLLM path: MERGED, not blocked** — core MLA support landed via #1801 (Layerwise), #2032 (P2P), #2697 (multi-TP), #2935/#2941. PR #2951 is a *follow-up* multi-group bugfix (GLM-5/DeepSeek V3), still OPEN — not a gate on the feature. **Try it.** Expect GLM-5-specific rough edges: OPEN bugs #2774 (GLM-5 FP8 tensor-shape mismatch), #2977 (GLM-5 cache hit rate 0), #2881 (DeepSeek V3.2 shape mismatch), #3388 (degenerate output under load). Smoke-test cache-hit-rate and output quality before trusting it on GLM-5.
  - **SGLang path: genuinely still blocked** — issue #3192 (2026-05-28) confirms LMCache doesn't support MLA on SGLang; PR #2629 ("Add MLA to SGLangLayerwise") is the open fix, stale since 2026-05-07. Use SGLang **HiCache** for MLA models, not LMCache.
- **NSA + HiCache**: works on SGLang; NSA fused `kv_buffer` handled natively since mid-2026.

---

### T3 — Speculative decode

**Goal**: get more output tokens per BW-bound decode step; biggest lever for per-user latency on low-concurrency workloads.

| Method | When to use | Typical Δ | Cost |
|--------|-------------|-----------|------|
| **EAGLE3** | Draft weights available; Blackwell preferred | 2.5–4× per-user tok/s | 3–5% aggregate throughput hit |
| **MTP** (native) | Model ships with MTP head (GLM-5, Qwen3-Next) | 1.5–2.5× per-user | 0–5% aggregate |
| **Draft-model** | Tiny draft runs alongside | 1.5–2× per-user | 10–20% aggregate (draft GPU time) |
| **None** | Compute-bound workloads (c > 128), or no draft | 1.0× | — |

**Canonical config**: EAGLE3 when a trained draft is available (CoreWeave's K2.6 result used this); MTP when the model ships with native speculative heads; no spec decode for batch/throughput workloads at high concurrency.

**Blueprint evidence**: GLM-5 llm-d MTP (blueprint). Kimi K2.6-speculative spec written, EAGLE3 not yet benchmarked — this is the single highest-priority open comparison.

**Conflicts**:
- **PCIe-only topology (g7e)**: MTP hurts throughput on PCIe GPUs — verification overhead > draft savings without NVLink.
- **Spec decode + disagg**: partially supported; verify before combining.

---

### T4 — Parallelism strategies

**Goal**: shape the replica for latency vs throughput, then scale horizontally.

| Strategy | When to use | Typical pattern |
|----------|-------------|-----------------|
| **TP=min-to-fit + DP replicas** | Default. Throughput + latency balance | TP=4 replica on 8-GPU node → 2 replicas (DP=2) |
| **TP=max single replica** | Latency-critical, low-QPS | TP=8 on 8-GPU node |
| **PP (pipeline)** | Model too big for one node; bubbles acceptable | PP=2 across 2 nodes |
| **EP (expert parallel)** | Wide-EP MoE; expert count > TP degree | EP=32 across NVLink domain |
| **Disagg P/D** (orthogonal to TP) | Long-prefix workloads | Prefill TP=2 + Decode TP=2 via NIXL |

**Canonical config**: TP-to-fit (smallest TP that makes the model fit with headroom) + DP replicas to saturate the node. Pure TP=max is rarely optimal.

**Blueprint evidence**: Qwen3-235B TP=4+DP=2 ceiling on B300 (13,877 tok/s peak); TP=2+DP=4 peaks 1.71× above TP=4 single-replica. K2.6-speculative spec Phase 5C proposes PP=2×TP=4 (unmeasured).

**Conflicts**:
- **Wide EP requires NVLink domain**: EP=32+ needs NVL72 (GB200/GB300). On p6-b300 (8-GPU NVSwitch) EP is capped at 8.

---

### T5 — Kernel / compile

**Goal**: close the gap between engine scheduling and the hardware ceiling. The "last 15%".

| Optimization | When to use | Typical Δ |
|--------------|-------------|-----------|
| **torch.compile** | vLLM ≥ 0.8, SGLang compile mode | 10–20% at low batch, 5–10% at saturation |
| **FLASHINFER_MLA** | MLA models (DeepSeek, Kimi, GLM-5) | 15–25% decode throughput vs default |
| **FlashAttention-3** | Non-MLA models, Hopper+ | 10–20% decode |
| **CUDA graphs** | Decode-heavy workloads | 5–15% at low concurrency |
| **Overlap scheduler** | vLLM latest, scheduling bubbles | 10–30% when framework overhead dominates |

**Canonical config**: enable `torch.compile` + best attention backend for the architecture. CUDA graphs on for low-concurrency latency paths. Overlap scheduler on for agent/tool-calling workloads.

**Blueprint evidence**: Kimi K2.6 uses FLASHINFER_MLA on B300 (feeds the 10,437 tok/s ceiling). GLM-5 cold-start dominated by DeepGEMM JIT + torch.compile (16 min on first run).

**Conflicts**:
- **torch.compile + dynamic batching**: first compile stall can mask as a health-check failure; bump startup probe timeout.
- **DeepGEMM JIT**: adds 15 min to first cold start; cache to persistent volume.

## Tier stack table (required in every Stage 6 report)

Every blueprint's benchmark report fills this out. One row per tier.

| Tier | Config landed | Δ vs T0 (tok/s) | Δ vs T0 (TTFT p99) | Blocked? |
|------|---------------|------------------|---------------------|----------|
| T0 | BF16, TP=8, no cache | 1.0× (ref: 850 tok/s) | 1.0× (ref: 450 ms) | — |
| T1 | FP8 E4M3 + KV | 1.8× | 0.85× | — |
| T2 | L1 prefix + HiCache | 2.9× | 0.55× | — |
| T3 | — | — | — | ⚠️ EAGLE3 draft not available |
| T4 | TP=4 + DP=2 | 4.2× | 0.48× | — |
| T5 | torch.compile + FLASHINFER_MLA | 4.8× | 0.42× | — |

Leave rows empty (or mark "not attempted") for tiers not applied. Flag any blocker in its own row so the gap is visible.

## Running the comparison automatically

The six tiers map directly to sidecar configurations. Each tier gets one sidecar; the benchmark runner sweeps them via a tag-per-tier pattern:

```bash
for tier in t0-baseline t1-fp8 t2-prefix-hicache t3-eagle3 t4-tp4-dp2 t5-compile; do
  ./run-benchmark.sh \
    --endpoint http://svc:8000 \
    --workload concurrency-sweep \
    --sidecar bench/${tier}.yaml \
    --tag $tier
done
```

The resulting artifacts share model + hardware + workload, so `runner/compare.py` can produce the tier table above directly.

## What's still not known (per-model frontier)

- **EAGLE3 deltas** for any model we serve — need to run it once to get a data point.
- **NVFP4 throughput** on top of FP8 on B300 — one sidecar flip away.
- **Disagg P/D (NIXL) aggregate throughput** vs colocated TP — never measured.
- **Wide EP beyond NVL8** — requires GB200/GB300 NVL72 access.
- **Mooncake KV tiering** — spec exists, no data.

See `domains/gpu-serving/specs/kimi-k2.6-speculative.md` Phase 5 for the most up-to-date single-node frontier plan (T3 + T5 expansion).

## Version-dependence

The canonical configs above pin to **2026-05**. Each tier's recommendation is vulnerable to engine releases (vLLM, SGLang, TRT-LLM), kernel libraries (FlashInfer, DeepGEMM, cuDNN), and hardware firmware. The CLAUDE.md "Version Refresh" protocol re-validates this doc when any of those move.

When a new release lands, walk the tier list and check:
1. Does the canonical config still apply? (e.g. did `attention_backend` defaults change?)
2. Did new methods land in a tier? (e.g. a new speculative decode method)
3. Did any conflict go away? (e.g. LMCache + MLA unblocks)

Update this file as part of the Version Refresh, not ad-hoc.

## Links

- Theory: [inference-optimization-guide.md](inference-optimization-guide.md)
- Visual: [inference-optimization-visual.html](inference-optimization-visual.html)
- Workload cards: [standards/benchmark-commons/workloads/](../standards/benchmark-commons/workloads/)
- Schema: [standards/benchmark-commons/container/schema/enriched-artifact.json](../standards/benchmark-commons/container/schema/enriched-artifact.json)
- Roofline/Pareto: [domains/gpu-serving/blueprints/kimi-k2.6-speculative/docs/roofline-explainer.html](../domains/gpu-serving/blueprints/kimi-k2.6-speculative/docs/roofline-explainer.html)
