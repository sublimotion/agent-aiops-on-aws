# AI Inference Infrastructure Optimization — Tiered Decision Framework

## TL;DR — the 80/20

**80% of production GPU LLM workloads need only Tier 1 + Tier 2.**
**~15% need Tier 3 (cold-start) or Tier 4 (routing).**
**<5% need Tier 5 (disagg P/D, request migration, multi-node TP).**

If you reach for Tier 5 before Tier 1 is solid, you are buying complexity you cannot operate. Workload pattern is the gating factor at every tier — get the pattern wrong and the higher-tier optimizations *cost* you, they don't help.

---

## The five tiers

| Tier | Lever | Workload pattern that justifies it | Lab spec | Operational complexity |
|---|---|---|---|---|
| **1. Engine-level defaults** | Chunked prefill, prefix cache, continuous batching, right TP, FP8 where supported | All workloads | Implicit in every blueprint | Trivial — config flags |
| **2. Production basics** | Slim image, AMI match, health-check tuning, min-1-replica-per-region | All production workloads | Spec A + `warm-node-pool.md` rule | Low |
| **3. Cold-start strategies** | Compile cache (C-EBS), weights store (MX/B), snapshot (E), MX warm-pool | Spiky traffic, multi-replica autoscale, multi-tenant fleets | Specs A, B, C-EBS, E, regional-mx-warm-pool | Medium |
| **4. Capacity & routing** | GAIE EPP scorers, Dynamo Router, multi-region capacity-aware routing | Multi-replica, prefix-shareable, SLO tiers, capacity events | global-inference-gateway, cold-start-autoscaler-policy | High |
| **5. Frontier-model patterns** | Disagg P/D (NIXL/DistServe/Mooncake), request migration (Llumnix), multi-node TP, KV offload, MTP/spec-decode | Frontier MoE on multi-node TP, RL/agent harnesses with extreme tail | Existing serving blueprints (kimi-k2.6-speculative, dynamo-hyperpod, glm5-llmd) | Very high |

---

## Tier 1 — Engine-level defaults (free; do this always)

These are config flags or defaults in modern serving engines. Zero infrastructure cost. Skipping them invalidates every higher-tier measurement because you're optimizing on top of an unfair baseline.

| Lever | Mechanism | Typical win | Source |
|---|---|---|---|
| **Chunked prefill** | Splits long prefills into chunks interleaved with decode batches; shrinks request-cost variance the rest of the stack would have to work around | 2-3× improvement in p99 TTFT under mixed prompt-length workloads | Sarathi-Serve (OSDI '24) |
| **Prefix caching** | Hash-keyed KV-cache reuse across requests sharing prompt prefixes | 20-50% prefill saving on multi-turn / shared-system-prompt workloads | Preble (arXiv 2407.00023) |
| **Continuous batching** | Replaces static-batch padding with token-level batching; new requests join mid-decode | Throughput multiplier ~3× vs static batching | Orca / vLLM |
| **Right TP size** | Choose tensor parallelism based on model fit and interconnect — TP=1 wherever weights fit, more only when forced | Avoids unnecessary NCCL allreduce overhead | MEMORY.md "Single-node GPU deployments" |
| **FP8 where supported** | Sub-FP16 quantization for MoE / dense; check `moe_intermediate_size / TP % 128 == 0` rule | 2× memory, ~1.5× throughput on Blackwell | MEMORY.md "FP8 MoE TP rule" |

**Skip-the-tier consequences**: the cold-start, routing, and disagg specs all assume these are on. Comparing without chunked prefill is the most common methodology mistake — vendor-blog "5× routing improvement" usually has chunked prefill OFF in the baseline.

---

## Tier 2 — Production basics (most workloads need)

The minimum viable production deployment. Costs are bounded and well-known.

| Lever | When to use | Lab status |
|---|---|---|
| **Slim image (Spec A)** | Always. 1.87× faster image pull (77 s vs 144 s measured). | Validated, EBS-prebake recommended |
| **Hardware match** | Pin model_type, TP, dtype, and engine to a known-working hw + driver tuple. NCCL 2.26.2+ on Blackwell PCIe. AL2023 AMI on B200. | MEMORY.md tracks the gotchas |
| **Health-check tuning** | Set `readinessProbe.initialDelaySeconds ≥ 900s` for first-startup JIT-heavy stacks (DeepGEMM 15-16 min cold, B200/B300 GLM-5/Kimi). | Steering rule in tech-stack.md |
| **Min-1-replica-per-region** | Critical-path workloads: at least one replica always-on per region. Codified in `warm-node-pool.md`. | Steering rule |
| **Image pre-pull DaemonSet** | On the warm-pool node. Eliminates image-pull from cold-start path for replicas-on-warm-node. | Spec A composes with warm-node-pool rule |

**Workload patterns that DON'T need Tier 2**: throwaway dev/test, single-shot batch jobs that tolerate 5-10 min cold start, research notebooks. Production = needs Tier 2.

---

## Tier 3 — Cold-start strategies (workload-shape dependent)

This is where workload pattern starts dominating the decision. Pick the wrong pattern → wasted spend.

### Decision matrix

| Pattern | Recommended Tier 3 levers | NOT recommended | Reason |
|---|---|---|---|
| **Steady-state, single-replica always-on** | None (Tier 2 sufficient) | Snapshot, warm pool | The deployment IS the warm pool; double-paying |
| **Steady-state, multi-replica autoscale (rare events <1/day)** | Spec C-EBS compile cache | Snapshot | Cold-start frequency too low to amortize snapshot bake cost |
| **Spiky autoscale (>5 events/day same model)** | Spec A + B + C-EBS three-tier | Snapshot if engine ≠ vLLM | Snapshot only worth it if vLLM AND stack frozen ≥1 week |
| **Spiky autoscale + frozen stack ≥1 week + vLLM** | A + B + C-EBS + Spec E snapshot | — | Snapshot's 25-40 s floor is load-bearing; produce-path cost amortized |
| **Multi-tenant fleet (model-per-tenant)** | A + B + C-EBS three-tier; consider regional-mx-warm-pool if N tenants > 10 | Snapshot fleet-wide | 60-snapshot-matrix invalidation tax dominates |
| **Spot-reclaim recovery sensitivity** | regional-mx-warm-pool (N=2 cross-AZ) | — | Warm pool is the failover floor |
| **Scale-to-zero with infrequent events (<1/week)** | Snapshot if vLLM, three-tier otherwise | Warm pool | Snapshot's restore beats $1,900/mo always-on warm pool |

### Three-tier vs fused snapshot — the architectural choice

| | Three-tier (A + B + C-EBS) | Fused snapshot (E) |
|---|---|---|
| Artifact count for 10 models × 3 vLLM × 2 hw fleet | 16 | 60 |
| Invalidation surface | independent per layer | any layer triggers full rebake |
| Engine compatibility | vLLM, SGLang, TensorRT-LLM | vLLM only (preview) |
| Replica-N cold start (Ministral-class) | 50-120 s | 25-40 s |
| Operational complexity | Low | Medium-High |
| Best for | Default for any production workload | Spiky autoscale on frozen vLLM stack |

**Steering rule** (codified in `.claude/steering/tech-stack.md`): default to three-tier. Add snapshot only when ALL THREE conditions hold: stack frozen ≥1 week, autoscale events frequent enough that 25-40 s vs 50-120 s is load-bearing, engine is vLLM.

---

## Tier 4 — Capacity & routing (multi-replica + variance-sensitive)

Pull-based / capacity-aware routing only matters when you have backends to route between AND the workload has request-cost variance OR prefix locality OR SLO tiers. Round-robin is correct for single-replica or fully-fungible-request workloads.

### Decision matrix

| Pattern | Recommended router | Why |
|---|---|---|
| **Single replica or fully fungible requests** | k8s Service round-robin | Nothing to schedule against; routing is theatrical |
| **Multi-replica, no prefix sharing, batch-only** | k8s Service round-robin or AIBrix `least-request` | Prefix-aware routing's published wins are 2-3× on prefix-heavy; sub-30% on batch |
| **Multi-replica, multi-turn chat or agent harness with shared prompts** | GAIE EPP v1.5.0 with `prefix` + `kvcacheutilization` scorers | Preble territory; 2-3× p99 TTFT win on prefix-heavy slices |
| **Multi-replica + SLO tiers (interactive/batch/async)** | GAIE + `sloheadroomtier` filter + `latencyslo` admitter | Tier separation prevents batch traffic from blowing interactive p99 |
| **Multi-region + capacity-event resilience** | GAIE + Dynamo `global_router` | <30 s automatic failover vs minutes of manual cutover |
| **Heterogeneous fleet (B200 interactive + g7e batch)** | Custom GAIE scorer extending `EndpointAttribute` | Open research; nobody published this |

### What's measurable now vs gestures

- **Measurable** (vendor-published, reproducible): KV-aware routing, prefix-aware routing, SLO-tiered endpoints. Land these.
- **Gestures** (sparse literature, opportunity): online-learned latency predictors, heterogeneous-fleet routing, multi-region under correlated capacity events. Lab is uniquely positioned to measure.

### What's NOT in this tier

- **AWS-native ALB LLM-awareness**: doesn't exist. `aws-application-networking-k8s` v2.1.0 (May 2026) does not implement the Inference Extension. Don't reach for ALB hoping it'll do anything LLM-aware.

---

## Tier 5 — Frontier-model patterns (only at scale, only for the biggest models)

These techniques have real wins but each costs an engineer-quarter to operate. The 80/20 cutoff is steep here — most workloads should never reach for Tier 5.

| Technique | When it pays | When it's a trap | Lab status |
|---|---|---|---|
| **Disagg P/D (DistServe / NIXL / Mooncake)** | Frontier MoE on multi-node TP, separate prefill / decode SLOs, model > single-node fit | Anything that fits on one node; mixing P/D at small scale costs latency you don't recover | `dynamo-hyperpod` and `glm5-llmd` blueprints; NIXL disabled cuda_ipc default per MEMORY.md |
| **Request migration (Llumnix)** | Long-running RL / agent harnesses with extreme tail latency, >10 min generations | Short-prompt chat; migration overhead exceeds the worst-case stall it prevents | Open spec; OSDI '24 paper, controlled measurement at our scale missing |
| **Multi-node TP** | Models that cannot fit on largest single node (GLM-5 744B, Kimi K2.6 1T, Qwen3-235B beyond TP=8 fit) | Anything that fits TP≤8 on one node — multi-node NCCL overhead dominates | MEMORY.md tracks all the gotchas; HyperPod blueprints |
| **KV cache offload (Mooncake-style host/CXL)** | Long-context (>32 K) workloads with KV-bound throughput | Short-context; offload latency dominates re-fetch cost | LMCache + glm5-lmcache; MLA/NSA gaps tracked in MEMORY.md |
| **Speculative decoding (MTP / draft-target)** | High decode-dominated workloads on single-replica large models | PCIe-interconnected GPUs (degrades throughput per MEMORY.md); short responses; speculative on synthetic random data triggers degenerate "hello hello" repetition | MEMORY.md feedback `synthetic_specdec_repetition` |
| **GMS (GPUDirect Storage)** | Snapshot artifact > 100 GB and bandwidth is the cold-start bottleneck | Smaller artifacts; gated on CUDA driver patch | Deferred in dynamo-snapshot spec |

### Hard truth on disagg P/D

Disagg P/D is **only really needed for frontier workloads** — agreed. The signs you actually need it:
- Model is too big for any single-node TP fit (GLM-5 744B, Kimi K2.6 1T, frontier MoE)
- Prefill SLO and decode SLO are explicitly different (e.g., interactive prefill ≤200ms, decode tokens streaming throughput dominant)
- KV transfer over NIXL/RDMA is faster than re-prefilling (only true at very long contexts)

If your workload is "Ministral-3B / Qwen3-30B / Mistral-Small fits TP≤4 on one node," you do not need disagg. Pretending you do is the most expensive complexity tax in the lab.

---

## The 80/20 read

### What 80% of workloads actually need

```
Tier 1: chunked prefill + prefix cache + continuous batching + right TP + FP8 where supported
Tier 2: slim image + AMI match + health-check tuning + min-1-replica
```

Not Tier 3. Not Tier 4. Not Tier 5.

### When you cross into the 20%

You move past Tier 2 if **one of these is true**:
- p99 TTFT SLO < baseline cold start (50-120 s for small models, 16 min for GLM-5 class)
- Autoscale events fire often enough that cold-start cost is a meaningful fleet $/hr
- Workload has prefix-sharing structure that round-robin throws away
- You're doing multi-region or capacity-resilient deployment
- Model genuinely doesn't fit on a single node

If none of those apply, **stay at Tier 2**. The complexity tax of higher tiers will eat the engineering budget.

### When you cross into the 5% (Tier 5)

You move into Tier 5 only if:
- Model is genuinely frontier (>200B activations or >700B total params; GLM-5, Kimi K2.6, Qwen3-235B class)
- AND your workload can't be served acceptably by Tier 1-4 on the largest single node available
- AND you have an engineer-quarter to dedicate to operating it

For any other workload, Tier 5 is a research project, not an optimization.

---

## Workload pattern → tier ceiling

The ceiling is determined by workload pattern, not by ambition. Forcing higher-tier optimizations onto a workload whose pattern doesn't justify them produces negative ROI.

| Workload pattern | Tier ceiling | Why |
|---|---|---|
| Single-shot batch (research, eval, async) | Tier 2 | Cold start is amortized over the batch; routing is theatrical for single-process |
| Steady-state production chat (small model, single replica) | Tier 2 | Deployment IS the warm pool |
| Steady-state production chat (small model, multi-replica) | Tier 4 routing | Tier 3 cold-start optional, depends on autoscale frequency |
| Spiky multi-tenant SaaS | Tier 4 + Tier 3 (warm pool) | Both layers needed; multi-region adds Tier 4 routing |
| Agent / RL harness with long-tail generations | Tier 4 routing + maybe Tier 5 migration | Llumnix territory IF tail is severe |
| Frontier MoE production (GLM-5 / Kimi class) | Tier 5 | Multi-node TP, disagg P/D, KV offload all in play |

---

## Practical recommendations for our lab fleet

Based on the workloads in MEMORY.md:

| Workload | Pattern | Current tier | Recommended tier |
|---|---|---|---|
| Ministral-3B benchmarks | Steady-state research, single-replica | 2 | Stay 2 |
| Qwen3-next-fp8 (TP=4) g7e | Steady-state benchmarks | 2 | Stay 2 (Tier 3 if production traffic emerges) |
| GLM-5 / Kimi K2.6 (B200/B300) | Frontier MoE, large context | Currently Tier 1+2 | Tier 5 candidate (already partial via lmcache, llmd, hyperpod) |
| Agent harness / verifier-reward | Long-tail generations, multi-turn | Tier 1+2 | Tier 4 + Tier 5 migration candidate (Llumnix) |
| RL Conductor (Sakana repro) | Multi-worker, prefix-shareable | Tier 1+2 | Tier 4 routing candidate (open research) |
| Cost-aware-routing experiment | Cross-region cost arbitrage | spec stage | Tier 4 + cost dimension (composes with global-inference-gateway) |

---

## What this framework does NOT promise

- A specific number for your workload — workload pattern dominates; the ranges in this doc are typical, not guarantees.
- That higher tiers always help — they help only when the pattern justifies them. Tier 5 on small workloads costs more than it saves.
- That all tiers compose without conflict — they do compose well in this lab's measurements, but compounding wins are sub-linear (snapshot's 25-40 s floor is on top of the EBS+FSR ~19 s overhead, not in addition to baseline cold start).

## References

- Cold-start specifics: `domains/ai-infra/reports/cold-start-progress-report.html` (v5)
- Cold-start tutorial: `domains/ai-infra/reports/cold-start-explainer.html`
- Three-tier vs fused architecture: `.claude/steering/tech-stack.md` § "Cold-start: prefer three-tier separation"
- Routing literature snapshot: `domains/ai-infra/specs/global-inference-gateway-research.md`
- Operationalization (per-pool tier selection controller): `domains/ai-infra/specs/cold-start-autoscaler-policy.md`
- Frontier-model gotchas: MEMORY.md (B200/B300 sections, NCCL Blackwell, LMCache+MLA/NSA, FP8 MoE TP rule)
