# Managed KV Cache Differentiation — Price/Performance Secret Sauce

## Status: DRAFT

## Overview

Managed KV cache is not a standalone feature — it is the foundation for a routing, scheduling, and elasticity stack that cannot exist without platform-level visibility into cache state. By treating KV cache as a shared, indexed service rather than per-instance ephemeral memory, HyperPod Managed Inference can deliver cross-request prefix routing, elastic prefill/decode disaggregation, and SLA-aware admission control — together achieving 2-3x better price/performance than OSS stacks on equivalent hardware.

## Problem Statement

Customers do not perceive ease-of-use or reduced TTM as sufficient differentiation for the HyperPod per-node premium. "I can run vLLM on EC2 myself" is the default objection. We need capabilities that are architecturally impossible without a managed platform layer — not just convenience features.

**Core insight**: Every advanced inference optimization (prefix sharing, P/D disagg, admission control, elastic scaling) requires a global view of KV cache state across instances. OSS solutions are per-instance or require complex manual orchestration. A managed cache service IS that global view.

## Differentiation Architecture

```
                    ┌─────────────────────────────────────┐
                    │     SLA-Aware Admission Controller    │
                    │  (knows cache residency + decode      │
                    │   queue depth across all replicas)    │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │     Prefix-Aware Request Router       │
                    │  (queries managed cache index before  │
                    │   dispatching — routes to cache hit)  │
                    └──────────────┬───────────────────────┘
                                   │
              ┌────────────────────┼────────────────────────┐
              ▼                    ▼                         ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
   │  Prefill Replica  │ │  Decode Replica   │ │  Decode Replica      │
   │  (GPU-heavy)      │ │  (memory-heavy)   │ │  (memory-heavy)      │
   └────────┬─────────┘ └────────┬──────────┘ └────────┬─────────────┘
            │                     │                      │
            └─────────────────────┼──────────────────────┘
                                  ▼
                    ┌─────────────────────────────────────┐
                    │       Managed KV Cache Service        │
                    │  (cross-instance, prefix-indexed,     │
                    │   predictive eviction, EFA-backed)    │
                    └─────────────────────────────────────┘
```

## Capability 1: Cross-Request Prefix Routing

### What

Route requests sharing a common prefix (system prompt, RAG document, few-shot examples) to the instance that already has that prefix's KV tensors cached. Compute the prefix ONCE, serve N requests.

### Why OSS can't do this

Requires a **global prefix index** across all serving instances with real-time cache residency tracking. OSS solutions (vLLM prefix caching, SGLang RadixAttention) are per-instance only — no cross-instance awareness. llm-d attempts this but is early-stage and requires manual setup.

### Platform lever

The managed KV cache service IS the prefix index. The router queries it in <1ms before dispatching. Multi-tenant: shared prefixes across workloads on the same endpoint.

### Expected impact

- 40-60% reduction in redundant prefill compute for RAG/agentic workloads
- TTFT improvement proportional to prefix length (32K prefix → ~10x TTFT reduction on cache hit)
- Higher effective throughput at same GPU count

## Capability 2: Elastic Prefill/Decode Disaggregation

### What

Dynamically scale prefill and decode replicas independently based on workload phase. KV state lives in the managed tier, so any decode instance can pick up where any prefill instance left off.

### Why OSS can't do this

P/D disaggregation in OSS (NIXL, Mooncake, DistServe) is architecturally fixed at deploy time. Changing the P:D ratio requires redeployment. Dynamic rebalancing needs:
- Live KV migration between instances (managed cache makes this a pointer swap, not a GPU-to-GPU copy)
- Admission control that knows decode queue depth
- Instance lifecycle management (spot reclaim, capacity blocks)

### Platform lever

Managed cache decouples KV state from compute. Prefill generates KV → writes to managed cache → ANY decode instance reads it. Scale each phase on independent autoscaling policies.

### Expected impact

- 2-3x better GPU utilization during bursty traffic (vs fixed P:D ratio)
- Seamless spot instance integration (decode instances can be reclaimed; KV state persists)
- Right-size compute: prefill on high-FLOPS instances (p5e), decode on high-memory instances (g7e)

## Capability 3: SLA-Aware Admission Control

### What

A platform-level admission controller that sees utilization across all replicas and makes routing/queuing decisions to maintain customer-defined SLA guarantees (e.g., p99 TTFT < 2s, p99 ITL < 50ms).

### Why OSS can't do this

vLLM/SGLang admit all requests and let latency degrade gracefully (or not). There is no SLA-aware admission. Customers who need latency guarantees must massively over-provision (3-5x) to handle tail cases.

### Platform lever

The admission controller knows:
- Current KV cache utilization per instance (from managed cache)
- Decode queue depth and estimated drain time
- Incoming request's expected KV footprint (from prompt length)
- Customer's SLA contract

Decision: admit, queue, or reroute — BEFORE the request hits the GPU.

### Expected impact

- Guaranteed latency at 70-80% utilization (vs OSS needing <40% for SLA compliance)
- 2x cost efficiency for latency-sensitive workloads
- Eliminates "noisy neighbor" problem in multi-tenant endpoints

## Capability 4: Custom CUDA Kernels (Managed Cache Integration)

### 4a. Fused Decode Attention with Managed KV Offload

Standard paged attention assumes flat GPU memory. Custom kernels that know about the managed cache tier can issue **async DMA prefetch** for KV blocks before the attention head needs them — eliminating cold-KV stalls.

### 4b. EFA-Aware Collectives

Custom allreduce/allgather kernels that exploit SRD packet ordering and multi-path EFA interfaces with topology-aware ring construction. ~15-20% better TP scaling than stock NCCL on 4+ GPU nodes.

### 4c. Predictive Cache Warming

Pattern detection on request streams → speculatively retain high-reuse prefixes, evict low-reuse. OSS cache policies are always LRU. Managed platform sees usage patterns across time and tenants.

## Capability 5: Agent Workload Unlock

### What

Agentic workloads (coding agents, research agents, tool-calling loops) have a unique access pattern: long-running sessions with **interleaved reasoning and tool execution**. During tool execution (bash commands, API calls, file reads), the GPU holds the session's KV cache hostage — allocated but idle. A managed platform can exploit these "GPU bubbles" to serve other requests, then seamlessly resume the agent session.

### The Agent Problem

```
Agent session (30 turns, ~5 min):

  Turn 1: [REASON 2s] [TOOL_EXEC 8s]  ← GPU idle 80% of wall time
  Turn 2: [REASON 3s] [TOOL_EXEC 12s] ← GPU idle 80% of wall time
  ...
  Turn 30: [REASON 1s] [DONE]

  GPU utilization for this session: ~20%
  KV cache held for entire 5 min: ~4-8 GB
```

With OSS: you either (a) keep KV resident and waste 80% of GPU time, or (b) evict and recompute on resume (TTFT penalty scales with context length — 32K context = 5-10s recompute).

### Platform lever

**Managed KV cache enables zero-cost session suspend/resume:**

```
Agent Turn N completes → REASON phase done → tool_call emitted
  1. Offload KV to managed cache tier (async DMA, <100ms for 8GB over EFA)
  2. GPU slots freed → backfill with other requests (chatbot, batch, other agents)
  3. Tool execution completes → agent needs Turn N+1
  4. Prefetch KV from managed cache (overlapped with request routing)
  5. Resume decode with zero recomputation
```

**ThunderAgent-style bubble filling, but without the fragility:**

| Approach | Session State | Backfill Capacity | Resume Cost |
|---|---|---|---|
| OSS (keep resident) | GPU memory | 0% | 0 (never freed) |
| OSS (evict + recompute) | Discarded | 100% | Full prefill (5-10s) |
| **HyperPod MI (managed cache)** | Managed tier | ~80% | <200ms (prefetch) |

### Why this matters NOW

Agentic workloads are the fastest-growing inference pattern:
- Claude Code: 30+ turns, 100K+ context, tool calls every turn
- Coding agents (SWE-bench): 10-30 turns, 65K context, 76% of time in tool execution
- Research agents: 50+ turns, multi-hour sessions, API calls between every reasoning step
- Customer support: multi-tool orchestration with database/CRM lookups

**At scale (1000 concurrent agent sessions), the managed cache approach serves 4-5x more agents on the same GPU fleet** because GPU time is never wasted holding idle KV state.

### Compound effect with other capabilities

| Combined with... | Agent-specific benefit |
|---|---|
| Prefix routing | All agents sharing same system prompt / tool definitions → single KV computation |
| P/D disaggregation | Prefill (first turn) on high-FLOPS GPU, decode (subsequent turns) on memory-optimized |
| Admission control | Admit new agent sessions only when managed cache has capacity; queue gracefully |
| Predictive prefetch | Learn agent tool execution patterns → pre-warm KV before tool returns |

### Quantified target

| Metric | OSS (keep-resident) | OSS (evict) | HyperPod MI |
|---|---|---|---|
| Concurrent agents per GPU | 2-4 | 8-12 (with recompute tax) | **10-15 (no tax)** |
| Effective GPU utilization | 20% | 80% (bursty) | **85% (smooth)** |
| Resume latency (32K ctx) | 0ms | 5-10s | **<200ms** |
| Agent throughput (issues/hr/GPU) | 3-5 | 3-5 (recompute eats gains) | **12-20** |

### Demo scenario

Run 100 concurrent SWE-bench coding agents (Qwen3.5-27B, 30 turns, 65K context) on 8×H200:
- **OSS baseline**: ~16 concurrent (KV-resident), 3 issues/hr/GPU
- **HyperPod MI**: ~80 concurrent (managed cache suspend/resume), 12 issues/hr/GPU
- **4x throughput improvement** at identical hardware cost

## Competitive Positioning

| Capability | OSS (vLLM/SGLang on EC2) | Managed Competitors | HyperPod MI |
|---|---|---|---|
| Prefix caching | Per-instance only | Fireworks (proprietary) | Cross-instance, multi-tenant |
| P/D disaggregation | Fixed at deploy | None at scale | Elastic, autoscaled |
| Admission control | None | Anyscale (basic) | SLA-guaranteed |
| Agent session mgmt | Keep-resident or evict | None | Suspend/resume <200ms |
| Cache intelligence | LRU eviction | Unknown | Predictive, pattern-aware |
| Custom kernels | Stock | Proprietary (Together) | EFA + managed cache fused |

## Quantified Differentiation (Target)

| Metric | OSS Baseline | HyperPod MI Target | Improvement |
|---|---|---|---|
| Effective throughput (tok/s/$) | 1x | 2-3x | Prefix routing + P/D elasticity |
| Latency guarantee headroom | 40% utilization | 75% utilization | Admission control |
| RAG/agent TTFT (cached prefix) | Full recompute | ~0ms (cache hit) | Prefix routing |
| Spot interruption recovery | Full restart | <5s (KV persists) | Managed cache |
| Multi-turn context reuse | Per-session only | Cross-session | Managed cache |

## Capability 6: llm-d + Dynamo Integration (Managed Orchestration Layer)

### What

llm-d (Gateway API Inference Extension) provides request-level routing via Kubernetes-native CRDs. NVIDIA Dynamo provides disaggregated serving with NIXL-based KV transfer. Both are OSS but **neither solves the orchestration problem alone**. HyperPod MI integrates them as a managed stack where the managed KV cache is the shared state plane.

### The Integration Gap

```
Today (OSS, manual assembly):

  llm-d EPP ──→ vLLM instance 1 (own KV cache, own prefix tree)
       │──→ vLLM instance 2 (own KV cache, own prefix tree)
       │──→ vLLM instance 3 (own KV cache, own prefix tree)

  Problem: EPP's prefix scorer has NO visibility into actual cache state.
           It guesses based on request history, not ground truth.
           Dynamo's NIXL transfers are point-to-point, not cache-aware.
```

```
HyperPod MI (managed integration):

  llm-d EPP ──→ Managed KV Cache Index ──→ Route to cache-hit instance
       │                                         │
       │         Dynamo/NIXL ◄────────────────────┘
       │         (transfers KV only when miss, not speculatively)
       │
       └──→ Admission Controller (knows cache capacity + decode queues)
```

### Why manual assembly fails

| Component | What it does well | What it can't do alone |
|---|---|---|
| **llm-d** | Request routing via Gateway API, model multiplexing, InferencePool abstraction | No cache state visibility, scoring heuristics only, no P/D awareness |
| **Dynamo** | NIXL KV transfer, disaggregated prefill/decode, CUDA-aware networking | No routing intelligence, no multi-tenant isolation, no elastic scaling |
| **vLLM/SGLang** | Serving engine, prefix caching (local), continuous batching | Per-instance only, no cross-instance coordination |

Assembling these into a production system requires:
- Custom glue code between EPP scorers and engine cache state
- Manual NIXL topology configuration per deployment
- No autoscaling coordination (scaling decode breaks NIXL connections)
- No session affinity for multi-turn (llm-d routes per-request, not per-session)

### Platform lever: Managed orchestration

HyperPod MI provides the **control plane** that ties these together:

| Layer | OSS Component | Managed Enhancement |
|---|---|---|
| **Routing** | llm-d EPP | EPP scorer backed by real-time managed cache index (not heuristic) |
| **KV Transfer** | Dynamo/NIXL | Transfers routed through managed cache — write-once, read-many |
| **Session Affinity** | None | Agent sessions pinned to managed cache region, not instance |
| **Scaling** | K8s HPA | Cache-aware autoscaler: scale decode without invalidating KV state |
| **Model Multiplexing** | InferenceModel CRD | Shared prefix cache across models with same tokenizer (e.g., Qwen family) |

### Concrete integration points

**1. llm-d EPP → Managed Cache Scorer**
```
Current:  EPP scores instances by load (LoadAwareScorer) or prefix hash guess
Enhanced: EPP queries managed cache service for exact cache residency
          → score = f(cache_hit_bytes, queue_depth, SLA_headroom)
```

**2. Dynamo NIXL → Managed Cache as Transfer Medium**
```
Current:  NIXL does GPU→GPU KV transfer over NVLink/RDMA (point-to-point)
Enhanced: Prefill writes KV to managed cache (one write)
          Any decode instance reads from managed cache (fan-out)
          Eliminates N:1 transfer topology for popular prefixes
```

**3. Session State for Agents**
```
Current:  llm-d has no session concept — each request routed independently
Enhanced: Agent session ID → managed cache region
          All turns in a session route to same cache region (not instance)
          Instance can change (spot reclaim, scaling) without session loss
```

**4. Elastic Scaling Without Cache Invalidation**
```
Current:  Scale up → new instance has cold cache → recompute everything
          Scale down → KV state lost forever
Enhanced: Scale up → new instance reads hot prefixes from managed cache
          Scale down → KV state persists in managed tier → zero cold-start
```

### Why this is defensible

- **llm-d is OSS** — anyone can run EPP, but without managed cache backing the scorer, it's just load balancing with extra steps
- **Dynamo/NIXL is OSS** — anyone can do P/D disagg, but without managed cache as the shared state plane, scaling breaks transfers
- **The integration** is the moat — the managed cache service is what makes llm-d smart and Dynamo elastic. Neither project will build this themselves (out of scope for both).

### Demo scenario

Deploy Qwen3-235B-A22B with:
- 2 prefill instances (p5e, TP4) + 8 decode instances (g7e, TP2)
- llm-d routing with managed cache scorer
- 50 concurrent coding agent sessions

**Measure**: Scale decode 8→16 during burst, back to 8 after burst.
- OSS: new decode instances cold-start (5-10s first response), old sessions broken on scale-down
- HyperPod MI: new instances warm in <500ms (managed cache prefetch), scale-down preserves all sessions

## Implementation Priority

1. **Prefix-aware routing + llm-d integration** (highest ROI, unlocks the rest) — Q3
2. **SLA admission control** (clearest premium justification) — Q3
3. **Agent session suspend/resume** (fastest-growing workload) — Q3-Q4
4. **Elastic P/D disaggregation + Dynamo integration** (requires managed cache maturity) — Q4
5. **Custom kernels** (hardest, longest lead time) — Q4+

## Non-Requirements

- Multi-region (single-region first)
- Training workloads (inference-only scope)
- Model fine-tuning integration
- Custom model architectures (standard transformer variants only)

## Success Criteria

- [ ] Demo: same hardware, same model, HyperPod MI delivers >2x tok/s/$ vs OSS vLLM on a RAG workload with shared prefixes
- [ ] Demo: bursty agent workload, HyperPod MI maintains p99 TTFT <2s at 70% utilization while OSS violates at 40%
- [ ] Customer POC: at least one customer validates price/performance claim in production-like workload
- [ ] Kernel contribution: at least one custom kernel merged into serving stack with measured improvement

## References

- NIXL (NVIDIA): KV transfer library for P/D disagg
- Mooncake (Moonshot AI): KV cache-centric disaggregated architecture
- DistServe (OSDI '24): Prefill/decode disaggregation
- llm-d (Red Hat/IBM): Gateway API inference extension with prefix scoring
- ThunderAgent: Program-aware scheduling for agent workloads
- SGLang RadixAttention: Per-instance prefix tree (our cross-instance version)

---

> **Note**: This spec defines the differentiation strategy. Implementation blueprints will be created per-capability as they move from DRAFT to IN_PROGRESS.
