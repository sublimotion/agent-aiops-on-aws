# Global Inference Gateway — Prior-Art Research Brief

**Status**: Research input for `global-inference-gateway.md`. Not a spec.
**Date**: 2026-06-01
**Scope**: Ground the spec's pull-based / capacity-aware / SLO-tiered hypothesis against the public state of LLM inference routing and scheduling.

> Caveats up front: (1) most of the strongest numerical claims in this space are vendor blog posts or paper abstracts, not peer-reviewed full-paper measurements. Where I could not access the full paper or the comparison was unspecified, that is called out explicitly. (2) Knowledge cutoff plus some 404s on llm-d release blogs mean a handful of numbers in this brief are quoted from one-paragraph summaries of those posts rather than the underlying tables.

---

## 1. Executive summary

1. **The "pull-based capacity-aware SLO-tiered" composition is no longer novel — it is the published direction of every major OSS router** (AIBrix, llm-d/GAIE, NVIDIA Dynamo). All three ship prefix/KV-cache-aware scoring + KV-utilization-aware load balancing + SLO-tier filters in their main branches as of 2026-Q2. Our spec hypothesis as written reads like a 2025 framing.
2. **The strongest measured public claim is "2x faster TTFT" (Dynamo + Baseten, Qwen3-Coder 480B) and "up to 3x P90 improvement on long-prefill workloads" (llm-d 0.3 predicted-latency).** Neither is a peer-reviewed comparison, neither is round-robin specifically, and the workload mixes are heavily prefix-shared agentic traces. The spec's ≥30% p99 TTFT target sits *below* these vendor claims, but the right framing is "we should beat round-robin, the question is by how much on *our* workload shape."
3. **The strongest peer-reviewed paper directly relevant to the spec is Llumnix (OSDI '24)** — its "order-of-magnitude tail latency improvement" came from in-flight request *migration*, not arrival-time routing. That's the actual paradigm shift in the literature, and the spec's "Out of scope: in-flight migration" line is the spec's biggest blind spot. Migration is what actually beats well-tuned routing under contention.
4. **Most-impactful spec changes**: (a) drop "is pull-based better than push-based" as a research question — it's a config flag in AIBrix (`slo-least-load` push vs `slo-least-load-pulling` pull) — and reframe as a measurement of which composition wins on our workload; (b) replace "custom global router cell (e)" with an integration of Dynamo's existing global_router (it already does priority-based pool override + grid lookup over TTFT/ITL targets); (c) add a Llumnix-style migration cell as Phase 4 once Phase 1-3 land.
5. **Where the research is heading next**: predicted-latency-based scheduling (online learned models replacing heuristic weights) is the active frontier — llm-d shipped a preview in 0.3 and committed deeper integration in their Mar 2026 blog. ML-in-the-data-plane is the open question, not pull vs push.

---

## 2. System-by-system

### 2.1 AIBrix (ByteDance / vllm-project)

**Repo**: `vllm-project/aibrix` (4.8K stars, last push 2026-06-01). Latest release **v0.6.0 (2026-03-05)**.

**What it actually does**:
- Envoy-Gateway-based gateway plugin (`pkg/plugins/gateway/algorithms/`) hosting **17 named routing strategies**: `random`, `least-request`, `least-busy-time`, `least-latency`, `least-kv-cache`, `least-gpu-cache`, `least-utilization`, `throughput`, `power-of-two`, `prefix-cache`, **`prefix-cache-preble`** (impl of Preble paper, arXiv 2407.00023), `vtc-basic` (Virtual Token Counter fairness), `slo`, `slo-pack-load`, **`slo-least-load`** (push), **`slo-least-load-pulling`** (pull), `pd` (P/D disagg), `session-affinity`.
- Strategies are switchable per request via `routing-strategy` HTTP header, or per "routing profile" via a config annotation. v0.6.0 added a `config-profile` header that lets the client pick a named profile.
- v0.6.0 introduced **mixed-workload PD routing** in the same deployment: long/prefill-heavy → PD pods, short/interactive → standard pods, with overflow into standard when PD saturates.
- The router maintains a **high-frequency local snapshot of pod metrics via periodic pulls + subscriptions** so scoring runs from cache, not live queries on the hot path. Explicit design comment: "scaling to thousands of QPS."

**Measured results**:
- **Paper (arXiv 2504.03648)**: only abstract retrievable as text. Quantified claim: "**50% throughput increase, 70% latency reduction**" but only attributable to the **distributed KV cache** component, not to the router. No baseline named in the abstract. ByteDance production-scale numbers, model count, GPU count are NOT in the abstract.
- **v0.6.0 release notes**: zero published benchmark numbers. Qualitative claims ("improved GPU utilization, stable latency") only.
- **KubeCon EU 2025 keynote (Google + ByteDance)**: titled "LLM-Aware Load Balancing in Kubernetes: A New Era of Efficiency" — could not retrieve text content; based on co-presenter (Clayton Coleman, the author of Kubernetes' default scheduler) the framing is *industry-consensus* rather than novel-paper.

**Mechanism**: Envoy ext-proc plugin (single Go binary `gateway-plugin`) implementing the `Router` interface; v0.6.0 also added an **Envoy-as-sidecar mode** that drops the Envoy Gateway controller dependency.

**What's NOT solved (per repo + release notes)**:
- No multi-region routing primitives. The router is in-cluster.
- No published comparison between any of the 17 strategies on a controlled workload. The lab shipping all of them with no "use this one by default" guidance is itself a signal that no single strategy dominates.
- No SLO-tier *admission control* — `slo` family routes against SLOs but doesn't shed.

**Verdict**: AIBrix is the most feature-complete in-cluster router; the breadth-over-depth approach makes it a useful baseline harness. The fact that they ship both `slo-least-load` (push) and `slo-least-load-pulling` (pull) with no published comparison means **AIBrix itself is the perfect testbed for the push-vs-pull question**: same code, two strategies, switchable by header.

### 2.2 llm-d / Gateway API Inference Extension (kubernetes-sigs/gateway-api-inference-extension)

**Repo**: `kubernetes-sigs/gateway-api-inference-extension` (678 stars, last push 2026-05-28). Latest release **v1.5.0 (2026-04-19)**.

**State of EPP scorers — directly relevant to MEMORY.md's "LoadAwareScorer not registered in v1.3.1"**:

The MEMORY.md note is **outdated**. Current `pkg/epp/framework/plugins/scheduling/` ships:

| Plugin | Path | What it does |
|---|---|---|
| `kvcacheutilization` scorer | `scorer/kvcacheutilization/` | Scores by `KVCacheUsagePercent` from `/metrics` |
| `prefix` scorer | `scorer/prefix/` | Scores by approximate prefix overlap (chunked rolling hash, see proposal 0602) |
| `prefixcacheaffinity` filter | `filter/prefixcacheaffinity/` | Filters to endpoints with high prefix overlap |
| `sloheadroomtier` filter | `filter/sloheadroomtier/` | Probabilistic two-tier filter on (TTFT, TPOT) headroom predictions; ε-exploration to recovering endpoints (default ε=0.01) |
| `latencyslo` admitter | `requestcontrol/admitter/latencyslo/` | Admits sheddable (priority<0) requests only when ≥1 endpoint passes hasValid/hasIdle/hasCold |
| `slodeadline` ordering | `flowcontrol/ordering/slodeadline/` | Earliest-deadline-first ordering inside the EPP queue |
| `approximateprefix` data producer | `requestcontrol/dataproducer/approximateprefix/` | The chunked-hash prefix index (no worker reporting needed) |
| `predictedlatency` data producer | `requestcontrol/dataproducer/predictedlatency/` | TPOT-SLO-queue maintenance for the predictor |

So the current EPP composes: **filter (SLO headroom tier + prefix affinity) → score (KV utilization + prefix) → admit (latency-SLO sheddable) → flow-control queue (SLO deadline ordering, fairness)**. This is much closer to the Sarathi-Serve / Llumnix scheduler-shape than what MEMORY.md described.

**InferencePool API**: stable at `inference.networking.k8s.io/v1` (matches MEMORY.md). v1alpha2 added `InferenceModelRewrite` and `InferenceObjective` (objective-based routing primitives). v1alpha1 keeps `InferencePoolImport` (cross-cluster). No breaking changes since 2026-Q1 in the GA API.

**Measured results (vendor blog only, no peer review)**:
- **llm-d 0.3 (Oct 2025)**: "improves P90 latency by up to **3x** in long-prefill workloads" (predicted-latency balancing preview). Baseline implied "previous llm-d default scheduler," not round-robin specifically. Also: "2.2k tokens/s per H200 GPU" — a throughput number, not a routing comparison.
- **llm-d Sept 2025 prefix-cache post**: "**10x cost differences** between cached and uncached tokens, **57x faster response time, 2x throughput** on identical hardware" — this is the cached-vs-uncached *token* delta, not the router-vs-router delta. The blog post is a primer on why prefix routing matters, not a head-to-head.
- **llm-d Mar 2026 "Predicted-Latency Based Scheduling"**: "lightweight ML model trained online from live traffic, replaces manually tuned heuristic weights." Direction signal; no numbers in the summary I retrieved.

**Mechanism**: Envoy ExtProc → EPP gRPC server → scheduler framework (filter chain, scorer chain, picker) → returns endpoint to Envoy. Same hot-path shape as AIBrix's gateway plugin. The novelty vs AIBrix is the **CRD-driven model**: `InferencePool`, `InferenceObjective`, `InferenceModelRewrite` are first-class K8s objects with controllers, not config flags.

**What's NOT solved**:
- Multi-region: `InferencePoolImport` exists but cross-region routing policy is not in the spec. Each cluster has its own EPP.
- Predicted-latency model lifecycle: how does the online model get trained, what does it require from workers, what's the cold-start behavior. The Mar 2026 blog announces the direction; production-readiness is unclear.
- Comparison with AIBrix: nobody has published a head-to-head EPP-vs-AIBrix-gateway-plugin benchmark.

### 2.3 NVIDIA Dynamo Router (ai-dynamo/dynamo)

**Repo**: `ai-dynamo/dynamo` (7.1K stars, last push 2026-06-01). Latest stable container **1.1.1**, headed to 1.5+.

**Two distinct routing components — important not to confuse them**:

1. **`lib/kv-router/`** (Rust crate `dynamo-kv-router`): in-cluster KV-aware routing. Uses `RadixTree` / `ConcurrentRadixTree` indexer over block-hashes published by workers via `RouterEvent`. Local scheduler with a configurable `RouterQueuePolicy` (FCFS for tail TTFT, **WSPT** = weighted shortest processing time for avg TTFT, LCFS).
   - Score function (per docs): combines decode cost (active blocks) + prefill cost (newly computed blocks), weighted by `--router-kv-overlap-score-weight` (default 1.0; higher = better TTFT, lower = better load balance).
   - Backpressure: `--router-queue-threshold 4.0` + queue policy.
   - **Workers report KV cache events automatically** — no worker-side config needed.

2. **`components/src/dynamo/global_router/`** (Python): hierarchical *cross-pool* routing. Two modes:
   - **disagg**: registers as both prefill and decode worker; routes prefill on (ISL, TTFT_target), decode on (context_length, ITL_target). 2D grid lookup → pool index.
   - **agg**: registers as Chat+Completions; routes on (TTFT_target, ITL_target). Same 2D grid.
   - Supports **priority-based override** via `nvext.agent_hints.priority` — explicitly built for **RL straggler mitigation** ("RL framework tags slow requests, redirects to dedicated min-latency pool"). This is exactly the agent-runtime use case from our autoresearch domain.

**Measured results**:
- README claims:
  - **2x faster TTFT** (KV-aware routing, Qwen3-Coder 480B) — citation: [Baseten benchmark blog](https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/). Baseline = Dynamo without KV-aware routing (i.e., Dynamo round-robin), workload = Qwen3-Coder 480B agentic traffic. Single-cluster.
  - **80% fewer SLA breaches at 5% lower TCO** — Planner autoscaling, Alibaba APSARA 2025 talk. This is the autoscaler, not the router.
  - **7x throughput, 750x throughput** — these are GB200/GB300 hardware comparisons, not router comparisons.
- `docs/benchmarks/kv-router-ab-testing.md` is a **harness guide** for users to run their own Router-ON vs Router-OFF benchmark on Qwen3-32B + 8 H100 workers. It does not include published numbers — explicit "DIY benchmark" framing.
- `benchmarks/router/` has Python harnesses for `prefix_ratio_benchmark.py`, `agent_benchmark.py`, `real_data_benchmark.py`, `real_data_priority_benchmark.py`. Code, not published numbers.

**NIXL integration**: The dynamo router itself does not invoke NIXL; NIXL is the KV transport between disagg prefill and decode workers. The router decides *where* a request goes; NIXL decides *how the KV moves*. They are decoupled.

**Comparison with Dynamo/GAIE gateway-mode**: README explicitly documents two deployment modes — "Standalone" (Frontend → Router → workers) and **"Gateway (GAIE)"** mode where Dynamo runs *behind* a GAIE gateway and ships its own EPP plugin (Dynamo Endpoint Picker Plugin), with the Frontend in `--router-mode direct`. **This means Dynamo and llm-d are not competing — Dynamo plugs into GAIE.** Dropping cell (d) Dynamo or cell (c) GAIE-EPP because they are alternatives is wrong; they are composable and the production target is "Dynamo backend behind GAIE gateway."

**What's NOT solved**:
- KV-aware mode requires `model_input=ModelInput.Tokens` (pre-tokenized) and dynamic discovery via etcd. Static endpoints unsupported.
- Global router supports vLLM + Mocker only; **SGLang and TensorRT-LLM are listed as not supported** for the bootstrap/async-KV-transfer path. Important: our benchmark fleet leans heavily on SGLang for GLM-5 and Kimi K2.6, so the global router cell would need to use vLLM backends.

### 2.4 sgl-router (sgl-project/sglang `experimental/sgl-router/`)

**State**: experimental, single-worker HTTP proxy, Rust, KV-aware *in name only* — multi-worker routing, service discovery, observability all listed as **pending**. The published BENCHMARKS.md is a CPU microbenchmark of policy-selection latency on an M1 MacBook (round-robin 2.5ns, random 16-471ns, power-of-two 1.75µs). No E2E throughput, no real fleet test.

**Verdict**: not a real comparator yet. **Recommend dropping any sgl-router cell from the spec.** When SGLang's PD-disagg story matures the right substrate is Dynamo+SGLang, not sgl-router.

### 2.5 AWS-native routing (ALB / NLB / VPC Lattice / App Mesh / EKS)

**Tried to find any LLM-aware routing announcement on aws.amazon.com/blogs**. Result:
- The "Application Load Balancer support for LLM-aware routing" URL hypothesized in the spec **does not exist** (404). I could not find any AWS-native announcement of an inference-aware ALB or NLB feature.
- The closest 2026 AWS-blog post on inference is a SageMaker observability post (May 29, 2026) — observability, not routing.
- **VPC Lattice / aws-application-networking-k8s v2.1.0 (May 2026)**: implements Gateway API but **no Inference Extension support** — no `InferencePool` controller, no EPP integration. No issues open in the repo titled "InferencePool" or "Gateway API Inference Extension."
- ALB has `least_outstanding_requests` (LOR) target-group algorithm, weighted target groups, slow-start mode. None of these are LLM-aware. LOR is the closest analog to least-conn but operates on TCP-level outstanding requests, not on KV utilization.

**Verdict**: **AWS native primitives buy you nothing for LLM-aware routing today.** Anything beyond round-robin / LOR has to come from Envoy/EPP/AIBrix/Dynamo running on top of the AWS data plane. The spec's reference to "ALB target-group health optimizations" should be downgraded to "AWS provides L4/L7 plumbing only; the LLM-aware layer is necessarily user-deployed."

**Implication**: the spec's "custom global router" cell can run entirely in user-managed Envoy + a controller; no AWS feature is on a roadmap that would simplify it.

---

## 3. MLSys / OSDI / NSDI literature snapshot

(Past ~18 months. Where the abstract was the only accessible text, that's flagged.)

### 3.1 Llumnix (OSDI '24, Sun et al., Alibaba) — arXiv 2406.03243

**Insight**: Reschedule *in-flight* requests across instances by live-migrating their KV cache. Treats LLM serving like OS context switching.

**Headline**: "Order-of-magnitude tail-latency improvement, 1.5x speedup for high-priority requests, 36% cost savings at similar tail latencies." Baselines named only as "state-of-the-art LLM serving systems" in the abstract.

**Why it matters for our spec**: The biggest tail-latency wins in the public literature come from **migration**, not arrival-time routing. The spec's "Out of scope: in-flight request migration" is the single most consequential exclusion. Migration solves what routing cannot — once a request is admitted to a saturated instance, no amount of clever routing helps it.

**Open source**: github.com/AlibabaPAI/llumnix (Apache).

### 3.2 Mooncake (Kimi / Moonshot, Qin et al.) — arXiv 2407.00079

**Insight**: KVCache-centric disaggregated architecture, prefill+decode separated, KV pool over CPU+DRAM+SSD across the cluster. Prediction-based early rejection for overload.

**Headline**: "Up to 525% throughput increase in simulated scenarios; Kimi handles 75% more requests in production."

**Routing/scheduling specifics not in the abstract**. The Mooncake paper is more about KV pool architecture than routing per se, but the early-rejection policy is directly relevant to the spec's admission story (mirrored in GAIE's `latencyslo` admitter).

### 3.3 DistServe (OSDI '24, Zhong et al.) — arXiv 2401.09670

**Insight**: Disaggregate prefill and decode to separate GPUs. Co-optimize parallelism per phase. Bandwidth-aware placement to minimize KV transfer cost.

**Headline**: "**7.4x more requests** at fixed SLO or **12.6x tighter SLO** at fixed load, >90% requests within latency target."

**Why it matters**: this is the *disagg* case for the spec's PD-disagg cell. Note the comparison is not against round-robin routing; it's against colocated PD serving. Different baseline.

### 3.4 Splitwise (ISCA '24, Patel et al., Microsoft) — arXiv 2311.18677

**Insight**: Same disagg PD framing as DistServe but motivated by power and heterogeneous hardware ("decode underutilizes compute, run it on cheaper GPUs"). 1.4x throughput at 20% lower cost; 2.35x throughput same cost+power.

**Difference vs DistServe**: less ML-centric scheduling depth, more datacenter/hardware-economics focus. Explicit heterogeneous-fleet design.

### 3.5 Sarathi-Serve (OSDI '24, Agrawal et al., Microsoft) — paywall on usenix.org

**Insight (from secondary sources)**: Chunked prefill + stall-free batching — the prefill phase is broken into chunks that can be co-batched with decode tokens, hiding prefill cost in decode latency. Now upstream in vLLM (`--enable-chunked-prefill`) and SGLang.

**Why it matters**: Sarathi-Serve is *backend* scheduling, not routing. But it eliminates one of the routing pressures: with chunked prefill, the prefill spikes that justify "skip the saturated backend" are smaller. This should *narrow* the gap between round-robin and KV-aware routing because the cost variance is reduced. We should benchmark with chunked prefill ON in *all* cells, otherwise we'd be measuring against an unfair baseline.

### 3.6 Preble (Srivatsa et al., 2024) — arXiv 2407.00023

**Insight**: Distributed prefix-aware scheduling co-optimizing KV reuse with load balancing.

**Headline**: "1.5x-14.5x avg latency improvement, 2x-10x p99 latency improvement vs SOTA serving systems." Baselines unnamed.

**Open source**: implementation referenced; AIBrix has a Preble port (`prefix-cache-preble` strategy), so this is a **directly testable** algorithm in our matrix without writing new code.

### 3.7 vAttention (ASPLOS '25, Microsoft) — not directly retrieved here

**Insight**: dynamic memory management for attention via OS virtual memory primitives. Memory-management not routing, but eliminates the KV-fragmentation pressure that motivates much of the routing logic.

### 3.8 FastServe / AlpaServe (older, OSDI/SOSP '23-24)

**Insight**: AlpaServe (OSDI '23) showed that statistical multiplexing + model-parallel placement can hide tail. FastServe (preemptive scheduling within a worker). Both are pre-disagg-era; less directly applicable.

### 3.9 The "predicted latency" frontier — no peer-reviewed paper yet

llm-d's online-trained latency predictor (Mar 2026 blog) is the visible bleeding edge. There is no published OSDI/NSDI paper on online-learned latency models for inference routing as of the cutoff of this brief — that's an open research opportunity (see §6).

---

## 4. Push vs pull — is the framing sound?

**Short answer**: the framing is *real* but increasingly nominal. It's a config flag, not a paradigm.

**Long answer**:

- **Llumnix is the canonical "pull" paper** in the sense that workers signal capacity and the scheduler reschedules continuously. But Llumnix's contribution is *migration*, not pull-vs-push at admission time.
- **AIBrix encodes the distinction in code**: `slo-least-load` (push: scheduler picks at arrival from a snapshot) vs `slo-least-load-pulling` (pull: scheduler peeks the SLO queue, lets workers pull when capacity frees). Same backend, two strategies, switchable by header. This is the only place I found push-vs-pull as an *empirical knob*.
- **Dynamo's KV router is structurally pull-shaped**: workers publish `RouterEvent`s with cache state, the router maintains a continuously updated index, and the `LocalScheduler` chooses based on current index state. There's a queue threshold (`--router-queue-threshold 4.0`) that gates admission. Whether you call this "pull" depends on definition; in practice the queue lives in the router and the scheduler is event-driven from worker-side updates.
- **GAIE's flow-control framework** has a **`managedqueue`** with eviction/sheddable filters and EDF ordering. This is queue-at-the-LB, not queue-at-the-backend — closer to "push with backpressure" than "true pull."

**Industry consensus**: Envoy ext-proc / GAIE is the de-facto control point for in-cluster LLM routing. AIBrix uses Envoy ext-proc. llm-d EPP is Envoy ext-proc. Dynamo's GAIE-mode uses Envoy ext-proc. The "Linux of LLM routing" framing is **directionally correct** but the actual common substrate is **gRPC ext-proc + a scoring/filtering pipeline**, not Envoy specifically — AIBrix even ships a sidecar mode that elides Envoy Gateway as a controller.

**Where the paradigm actually matters**: the push-vs-pull distinction collapses once you have a centralized capacity index that updates faster than the request arrival rate. In practice every modern router maintains such an index (AIBrix's high-frequency snapshot, Dynamo's RouterEvent stream, GAIE's metrics extractor). The remaining differentiator is **how stale the index is allowed to be** under load — which is a tuning decision, not an architecture decision.

**Implication for the spec**: framing the experiment as "push vs pull" risks measuring a config knob rather than a hypothesis. Reframe as **"capacity-aware scoring vs round-robin"** (the actual paradigm shift) and treat push/pull as a sub-axis within capacity-aware.

---

## 5. Implications for our spec

Concrete changes to recommend in `global-inference-gateway.md`:

1. **Drop "pull-based" as the headline framing**. Replace with "capacity-aware + prefix-aware + SLO-tiered." The pull-vs-push distinction is an axis under capacity-aware, not the primary paradigm shift.
2. **Update the MEMORY.md note about EPP scorers** — the current GAIE main branch ships kvcacheutilization, prefix, sloheadroomtier, latencyslo, slodeadline, approximateprefix, predictedlatency. The "LoadAwareScorer not registered in v1.3.1" caveat is two release cycles stale (v1.5.0 is current). Stage 0 prerequisite #1 should be "smoke-test EPP v1.5.0 with the current scorer set" not "fix LoadAwareScorer registration."
3. **Replace cell (e) "custom global router"** with **"Dynamo global_router with our capacity-state CRD adapter"**. The grid-lookup over (ISL, TTFT_target) × (context_length, ITL_target) with priority override is exactly what we'd build, already in production with a known config schema. We integrate, not reinvent.
4. **Cell (b) "Envoy + naive least-conns" should be replaced with `aibrix least-request`** (real LLM-aware least-request) so the gradient between "no LLM awareness" and "full LLM awareness" is measurable. Round-robin Service is the floor; AIBrix least-request is "naive LLM-aware"; AIBrix prefix-cache or GAIE EPP is "full LLM-aware."
5. **Add an explicit "chunked prefill ON" baseline note** to the Measurement section. Sarathi-Serve-style chunked prefill is upstream and shrinks the cost variance the router exploits. Comparing routers without chunked prefill measures an unfair baseline. Lock chunked prefill ON for all cells; document it.
6. **Add a Phase 4 "request migration" cell** — Llumnix's "order-of-magnitude tail latency" claim is too consequential to leave as "out of scope." Either (a) explicitly defer with a timeline, or (b) add a Llumnix-on-Alibaba-PAI cell as a stretch goal. The spec as written measures only arrival-time routing and will hit a tail-latency ceiling that migration could break through.
7. **Drop sgl-router from any candidate cell**. It's a single-worker proxy with multi-worker routing listed as future work. Dynamo+SGLang in standalone mode is the right comparator if SGLang backends are required.
8. **Falsification thresholds** — the spec's "<15% improvement → fail" is too coarse given that vendor blogs claim 2-3x. If our number is "10% improvement on mixed workload" that's still potentially meaningful given the workload-shape sensitivity of the published claims. Consider a per-workload decomposition: prefix-shareable workload should show >2x (Preble territory); non-shareable batch should show <30%; the spec's success criterion is whether the prefix-shareable improvement materializes on *our* workload mix, not whether the total p99 number drops by 30%.
9. **The "Cross-region routing" assumption needs an explicit RTT budget table**. ALB doesn't help. Envoy + Route53-latency-routing or Cloud Map can shave a few ms but the speed of light is the floor (us-west-2 ↔ us-east-2 ≈ 65ms RTT). For interactive-tier (SLO p99 ≤ 1s) the cross-region penalty is tolerable; for sub-200ms p99 tiers it's structurally incompatible. Codify the falsification line ">200ms RTT median penalty" against a real RTT measurement, not an a priori threshold.
10. **Capacity-state CRD design** — the spec proposes backends advertise (KV free%, in-flight, prefix-hit prob, warm-pool membership, snapshot freshness). GAIE has standardized `LatencyPredictionInfo` (TTFT/TPOT headroom + dispatched-request-count + KV usage %). Don't reinvent — extend GAIE's `EndpointAttribute` schema with the warm-pool / snapshot-freshness fields. This makes our work upstreamable as a GAIE plugin instead of a parallel CRD.

---

## 6. Open research questions

What is *not* in the literature that a measurement from this lab would actually contribute:

1. **Online-learned latency-prediction models in production routing** — llm-d has a preview (Mar 2026 blog), no paper yet. Open questions: what features (prompt length, system prompt hash, recent worker latency, KV state)? How does the model handle distribution shift? What's the cold-start when workers are new? **A measurement from us comparing heuristic-weight scoring vs online-learned latency-predictor on our agent-runtime workload would be publishable** — nobody has published a controlled comparison.
2. **Heterogeneous-fleet routing** — Splitwise gestures at it; the spec's composition with cold-start work (warm pool g7e + B200 spot + AZ failover) is a heterogeneous-fleet routing problem. **No published paper measures routing across heterogeneous LLM accelerators** (e.g., interactive on B200, batch on g7e PRO 6000 with 4× lower TFLOPS but cheaper). This is a gap our fleet shape uniquely lets us measure.
3. **Multi-region LLM routing under spot-reclaim events** — the spec's "synthetic AZ-scale spot reclaim" cell is genuinely novel. Public literature has nothing on multi-region LLM router behavior under correlated capacity events. The result either way would be a contribution.
4. **Verification + routing co-design** — our verifier-reward and verification-primitives experiments showed agentic workloads have heavy prefix-sharing patterns *and* RL-style straggler tail. Dynamo's `priority` override hint is exactly the lever, but no published study measures whether priority-override + KV-aware routing dominates a Llumnix-style migration approach for RL/agent harness workloads. The agent-runtime + cost-aware-routing + global-inference-gateway specs together cover this surface; a unified measurement is novel.
5. **The "is migration worth the engineering cost vs better routing" question** — Llumnix says yes, but Llumnix's baselines are unspecified. Whether arrival-time routing with state-of-the-art scoring (KV+prefix+predicted-latency) closes the gap to in-flight migration is **open**. A faithful comparison would be a contribution.
6. **EPP scaling under MoE / disagg-PD shape** — all the published numbers are dense models or aggregated-PD. GLM-5/Kimi K2.6/Qwen3-235B are MoE with disagg-PD shapes where the router's job is fundamentally harder (expert-shard pinning, prefill node selection, decode node selection, KV transit cost). Our serving fleet is the natural measurement substrate.

**Where the literature is sparse and that's a signal of opportunity vs sparse-and-already-saturated**:

- *Sparse + opportunity*: heterogeneous-fleet routing, online-learned predictors, multi-region under capacity events, MoE-disagg routing.
- *Sparse + saturated*: pull-vs-push paradigm itself (everyone has converged on event-driven snapshots; not a fertile axis).
- *Saturated*: prefix-cache-aware routing on dense single-region workloads — Preble + Dynamo + AIBrix + GAIE all measured; another data point doesn't add much unless our workload shape is genuinely different.

---

## 7. Sources / artifact pointers

- **AIBrix**: `vllm-project/aibrix` v0.6.0; `pkg/plugins/gateway/algorithms/` (17 strategies); `docs/source/designs/aibrix-router.rst`; arXiv 2504.03648 (paper, abstract only retrieved as text); KubeCon EU 2025 keynote (not retrievable); KubeCon NA 2025 keynote (YouTube, not retrieved).
- **GAIE / llm-d**: `kubernetes-sigs/gateway-api-inference-extension` v1.5.0; `pkg/epp/framework/plugins/scheduling/` (filter+scorer plugins listed in §2.2); `pkg/epp/framework/plugins/requestcontrol/admitter/latencyslo/`; `pkg/epp/framework/plugins/flowcontrol/`; `docs/proposals/0602-prefix-cache-aware-routing-proposal/README.md` (algorithm spec); llm-d.ai/blog (post-by-post summaries; some 404 on direct slug fetches).
- **Dynamo**: `ai-dynamo/dynamo` v1.1.1+; `lib/kv-router/` (Rust); `components/src/dynamo/global_router/`; `components/src/dynamo/planner/`; `docs/benchmarks/kv-router-ab-testing.md` (harness, no published numbers); `benchmarks/router/`. Baseten "2x TTFT" blog cited from README.
- **Papers**: Llumnix arXiv 2406.03243 (OSDI '24); Mooncake arXiv 2407.00079; DistServe arXiv 2401.09670 (OSDI '24); Splitwise arXiv 2311.18677 (ISCA '24); Sarathi-Serve OSDI '24 (paywalled, not directly retrieved); Preble arXiv 2407.00023.
- **AWS**: confirmed *no* native LLM-aware ALB/NLB routing; `aws/aws-application-networking-k8s` v2.1.0 has no Inference Extension support.
- **sgl-router**: `sgl-project/sglang/experimental/sgl-router/` — early/single-worker; recommend dropping from spec.
