# Global Inference Gateway — Capacity-Aware, Prefix-Aware, SLO-Tiered Routing

## Status: DRAFT (rewritten 2026-06-01 after research brief; superseded the original push-vs-pull framing)

## Hypothesis

Capacity-aware + prefix-aware + SLO-tiered routing — composing GAIE EPP scorers with NVIDIA Dynamo's `global_router` — delivers two distinct, separately-falsifiable wins on production-shaped workload:

1. **Prefix-shareable workloads (multi-turn chat, agent harnesses with repeated tool prompts)**: ≥2× p99 TTFT improvement vs k8s Service round-robin, matching Preble / Dynamo / AIBrix vendor claims.
2. **Capacity-event resilience (synthetic 50% AZ-scale spot reclaim)**: <30 s automatic cross-region failover with no manual intervention; round-robin baseline requires manual cutover (minutes).

The win on **batch / non-shareable workloads** is expected to be modest (<30%) and is NOT the headline. The spec's success criterion is whether the prefix-shareable improvement materializes on *our* workload mix.

The push-vs-pull paradigm framing is dropped — research surfaced that modern routers all converge on event-driven worker capacity snapshots. The real paradigm shift is **"the LB consumes backend signals (KV %, prefix locality, SLO headroom) instead of treating backends as fungible."** Push vs pull is an implementation axis, not the load-bearing change.

## Falsification criteria (workload-decomposed)

- **Prefix-shareable workload**: < 1.5× p99 TTFT improvement vs round-robin → prefix-aware routing's published wins don't materialize at our scale; either our workload isn't shareable enough or the routing overhead eats the locality bonus.
- **Capacity-event recovery**: > 60 s automatic failover, OR manual intervention required → global capacity model is too lagging or coarse to be load-bearing.
- **SLO headroom predictor accuracy**: GAIE's `predictedlatency` data producer mispredicts > 25% of the time on our workload → the predictor needs retraining or our workload distribution is too far from its training set.
- **Cross-region RTT penalty**: > 200 ms median for interactive-tier requests → speed-of-light floor is incompatible with sub-200ms p99 SLO; constrain the rule to single-region within an SLO tier.
- **Operational complexity**: number of CRDs, controllers, scoring plugins, and on-call paging incidents > 3× the round-robin baseline AND prefix-shareable win < 2× → the trade is bad.

## Why this matters

The cold-start lab solved the *backend* problem. None of those specs affect what happens once a request arrives at the cluster. Three production failure modes the lab can't fix from the backend side:

1. **Capacity-driven cold-start chains** — iter-5b `UnfulfillableCapacity` in us-west-2a + 2d. A capacity-aware router would have routed to us-east-2 g7e and avoided the cold start entirely.
2. **Tail latency from ignored backend state** — vLLM at 92% KV utilization serves a 50-token request fine but stalls on a 4K-token prompt. Round-robin guarantees one of the N replicas stalls under load. KV-aware routing skips the saturated backend.
3. **Prefix-cache locality wasted** — multi-turn chat: turn 2 of a conversation has the entire prior context already in the replica's KV cache that served turn 1. Round-robin → re-prefill the entire context → latency bomb. Preble (arXiv 2407.00023) measures 2-3× win on prefix-heavy workloads; we've never measured it at our scale.

## Stage-budget claim

This spec doesn't measure a stage of cold start. It measures **steady-state request latency** and **fleet capacity utilization** under a workload mix.

Predicted shapes (workload-decomposed per the falsification criteria):

| Metric | Round-robin | + AIBrix least-request (LLM-aware naive) | + GAIE EPP (KV+prefix+SLO scorers) | + Dynamo global_router on top |
|---|---|---|---|---|
| p99 TTFT — prefix-shareable | 1.5-5 s | 1-3 s | **0.4-1 s (target)** | 0.4-1 s |
| p99 TTFT — non-shareable batch | 1.5-5 s | 1.2-3.5 s | 1-2.5 s | 1-2.5 s |
| Prefill savings — multi-turn | 0% | 0% | **20-50% (Preble-validated range)** | 20-50% |
| Cross-region failover MTTR | manual (minutes) | manual | manual (single-region only) | **<30 s automatic** |
| Capacity-driven cold starts / week | 5-20 | 5-20 | 5-20 | <2 |

## Matrix — workload-decomposed gradient

The matrix is now structured as a gradient from "no LLM awareness" to "full stack composed":

| Cell | Router | LLM awareness level | Notes |
|---|---|---|---|
| (a) | k8s Service round-robin | None — floor baseline | Required for absolute speedup numbers |
| (b) | **AIBrix `least-request`** | LLM-aware naive — counts in-flight LLM requests, not bytes | Replaces "Envoy least-conns"; gradient point between zero awareness and full awareness |
| (c) | **GAIE EPP v1.5.0** with `kvcacheutilization` + `prefix` + `sloheadroomtier` + `latencyslo` admitter + `slodeadline` ordering | Full LLM-aware routing | Current GAIE scorer set; replaces the v1.3.1-era LoadAwareScorer caveat |
| (d) | **GAIE EPP + Dynamo `global_router`** behind it | Full LLM-aware + 2D-grid SLA scheduler + priority override | Dynamo's "Gateway (GAIE)" mode; not an alternative to (c), it stacks on top |
| (e) | (d) + cross-region capacity-state CRD adapter | Multi-region capacity-aware | Extends GAIE's `EndpointAttribute` schema with warm-pool / snapshot-freshness fields rather than a parallel CRD |

**Workload axis** (each cell measured against all three workloads, decomposed in results):

| Workload | Pattern | Why it matters |
|---|---|---|
| (i) **prefix-shareable** | Multi-turn chat, agent harness with repeated tool/system prompts, 70% prefix overlap | This is where vendors claim 2-3×; primary success criterion |
| (ii) **non-shareable batch** | Independent prompts, long contexts, throughput-bound | This is where prefix routing is theatrical; expect modest wins only |
| (iii) **mixed** (60/30/10 prefix-shareable / batch / async) | Realistic production-like mix | Tail-of-tail (p99.9) is the operational metric here |

**Topology axis**:

- **Phase 1 — single-region**: cells (a) (b) (c) (d) on workloads (i) (ii) (iii). 12 measurements.
- **Phase 2 — multi-region**: cells (d) (e) on workload (iii) + synthetic capacity event. 4 measurements (with-event / without-event for each).
- **Phase 3 — SLO tiers**: best from Phase 2 evaluated against tier-mixed workload (interactive/batch/async with separate backend pools). 2 measurements.
- **Phase 4 (stretch goal)** — request migration: Llumnix-style in-flight migration as a control. **Add OR explicitly defer** — see §"On Llumnix and request migration" below. 2 measurements if run.

**Total ~18-22 cells over 5-7 days wall-clock.**

## On Llumnix and request migration

The research brief flagged Llumnix (OSDI '24, arXiv 2406.03243) as the spec's biggest blind spot: an order-of-magnitude tail-latency win came from **in-flight request migration**, not arrival-time routing. Our spec measures only arrival-time scoring and will hit a tail-latency ceiling that migration could break through.

Two honest options:

1. **Add Phase 4** with one Llumnix-on-vLLM cell as a stretch goal. Cost: ~$15 if the upstream PR is mergeable, much more if not. Result either contributes a real number to the migration-vs-routing question (genuinely open in the literature) or documents why migration is operationally too painful to ship at our fleet shape.
2. **Explicitly defer** with a date. Rule produced: "this spec measures arrival-time routing; in-flight migration is a separate spec at `domains/ai-infra/specs/request-migration.md` to be drafted by 2026-Q3 if Phase 1 results are inconclusive on tail latency."

Recommendation: option 1 if Llumnix is mergeable against current vLLM; option 2 otherwise. Stage 0 prerequisite #5 verifies this.

## Stage 0: prerequisites

Five blockers before Phase 1 runs:

1. **GAIE EPP v1.5.0 smoke** — current scorer set (`kvcacheutilization`, `prefix`, `sloheadroomtier`, `latencyslo` admitter, `slodeadline` ordering, `approximateprefix`, `predictedlatency`). The MEMORY.md note about "LoadAwareScorer not registered in v1.3.1" is two release cycles stale (v1.5.0 is current). Stage 0 confirms scorers fire end-to-end with no missing-plugin errors.

2. **AIBrix v0.6.0 deploy** — the gradient cell (b) needs `aibrix least-request` running standalone (not as a backend behind GAIE). Vendor: `vllm-project/aibrix` v0.6.0; algorithms in `pkg/plugins/gateway/algorithms/`.

3. **NVIDIA Dynamo `global_router`** — vendored under `domains/ai-infra/blueprints/dynamo-snapshot/upstream-snapshot/` we have the snapshot agent but NOT the global router. Need to either deploy from upstream `ai-dynamo/dynamo` separately or substitute another SLA-aware router (Preble's open implementation if available).

4. **Multi-region cluster pair** — currently `qn-sglang-eks-cluster` (us-west-2). Need a sibling cluster in us-east-2 OR us-east-1 for Phase 2. Phase 1 single-region runs without this.

5. **Llumnix mergeability check** — verify if the Llumnix migration patches apply against vLLM 0.10.2+. If yes, Phase 4 is in scope. If no, defer per §"On Llumnix" option 2.

6. **Synthetic load generator** — needs configurable prefix-reuse rate (~70% for workload (i), ~0% for (ii), ~30% for (iii)). `vllm bench serve` doesn't model multi-turn well. Either extend `standards/benchmark-commons/` or vendor an LLM-aware generator (e.g., LMBench).

If any of (1) (2) (3) (4) (5) (6) blocks: degrade gracefully — drop the corresponding cell rather than blocking the whole experiment. The matrix is structured as a gradient so partial completion still yields signal.

## Baseline and methodology guardrails

Each variant deploys the **same backend pool** (Ministral-3B at TP=1, replicas=4) and serves the same workload generator. Round-robin via Kubernetes Service is the reference baseline (cell a). All variants measured side-by-side on identical hardware in identical conditions.

**Hard methodology requirements** (research brief §5.5):

- **Chunked prefill ON for all cells** (Sarathi-Serve / vLLM `--enable-chunked-prefill`). This is upstream and shrinks the cost variance routers exploit. Comparing routers without chunked prefill measures an unfair baseline. Lock it ON; document the version.
- **vLLM version frozen** for the duration of each phase. Prefix-cache implementation has shifted in 0.10.x→0.11.x; mid-experiment upgrades invalidate cells.
- **Replicas=4 minimum** per cell — fewer than 4 hides routing decisions in the noise of "almost everything goes to the same place."
- **Workload generator pinned** — don't switch generators mid-phase.

For Phase 2 multi-region, baseline is "no cross-region routing — clients are pinned to one region; failover is manual."

For Phase 3 SLO tiers, baseline is "all requests share one pool — no tier separation."

## Cross-region RTT budget

| AZ pair | Median RTT | Interactive-tier (≤200ms p99 TTFT) | Batch-tier (≤5s p99 TTFT) | Async (best-effort) |
|---|---|---|---|---|
| us-west-2 ↔ us-east-2 | ~65 ms | **structurally incompatible** | OK | OK |
| us-west-2a ↔ us-west-2d (intra-region) | ~1-2 ms | OK | OK | OK |
| us-west-2 ↔ us-east-1 | ~75 ms | structurally incompatible | OK | OK |

The falsification line "> 200 ms median RTT penalty" is measured against actual RTT, not an a priori threshold. For interactive-tier, cross-region is ruled out by the speed of light; the rule must constrain interactive routing to within-region.

## Measurement

- **Primary metrics** (decomposed by workload):
  - p50 / p95 / p99 / p99.9 TTFT per workload (i, ii, iii) per SLO tier
  - Throughput tokens/sec per backend at steady state
  - Tail-of-tail: p99.9 TTFT under sustained load (the SLO violation rate is the actual operational metric)
- **Secondary metrics**:
  - Prefix-cache hit rate per replica (KV reuse %) — must be high for workload (i), zero for (ii)
  - GAIE `predictedlatency` accuracy (predicted vs measured TTFT, error histogram)
  - Backend KV-cache utilization distribution (heatmap over time)
  - Cross-region traffic shift events (when did the router move traffic, how fast did it react)
  - Capacity-driven cold-start count (events per hour)
  - Router latency overhead (added gRPC ext-proc time per request)
- **Sample size**: ≥10 K requests per cell at ≥10 QPS sustained; capacity events triggered synthetically per Phase-2 cell.
- **Output**: enriched JSON per `standards/benchmark-commons/PROPOSAL.md`, with a `routing_decisions` block (per-request: which backend chosen, scoring breakdown, was prefix hit, KV state at routing time, predicted vs actual latency).
- **Tool**: extend `standards/benchmark-commons/container/` with a multi-turn workload generator with configurable prefix-reuse rate. Dashboards via `.claude/skills/benchmark-runner/templates/prometheus-bench.yaml`.

## Fixtures

- `domains/gpu-serving/blueprints/ministral-3b/` — primary substrate (replicas=4 across cells)
- `domains/gpu-serving/blueprints/glm5-llmd/` — EPP cell substrate (already wired; needs v1.3.1 → v1.5.0 upgrade per Stage 0 #1)
- New: `domains/ai-infra/blueprints/global-inference-gateway/` — for cells (d) and (e):
  - GAIE v1.5.0 manifests with full scorer set
  - Dynamo `global_router` deployment behind GAIE (Dynamo Gateway-mode integration)
  - GAIE `EndpointAttribute` schema extension for warm-pool / snapshot-freshness fields (extending upstream, not parallel CRD — research brief §5.10)
  - Multi-region cluster bootstrap and cross-region BGP/Cloud Map setup
  - SLO tier endpoint definitions per HTTPRoute / InferencePool
  - Synthetic capacity-event injector

## Rule the experiment would produce

If prefix-shareable wins ≥2× and capacity recovery <30 s:

> **Default inference gateway pattern** for any production gpu-serving deployment with > 1 replica or multi-region SLO concerns: deploy GAIE v1.5.0 with full scorer set + Dynamo `global_router` behind it for SLA-aware grid scheduling. Backend signals (KV %, prefix locality, SLO headroom, warm-pool membership, snapshot freshness) flow as GAIE `EndpointAttribute` extensions. Round-robin Kubernetes Service is acceptable only for single-replica deployments. Cross-region routing applies to batch / async tiers only — interactive tier (≤200ms p99 TTFT) is structurally constrained to within-region.

> **Composition with existing cold-start work**: snapshot freshness + warm-pool MX cache hit + compile-cache PVC presence are **backend-advertised state** the router consumes via the GAIE `EndpointAttribute` schema extension. The backend says "I'm a warm-pool node with model X cache hot" or "I'm a snapshot-restored replica at 30% KV utilization." The router scores accordingly. Cold-start optimization is upstream of (not parallel to) routing — backends signal what they have, router routes accordingly.

If prefix-shareable wins < 1.5×:

> Stay on round-robin + manual cross-region failover. Document the failure mode (was prefix-cache hit rate too low at our scale? was ext-proc overhead the bottleneck? was vLLM's chunked-prefill already absorbing most of the variance routers exploit?) so a future attempt has a sharper hypothesis. Re-evaluate when our fleet shape changes (multi-tenant model-per-tenant) or when the GAIE schema-extension upstream PR lands.

## Out of scope

- **In-flight request migration** — Llumnix-class technique. Phase 4 stretch goal addresses this directly; if Phase 4 is deferred, file a separate spec at `domains/ai-infra/specs/request-migration.md` with a 2026-Q3 deadline if Phase 1 results inconclusive on tail latency.
- **Cost-aware routing** ($/token spot vs on-demand) — covered by `domains/autoresearch/specs/cost-aware-routing.md`. This spec composes with it; the autoresearch spec is the cost dimension, this spec is the latency/capacity dimension.
- **Multi-tenant fairness / quotas** — orthogonal control plane concern.
- **Disagg P/D in-flight transfer** — NIXL territory.
- **Caching outside KV cache** (response cache, RAG hit cache) — separate optimization.
- **AWS-native LLM-aware load balancing** — research brief confirmed there is no LLM-aware ALB feature; `aws-application-networking-k8s` v2.1.0 (May 2026) does not implement the Inference Extension. Down-graded from "consider" to "not viable."

## Persistent caches via EBS snapshot

Not directly applicable. Router state (capacity model, recent decisions, scoring weights, online-learned predictor weights if used) is small and ephemeral; lives in etcd or controller cache.

The `EndpointAttribute` schema extension that backends update IS persisted via etcd through the API server but doesn't need EBS-snapshot semantics.

If we land an online-learned predictor (research brief §6.1 — open question), its weight checkpoints might warrant persistent storage; defer that decision until the predictor is in scope.

## Cost estimate

- **Stage 0 prerequisites**: GAIE v1.5.0 smoke (~$5), AIBrix deploy (~$5), Dynamo `global_router` deploy (~$5), multi-region cluster pair (us-east-2 EKS control plane prorated ~$70/mo, but only need 5 days for Phase 2 = ~$12), Llumnix mergeability check (~$2), workload generator extension (~$0). **~$29**.
- **Phase 1 single-region** (12 cells × 1 hr each on g7e.24xl spot): ~$45.
- **Phase 2 multi-region** (4 cells × 1 hr each, 2 regions × g7e.24xl spot): ~$30.
- **Phase 3 SLO-tiered** (2 cells × 2 hr each): ~$15.
- **Phase 4 (Llumnix migration, stretch goal)** — only if Stage 0 #5 confirms mergeability: ~$15.
- **Synthetic capacity-event simulation**: kill-and-wait pattern, no extra cost.
- **Buffer + retries**: ~$30.
- **Total cap: ~$165** (with Phase 4) / **~$150** (without Phase 4).

This is cheap because the gateway runs on small instances; the bulk of cost is the backend GPU pool that already exists.

## References

### Primary sources (versions current as of research brief 2026-06-01)
- **Research brief**: `domains/ai-infra/specs/global-inference-gateway-research.md` — citation-grounded survey of AIBrix, GAIE, Dynamo Router, sgl-router, AWS-native, plus 8 MLSys papers
- GAIE v1.5.0: https://github.com/kubernetes-sigs/gateway-api-inference-extension (current scorer set + algorithm spec at `docs/proposals/0602-prefix-cache-aware-routing-proposal/README.md`)
- AIBrix v0.6.0: https://github.com/vllm-project/aibrix (`pkg/plugins/gateway/algorithms/`)
- NVIDIA Dynamo: https://github.com/ai-dynamo/dynamo (`global_router` + Gateway-mode docs)
- Envoy ext-proc: https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter

### MLSys papers (selected by research brief; full annotations in research brief §3)
- **Llumnix** (OSDI '24, Sun et al.) arXiv 2406.03243 — in-flight request migration; the migration-vs-routing question is the spec's biggest open thread
- **Mooncake** (Qin et al.) arXiv 2407.00079 — KV-cache-centric disagg P/D; what Kimi K2.6 actually runs
- **DistServe** (OSDI '24, Zhong et al.) arXiv 2401.09670 — P/D disagg scheduling
- **Splitwise** (ISCA '24, Patel et al.) arXiv 2311.18677 — heterogeneous-fleet gestures, no controlled measurements
- **Sarathi-Serve** (OSDI '24, Agrawal et al.) — chunked prefill; methodology requirement #1 in this spec
- **Preble** (Srivatsa et al., 2024) arXiv 2407.00023 — prefix-aware routing measured 2-3× win on prefix-heavy workloads; primary published reference for our prefix-shareable cell

### Existing prior art in this workbench
- `domains/gpu-serving/blueprints/glm5-llmd/` — EPP wiring, manifests already authored (needs v1.3.1 → v1.5.0)
- `domains/gpu-serving/blueprints/nemotron-super/scripts/pd_router.py` — smaller-scope precursor
- MEMORY.md § "llm-d / Gateway API Inference Extension" — wiring lessons; **note: scorer registration caveat is now stale**, see Stage 0 #1
- Cold-start composition:
  - `domains/ai-infra/specs/dynamo-snapshot-eks-multinode.md` (snapshot freshness as router input)
  - `domains/ai-infra/specs/regional-mx-warm-pool.md` (warm-pool membership as router input)
  - `domains/ai-infra/specs/cold-start-autoscaler-policy.md` (per-pool tier selection composes cleanly; that's the per-pool decision, this is the per-request decision)
- Cost-aware adjacency: `domains/autoresearch/specs/cost-aware-routing.md`
- Steering: `.claude/steering/tech-stack.md` § "Cold-start: prefer three-tier separation" — this spec is a layer above; routing consumes backend signals
