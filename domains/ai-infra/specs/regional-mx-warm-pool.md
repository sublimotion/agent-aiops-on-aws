# Regional MX Warm Pool — Dual-Purpose Warm Slot + ModelExpress Proxy

## Status: DRAFT

## Hypothesis

A small fleet of GPU-light, memory-heavy spot instances per region — held warm, image pre-pulled, AND running as ModelExpress (MX) proxy peers caching popular model weights — eliminates **two** stages from the cold-start pipeline simultaneously: node provision (60-90 s with Karpenter, longer with Cluster Autoscaler) and a meaningful fraction of model load (60-120 s baseline → 20-40 s peer-fetch).

Threshold: ≥40 s p50 wall-time reduction on first-replica cold start vs the existing warm-pool-only pattern, AND ≥30 s vs Karpenter+pause-pods only, on a representative serving workload (Ministral-3B class). For a Kimi-class workload (700 GB), the win compounds — model load drops from S3 PCIe-bound (~245 s on TP=8) to intra-VPC-bandwidth-bound, hypothesized 60-120 s.

## Traffic-shape applicability gate

This spec is **only relevant for spiky traffic**. It is explicitly NOT a fleet-wide default. The warm pool's economics depend on **event frequency × penalty-per-event** being large enough to amortize ~$1,900/mo per region of always-on warm slots.

Three traffic shapes where the pattern pays:

| Shape | Pattern | Warm pool sizing | MX-peer value |
|---|---|---|---|
| **Diurnal burst** | 3-5× peak/trough, scale up at 9am / down at 6pm | N=1-2 fixed | High — same models warm-fetched daily |
| **Bursty multi-tenant** | model-per-tenant, unpredictable per-tenant peaks, 10+ events/hr during peaks | N=2-4 sized to tenant count | Very high — peer-fetch hides per-tenant model swap |
| **Spot-reclaim recovery** | rare but catastrophic; 8-GPU node dies, 3-16 min baseline restore | N=1 minimum failover floor | Medium — only first reclaim per model benefits |

Two regimes where this spec does NOT apply (document as exemptions in the steering rule the experiment produces):

- **Single-replica always-on workloads** — the deployment IS the warm pool. Adding a separate warm pool is double-paying.
- **Fully scale-to-zero with infrequent traffic** (<1 event/day per pool) — Spec E snapshot's ~25-40 s restore beats a permanently-warm pool that costs ~$1,900/mo to absorb 1-2 events/week. Cost-per-cold-start-avoided is the wrong shape.

If your traffic isn't spiky in one of the three shapes above, **stay on the three-tier path** (`tech-stack.md` § "Cold-start: prefer three-tier separation"). This spec is a specialized pattern, not a default upgrade.

## Falsification criteria

- MX peer-fetch from a same-VPC warm-pool node delivers <30% reduction in model-load wall time vs S3 fetch on the burst node → locality bonus isn't real, or PCIe-bound stage dominates and intra-VPC bandwidth doesn't help.
- Spot reclaim rate of the warm-pool nodes exceeds 2×/week per region → can't maintain N=2 reliably without on-demand fallback that erodes the economics. Pattern degenerates to existing warm-pool spec.
- Steady-state cost per region exceeds 1.5× the existing always-on managed-nodegroup warm pool → savings the dual-purpose framing was supposed to deliver don't materialize.
- Operational complexity (CRD count, controller LOC, on-call paging on MX-peer reclaim) more than doubles vs vanilla warm pool → the second value stream isn't worth the failure modes it introduces.

## Why this matters

The cold-start lab has been treating "node provision" as a 0 s assumption — but in practice it's 60-90 s with Karpenter and 120-180 s with Cluster Autoscaler. Capacity-driven failures (iter 5b's `UnfulfillableCapacity`) make this worse: when the AZ has no g7e spot, the cold-start clock can stretch into minutes or hours.

Two existing specs address pieces of this:
- `warm-node-pool.md` — operational pattern, no experiment, says "keep N=1 on-demand always-on."
- `model-decoupling-and-load.md` (Spec B) — RunAI/MX as model load accelerators, currently deprioritized per `.claude/steering/tech-stack.md` because PCIe is the host→GPU bottleneck and snapshot covers replica-N.

This spec **composes** them. A warm-pool node sitting idle waiting for a burst event is wasted GPU spend. Repurposing the same node as an MX peer gives you two value streams from one piece of infrastructure: cold-replica capacity AND a regional weights cache. The MX deprioritization gets re-examined when the node is "free" — the only marginal cost is the storage for cached weights.

Adjacent benefit: an MX proxy at the regional edge becomes the **failure-mode floor** for snapshot-invalidation events. When a snapshot is invalidated (vLLM bump, weights swap), produce-path cost on the burst node is reduced because weights come from the warm peer instead of S3.

## Stage-budget claim

**Replica-1 cold start, Ministral-3B class, with Karpenter + warm pool + MX peer**:

| Stage | Baseline (vanilla Karpenter) | + warm pool only | + warm pool as MX peer | Why |
|---|---|---|---|---|
| Karpenter pod-pending → node-Ready | 60-90 s | 0-30 s (preempt pause pod) | 0-30 s | unchanged from warm pool |
| Image pull | 5-15 s | 0-5 s (DaemonSet pre-pulled) | 0-5 s | unchanged |
| Container start | 5-10 s | 5-10 s | 5-10 s | unchanged |
| Model load | 60-120 s (S3) | 60-120 s | **20-40 s (peer-fetch)** | intra-VPC bandwidth + RunAI parallel streams |
| JIT/compile | 30-90 s | 30-90 s | 30-90 s | unchanged (Spec C-EBS handles) |
| First token | 1-5 s | 1-5 s | 1-5 s | unchanged |
| **Total** | **160-330 s** | **95-260 s** | **55-180 s** | **2-3× vs baseline, 1.5-2× vs warm pool only** |

For Kimi-class (700 GB model on TP=8), model load on S3 is the dominant ~245 s PCIe-bound stage. MX peer cuts this materially because peers can stream weights to the burst node in parallel from N peers, saturating the burst node's PCIe rather than its single S3 connection — predicted 60-120 s, ~2-4× win on the model-load stage alone.

Stage applies to **replica-1 / cold-cluster**. For replica-N≥2, the snapshot path (Spec E) still wins on absolute time; this spec is about the cold-cluster floor.

## Matrix

| Axis | Values |
|------|--------|
| Hardware (warm pool) | g6.4xlarge (1×L4 24GB, 64GB RAM, 600GB NVMe, ~$1.30/hr spot), g6e.4xlarge (1×L40S 48GB, 128GB RAM, 1.9TB NVMe, ~$2/hr spot) |
| Hardware (burst, served) | g7e.24xlarge for Ministral-3B / Mistral-Small-4 class; p5e.48xlarge for Kimi K2.6 class |
| Pool size | N=1 (single-AZ), N=2 (cross-AZ in single region), N=3 (one per AZ) |
| MX backend | ModelExpress with weights cached in RAM, NVMe, or both (tiered) |
| Pool replacement strategy | reactive (on `instance-action: terminate`), eager (on `capacity-rebalance` signal), hybrid |
| Workload | Ministral-3B (small, sanity), Qwen3-next-fp8 (medium, TP=4), Kimi K2.6 (large, TP=8 — only on the few cells where p5e is available) |
| Variants | (a) vanilla Karpenter, (b) Karpenter + pause-pod warm pool, (c) Karpenter + warm pool + MX peer (this spec) |

**Cells run** (12): {g6.4xlarge, g6e.4xlarge} × {N=2 cross-AZ} × {hybrid replacement} × {3 workloads} × {variant b vs c}. Variant a runs as a single baseline per workload (3 cells). **Total 15 measurements, ~3-5 days.**

## Stage 0: prerequisites

1. **Karpenter on `qn-sglang-eks-cluster`**: cluster currently uses managed nodegroups. Either install Karpenter (additive, ~30 min) or use managed nodegroup `minSize=2` with image pre-pull DaemonSet as the warm-pool substrate. Karpenter preferred because it's the platform direction, but managed nodegroup is fine as a fallback.
2. **MX (ModelExpress) deployment**: vendor or build from upstream. ModelExpress is the assumed-but-not-yet-validated technique referenced in `model-decoupling-and-load.md`. **Block here if MX has no production-ready deployment** — fall back to RunAI Streamer with peer-aware S3 endpoint as a substitute, or measure the warm-pool-no-MX cell only.
3. **Spot capacity-rebalance handler**: AWS Node Termination Handler v1.21+ deployed cluster-wide. Without it, spot reclaim of warm-pool nodes is a hard event instead of a graceful drain.

If any of these are blocked, the spec degrades to "warm pool with image pre-pull only" (which is the existing `warm-node-pool.md`) and the MX-peer cells defer.

## Baseline

Three baselines, one per variant:
- **Vanilla Karpenter** — `minSize=0`, scale on demand, no warming. Measures the worst case.
- **Karpenter + pause-pod warm pool** — N=2 nodes held warm via low-priority pause pods, image pre-pulled. Measures the existing `warm-node-pool.md` pattern.
- **Karpenter + warm pool + MX peer** — pause pods on N=2 nodes ALSO run MX backend, weights cached on NVMe. Measures this spec.

For each, deploy the same blueprint (e.g., `domains/gpu-serving/blueprints/ministral-3b/`) and trigger replica-1 cold start by deleting the node + recreating. Measure pod-create → first-token-streamed.

## Measurement

- **Primary**: pod-create → first-token-streamed wall time, p50/p95 over 10 cold starts per cell.
- **Secondary**:
  - Per-stage breakdown (Karpenter provision time, image pull, container start, model load, JIT, first token) — uses existing `domains/ai-infra/shared/cold_start_harness.py`.
  - **MX cache hit rate** under realistic workload mix (which models are actually fetched from peer vs S3).
  - **Warm-pool reclaim incidents** during the run window — count + recovery time per event.
  - **Cost per cold start avoided** — (warm pool $/hr × hours) ÷ (count of cold-start events absorbed) for each variant, dollar-normalized.
- **Sample size**: 10 cold starts per cell (3 variants × 3 workloads + 6 hardware/pool-size cells). 7-day capture window.
- **Output**: enriched JSON per `standards/benchmark-commons/PROPOSAL.md`, with a `warm_pool_state` block (peer reachability, cache hit/miss, reclaim events).
- **Tool**: extend `cold_start_harness.py` with `mx_peer_fetch_breakdown` and `karpenter_provision_breakdown` blocks.

## Fixtures

- `domains/gpu-serving/blueprints/ministral-3b/` — Ministral cell substrate
- `domains/gpu-serving/blueprints/qwen3-next/` — Qwen3-next cell substrate (TP=4)
- `domains/gpu-serving/blueprints/kimi-k2.6-speculative/` — Kimi cell substrate (TP=8, gated on p5e capacity)
- New: `domains/ai-infra/blueprints/regional-mx-warm-pool/` — Helm chart and manifests for:
  - Warm-pool NodePool / managed nodegroup definition
  - Image pre-pull DaemonSet (image set per `.claude/steering/tech-stack.md` blessed-image list)
  - Pause-pod Deployment with PriorityClass `-1`, `karpenter.sh/do-not-disrupt: "true"`, GPU resource request to hold the slot
  - MX backend StatefulSet with hostPath NVMe mount for weights cache, S3-protocol-compatible endpoint exposed via headless Service for peer discovery
  - Cluster-wide ConfigMap with peer-discovery config that burst-pool pods consume to know which warm-pool nodes have which weights cached
  - AWS Node Termination Handler

## Rule the experiment would produce

If hypothesis holds at the predicted threshold:

> **For spiky-traffic GPU LLM workloads** (diurnal burst, bursty multi-tenant, or spot-reclaim-sensitive), default warm-pool pattern: deploy `regional-mx-warm-pool` as a 2-node spot pool (one per AZ), GPU-light hardware (g6.4xlarge or g6e.4xlarge depending on weight-cache RAM requirements), image pre-pulled via DaemonSet, MX backend running on NVMe-cached weights for the 5-10 most-fetched models in the region. Cold-start replica-1 budget for any model in the cache: 55-180 s instead of 160-330 s. Update `warm-node-pool.md` § "What lands in steering" to require MX peer mode for any blueprint with cold-start SLO < 120 s for replica-1.
>
> Also reactivates Spec B (model decoupling) on the strength of the locality bonus: MX is no longer evaluated as a standalone optimization; it's the second value stream of the warm pool you already pay for.

If falsified (MX peer doesn't deliver the predicted ≥30% reduction, OR reclaim rate exceeds budget):

> Stay on existing `warm-node-pool.md` pattern (image pre-pull only, no MX). Spec B remains deprioritized. Document where MX peer-fetch broke down (was it PCIe-bound at the burst node? was peer discovery the bottleneck? was reclaim too aggressive?) so the next attempt has a sharper hypothesis.

## Out of scope

- **Cross-region MX coordination** — this spec is per-region. Multi-region weights replication is a separate concern.
- **Snapshot integration** — orthogonal. Snapshot replicates a warm worker's full state; MX peer replicates weights only. They compose: snapshot's produce-path also benefits from MX peer-fetch.
- **Cost-aware warm-pool sizing** — N=2 is a fixed assumption. Dynamic N based on traffic forecast is a follow-on.
- **Karpenter consolidation tuning** — assumed to be on default settings; aggressive consolidation could disrupt warm-pool pause pods, but that's a Karpenter operational concern.
- **GPU-tier proxies** — using a g7e/p5e as a warm pool defeats the cost argument; explicitly excluded. The pattern is GPU-LIGHT proxies serving GPU-HEAVY burst.

## Persistent caches via EBS snapshot

Not applicable directly. The MX cache is in-RAM + on-NVMe per warm-pool node, not an EBS-snapshotted artifact. NVMe is ephemeral — weights are re-staged to NVMe on warm-pool node bootstrap (~3 min for a 6 GB model from S3). For larger weights (Kimi 700 GB), bootstrap time becomes a real consideration; falls back to S3-streaming-into-NVMe-as-needed-cache rather than full pre-population.

This spec MAY produce a steering rule about **EBS-snapshotting NVMe-cached weights** if bootstrap time on large weights becomes a bottleneck — but that's an outcome, not an input.

## Cost estimate

- **Stage 0 prerequisites**: Karpenter install ~$0 (additive); MX deploy + bake test on m6i.xlarge spot ~$2; AWS NTH install ~$0.
- **Variant baselines**: 3 baselines × 1 g7e.24xl spot × 1 hr each = ~$11.
- **MX-peer cells (the experiment)**: 2 g6e.4xlarge spot × 4 hr × 12 cells = ~$190 worst case; realistic ~$60 if cells share warm pool.
- **Burst nodes (g7e + p5e)**: 1 g7e.24xl spot × 6 hr = ~$30; 1 p5e.48xl spot × 4 hr × 1 Kimi cell = ~$67 (gated on capacity, may skip).
- **Storage for MX NVMe cache**: hostPath, $0 marginal beyond instance cost.
- **Buffer**: ~$30.
- **Total cap: ~$330** (full matrix incl. Kimi cell) / **~$250** (skip Kimi cell, defer to a separate run if p5e capacity opens).

## References

- Existing operational pattern: `domains/ai-infra/specs/warm-node-pool.md` (this spec extends it with the MX-peer mechanism)
- MX baseline: `domains/ai-infra/specs/model-decoupling-and-load.md` (Spec B; deprioritized per current steering, this spec reactivates it)
- Karpenter: https://karpenter.sh
- AWS Node Termination Handler: https://github.com/aws/aws-node-termination-handler
- Cold-start report: `domains/ai-infra/reports/cold-start-progress-report.html` (v5; this spec, if positive, adds a "node provision" row to the stacked Pareto)
- Steering: `.claude/steering/tech-stack.md` § "Cold-start: prefer three-tier separation" + "Spec B: keep on the roadmap" — this spec is one of the concrete reactivation paths
- Adjacent: `domains/ai-infra/specs/cold-start-autoscaler-policy.md` (operationalization spec; the policy controller would manage warm-pool sizing per pool)
