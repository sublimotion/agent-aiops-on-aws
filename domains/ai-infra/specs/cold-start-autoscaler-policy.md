# Cold-Start Autoscaler Policy — SLO-Driven Per-Pool Selection

## Status: DRAFT

## Hypothesis

A pool-level autoscaler policy that selects between **three-tier** (slim image + compile cache + weights store) and **fused snapshot** (Dynamo + CRIU) per replica-pool — keyed on the pool's stack-churn rate and SLO — produces lower aggregate cold-start cost across a multi-tenant fleet than uniformly applying either technique.

Threshold for "lower aggregate cost": ≥20% reduction in p95 first-token latency across the fleet vs uniform three-tier baseline, AND ≤30% increase in operational artifact count (the fused-everywhere worst case is 60 snapshots for a 10-model × 3-vLLM × 2-hw fleet vs 16 artifacts for three-tier; the policy should land closer to three-tier on artifact count while still capturing snapshot's latency benefit on hot pools).

## Falsification criteria

- The policy delivers < 10% p95 latency improvement vs uniform three-tier on a representative fleet → snapshot adoption isn't load-bearing enough to justify the operational complexity; mark Spec E as research-validated but production-deferred.
- Snapshot-eligible pools end up < 20% of the fleet by replica-count → the eligibility criteria are too narrow; either relax or recognize that snapshot is a niche optimization, not a fleet-wide pattern.
- The policy's stack-churn detector emits false-positive snapshot invalidations more than 1×/week per pool → snapshot pools are constantly rebaking and the latency benefit is eaten by produce-path cost.
- Operational complexity (k8s CRD count, controller LOC, on-call pages per month) exceeds 2× the three-tier-only baseline → the policy is not sustainable without dedicated platform team.

## Why this matters

The lab phase is ending. Specs A, B, C-EBS, E have each been validated in isolation, and `.claude/steering/tech-stack.md` now encodes the production rule for **when to pick each**. But the rule is currently human-applied: a platform engineer reads the steering file and decides per-blueprint. That doesn't scale to a fleet with weekly vLLM bumps, multi-tenant model swaps, and elastic autoscaling.

This spec operationalizes the steering rule as a **controller** that applies it automatically per replica-pool. Positive result moves the lab from "we know what to do" to "the platform does it." Negative result tells us snapshot's complexity isn't worth the latency win at fleet scale and we can simplify the recommendation.

Adjacent benefit: the same controller produces the cost telemetry needed to spot stack-churn patterns (e.g., "pool X invalidates 3×/week because the team rebuilds the image on every commit") — actionable signal even before the policy itself ships.

## Stage-budget claim

This spec doesn't measure a stage of cold start; it measures **fleet-aggregate** cold-start cost. The substrate is whatever pools the controller manages.

Predicted fleet behavior:
- Eligibility split: ~30-50% of pool-replica-events qualify for snapshot (frozen stack + vLLM + autoscale-sensitive), ~50-70% stay three-tier.
- p95 first-token across fleet: ~40 s (snapshot pools) blended with ~80-160 s (three-tier pools) → fleet p95 ~70-100 s, vs uniform three-tier ~80-160 s, vs uniform snapshot ~40 s + 60-snapshot operational tax.
- Artifact count: ~20-30 (16 three-tier base + 5-15 snapshots for the eligible pools), vs 16 (three-tier-only) or 60 (fused-everywhere).

## Matrix

| Axis | Values |
|------|--------|
| Pool archetype | (a) hot-stable: vLLM frozen ≥2 weeks, weights frozen ≥1 month, autoscale events 10+/day. (b) churning-dev: vLLM rebuilt 3+×/week, autoscale rare. (c) multi-tenant: 5+ models per pool, weights swap multiple times per day. (d) sglang-only: GLM-5/Kimi K2.6 on SGLang, snapshot ineligible by engine. |
| Engine | vLLM, SGLang |
| Stack-churn rate | low (<1 invalidation/week), medium (1-5/week), high (>5/week) |
| Autoscale frequency | low (<1 event/day), medium (1-10/day), high (>10/day) |
| Variants | uniform three-tier, uniform snapshot, **policy-driven** (this spec) |

**Cells run**: simulate 4 pool archetypes under 3 stack-churn rates × 3 autoscale frequencies → 36 simulated pools, run for 7 days each. Real measurement on **2 production-shaped pools** (one snapshot-eligible, one three-tier-only) for ground truth.

## Stage 0: prerequisites

Before building the controller, three things must already exist:

1. **Spec E E1 result** — at least one passing E1 cell (Ministral-3B 4-replica TP=1 EKS) with measured p50/p95 first-token latency under concurrent restore. The policy assumes a real number, not the predicted 25-40 s. If E1 is still pending, this spec **defers to its result** before scoping the controller's latency model.
2. **Spec C-EBS production blueprint** — there isn't a turnkey `domains/gpu-serving/blueprints/<name>-with-compile-cache/` today. The compile-cache pattern is documented but not encoded as a deployable. The policy needs both options (three-tier AND snapshot) as commodity blueprints to switch between.
3. **Stack-churn telemetry** — the controller's main input is "how often does this pool's stack invalidate?" Today we don't measure this. Need a CRD or label convention that records each invalidation event (image bump, vLLM bump, weights swap, config change) with timestamps, so the controller has data to decide eligibility.

Stage 0 is **gate** — falsification of any of the three blocks the spec entirely. Especially gate #1: if E1 falsifies, snapshot is research-only and there's nothing to operationalize.

## Baseline

The "off" position for this experiment is **uniform three-tier** applied across all pools — the current production rule's default branch. Cost-per-replica-cold-start is measured at the pool level, aggregated to fleet level, and compared against the policy-driven variant.

## Measurement

- **Primary metric**: fleet p95 first-token latency over a 7-day window, weighted by pool replica-event count. Lower is better.
- **Secondary metrics**:
  - Fleet artifact count (snapshots + EBS volumes + compile-cache PVCs)
  - Per-pool produce-path frequency (how often each pool rebakes)
  - Cost-per-replica-restored ($/event) per pool tier
  - Eligibility-detector false-positive rate (snapshots invalidated mid-day vs at scheduled refresh)
  - Controller CPU/memory + reconcile-loop latency (operational complexity proxy)
- **Sample size**: 7 days per cell, 3 cells (uniform-three-tier, uniform-snapshot, policy-driven). Replica events vary by pool but baseline target is ≥1000 cold starts per cell to get tight CIs on p95.
- **Output**: enriched JSON per `standards/benchmark-commons/PROPOSAL.md`, with a `pool_breakdown` block that captures per-pool decisions and outcomes.
- **Tool**: extend `domains/ai-infra/shared/cold_start_harness.py` with a `policy_decision` field; add a new `pool_simulator.py` for the 36-cell simulation.

## Fixtures

- `domains/ai-infra/blueprints/dynamo-snapshot-eks/` — substrate for snapshot-eligible pools (E1 first)
- `domains/gpu-serving/blueprints/ministral-3b/` — three-tier substrate for hot-stable archetype
- `domains/gpu-serving/blueprints/glm5/` (or `glm5-lmcache/`) — three-tier substrate for sglang-only archetype
- New: `domains/ai-infra/blueprints/cold-start-autoscaler/` — the controller itself:
  - CRD: `ColdStartPolicy` (per-pool spec: SLO, stack-churn budget, engine, eligibility-override)
  - Controller: watches Deployments + the new CRD, decides per-pool tier, manages compile-cache PVCs and snapshot-agent DaemonSet membership accordingly
  - Stack-churn detector: subscribes to image push events (ECR), Helm release events, ConfigMap changes; emits invalidation timestamps
  - Latency-model module: predicts cold-start cost for each (pool, tier) pair using measured Spec A/B/C-EBS/E numbers; updates the prediction from observed events

## Rule the experiment would produce

If the policy hits its 20% p95 reduction with ≤30% artifact-count increase:

> Default cold-start posture for any new gpu-serving blueprint: deploy with the `ColdStartPolicy` controller managing tier selection. Pool-authors set the SLO and the stack-churn budget; the controller picks three-tier vs snapshot per pool, manages the compile-cache PVC lifecycle, and drives the snapshot-agent DaemonSet membership. Override only when the pool has a specific reason (e.g., the engine is SGLang — controller will pick three-tier automatically; or the model is research-only — pool-author can pin three-tier to skip snapshot's produce-path cost). Snapshot is no longer a separate decision a platform engineer makes per-blueprint — it's an outcome of the policy.

If the policy falsifies (<10% p95 win or >2× operational complexity):

> Snapshot adoption is research-validated but production-deferred. Stay on three-tier as the fleet default. Revisit when (a) snapshot's produce-path is faster (upstream improvements), or (b) the fleet is large enough that 25-40 s vs 80-120 s on a subset of pools justifies a dedicated platform team. Document the fleet-shape threshold under which adoption pencils out — the controller LOC is not the issue, the operational tax is.

## Out of scope

- **Multi-node TP > 1 cold start**: still upstream-blocked. Policy can flag eligibility but not deliver snapshot value.
- **Cross-cluster fleet**: this is a single-EKS-cluster controller. Federated multi-cluster coordination is a separate spec.
- **Cost-aware scheduling**: this spec is about cold-start latency. Spot vs on-demand selection per pool is `cost-aware-routing` territory.
- **GPU-fault-aware autoscaling**: see `domains/gpu-serving/specs/ray-serve-ft.md` and adjacent. The cold-start policy is orthogonal — it improves cold start whether or not the trigger was a fault.

## Persistent caches via EBS snapshot

The controller itself relies on EBS snapshots (for both the compile cache and the process snapshot tiers). No new EBS-snapshot pattern is introduced; this spec composes the existing patterns.

The new artifact is the controller's **decision history** — a CRD `ColdStartDecision` that records each pool-event's chosen tier and outcome. This is small (KB-scale per event) and fits a ConfigMap or DynamoDB; not an EBS-snapshot use case.

## Cost estimate

- Stage 0 dependencies (E1 + compile-cache blueprint + telemetry CRD): assumed sunk cost from prior specs; this spec doesn't pay for E1.
- Controller development on a small EC2: m6i.xlarge spot ~$0.04/hr × ~40 hr = **~$2** (Go or Python controller-runtime; iterate against a kind cluster locally then EKS for integration).
- Fleet simulation (36-cell, 7-day virtual time): pure-CPU work on the same instance, **negligible**.
- Real production-shaped runs (2 pools × 7 days): one g7e.24xlarge spot pool + one p5e/SGLang pool, gated on capacity. ~$3.50/hr × 7 days × 2 pools ≈ **~$1,200** if always-on; realistically ~$200 if scaled down between events.
- **Total cap: ~$250** (controller dev + 7-day production-shaped real runs scaled to event-driven, not always-on)

This is the most expensive spec in the lab because it requires real fleet behavior, not a single deployment.

## References

- Steering rule: `.claude/steering/tech-stack.md` § "Cold-start: prefer three-tier separation over fused process snapshot as the default" (this spec operationalizes the rule)
- Spec E: `domains/ai-infra/specs/dynamo-snapshot-eks-multinode.md` (Stage 0 gate)
- Spec C-EBS: `domains/ai-infra/specs/compile-cache-ebs-snapshot.md`
- Spec A: `domains/ai-infra/specs/image-pull-acceleration.md`
- Spec B: `domains/ai-infra/specs/model-decoupling-and-load.md`
- Cold-start report: `domains/ai-infra/reports/cold-start-progress-report.html` (v5 has the stacked Pareto + three-tier-vs-fused tables this controller is encoding)
- Adjacent: `domains/autoresearch/specs/cost-aware-routing.md` (cost-aware scheduler — orthogonal but composes well)
- K8s controller patterns: kubebuilder, controller-runtime
