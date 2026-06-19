# Spec D — Cold-Start Stacked End-to-End

## Status: DRAFT (depends on A, B, C, E, F)

## Hypothesis

When the winning variant from each independent spec (A image-pull, B model-load, C compile-cache) is stacked, end-to-end cold start for a frontier model (GLM-5-FP8 or Kimi K2.6) drops from baseline **15-25 minutes** to **under 90 seconds for replica N≥2** on a warm node pool — a ~10-15× improvement.

For replica 1 (cold cluster, no peer for ModelExpress, no warm cache), the improvement is bounded by node-provision (60-120s) + bake-only image pull + model decoupling without P2P + cached compile artifacts: target **under 5 minutes**.

## Falsification criteria

- Replica-N stacked cold start > 3 minutes → either one of the underlying specs underperformed its hypothesis, or stacking introduces unanticipated interference.
- Replica-1 stacked cold start > 8 minutes → node provisioning or non-P2P paths are not converging.
- Any stacked variant performs *worse* than its sub-best baseline → cross-stage interaction (e.g., compile cache invalidated by image change). Diagnose and refactor.

## Why this matters

This is the only spec that produces the number Lila — and we — actually care about: **achievable cold-start floor on EKS for a frontier model, with all known optimizations stacked**. Most published cold-start work measures one stage at a time and doesn't validate that the wins compose. Stacking is where second-order interactions (e.g., image-baked compile cache fights image-pull-acceleration) surface.

## Stage-budget claim

For frontier model (GLM-5-FP8 ~733 GB, B300 hardware, replica-N≥2 on warm pool):

| Stage | Baseline (sec) | Stacked (sec) | Source spec |
|---|---|---|---|
| Node provision | 60-120 | 60-120 (warm pool) → effectively 0 for replica-N | warm pool steering rule |
| Image pull | 300-600 | 5-15 (EBS prebake) | Spec A |
| Container start | 5-10 | 5-10 | (no change) |
| Model load | 600-900 | 10-30 (ModelExpress P2P) | Spec B |
| JIT / compile | 600-900 | 60-120 (PVC compile cache) | Spec C |
| First token warmup | 1-5 | 1-5 | (no change) |
| **Total replica-N** | **1500-2500** | **80-180** | **~10-15×** |
| **Total replica-1** | **1500-2500** | **240-360** | (no peer for P2P, falls back to Run:ai Streamer + FSx) | **~5×** |

## Matrix

| Axis | Values |
|------|--------|
| Models | GLM-5-FP8 (worst-case JIT), Kimi K2.6 (worst-case weights), Qwen3-8B (sanity check, low JIT/weights) |
| Stack variant | (1) all-defaults baseline, (2) winner-of-A only, (3) (2) + winner-of-B, (4) (3) + winner-of-C — produces the additive curve, (5) full-stack with PVC instead of bake (cache update story) |
| Replica index | 1 (cold), 2 (P2P available), 4, 16 (scaling validation) |
| Hardware | p6-b300.48xlarge (frontier-class) |

~24 cells. Each cell is expensive (full B300 cold start) — cap to representative subset.

## Baseline

The "all-defaults" variant: ECR pull, model in image, no compile cache, no Run:ai Streamer, no ModelExpress, no warm pool. This is the worst-case but also the most common starting state.

## Measurement

End-to-end timing only. Use `shared/cold_start_harness.py` with the `--out` artifact tagged with the stack variant and replica index. Stage decomposition comes from the underlying specs' instrumentation; this spec just validates the sum.

Two extra requirements:
1. **Stage-time additivity check**: verify that stacked total ≈ sum of independent stage wins. If not, identify the interaction.
2. **Replica-N scaling**: at N=16 for ModelExpress, confirm scaling is sublinear (P2P broadcast tree, not naive serial).

Sample size: 5 runs per cell (B300 is expensive).

## Fixtures

- All upstream blueprints from Specs A, B, C, E, F.
- The ModelExpress server, Run:ai Streamer container, EBS prebake snapshot, FSx PVC with compile caches — all running in `ai-infra` namespace before this spec starts.

## Rule the experiment would produce

> **Production cold-start floor for frontier-model serving on EKS** (publishable):
> - Replica-N≥2: under 90 s achievable; under 60 s aspirational.
> - Replica-1: under 5 min achievable; bounded by node provision.
> - Required components: (a) warm node pool with EBS-prebaked images, (b) decoupled weights via S3 + Run:ai Streamer + ModelExpress P2P on RDMA fabric, (c) PVC-mounted per-SKU compile caches.
> - **Failure modes** (from interaction analysis): document specific cases where stacking regressed below independent best, e.g. "image-baked compile cache + EBS prebake → snapshot rebuild on every cache update; use PVC mount instead."

## Out of scope

- Independent variant comparison (handled in upstream specs).
- Production rollout (separate PRs against `gpu-serving` blueprints).
- Sub-second cold start (Modal-class infrastructure rebuild required, out of EKS-portable scope).

## Cost estimate

~$3,000-5,000. Most expensive spec in the lab. Run last, after upstream specs have validated their hypotheses.

## References

- All upstream specs (A, B, C, E, F).
- Lila's published platform claims (2,591 tok/s peak, 49 models, FSx + sticky sessions) — implicit comparison target for the cold-start dimension they don't quantify.
- Modal's sub-second cold start as the "Modal-class" reference point.
