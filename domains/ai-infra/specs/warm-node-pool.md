# Warm Node Pool Pattern

## Status: STEERING-DRAFT (no experiment, codifies operational consensus)

This is not a hypothesis-driven experiment. It is a **steering rule waiting for codification**. Operational consensus from prior art (ScaleOps survey, AWS Labs guidance, our own deployment memory) — no measurement adds value. Drafted as a spec so the rule has a paper trail; lands as a direct edit to `.claude/steering/tech-stack.md` rather than a `blueprints/` execution.

## The rule

**Two-tier GPU node pool**:

- **Primary pool**: `minNodes=1` on-demand, always-on. Holds at least one warm node per critical region. Image pre-pulled via DaemonSet. Eats node-provision and image-pull stages off the cold-start path entirely for the first replica.
- **Burst pool**: `minNodes=0` spot/preemptible, scale on demand. Pays full cold start when invoked.
- **Image pre-pull DaemonSet** on the primary pool. Pulls every blessed serving image to local disk. New pods on the primary skip the image-pull stage.

Cost model: ~$2-32/hr per primary node (per ScaleOps's GPU instance range), depending on SKU. Justified when:
- The serving deployment has a strict cold-start SLO (< 90 s replica-1) that node provisioning alone cannot meet.
- The deployment scales above 1 replica routinely; otherwise the primary node is paid for to serve as a permanent home.
- Spot interruption is non-zero in the region/SKU; primary pool is the failover floor.

## Why not an experiment

Three reasons:
1. **No falsifiable hypothesis.** "Warm pool reduces cold start" is tautological — the pool *is* the cold start avoidance.
2. **Variant space is operational, not technical.** Pool sizing (1 vs 2 vs N), instance-type mixing, scheduled scaling — these are deployment decisions per blueprint, not technique comparisons.
3. **Prior art is solid.** ScaleOps, AWS, every cloud serving framework converges on this pattern. Re-running it produces no new information.

## What lands in steering

A steering rule under `.claude/steering/tech-stack.md` § Serving Operations:

> **Warm GPU node pool**: any `gpu-serving` blueprint with a cold-start SLO < 90 s for replica-1 must specify a primary on-demand pool with `minNodes >= 1` and an image pre-pull DaemonSet. Burst capacity comes from spot. Single-replica-only deployments are exempt — the deployment itself is the warm pool.

And per-blueprint in the spec template's deployment section: "Warm pool required: yes/no, primary pool sizing, image pre-pull DaemonSet manifest path."

## What this spec does NOT cover

- **Karpenter consolidation behavior** under bursty traffic — separate operational concern.
- **Cross-AZ failover** — separate spec if/when we hit it.
- **Reserved instances vs spot economics** — domain-specific cost optimization, out of scope.
- **The specific image pre-pull mechanism** — that's Spec A (image-pull-acceleration) territory; the *rule* here is "pre-pull on primary pool"; the *mechanism* is whatever Spec A's winning variant produces.

## Adoption checklist

When landing the steering rule:

1. Add the rule under `.claude/steering/tech-stack.md` § Serving Operations.
2. Update `domains/gpu-serving/specs/_template.md` to include a "Warm pool" section in the deployment configuration.
3. PR each existing blueprint that misses it: add or document why it's exempt.
4. Reference: ScaleOps [GPU cold-start patterns](https://scaleops.com/blog/reducing-gpu-cold-start-times-in-kubernetes-patterns-and-solutions/) Pattern 4.

## References

- ScaleOps blog (Pattern 4: Warm Node Pools).
- AWS Labs ai-on-eks container startup guide.
- Our memory: existing serving blueprints (e.g., `glm5-fp8`, `kimi-k2.6-speculative`) implicitly use this pattern.
