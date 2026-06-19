# Spec A — Image Pull Acceleration

## Status: DRAFT (variant priority depends on Spec F outcome)

## Hypothesis

For container images ≥10 GB on EKS GPU nodes:

- **EBS prebake** cuts image-pull stage time by ≥5× vs default ECR pull, regardless of access pattern.
- **SOCI lazy snapshotter with default tuning** cuts image-pull stage time by ≥2× vs default ECR pull, but only when the cold-start access ratio is ≤20% (validated by Spec F).
- **ECR pull-through cache** provides ≤30% improvement when ECR is already in-region; not worth the operational cost.

## Falsification criteria

- EBS prebake improvement < 3× → CI complexity not justified, fall back to default + Run:ai Streamer alone.
- SOCI improvement < 1.5× even after FUSE tuning (Spec E) → drop SOCI from the playbook.
- Pull-through cache adds latency on warm-cache hits → drop entirely.

## Why this matters

For a 49-model fleet (per Lila's published platform), each model's vLLM container has model-specific patches and is 10-30 GB. Image pull on a fresh autoscale event blocks first-token. This stage is independent of model-load and JIT, so any win compounds with Spec B and Spec C.

## Stage-budget claim

| Stage | Baseline (sec) | Predicted with EBS prebake | Predicted with SOCI+FUSE-tuned | Why |
|---|---|---|---|---|
| Node provision | 60-120 | 60-120 | 60-120 | unchanged |
| Image pull | 180-600 | 5-15 | 30-90 | EBS = on-disk; SOCI = lazy + tuned FUSE |
| Container start | 5-10 | 5-10 | 5-30 | SOCI adds FUSE mount overhead |
| Model load | unchanged | unchanged | unchanged | Spec B |
| JIT / compile | unchanged | unchanged | unchanged | Spec C |
| First token | 1-5 | 1-5 | 1-5 | unchanged |

Replica index: **Nth replica on warm node pool** primarily; 1st-replica EBS numbers depend on snapshot-attach latency (~5-15 s on AWS).

## Matrix

| Axis | Values |
|------|--------|
| Image size class | (S) 3 GB vLLM-base, (M) 10 GB vLLM+single-model, (L) 30 GB vLLM+full-CUDA-debug |
| Pull mechanism | (1) baseline ECR direct, (2) ECR pull-through cache, (3) SOCI default config, (4) SOCI + Modal-style FUSE tuning (depends on Spec E), (5) EBS prebake (Bottlerocket), (6) OCI Image Volume (K8s 1.31+) |
| Hardware | g7e.24xlarge (AL2023), p6-b300.48xlarge (AL2023), Bottlerocket variant for (5) |
| Replica scenario | (a) cold cluster, no warm cache, (b) warm node pool with one prior pull |

3 × 6 × 2 × 2 = 72 cells. Run a representative subset (~30) — prioritize size-L, scenario-b, and the variants Spec F flags as promising.

## Baseline

Default containerd snapshotter, ECR direct pull, AL2023 NVIDIA AMI, no pre-warm. Same node type, same image, same K8s version (1.31+).

## Measurement

Reuse `shared/cold_start_harness.py`. Primary metric: pod-create → pod-Running (image-pull-bound). Secondary: pod-Running → first-token (model-load-bound, sanity check it's unchanged).

Additional measurements:
- Bytes pulled (containerd + FUSE counters).
- Layer count and largest-layer size (image structural).
- Negative-lookup latency for SOCI variants (eBPF, ties to Spec F).

Sample size: 10 runs per cell, drop highest+lowest.

## Fixtures

- Existing `vllm/vllm-openai:latest` for size-S baseline.
- Existing `domains/gpu-serving/blueprints/glm5-fp8/` container build for size-M.
- Custom build (this experiment) of vLLM with full CUDA toolkit debug symbols for size-L (intentionally large to stress-test).
- Reference Karpenter EC2NodeClass userData from AWS Labs ai-on-eks for SOCI bootstrap.
- Bottlerocket EBS-prebake CI from AWS Labs guide.

## Rule the experiment would produce

> **Default image-pull strategy by image size**:
> - **< 5 GB**: default containerd, no acceleration needed.
> - **5-15 GB**: SOCI + FUSE tuning (Spec E config) on AL2023 with Karpenter userData. Skip if access ratio (Spec F) is high.
> - **15-50 GB**: EBS prebake on Bottlerocket. CI rebuilds the snapshot on image push; Karpenter EC2NodeClass references snapshot ID.
> - **OCI Image Volume**: track for K8s 1.32+ maturity but not ready as default.
> - **ECR pull-through cache**: only for cross-region or burst scenarios where regional cache hit rate offsets configuration cost.

## Out of scope

- Model artifact loading (Spec B).
- Compile caches (Spec C).
- Image-size reduction techniques (multi-stage builds, base-image choice). Treated as a prerequisite, not a variant — every image in this matrix is already minimally built.
- gVisor-style runtime bypass — not portable to EKS.

## Cost estimate

~$200-400 GPU time + ~$50 EBS snapshot storage. Most cells run on g7e (cheap); only a handful re-validate on B300.

## References

- AWS Labs: [Container startup time on EKS](https://awslabs.github.io/ai-on-eks/docs/guidance/container-startup-time)
- ScaleOps: [GPU cold-start patterns](https://scaleops.com/blog/reducing-gpu-cold-start-times-in-kubernetes-patterns-and-solutions/)
- SOCI snapshotter: https://github.com/awslabs/soci-snapshotter
- Spec F (`cold-start-access-profiling.md`) — gates SOCI variant priority
- Spec E (`fuse-tuning-for-snapshotters.md`) — provides tuned FUSE config
