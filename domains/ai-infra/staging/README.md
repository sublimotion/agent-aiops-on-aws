# Staging — AI-Infra Lab Launch Plan

Operations manual for running the experiment battery on the existing `qn-sglang-eks-cluster` in **us-west-2**, leveraging **B300 spot capacity in us-west-2b** (~$26.30/hr).

## Locked context

| | |
|---|---|
| Region | us-west-2 |
| AZ | us-west-2b (only place B300 spot is available) |
| Subnet | `subnet-001db6882dbb5ac72` |
| EKS cluster | `qn-sglang-eks-cluster` (k8s 1.32) |
| VPC | `vpc-0bd6abcecded8edf6` |
| Model bucket | `s3://kimi-k2-bench-models-20260216163240701700000006/models/Kimi-K2.6/` (1.18 TB, ready) |
| B300 instance type | `p6-b300.48xlarge` (8× B300 275GB, NVSwitch, sm_103) |
| ECR | created by `terraform apply` here |

## Out-of-scope decisions (locked)

- **Cross-node EFA P2P for ModelExpress is dropped from Spec B.** Same-node multi-pod only.
- **No new EKS cluster** — append to existing one.
- **No Karpenter** — managed nodegroup with `desired_size=0` when idle.
- **No CI for image builds** — built on a small build host, pushed to ECR. Reproducible from the Dockerfiles.
- **Single B300 only.** If two ever become available simultaneously, that's a follow-up.

## Prerequisites

1. AWS credentials with permissions to provision EC2/EKS/ECR/IAM in this account.
2. `kubectl` context set to `qn-sglang-uw2` (already configured per your local).
3. Slim images already built on the build host and pushed to ECR (see `domains/ai-infra/blueprints/build-host/README.md`).
4. `python3` with `pyyaml` installed locally.
5. `envsubst` (gettext) for manifest substitution.

## Run sequence

The `run-plan.sh` script drives the experiment battery in cost order. Each phase can be killed and re-run; results land in `domains/ai-infra/blueprints/<spec>/results/`.

### Phase 1 — `prepare`

```bash
cd domains/ai-infra/staging/terraform
terraform init
terraform apply

# Then build slim images on the build host (separate blueprint).
# Push to the ECR repos created by terraform.
```

This phase costs only ECR storage (~$0.10/GB-month for slim images; <$3/month).

### Phase 2 — `small` (cheap experiments first)

```bash
cd domains/ai-infra/staging
./scripts/run-plan.sh small
```

Runs:
- **Spec 0 small fixture** — 5 cold starts of qwen3-next on existing GPU capacity.
- **Spec E (FUSE tuning)** — no GPU needed; reminder to run on build host.
- **Spec F (access profiling)** — needs eBPF on a small GPU.

Cost: ~$50-100. Validates the profiler before any expensive work.

### Phase 3 — `b300-up`

```bash
./scripts/run-plan.sh b300-up
```

Scales the `ai-infra-b300-spot` nodegroup to `desired_size=1`. Waits up to 30 min for capacity. **The clock starts here.** B300 burns ~$0.43/min.

### Phase 4 — `b300` (the main event)

```bash
./scripts/run-plan.sh b300
```

Runs in this order on the live B300:
1. **Spec 0 large fixture** — 5 cold starts of Kimi K2.6 (~30 min total).
2. **Spec B variants** — model load mechanisms × replica scenarios (~2-3 hrs).
3. **Spec C variants** — compile cache strategies (~3-4 hrs).
4. **Spec D** — stacked end-to-end (post-analysis; not auto-run).

Cost target: ~$200-400 for Specs 0+B+C. Hard timeout in run-plan: kill and tear down if elapsed > 18h.

### Phase 5 — `b300-down`

```bash
./scripts/run-plan.sh b300-down
```

Scales the nodegroup to 0. **Run this immediately after Phase 4 to stop the spot bill.**

### Phase 6 — `teardown` (optional, when done)

```bash
./scripts/run-plan.sh teardown
cd domains/ai-infra/staging/terraform && terraform destroy
```

Clears the namespace, scales the nodegroup to 0, and (with `terraform destroy`) removes the ECR repos and the nodegroup definition itself.

## Results location

Each spec's results land in:

```
domains/ai-infra/blueprints/<spec>/results/<variant>-<timestamp>.json
```

Cross-spec analysis after the run:

```bash
python3 shared/stage_compare.py \
  domains/ai-infra/blueprints/*/results/*.json \
  --group-by experiment
```

## Cost model

| Phase | Estimated cost |
|---|---|
| Phase 1 (prepare) | <$5 (ECR storage) |
| Phase 2 (small) | $50-100 |
| Phase 3 (b300-up) | <$5 (the wait) |
| Phase 4 (b300) | $200-400 |
| Phase 5 (b300-down) | $0 |
| Phase 6 (teardown) | $0 |
| **Total** | **$255-510** |

The B300 spot rate (~$26/hr) is the dominant cost. Hard rule: **always run b300-down after Phase 4**. A forgotten B300 spot costs ~$630/day.

## Resilience

- **Spot interruption**: B300 spot can be reclaimed with 2 minutes notice. The `run-plan.sh` script's per-phase resume logic means a single cell loses at most one cold-start measurement. The model weights persist on the node's NVMe across pod restarts.
- **Capacity exhaustion**: if `b300-up` cannot find capacity within 30 min, the script exits non-zero. Re-run later.
- **Stop button**: `Ctrl-C` → manual `b300-down`. Always run `b300-down` if killing mid-experiment.

## What's not in this directory

- **Spec A (image-pull-acceleration) execution** — runs on g7e/p5e *after* the B300 window closes. Cheaper, can be slotted in later.
- **Slim image builds** — `domains/ai-infra/blueprints/build-host/` and `domains/ai-infra/shared/images/`.
- **Profiler tooling** — `domains/ai-infra/shared/profiler.py` and friends.

## Files

- `terraform/main.tf` — ECR repos + B300 spot nodegroup pinned to us-west-2b. `terraform validate` passes.
- `manifests/kimi-k2.6-fixture.yaml` — Pod + Service for Kimi K2.6 on B300.
- `manifests/qwen3-next-fixture.yaml` — Pod + Service for qwen3-next on any GPU.
- `scripts/run-plan.sh` — phase-driven experiment runner.
