---
blueprint: dynamo-snapshot-eks
spec: domains/ai-infra/specs/dynamo-snapshot-eks-multinode.md
status: blocked
last_stage: "0.3"
last_updated: 2026-05-31T23:17:00Z
stages:
  "0.1":
    status: complete
    note: "Retired per iter-4 user direction. Dump-path Gates 1/2/4 PASS in iter 1. Restore-path single-VM placeholder strategy was structurally wrong (5 blockers iter 2+3) — restore validation deferred to Stage 0.3 EKS DaemonSet where the upstream architecture is correct."
  "0.2":
    status: not_started
    note: "EBS+FSR pipeline. Will run inline with E1 in iter 5b Stage B."
  "0.3":
    status: blocked
    note: "Iter 5b halted on g7e spot UnfulfillableCapacity in us-west-2a + 2d (both .24xl + .48xl), 5 ASG attempts 23:05-23:13Z, no instance launched, GPU spend $0. Pre-reqs done: external-snapshotter v8.1.0 CRDs + controller installed; manifests 00/10/20/25/40 applied; nodegroup dynamo-snapshot-g7e ACTIVE at desired=0. Awaiting iter 5c capacity retry or 2c subnet add or on-demand authorization. Earlier iter-5 halt cause (three structural findings: (1) snapshot-agent image build needs ~5 GiB CUDA-devel base + 10-15 min CRIU compile — must NOT happen on a $5/hr GPU clock; (2) orchestrator Job is non-trivial controller work (build CheckpointJob spec from Deployment, watch annotations, drive EBS-snapshot side-channel, drive PVC-from-snapshot fan-out) — 8-12 hr authoring vs 1-2 hr in directive; (3) vLLM has no built-in ready-for-checkpoint sentinel hook — needs sidecar/initContainer/curl-loop wrapper not anticipated in directive. Re-plan: split iter 5 into 5a (cheap m6i build+authoring, ~$2) and 5b (g7e GPU clock E1, ~$25), gated."
  E1:
    status: not_started
  E2:
    status: not_started
  E3:
    status: not_started
  B1:
    status: not_started
  B2:
    status: not_started
---

# Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 0.1 — containerized single-GPU C/R smoke | complete (retired) | Dump-path proven iter 1. Restore-path moved to Stage 0.3 (correct architecture). |
| 0.2 — EBS+FSR pipeline | not_started | Will run inline with E1 in iter 5b. |
| 0.3 — EKS g7e nodegroup + snapshot-agent DaemonSet | blocked (pre-flight halt iter 5) | Three structural findings — build placement, orchestrator scope, ready-for-checkpoint hook gap. See lessons iter-5. |
| E1 — 4 pods × TP=1 Ministral-3B | not_started | Headline cell. Re-budgeted at ~$25 alone for iter 5b. |
| E2 — 2 pods × TP=1 × 2 nodes | not_started | |
| E3 — 1 pod × TP=4 Qwen3-next-fp8 | not_started | |
| B1 — Ministral-3B baseline | not_started | |
| B2 — Qwen3-next-fp8 baseline | not_started | |

## Spend
- Iter 1: ~$0.40 (g6.xlarge spot reclaimed)
- Iter 2: ~$0.65 (g6.xlarge spot reclaimed)
- Iter 3: ~$0.40 (g6.xlarge on-demand 30 min)
- Iter 4: $0.00 (halted pre-launch)
- Iter 5: $0.00 (halted pre-launch — no GPU, no dev VM yet)
- Iter 5a: ~$0.20 (m6i.xlarge spot ~50 min for image build + manifest authoring)
- Iter 5b: $0.00 (halted on g7e spot UnfulfillableCapacity, no instance launched)
- **Total so far: ~$1.65 of $80-100 cap**

## Halt rationale (iter 5, pre-flight)

User directed an all-in-one g7e session including the option "build images directly on the g7e if you prefer all-on-node." Reading the upstream Go source + Dockerfile against the directive surfaced three structural facts that make a single-sitting all-in-one session a cost blunder regardless of the $80 cap:

1. **Image builds belong on a non-GPU dev VM**, not on the g7e clock. The agent base is ~5 GiB and CRIU compiles for 10-15 min; failures during build (NGC auth, network, lint) on a $5/hr GPU clock contradict the iter-3 halt rule.
2. **The "snapshot orchestrator Job"** the directive describes is actually a 200+ line controller (build CheckpointJob, watch annotations, drive EBS snapshot+FSR, drive PVC-from-snapshot fan-out for 4 restores). 8-12 hr authoring, not the directive's implied 1-2 hr.
3. **vLLM has no built-in `ready-for-checkpoint` sentinel hook.** Dynamo's checkpoint contract requires the workload write `/var/run/snapshot/<container>/ready-for-checkpoint` after warmup — standard `vllm/vllm-openai:v0.10.2` does not. We need a wrapper (sidecar / initContainer / curl-loop) that's not in the directive.

Same conclusion as iter-3 and iter-4: split into authoring (cheap) and GPU clock (expensive), gated.

## Iter 5 re-plan (proposed, awaiting user ack)

**Iter 5a (this loop, no GPU, ~$2 cap):**
1. Launch m6i.xlarge in us-west-2 (any AZ, on-demand for stability — ~$0.20/hr).
2. Clone `domains/ai-infra/blueprints/dynamo-snapshot/upstream-snapshot/`, run `make docker-build-agent IMG=615299764834.dkr.ecr.us-west-2.amazonaws.com/dynamo-snapshot-agent:iter5` (CRIU compile + cuda-checkpoint pull, ~15 min).
3. `make docker-build-placeholder PLACEHOLDER_BASE_IMG=vllm/vllm-openai:v0.10.2 PLACEHOLDER_IMG=615299764834.dkr.ecr.us-west-2.amazonaws.com/dynamo-vllm-placeholder:v0.10.2-iter5` (~10 min).
4. Create ECR repos, auth, push both.
5. Author manifests at `domains/ai-infra/blueprints/dynamo-snapshot-eks/k8s/`:
   - `00-namespace.yaml` (`dynamo-snapshot`)
   - `10-rbac.yaml` (SA `snapshot-agent` + ClusterRole: pods list/watch/patch, pods/exec, configmaps/get; ClusterRoleBinding)
   - `20-seccomp-configmap.yaml` (block-iouring.json profile materialized to `/var/lib/kubelet/seccomp/profiles/block-iouring.json` via DaemonSet hostPath copy or `init` step on each node)
   - `25-agent-config-configmap.yaml` (the `/etc/snapshot/config.yaml` per `internal/types/config.go`: `storage.basePath: /checkpoints`, `accessMode: agentMount`, `restore.nsRestorePath: /usr/local/bin/nsrestore`, `restore.restoreTimeoutSeconds: 600`, `criu.tcpEstablished: true`, `criu.fileLocks: true`, `criu.linkRemap: true`, `criu.manageCgroupsMode: full`)
   - `30-snapshot-agent-daemonset.yaml` (privileged, hostPID=true; mounts `/run/containerd/containerd.sock`, `/var/lib/kubelet/pods`, `/checkpoints` from hostPath gp3 (or PV later), `/etc/snapshot/config.yaml` from configmap, `/var/lib/kubelet/seccomp/profiles` from hostPath; nodeSelector `nvidia.com/gpu.product=RTX-PRO-6000-Blackwell` + `agent-aiops/snapshot-eligible=true`; image=ECR push from step 4)
   - `40-storageclass-ebs-fsr.yaml` (gp3 + a separate VolumeSnapshotClass for EBS CSI)
   - `50-ministral-3b-deployment.yaml` (4 replicas DESIRED but start at 1; container `ministral` with vLLM `--enable-sleep-mode --tool-call-parser mistral --enable-auto-tool-choice`, weights from S3; sidecar `ready-watcher` busybox curl-loop that polls `localhost:8000/v1/models` and `touch /var/run/snapshot/ministral/ready-for-checkpoint` on first 200 OK; both containers mount `snapshot-control` emptyDir at `/var/run/snapshot` with subPath=container-name; pod has `nvidia.com/snapshot-target-containers: ministral` annotation)
   - `60-orchestrator-job.yaml` (Python orchestrator image with kubectl + boto3, drives the 4-step protocol: (a) clone Deployment template into a CheckpointJob with `nvidia.com/snapshot-is-checkpoint-source=true` label and target-containers annotation, wait for `nvidia.com/snapshot-checkpoint-status=completed`, (b) `aws ec2 create-snapshot --volume-id` of the bound EBS volume + FSR enable, (c) build a VolumeSnapshotContent referencing the AWS snapshot, (d) scale Deployment to 4 with restore-target labels and per-replica volumeClaimTemplates from the VolumeSnapshot)
6. `kubectl --kubeconfig <cluster-kubeconfig> apply --dry-run=server -f k8s/` to validate manifests against live API server (only requires read RBAC on the user, no mutation).
7. Author `scripts/run-e1.sh` end-to-end driver, `scripts/measure-baseline.sh` for B1 (cold-start to first-token, 5 runs).
8. Terminate m6i.

**Iter 5b (next loop, gated on user ack after seeing 5a artifacts, ~$25 cap):**
1. Add g7e managed nodegroup to `qn-sglang-eks-cluster`: `instanceTypes=[g7e.24xlarge, g7e.48xlarge]`, AMI type `AL2023_x86_64_NVIDIA`, capacity SPOT, subnets us-west-2a + 2d only (skip 2b $6.55/hr; 2c not in VPC), min=0/max=2/desired=1, labels `nvidia.com/gpu.product=RTX-PRO-6000-Blackwell` + `agent-aiops/snapshot-eligible=true`. Verify driver ≥580 + NCCL ≥2.26.2 (NCCL Broken on Blackwell PCIe gate).
2. Wait Ready node. Re-check spot price if stuck > 5 min.
3. Apply 00-50 manifests in order. Confirm DaemonSet rollout to the g7e node, `kubectl logs ds/snapshot-agent` shows "node controller started and caches synced".
4. Scale Deployment to 1. Wait until `ministral` Pod is Ready and the `ready-watcher` sidecar has touched the sentinel.
5. **B1 baseline (run BEFORE snapshot)**: 5 cold pod restarts of replicas=1 with `--load-format auto`. Measure pod-create → first-token. Save to `results/e1/b1-baseline.json` per `standards/benchmark-commons/PROPOSAL.md`.
6. Apply 60-orchestrator-job.yaml. Watch the Job log for the 4 phases. EBS snapshot tagged `(model=ministral-3b, vllm_version=0.10.2, tp_size=1, gpu_arch=sm_120, driver=580+, config_hash=<h>, blueprint=dynamo-snapshot-eks, iter=5)`.
7. Restore phase: orchestrator scales to 4 replicas. Capture per-replica timestamps: pod-create → volume-bound → CRIU-restore-start → CRIU-restore-end → cuda-restore-end → first-token. Capture concurrent-restore time-to-all-4-ready.
8. Run gates:
   - **Gate 1**: SHA256(first-64-token-IDs) per restored replica == freshly warmed replica with deterministic gen kwargs (temperature=0, seed=42, top_p=1.0, top_k=-1).
   - **Gate 2**: artifact size ≤ weights + 4 + 1×TP = 6 + 5 = 11 GiB (Ministral-3B ~6 GiB).
   - **Gate 3**: p50 pod-create-to-first-token ≤ 30 s for snapshot variant.
9. Halt at first failure or $25 spend. Surface E1 result.
10. Cleanup: scale Deployment to 0, scale g7e nodegroup desired=0 (preserve manifests + nodegroup config for E2/E3).

## Iter 5b run (2026-05-31, 23:00-23:17Z) — capacity halt

**Outcome: HALT on g7e spot UnfulfillableCapacity. Zero GPU spend.**

### What ran (~$0)

1. **Step 0 (cluster prereqs)**: external-snapshotter v8.1.0 CRDs + controller installed. Manifests 00 (namespace), 10 (RBAC), 20 (seccomp installer), 25 (agent config), 40 (StorageClass + VolumeSnapshotClass) applied. All persistent for next iter.
2. **Step 1 (nodegroup)**: `dynamo-snapshot-g7e` managed nodegroup created on `qn-sglang-eks-cluster` — subnets 2a-private + 2d-private, instanceTypes `[g7e.24xlarge, g7e.48xlarge]`, AL2023 NVIDIA, SPOT, role `ai-infra-b300-node`, labels `agent-aiops/snapshot-eligible=true` + `nvidia.com/gpu.product=RTX-PRO-6000-Blackwell`, taint `nvidia.com/gpu=true:NoSchedule`. ACTIVE in 65 s.

### What halted

3. **Step 2 (GPU clock start)**: Bumped desiredSize 0→1 at 23:04:18Z (spot prices re-checked: 2d $4.91, 2a $5.11, both within $7/hr cap). ASG made 5 spot-fleet attempts at 23:05:38, 23:07:34, 23:09:30, 23:11:26, 23:13:22 — **all returned `UnfulfillableCapacity`** for both .24xl and .48xl in both subnets. Re-checked spot price (one re-check per directive): pricing normal across all four AZs, but 2c is not in the cluster VPC. Per halt rule, scaled back to desired=0 at 23:16:31Z.

### Steps not reached (deferred to iter 5c)

- Step 3 (apply 30-snapshot-agent-daemonset.yaml) — needs g7e node.
- Step 4 (apply 50-ministral-3b-deployment.yaml + warmup) — needs g7e node.
- Step 5 (B1 baseline) — needs running pod.
- Step 6 (orchestrator-job snapshot produce) — needs warmed pod.
- Step 7 (E1 restore at N=4) — needs snapshot.
- Steps 8-9 (Gates 1/2/3, halt rules).

### Iter 5c options (surface to user)

1. **Time-shift retry**: re-attempt during off-peak Pacific hours; g7e spot in us-west-2 has historically had multi-hour pockets of unavailability.
2. **Add us-west-2c subnet**: cheapest spot price ($4.75 for .24xl) and a third capacity pool. Requires Terraform change to cluster VPC.
3. **On-demand fallback for E1 only**: g7e.24xl on-demand ~$15/hr × ~10 min for B1+E1+gates ≈ $2.50. Negligible vs $30 cap. Tradeoff: locks in cost regardless of capacity.

### Cumulative spend

- Iters 1-5a: ~$1.65
- Iter 5b: $0.00
- **Total: ~$1.65 of $80-100 cap**

### Ready-or-not

- E1: NOT READY (capacity gate).
- E2 (cross-node): NOT READY (E1 must pass first).
- E3 (TP=4 Qwen3-next-fp8): NOT READY (E1 must pass first).
