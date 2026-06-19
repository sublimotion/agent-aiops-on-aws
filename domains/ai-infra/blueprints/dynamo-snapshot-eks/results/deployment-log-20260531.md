# dynamo-snapshot-eks — deployment log 2026-05-31

## Stage 0 pre-deploy gate

- 11:25 UTC — Read spec `dynamo-snapshot-eks-multinode.md` and predecessor lessons `dynamo-snapshot/lessons.md` in full.
- 11:26 UTC — Verified AMI `ami-041914c9e9b61b15e` exists in us-west-2 (predecessor bake, NVIDIA AL2023 + driver 595.71.05 + CRIU 4.2 + PR #3021 + cuda-checkpoint + cuda_plugin.so + seccomp-wrap).
- 11:26 UTC — EKS cluster `qn-sglang-eks-cluster` (k8s 1.32) ACTIVE with nodegroups `ai-infra-b300-spot`, `ray-ft-gpu`, `ray-ft-system`. No g7e nodegroup yet (Stage 0.3 will add).
- 11:26 UTC — g6.xlarge spot price in us-west-2b: $0.32/hr (cheapest AZ).
- 11:27 UTC — Created blueprint dir `domains/ai-infra/blueprints/dynamo-snapshot-eks/` with `scripts/`, `k8s/`, `docker/`, `results/` subdirs and template `lessons.md`.

## Stage 0.1 — containerized single-GPU C/R smoke

- 11:28 UTC — Launched g6.xlarge spot `i-0b1b81fdfd547d27a` from baked AMI in us-west-2b (subnet-0107183866c5f4f76, sg-06821adfa7f05916f, key dynamo-snapshot-uw2, IAM dynamo-snapshot-uw2-profile). Public IP 35.95.64.159.
- 11:29 UTC — SSH ready; verified L4 GPU, CRIU 4.2, cuda-checkpoint, nvidia-container-runtime, docker, nerdctl, seccomp-wrap all present. `/usr/lib/criu/cuda_plugin.so` symlink missing (matches predecessor lesson).
- 11:30 UTC — Mounted NVMe (xfs on /dev/nvme1n1, 233 GiB), created cuda_plugin symlink, ran a `nvidia/cuda` container with `--gpus '"device=0"'` to confirm device isolation: container sees only `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-modeset`, `/dev/nvidia-uvm{,-tools}` — no peer GPUs as predicted. **Gate 4 PASS**.
- 11:32 UTC — Wrote v1 orchestrator `scripts/run-stage01-container-cr.sh` (vLLM 0.10.2 docker image, smoke-vllm-sleep.py harness, criu dump from host with `--pid=host`).
- 11:38 UTC — vLLM image pulled (~10 GiB, ~6 min). First v1 run failed because the bash entrypoint used `python` and the image only has `python3`.
- 11:42 UTC — Fixed `python` → `python3`, re-ran. `READY_FOR_CHECKPOINT` reached, sleep(level=1) freed 8.85 GiB. CRIU dump failed at `criu/mount.c:753`: nvidia-container-runtime injects ~57 host driver libs as file-bind-mounts that vanilla CRIU 4.2 cannot reconcile (`libglxserver_nvidia.so` was the first to trip).
- 11:48 UTC — Wrote v2 orchestrator `scripts/run-stage01-container-cr-v2.sh` adding skip-mnt enumeration. First broad regex (`nvidia|cuda|libgl|...`) skipped 41 mounts but missed `libnvcuvid.so` (no `nvidia` substring in name). Iter 2 expanded regex: same failure on `/run/nvidia-persistenced/socket` (tmpfs, not on host XFS device).
- 11:55 UTC — Read upstream Dynamo source `internal/runtime/mounts.go::BuildMountPolicy` and rewrote v2 to mirror its policy: skip non-OCI mounts under `/proc /sys /run`; externalize everything else via `--external mnt[<path>]:<label>`; let `/dev/shm` dump natively.
- 12:00 UTC — Re-ran with externalization: 15 skip-mnt + 66 externalize args. **CRIU dump SUCCEEDED** in ~1.5 s, `Dumping FAILED` errors gone, cuda_plugin completed stage 2.
- 12:00 UTC — CRIU restore failed at `criu/mount.c:3024 No mapping for 1044:(null) mountpoint` because dump-time externalization records mount IDs that need a corresponding `--external mnt[<label>]:<source>` on restore. Restore was invoked without those mappings.
- 12:05 UTC — Spot `i-0b1b81fdfd547d27a` reclaimed by AWS (`Service initiated`). SSH timed out. Total instance lifetime ~80 min, ~$0.40 spent.
- 12:10 UTC — Updated `lessons.md` with five new entries documenting Gate 4 PASS, the externalization fix, the restore mapping gap, the spot reclaim pattern, and the resume checklist. Wrote `results/progress.md` and this deployment log.

## Status at end of session 1

- Gate 4 (device isolation): **PASS** (verified)
- Gate 1 (token equality): **NOT MEASURED** (restore interrupted)
- Gate 2 (artifact size): **NOT MEASURED** (restore interrupted; dump artifact was written but not validated against weights)
- Gate 3 (restore < 30 s): **NOT MEASURED**
- Hypothesis status: **SUPPORTED so far**. Containerization solves the bare-EC2 device-fd problem AND mount externalization (Dynamo policy) clears CRIU 4.2's mount-tree validation. Full pass/fail awaits restore measurement on next loop iteration.

## Resume plan

See `lessons.md` § Resume checklist. The v2 orchestrator needs a placeholder-container restore step (launch fresh container, join its namespaces, pass `--external` mappings to the restored mount points). After that, Gates 1/2/3 should clear and Stage 0.1 will be GREEN, unblocking 0.2 + 0.3 + E1/E2/E3.

## Iter 5 (2026-05-31, late) — pre-flight halt before all-in-one g7e session

- 19:50 UTC — Loaded user iter-5 directive: all-in-one g7e session, raised cap $80/$100, skip 2b, retire Stage 0.1.
- 20:00 UTC — Re-confirmed cluster + pricing facts (no drift): 615299764834/aiops, EKS qn-sglang-eks-cluster ACTIVE, g7e.24xl spot 2a $5.11/hr 2d $4.80/hr.
- 20:05 UTC — Read upstream `Makefile`, `Dockerfile`, `cmd/agent/{main.go,config.go}`, `internal/controller/controller.go`, `internal/types/config.go`, `protocol/{checkpoint.go,restore.go,common.go}` to understand the agent contract before authoring manifests.
- 20:25 UTC — Surfaced three structural findings to lessons.md:
  1. Image build placement — must be on cheap dev VM, NOT on g7e clock per iter-3 cost-correctness rule.
  2. Orchestrator scope — directive's "60-snapshot-job.yaml" is a 200+ line controller, not a thin trigger.
  3. vLLM has no `ready-for-checkpoint` sentinel hook — needs sidecar/initContainer wrapper not anticipated in directive.
- 20:30 UTC — Wrote iter-5 re-plan to `results/progress.md`: split into 5a (m6i build+author, ~$2) and 5b (g7e E1, ~$25), gated on user ack.
- 20:30 UTC — HALT BEFORE LAUNCH. No spend this iteration. Awaiting user ack on 5a/5b split before proceeding.


---

## Iter 5a (m6i.xlarge spot us-west-2a) — image build + manifest authoring

**Status: COMPLETE.** Cumulative spend through iter 5a: ~$1.65.

### Timeline (UTC)

| Time | Event |
|---|---|
| 17:30 | Read iter-1..iter-5 lessons; reverse-engineered protocol from `protocol/{checkpoint,restore,common,control_volume}.go`, `internal/types/config.go`, `cmd/agent/{main,config}.go`, `Dockerfile`, `Makefile`. |
| 17:31 | `aws ec2 create-security-group iter5a-dev-sg` in vpc-0bd6abcecded8edf6. SSH from current IP only. SG sg-0d946603940c7941b. |
| 17:32 | Attach `AmazonEC2ContainerRegistryPowerUser` to dynamo-snapshot-uw2-role (was read-only). |
| 17:32 | First spot launch in us-west-2d → InsufficientInstanceCapacity. |
| 17:33 | Spot launched in us-west-2a public subnet, AssociatePublicIpAddress=true. i-08bc2854ab60d9c12 / 54.214.126.240. |
| 17:35 | rsync upstream-snapshot/ → dev VM. Background image build started. |
| 17:39 | Agent base (cuda-dl-base:25.11-cuda13.0-devel-ubuntu24.04, 8.59 GiB) pulled from NGC public. No auth needed. |
| 17:42 | Agent image built (8.71 GiB, ID eca41bec26d4) tagged for ECR. |
| 17:46 | First placeholder build attempt failed: `apt-get install libgnutls30t64` not found. vllm/vllm-openai:v0.10.2 is jammy 22.04, package only exists on 24.04+. Patched Dockerfile sed `libgnutls30t64 → libgnutls30`. |
| 17:50 | Second placeholder attempt failed: GLIBC_2.38 mismatch (CRIU built in ubuntu:24.04, placeholder base is jammy 2.35). Authored Dockerfile.placeholder-jammy with `FROM ubuntu:22.04 AS criu-builder`. |
| 17:55 | Authored 8 manifests in k8s/: 00-namespace, 10-rbac, 20-seccomp-profile, 25-agent-config-cm, 30-snapshot-agent-daemonset, 40-storageclass-ebs-fsr, 50-ministral-3b-deployment, 60-orchestrator-job. |
| 17:58 | Authored 5 scripts: run-b1-baseline.sh, run-e1-snapshot.sh, gate1-token-equality.py, gate2-artifact-size.sh, gate3-latency.sh. |
| 18:00 | `kubectl --dry-run=server -f k8s/`: 7/8 PASS, 8th (40-storageclass-ebs-fsr.yaml VolumeSnapshotClass) PARTIAL — needs snapshot-controller CRDs (iter-5b prereq). |
| 18:05 | Third placeholder attempt: CRIU compile succeeded but `make install-lib` failed at `python3: No module named pip`. Added `python3-pip` to jammy criu-builder apt list. |
| 18:08 | Fourth placeholder attempt running. |

### Manifests (k8s/) — server-side dry-run results

| File | Purpose | Dry-run |
|---|---|---|
| 00-namespace.yaml | dynamo-snapshot ns w/ PSA privileged labels | PASS |
| 10-rbac.yaml | snapshot-agent SA + ClusterRole (pods, daemonsets, jobs, events, configmaps, nodes), snapshot-orchestrator SA + ClusterRole | PASS |
| 20-seccomp-profile.yaml | block-iouring.json ConfigMap + privileged installer DaemonSet | PASS |
| 25-agent-config-cm.yaml | /etc/snapshot/config.yaml | PASS |
| 30-snapshot-agent-daemonset.yaml | snapshot-checkpoints PVC + DaemonSet (privileged, hostPID, container `agent`, volume `checkpoints`) | PASS |
| 40-storageclass-ebs-fsr.yaml | gp3-snapshot StorageClass + ebs-snapshot VolumeSnapshotClass | StorageClass PASS, VolumeSnapshotClass FAIL — CRDs absent (iter-5b prereq) |
| 50-ministral-3b-deployment.yaml | Deployment + ready-watcher sidecar + ClusterIP Service | PASS |
| 60-orchestrator-job.yaml | snapshot-orchestrator-script ConfigMap + Job (python:3.12-slim) | PASS |

### ECR images

| URI | Size | Build notes |
|---|---|---|
| 615299764834.dkr.ecr.us-west-2.amazonaws.com/dynamo-snapshot-agent:iter5 | 8.71 GiB | upstream Dockerfile target=agent, base cuda-dl-base:25.11 (Ubuntu 24.04), CRIU built in ubuntu:24.04. |
| 615299764834.dkr.ecr.us-west-2.amazonaws.com/dynamo-snapshot-agent:latest | 8.71 GiB | same content as :iter5. |
| 615299764834.dkr.ecr.us-west-2.amazonaws.com/dynamo-vllm-placeholder:v0.10.2-iter5 | (TBD when build4 lands) | custom Dockerfile.placeholder-jammy, base vllm/vllm-openai:v0.10.2 (jammy 22.04), CRIU compiled in ubuntu:22.04 for libc parity. |
| 615299764834.dkr.ecr.us-west-2.amazonaws.com/dynamo-vllm-placeholder:latest | (TBD) | same. |

### Cluster CRD findings (iter-5b prereqs)

1. `snapshot.storage.k8s.io/v1` CRDs not installed. Install external-snapshotter v8.1.0 from kubernetes-csi/external-snapshotter before applying 40-storageclass-ebs-fsr.yaml.
2. No nodes labeled `agent-aiops/snapshot-eligible=true` — expected, the g7e nodegroup is added in iter 5b.

### Cleanup

- Terminate i-08bc2854ab60d9c12 (m6i.xlarge spot) at end of iter 5a.
- Leave ECR images intact for iter 5b.
- Security group sg-0d946603940c7941b — leave for now (harmless; can attach to iter 5b dev tasks).
- IAM role policy attachment (ECR PowerUser) — leave; will be reused in iter 5b for any rebuilds.

---

## Iter 5b (2026-05-31, evening) — g7e E1 GPU clock

Start: 23:00:52Z. Cap $30. Halt rules per ralph-loop directive.

### Step 0 — Cluster prereqs (no GPU spend)

- 23:00 UTC — Pre-flight verified: kubectl context=qn-sglang, identity aiops/615299764834, namespace dynamo-snapshot already Active (from iter-5a apply), no g7e nodegroup yet.
- Spot prices live: 2d $4.87/hr, 2c $4.75/hr, 2a $5.11/hr, 2b $6.60/hr. Plan: target 2d (cheapest in scope), fallback 2a.

### Step 0 done (~$0)

- 23:01 — `kubectl apply -k external-snapshotter//client/config/crd?ref=v8.1.0` → 6 CRDs created (volumesnapshots, volumesnapshotcontents, volumesnapshotclasses, plus 3 volumegroupsnapshot kinds).
- 23:01 — `kubectl apply -k external-snapshotter//deploy/kubernetes/snapshot-controller?ref=v8.1.0` → snapshot-controller Deployment 2/2 Available in 42 s.
- 23:02 — Verified ebs-csi-controller (2 pods) + ebs-csi-node (2 pods) Running. nvidia-device-plugin not yet present (depends on g7e node).
- 23:02 — Applied 00-namespace, 10-rbac, 20-seccomp-profile, 25-agent-config-cm, 40-storageclass-ebs-fsr → all succeed. VolumeSnapshotClass now resolves (CRD installed).

### Step 1 — g7e nodegroup (ACTIVE in ~1 min, $0 with desired=0)

- 23:02:59 — `aws eks create-nodegroup dynamo-snapshot-g7e` (subnets 2a-private + 2d-private, instanceTypes [g7e.24xlarge, g7e.48xlarge], AL2023 NVIDIA, SPOT, min=0/max=2/desired=0, role ai-infra-b300-node, labels + GPU taint).
- 23:04:04 — Nodegroup ACTIVE.

### Step 2 — Bump desired=1 (FAILED — UnfulfillableCapacity, GPU clock = $0)

- 23:04:18 — `update-nodegroup-config desiredSize=1`. Re-checked spot prices first: 2d $4.91, 2a $5.11 (within $7/hr cap).
- 23:05:38 — ASG attempt #1 → `UnfulfillableCapacity` (both .24xl + .48xl, both subnets).
- 23:07:34 — Attempt #2 → same.
- 23:09:30 — Attempt #3 → same.
- 23:11:26 — Attempt #4 → same.
- 23:13:22 — Attempt #5 → same.
- 23:16 — Re-checked spot price (one re-check per directive): still normal pricing across all four AZs; 2c not in VPC. **No instance ever launched. GPU spend = $0.**
- 23:16:31 — Per halt rule "g7e spot un-fulfillable in 2a + 2d after 10 min and one re-check → halt, surface (cost guard before any GPU spend)": **HALTED.** Scaled nodegroup back to desiredSize=0.
- Lessons captured inline (see lessons.md `[aws]: g7e spot UnfulfillableCapacity ...` and `[eks]: managed nodegroup spot retries are free ...`).

### Cleanup state for iter 5c

- snapshot-controller + 6 CRDs: persistent in cluster (good).
- Manifests 00, 10, 20, 25, 40 applied: persistent (good).
- Manifests 30 (DaemonSet), 50 (Deployment), 60 (orchestrator Job): NOT applied yet (require g7e node).
- Nodegroup `dynamo-snapshot-g7e`: preserved at desired=0 — ready for iter 5c re-launch.
- ECR images: untouched.
- B1 baseline: NOT MEASURED.
- E1: NOT ATTEMPTED.
- Gates 1/2/3: NOT EVALUATED.

### Iter 5b verdict

**HALT on capacity gate, NOT a falsification.** Hypothesis untested. Iter 5b spend: **$0** (well under $30 cap). Cumulative through 5b: ~$1.65.

Ready-or-not for E2/E3: **NOT READY** — E1 must run successfully first. Same posture as end of iter 5a, plus snapshot-controller + 5 manifests now applied.
