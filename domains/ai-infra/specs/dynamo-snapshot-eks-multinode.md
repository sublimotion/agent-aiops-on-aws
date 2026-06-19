# Dynamo Snapshot — EKS Multi-Node + Multi-GPU Containerized Validation

## Status: DRAFT

## Hypothesis

NVIDIA Dynamo Snapshot (CRIU + `cuda-checkpoint`) restores warmed vLLM workers in **≤15 s** when run **inside `nvidia-container-runtime` containers on EKS**, both for (a) replica-N TP=1 pods scheduled across multiple g7e nodes and (b) a single TP=4 pod on one g7e.48xlarge — clearing the bare-metal multi-GPU halt observed in the predecessor spec.

The containerized mount namespace exposes only the requested `/dev/nvidiaN` device nodes, so CRIU's `cuda_plugin` no longer trips on the 51-fd-across-all-GPUs problem that killed the p5e bare-EC2 attempt.

## Falsification criteria

- **E1 (replica-N TP=1)**: median pod-create-to-first-token > 30 s across 4 replicas restoring concurrently from one canonical EBS snapshot → containerized restore is not delivering the predicted single-GPU win at replica scale.
- **E2 (cross-node)**: cross-node restore from a region-shared EBS snapshot regresses > 25% vs same-node restore → snapshot artifact is not portable across nodes in practice.
- **E3 (TP=4 single-pod)**: any of (a) `criu dump` fails on `/dev/nvidiactl` or `/dev/nvidia[0-3]` fds inside the container, (b) restored worker's TTFT or throughput regresses > 5% vs freshly warmed TP=4, (c) NCCL state corruption (rank mismatch, hang) on first post-restore generation → containerized execution does NOT clear the multi-GPU C/R gap; multi-GPU snapshot remains blocked on upstream `HANDLE_DEVICE_FD` plugin work.
- Any cell: artifact size > `weights + 4 GiB + 1 GiB × TP` → vLLM `sleep(level=1)` integration broken or NCCL shm not released.

## Why this matters

The predecessor spec (`dynamo-snapshot-coldstart.md`) proved the single-GPU primitive works on bare g6 (Ada sm_89) but **halted on bare p5e** because the NVIDIA driver opens all 8 `/dev/nvidiaN` fds regardless of `CUDA_VISIBLE_DEVICES`, and CRIU's `cuda_plugin` lacks a `HANDLE_DEVICE_FD` hook. The lessons.md noted the exact pivot:

> NVIDIA's K8s-native `snapshot-agent` flow likely sidesteps this by running the workload in a container under `nvidia-container-runtime`, which injects ONLY the requested GPU's device nodes via OCI hooks — the workload pod literally doesn't see `/dev/nvidia[1-7]` because they're not bind-mounted into the container's mount namespace.

This spec runs that pivot. A positive E1+E2 result enables the production rule from the predecessor spec on multi-GPU **hosts** (any g7e/p5e), even though TP must stay = 1 per pod. A positive E3 additionally extends the rule into the "limited validation" multi-GPU TP regime NVIDIA flags as preview, opening Spec E to TP>1 production workloads.

Cold-start pain points this addresses on g7e specifically: Qwen3.5-MoE 122B-A10B-FP8 cold start (model load + CUDA graph capture across TP=4) is currently 2–4 min per replica.

## Stage-budget claim

**E1 — Ministral-3B, replica-N≥2, TP=1 per pod, EBS+FSR**:

| Stage | Baseline (sec) | Predicted with technique (sec) | Why |
|---|---|---|---|
| Node provision | 0 (warm pool) | 0 | unchanged |
| Image pull | 5–15 (EBS prebake) | 5–15 | unchanged |
| Container start | 5–10 | 5–10 | unchanged |
| Model load | 30–60 | 0 | rolled into restore |
| JIT / compile | 10–30 | 0 | rolled into restore |
| EBS volume-create from snapshot | — | 8–10 | per-replica, FSR-warm |
| EBS attach + mount | — | 9–12 | per-replica |
| CRIU + cuMemMap restore | — | 3–8 | parallel-memfd + AIO |
| First token | 1–5 | 1–5 | unchanged |
| **Total** | **50–120** | **25–60** | ~2–4× |

Replica index: **N ≥ 2** (replica-1 still pays first-warm cost to *produce* the snapshot).

**E3 — Qwen3-next-fp8, TP=4, single pod, EBS+FSR**:

| Stage | Baseline (sec) | Predicted with technique (sec) | Why |
|---|---|---|---|
| Image pull | 5–15 | 5–15 | unchanged |
| Container start | 5–10 | 5–10 | unchanged |
| Model load (TP=4 shard) | 60–120 | 0 | rolled into restore |
| JIT + CUDA graph capture (×4 ranks) | 60–180 | 0 | rolled into restore |
| EBS volume-create + attach | — | 10–15 | larger artifact (~130 GB FP8) |
| CRIU restore (×4 ranks) | — | 10–25 | TP=4 process tree, parallel-memfd |
| NCCL re-init / sanity | — | 2–5 | new step — open question whether needed |
| First token | 1–5 | 1–5 | unchanged |
| **Total** | **130–330** | **30–75** | ~3–5× |

E3 has the highest uncertainty — first measurement of containerized multi-GPU CRIU on AWS as far as we know.

## Matrix

| Axis | Values |
|------|--------|
| Models | **Ministral-3-3B-Instruct-2512** (TP=1, ~6 GiB, already at `s3://vllm-model-cache-615299764834/`); **Qwen3-next-fp8** (TP=4, ~130 GB FP8, already at `s3://qwen3-next-bench-models-...` and `s3://qn-sglang-models-...`) |
| Hardware | **g7e family — instance size flexible** based on spot capacity. Preferred: g7e.24xlarge (4× RTX PRO 6000 Blackwell PCIe sm_120, 96 GB VRAM each, 1 EFA). Acceptable: g7e.48xlarge (8× same GPU, 4 EFA) — note 48xl can host E1+E3 simultaneously. Region: **us-west-2** (g7e available in all 4 AZs). |
| Topology | (a) 4 replicas × TP=1 on **1 g7e node** (24xl: fills the node; 48xl: half-fills); (b) 2 replicas × TP=1 on **each of 2 g7e nodes** (cross-node, sized per spot availability); (c) 1 replica × TP=4 on **1 g7e node** (24xl: fills; 48xl: fills 4 of 8 GPUs) |
| Storage backend | **EBS gp3 + FSR** (primary); FSx Lustre PERSISTENT-1000 (deferred — see end of spec) |
| Engine | vLLM only (Dynamo snapshot supports vLLM exclusively in preview) |
| Variants | baseline cold start, snapshot restore |

**Cells run** (4 + 2 baselines):

| Cell | Topology | Model | Storage | Notes |
|---|---|---|---|---|
| **E1** | 4 pods × TP=1 on 1 node | Ministral-3B | EBS+FSR | Validates containerized C/R + replica-N concurrent restore |
| **E2** | 2 pods × TP=1 × 2 nodes (4 total) | Ministral-3B | EBS+FSR (region-shared snapshot) | Validates cross-node portability |
| **E3** | 1 pod × TP=4 on 1 node | Qwen3-next-fp8 | EBS+FSR | First containerized multi-GPU CRIU test on AWS |
| **B1** | 1 pod × TP=1 baseline | Ministral-3B | S3 → vLLM `--load-format auto` | Baseline for E1/E2 |
| **B2** | 1 pod × TP=4 baseline | Qwen3-next-fp8 | S3 → vLLM `--load-format auto` | Baseline for E3 |

**Deferred** (added to end of spec, see §FSx addendum): FSx Lustre PERSISTENT-1000 cells for E1, E2, E3 — produces the latency-vs-$/mo Pareto frontier. Run only after EBS+FSR cells pass.

## Stage 0: prerequisites and gates

Before booking any g7e capacity, three gates must pass on cheaper hardware:

1. **Containerized single-GPU C/R on g6.xlarge** (extends predecessor spec). Run vLLM in a `nvidia-container-runtime` container with `--gpus '"device=0"'`. CRIU runs from a privileged sidecar against the workload pod's PID. Confirms: (a) seccomp profile blocking io_uring works in containerized vLLM; (b) only `/dev/nvidia0` + `/dev/nvidiactl` + `/dev/nvidia-uvm` appear inside the container; (c) Gates 1/2/3 from predecessor spec all PASS in containerized mode. Cost: ~$2 / ~4 hr.
2. **EBS+FSR snapshot pipeline validated standalone**. Bake one canonical EBS snapshot from a warmed Ministral-3B C/R artifact in us-east-1. Confirm `aws ec2 create-volume --snapshot-id` from FSR-enabled snapshot completes in < 12 s in target AZ. Cost: ~$3.
3. **EKS cluster prep**. Reuse `qn-sglang-eks-cluster` (us-west-2, k8s 1.32, ACTIVE) — currently has `ai-infra-b300-spot`, `ray-ft-gpu`, `ray-ft-system` nodegroups but no g7e. Add a g7e managed nodegroup (instance type list: `g7e.24xlarge`, `g7e.48xlarge` for capacity flexibility, min=0, max=2, desired=0) with the AL2023 NVIDIA AMI (driver ≥ 580, NCCL ≥ 2.26.2). Install `snapshot-agent` DaemonSet (privileged, hostPID=true) and `seccomp-profiles` ConfigMap. Hard constraint: **all C/R must run on K8s**, never bare-EC2 — bare-metal multi-GPU is the documented dead end from the predecessor spec. Cost: ~$0.10/hr control plane (already running).

If Stage 0.1 fails, **halt** — the containerized hypothesis is wrong and we file the cuda_plugin gap upstream. If 0.2 or 0.3 fail, fix and retry — they're operational, not hypothesis-killing.

## Baseline

For each model:
- Standard `domains/gpu-serving/blueprints/<name>/` deploy on g7e (Ministral-3B blueprint exists; Qwen3-next requires the existing `qwen3-next/` blueprint adapted for g7e — TP=4 confirmed working in MEMORY.md "Qwen3.5 MoE on vLLM 0.18" notes).
- `--load-format auto`, weights pulled from S3 → instance NVMe → vLLM
- Cold pod, full warmup including a synthetic prompt to force CUDA graph capture
- Measured pod-create → first-token-streamed

Snapshot variant uses identical blueprint + `snapshot-agent` DaemonSet + entrypoint hooks (`/tmp/ready-for-checkpoint` signal, `/tmp/restore-complete` poll).

## Measurement

- **Primary**: pod-create timestamp → first-token-streamed timestamp (seconds), per replica, p50 + p95 over 5 cold restores.
- **Concurrent-restore signal** (E1/E2): time-to-all-replicas-ready when N replicas restore simultaneously from the canonical snapshot. Captures EBS+FSR throughput contention if any.
- **Correctness gates** (per predecessor spec):
  - Gate 1: byte-identical token IDs vs freshly warmed worker (SHA256 of first 64 tokens).
  - Gate 2: artifact size ≤ `weights + 4 GiB + 1 GiB × TP` (size-tiered rule from predecessor spec, extended for TP).
  - Gate 3: restore wall-clock < 30 s for E1/E2, < 60 s for E3.
- **TP=4-specific** (E3): NCCL allreduce sanity check post-restore (small synthetic tensor), confirm rank topology matches pre-checkpoint.
- **Cost-per-restore-second**: per-replica $/restored-replica-second for EBS+FSR tier (volume + FSR amortization).
- **Output**: enriched JSON per `standards/benchmark-commons/PROPOSAL.md`, stage breakdown matching the budget tables above.
- **Tool**: extend `domains/ai-infra/shared/cold_start_harness.py` to emit a `restore_breakdown` block with cells for `volume_create`, `attach`, `mount`, `criu_restore`, `cuda_restore`, `nccl_reinit`, `first_token`.

## Fixtures

- `domains/gpu-serving/blueprints/ministral-3b/` — primary substrate for E1/E2/B1
- `domains/gpu-serving/blueprints/qwen3-next/` — substrate for E3/B2 (override TP=4 if not already)
- New: `domains/ai-infra/blueprints/dynamo-snapshot-eks/` — sibling to `dynamo-snapshot/`, holds:
  - Helm chart for `snapshot-agent` DaemonSet (privileged, hostPID, mounts `/var/lib/kubelet/pods` for pod PID resolution)
  - Sidecar image with CRIU 4.2 + PR #3021 (parallel-memfd) and `cuda-checkpoint` binary, seccomp profile blocking `io_uring_*` syscalls (matches Dynamo `protocol/checkpoint.go` behavior)
  - Snapshot orchestrator Job that drives checkpoint + EBS-snapshot-create + multi-replica restore via Kubernetes API
  - Reuses the AMI baked in predecessor spec (`ami-041914c9e9b61b15e`) where possible — driver/CRIU stack is sm-agnostic per the p5e cell finding

## Rule the experiment would produce

If E1+E2 pass (containerized single-GPU at replica-N):

> For single-GPU vLLM inference workloads on EKS where replica-N cold-start latency matters (autoscaling small/medium models, cost-aware-routing tiers), use the `dynamo-snapshot-eks` DaemonSet pattern with **EBS gp3 + FSR** as the snapshot backend. Each replica creates an independent gp3 volume from one canonical FSR-enabled snapshot via the EBS CSI driver. Expected p50: 25–60 s pod-create-to-first-token vs 50–120 s baseline. Skip when (a) restored worker is the only replica (no warm-pool peer to inherit snapshot from), (b) model is multi-GPU TP > 1 (defer to the E3 result), or (c) the engine is SGLang (snapshot/restore is vLLM-only in preview).

If E3 also passes (containerized TP=4):

> Extension: same pattern works for single-pod TP > 1 vLLM on g7e/p5e provided the pod runs under `nvidia-container-runtime` with explicit `--gpus '"device=0,…,N-1"'` (so peer GPUs are NOT bind-mounted into the container's mount namespace). NCCL state survives CRIU dump+restore on the same node; cross-node TP > 1 restore is still upstream-blocked.

If E3 falsifies:

> Multi-GPU TP > 1 snapshot is still blocked on upstream CRIU `HANDLE_DEVICE_FD` plugin work even with containerization, because [observed-failure-mode]. File the gap upstream against `cuda_plugin`. Single-GPU rule (E1/E2) stands.

## Out of scope

- Multi-node TP > 1 (NCCL multi-node + NIXL quiesce) — upstream roadmap, revisit when shipped.
- TensorRT-LLM (upstream roadmap).
- GPUDirect Storage (GMS) restore path — gated on CUDA driver patch.
- SGLang — snapshot/restore is vLLM-only in preview.
- Replica-1 (first-ever) cold start — by construction, snapshot only helps replica-N.
- Cross-AZ snapshot replication — single-AZ in scope only; cross-AZ is incremental EBS-snapshot work, not C/R work.

## Persistent caches via EBS snapshot — applicable

Unlike the predecessor spec's stance (which favored FSx Lustre), this spec **leads with EBS+FSR** for the snapshot artifact:

- Snapshot artifact is per `(model, vllm_version, hardware, driver, config_hash)` — same lifecycle as the spec-c-ebs-snapshot AOT compile cache.
- Predecessor spec's production-rule table already pegs EBS+FSR as the production default (3–15× cheaper than FSx P-125 and 20–100× cheaper than P-1000 for single-artifact intermittent-burst workloads, with ~10–20 s latency penalty).
- This spec validates that recommendation directly rather than starting from FSx as a counterfactual.

Setup mirrors `domains/ai-infra/blueprints/spec-c-ebs-snapshot/`:
1. Bake snapshot once on a warmed-up replica-1 (CRIU dump → write artifact tree to a per-AZ gp3 volume → `aws ec2 create-snapshot`).
2. Tag with `(model, vllm_version, tp_size, gpu_arch, driver, config_hash)`.
3. Enable FSR in the target AZ during scale events; toggle off when cold.
4. Each replica's PVC creates an independent gp3 volume from the snapshot via EBS CSI driver. No Multi-Attach.

## Cost estimate

- **Stage 0.1 containerized smoke**: 1× g6.xlarge spot, ~$0.50/hr × ~4 hr = **~$2**
- **Stage 0.2 EBS+FSR pipeline**: snapshot create + FSR for 1 hr in 1 AZ = **~$3**
- **Stage 0.3 EKS prep**: control plane prorated negligible; nodegroup creation cost = $0 (desired=0)
- **E1 + E2** (Ministral-3B replica-N + cross-node): 2× g7e.24xlarge spot ~$2.20/hr × ~4 hr = **~$18** (or 2× 48xl ~$4.40/hr × 4 hr = ~$36 if only 48xl has spot)
- **E3** (Qwen3-next-fp8 TP=4): 1× g7e.24xlarge spot ~$2.20/hr × ~6 hr = **~$14** (longer because of larger artifact + TP=4 NCCL gating; bumps to ~$26 on 48xl)
- **EBS gp3 volumes**: 4 × 50 GiB × few hours, prorated **~$2**; FSR per-snapshot per-AZ ~$0.75/hr × 8 hr = **~$6**
- **S3 transfer**: model staging from existing buckets, intra-region negligible
- **Buffer**: ~$10
- **Total cap: ~$80 (24xl spot) / ~$100 (48xl fallback)** — revised iter-4 from original $60/$90 after pre-flight pricing check showed g7e.24xl spot drifted 2-3× since spec was drafted (us-west-2a $5.11/hr, 2d $4.80/hr; 2b $6.55/hr skipped). E1 alone is now ~$20-25 vs spec's original ~$9. Cluster VPC has no us-west-2c subnet so cheapest AZ is unreachable without VPC changes.

## §FSx Lustre addendum (deferred — run only if EBS+FSR cells PASS)

If E1/E2/E3 pass on EBS+FSR, run a parallel FSx Lustre PERSISTENT-1000 cell for each (E1', E2', E3') to produce the latency-vs-$/mo Pareto frontier promised by the predecessor spec's production rule:

| Tier | Restore latency (130 GB Qwen3-next artifact) | Steady-state $/mo (1 artifact) | When to use |
|---|---|---|---|
| **EBS gp3 + FSR** | (measured in E1/E2/E3) | ~$10–100 (FSR on-demand) to ~$550 (always-on, 1 AZ) | **Production default** |
| **FSx Lustre PERSISTENT-1000** | (measured in E1'/E2'/E3') | ~$1,095+ (1.2 TiB minimum, always-on) | Many concurrent restores across AZs, always-warm pool |

Cost for FSx addendum: ~$15 (FSx P-1000 × 10 hr) + ~$15 (g7e re-runs) = **~$30**.

If EBS+FSR results dominate FSx by enough margin (latency penalty < 30 s for the smaller artifacts; cost advantage > 10×), the addendum can be skipped and the rule defaults to EBS+FSR with FSx as a "consult an adult" footnote.

## References

- Predecessor spec: `domains/ai-infra/specs/dynamo-snapshot-coldstart.md`
- Predecessor lessons (especially p5e halt diagnosis): `domains/ai-infra/blueprints/dynamo-snapshot/lessons.md`
- Spec C-EBS for the EBS+FSR pattern: `domains/ai-infra/blueprints/spec-c-ebs-snapshot/`
- Upstream Dynamo: https://github.com/ai-dynamo/dynamo
- CRIU: https://github.com/checkpoint-restore/criu (PRs #3021, #3022 by `dfeigin-nv`)
- NVIDIA blog: https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/
- Memory: `MEMORY.md` § "Qwen3.5 MoE on vLLM 0.18" (TP=4 g7e config), § "Critical: NCCL Broken on Blackwell PCIe (g7e)" (NCCL 2.25.1 broken — must use NCCL 2.26.2+ / NGC 25.03+ in the snapshot-agent and workload images)
