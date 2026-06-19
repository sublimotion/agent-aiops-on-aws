---
blueprint: fin-rag-answer-g7e
spec: domains/gpu-serving/specs/fin-rag-answer-g7e.md
parent: fin-rag-answer (B200)
status: benchmarked
last_stage: 6
date: 2026-06-12
---

# fin-rag-answer-g7e — Progress

## SESSION 2026-06-12 (cont.): g7e CAPACITY WON via multi-region ODCR — 3-way SHIPS
The us-east-2 capacity shortage was real and durable (see prior section), but the
**multi-region ODCR chase** (`scripts/chase-g7e-multiregion.sh`) broke it: probing all 10
g7e AZs across 4 regions with `create-capacity-reservation --instance-match-criteria open`
(fails in ~2s on InsufficientInstanceCapacity vs ~33min for an EKS nodegroup) **won Tokyo
ap-northeast-1a** (`cr-0802787bafa24b760`) in ~30s. Launched a plain g7e.12xlarge EC2 there
(`i-0de1456111eb3b30e`, 54.199.69.41), downloaded FP8 weights into NVMe via a hf_transfer
container, and ran BOTH engines.

- **vLLM 0.22.1 (tp2x1, single replica)** — full sweep c8/130/200/256/512. Stage-0 smoke PASS
  (NCCL sm_120+PCIe bug avoided via vLLM custom all-reduce). Single replica on 2 PCIe GDDR7
  GPUs **saturates hard**: e2e_p50 7.1s @ c8 (already over the 6.5s SLO), 66.6s @ c130,
  206s @ c512. Cost floor ~$0.15/1M total tokens (c256). SLO FAIL at every point — expected,
  this is 1 replica absorbing load that B200/H200 spread across 4–8 replicas.
- **SGLang v0.5.12.post1-cu130 (tp2x1)** — cu130 ships NCCL ≥2.26.2 → NCCL init clean. MoE
  backend search on sm_120: `triton` (SMEM overflow 147456>101376), `triton_kernel` (weight
  shape mismatch 1344 vs 1024), `flashinfer` (invalid name) ALL FAILED; **`flashinfer_cutlass`
  WORKS** — the first MoE backend that serves Nemotron-3-Super on sm_120. Smoke PASS, sweep
  running. SGLang notably faster at low conc (c8 e2e_p50 4.6s vs vLLM 7.1s) — radix cache + a
  working MoE kernel.

**3-way cost comparison now SHIPS** (B200 vs H200 vs g7e). Enriched via
`../fin-rag-answer/scripts/enrich-to-standard.py --platform g7e` (PLATFORMS[g7e] corrected to
g7e.12xlarge / 2 GPU / ap-northeast-1 / $12.01727/hr on-demand; engine derived from config tag),
schema-valid, symlinked into results-vault.

## SUPERSEDED — prior "capacity-BLOCKED, ships as 2-way" conclusion
The us-east-2-only chase (`chase-g7e-capacity.sh` 60 rounds + `watch-g7e-capacity.sh` 180
rounds, ~5h) never freed capacity in 2a/2b — that read was correct *for us-east-2*. The fix
was not waiting longer but **widening the search surface**: ODCR probing across all 4 g7e
regions. Lesson captured in memory `infra-g7e-capacity-chase`.

## RESUME HERE (paused 2026-06-11 ~17:15 EDT, user shutting down laptop)
All GPU nodegroups DELETED — **$0 billing**, nothing running. Persisted at ~no cost: FSx
weights `vllm-qwen3-fsx-pvc` (~124GB FP8 + ~240GB BF16, the reuse premise), ECR images
(`vllm-openai:v0.22.1`, `sglang:v0.5.12.post1-cu130`), EKS cluster `qwen3-next-bench-eks-cluster`,
all specs/manifests/blueprints.

**What happened:** g7e.48xl capacity-exhausted across all 3 us-east-2 AZs. A race (od+spot ×
3 AZ × 3 sizes) landed a **SPOT g7e.12xlarge** (2 GPUs) in us-east-2b — but I mistakenly kept
the spot winner (deleted the on-demand loser), and spot reclaimed it after ~10 min before the
weight copy ran. Node fix work that IS preserved in manifests/scripts: device-plugin needs the
`ai-infra/g7e` toleration patch + node `nvidia.com/gpu.present=true` label; `setup-nvme-raid.sh`
now handles single-disk (12xl) AND multi-disk (48xl) NVMe.

**Next session, do in parallel (per user 2026-06-11):**
1. **g7e** — relaunch ON-DEMAND (not spot — it gets reclaimed mid-bench), multi-AZ range
   (g7e 48/24/12xl across all 3 private subnets). desiredSize 1.
2. **H200** — NEW third comparison point. Spec written: `domains/gpu-serving/specs/fin-rag-answer-h200.md`.
   Launch p5e.48xlarge (8× H200 141GB, Hopper sm_90, NVSwitch, mature NCCL) **SPOT in us-east-2a**
   (use2-az1, subnet-0fced510ea62b874e). Live spot ~$9.77/hr (2026-06-11). 8× 3800GB NVMe (RAID0).
   Blueprint dir not yet created. Reuses same cluster/FSx/ECR/namespace as B200+g7e.
3. Both: cross-AZ FP8 copy FSx(2c)→NVMe, then vLLM 0.22.1 Stage-0 smoke → SGLang smoke → fin-support
   SLO bench @ conc 8 & 130 → 3-way $/1M-tok comparison (B200 baseline $0.040/1M @ spot $32).
   NOTE g7e 12xl = 2 GPUs → agg-tp2-x1 single replica, per-replica throughput only.



Cost-comparison addendum: re-run the fin-support RAG SLO benchmark on g7e (sm_120, PCIe)
vs the B200 winner. Two engines: vLLM (apples-to-apples vs B200) + SGLang (reproduce
upstream #20541 known-good sm_120 config; the only engine where prefix cache works here).
Sweep guided by B200 results but verified empirically — do NOT assume sm_120 transfer.

## Reuse plan (us-east-2, same cluster as B200)
- Cluster `qwen3-next-bench-eks-cluster` (context `fin-rag-b200`) — REUSED.
- FSx PVC `vllm-qwen3-fsx-pvc` (holds FP8 ~124GB + BF16 ~240GB) — REUSED, no HF re-download.
  FSx in us-east-2c; g7e node in us-east-2a → one-time CROSS-AZ bulk copy to local NVMe.
- ECR `vllm-openai:v0.22.1` — REUSED (PRIMARY, smoke-gated; fallback :v0.18.1). SGLang
  `sglang:v0.5.12.post1-cu130` — STAGED to ECR via skopeo (13GB, pushed 16:17, cu130 clears
  the NCCL-2.25.1 sm_120+PCIe bug).
- Namespace `ml-inference` — REUSED.

## Stage log
- **Stage 1 (compute)** — DONE. **g7e.48xlarge capacity-exhausted across ALL us-east-2 AZs**
  (on-demand + spot, use2-az1/az2/az3 all returned `InsufficientInstanceCapacity`). The earlier
  "stuck CREATING / no ASG" read was a red herring — the ASG existed and was silently retrying
  failed launches; `modified==created` on the nodegroup does not tick during provisioning.
  Real signal = ASG scaling activities. Resolution: raced **two nodegroups** (on-demand +
  spot) × **3 AZs** × **3 sizes** (g7e 48/24/12xl) = 18 pools. Winner: **SPOT g7e.12xlarge in
  us-east-2b** (`i-0ae101c04d62c4d57`, node `ip-10-0-28-185`), Ready in 39s. Deleted all losing
  nodegroups. **SCALE DELTA: 12xl = 2 GPUs (not 8).** → agg-tp2-x1 single replica; FP8 still
  needs TP=2 (124GB > 96GB/GPU). Engines run SEQUENTIALLY (each TP2 replica = both GPUs).
  Benchmark is per-replica, not 8-GPU aggregate; $/1M-tok comparison still valid (scales w/ GPU-hr).
- **Device plugin fix** — DONE. AL2023 NVIDIA AMI node did NOT self-label `nvidia.com/gpu.present`,
  and the shared `nvidia-device-plugin` DS (kube-system) only tolerated `ai-infra/b200`. Fixed:
  labeled node `nvidia.com/gpu.present=true` + added additive `ai-infra/g7e` toleration to the DS
  (kept b200). 2 GPUs now allocatable.
- **NVMe** — g7e.12xl has a SINGLE 3800GB NVMe (no RAID needed, unlike 48xl multi-disk). Format+mount only.
- **FSx SG** — DONE. Opened FSx SG sg-07c6da755ffe8af2d to cluster SG sg-0abb08cf4d13be131
  on Lustre ports 988 + 1018-1023 (LT-less managed nodes carry only the cluster SG, not the
  node shared SG that FSx already trusted — B200 lesson #11).

## Open dependencies before serving
1. g7e node Ready (spot fulfillment or on-demand fallback).
2. NVMe RAID0 built (`nvme-raid-setup.yaml` — auto-detects disks).
3. FP8 weights copied FSx→NVMe (`copy-fp8-to-nvme.yaml`, cross-AZ, ~124GB).
4. vLLM smoke (TRITON_ATTN baseline) — confirm sm_120 FP8 loads + coherent Fin output.
5. SGLang image to ECR + NCCL>=2.26.2 verified, then #20541 config smoke.

## Key risk carried from B200
Mamba2 automatic prefix caching is NON-FUNCTIONAL on vLLM 0.18.1 at any TP (0 hits / 1.8M
queries, B200 Leg 4). So on the vLLM g7e path the RAG prefix-cache win does NOT materialize —
SLO rests on raw prefill throughput, which is g7e's weakest axis (GDDR7 « HBM3e). SGLang radix
cache DOES work on g7e (#20541) → likely the most interesting g7e finding.
