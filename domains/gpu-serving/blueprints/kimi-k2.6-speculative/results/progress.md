---
blueprint: "kimi-k2.6-speculative"
domain: "gpu-serving"
spec: "domains/gpu-serving/specs/kimi-k2.6-speculative.md"
status: "complete"
last_updated: "2026-05-13T22:40:00Z"
last_stage: "stage-8"
region: "us-east-1"
az_id: "use1-az6"
subnet_id: "subnet-05569398360910f46"
vpc_id: "vpc-07ff0e6bdc3cac475"
eks_cluster: "inference-eks-v132"
model_bucket: "s3://kimi-k2-bench-models-20260216163240701700000006"
results_bucket: "s3://kimi-k2-bench-results-20260216163240701700000007"
ami_id: "ami-027c3ae8019fc0d3a"

stages:
  - id: "stage-0"
    name: "Deployment card lookup"
    status: "pending"
  - id: "stage-1"
    name: "Foundation (subnet routing, IAM, SG)"
    status: "in_progress"
  - id: "stage-2"
    name: "Image build (vLLM EAGLE3, SGLang, bench runner)"
    status: "pending"
  - id: "stage-3"
    name: "Storage and model staging (K2.6 + kimi-k2.6-eagle3 to S3)"
    status: "pending"
  - id: "stage-4"
    name: "GPU spot node launch (p6-b300 in use1-az6)"
    status: "pending"
  - id: "stage-4a"
    name: "GPU health validation"
    status: "pending"
  - id: "stage-5"
    name: "Serving stack deployment"
    status: "pending"
  - id: "stage-6"
    name: "Benchmark Phases 0-5"
    status: "pending"
  - id: "stage-7"
    name: "Readiness audit"
    status: "pending"
  - id: "stage-8"
    name: "Compound"
    status: "pending"

phases:
  phase-0-roofline: {status: "complete", artifact: "results/phase-0-roofline/"}
  phase-1-sglang-eagle3: {status: "complete", artifact: "lessons.md Phase 1 results"}
  phase-1b-eagle3-sweep: {status: "complete", artifact: "results/phase-1b/", winner: "s4_d4_k1"}
  phase-2-vllm-eagle3: {status: "dropped", artifact: null, reason: "L13 vision-tower blocker; user decision: no source builds"}
  phase-3-dynamic-mla: {status: "dropped", artifact: null, reason: "Low expected delta vs prefix-cache baseline"}
  phase-4-fullstack: {status: "complete", artifact: "results/phase-4/"}
  phase-5a-default-stack: {status: "complete", artifact: "results/phase-5/5a-default-stack/"}
  phase-5b-no-cuda-graph: {status: "complete", artifact: "results/phase-5/5b-no-cuda-graph/"}
  phase-5c-tp4-dp2: {status: "complete", artifact: "results/phase-5/5c-tp4-dp2/"}
  phase-5d-fp4-probe: {status: "complete", artifact: "results/phase-5/5d-fp4-probe/", note: "FP4 requires cutlass 3.x, not shipped"}

artifacts:
  lessons: false
  readiness_audit: []
  deployment_log: []
  compound: []
  benchmark_report: false
---

# Progress: kimi-k2.6-speculative

## Session 2026-05-13

Kickoff — moving fast per user direction. Full phase 0-5 EAGLE3 + dynamic MLA study on B300 spot.

### Stage 1 — Foundation (done)

- [x] Verified spot availability: `aws ec2 describe-spot-price-history` → use1-az6 @ $26.41/hr, placement score 9/10
- [x] Confirmed existing EKS cluster `inference-eks-v132` in us-east-1
- [x] Existing subnet `subnet-05569398360910f46` in az6 has NAT-only routing (no inbound) — replaced
- [x] Created **public subnet in us-east-1c**: `subnet-079555bada92f1c9b` (10.192.13.0/24, route table `rtb-053b3234bccd30bd5` with IGW + S3 endpoint)
- [x] Security group `sg-02c4dbf438ee20741` (kimi-k26-spec-gpu) — SSH/vLLM/SGLang from my IP, self-ref, EKS cluster cross-ref both ways
- [x] IAM role `kimi-k26-gpu` with S3 read (model bucket) + write (results bucket) + SSM + ECR read
- [x] IAM role `kimi-bench-helper` with S3 read/write on model bucket

### Stage 4 — GPU spot node

- [x] Launched `i-0072c37171bfb66f9` in NAT-only subnet — **terminated** (no inbound SSH)
- [x] Relaunched `i-090e91fde6f184077` in public subnet — RUNNING at 13.217.57.102 / 10.192.13.65
- [x] Bootstrap: `dnf install mdadm xfsprogs jq awscli` added (was missing on DLAMI), NVMe RAID0 28TB mounted at /mnt/nvme
- [x] GPU detected: 8× B300 SXM6 AC @ 275 GiB each (matches memory)
- [x] Docker + ECR login working
- [x] sync-loop-gpu.sh and spot-reclaim-watcher.sh running in background

### Stage 3 — Weight staging (in progress)

- [x] Helper EC2 `i-003e74da2e0488be9` (98.93.9.233) — hf download stalled at ~70/95 shards, killed
- [x] **Switched to direct GPU-node HF download** (p6-b300 has 3200 Gbps network)
- [x] Draft model `lightseekorg/kimi-k2.6-eagle3` DONE — 5 files, 6 GB on /mnt/nvme/models/kimi-k26-eagle3/
- [ ] Target `moonshotai/Kimi-K2.6` downloading on GPU node to /mnt/nvme/models/kimi-k26-fp8/
- [ ] Background S3 mirror waiting for target completion, then syncs both to s3://kimi-k2-bench-models-.../models/

## Session 2026-05-13 19:00 UTC — Resumption

User scope: complete all phases. After discussion, pruned to Phase 0, 1b, 4, 5 (dropping Phase 2 vLLM EAGLE3 blocked by L13 and Phase 3 dynamic MLA — low expected delta, no source builds).

### Stage 1 — Foundation (re-verified)
- [x] Subnet + SG + IAM from prior session still exist
- [x] HF token uploaded to SSM SecureString `/kimi-k26/hf-token` (then skipped in favor of SCP — L14 below)
- [x] IAM role `kimi-k26-gpu` extended with SSM read + KMS decrypt
- [x] Phase 5 script `serve-vllm-phase5.sh` + config written
- [x] Phase 1b sweep script `run-phase1b-sweep.sh` written (24-ish configs, pruned `num_steps <= num_draft_tokens`)
- [x] Phase 4 `run-phase4-fullstack.sh` written (reads `WINNER.env` from Phase 1b)
- [x] Phase 0 roofline analysis math done (`results/phase-0-roofline/analysis.md`)
- [x] Benchmark runner `bench-one.py` written (sglang + vllm endpoints)

### Stage 4 — GPU spot node (new)
- [x] Launched p6-b300.48xlarge spot `i-0625a28290679d2ec` @ 18.234.156.117 (spot price $25.65/hr in us-east-1c)
- [x] NVMe RAID0 28 TB mounted at /mnt/nvme
- [x] 8× B300 SXM6 detected
- [x] SGLang + PyTorch images pulled
- [x] Bootstrap hit ECR connect timeout — not a blocker; used public images only

### Stage 2 — Weight staging (2nd attempt, live now)
- [x] HF CLI + hf_transfer installed for ec2-user
- [x] Draft weights `lightseekorg/kimi-k2.6-eagle3` (6 GB) — **READY**
- [ ] Target weights `moonshotai/Kimi-K2.6` — **~13% at T+8 min, ETA ~45 min more**

### Stage 4a — Phase 0 GPU health (launched in parallel)
- [ ] NCCL all_reduce / all_gather 8-GPU running in background
- [ ] Topology captured

### Instance registry
```
GPU (spot): i-0625a28290679d2ec pub=18.234.156.117 priv=10.192.13.9 subnet=subnet-079555bada92f1c9b sg=sg-02c4dbf438ee20741
```

### L14 — VPC endpoint interference with ECR/SSM
**Severity**: MEDIUM · **Category**: networking
The public subnet `subnet-079555bada92f1c9b` route table or VPC endpoint config blocks outbound to `api.ecr.us-east-1.amazonaws.com` and `ssm.us-east-1.amazonaws.com` (both curl time out after 5s, 000 response code). S3 and HuggingFace work fine. This caused the initial bootstrap script to bail at ECR login before weight staging started.
**Workaround**: Used `scp` to place HF token, and skipped ECR (public images only). SSM route still works from the controller's local machine, so the token could have been pushed to SSM from here and then pulled — but we already have it locally, so SCP is simpler.
**Rule**: When routing through this VPC subnet, don't assume AWS regional endpoints are reachable. Test `curl -m 5 https://api.ecr.<region>.amazonaws.com/` in userdata; fall back to public-image paths if it fails.
