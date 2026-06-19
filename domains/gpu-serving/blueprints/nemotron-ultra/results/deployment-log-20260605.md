# Nemotron-3-Ultra-550B-A55B-NVFP4 — Deployment Log 2026-06-05

Target: p6-b300.48xlarge spot, us-west-2b (usw2-az2), vLLM v0.22.0-cu130, TP4 single
replica. Scope: deploy up to and INCLUDING the Stage 5 / P0 smoke gate, then STOP and
report. P1-P4 benchmark phases are explicitly deferred.

Account 615299764834, user aiops. All AWS calls pass `--region us-west-2` (CLI default is
us-east-1).

---

## Stage 0 — Pre-deployment gate

### 0a. Blueprint review
- Blueprint scaffolded fresh from spec; sibling `nemotron-super/` used as structural
  template. blueprint-reviewer to be run against the populated directory before apply.

### 0b. Deployment card lookup
- `mdc get nemotron-3-ultra --engine vllm` -> **No card found**.
- `mdc sync` -> ran clean; `mdc list | grep nemotron` shows only `nemotron-49b` (different
  dense_nas model, NOT this 550B MoE). No card for nemotron-3-ultra.
- `mdc prs nemotron-3-ultra` -> no watch_prs/search_terms defined; nothing to track.
- `gpu-infra card b300` -> **No card found**. Available: g7e, p5e, p6-b200, aicr.
  `p6-b200` is the closest reference (same AL2023/nodeadm/Fabric-Manager/NVSwitch facts;
  its `arch=sm_120` is imprecise — B200=sm_100, B300=sm_103).
- DECISION: proceed from the HF model card transcribed verbatim in the spec (per the
  deployment instruction's fallback). Recorded card gaps in lessons.md.
- Pulled HF `config.json` (no weight download): `model_type=nemotron_h`,
  `architectures=[NemotronHForCausalLM]`, `max_position_embeddings=262144`,
  `moe_intermediate_size=5120`, `moe_latent_size` present (LatentMoE), `num_nextn_predict_layers`
  + `mtp_layers_block_type` present (MTP heads), `quantization_config` mixed FP8/NVFP4.
  -> 5120/TP4 = 1280, 1280 % 128 == 0 (divisibility-safe even under the FP8-block rule).

### 0c. Serving-config gate (fail-closed)
- Wrote `benchmark.yaml` sidecar modeling the smoke config (TP4, NVFP4, mamba-hybrid, MTP,
  B300, AL2023).
- `validate-serving-config.py --sidecar ... --corpus-root .` -> **EXIT 0** (PASS, no FAIL).
  - WARN `mamba-mtp-prefix-cache`: MTP + `--enable-prefix-caching` on mamba-hybrid may need
    `--no-enable-prefix-caching`. Recorded as #1 predicted failure mode + fallback in
    lessons.md. First attempt keeps verbatim card config.
  - INFO `specdec-acceptance-gate`: MTP acceptance not yet measured (smoke item #5).
  - No `prior-failure:*` findings surfaced from the lessons corpus for this config.
- NVFP4 caveat: resolver's fp8-moe-tp-divisibility rule does NOT fire for fp4 quant —
  recorded in lessons.md; vLLM load-time error is the authoritative NVFP4-layout check.

**Stage 0 validation: PASS.** Cards absent (documented); resolver exit 0; conflicts flagged.

---

## Stage 1 — Foundation (reuse existing infra)

Inventory of reusable us-west-2 infra (verified, not assumed):
- EKS cluster `qn-sglang-eks-cluster`: ACTIVE, v1.32, vpc-0bd6abcecded8edf6,
  clusterSG sg-070da338e3796648d, endpoint reachable.
- Node group `ai-infra-b300-spot`: ACTIVE, p6-b300.48xlarge, AL2023_x86_64_NVIDIA, SPOT,
  scaling min0/max1/desired0, disk 500GB, subnet subnet-001db6882dbb5ac72
  (us-west-2b = usw2-az2 — EXACT target AZ), taint `ai-infra/b300=true:NoSchedule`,
  label `ai-infra/role=b300-spot`. Access entry `ai-infra-b300-node` already present.
- S3: `qn-sglang-models-20260303161715850900000007` EXISTS (the short name in the
  instruction resolves to this). Will create a dedicated `nemotron-ultra-models` prefix or
  reuse this bucket for ~335 GB NVFP4 staging.
- No B300 instance currently running (desired=0) — $0/hr burn at start.

DECISION: reuse cluster + node group + AMI wholesale. No new Terraform foundation needed;
the blueprint Terraform will (like nemotron-super) attach to the existing cluster via data
sources and add only the K8s serving/staging objects.

### Blueprint scaffolding + Stage 0a review (post-scaffold)
- Scaffolded from spec using nemotron-super as template. Files: main.tf, variables.tf,
  outputs.tf, nemotron-ultra-b300.tfvars, benchmark.yaml, README.md, lessons.md,
  configs/baseline.sh, scripts/{stage-model,scale-node,smoke-test}.sh,
  results/{progress.md, deployment-log-20260605.md}.
- Terraform: `fmt -check` CLEAN, `init` OK (aws 5.x + kubernetes 2.38), `validate` SUCCESS.
- Scripts: all pass `bash -n`.
- blueprint-reviewer-equivalent pre-deploy gate: all README file refs resolve; spec exists
  with `## Verification Criteria` + Stage 5 section; AGENTS.md routing entry present
  (line 65). P0 issues: NONE -> deployment NOT blocked.
  - P2 (non-blocking): project-structure.md tree lacks a nemotron-ultra entry; defer to
    compound step (cosmetic steering update).

**Stage 1 validation: PASS** (cluster reachable via kubectl get nodes — 2 system nodes
Ready; node group + AMI + AZ all reused/verified). No new foundation Terraform required.

---

## Stage 3 — Storage / model staging (PREPARED, not executed)
- HF repo `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` verified accessible WITHOUT a
  token (OpenMDW public): sha 02c7d9e6, 113 safetensors shards, 352.4 GB total.
- S3 prefix `s3://qn-sglang-models-20260303161715850900000007/nemotron-3-ultra-nvfp4/`
  currently EMPTY — no prior staging.
- Fixed `scripts/stage-model.sh` for huggingface_hub v1.x (dropped removed
  `local_dir_use_symlinks` arg). Control host (macOS, 58 GB free, no /mnt/nvme) cannot
  stage 352 GB — staging must run on the B300 node's NVMe.
- NOT executed: requires the GPU node (or a build host with bandwidth + 400GB scratch).

## Stage 4 / 4a — GPU node + health (BLOCKED — stopped before burn)
- Plan: scale `ai-infra-b300-spot` 0->1 (usw2-az2), stage to NVMe, run gpu-infra MCP
  pre-flight, deploy vLLM, smoke test.
- BLOCKER: Stage 4a GPU pre-flight is NOT executable here:
  - `~/.ssh/gpu-cluster-key` (gpu-infra MCP's configured key) absent; no *.pem present.
  - `ai-infra-b300-spot` has remoteAccess=null, launchTemplate=null -> no SSH path to new
    nodes; SSM posture of the NG unverified (only 2 system nodes are SSM Online).
  - gpu-infra MCP live-diagnostic tools not in the current callable tool set.
- DECISION (cost discipline): did NOT scale up. Node group remains desiredSize=0, no B300
  instances running -> **$0/hr burn**. All blueprint files checkpointed to disk.

**Stage 4a validation: BLOCKED.** See lessons.md "blocker: Stage 4a GPU pre-flight" for
the unblock recipe.

## Cost summary
- Total spend this session: **$0** (no GPU instance launched).
- Current burn: **$0/hr** (ai-infra-b300-spot at desiredSize=0).

---

## SESSION 2 (resume) — Stage 3 → Stage 5 gate

Re-validation on resume:
- `aws sts get-caller-identity`: account 615299764834, user aiops. CLI default region us-east-1 (pass --region us-west-2 everywhere).
- `terraform validate`: SUCCESS; `terraform fmt -check`: clean.
- EKS `qn-sglang-eks-cluster`: ACTIVE, v1.32. `kubectl get nodes`: 2x m6i.xlarge system nodes Ready.
- NG `ai-infra-b300-spot`: ACTIVE, p6-b300.48xlarge SPOT, AL2023_x86_64_NVIDIA, min0/max1/desired0, subnet-001db6882dbb5ac72 (us-west-2b), nodeRole ai-infra-b300-node.
- `mdc get nemotron-3-ultra --engine vllm`: still No card. `mdc prs`: nothing to track. (cards remain absent — documented in lessons.)
- Stage 0c resolver re-run: EXIT 0. WARN mamba-mtp-prefix-cache (fallback ready), INFO specdec-acceptance-gate. No prior-failure findings.

### PRIOR BLOCKER RESOLVED — SSM, not SSH
`aws iam list-attached-role-policies --role-name ai-infra-b300-node` ->
`AmazonSSMManagedInstanceCore` present. Stage 4a will run via kubectl exec into the vLLM
pod (and SSM start-session as fallback). No SSH key needed; gpu-infra MCP SSH tooling out
of scope. Logged in lessons.md.

### Stage 3 — Storage / model staging (EXECUTING, no GPU burn)
- HF repo re-verified: sha 02c7d9e6, 113 safetensors shards, 352.3 GB shard bytes /
  352.4 GB total / 243 files. Public (OpenMDW, no token). Largest shard 8.4 GB.
- System nodes (m6i.xlarge, 47 GB ephemeral) and control host (58 GB free) both too small
  to hold 352 GB at once -> stream one file at a time (download/upload/delete) from the
  control host directly to S3. Peak local disk < 10 GB. $0 GPU burn.
- Running `/tmp/stage-nemotron-stream.py` -> s3://qn-sglang-models-...7/nemotron-3-ultra-nvfp4/.
  Idempotent (skips files already in S3 with matching size). In progress.

### Stage 3 staging — bandwidth note
- In-cluster Job rejected as the staging path: the system-node role
  (`system-eks-node-group-...24`) is S3 read-only on the staging bucket
  (`ray-video-s3-write` only covers `ray-video-poc-intermediate`). The b300 node role
  (`ai-infra-b300-node`) likewise has only read policies (`kimi-models-read`,
  `vllm-model-cache-read`). So in-cluster pods cannot PutObject to
  `qn-sglang-models-...7`. Staging runs from the control host (user `aiops`, full S3).
- Throughput ceiling is the control host uplink to S3 (~25-40 MB/s), NOT the HF download
  leg (hf_transfer made no difference). ~352 GB => ~2.5-3 hr. Peak local scratch < 5 GB
  (one file at a time). $0 GPU burn — node still desired=0.
- Streaming proceeding; idempotent (resumes on S3 size match). Will verify 113 shards +
  total bytes in S3 BEFORE scaling the node.
