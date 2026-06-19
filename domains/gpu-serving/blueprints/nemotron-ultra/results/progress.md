---
# Progress Schema v1
blueprint: "nemotron-ultra"
domain: "gpu-serving"
spec: "domains/gpu-serving/specs/nemotron-ultra.md"
status: "complete"
last_updated: "2026-06-06T00:00:00Z"
last_stage: "stage-6"

stages:
  - id: "stage-0"
    name: "Pre-deployment gate (cards + serving-config resolver)"
    status: "complete"
    notes: "No mdc/gpu-infra card; proceeded from HF card in spec. Stage 0c resolver exit 0 (2 WARN/INFO, no FAIL)."
  - id: "stage-1"
    name: "Foundation"
    status: "complete"
    notes: "Reused qn-sglang-eks-cluster + ai-infra-b300-spot NG. kubectl get nodes OK. TF init/validate/fmt clean."
  - id: "stage-2"
    name: "Build machine"
    status: "skipped"
    notes: "No custom image. PIVOT: vllm/vllm-openai:v0.22.0-cu130 does NOT exist (cu130 caps at v0.20.0; v0.22.1 is cu129-only). B300 smoke uses lmsysorg/sglang:v0.5.12.post1-cu130 (pulled to node, 12.1GB). See lessons.md."
  - id: "stage-3"
    name: "Storage and model staging"
    status: "complete"
    notes: "PIVOT to direct HF->NVMe on node (S3 sync was 7/113 after ~40min). snapshot_download to /mnt/nvme/models/nemotron-3-ultra-nvfp4: 329GB/113 shards + config/tokenizer/chat_template/reasoning_parser, 0 incomplete, ~2.5min. NVMe = 8x3.5TB instance-store RAID0 (28TB ext4)."
  - id: "stage-4"
    name: "Capacity reservation and GPU node"
    status: "complete"
    notes: "Scaled ai-infra-b300-spot to desiredSize=1. i-0a2877d7deda0e556 (10.2.21.188) Ready, 8 GPUs advertised after nvidia.com/gpu.present label. ~$27/hr burn active. EBS root bumped to 1000MB/s/6000IOPS."
  - id: "stage-4a"
    name: "GPU health validation"
    status: "complete"
    notes: "PASS via SSM (node role has AmazonSSMManagedInstanceCore). 8x B300 SXM6 275GB, temps 27-31C, 0 ECC uncorrected, 0 Xid, 0 row remaps. Full NVLink mesh NV18 (144 links), fabric-manager active. Driver 580.159.03. NCCL bandwidth test still pending (will run inside SGLang container)."
  - id: "stage-5"
    name: "Serving stack deployment (P0 smoke gate)"
    status: "complete"
    notes: "GATE GREEN (STOP POINT). SGLang TP4 + flashinfer_trtllm MoE + EAGLE MTP, no EP, no radix cache. All 6 smoke items pass: health 200, model registered, completion=SMOKE_OK, reasoning toggle (nemotron_3) works, tool call (qwen3_coder) works, MTP accept len 1.8-4.47/accept rate 0.16-0.69, warm single-stream 188-244 tok/s (peak 243.7). Config corrections vs card: kv-cache-dtype fp8_e4m3 (not fp8), drop --ep-size, flashinfer_trtllm MoE (not triton), --disable-radix-cache, no_buffer. See lessons.md."
  - id: "stage-6"
    name: "Pre-benchmark validation + P1-P4 benchmark suite (TWO RUNS)"
    status: "complete"
    notes: "RUN 1 (SGLang B300 TP8): single-stream 177 tok/s decode, accept_len 2.4, agg ~1040 — concluded 300 tok/s unreachable. RUN 2 (vLLM v0.22.0 B200 TP4, AUTHORITATIVE) OVERTURNS that: SGLang bug #21138 broke NemotronH MTP. vLLM native nemotron_h_mtp k=5 gets accept_len 3.54, single-stream decode median 297.7 tok/s (0.99x, 6/12 prompts clear 300), wall 267.6. Peak agg 1883 @ c=256 (1.8x SGLang), sustained 1847, 0 errors. P1 every workload 2.4-4.9x SGLang. P4 64k/128k/256k = 468-514 tok/s decode (3.7-4.3x). Cost ~$6.00/M output @ $40.57/hr B200 spot vs DeepInfra $2.50/M (2.4x). Engine choice dominates hardware. B200 node torn down after run."
  - id: "stage-7"
    name: "Readiness audit"
    status: "not_started"
  - id: "stage-8"
    name: "Compound"
    status: "not_started"

phases:
  - id: "P0"
    name: "Smoke + tool-call + reasoning toggle + MTP gate"
    status: "complete"
    notes: "PASSED 2026-06-05. Single-stream warm decode 188-244 tok/s vs 300 tok/s target — close pre-tuning. P1-P4 NOT started (deferred per instruction)."
  - id: "P1"
    name: "Standard workload suite"
    status: "complete"
    notes: "Ran chatbot-short/long, rag-long-context, coding-agent, sharegpt-mix at real token shapes on TP8 wide-tree. Results in results/standard/."
  - id: "P2"
    name: "Concurrency sweep + ablations"
    status: "complete"
    notes: "TP4+MTP sweep c=1..256 (agg saturates ~1040 tok/s @ c>=64). Single-stream ablation across TP4/TP8/TP8-wide-tree. accept_len flat ~1.9-2.4 — acceptance is the gate."
  - id: "P3"
    name: "Speed + cost analysis vs DeepInfra"
    status: "complete"
    notes: "SPEED: not beaten (best 130 tok/s e2e vs 300). COST: $7.21/M output @ 1040 tok/s agg vs DeepInfra $2.50/M — DeepInfra ~2.9x cheaper at retail; break-even needs ~3000 tok/s agg."
  - id: "P4"
    name: "1M long-context (B300)"
    status: "complete"
    notes: "1M NOT POSSIBLE — model max_position_embeddings=262144 (256k). Feasible 64k/128k/256k all stable ~119-131 tok/s decode, no OOM. 512k/1m tiers excluded."

artifacts:
  lessons: true
  readiness_audit: []
  deployment_log: ["20260605"]
  compound: []
  benchmark_report: true
---

# Nemotron-3-Ultra Progress

Status: **BLOCKED at Stage 4a** (GPU pre-flight not executable from this environment).
$0 spent — node group never scaled up. All scaffolding + gates checkpointed to disk.

| Stage | Status | Notes |
|-------|--------|-------|
| 0 — Pre-deploy gate | complete | No cards exist; HF card verbatim. Resolver exit 0 (no FAIL). |
| 1 — Foundation | complete | Reuse qn-sglang-eks-cluster + ai-infra-b300-spot NG (usw2-az2). |
| 2 — Build machine | skipped | Stock vLLM image. |
| 3 — Storage / staging | blocked | Tooling ready; HF 113 shards/352GB verified; needs node NVMe. |
| 4 — GPU node | blocked | Not scaled up — cost discipline. $0/hr. |
| 4a — GPU health | blocked | No SSH key / NG remoteAccess=null / MCP tools absent. |
| 5 — Serving smoke | not_started | **STOP POINT** — gated behind 4a unblock. |
| 6-8 | not_started | DEFERRED. |

## Unblock recipe
See `lessons.md` "blocker: Stage 4a GPU pre-flight". TL;DR: give the b300 node group an SSH
key/remote-access path (or confirm SSM + instance profile), place
`~/.ssh/gpu-cluster-key`, then `scale-node.sh 1` -> `stage-model.sh` -> Stage 4a ->
`terraform apply` -> `smoke-test.sh`.
