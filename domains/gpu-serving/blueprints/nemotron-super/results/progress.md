---
blueprint: "nemotron-super"
domain: "gpu-serving"
spec: "domains/gpu-serving/specs/nemotron-super.md"
status: "not_started"
last_updated: "2026-03-21T15:11:17Z"
last_stage: ""

stages:
  - id: "stage-0"
    name: "Deployment card lookup"
    status: "not_started"
  - id: "stage-1"
    name: "Foundation"
    status: "not_started"
  - id: "stage-2"
    name: "Build machine"
    status: "not_started"
  - id: "stage-3"
    name: "Storage and model staging"
    status: "not_started"
  - id: "stage-4"
    name: "Capacity reservation and GPU node"
    status: "not_started"
  - id: "stage-4a"
    name: "GPU health validation"
    status: "not_started"
  - id: "stage-5"
    name: "Serving stack deployment"
    status: "not_started"
  - id: "stage-6"
    name: "Pre-benchmark validation"
    status: "not_started"
  - id: "stage-7"
    name: "Readiness audit"
    status: "not_started"
  - id: "stage-8"
    name: "Compound"
    status: "not_started"

phases:

artifacts:
  lessons: true
  readiness_audit: []
  deployment_log: []
  compound: []
  benchmark_report: false
---

# Progress: nemotron-super

## Deployer Stages

| Stage | Name | Status |
|-------|------|--------|
| stage-0 | Deployment card lookup | -- |
| stage-1 | Foundation | -- |
| stage-2 | Build machine | -- |
| stage-3 | Storage and model staging | -- |
| stage-4 | Capacity reservation and GPU node | -- |
| stage-4a | GPU health validation | -- |
| stage-5 | Serving stack deployment | -- |
| stage-6 | Pre-benchmark validation | -- |
| stage-7 | Readiness audit | -- |
| stage-8 | Compound | -- |

## Artifacts

| Artifact | Present |
|----------|---------|
| lessons.md | true |
| readiness audits | (none) |
| deployment logs | (none) |
| compound summaries | (none) |
| benchmark report | false |

## fin-rag Experiment Axes

| Axis | Status | Result |
|------|--------|--------|
| P0 precision (FP8 vs BF16) | done | both deployed |
| P1 chunked-prefill sweep | done | mnbt=16384 winner |
| P1 spec-decode (MTP + n-gram) | **CLOSED** | NOT deployable on vLLM 0.18.1 TP2. MTP blocked at TP2; n-gram crashes in Mamba-2 CUDA-graph capture, only starts with `--enforce-eager`. Operator decision: did NOT measure eager acceptance — eager latency non-representative, crashed graph-capture path is required in production, so eager number is not actionable. Verdict stands. |
| P1 FP8 KV on/off | pending | on graph-captured winner |
| P1 TRITON_ATTN vs FlashInfer | confirmed (split) | Two independent subsystems: attention runs `AttentionBackendEnum.TRITON_ATTN` (honors `--attention-backend`); FP8 MoE GEMM auto-picks `FLASHINFER_TRTLLM` (out of AITER/FLASHINFER_CUTLASS/DEEPGEMM/TRITON/MARLIN). Linear kernel = `FlashInferFP8ScaledMMLinearKernel`. So "FlashInfer vs Triton" applies to MoE backend, not attention. Confirmed from engine init log v0.18.1. |
| P2 parallelism (tp4-x2 vs tp2-x4) | pending | TP4 safe (per-tensor FP8) |
| P2 disagg 4p4d/2p6d | pending | time-permitting only |
| TP1 prefix-cache leg | pending | does prefix_cache_hits_total go >0 at TP1 (vs 0 at TP2)? |

## Current Serving State

Reverted from the eager n-gram leg to the graph-captured FP8 winner on 2026-06-11:
`fin-rag-vllm-fp8` (ml-inference): agg-tp2-x4, mnbt=16384, kv-cache-dtype fp8, attention-backend TRITON_ATTN, NO spec-decode, NO enforce-eager. Rolling restart in progress (CUDA-graph capture ~3 min cold start).
