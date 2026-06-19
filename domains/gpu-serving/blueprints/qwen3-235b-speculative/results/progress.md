---
blueprint: "qwen3-235b-speculative"
domain: "gpu-serving"
spec: "domains/gpu-serving/specs/qwen3-235b-speculative.md"
status: "pending"
last_updated: "2026-05-14T00:00:00Z"
last_stage: "stage-0"
region: "us-west-2"
az_id: "usw2-az2"
subnet_id: "(to provision)"
vpc_id: "(to provision)"
model_bucket: "(to provision)"
results_bucket: "(to provision)"
ami_id: "(look up for us-west-2)"

stages:
  - id: "stage-0"
    name: "Deployment card lookup + blueprint review"
    status: "pending"
  - id: "stage-1"
    name: "Foundation (subnet routing, IAM, SG, SSM HF token)"
    status: "pending"
  - id: "stage-3"
    name: "Storage and model staging (Qwen3-235B FP8 + EAGLE3 draft to S3+NVMe)"
    status: "pending"
  - id: "stage-4"
    name: "GPU spot node launch (p6-b300 in usw2-az2)"
    status: "pending"
  - id: "stage-4a"
    name: "GPU health + NCCL validation"
    status: "pending"
  - id: "stage-4b"
    name: "Observability stack (Prometheus + DCGM + node-exporter)"
    status: "pending"
  - id: "stage-5"
    name: "Serving stack deployment (SGLang EAGLE3)"
    status: "pending"
  - id: "stage-6"
    name: "Benchmarks — Phases 0, 1, 1b, 4, 5a/b/c/d"
    status: "pending"
  - id: "stage-7"
    name: "Readiness audit + vault sync"
    status: "pending"
  - id: "stage-8"
    name: "Compound"
    status: "pending"

phases:
  phase-0-roofline: {status: "pending", artifact: null}
  phase-1-defaults: {status: "pending", artifact: null}
  phase-1b-eagle3-sweep: {status: "pending", artifact: null, note: "13-config num_steps×num_draft sweep"}
  phase-4-fullstack: {status: "pending", artifact: null, note: "winner + HiCache 200 GB/rank"}
  phase-5a-default-stack: {status: "pending", artifact: null}
  phase-5b-no-cuda-graph: {status: "pending", artifact: null, note: "L19 generalization check"}
  phase-5c-tp2-dp2: {status: "pending", artifact: null, note: "TP2+DP2 on 4-GPU (smaller model, TP4 comfortable)"}
  phase-5d-fp4-probe: {status: "pending", artifact: null, note: "profile-only (cutlass 3.x not shipped)"}

baseline_reference:
  blueprint: "domains/gpu-serving/blueprints/qwen3-235b-b300"
  vllm_tp4_peak_tok_per_s: 11820
  vllm_tp2dp4ep_peak_tok_per_s: 13877
  vllm_tp4_c1_per_req: 102.8

artifacts:
  lessons: false
  readiness_audit: []
  deployment_log: []
  compound: []
  benchmark_report: false
---

# Progress: qwen3-235b-speculative

## Session: (not started)

Scaffolded 2026-05-14 as a mirror of `kimi-k2.6-speculative` methodology. Baseline `qwen3-235b-b300` already complete; this session adds the SGLang + EAGLE3 + HiCache + TP2+DP2 optimization tiers.

### Planned runbook

See `../README.md` §Runbook for the full sequence.

### Observability mandate

Per `.claude/steering/tech-stack.md`, Stage 4b must run `bootstrap-observability.sh` + pass `observability-smoke-test.sh` before Stage 5. No Kimi-style TTFT loss on this session.

### Instance registry

```
GPU (spot): (not yet launched)
```
