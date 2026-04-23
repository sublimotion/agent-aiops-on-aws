---
blueprint: "qwen3-next-custbench"
domain: "gpu-serving"
spec: "domains/gpu-serving/specs/qwen3-next-custbench.md"
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
  - id: "t1"
    name: "Customer Reproduction (Config A)"
    status: "not_started"
  - id: "t2"
    name: "Optimized Head-to-Head (Config B vs A)"
    status: "not_started"
  - id: "t3"
    name: "MTP Isolation (Config C vs B)"
    status: "not_started"
  - id: "t4"
    name: "Load Scaling (Config B)"
    status: "not_started"
  - id: "t5"
    name: "Simulated Memory-Constrained KV Cache Offloading"
    status: "not_started"
  - id: "t6"
    name: "2x Replica + CPU Offload (Config E)"
    status: "not_started"
  - id: "t7"
    name: "Stress Test at 1500 Concurrent (Config E)"
    status: "not_started"

artifacts:
  lessons: true
  readiness_audit: []
  deployment_log: []
  compound: []
  benchmark_report: false
---

# Progress: qwen3-next-custbench

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

## Spec Phases

| Phase | Name | Status |
|-------|------|--------|
| t1 | Customer Reproduction (Config A) | -- |
| t2 | Optimized Head-to-Head (Config B vs A) | -- |
| t3 | MTP Isolation (Config C vs B) | -- |
| t4 | Load Scaling (Config B) | -- |
| t5 | Simulated Memory-Constrained KV Cache Offloading | -- |
| t6 | 2x Replica + CPU Offload (Config E) | -- |
| t7 | Stress Test at 1500 Concurrent (Config E) | -- |

## Artifacts

| Artifact | Present |
|----------|---------|
| lessons.md | true |
| readiness audits | (none) |
| deployment logs | (none) |
| compound summaries | (none) |
| benchmark report | false |
