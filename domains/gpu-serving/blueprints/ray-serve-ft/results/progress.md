---
blueprint: "ray-serve-ft"
domain: "gpu-serving"
spec: "domains/gpu-serving/specs/ray-serve-ft.md"
status: "complete"
last_updated: "2026-03-21T15:12:40Z"
last_stage: "t5"

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
    name: "Replica Crash Recovery"
    status: "complete"
  - id: "t2"
    name: "Worker Node Drain"
    status: "complete"
  - id: "t3"
    name: "Head Node Failure (GCS FT)"
    status: "complete"
  - id: "t4"
    name: "Head Node Failure WITHOUT GCS FT (control)"
    status: "not_started"
  - id: "t5"
    name: "HTTP Proxy Failover"
    status: "complete"
  - id: "t6"
    name: "ElastiCache Connectivity Disruption"
    status: "not_started"
  - id: "phase-0"
    name: "Infrastructure (2 hrs)"
    status: "not_started"
  - id: "phase-1"
    name: "Baseline Deployment (30 min)"
    status: "not_started"
  - id: "phase-2"
    name: "Fault Injection (2-3 hrs)"
    status: "not_started"
  - id: "phase-3"
    name: "Analysis (1 hr)"
    status: "not_started"

artifacts:
  lessons: true
  readiness_audit: []
  deployment_log: []
  compound: []
  benchmark_report: false
---

# Progress: ray-serve-ft

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
| t1 | Replica Crash Recovery | DONE |
| t2 | Worker Node Drain | DONE |
| t3 | Head Node Failure (GCS FT) | DONE |
| t4 | Head Node Failure WITHOUT GCS FT (control) | -- |
| t5 | HTTP Proxy Failover | DONE |
| t6 | ElastiCache Connectivity Disruption | -- |
| phase-0 | Infrastructure (2 hrs) | -- |
| phase-1 | Baseline Deployment (30 min) | -- |
| phase-2 | Fault Injection (2-3 hrs) | -- |
| phase-3 | Analysis (1 hr) | -- |

## Artifacts

| Artifact | Present |
|----------|---------|
| lessons.md | true |
| readiness audits | (none) |
| deployment logs | (none) |
| compound summaries | (none) |
| benchmark report | false |
