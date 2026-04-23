# Kubernetes GPU Scheduler Landscape (April 2026)

## Context

When running multiple LLM inference workloads on shared GPU clusters, the default K8s scheduling model — whole-GPU exclusive assignment via the NVIDIA device plugin — works for simple cases but breaks down at scale. This note maps the available schedulers, what problems they solve, and when to reach for each one.

## The Default: NVIDIA Device Plugin

The baseline on every managed K8s GPU platform (EKS, GKE, AKS, HyperPod).

- GPUs are exposed as integer resources (`nvidia.com/gpu: "2"`)
- Exclusive assignment — one pod per GPU, no sharing
- No topology awareness — GPUs treated as fungible
- No gang scheduling — pods scheduled independently

**When it works:** Small clusters, few models, plenty of spare GPUs. We used this for the HyperPod 3-model session (Mistral SM4 + Gemma 31B + Gemma E4B on one p5.48xlarge, 5 of 8 GPUs) and it was fine.

**When it breaks:** Multi-tenant clusters with GPU fragmentation, workloads that need topology-aware placement, models that don't fill a whole GPU, or multi-node TP jobs that need all-or-nothing scheduling.

---

## Schedulers by Category

### Inference-Specific

#### Grove (NVIDIA / ai-dynamo) — Alpha
- **Repo**: https://github.com/ai-dynamo/grove
- **Version**: v0.1.0-alpha.7 (March 2026)
- **License**: Apache 2.0

The only scheduler designed specifically for **multi-node disaggregated inference**. From the same NVIDIA Dynamo project as the disaggregated prefill/decode engine.

**Three-level hierarchy:**
```
PodCliqueSet (PCS)              ← system: canary, A/B, multi-AZ
  └─ PodCliqueScalingGroup (PCSG)  ← component: one complete model instance
       ├─ PodClique: prefill         ← role: prefill workers
       └─ PodClique: decode          ← role: decode workers
```

Each level scales independently — add decode workers without touching prefill, or add a whole new model replica for HA.

**Key features:**
- Hierarchical gang scheduling (nested pod groups with proportional scaling)
- Built-in startup ordering (workers before leaders)
- Rolling updates for multi-node serving units
- MNNVL ComputeDomain integration for automatic NVSwitch topology placement
- Multi-level autoscaling (system, component, role)

**Unique strength:** First-class support for disaggregated P/D inference. No other scheduler can express "deploy 1 prefill group + 3 decode groups, scale decode independently."

**Maturity:** Alpha. Moving fast but not production-ready yet.

---

### GPU Partitioning & Sharing

#### KAI Scheduler (NVIDIA, open-sourced from Run:ai)
- **Origin**: Scheduling engine from Run:ai (acquired by NVIDIA 2024), open-sourced March 2025

**Key features:**
- GPU fractions — request 0.5 GPU, two pods share with memory isolation
- Topology-aware placement within a node (NVSwitch/PCIe domain awareness)
- Bin packing — fill GPUs densely before spilling
- Gang scheduling (flat, not hierarchical)
- Fair-share queuing with borrowing/preemption

**Unique strength:** GPU fractions. The only scheduler that lets you pack multiple small models onto one GPU without MIG. For our Gemma E4B (15 GiB on an 80 GiB H100), this would save 65 GiB of wasted VRAM.

#### Run:ai (NVIDIA, commercial)
- Commercial platform built on top of KAI scheduler
- Adds UI/dashboard, workload management, oversubscription policies, fleet management
- Enterprise GPU orchestration — the "vCenter for GPUs" play

---

### Batch / Training Job Scheduling

#### Volcano (CNCF Incubating)
- **Origin**: Huawei, ~2019
- **Maturity**: Stable, large community

The original K8s batch scheduler for HPC/AI training.

**Key features:**
- Gang scheduling via PodGroup (flat all-or-nothing)
- Hierarchical queue management with fair-share, priority, preemption
- Native MPI, PyTorch, TensorFlow distributed job support
- NUMA/GPU affinity via plugins

**Strengths:** Battle-tested, CNCF backing, well-understood. Common in on-prem HPC-to-K8s migrations.
**Weaknesses:** Heavy footprint (CRDs, controllers, admission webhooks). Designed for batch jobs, not inference serving. No GPU fractions. Flat gang scheduling doesn't express multi-role hierarchies.

**Used by:** Baidu, Huawei, many Chinese cloud providers, on-prem HPC clusters.

#### Kueue (K8s SIG Scheduling)
- **Origin**: Google-led, K8s SIG project, started 2022
- **Maturity**: Stable

The upstream K8s approach to job queueing and GPU quota management.

**Key features:**
- ResourceFlavors — define GPU types (H100, A100, L4) with per-type quotas
- ClusterQueues / LocalQueues — multi-tenant quota management with borrowing
- Preemption and fair sharing
- Integration with Job, RayJob, PyTorchJob, MPIJob

**Strengths:** Upstream K8s project (will likely become the standard), lightweight, composable. Integrated into GKE Autopilot.
**Weaknesses:** Quota/admission layer only — doesn't replace the scheduler. No GPU fractions, no topology awareness, no gang scheduling.

#### Yunikorn (Apache Incubating)
- **Origin**: Cloudera
- Designed for mixed big-data + ML clusters (Spark + training)
- Hierarchical queue management, gang scheduling, bin packing
- Less GPU-specific than KAI, smaller community for pure GPU workloads

---

### Upstream K8s

#### Dynamic Resource Allocation (DRA)
- **Status**: Beta since K8s 1.31
- Built into K8s itself (ResourceClaims, ResourceSlices)
- Lets the scheduler understand GPU topology natively
- Will eventually replace the device plugin model
- Still maturing — not production-ready for complex GPU scheduling

---

## Comparison Matrix

| Feature | Device Plugin | Grove | KAI | Volcano | Kueue | Yunikorn | DRA |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPU fractions | - | - | **Yes** | - | - | - | - |
| Gang scheduling | - | **Hierarchical** | Flat | Flat | - | Flat | - |
| Topology aware | - | **NVSwitch/MNNVL** | Within-node | Basic | - | - | Native |
| Fair-share queues | - | - | **Yes** | **Yes** | **Yes** | **Yes** | - |
| Preemption | Basic | - | **Yes** | **Yes** | **Yes** | **Yes** | - |
| GPU bin packing | - | - | **Yes** | Basic | - | Basic | - |
| Disaggregated P/D | - | **First-class** | - | - | - | - | - |
| Multi-level autoscale | - | **Yes** | - | - | - | - | - |
| Rolling updates | - | **Multi-node** | - | - | - | - | - |
| Startup ordering | - | **Built-in** | - | Plugin | - | - | - |
| Maturity | Stable | Alpha | Early | Stable | Stable | Stable | Beta |
| Governance | NVIDIA | NVIDIA OSS | NVIDIA OSS | CNCF | K8s SIG | Apache | K8s |

## Decision Guide

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Small cluster, few models, plenty of GPUs | Default device plugin | Simple, no overhead |
| Multi-tenant GPU cluster, need quotas | Kueue | Upstream, lightweight, will be standard |
| Small models wasting VRAM on large GPUs | KAI | GPU fractions, bin packing |
| Large training jobs, on-prem HPC | Volcano | Battle-tested gang scheduling |
| Disaggregated prefill/decode inference | Grove | Only option with hierarchical P/D primitives |
| Mixed Spark + ML workloads | Yunikorn | Big data + ML hybrid queues |
| Multi-node TP with topology requirements | Grove + DRA | NVSwitch domain placement |

## The Convergence Path

The stack is fragmented today but converging:

```
2024-2025:  Each scheduler solves one layer independently
2026-2027:  Kueue (quotas) + DRA (devices) + Grove/KAI (placement) compose together
2028+:      DRA matures to handle topology natively, KAI fractions may go upstream
```

The likely production stack in 1-2 years:
- **Kueue** for multi-tenant quota management (who gets GPUs)
- **DRA** for device-level topology representation (what GPUs look like)
- **Grove** for inference workload composition (how serving units are structured)
- **KAI** for GPU partitioning within nodes (how GPUs are shared)

These are complementary layers, not competitors.

---

## Relevance to Our HyperPod Work

For the current blueprint pattern (single-node, 1-3 models, exclusive GPU assignment), the default device plugin is sufficient. But as we move toward:

- **Disaggregated inference** (Dynamo P/D split for 119B+ models) → Grove
- **Multi-model density** (packing 4B models onto shared GPUs) → KAI fractions
- **Shared cluster with multiple teams** → Kueue quotas
- **Multi-node TP across NVSwitch domains** → Grove + DRA

...these schedulers become necessary. Grove is the most interesting for our roadmap because it directly addresses the Dynamo disaggregated serving pattern we're evaluating in `llmd-hyperpod` and `dynamo-hyperpod` blueprints.

---

*Last updated: 2026-04-07. Based on field research during HyperPod 3-model benchmark session.*
