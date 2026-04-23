# [Blueprint Name] Requirements

## Status: DRAFT | IN_PROGRESS | DEPLOYED | COMPLETED

## Overview
Brief description of what this deployment does.

## Components

### 1. Compute
- **Platform**: EKS / SageMaker Endpoints / Lambda / etc.
- **Instance Types**: (specify with fallbacks for GPU)
- **Scaling**: Min/Max/Desired

### 1a. GPU & NCCL Pre-Flight (required for multi-GPU)
Before any multi-GPU workload, run these diagnostics and record results:

1. **GPU inventory**: `nvidia-smi` — driver version, CUDA version, GPU names, memory, ECC status
2. **Topology**: `nvidia-smi topo -m` — verify NVLink vs PCIe-only, identify GPU groupings
3. **PCIe link**: `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current` — verify expected gen/width under load
4. **ECC/Xid errors**: `nvidia-smi --query-gpu=ecc.errors.*` + `dmesg | grep "NVRM: Xid"` — zero tolerance for uncorrected errors
5. **NCCL collective test**: Run `nccl_diag.py` (all_reduce, broadcast, barrier across all GPUs) — must pass before deploying distributed training or tensor-parallel serving
6. **NCCL transport**: Check `NCCL_DEBUG=INFO` output for P2P support, transport type (NVLink/SHM/NET), channel count

**Known blockers** (see `devstral-sera/lessons.md`):
- NCCL ≤ 2.25.1 has shared memory bug on Blackwell (sm_120) PCIe-only topology — upgrade to ≥ 2.26.2
- g7e instances (RTX PRO 6000 Blackwell) are PCIe-only, no NVLink — affects multi-GPU training
- vLLM inference unaffected (uses custom allreduce, not NCCL)

### 2. Model
- **Model ID**: HuggingFace model ID or path
- **Format**: safetensors / GGUF / etc.
- **Serving**: vLLM / TGI / Triton / etc.
- **Required Args**: Any model-specific arguments
- **Deployment Card**: Run `mdc get <model> --engine <engine>` before deploying. If no card exists, run `mdc sync` and create one from upstream docs.

### 3. Networking
- **VPC**: CIDR, AZs
- **Access**: Public / Private / VPN
- **Endpoints**: Required VPC endpoints

### 4. Storage
- **Model Storage**: S3 / EFS / Local
- **Caching**: PVC size if needed

### 5. Development Environment
- **IDE**: SageMaker Studio / Cloud9 / None
- **Connectivity**: To compute cluster

## Non-Requirements
List what's explicitly out of scope:
- Multi-region?
- HA/DR?
- Production monitoring?

## Security Requirements
- Encryption at rest
- Network isolation
- IAM/RBAC

## Cost Considerations
Rough estimates or cost-saving recommendations.

## Known Limitations
Known issues or constraints to be aware of before deployment.

Check `mdc prs <model>` for recently merged upstream PRs that may affect this deployment.

## Verification Criteria

Concrete, mechanically checkable conditions for each stage. The deployer agent checks each criterion and records pass/fail in the readiness audit. A criterion is either deterministic (command returns expected output) or metric-bounded (value within threshold).

### Stage 4a — GPU Health
- [ ] All GPUs report ECC enabled, 0 uncorrectable errors: `nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader` returns all zeros
- [ ] No pending row remaps: `nvidia-smi --query-gpu=retired_pages.pending --format=csv,noheader` returns `No`
- [ ] GPU thermals < 85°C under idle: `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader`
- [ ] NCCL all-reduce bandwidth > _____ GB/s for TP=_____ (fill in from deployment card)
- [ ] No Xid errors in dmesg: `dmesg | grep "NVRM: Xid"` returns empty

### Stage 5 — Serving Stack
- [ ] Health endpoint responds: `curl -s -o /dev/null -w '%{http_code}' localhost:8000/health` returns `200`
- [ ] Test completion succeeds: single request to `/v1/completions` returns valid output
- [ ] Model loads without OOM: container logs show no `CUDA out of memory` errors
- [ ] Startup time < _____ minutes (fill in; account for DeepGEMM JIT if applicable)

### Stage 6 — Benchmark
- [ ] TTFT P50 < _____ ms at concurrency _____
- [ ] Throughput > _____ tok/s at batch size _____
- [ ] No OOM at max concurrent requests = _____
- [ ] No request timeouts during benchmark run

### Stage 7 — Readiness Audit
- [ ] All readiness audit categories pass
- [ ] No unresolved lessons with severity >= HIGH in `lessons.md`
- [ ] All verification criteria above are checked and recorded
- [ ] Deployment card recommendations are either followed or explicitly overridden with justification

> **How to fill in thresholds**: Run `mdc get <model> --engine <engine>` and use the card's recommended performance targets. If no card exists, leave blank and establish baselines from the first deployment.

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes,
> design evaluations) belong in the blueprint directory, not in this spec.
> See `blueprints/<name>/lessons.md`, `blueprints/<name>/results/`, etc.
