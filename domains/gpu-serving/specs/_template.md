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

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes,
> design evaluations) belong in the blueprint directory, not in this spec.
> See `blueprints/<name>/lessons.md`, `blueprints/<name>/results/`, etc.
