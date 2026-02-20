# Kimi K2.5 MoE Inference Blueprint

Deploys moonshotai/Kimi-K2.5 (1T parameter MoE model) on p5e.48xlarge with 8x H200 GPUs, FSx Lustre, and EKS.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  EKS Cluster (us-east-2)            │
│                                                     │
│  ┌───────────┐  ┌────────────────────────────────┐  │
│  │  System   │  │   p5e.48xlarge (Capacity Block) │  │
│  │  2x m6i   │  │   8x H200 (1.1TB HBM)          │  │
│  │           │  │   ┌──────────────────────┐      │  │
│  │           │  │   │ vLLM (Kimi K2.5)     │      │  │
│  │           │  │   │ TP=8, enforce-eager  │      │  │
│  │           │  │   └──────────────────────┘      │  │
│  └───────────┘  │   NVMe RAID0 (30TB)             │  │
│                 └────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  FSx Lustre (100 TiB SCRATCH_2)               │  │
│  │  Model storage + KV cache                     │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────┐                                      │
│  │Prometheus │  Port 30090                          │
│  └───────────┘                                      │
└─────────────────────────────────────────────────────┘
```

## Model Specifications

| Attribute | Value |
|-----------|-------|
| Architecture | Mixture-of-Experts (MoE) |
| Total Parameters | 1 Trillion |
| Activated Parameters | 32B per forward pass |
| Experts | 384 (8 active per token) |
| Context Length | 256K tokens |
| Model Size | ~540 GB (64 safetensor shards) |
| Attention | Multi-head Latent Attention (MLA) |
| Quantization | Native INT4 |

## Prerequisites

- AWS account with p5e.48xlarge capacity block access in us-east-2
- Terraform >= 1.0
- kubectl configured
- HuggingFace access to moonshotai/Kimi-K2.5

## Quick Start

```bash
# 1. Deploy infrastructure
cd blueprints/kimi-k2.5
terraform init
terraform apply

# 2. Launch GPU node (capacity block - manual)
aws ec2 run-instances \
  --instance-type p5e.48xlarge \
  --capacity-reservation-specification 'CapacityReservationTarget={CapacityReservationId=<cr-id>}' \
  --instance-market-options 'MarketType=capacity-block' \
  ...

# 3. Mount FSx and set up NVMe on the GPU node
ssh ec2-user@<gpu-node>
mount-fsx.sh <fsx-dns> <mount-name>
# Set up NVMe RAID0 (see docs/moe-loading-best-practices.md section 5.3)

# 4. Copy model and start vLLM
rsync -av /mnt/fsx/models/Kimi-K2.5/ /mnt/nvme/models/Kimi-K2.5/
# vLLM starts via Kubernetes deployment
```

## Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vllm_model_id` | `moonshotai/Kimi-K2.5` | Model identifier |
| `gpu_instance_types` | `["p5e.48xlarge"]` | GPU instance (8x H200) |
| `fsx_storage_capacity` | `100800` | ~100 TiB FSx Lustre |
| `vllm_tensor_parallel_size` | `8` | Must match GPU count |
| `vllm_max_model_len` | `32768` | Context length limit |
| `vllm_gpu_memory_utilization` | `0.85` | Leave headroom for MoE routing |

## Critical vLLM Flags

These are always applied:
- `--enforce-eager` -- Prevents torch.compile hangs on MoE kernels
- `--trust-remote-code` -- Required for Kimi architecture
- `--tensor-parallel-size 8` -- Must match GPU count
- `--enable-prefix-caching` -- For KV cache benchmarks
- `--mm-encoder-tp-mode data` -- Vision encoder parallelism
- `--tool-call-parser kimi_k2` / `--reasoning-parser kimi_k2` -- Kimi-specific parsers

## Documentation

| Document | Description |
|----------|-------------|
| [Lessons Learned](lessons.md) | Operational lessons from p5e deployments |
| [Benchmark Report](results/benchmark-report.md) | Consolidated LMCache + Dynamo KVBM results |
| [Execution Log](results/execution-log.md) | kubectl commands used to run benchmarks |
| [MoE Loading Best Practices](docs/moe-loading-best-practices.md) | Guide for loading and serving Kimi K2.5 |
| [Dynamo KV Cache GDS](docs/dynamo-kv-cache-gds.md) | NVIDIA Dynamo architecture reference |
| [Dynamo GDS Benchmark Plan](docs/dynamo-gds-benchmark-plan.md) | GPU Direct Storage benchmarking plan |
| [Mooncake Assessment](docs/mooncake-assessment.md) | Mooncake as L3 storage backend evaluation |

## Cost Estimate

| Resource | Cost |
|----------|------|
| p5e.48xlarge (capacity block) | ~$60-98/hr |
| FSx Lustre 100 TiB | ~$1,400/month |
| EKS cluster | ~$0.10/hr |
| NAT Gateway | ~$0.045/hr + data |
| **Per 8-hour session** | **~$500-800** |

## Spec Reference

See [specs/kimi-k2.5.md](../../specs/kimi-k2.5.md) for full requirements.
