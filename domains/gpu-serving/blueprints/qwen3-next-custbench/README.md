# Qwen3-Next Customer Benchmark

Reproduces and optimizes a customer's Qwen3-Next-80B-A3B-Instruct serving configuration on p5en.48xlarge (8x H200). Compares customer's exact config against optimized configs to quantify the impact of prefix caching, chunked prefill, FP8 quantization, and KV cache offloading.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              EKS Cluster (us-east-2c)               │
│                                                     │
│  ┌───────────┐  ┌────────────────────────────────┐  │
│  │  System   │  │  p5en.48xlarge (Capacity Block) │  │
│  │  2x m6i   │  │  8x H200 (141 GB HBM3e each)   │  │
│  │           │  │                                  │  │
│  │           │  │  Config A: Customer baseline     │  │
│  │           │  │  Config B: Optimized (prefix+FP8)│  │
│  │           │  │  Config C: Optimized - no MTP    │  │
│  │           │  │  Config D: + CPU offload         │  │
│  │           │  │  Config E: 2x TP=4 replicas      │  │
│  │           │  │                                  │  │
│  └───────────┘  │  NVMe RAID0 (~30 TB)            │  │
│                 └────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  FSx Lustre (4.8 TiB PERSISTENT_2)            │  │
│  │  Model storage (persistent across sessions)   │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  FSx Lustre EFA (4.8 TiB PERSISTENT_2)        │  │
│  │  Dynamo KVBM tiered KV cache offloading (T5d) │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Benchmark Tiers

| Tier | Config | Description |
|------|--------|-------------|
| T1 | A | Customer reproduction (1000 concurrent, 10K in / 1K out) |
| T2 | B | Optimized head-to-head (same workload) |
| T2b | B vs A | Prefix sharing (8K shared prefix + 128 unique) |
| T3 | C vs B | MTP isolation (no MTP vs MTP) |
| T4 | B | Load scaling (0.5 / 5.0 / inf qps) |
| T5 | Various | Simulated memory-constrained KV offloading (gpu-mem=0.30) |
| T6 | E | 2x replica + CPU offload (1000 concurrent) |
| T7 | E | Stress test at 1500 concurrent |

## Quick Start

```bash
# 1. Reserve capacity block (update CAPACITY_RESERVATION_ID)
./scripts/launch-capacity-block.sh

# 2. Wait for node to join EKS
kubectl get nodes -w

# 3. Copy model from FSx to NVMe
ssh <node>  # or SSM
cp -r /mnt/fsx/models/qwen3-next /mnt/nvme/models/qwen3-next-fp8

# 4. Start a config and run benchmarks
./configs/vllm-customer-baseline.sh   # Config A
./scripts/run-benchmarks.sh t1        # T1: Customer reproduction

./configs/vllm-optimized.sh           # Config B
./scripts/run-benchmarks.sh t2        # T2: Optimized
```

## Configs

| File | Config | Description |
|------|--------|-------------|
| `configs/vllm-customer-baseline.sh` | A | Customer's exact flags |
| `configs/vllm-optimized.sh` | B | + prefix caching, chunked prefill, FP8, qwen3_coder |
| `configs/vllm-optimized-nomtp.sh` | C | Config B without MTP |
| `configs/vllm-optimized-cpuoffload.sh` | D | Config B + cpu-offload-gb 64 |
| `configs/vllm-2replica-cpuoffload.sh` | E | 2x TP=4 replicas + CPU offload |
| `configs/vllm-constrained-dynamo-fsx.sh` | T5d | Dynamo KVBM tiered offload to FSx |

## Parent Blueprint

Reuses infrastructure from [qwen3-next](../qwen3-next/) (EKS, VPC, FSx, model). See [spec](../../specs/qwen3-next-custbench.md) for full details.
