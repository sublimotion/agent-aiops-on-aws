# qwen3-next-sglang

Benchmark Qwen3-Next-80B on SGLang + HiCache on g7e.24xlarge (4x L40S, TP=4) to validate coding agent viability.

**Spec**: [`domains/gpu-serving/specs/qwen3-next-sglang.md`](../../specs/qwen3-next-sglang.md)
**Parent specs**: [`qwen3-next.md`](../../specs/qwen3-next.md), [`qwen3-next-g7e.md`](../../specs/qwen3-next-g7e.md)

## Prerequisites

1. **qwen3-next blueprint deployed** — provides the S3 model bucket (`model_s3_bucket_id` variable).
   ```bash
   terraform -chdir=../qwen3-next output models_bucket_name
   ```
2. Model staged as FP8 in S3 (Qwen3-Next-80B-FP8).
3. AWS credentials with EKS and EC2 access.

## Instance

| Property | Value |
|----------|-------|
| Instance | g7e.24xlarge |
| GPUs | 4x NVIDIA L40S (48 GB each, 192 GB total) |
| VRAM + CPU KV budget | ~264 GB GPU + ~300 GB CPU ≈ 564 GB |
| Interconnect | PCIe (no NVLink, no EFA/GDS) |
| On-demand | $16.57/hr |
| Spot (us-west-2d) | ~$2.94/hr |

## Benchmark Phases

| Phase | Name | Duration | Description |
|-------|------|----------|-------------|
| S0 | Smoke test | 30 min | Model loads, basic inference, tool-calling |
| S1 | Throughput baseline | 1 hr | QPS sweep 0.5–8.0, target ≥150 tok/s |
| S2 | BFCL tool-use eval | 1 hr | 200 scenarios, gate: BFCL ≥75 for coding agents |
| S3 | HiCache tiers | 1.5 hr | L1-only → L1+L2 (CPU) → L1+L2+L3 (NVMe) |
| S4 | Swarm concurrency | 1 hr | 4–64 concurrent agents with tool-call delays |

**Estimated cost**: ~$100 for full run (6 hours on g7e.24xl).

## Quick Start

```bash
# 1. Deploy infrastructure
terraform init && terraform apply

# 2. SSH to GPU node and copy model to NVMe
./scripts/copy-to-nvme.sh   # if available, or see outputs.tf

# 3. Start SGLang baseline
bash configs/sglang-baseline.sh

# 4. Run benchmarks (all phases)
./scripts/run-benchmarks.sh all

# 5. Run a single phase
./scripts/run-benchmarks.sh s0
```

## Config Files

| Config | Use Case |
|--------|----------|
| `sglang-baseline.sh` | TP=4 baseline, no HiCache |
| `sglang-hicache-l2.sh` | HiCache L1+L2 (CPU offload) |
| `sglang-hicache-nvme.sh` | HiCache L1+L2+L3 (NVMe tier) |
| `sglang-tp8-baseline.sh` | TP=8 baseline (if multi-node) |
| `sglang-tp8-hicache-l2.sh` | TP=8 + HiCache L2 |
| `sglang-2replica-tp4.sh` | 2 replicas, TP=4 each (GPUs 0-3 / 4-7) |

## Key Caveats

- **`--disable-cuda-graph` required** for HiCache with hybrid attention models.
- **`write_through` policy only** — `write_back` risks data loss on eviction with MambaRadixCache.
- **PR #19663** (HiCache for MambaRadixCache) may need cherry-picking; see `docker/` fallback Dockerfile if needed.
- **cu131 image required** for Blackwell sm_100 support.

## Decision Gates

- BFCL < 70 → **Stop** (not viable for coding agents)
- BFCL 70–75 → **Caution** (viable for swarms only)
- BFCL ≥ 75 → **Proceed** (viable for both swarm and interactive)
- BFCL ≥ 80 → **Strong** (competitive with Claude Sonnet)
