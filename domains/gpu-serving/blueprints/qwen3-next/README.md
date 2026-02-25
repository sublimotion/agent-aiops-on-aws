# Qwen3-Next-80B MoE Inference Blueprint

Deploys Qwen/Qwen3-Next-80B-A3B-Instruct (80B MoE, 3B active) on p5en.48xlarge with 8x H200 GPUs, FSx Lustre PERSISTENT_2, and latency-optimized benchmarking across vLLM and SGLang.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  EKS Cluster (us-east-2)            │
│                                                     │
│  ┌───────────┐  ┌────────────────────────────────┐  │
│  │  System   │  │  p5en.48xlarge (Capacity Block) │  │
│  │  2x m6i   │  │  8x H200 (141 GB HBM3e each)   │  │
│  │           │  │  ┌──────────────────────┐       │  │
│  │           │  │  │ vLLM / SGLang        │       │  │
│  │           │  │  │ Qwen3-Next-80B FP8   │       │  │
│  │           │  │  │ TP=4 or DP=8+EP      │       │  │
│  │           │  │  └──────────────────────┘       │  │
│  └───────────┘  │  NVMe RAID0 (~30 TB)            │  │
│                 └────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  FSx Lustre (4.8 TiB PERSISTENT_2)            │  │
│  │  Model storage (persistent across sessions)   │  │
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
| Architecture | Hybrid-attention Mixture-of-Experts (MoE) |
| Total Parameters | 80B |
| Activated Parameters | 3B per forward pass |
| Experts | 512 (10 active per token + 1 shared) |
| Layers | 48: 12 x (3x Gated DeltaNet-MoE + 1x Gated Attention-MoE) |
| Context Length | 262,144 tokens native |
| Model Size | ~80 GB (FP8 quantized) |
| Attention | Hybrid: DeltaNet (linear) + Gated Attention |
| Quantization | FP8 (block_k=128) |
| MTP | Native multi-token prediction head |

## Prerequisites

- AWS account with p5en.48xlarge capacity block access in us-east-2
- Terraform >= 1.0
- kubectl configured
- HuggingFace access to Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
- Python 3.11+ with `openai`, `pandas`, `aiohttp` (for benchmarks)

## Reproducing the Blueprint End-to-End

### Step 1: Deploy Infrastructure

```bash
cd blueprints/qwen3-next
terraform init
terraform apply
```

This creates the EKS cluster, FSx Lustre PERSISTENT_2 filesystem (4.8 TiB), networking, and node groups. Note the FSx DNS name and mount name from the Terraform outputs.

### Step 2: Stage Container Images

Build and push custom images with Qwen3-Next dependencies (transformers-main, flash-linear-attention, causal-conv1d) to ECR:

```bash
bash ../../scripts/stage-images-ecr.sh
```

Or build individually:

```bash
docker build --platform linux/amd64 \
  -f docker/Dockerfile.vllm-qwen3next \
  -t <ecr>/vllm-qwen3next:v0.15.0 docker/
docker push <ecr>/vllm-qwen3next:v0.15.0
```

### Step 3: Stage Model Weights

From an internet-connected host, download and copy FP8 weights to FSx:

```bash
bash scripts/stage-model.sh
```

This runs `huggingface-cli download Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` and copies to `/fsx/models/qwen3-next-fp8/`.

### Step 4: Launch GPU Node

Reserve and launch the p5en.48xlarge capacity block:

```bash
aws ec2 run-instances \
  --instance-type p5en.48xlarge \
  --capacity-reservation-specification \
    'CapacityReservationTarget={CapacityReservationId=<cr-id>}' \
  --instance-market-options 'MarketType=capacity-block' \
  ...
```

### Step 5: Copy Model to NVMe

SSH into the GPU node, create NVMe RAID0, and copy model from FSx for fastest loading (~30-60s deserialization vs minutes from FSx):

```bash
bash scripts/copy-to-nvme.sh
```

### Step 6: Start a Serving Configuration

Launch scripts for each configuration are in `configs/`. All use `nerdctl` with `--gpus all` and mount `/mnt/fsx` and `/mnt/nvme`:

| Config | Script | Engine | Description |
|--------|--------|--------|-------------|
| vLLM baseline | `configs/vllm-baseline.sh` | vLLM | TP=4, FP8, prefix caching |
| vLLM TP4+MTP | `configs/vllm-tp4-mtp.sh` | vLLM | TP=4 with MTP speculative decoding (2 tokens) |
| vLLM TP8+MTP | `configs/vllm-tp8-mtp.sh` | vLLM | TP=8 with MTP (requires BF16, FP8 blocked) |
| vLLM DP8+EP | `configs/vllm-dp8-ep.sh` | vLLM | Data-parallel=8 + expert parallel, max throughput |
| SGLang baseline | `configs/sglang-baseline.sh` | SGLang | TP=8, RadixAttention |
| SGLang TP4 | `configs/sglang-tp4.sh` | SGLang | TP=4 for cross-engine comparison |
| SGLang TP8+MTP | `configs/sglang-tp8-mtp.sh` | SGLang | TP=8 with NEXTN speculative decoding |

**Baseline example** (start here to verify the model loads correctly):

```bash
bash configs/vllm-baseline.sh
```

### Step 7: Run Benchmarks

```bash
bash scripts/run-benchmarks.sh
```

The benchmark script runs P0/P1/P2 tiered workloads against the active serving configuration and writes results to `results/session-<date>/`.

### Step 8: Tear Down

```bash
terraform destroy
```

## Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `serving_engine` | `vllm` | `vllm` or `sglang` |
| `parallelism_config` | `tp8-x1` | `tp8-x1`, `tp4-x1`, `dp8-ep` |
| `enable_mtp` | `true` | Enable MTP speculative decoding |
| `model_path` | `/local/models/qwen3-next-fp8` | Path to staged FP8 weights on NVMe |
| `fsx_storage_capacity` | `4800` | 4.8 TiB FSx PERSISTENT_2 |
| `vllm_gpu_memory_utilization` | `0.92` | Leave headroom for EPLB redundant experts |
| `vllm_max_model_len` | `131072` | Context length (32768 for dp8-ep) |

## Critical Serving Flags

### vLLM

- `--quantization fp8` -- FP8 quantized checkpoint
- `--tensor-parallel-size 4` -- TP=4 baseline (FP8 block_k=128 incompatible with TP=8)
- `--enable-prefix-caching` -- Agentic workloads reuse large system prompts
- `--tool-call-parser qwen3_coder` -- Qwen3-Next tool call support
- `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` -- Native MTP
- `VLLM_ATTENTION_BACKEND=FLASHINFER` -- FlashInfer attention backend

### SGLang

- `--tp-size 8` -- Full TP=8 (SGLang handles FP8 differently)
- `--dtype bfloat16` -- Explicit compute dtype for non-quantized layers
- `--speculative-algo NEXTN` -- MTP speculative decoding
- RadixAttention prefix caching is on by default

> **Note**: `--trust-remote-code` is **not required**. DeltaNet attention and tokenizer are natively supported in transformers main branch.

## File Reference

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/stage-model.sh` | Download and stage FP8 model weights to FSx |
| `scripts/copy-to-nvme.sh` | Copy model from FSx to NVMe RAID0 for fast loading |
| `scripts/run-benchmarks.sh` | Full benchmark suite (P0/P1/P2 tiered workloads) |

### Configs

| Config | Purpose |
|--------|---------|
| `configs/vllm-baseline.sh` | vLLM TP=4, FP8, prefix caching |
| `configs/vllm-tp4-mtp.sh` | vLLM TP=4 with MTP (2 speculative tokens) |
| `configs/vllm-tp8-mtp.sh` | vLLM TP=8 with MTP (requires BF16) |
| `configs/vllm-dp8-ep.sh` | vLLM DP=8 + expert parallel, max throughput |
| `configs/sglang-baseline.sh` | SGLang TP=8, RadixAttention |
| `configs/sglang-tp4.sh` | SGLang TP=4 for cross-engine comparison |
| `configs/sglang-tp8-mtp.sh` | SGLang TP=8 with NEXTN speculative decoding |
| `configs/vllm-tp4-cpuoffload.sh` | vLLM TP=4 extended context (262K native); cpu-offload blocked on V1 engine |

### Docker

| File | Purpose |
|------|---------|
| `docker/Dockerfile.vllm-qwen3next` | vLLM + transformers-main + flash-linear-attention + causal-conv1d |
| `docker/Dockerfile.sglang-qwen3next` | SGLang + transformers-main + flash-linear-attention + causal-conv1d |

### Results

| File | Purpose |
|------|---------|
| `results/benchmark-report.md` | Consolidated findings across all configs |
| `results/execution-log.md` | Commands and logs from benchmark runs |
| `results/benchmark-visual-20260224.html` | Visual benchmark comparison |
| `results/readiness-audit-20260224.md` | Pre-deployment readiness audit |
| `results/session-20260224/` | Raw JSON results from benchmark session |
| `results/compound-20260224.md` | Elevated lessons from deployment session |
| `results/compound-20260224-benchmarks.md` | Benchmark-specific lessons and findings |

### Docs

| Document | Purpose |
|----------|---------|
| `serving-best-practices.md` | Serving configuration guidance for Qwen3-Next on H200 |
| `lessons.md` | Operational lessons from p5en deployments |

## Cost Estimate

| Resource | Cost |
|----------|------|
| p5en.48xlarge (capacity block) | ~$41.61/hr |
| FSx Lustre 4.8 TiB PERSISTENT_2 | ~$696/month |
| EKS cluster | ~$0.10/hr |
| m6i.xlarge system nodes (x2) | ~$0.38/hr |
| **Per 5-hour session** | **~$210** |

## Spec Reference

See [specs/qwen3-next.md](../../specs/qwen3-next.md) for full requirements.
