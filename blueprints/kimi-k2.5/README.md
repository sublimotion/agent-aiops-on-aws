# Kimi K2.5 MoE Inference Blueprint

Deploys moonshotai/Kimi-K2.5 (1T parameter MoE model) on p5e.48xlarge with 8x H200 GPUs, FSx Lustre, and KV cache benchmarking across three serving stacks.

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
- Python 3.11+ with `openai`, `pandas`, `aiohttp` (for benchmarks)

## Reproducing the Blueprint End-to-End

### Step 1: Deploy Infrastructure

```bash
cd blueprints/kimi-k2.5
terraform init
terraform apply
```

This creates the EKS cluster, FSx Lustre filesystem, networking, and node groups. Note the FSx DNS name and mount name from the Terraform outputs.

### Step 2: Validate Storage

Before launching the GPU instance, verify the infrastructure is ready:

```bash
bash scripts/validate-storage.sh
```

This checks S3, FSx, DRA, EFA, and connectivity. All checks should pass before proceeding.

### Step 3: Launch GPU Node

Capacity block instances are provisioned manually:

```bash
aws ec2 run-instances \
  --instance-type p5e.48xlarge \
  --capacity-reservation-specification \
    'CapacityReservationTarget={CapacityReservationId=<cr-id>}' \
  --instance-market-options 'MarketType=capacity-block' \
  ...
```

### Step 4: Set Up NVMe RAID and Copy Model

SSH into the GPU node, then run the setup script. This creates a RAID0 array across the 8x NVMe drives (~30TB), mounts FSx, and copies the model to NVMe for faster loading (~25 GB/s vs ~1-3 GB/s from FSx):

```bash
ssh ec2-user@<gpu-node>
sudo bash scripts/setup-nvme-model.sh \
  --fsx-dns <fsx-dns> \
  --fsx-mount <mount-name> \
  --model-name Kimi-K2.5
```

For better FSx throughput when loading directly, stripe model files across all OSTs first:

```bash
sudo bash scripts/stripe-model-fsx.sh \
  --model-name Kimi-K2.5 \
  --fsx-mount /mnt/fsx-dynamo
```

See [docs/moe-loading-best-practices.md](docs/moe-loading-best-practices.md) for details on NVMe RAID0 setup (section 5.3) and FSx striping.

### Step 5: Start a Serving Configuration

Launch scripts for each configuration are in `configs/`. All use Docker with `--gpus all` and mount `/mnt/fsx` and `/mnt/nvme`:

| Config | Script | Description |
|--------|--------|-------------|
| Baseline | `configs/baseline.sh` | Native vLLM prefix caching, no external KV cache |
| LMCache + GDS | `configs/lmcache.sh` | KV cache offloading to FSx via GPU Direct Storage |
| LMCache + POSIX | `configs/lmcache-posix.sh` | KV cache offloading to FSx via CPU bounce (no GDS) |
| Dynamo KVBM | `configs/dynamo.sh` | NVIDIA Dynamo tiered KV cache (VRAM -> DRAM -> NVMe -> FSx) |
| Mooncake | `configs/mooncake.sh` | Mooncake as L3 storage backend |

**Baseline example** (start here to verify the model loads correctly):

```bash
bash configs/baseline.sh
```

**LMCache** (requires additional setup):

```bash
bash scripts/setup-lmcache-p5e.sh   # install LMCache + dependencies
bash configs/lmcache.sh              # start vLLM with LMCache GDS backend
```

**Dynamo KVBM** (requires additional setup):

```bash
bash scripts/setup-dynamo-p5e.sh     # install Dynamo, verify GPUs, GDS drivers
bash configs/dynamo.sh               # start vLLM with Dynamo KVBM
```

Docker images for Dynamo and Mooncake are in `docker/`:
- `docker/Dockerfile.dynamo-kvbm` -- Dynamo KVBM image with MLA patch (`docker/dynamo-kvbm-mla.patch`)
- `docker/Dockerfile.vllm-mooncake` -- vLLM with Mooncake integration
- `docker/dynamo-config.yaml` / `docker/dynamo-config-posix.yaml` -- Dynamo runtime configs

### Step 6: Run Benchmarks

The benchmark suite tests all serving configurations across multiple workload types (reasoning, code generation, multi-turn, long context, multi-tenant, stress):

```bash
# From local machine, port-forward to the vLLM endpoint
kubectl port-forward -n ml-inference svc/vllm-benchmark 30080:8000

# Or if running Docker directly on the GPU node, the endpoint is localhost:8000

# Run the full suite
python3 scripts/run-benchmarks.py
```

The benchmark script (`scripts/run-benchmarks.py`) runs 10 workload categories against the active serving configuration and writes JSON results to `results/`. Switch the serving config (step 5), re-run the benchmarks, and compare.

For the exact kubectl patch commands used during our benchmark runs, see [results/execution-log.md](results/execution-log.md).

### Step 7: Tear Down

```bash
terraform destroy
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

These are always applied (see `configs/baseline.sh` for the canonical reference):
- `--enforce-eager` -- Prevents torch.compile hangs on MoE kernels
- `--trust-remote-code` -- Required for Kimi architecture
- `--tensor-parallel-size 8` -- Must match GPU count
- `--enable-prefix-caching` -- For KV cache benchmarks
- `--tool-call-parser kimi_k2` / `--reasoning-parser kimi_k2` -- Kimi-specific parsers
- `VLLM_ATTENTION_BACKEND=FLASHINFER` -- FlashInfer attention backend

## File Reference

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/setup-nvme-model.sh` | Create NVMe RAID0, mount FSx, copy model to local NVMe |
| `scripts/stripe-model-fsx.sh` | Re-stripe model files across all FSx OSTs for max throughput |
| `scripts/setup-dynamo-p5e.sh` | Install NVIDIA Dynamo KVBM, verify GPUs and GDS drivers |
| `scripts/setup-lmcache-p5e.sh` | Install LMCache with vLLM integration |
| `scripts/validate-storage.sh` | Pre-flight validation of S3, FSx, DRA, EFA |
| `scripts/validate-gds.md` | GDS validation reference (manual steps) |
| `scripts/run-benchmarks.py` | Full benchmark suite (10 workload types, JSON output) |

### Configs

| Config | Purpose |
|--------|---------|
| `configs/baseline.sh` | vLLM with native prefix caching (no external KV) |
| `configs/lmcache.sh` | vLLM + LMCache with GDS to FSx |
| `configs/lmcache-posix.sh` | vLLM + LMCache with POSIX fallback (no GDS) |
| `configs/dynamo.sh` | vLLM + NVIDIA Dynamo KVBM |
| `configs/mooncake.sh` | vLLM + Mooncake L3 backend |
| `configs/comparison.yaml` | Side-by-side config comparison matrix |
| `configs/dynamo-gds.yaml` | Dynamo GDS-specific Kubernetes manifest |

### Docker

| File | Purpose |
|------|---------|
| `docker/Dockerfile.dynamo-kvbm` | Dynamo KVBM image with MLA patch |
| `docker/Dockerfile.vllm-mooncake` | vLLM + Mooncake image |
| `docker/dynamo-config.yaml` | Dynamo GDS runtime config |
| `docker/dynamo-config-posix.yaml` | Dynamo POSIX fallback runtime config |
| `docker/dynamo-kvbm-mla.patch` | Patch for MLA support in Dynamo |

### Results

| File | Purpose |
|------|---------|
| `results/benchmark-report.md` | Consolidated findings across all configs |
| `results/execution-log.md` | kubectl commands used during benchmark runs |
| `results/kimi-k2.5-p5e/` | Raw JSON results from Dynamo runs |
| `results/kimi-k2.5-p5e-baseline/` | Raw JSON results from baseline runs |
| `results/kimi-k2.5-p5e-baseline-full/` | Full baseline benchmark results |
| `results/kimi-k2.5-p5e-v2/` | Second iteration benchmark results |
| `results/dynamo_*.json` | Individual Dynamo scenario results |

### Docs

| Document | Purpose |
|----------|---------|
| `docs/moe-loading-best-practices.md` | Loading and serving Kimi K2.5 on p5e |
| `docs/dynamo-kv-cache-gds.md` | NVIDIA Dynamo architecture reference |
| `docs/dynamo-gds-benchmark-plan.md` | GPU Direct Storage benchmarking plan |
| `docs/mooncake-assessment.md` | Mooncake as L3 storage backend evaluation |
| `docs/storage-interconnect-io-best-practices.md` | Storage and interconnect I/O guide |
| `docs/moe-checkpointing-best-practices.md` | MoE model checkpointing guide |
| `docs/eagle-vllm-vs-sglang.md` | Eagle speculative decoding comparison |
| `docs/sota-reasoning-models-analysis.md` | SOTA reasoning models landscape |
| `docs/together-ai-infrastructure.md` | Together AI infrastructure reference |
| `docs/use-cases.md` | Target use cases for KV cache offloading |
| `lessons.md` | Operational lessons from p5e deployments |

## Cost Estimate

| Resource | Cost |
|----------|------|
| p5e.48xlarge (capacity block) | ~$60-98/hr |
| FSx Lustre 100 TiB | ~$1,400/month |
| EKS cluster | ~$0.10/hr |
| NAT Gateway | ~$0.045/hr + data |
| **Per 8-hour session** | **~$500-800** |

## Spec Reference

See [specs/vllm-kv-cache-benchmark.md](../../specs/vllm-kv-cache-benchmark.md) for full requirements.
