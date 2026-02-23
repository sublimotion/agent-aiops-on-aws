# Ministral-3B Blueprint

Deploy vLLM serving Mistral's Ministral-3-3B-Instruct model on EKS with optional LMCache integration for KV cache optimization.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                            VPC                                   │
│  ┌─────────────────┐  ┌───────────────────────────────────────┐ │
│  │  SageMaker      │  │             EKS Cluster                │ │
│  │  Studio         │──│  ┌─────────┐   ┌───────────────────┐  │ │
│  │  (Code Editor)  │  │  │ System  │   │    GPU Nodes      │  │ │
│  └─────────────────┘  │  │ Nodes   │   │    (g6e/g6)       │  │ │
│                       │  └─────────┘   │                   │  │ │
│  ┌─────────────────┐  │                │ ┌───────────────┐ │  │ │
│  │  FSx Lustre     │──│────────────────│ │ vLLM+LMCache  │ │  │ │
│  │  (KV Cache)     │  │                │ │  Ministral-3B │ │  │ │
│  └─────────────────┘  │                │ └───────────────┘ │  │ │
│       (optional)      │                └───────────────────┘  │ │
│                       └───────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Benchmark Results

| Configuration | Avg TTFT | Throughput | Improvement |
|---------------|----------|------------|-------------|
| Native vLLM | 0.444s | 60.9 t/s | baseline |
| **LMCache + FSx** | **0.197s** | 61.0 t/s | **55% faster** |

**Key finding**: LMCache reduces TTFT by 40-55% through KV cache reuse.

See [kv-cache-benchmark/RESULTS.md](../kv-cache-benchmark/RESULTS.md) for detailed findings.

## Prerequisites

- AWS CLI configured
- Terraform >= 1.0
- HuggingFace account (for model access)

## Quick Start

```bash
# Initialize
terraform init

# Deploy with native vLLM (default)
terraform apply

# Deploy with LMCache + FSx for optimized KV caching
terraform apply \
  -var="kv_cache_backend=lmcache" \
  -var="lmcache_config=fsx" \
  -var="enable_fsx_lustre=true"
```

## Configuration

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `aws_region` | us-east-1 | AWS region |
| `gpu_instance_types` | g6e.xlarge, g6.xlarge, ... | GPU instances (priority order) |
| `vllm_model_id` | mistralai/Ministral-3-3B-Instruct-2512 | Model to serve |
| `enable_sagemaker` | true | Deploy SageMaker Studio |

### KV Cache Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `kv_cache_backend` | native | `native` or `lmcache` |
| `kv_cache_config` | none | Native preset: `none`, `cpu-light`, `fsx-swap` |
| `lmcache_config` | cpu | LMCache preset: `cpu`, `disk`, `fsx` |
| `enable_fsx_lustre` | false | Enable FSx Lustre for KV cache storage |

### LMCache Presets

| Preset | Backend | Use Case |
|--------|---------|----------|
| `cpu` | CPU memory | Low latency, moderate capacity |
| `disk` | Local NVMe | High capacity, higher latency |
| `fsx` | FSx Lustre | High throughput, shared storage |

## Usage

After deployment:

```bash
# Configure kubectl
aws eks update-kubeconfig --region us-east-1 --name aiops-eks-cluster

# Test vLLM
curl http://<load-balancer>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistralai/Ministral-3-3B-Instruct-2512", "messages": [{"role": "user", "content": "Hello!"}]}'

# Check LMCache config (when enabled)
kubectl get configmap lmcache-config -n ml-inference -o yaml
```

## When to Use LMCache

**Recommended for:**
- Multi-turn conversations with shared context
- RAG applications with common document prefixes
- Chatbots with system prompts
- Any workload where TTFT latency is critical

**Not needed for:**
- Single-shot completions with unique prompts
- Small models that fit entirely in GPU memory
- Maximum throughput without latency requirements

## Known Limitations

1. **Model format**: Ministral requires `--load_format mistral`, incompatible with S3 streaming
2. **GPU capacity**: g6e instances have limited availability; uses multi-instance fallback
3. **LMCache requirement**: Requires `--enforce-eager` flag (disables CUDA graphs)
4. **FSx provisioning**: FSx Lustre takes ~7 minutes to provision

## Cost Estimate

| Resource | Type | ~Monthly Cost |
|----------|------|---------------|
| EKS Cluster | Control plane | $73 |
| System Nodes | 2x m6i.large | $140 |
| GPU Nodes | 1x g6e.xlarge | $380 |
| NAT Gateway | Single | $45 |
| FSx Lustre (optional) | 1.2 TiB SCRATCH_2 | $168 |
| **Total (without FSx)** | | **~$640/mo** |
| **Total (with FSx)** | | **~$810/mo** |

*Costs vary by usage. GPU nodes can scale to 0 when idle.*

## Benchmarking

Run benchmarks using the included tools:

```bash
# Clone benchmark repos
git clone https://github.com/tteon/kvcache-offloading.git
git clone https://github.com/LMCache/LMBenchmark.git

# Run kvcache-offloading benchmark
cd kvcache-offloading
python benchmark.py \
  --api-base "http://<load-balancer>/v1" \
  --model "mistralai/Ministral-3-3B-Instruct-2512" \
  --label "my-test"

# Run LMBenchmark multi-round QA
cd LMBenchmark/synthetic-multi-round-qa
python multi-round-qa.py \
  --model "mistralai/Ministral-3-3B-Instruct-2512" \
  --base-url "http://<load-balancer>" \
  --num-users 5 --num-rounds 5
```
