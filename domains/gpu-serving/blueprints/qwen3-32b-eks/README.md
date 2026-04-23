# Qwen3-32B EKS Benchmark — Apple-to-Apple HyperPod Comparison

## Objective

Compare plain EKS vLLM serving vs HyperPod Inference Operator for Qwen3-32B-FP8
on L40S GPUs. Isolates the value-add of HyperPod operator features (LMCache L1 CPU
offload, KV-transfer config, prefix-aware routing) vs vanilla vLLM.

## Setup

| Property | EKS (this blueprint) | HyperPod (qwen3-32b-hyperpod) |
|----------|---------------------|-------------------------------|
| Cluster | `finetune-eks` (K8s 1.33) | `qwen3-32b-g6e-cluster` |
| Instance | g6e.2xlarge (1x L40S) | ml.g6e.48xlarge (8x L40S, 1 used) |
| Model | RedHatAI/Qwen3-32B-FP8-dynamic | Same |
| Image | lmcache/vllm-openai:latest-nightly | Same |
| GPU count | 1 | 1 |
| max-model-len | 24000 | 24000 |
| gpu-memory-utilization | 0.95 | 0.95 |

### Key differences

- **No HyperPod operator**: No LMCache connector injection, no `--kv-transfer-config`,
  no `LMCACHE_LOCAL_CPU=true` env var
- **No operator prefetch**: Model downloaded via S3 init container (not tmpfs prefetch)
- **Smaller instance**: g6e.2xlarge (8 vCPU, 64 GiB) vs g6e.48xlarge (192 vCPU, 1536 GiB).
  The model uses 1 GPU in both cases, but system resources differ.

## Configs

| Config | Prefix Cache | HyperPod Equivalent | What it tests |
|--------|-------------|---------------------|---------------|
| config0-nocache | Off | N/A (HyperPod had no pure no-cache config) | True baseline floor |
| config1-prefix-cache | vLLM `--enable-prefix-caching` | config1-baseline | vLLM-only prefix cache |

### HyperPod configs NOT replicated here

- **config0-nocache (HyperPod)**: Had `enableL1Cache: true` + operator LMCache injection.
  Not comparable to our config0 — our config0 is a true no-cache baseline.
- **config2-l1l2**: Operator L1+L2 tiered cache. Requires HyperPod operator.

## Workloads (identical to HyperPod)

| ID | Name | Description |
|----|------|-------------|
| W1 | Multi-Turn Chat | Sweep rounds (1/5/10) x concurrent (1/4/8) x QPS (1/4) |
| W2 | RAG / Long Doc QA | Shared doc prefix, warmup vs cached query |
| W3 | Agentic Tool Calling | Multi-turn with tool-call pauses (0.5/2/5s) |
| W4 | Shared System Prompt | Long shared prompt + short unique queries |
| W5 | ShareGPT Conversations | Variable-length real conversations at various QPS |
| W6 | Long Context Scaling | Input lengths 1K/4K/8K/16K |

## Quick Start

```bash
# 1. Create nodegroup
bash scripts/setup-nodegroup.sh

# 2. Update kubeconfig
aws eks update-kubeconfig --name finetune-eks --region us-east-1

# 3. Verify GPU node
kubectl get nodes -l node.kubernetes.io/instance-type=g6e.2xlarge
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'

# 4. Deploy L0 (no cache)
kubectl apply -f configs/config0-nocache.yaml
kubectl wait --for=condition=available deployment/qwen3-32b-nocache --timeout=900s

# 5. Port-forward and run benchmark
kubectl port-forward svc/qwen3-32b-nocache 8000:8000 &
python scripts/benchmark.py --config config0-nocache

# 6. Delete L0, deploy L1
kubectl delete -f configs/config0-nocache.yaml
kubectl apply -f configs/config1-prefix-cache.yaml
kubectl wait --for=condition=available deployment/qwen3-32b-prefix-cache --timeout=900s

# 7. Port-forward and run benchmark
kubectl port-forward svc/qwen3-32b-prefix-cache 8000:8000 &
python scripts/benchmark.py --config config1-prefix-cache

# 8. Scale down when done
aws eks update-nodegroup-config --cluster-name finetune-eks \
  --nodegroup-name g6e-benchmark --scaling-config minSize=0,maxSize=1,desiredSize=0 \
  --region us-east-1
```

## Expected Comparison Points

| Metric | EKS L0 (no cache) | EKS L1 (prefix) | HyperPod config1 (prefix) | HyperPod config0 (L1+LMCache) | HyperPod config2 (L1+L2) |
|--------|-------------------|-----------------|--------------------------|-------------------------------|--------------------------|
| W1 TTFT p50 | ? | ? | 62-143ms | 62-143ms | 63-143ms |
| W2 RAG improvement | N/A | ? | 1.06-1.21x | - | 1.50-3.40x |
| W3 degradation @10t | ? | ? | - | - | 0.55-1.52x |
| ITL p50 | ? | ? | 53-100ms | 53-100ms | 54-101ms |
