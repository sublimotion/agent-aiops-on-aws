# Benchmark Runs Documentation

This document captures all Kubernetes benchmark commands used during the vLLM KV cache benchmark evaluation.

## Prerequisites

### Cluster Access
```bash
aws eks update-kubeconfig --name vllm-kv-bench-eks-cluster --region us-east-1
```

### Port Forwarding
```bash
# vLLM API access
kubectl port-forward -n ml-inference svc/vllm-benchmark 30080:8000

# Prometheus metrics
kubectl port-forward -n monitoring svc/prometheus-server 9090:80
```

### Python Environment
```bash
cd blueprints/vllm-kv-benchmark
python3 -m venv venv
source venv/bin/activate
pip install openai pandas aiohttp
```

---

## Baseline Configuration (Native Prefix Caching)

### Deploy Baseline vLLM
```bash
kubectl patch deployment vllm-benchmark -n ml-inference --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/args",
    "value": [
      "--model", "s3://vllm-model-cache-615299764834/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/cfcb068fa7c44114cf77a462357c6cdcd2c304b4/",
      "--port", "8000",
      "--gpu-memory-utilization", "0.9",
      "--max-model-len", "32768",
      "--enable-prefix-caching",
      "--disable-log-requests"
    ]
  }
]'
```

### Restart Deployment (GPU workloads)
```bash
# Scale to 0 first to avoid scheduling conflicts
kubectl scale deployment/vllm-benchmark -n ml-inference --replicas=0
sleep 5
kubectl scale deployment/vllm-benchmark -n ml-inference --replicas=1
kubectl rollout status deployment/vllm-benchmark -n ml-inference --timeout=300s
```

---

## LMCache + FSx Configuration

### Deploy LMCache with FSx Backend
```bash
kubectl patch deployment vllm-benchmark -n ml-inference --type='json' -p='[
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/env",
    "value": [
      {"name": "LMCACHE_CHUNK_SIZE", "value": "256"},
      {"name": "LMCACHE_LOCAL_DISK", "value": "file:///fsx/kv-cache/"},
      {"name": "LMCACHE_MAX_LOCAL_DISK_SIZE", "value": "100.0"}
    ]
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/args",
    "value": [
      "--model", "s3://vllm-model-cache-615299764834/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/cfcb068fa7c44114cf77a462357c6cdcd2c304b4/",
      "--port", "8000",
      "--gpu-memory-utilization", "0.9",
      "--max-model-len", "32768",
      "--enable-prefix-caching",
      "--disable-log-requests",
      "--kv-transfer-config", "{\"kv_connector\":\"LMCacheConnectorV1\", \"kv_role\":\"kv_both\"}"
    ]
  }
]'
```

### Remove LMCache (Switch to Baseline)
```bash
kubectl patch deployment vllm-benchmark -n ml-inference --type='json' -p='[
  {"op": "remove", "path": "/spec/template/spec/containers/0/env"},
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/args",
    "value": [
      "--model", "s3://vllm-model-cache-615299764834/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/cfcb068fa7c44114cf77a462357c6cdcd2c304b4/",
      "--port", "8000",
      "--gpu-memory-utilization", "0.9",
      "--max-model-len", "32768",
      "--enable-prefix-caching",
      "--disable-log-requests"
    ]
  }
]'
```

### Check FSx Cache Size
```bash
kubectl exec deployment/vllm-benchmark -n ml-inference -- du -sh /fsx/kv-cache/
```

### Clear FSx Cache
```bash
kubectl exec deployment/vllm-benchmark -n ml-inference -- rm -rf /fsx/kv-cache/*
```

---

## LMBench Workload Commands

### Clone LMBench
```bash
git clone --depth 1 https://github.com/LMCache/LMBench.git
```

### Synthetic Multi-Round QA
```bash
cd LMBench/3-workloads/synthetic

# Low QPS (0.5)
python multi-round-qa.py \
  --num-users 10 \
  --shared-system-prompt 1000 \
  --user-history-prompt 500 \
  --answer-len 200 \
  --num-rounds 5 \
  --qps 0.5 \
  --base-url http://localhost:30080 \
  --model "s3://vllm-model-cache-615299764834/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/cfcb068fa7c44114cf77a462357c6cdcd2c304b4/" \
  --output results/synthetic_low.csv

# Medium QPS (2.0)
python multi-round-qa.py \
  --num-users 10 \
  --shared-system-prompt 1000 \
  --user-history-prompt 500 \
  --answer-len 200 \
  --num-rounds 5 \
  --qps 2.0 \
  --base-url http://localhost:30080 \
  --model "s3://vllm-model-cache-615299764834/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/cfcb068fa7c44114cf77a462357c6cdcd2c304b4/" \
  --output results/synthetic_medium.csv

# High QPS (4.0)
python multi-round-qa.py \
  --num-users 10 \
  --shared-system-prompt 1000 \
  --user-history-prompt 500 \
  --answer-len 200 \
  --num-rounds 5 \
  --qps 4.0 \
  --base-url http://localhost:30080 \
  --model "s3://vllm-model-cache-615299764834/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/cfcb068fa7c44114cf77a462357c6cdcd2c304b4/" \
  --output results/synthetic_high.csv
```

### Agentic Workload
```bash
cd LMBench/3-workloads/agentic

python agentic-qa.py \
  --num-agents 5 \
  --num-rounds 10 \
  --base-url http://localhost:30080 \
  --model "s3://vllm-model-cache-615299764834/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/cfcb068fa7c44114cf77a462357c6cdcd2c304b4/" \
  --output results/agentic.csv
```

### Long Context Stress Test
```bash
cd LMBench/3-workloads/synthetic

# 8K context
python multi-round-qa.py \
  --num-users 5 \
  --shared-system-prompt 4000 \
  --user-history-prompt 4000 \
  --answer-len 200 \
  --num-rounds 3 \
  --qps 0.5 \
  --base-url http://localhost:30080 \
  --model "..." \
  --output results/long_8k.csv

# 16K context
python multi-round-qa.py \
  --num-users 5 \
  --shared-system-prompt 8000 \
  --user-history-prompt 8000 \
  --answer-len 200 \
  --num-rounds 3 \
  --qps 0.5 \
  --base-url http://localhost:30080 \
  --model "..." \
  --output results/long_16k.csv

# 24K context (expect preemptions)
python multi-round-qa.py \
  --num-users 5 \
  --shared-system-prompt 12000 \
  --user-history-prompt 12000 \
  --answer-len 200 \
  --num-rounds 3 \
  --qps 0.3 \
  --base-url http://localhost:30080 \
  --model "..." \
  --output results/long_24k.csv
```

---

## Multi-Tenant Benchmark (Custom)

### Basic Usage
```bash
cd blueprints/vllm-kv-benchmark
source venv/bin/activate

python multi_tenant_benchmark.py \
  --base-url http://localhost:30080 \
  --model "s3://vllm-model-cache-615299764834/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/cfcb068fa7c44114cf77a462357c6cdcd2c304b4/" \
  --num-tenants 5 \
  --users-per-tenant 20 \
  --system-prompt-length 2000 \
  --questions-per-user 3 \
  --qps 2.0 \
  --output results/multi_tenant.csv
```

### High System Prompt Variety (50 tenants)
```bash
python multi_tenant_benchmark.py \
  --base-url http://localhost:30080 \
  --model "s3://vllm-model-cache-615299764834/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/cfcb068fa7c44114cf77a462357c6cdcd2c304b4/" \
  --num-tenants 50 \
  --users-per-tenant 5 \
  --system-prompt-length 4000 \
  --questions-per-user 2 \
  --qps 3.0 \
  --output results/multi_tenant_50tenants.csv
```

### Cold Start Test
```bash
# 1. Warm up cache
python multi_tenant_benchmark.py \
  --base-url http://localhost:30080 \
  --model "..." \
  --num-tenants 5 \
  --users-per-tenant 3 \
  --system-prompt-length 4000 \
  --questions-per-user 2 \
  --qps 1.0 \
  --output results/warmup.csv

# 2. Check FSx cache populated
kubectl exec deployment/vllm-benchmark -n ml-inference -- du -sh /fsx/kv-cache/

# 3. Restart vLLM
kubectl scale deployment/vllm-benchmark -n ml-inference --replicas=0
sleep 5
kubectl scale deployment/vllm-benchmark -n ml-inference --replicas=1
kubectl rollout status deployment/vllm-benchmark -n ml-inference --timeout=300s

# 4. Run cold start benchmark
python multi_tenant_benchmark.py \
  --base-url http://localhost:30080 \
  --model "..." \
  --num-tenants 5 \
  --users-per-tenant 3 \
  --system-prompt-length 4000 \
  --questions-per-user 2 \
  --qps 1.0 \
  --output results/coldstart.csv
```

---

## Monitoring Commands

### Check vLLM Metrics
```bash
curl -s http://localhost:30080/metrics | grep -E "(prefix_cache|preemption|kv_cache)"
```

### Prometheus Queries
```promql
# Prefix cache hit rate
vllm:prefix_cache_hit_rate

# Preemptions (memory pressure indicator)
vllm:num_preemptions_total

# KV cache utilization
vllm:kv_cache_usage_percent
```

### Pod Logs
```bash
kubectl logs deployment/vllm-benchmark -n ml-inference --tail=100
kubectl logs deployment/vllm-benchmark -n ml-inference -f  # Follow
```

### Resource Usage
```bash
kubectl top pods -n ml-inference
kubectl describe pod -l app=vllm-benchmark -n ml-inference
```

---

## Test Matrix Summary

| Test | Command | Key Parameters |
|------|---------|----------------|
| Baseline 5 tenants | `multi_tenant_benchmark.py` | `--num-tenants 5 --system-prompt-length 2000` |
| Baseline 50 tenants | `multi_tenant_benchmark.py` | `--num-tenants 50 --system-prompt-length 4000` |
| LMCache 5 tenants | Deploy LMCache + `multi_tenant_benchmark.py` | Same as baseline |
| LMCache 50 tenants | Deploy LMCache + `multi_tenant_benchmark.py` | Same as baseline |
| Cold start baseline | Restart + `multi_tenant_benchmark.py` | `--num-tenants 5` |
| Cold start LMCache | Deploy LMCache, warmup, restart, benchmark | `--num-tenants 5` |
| Long context 8K | `multi-round-qa.py` | `--shared-system-prompt 4000 --user-history-prompt 4000` |
| Long context 16K | `multi-round-qa.py` | `--shared-system-prompt 8000 --user-history-prompt 8000` |
| Long context 24K | `multi-round-qa.py` | `--shared-system-prompt 12000 --user-history-prompt 12000` |

---

## Results Location

All CSV results are stored in `blueprints/vllm-kv-benchmark/results/`:

```
results/
├── vllm-baseline_synthetic_low.csv
├── vllm-baseline_synthetic_medium.csv
├── vllm-baseline_synthetic_high.csv
├── vllm-baseline_agentic.csv
├── vllm-baseline_rag_low.csv
├── multi_tenant_baseline.csv
├── multi_tenant_baseline_50tenants.csv
├── multi_tenant_lmcache_fsx.csv
├── multi_tenant_lmcache_50tenants.csv
├── warmup_lmcache.csv
├── coldstart_baseline.csv
├── coldstart_lmcache.csv
└── benchmark_summary.md
```
