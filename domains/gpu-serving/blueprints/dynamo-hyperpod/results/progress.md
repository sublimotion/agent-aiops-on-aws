# Dynamo on HyperPod — Progress

## Session 2026-03-31

### Deployment Summary

| Component | Status | Details |
|-----------|--------|---------|
| Namespace | DEPLOYED | `dynamo-validation` |
| FSx PVC | BOUND | `fsx-dynamo` → `fsx-dynamo-pv` (same 1.2 TiB filesystem as llmd-validation) |
| etcd | RUNNING | v3.5.17, healthy, GPU node scheduling |
| Dynamo vLLM Worker | RUNNING | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1` (vLLM 0.16.0) |
| Model | SERVING | Qwen/Qwen3-0.6B, prefix caching enabled |
| GPU | A10G (sm_86) | Confirmed compatible with Dynamo vLLM runtime |

### Smoke Test Results

**29 passed, 0 failed, 5 skipped**

| Stage | Result | Notes |
|-------|--------|-------|
| 1. Infrastructure Discovery | 6 PASS, 1 SKIP | ADOT not installed (expected) |
| 2. Dynamo Components | 5 PASS | etcd healthy, FSx PVC bound, 54 CRDs |
| 3. vLLM Baseline | 6 PASS | Health 200, chat completion works, prefix cache metrics exposed |
| 4. KVBM G2 CPU Cache | 1 PASS, 2 SKIP | Env var set (8GB) but KVBM not active in vllm-runtime image |
| 5. KVBM G3 FSx Cache | 6 PASS, 1 SKIP | FSx mounted, writable, env vars set, llm-d cache dir coexists |
| 6. Observability | 2 PASS, 1 SKIP | vLLM metrics accessible, PodMonitor CRD available |
| 7. CRD Coexistence | 3 PASS | Gateway API + IEC CRDs coexist, no operator errors |

### Key Findings

1. **NGC entrypoint incompatibility**: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1` uses `/opt/nvidia/nvidia_entrypoint.sh` which fails with bare `--model` args. Fix: explicit `command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]`.

2. **KVBM not active in vllm-runtime image**: The image bundles standard vLLM 0.16.0. KVBM env vars (`DYN_KVBM_*`) are set but not consumed. KVBM requires the full Dynamo orchestration stack (Frontend + Router + Planner), not just the vLLM worker.

3. **Disk pressure from 12.3 GB image**: g5.4xlarge has ~95 GB ephemeral storage. The Dynamo image (12.3 GB) plus existing llm-d/HyperPod images caused DiskPressure taint. Resolved by scaling down llmd-validation pods and letting kubelet GC reclaim.

4. **A10G (sm_86 Ampere) confirmed compatible**: Despite concerns about NGC targeting L40S/H100/B200, the vLLM runtime works fine on A10G with FlashAttention backend.

5. **FSx coexistence works**: Separate PV (`fsx-dynamo-pv`) with different volumeHandle suffix points to same filesystem. Dynamo uses `/mnt/fsx/kv-cache/` while llm-d uses `/mnt/fsx/kvcache` — no conflict.

6. **etcd health check**: The etcd v3.5.17 image doesn't include `wget`. Use `etcdctl endpoint health` instead.

### Deployment Order

```bash
# 1. Scale down llmd-validation to free GPU
kubectl scale deployment -n llmd-validation ms-inference-scheduling-llm-d-modelservice-decode --replicas=0
kubectl scale deployment -n llmd-validation gaie-inference-scheduling-epp --replicas=0
kubectl scale deployment -n llmd-validation infra-inference-scheduling-inference-gateway-istio --replicas=0

# 2. Apply Dynamo manifests
kubectl apply -f manifests/namespace.yaml
kubectl apply -f manifests/fsx-pvc.yaml
kubectl apply -f manifests/etcd.yaml
kubectl apply -f manifests/dynamo-worker.yaml

# 3. Run smoke test
bash scripts/smoke_test.sh
```

### Next Steps

- [ ] Deploy full Dynamo stack (Frontend + Router + Planner) to activate KVBM
- [ ] Test KVBM G2 CPU cache with full Dynamo orchestration
- [ ] Test KVBM G3 FSx disk cache population
- [ ] Benchmark prefix cache hit rate vs llm-d LMCache
