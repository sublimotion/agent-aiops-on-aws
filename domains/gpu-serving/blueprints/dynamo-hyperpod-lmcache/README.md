# Dynamo HyperPod LMCache Recipe

Deploy NVIDIA Dynamo on SageMaker HyperPod and connect Dynamo's vLLM worker to the HyperPod managed tiered-storage daemon through LMCache.

## Result Snapshot

- Region: `us-west-2`
- Requested hardware: `ml.g7.2xlarge`
- Live fallback: `ml.g6e.xlarge` because the SageMaker HyperPod API model rejected `ml.g7.*`
- Model: `Qwen/Qwen3-0.6B`
- Image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1`
- Namespace: `dynamo-hp-lmcache`
- HyperPod L2 daemon: `sagemaker-hyperpod://10.2.37.31:9200`

## What This Example Proves

- HyperPod tiered storage is enabled and `ai-toolkit` is reachable on the GPU node.
- The Dynamo frontend serves OpenAI-compatible `/v1/chat/completions`.
- The Dynamo frontend discovers the backend worker and routes requests to `dyn://dynamo-hp-lmcache.backend.generate`.
- The Dynamo vLLM worker initializes `LMCacheConnectorV1`, opens `ai_toolkit_cache`, and connects to the HyperPod managed L2 daemon.
- Metrics expose Dynamo, vLLM, and LMCache counters on `DYN_SYSTEM_PORT=8081`.
- Dynamo frontend store/restart/replay hit HyperPod L2: replay returned `cached_tokens=1102`, `vllm:external_prefix_cache_hits_total=1102`, and `lmcache:num_hit_tokens_total=1103`.

## Deploy

```bash
kubectl apply -f domains/gpu-serving/blueprints/dynamo-hyperpod-lmcache/manifests/dynamo-lmcache-smoke.yaml
kubectl rollout status deployment/dynamo-lmcache-worker -n dynamo-hp-lmcache --timeout=15m
```

The manifest runs the Dynamo frontend and Dynamo vLLM worker in one pod with file discovery. This avoids pulling additional control-plane images on a cluster where workload pods may not have normal internet egress. The frontend listens on port `8000`; worker/Dynamo/LMCache metrics are exposed on port `8081`.

## Validate The Frontend Path

```bash
kubectl port-forward -n dynamo-hp-lmcache svc/dynamo-lmcache-frontend 18000:8000 18081:8081

curl -s http://127.0.0.1:18000/v1/models
curl -s http://127.0.0.1:18000/health
curl -s http://127.0.0.1:18081/metrics | grep -E '^(dynamo_|vllm:|lmcache:)'
```

Run the telemetry probe:

```bash
python3 domains/gpu-serving/blueprints/dynamo-hyperpod-lmcache/scripts/dynamo_frontend_l2_probe.py store
kubectl rollout restart deployment/dynamo-lmcache-worker -n dynamo-hp-lmcache
kubectl rollout status deployment/dynamo-lmcache-worker -n dynamo-hp-lmcache --timeout=15m
python3 domains/gpu-serving/blueprints/dynamo-hyperpod-lmcache/scripts/dynamo_frontend_l2_probe.py replay
```

Artifacts are written under `results/`.

## Key Artifacts

- `manifests/dynamo-lmcache-smoke.yaml` - runnable Kubernetes recipe.
- `scripts/dynamo_frontend_l2_probe.py` - repeatable frontend store/restart/replay probe.
- `results/e2e-telemetry-dynamo-frontend-short-store-20260709.json` - Dynamo frontend store pass.
- `results/e2e-telemetry-dynamo-frontend-short-replay-20260709.json` - Dynamo frontend post-restart L2 hit proof.
- `results/e2e-telemetry-force-l2-replay-20260709.json` - raw vLLM control proof.
- `results/deployment-log-20260709.md` - full deployment timeline.
- `lessons.md` - reusable field notes and failure modes.

## Notes

In this image, `python3 -m dynamo.vllm --connector lmcache` is rejected for vLLM and must be expressed as:

```bash
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"sagemaker_hyperpod_shared_memory_name":"ai_toolkit_cache"}}'
```

`dnsPolicy: Default`, `PYTHONHASHSEED=0`, and `save_unfull_chunk: true` were required for this HyperPod run.
