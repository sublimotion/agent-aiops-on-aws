# llm-d HyperPod LMCache Recipe

Deploy **llm-d** on SageMaker HyperPod and connect llm-d's vLLM replica to the HyperPod managed tiered-storage daemon (`ai-toolkit`, port 9200) through vLLM's `LMCacheConnectorV1`. This is the llm-d twin of the [`dynamo-hyperpod-lmcache`](../dynamo-hyperpod-lmcache/) recipe — same L2 outcome, different orchestrator.

Spec: [`domains/gpu-serving/specs/llmd-hyperpod-lmcache.md`](../../specs/llmd-hyperpod-lmcache.md)

## Result Snapshot — PASS (2026-07-10)

- Region: `us-west-2` (reused the existing `dynamo-hyperpod-lmcache` cluster)
- Selected hardware: `ml.g6e.xlarge` (1 GPU) — `ml.g7.*` not accepted by the SageMaker HyperPod API
- Model: `Qwen/Qwen3-0.6B`
- Worker image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1` (ships the SageMaker HyperPod LMCache adapter)
- llm-d: `llm-d-router-standalone` chart `v0.9.0`, EPP `v0.9.0`, GAIE `v1.5.0`, vLLM `0.16.0`, LMCache `0.3.14`
- Namespace: `llm-d-hp-lmcache`
- HyperPod L2 daemon: `sagemaker-hyperpod://10.2.37.31:9200` (shm `ai_toolkit_cache`)
- **Proof**: store → restart → replay produced `external_prefix_cache_hits_total 0→742`, `Retrieved 743/743 tokens`, 99.9% external hit rate, latency 3375ms→379ms (8.9x). Artifact: `results/e2e-telemetry-llmd-l2-proof-20260710.json`.
- Stage 6 (two-replica sharing) SKIPPED — single-GPU SKU.

> **Two gotchas that dominated the deploy** (see `lessons.md`): (1) mount the shm **file**
> `/dev/shm/ai_toolkit_cache` with `type: File`, NOT the whole `/dev/shm` — a Directory mount lets a
> terminating client `shm_unlink` the daemon-owned segment, poisoning all future clients (`LMCache is
> unhealthy, skipping store operation`). (2) The router had to be co-located on the GPU node because
> pod-to-pod traffic from vanilla EKS nodes to the HyperPod node is blocked.

## What This Example Proves

- HyperPod tiered storage is enabled and `ai-toolkit` is reachable on the GPU node.
- llm-d serves OpenAI-compatible `/v1/chat/completions` end-to-end through the gateway (NLB → gateway → EPP → vLLM).
- The llm-d-managed vLLM replica — **not** the HyperPod Inference Operator — initializes `LMCacheConnectorV1`, opens `ai_toolkit_cache`, and connects to the HyperPod managed L2 daemon.
- A **store → restart → replay** probe proves an L2 cache hit that survives a replica restart (L0/L1 wiped, only the managed daemon persists), served through the gateway.

## Deploy

llm-d is installed via its helmfile guides, then this blueprint's overlay is layered onto the `inference-scheduling` model service.

```bash
export LLMD_NS=llmd-hp-lmcache
kubectl create namespace $LLMD_NS --dry-run=client -o yaml | kubectl apply -f -

# 1. LMCache config ConfigMap (shared-memory name + save_unfull_chunk)
kubectl apply -n $LLMD_NS -f manifests/lmcache-configmap.yaml

# 2. Gateway provider (once per cluster)
git clone https://github.com/llm-d/llm-d.git
cd llm-d/guides/prereq/gateway-provider
./install-gateway-provider-dependencies.sh
helmfile apply -f istio.helmfile.yaml

# 3. Model service with the LMCache→L2 overlay
cd ../../inference-scheduling
helmfile apply --set ns=$LLMD_NS \
  -f gaie-inference-scheduling \
  -f ms-inference-scheduling  # merge configs/ms-values-hyperpod-lmcache.yaml as the ms values overlay
```

> The overlay in `configs/ms-values-hyperpod-lmcache.yaml` sets the vLLM `--kv-transfer-config`, the `LMCACHE_*` env, `PYTHONHASHSEED=0`, and mounts the LMCache ConfigMap. See the spec for the EPP RBAC / EnvoyExtensionPolicy `messageTimeout` prerequisites that must be satisfied first.

## Validate The Gateway Path

```bash
# Gateway endpoint
export GW=$(kubectl get gateways -n $LLMD_NS -o jsonpath='{.items[0].status.addresses[0].value}')
curl -s http://$GW/v1/models
curl -s http://$GW/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'

# vLLM + LMCache metrics from the replica
POD=$(kubectl get pod -n $LLMD_NS -l llm-d.ai/model=Qwen3-0.6B -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n $LLMD_NS $POD -c vllm -- \
  sh -c 'curl -s localhost:8000/metrics | grep -E "^(vllm:external_prefix_cache|lmcache:)"'
```

## Run The Store → Restart → Replay Proof

```bash
# Point the probe at the gateway (default http://127.0.0.1:8080 — port-forward first, or pass --base-url)
kubectl port-forward -n llmd-hp-lmcache svc/<gateway-svc> 8080:80 &

python3 scripts/llmd_l2_probe.py store
kubectl rollout restart deployment -n llmd-hp-lmcache -l llm-d.ai/model=Qwen3-0.6B
kubectl rollout status  deployment -n llmd-hp-lmcache -l llm-d.ai/model=Qwen3-0.6B --timeout=15m
python3 scripts/llmd_l2_probe.py replay
```

A PASS requires the `replay` artifact to show a non-zero `vllm:external_prefix_cache_hits_total` delta (and/or `lmcache:num_hit_tokens_total`) after the restart — the only surviving tier is the ai-toolkit daemon. Artifacts land in `results/`.

## Key Artifacts

- `configs/ms-values-hyperpod-lmcache.yaml` — llm-d modelservice overlay wiring LMCache → HyperPod L2.
- `configs/lmcache-config.yaml` — LMCache config content (`ai_toolkit_cache` shm name + `save_unfull_chunk: true`).
- `manifests/lmcache-configmap.yaml` — ConfigMap wrapper for the above.
- `scripts/llmd_l2_probe.py` — repeatable gateway store/restart/replay probe → telemetry JSON.
- `results/progress.md` — stage-by-stage deployment status.
- `lessons.md` — reusable field notes and failure modes.

## Notes (carried from prior HyperPod L2 runs)

- **shm name**: set `sagemaker_hyperpod_shared_memory_name: ai_toolkit_cache` in BOTH `--kv-transfer-config` and the `LMCACHE_CONFIG_FILE` — the adapter reads `config.extra_config`; the default is `shared_memory`.
- **shm permissions**: ai-toolkit creates `/dev/shm/ai_toolkit_cache` as `0600 uid 1000`; vLLM runs `uid 2000` → `chmod 666` via init container or unify GID.
- **env ordering**: `NODE_IP` must be defined before `LMCACHE_REMOTE_URL` for `$(NODE_IP)` substitution.
- **`PYTHONHASHSEED=0`** and **`save_unfull_chunk: true`** are required for a deterministic cross-restart L2 hit (Stage 5).
- **`hostPath: /dev/shm`** (not `emptyDir`) to share POSIX shm with the host daemon.
- **`dnsPolicy: Default`** if Hugging Face pulls fail on the HyperPod node.
