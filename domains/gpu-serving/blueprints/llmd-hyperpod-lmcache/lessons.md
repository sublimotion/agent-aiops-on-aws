---
model: "Qwen/Qwen3-0.6B"
engine: "llmd"
hardware: "ml.g6e.xlarge"
gpu_arch: "sm_89"
deployment_date: "2026-07-10"

outcome: "success"
failure_categories: []

cards_used:
  mdc: []
  gpu_infra: []

card_helped: null

benchmark:
  throughput_toks_s: null
  ttft_p50_ms: null
  ttft_p99_ms: null
  concurrent_users: null
  gpu_util_pct: null

ralph_iterations: null


learn_commands:
  - 'mdc learn "Qwen/Qwen3-0.6B" vllm "SageMaker HyperPod ai-toolkit L2 via LMCacheConnectorV1: requires type:File shm mount (/dev/shm/ai_toolkit_cache), not Directory — Directory mount lets terminating client shm_unlink poison daemon segment. Set sagemaker_hyperpod_shared_memory_name in BOTH kv_connector_extra_config AND LMCACHE_CONFIG_FILE extra_config. PYTHONHASHSEED=0 + save_unfull_chunk:true for deterministic cross-restart hits. Worker image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1 (vLLM 0.16.0 + LMCache 0.3.14, has SageMakerHyperPodConnectorAdapter)."'
  - 'mdc learn "Qwen/Qwen3-0.6B" llmd "llm-d on SageMaker HyperPod with ai-toolkit L2 via vLLM LMCacheConnectorV1: use GAIE v1.5.0 + llm-d-router-standalone chart (tiered-prefix-cache path). Cross-node pod networking may be blocked (HyperPod GPU node ↔ vanilla EKS nodes) — co-locate router on GPU node if data plane fails."'
  - 'gpu-infra learn -c platform "SageMaker HyperPod ai-toolkit tiered-storage daemon: mount the shm segment FILE (/dev/shm/ai_toolkit_cache) with type:File, NOT whole /dev/shm as Directory. Directory mount lets a terminating LMCache client shm_unlink the daemon-owned segment; daemon (create=False) never recreates it → health check fails → all store/lookup silently skipped. Recovery: scale clients to 0 → delete daemon pod (init container recreates segment) → scale clients back. Worker must run uid 1000 or chmod 666. Observed: dynamo-hyperpod-lmcache + llmd-hyperpod-lmcache."'
---

# llm-d HyperPod LMCache — Lessons

> Field notes captured during deployment. The compound-learner adds YAML
> frontmatter (model, engine, hardware, outcome, failure_categories,
> mdc_learn_commands, gpu_infra_learn_commands) as the final step.

## Carried Forward (do not rediscover)

From `llmd-hyperpod` and `dynamo-hyperpod-lmcache` — these are known-good and
already encoded in the configs/manifests:

1. **shm name mismatch** — LMCache defaults to `shared_memory`; ai-toolkit uses
   `ai_toolkit_cache`. Set it in BOTH `--kv-transfer-config` extra_config AND
   `LMCACHE_CONFIG_FILE` (the adapter reads `config.extra_config`).
2. **shm permissions** — ai-toolkit creates the shm file `0600` as uid 1000;
   vLLM runs uid 2000. `chmod 666` via init container (done in the overlay) or
   unify GID in production.
3. **env ordering** — `NODE_IP` must precede `LMCACHE_REMOTE_URL` for `$(NODE_IP)`
   substitution.
4. **`PYTHONHASHSEED=0` + `save_unfull_chunk: true`** — required for a
   deterministic cross-restart L2 hit (Stage 5).
5. **`hostPath: /dev/shm`** (not emptyDir) to share POSIX shm with the daemon.
6. **`dnsPolicy: Default`** if HF pulls fail on the HyperPod node.
7. **EKS ≤ 1.32** for RestrictedInstanceGroups (system nodes).
8. **EPP operational wiring** (from glm5-llmd): SA RBAC on pods+inferencepools,
   EnvoyExtensionPolicy CRD install + controller restart, `messageTimeout>=30s`
   + `failOpen: true`, non-empty scorer config.

## New Lessons

### 2026-07-10 — Upstream llm-d repo restructured (spec/overlay assumptions stale)

The spec and the pre-built `configs/ms-values-hyperpod-lmcache.yaml` assumed the old llm-d
helmfile layout (`guides/prereq/gateway-provider`, `guides/inference-scheduling`,
`ms-inference-scheduling/values.yaml`, image `ghcr.io/llm-d/llm-d-cuda:v0.5.1`). That layout
is **gone** on `main`. Current structure:

- **GAIE CRDs**: `kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.5.0/v1-manifests.yaml` (`GAIE_VERSION=v1.5.0`).
- **Router**: Helm chart `oci://ghcr.io/llm-d/charts/llm-d-router-standalone` (or `-router-gateway`), `ROUTER_CHART_VERSION=v0.9.0`, EPP image `ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.9.0`. Values from `guides/recipes/router/base.values.yaml` + a guide-specific values file.
- **Model server**: `kubectl apply -k` kustomize overlays under `guides/tiered-prefix-cache/modelserver/gpu/vllm/...`. There is a dedicated **LMCache** path: `.../lmcache-connector/{cpu,fs}/{base,gke}/`.
- All versions centralized in `guides/env.sh`.

Upstream defaults target Qwen3-32B / TP2 / H100 / 500Gi mem — must override for Qwen3-0.6B / TP1 / g6e.

### 2026-07-10 — g6e.xlarge is CPU/mem constrained

`hyperpod-i-0ebd814e4a113cdce` allocatable: **cpu=1930m**, mem≈29Gi, gpu=1. vLLM CPU requests
must be tiny (dynamo run used `cpu: 500m`). The 8-CPU/400-500Gi upstream requests will never schedule.
Router/EPP must avoid the tainted system nodes (`node-role=system:NoSchedule`) — they land on the
two vanilla EKS nodes (`ip-10-2-10-75`, `ip-10-2-40-177`).

### 2026-07-10 — Cross-node pod networking is BROKEN between HyperPod and vanilla EKS nodes (P0)

Symptom: router pod on a vanilla EKS node returned HTTP 000 (timeout) to the decode pod on the
HyperPod GPU node. Isolation proved it's not the router:
- in-pod `localhost:8000` completion → HTTP 200, 50ms (engine healthy)
- same-node pod → decode pod IP:8000 → HTTP 200, **1ms**
- cross-node (vanilla → HyperPod) → HTTP 000, times out
- EPP (control plane, via K8s API) discovered the decode pod fine — only the DATA plane across the
  node boundary is blocked. Likely the HyperPod node ENI security group does not allow pod-CIDR
  traffic from the vanilla EKS node group (the two node groups were created separately; SG-for-pods
  gap). The dynamo run never hit this because it ran frontend+worker in ONE pod on the GPU node.

**Fix for smoke test (no SG changes to shared infra):** co-locate the router on the GPU node.
The chart (`llm-d-router-standalone` v0.9.0) does not expose nodeSelector/tolerations, so patch the
`tpc-epp` Deployment directly:
```
kubectl patch deploy tpc-epp -n llm-d-hp-lmcache --type merge -p '{"spec":{"template":{"spec":{
  "nodeSelector":{"sagemaker.amazonaws.com/instance-group-name":"g6e-workers"},
  "tolerations":[{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}]}}}}'
```
GPU node has only ~350m CPU free after decode(500m)+system daemons, so also shrink router
epp+proxy CPU requests to 150m each (container order: envoy-proxy=0, epp=1). At conc≈1 this is fine.
After co-location: POST /v1/completions via router → **HTTP 200, 300ms**, EPP routed it (not FailOpen
bypass), full GPU+CPU prefix-cache scorer set active. Stage 3 PASS.
(Production fix would be to open the HyperPod node SG to the EKS pod CIDR, or run llm-d entirely on
HyperPod-managed GPU nodes.)

### 2026-07-10 — MUST mount the shm FILE, not the whole /dev/shm — else stores are silently skipped (P0)

This cost the most time. Symptom: LMCache connects to `sagemaker-hyperpod://<ip>:9200` and opens
`ai_toolkit_cache` at startup, requests succeed, but NOTHING is ever stored to L2 and every replay
misses (`need to load: 0`, `external_prefix_cache_hits_total=0`). The tell is:
```
LMCache WARNING: LMCache is unhealthy, skipping store operation
```

Mechanism:
- The SageMaker HyperPod connector opens the segment with `shm_open(..., create=False)` — it does NOT
  create it; the ai-toolkit daemon owns it.
- LMCache's RemoteBackend health monitor pings every 30s. On each ping it nulls + re-inits the
  connector (`Connector is None, re-initializing`), which re-runs `shm_open(create=False)`.
- If the segment isn't present at that instant → `Shared memory segment 'ai_toolkit_cache' not found`
  → health check fails → backend marked unhealthy → **all store AND lookup ops skipped** thereafter.
- Why the segment goes missing: with a **Directory** hostPath mount of `/dev/shm`, LMCache's
  `shm_unlink` when a client pod terminates propagates to the HOST segment and destroys the
  daemon-owned `ai_toolkit_cache`. The daemon (create=False on its clients) never recreates it, so it
  stays gone and every subsequent pod is poisoned.

**Fix (matches the dynamo run):** mount ONLY the segment file, not the directory:
```yaml
volumeMounts: [{name: hp-tiered-cache, mountPath: /dev/shm/ai_toolkit_cache}]
volumes:
  - name: hp-tiered-cache
    hostPath: {path: /dev/shm/ai_toolkit_cache, type: File}
```
A `type: File` bind mount of the single file isolates the client's `shm_unlink` from the host segment.

**Prerequisite / recovery sequence** (the `type: File` mount FAILS if the segment doesn't already
exist, and a prior Directory-mounted pod may have already destroyed it):
1. `kubectl scale deploy/llmd-lmcache-decode -n llm-d-hp-lmcache --replicas=0` (detach all clients)
2. `kubectl delete pod -n aws-hyperpod -l app=ai-toolkit` (daemon `setup` init container recreates the
   1GiB segment; wait for Ready)
3. `kubectl scale deploy/llmd-lmcache-decode --replicas=1` (segment now exists → File mount passes)

After this: `Shared memory opened: ai_toolkit_cache (1024.00 MB)`, zero health failures across ping
cycles, and stores land: `Stored 804 out of total 804 tokens ... throughput: 9.62 GB/s`. Stage 4 PASS.
