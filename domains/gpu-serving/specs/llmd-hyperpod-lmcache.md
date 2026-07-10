# llm-d on SageMaker HyperPod — Managed L2 via LMCache Connector Recipe Spec

## Status: VALIDATED — PASS (2026-07-10)

Deployed on the reused `dynamo-hyperpod-lmcache` cluster (us-west-2, EKS `qn-sglang-eks-cluster`, K8s
1.32, `ml.g6e.xlarge`). Store→restart→replay proved a managed-L2 hit on a fresh pod: `external_prefix_cache_hits_total 0→742`, 99.9% hit rate, 3375ms→379ms (8.9×). Blueprint:
`domains/gpu-serving/blueprints/llmd-hyperpod-lmcache/` (see `results/` + `lessons.md`). The §3
AS-DEPLOYED note and §3b node-placement section reflect what actually worked; the original DRAFT
assumptions (old llm-d helmfile layout, whole-`/dev/shm` mount) were WRONG and are corrected inline.

## Overview

Produce a clean, reproducible **recipe** for running llm-d on SageMaker HyperPod EKS with llm-d's vLLM replicas using the HyperPod managed tiered-storage daemon (`ai-toolkit`, port 9200) as the KV cache L2 backend through vLLM's `LMCacheConnectorV1`.

This is the llm-d twin of `dynamo-hyperpod-lmcache`. That spec proved the *Dynamo* orchestrator can plug its vLLM worker into HyperPod's managed L2 daemon without the HyperPod Inference Operator, and packaged it as a runnable manifest + store/restart/replay probe + telemetry artifacts. This spec produces the same **outcome** for llm-d:

```yaml
# vLLM args on the llm-d-managed replica
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"sagemaker_hyperpod_shared_memory_name":"ai_toolkit_cache"}}'
# env
LMCACHE_REMOTE_URL=sagemaker-hyperpod://$(NODE_IP):9200
```

The prior `llmd-hyperpod` blueprint already *validated* that this connection works (Stage 4 PASS, 2026-03-31). This spec is not a re-validation — it is the **productization** of that finding into a published recipe that mirrors the dynamo one: one runnable manifest, a repeatable store → restart → replay L2 probe that proves a cross-restart cache hit, and a telemetry artifact schema. The goal is a recipe a customer can copy, not a new research question.

## Integration Thesis

HyperPod's managed cache tier is exposed to inference pods through the LMCache protocol, and it is agnostic to which orchestration layer spawned the vLLM pod:

- L0: vLLM GPU prefix cache
- L1: LMCache in-process CPU memory
- L2: HyperPod managed tiered-storage daemon (`ai-toolkit`) on each GPU node, reachable at `sagemaker-hyperpod://<node-ip>:9200` over POSIX shared memory `ai_toolkit_cache`
- L3: FSx Lustre through a normal POSIX mount

llm-d manages vLLM replicas via Helm/helmfile (`llm-d-infra` + `llm-d-modelservice` + GAIE `inferencepool`), fronted by an Envoy/Istio gateway and the EPP endpoint-picker. The critical difference from Dynamo: **llm-d owns pod scheduling and EPP routing, not the KV connector.** The LMCache connector is set on the vLLM replica exactly as the Inference Operator would inject it, but through llm-d's `ms-*/values.yaml` instead of the IEC CRD. The recipe must prove the L2 hit survives a replica restart and is observable end-to-end through the gateway path, not just on a raw pod.

## Components

### 1. Compute

| Field | Value |
|---|---|
| Platform | SageMaker HyperPod with EKS orchestrator, managed tiered storage enabled |
| Preferred target | `ml.g7.2xlarge` only if the live SageMaker HyperPod API accepts it |
| Approved fallback | `ml.g6e.xlarge`, then `ml.g5.4xlarge` (the SKU the prior `llmd-hyperpod` run used) |
| Current API caveat | The local AWS CLI `sagemaker create-cluster` enum does not list `ml.g7.*`; it does list `ml.g6e.*` and `ml.g5.*`. Availability must be verified live, same as `dynamo-hyperpod-lmcache`. |
| EKS version | ≤ 1.32 required for `RestrictedInstanceGroups` (system-nodes); 1.33 not supported |
| Namespace | `llmd-hp-lmcache` |
| Replicas | 1 for the store/restart/replay proof; 2 same-node only for the cross-pod sharing stage |
| Region | `us-east-2` primary, `us-west-2` fallback (match the running HyperPod inference cluster) |

Enable managed tiered storage when creating or updating the cluster (JSON flag, not Boolean):

```bash
aws sagemaker update-cluster \
  --region <region> \
  --cluster-name <hyperpod-cluster-name> \
  --tiered-storage-config '{"Mode":"Enable","InstanceMemoryAllocationPercentage":20}'
```

Try `ml.g7.2xlarge` first, then fall back to `ml.g6e.xlarge`, then `ml.g5.4xlarge`, recording the reason in `results/deployment-log-<date>.md`. The `ai-toolkit` daemon must be running on the GPU node (port 9200 listening, `/dev/shm/ai_toolkit_cache` present) before this blueprint proceeds.

### 2. Model

| Field | Value |
|---|---|
| Model ID | `Qwen/Qwen3-0.6B` |
| Reason | Non-gated, <2 GB, fast startup, already proven on the llm-d + ai-toolkit L2 path in `llmd-hyperpod`, and matches the model used by `dynamo-hyperpod-lmcache` for a direct cross-orchestrator comparison |
| Precision | BF16 or engine default |
| Tensor parallelism | 1 |
| Max model length | 8192 for smoke; 16384 for cache pressure if stable |
| Engine | vLLM inside the llm-d modelservice |

Intentionally too small for a performance claim. This is a connector + managed-component recipe, not a benchmark.

### 3. Serving Stack

> **AS-DEPLOYED (2026-07-10, upstream `main`).** The llm-d repo was **restructured** since this spec
> was first drafted — the old `llm-d-infra` + `llm-d-modelservice` helmfile / `ms-values.yaml` layout
> (`guides/prereq/gateway-provider`, `guides/inference-scheduling`) **no longer exists**. What actually
> works now is captured in this table and Stage 2 below. Re-verify against `guides/env.sh` before deploy.

| Component | Requirement |
|---|---|
| GAIE CRDs | `kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${GAIE_VERSION}/v1-manifests.yaml` — `GAIE_VERSION=v1.5.0` (from `guides/env.sh`). Installs the GA `inferencepools.inference.networking.k8s.io` CRD. |
| Router | Helm chart `oci://ghcr.io/llm-d/charts/llm-d-router-standalone` (standalone = Envoy proxy + EPP in one Deployment, no separate Istio gateway). `ROUTER_CHART_VERSION=v0.9.0`, EPP image `ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.9.0`. Values: `guides/recipes/router/base.values.yaml` + `guides/tiered-prefix-cache/router/tiered-prefix-cache-cpu.values.yaml`. **The chart auto-provisions the EPP SA + Role/RoleBinding + InferencePool + Envoy ConfigMap** — the RBAC prereq below is satisfied by the chart, not by hand. |
| Model server | Deploy your own Deployment (or `kubectl apply -k` a `guides/tiered-prefix-cache/modelserver/gpu/vllm/lmcache-connector/*` overlay). The upstream overlays hardwire Qwen3-32B / TP2 / H100 / 500Gi and stock `vllm/vllm-openai` (which lacks the SageMaker adapter) — override all of that. The decode pod MUST carry the InferencePool selector label (default `llm-d.ai/guide: tiered-prefix-cache`) or the router won't route to it. targetPort 8000. |
| Worker image | Must include vLLM **and** the SageMaker HyperPod LMCache adapter. **Proven: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1`** (vLLM 0.16.0 + LMCache 0.3.14, has `SageMakerHyperPodConnectorAdapter`). Stock `vllm/vllm-openai` and `llm-d-cuda`/`llm-d-aws` are NOT confirmed to ship the adapter — verify `python -c 'import lmcache'` AND that the adapter opens `ai_toolkit_cache` before trusting them. |
| Connector | vLLM `--kv-transfer-config` with `LMCacheConnectorV1`, `kv_role=kv_both`, and `kv_connector_extra_config.sagemaker_hyperpod_shared_memory_name=ai_toolkit_cache`. Aggregated serving — **not** P/D disaggregation (see Non-Requirements). |
| Routing | GAIE InferencePool (GA API `inference.networking.k8s.io/v1`, `endpointPickerRef`) + EPP ext-proc, both created by the router chart. The `tiered-prefix-cache-cpu` values ship the full scorer set (GPU + CPU prefix-cache producers/scorers, queue, kv-cache-utilization, no-hit-lru). |
| Ingress | The router `tpc-epp` Service (ClusterIP, port 80 → Envoy 8081) is enough for the recipe; port-forward it for the probe. |

EPP operational wiring — **mostly handled by the `llm-d-router-standalone` chart now**, but still verify (silent-failure traps carried from `glm5-llmd/lessons.md`):

- **EPP ServiceAccount RBAC**: the chart creates the SA + Role/RoleBinding with list/watch on `pods` and `inferencepools`. If you deploy EPP by hand instead, it needs list/watch on `pods`, `inferencepools` (`inference.networking.k8s.io`), and the `x-k8s.io` group objects. Without it EPP can't discover endpoints.
- **EPP scheduler config must be non-empty**: an empty plugins list makes ext-proc accept the gRPC stream but never respond → requests hang forever. The `tiered-prefix-cache-cpu.values.yaml` ships a valid non-empty scorer set (GAIE v1.5.0 registers them all). Confirm the effective set in EPP startup logs; don't assume.
- **Standalone router uses Envoy directly, not EnvoyExtensionPolicy** — the `messageTimeout`/`--skip-crds`/EnvoyExtensionPolicy-CRD notes apply only to the Istio/Envoy-Gateway path. In standalone mode the ext-proc timeout is baked into the chart's Envoy ConfigMap; if you switch to the gateway path, re-check `messageTimeout≥30s` + `failOpen: true`.
- **Redis placement** (if a Redis-backed scorer is enabled): HyperPod system nodes are tainted/small; give it a GPU-node toleration or place it on a vanilla node.

Minimum modelservice vLLM args + env (set through `ms-*/values.yaml`, then rendered into the replica):

```yaml
args:
  - "--enable-prefix-caching"
  - "--max-model-len"
  - "8192"
  - "--gpu-memory-utilization"
  - "0.80"
  - "--kv-transfer-config"
  - '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"sagemaker_hyperpod_shared_memory_name":"ai_toolkit_cache"}}'
env:
  - name: NODE_IP
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: LMCACHE_CONFIG_FILE
    value: "/etc/lmcache/config.yaml"    # adapter reads extra_config here; without it, shm name defaults to `shared_memory`
  - name: LMCACHE_USE_EXPERIMENTAL
    value: "True"
  - name: LMCACHE_LOCAL_CPU
    value: "True"                        # L1
  - name: LMCACHE_MAX_LOCAL_CPU_SIZE
    value: "8"
  - name: LMCACHE_CHUNK_SIZE
    value: "256"
  - name: LMCACHE_REMOTE_URL
    value: "sagemaker-hyperpod://$(NODE_IP):9200"   # L2 — MUST come after NODE_IP
  - name: PYTHONHASHSEED
    value: "0"                           # stable LMCache keys across replica restarts (from dynamo-hyperpod-lmcache)
```

`LMCACHE_CONFIG_FILE` must point at a mounted ConfigMap. The adapter reads `extra_config` from this file, and `save_unfull_chunk: true` is what made the dynamo store→restart→replay proof deterministic (`dynamo-hyperpod-lmcache/lessons.md` #56 — without it the cross-restart hit was flaky). Minimum content:

```yaml
# ConfigMap → /etc/lmcache/config.yaml
extra_config:
  sagemaker_hyperpod_shared_memory_name: ai_toolkit_cache
save_unfull_chunk: true   # required for deterministic cross-restart L2 hit (Stage 5)
```

```yaml
volumeMounts:
  - name: lmcache-config
    mountPath: /etc/lmcache
    readOnly: true
volumes:
  - name: lmcache-config
    configMap:
      name: lmcache-config
```

**CRITICAL — mount the shm FILE with `type: File`, NOT the whole `/dev/shm`.** This is the single
biggest failure mode (cost the most time on the 2026-07-10 deploy). The connector opens the segment
with `shm_open(create=False)` — the ai-toolkit daemon owns it. If you mount the whole `/dev/shm` as a
`Directory`, LMCache's `shm_unlink` when a client pod terminates propagates to the host and **destroys
the daemon-owned segment**. The daemon never recreates it, so LMCache's ~30s health-monitor re-init
fails with `Shared memory segment 'ai_toolkit_cache' not found`, marks the backend unhealthy, and
**silently skips ALL store and lookup operations** (`LMCache is unhealthy, skipping store operation`) —
the connector "connects" but nothing ever persists to L2 and every replay misses. Bind-mounting the
single file isolates the client's unlink from the host segment:

```yaml
volumeMounts:
  - name: hp-tiered-cache
    mountPath: /dev/shm/ai_toolkit_cache   # the FILE, not the directory
volumes:
  - name: hp-tiered-cache
    hostPath:
      path: /dev/shm/ai_toolkit_cache
      type: File                           # File, NOT Directory
```

The `type: File` mount **fails to schedule if the segment doesn't already exist**, and a prior
Directory-mounted (or terminating) pod may have already destroyed it. Recovery sequence:

1. `kubectl scale deploy/<decode> --replicas=0` (detach all clients)
2. `kubectl delete pod -n aws-hyperpod -l app=ai-toolkit` (the daemon's `setup` init container recreates the 1 GiB segment; wait for Ready)
3. `kubectl scale deploy/<decode> --replicas=1` (segment now exists → File mount passes and is protected)

Do not run the client as a random uid: the daemon creates the segment `0600` as uid 1000. The proven
`ai-dynamo/vllm-runtime` image runs as uid 1000, so the shm perms line up with no chmod. If your image
runs as a different uid, add an init container to `chmod 666` or unify the GID.

### 3b. Node placement (cross-node pod networking is broken)

On the reused HyperPod cluster, pod-to-pod data-plane traffic from the **vanilla EKS nodes** to the
**HyperPod GPU node** is blocked (control plane / K8s API is fine — EPP still discovers pods, but the
Envoy→vLLM request times out with HTTP 000). The likely cause is the HyperPod node ENI security group
not allowing the EKS node group's pod CIDR. Two options:

- **Recipe workaround (used 2026-07-10):** co-locate the router on the GPU node. The `llm-d-router-standalone`
  chart does not expose `nodeSelector`/`tolerations`, so `kubectl patch` the `*-epp` Deployment to add
  `nodeSelector: {sagemaker.amazonaws.com/instance-group-name: <gpu-group>}` + the `nvidia.com/gpu`
  toleration, and shrink epp+proxy CPU requests (e.g. 150m each) to fit the GPU node's spare CPU
  (g6e.xlarge leaves only ~350m after the model server + system daemons).
- **Production fix:** open the HyperPod GPU node SG to the EKS pod CIDR, or run llm-d entirely on
  HyperPod-managed GPU nodes.

### 4. Optional FSx L3

FSx is optional for the first pass. If enabled, mount it separately from the ai-toolkit L2 path and switch the tier explicitly:

```bash
LMCACHE_REMOTE_URL=file:///mnt/fsx/llmd-hp-lmcache
```

FSx root is `755 root:root` — create `/mnt/fsx/llmd-hp-lmcache` with an init container (vLLM runs as uid 2000). Do not silently replace L2 with FSx and then call L2 validation complete.

### 5. Observability

Required direct checks:

- vLLM `/metrics` on port 8000 exposes `vllm:external_prefix_cache_queries_total` / `vllm:external_prefix_cache_hits_total` and `lmcache:*` counters when the connector is active.
- EPP metrics on port 9090/9002 show routing decisions.
- Replica logs show `LMCacheConnectorV1` initialized, `ai_toolkit_cache` opened, and a connection to `sagemaker-hyperpod://<node-ip>:9200`.
- ai-toolkit daemon logs/metrics show a connection from the llm-d replica pod.
- The e2e probe writes a telemetry artifact under `domains/gpu-serving/blueprints/llmd-hyperpod-lmcache/results/e2e-telemetry-<date>.json`.

AMP/AMG integration is optional for this recipe.

## Validation Stages

### Stage 0 — Carryover Audit

- [ ] Review `domains/gpu-serving/blueprints/llmd-hyperpod/results/progress.md`: llm-d + LMCache → ai-toolkit L2 already PASSED (Stage 4) with the shared-memory-name gotcha, the uid 2000 vs 1000 `chmod 666` gotcha, the `NODE_IP`-before-`LMCACHE_REMOTE_URL` ordering rule, and `hostPath: /dev/shm` requirement. Carry every one of these forward — do not rediscover them.
- [ ] Review `domains/gpu-serving/blueprints/dynamo-hyperpod-lmcache/` README + progress: `LMCACHE_CONFIG_FILE`, `PYTHONHASHSEED=0`, and `save_unfull_chunk: true` were required for a cross-restart L2 hit; `dnsPolicy: Default` was required to resolve Hugging Face on that HyperPod node.
- [ ] Confirm the live SageMaker HyperPod API's accepted instance types in the target region (`ml.g7.*` likely rejected → g6e → g5.4xlarge fallback chain).
- [ ] Confirm EKS ≤ 1.32 for `RestrictedInstanceGroups`.
- [ ] Confirm the worker image ships a LMCache version compatible with the running ai-toolkit daemon (`import lmcache`, compare major.minor). Proven adapter-capable image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1`.
- [ ] **shm mount** (§3): plan to mount the FILE `/dev/shm/ai_toolkit_cache` with `type: File`, NOT the whole `/dev/shm`. Know the daemon-recreate recovery sequence before you start.
- [ ] **Cross-node networking** (§3b): verify pod-to-pod reachability between vanilla EKS nodes and HyperPod GPU nodes (the reused `dynamo-hyperpod-lmcache` cluster has a data-plane gap). Plan router co-location or an SG fix before Stage 2.
- [ ] Record all findings in the deployment log before applying manifests.

### Stage 1 — HyperPod Tiered Storage Discovery

| Check | Method | Pass Criteria |
|---|---|---|
| Tiered storage enabled | `aws sagemaker describe-cluster --cluster-name <name>` | `TieredStorageConfig.Mode` is `Enable` |
| Selected SKU accepted | Live `create-cluster`/`update-cluster` validation | Exact `ml.g7.*`/`ml.g6e.*`/`ml.g5.4xlarge` result recorded |
| CPU architecture | `kubectl exec <pod> -- uname -m` | `x86_64` (LMCache adapter is x86-only) |
| GPU node is HyperPod-managed | `kubectl get nodes --show-labels` | SageMaker/HyperPod instance-group labels present |
| ai-toolkit daemon present | `kubectl get ds -A` / node port check | Daemon running on GPU node; port 9200 listening |
| IPC path present | Debug pod checks `/dev/shm/ai_toolkit_cache` | Directory exists |
| FSx CSI (if L3) | `kubectl get csidriver fsx.csi.aws.com` | Registered |
| GPU health | `nvidia-smi`, ECC/Xid checks | No uncorrected ECC, pending remaps, or Xid errors |

Do not continue if the daemon is absent.

### Stage 2 — llm-d Router Stack (standalone)

Install GAIE CRDs + the standalone router chart. (The old `git clone … guides/prereq/gateway-provider;
helmfile apply` flow is GONE — see the AS-DEPLOYED note in §3.)

```bash
git clone https://github.com/llm-d/llm-d.git && cd llm-d
export REPO_ROOT=$(git rev-parse --show-toplevel); source guides/env.sh   # sets GAIE_VERSION, ROUTER_*
kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${GAIE_VERSION}/v1-manifests.yaml"
helm install <name> oci://ghcr.io/llm-d/charts/llm-d-router-standalone --version ${ROUTER_CHART_VERSION} \
  -f guides/recipes/router/base.values.yaml \
  -f guides/tiered-prefix-cache/router/tiered-prefix-cache-cpu.values.yaml \
  -f <blueprint>/configs/router-overrides.yaml \
  -n <namespace>
```

| Check | Method | Pass Criteria |
|---|---|---|
| InferencePool CRD | `kubectl get crd inferencepools.inference.networking.k8s.io` | Present (from GAIE v1-manifests) |
| Router installed | `helm status <name> -n <ns>` | `deployed` |
| EPP + Envoy pod Running | `kubectl get pods -l llm-d.ai/igw-mode` | `*-epp` pod `2/2` (envoy-proxy + epp) Running |
| EPP SA + RBAC created | `kubectl get sa,role,rolebinding -n <ns>` | Chart created `*-epp` SA + Role/RoleBinding (list/watch pods + inferencepools) |
| InferencePool endpoint-picker wired | `kubectl get inferencepool <name> -o yaml` | `endpointPickerRef` → `*-epp:9002`, `selector.matchLabels` = model-server label |
| EPP scorer set loaded | `kubectl logs <epp-pod> -c epp \| grep 'parsed config'` | Non-empty scorer list (GPU+CPU prefix-cache, queue, kv-cache-utilization) |
| Router service | `kubectl get svc <name>-epp` | ClusterIP, port 80 → 8081 |

### Stage 3 — llm-d vLLM Baseline (L0 only)

Deploy Qwen3-0.6B as a Deployment carrying the InferencePool selector label (default
`llm-d.ai/guide: tiered-prefix-cache`), no LMCache yet. (Written as a plain Deployment, not the
upstream kustomize overlay — the overlay hardwires Qwen3-32B/TP2/H100/stock-vllm.)

| Check | Method | Pass Criteria |
|---|---|---|
| Manifest applies | `kubectl apply -f modelserver-baseline.yaml` | Deployment created |
| Replica scheduled | `kubectl get pods -o wide` | Running on the HyperPod GPU node |
| Health | in-pod `curl localhost:8000/v1/models` | HTTP 200 |
| Chat/completions via router | `curl <router-svc>/v1/completions` (port-forward) | HTTP 200, valid model output through Envoy → EPP → vLLM |
| EPP actually routed it | `kubectl logs <epp-pod> -c epp \| grep 'sent request body response'` | targetModelName set (not a FailOpen bypass) |
| Prefix caching | in-pod `curl localhost:8000/metrics \| grep prefix_cache` | `vllm:prefix_cache_*` and `vllm:external_prefix_cache_*` present |
| InferencePool endpoints | EPP logs `Pod already exists` for the decode pod | EPP discovered the decode pod |

> If the request times out (HTTP 000) but in-pod localhost works, it's the cross-node networking gap —
> see §3b (co-locate the router on the GPU node).

**Gate**: chat completion works end-to-end through the router before touching LMCache.

### Stage 4 — LMCache Connector to HyperPod L2

Patch the modelservice replica with the connector, env, and IPC mount.

| Check | Method | Pass Criteria |
|---|---|---|
| Connector accepted | Replica restarts with `--kv-transfer-config` | No crash loop, no vLLM schema/import error |
| LMCache installed | `kubectl exec -- python -c 'import lmcache'` | Import succeeds; version recorded and compatible with daemon |
| `NODE_IP` expansion | Replica env/logs | `LMCACHE_REMOTE_URL` contains the host IP, not literal `$(NODE_IP)` |
| shm FILE mount | pod spec | `hostPath type: File` on `/dev/shm/ai_toolkit_cache` (NOT Directory — see §3) |
| shm name correct | LMCache logs | Opens `ai_toolkit_cache`, not default `shared_memory` |
| shm permissions | `ls -l /dev/shm/ai_toolkit_cache` | Daemon creates it `0600 uid 1000`; the `ai-dynamo/vllm-runtime` image runs uid 1000 so perms align with no chmod. Different uid → chmod 666 / unify GID. |
| L2 connection stable | Replica logs across ≥2 health pings (~70s) | `Shared memory opened: ai_toolkit_cache (1024.00 MB)`; **no** `Health check failed` / `unhealthy` / `not found` after connect |
| L1 active | LMCache logs/metrics | Local CPU tier active |
| L2 write persists | Send long-prefix request, wait, grep logs | `Stored N out of total N tokens` appears (NOT `LMCache is unhealthy, skipping store operation`); `vllm:external_prefix_cache_queries_total` increments |

**Gate**: replica connects, stays healthy across health pings, AND a `Stored N/N tokens` line proves KV
was written to L2. If you see `skipping store operation`, the shm mount/segment is wrong — go to §3.

### Stage 5 — Store → Restart → Replay L2 Hit Proof (Core Recipe Deliverable)

This is the deliverable that mirrors `dynamo-hyperpod-lmcache`'s passing proof. A single repeatable probe (`scripts/llmd_l2_probe.py`) must prove an L2 hit **survives a replica restart**, so the hit can only come from the managed daemon, not the in-process L0/L1.

Procedure:

1. **Store** — send one cold request through the gateway with a deterministic shared prefix of ≥1024 tokens, `temperature: 0`, `max_tokens: 32`. Capture before/after metrics.
2. **Restart** — `kubectl rollout restart` the modelservice deployment and wait for Ready. This wipes L0 GPU cache and L1 CPU cache; only L2 (ai-toolkit) survives.
3. **Replay** — send the identical request again. Capture metrics + logs.

| Check | Method | Pass Criteria |
|---|---|---|
| Store writes L2 | Daemon metrics/logs after step 1 | Put/write activity from this pod |
| Replica actually restarted | `kubectl get pod` (new pod name/UID) | Fresh pod, cold L0/L1 |
| Replay hits L2 | `vllm:external_prefix_cache_hits_total` delta + LMCache `num_hit_tokens_total` + log `Retrieved N out of N required tokens` | Non-zero external hit after restart |
| Served through gateway | EPP/gateway logs | Replay request routed through the gateway, not curled directly at the pod |
| Telemetry artifact | `results/e2e-telemetry-<date>.json` | Written and referenced from the deployment log |

Telemetry artifact schema (mirror of the dynamo one):

```json
{
  "region": "",
  "selected_instance_type": "ml.g6e.xlarge|ml.g5.4xlarge",
  "fallback_reason": "",
  "model": "Qwen/Qwen3-0.6B",
  "orchestrator": "llm-d",
  "endpoint": "llm-d-gateway",
  "versions": {"llm_d_infra":"","llm_d_modelservice":"","gaie":"","vllm":"","lmcache":"","cuda":""},
  "requests": [
    {"name":"cold_store","http_status":200,"ttft_ms":0,"e2e_ms":0,"output_tokens":0},
    {"name":"warm_replay_after_restart","http_status":200,"ttft_ms":0,"e2e_ms":0,"output_tokens":0}
  ],
  "telemetry": {
    "vllm_prefix_cache": {"before":{},"after":{},"delta":{}},
    "lmcache": {"before":{},"after":{},"delta":{},"log_evidence":[]},
    "hyperpod_l2": {"log_evidence":[],"metric_delta":{}},
    "epp": {"route_log_evidence":[]},
    "gpu": {"memory_used_mb":0,"xid_errors":0}
  },
  "result": "PASS|PARTIAL|FAIL",
  "notes": []
}
```

If the model is too small for a stable TTFT delta, cache-hit telemetry is the source of truth. Do not mark this stage complete from latency alone.

### Stage 6 — Two-Replica Same-Node Cache Sharing (Optional)

Only if the selected SKU has ≥2 usable GPUs. Prove the managed L2 daemon shares KV across llm-d replicas on the same node and that EPP is aware.

| Check | Method | Pass Criteria |
|---|---|---|
| Two replicas on one node | `kubectl get pods -o wide` | Both on same node, one GPU each |
| Both connect to same L2 | Logs | Same node IP + port 9200 |
| Replica A warms prefix | Gateway request | Cache write observed |
| Replica B reads prefix | Route/force to B | Cache hit or TTFT improvement |
| EPP routing awareness | EPP logs | Router prefers the cached replica when cache state is visible |

### Stage 7 — Optional FSx L3

Only after L2 behavior is understood. Switch `LMCACHE_REMOTE_URL` to `file:///mnt/fsx/llmd-hp-lmcache`, prove cross-restart persistence, keep it separate from the L2 path in reporting.

## Success Criteria

| Criteria | Stage | Type |
|---|---|---|
| Live SageMaker HyperPod API accepts the selected SKU (g7 → g6e → g5.4xlarge) | 1 | Critical |
| HyperPod tiered storage enabled; ai-toolkit daemon running on the GPU node | 1 | Critical |
| llm-d gateway + EPP + InferencePool serve Qwen3-0.6B end-to-end via NLB | 3 | Critical |
| llm-d-managed replica initializes `LMCacheConnectorV1`, opens `ai_toolkit_cache`, connects to `sagemaker-hyperpod://<node-ip>:9200` | 4 | Critical |
| ai-toolkit daemon receives KV writes from a non-operator llm-d replica | 4 | Critical |
| Store → restart → replay proves an L2 hit that survives replica restart, served through the gateway | 5 | Critical |
| Telemetry artifact proves serving, cache activity, L2 daemon evidence, and EPP routing | 5 | Critical |
| Two-replica same-node cache sharing works or is skipped for one-GPU SKU with reason | 6 | Important |
| FSx L3 not confused with managed L2 in reporting | 7 | Important |

## Known Risks and Blockers

| Risk | Severity | Detail | Mitigation |
|---|---|---|---|
| ai-toolkit daemon rejects non-operator pods | High | Daemon may key off operator-injected labels/annotations. | Prior `llmd-hyperpod` run showed it does NOT reject (Stage 4 PASS); re-confirm with a bare vLLM+LMCache pod first if a new daemon version is running. |
| shm name mismatch | High | LMCache defaults to `shared_memory`; ai-toolkit uses `ai_toolkit_cache`. | Set via BOTH `kv_connector_extra_config` AND `LMCACHE_CONFIG_FILE` (the adapter reads `config.extra_config`). |
| **Whole `/dev/shm` (Directory) mount silently kills stores** | **CRITICAL** | The #1 time sink on the 2026-07-10 deploy. Directory hostPath mount of `/dev/shm` lets a terminating client pod `shm_unlink` the daemon-owned segment; the daemon (`create=False`) never recreates it, so LMCache goes unhealthy and **silently skips ALL store/lookup ops** (`LMCache is unhealthy, skipping store operation`) — connects fine, persists nothing, every replay misses. | Mount ONLY the FILE `/dev/shm/ai_toolkit_cache` with `hostPath type: File`. If already poisoned: scale clients to 0 → `kubectl delete pod -n aws-hyperpod -l app=ai-toolkit` (recreates the segment) → scale back to 1. See §3. |
| shm permissions | High | Daemon creates shm `0600` as uid 1000. | Run the client as uid 1000 (`ai-dynamo/vllm-runtime` does) so perms align with no chmod; other images → `chmod 666` init container or unify GID. |
| No cross-restart hit without stable keys | High | LMCache keys drift across pod restarts. | `PYTHONHASHSEED=0` (carried from dynamo run) — required for Stage 5 replay to hit. |
| Cross-node pod networking blocked | High | On the reused cluster, HyperPod GPU node ENI SG does not allow pod-CIDR traffic from vanilla EKS node group → router→vLLM times out (HTTP 000) though in-pod localhost works and EPP still discovers the pod. | Co-locate the router on the GPU node (§3b `kubectl patch` + shrink epp/proxy CPU), or open the HyperPod node SG to the EKS pod CIDR. |
| Image lacks the SageMaker adapter | High | Stock `vllm/vllm-openai` and `llm-d-cuda`/`llm-d-aws` are NOT confirmed to ship `SageMakerHyperPodConnectorAdapter`. | **Proven: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1`.** For any other image, verify `import lmcache` AND that it actually opens `ai_toolkit_cache` before trusting it. |
| `NODE_IP` env ordering | Medium | `LMCACHE_REMOTE_URL` must expand after `NODE_IP`. | Define `NODE_IP` before dependent env vars; verify runtime env. |
| EKS 1.33 breaks RestrictedInstanceGroups | Medium | System nodes won't schedule. | Use EKS ≤ 1.32. |
| DNS resolution on HyperPod node | Medium | HF download may fail without node resolver. | `dnsPolicy: Default` (carried from dynamo run) if HF pulls fail. |
| LMCache sync I/O regression | Medium | LMCache can serialize the vLLM scheduler under cache pressure. | Keep concurrency low; pressure-test only after functionality passes. |
| EPP ext-proc silent hang | Medium | Empty EPP scheduler plugins list → ext-proc accepts the gRPC stream but never responds; requests hang forever (glm5-llmd). | Ship a non-empty, version-valid scorer config; `failOpen: true` degrades to bypass rather than hang. |
| EnvoyExtensionPolicy silently ignored | Medium | CRD omitted by `--skip-crds`, or controller not restarted after CRD install (glm5-llmd). | Install the CRD from chart CRDs and `kubectl rollout restart` the Envoy Gateway controller. |
| EPP RBAC missing | Medium | EPP SA without list/watch on pods + inferencepools crash-loops or discovers no endpoints. | Bind the ClusterRole from Stage 2 before deploying EPP. |

## Non-Requirements

- Production throughput/latency benchmarking (this is a recipe + smoke proof).
- **P/D disaggregation with NIXL** — the AWS blog ("Introducing disaggregated inference on AWS powered by llm-d") covers the `NixlConnector` + EFA/`LIBFABRIC` disagg path with `ms-pd/values.yaml`; that is a *separate* recipe. This spec is aggregated serving with the LMCache→L2 connector, mirroring `dynamo-hyperpod-lmcache`.
- Wide-EP / multi-node inference.
- Replacing the HyperPod Inference Operator's managed serving path.
- Managed llm-d operator integration (Operator v2 proposal — out of scope).
- Serving MLA/NSA models; this uses a simple standard-attention model on purpose.

## Deployment Notes

Before deployment, run:

```bash
export AWS_REGION=us-east-2
mdc get Qwen/Qwen3-0.6B --engine vllm
mdc prs Qwen/Qwen3-0.6B
gpu-infra card g7 || true
gpu-infra card g6e || true
```

If the primary region cannot provide HyperPod capacity, set `AWS_REGION=us-west-2` and record the fallback reason. If SageMaker rejects `ml.g7.*`, fall back to `ml.g6e.xlarge`, then `ml.g5.4xlarge`, recording the reason. Record exact llm-d chart versions, vLLM, LMCache, CUDA, region, selected instance type, GPU model/architecture, and ai-toolkit daemon version in `domains/gpu-serving/blueprints/llmd-hyperpod-lmcache/lessons.md`.

## References

- Companion recipe (Dynamo twin): `domains/gpu-serving/specs/dynamo-hyperpod-lmcache.md` + `domains/gpu-serving/blueprints/dynamo-hyperpod-lmcache/`
- Prior llm-d + ai-toolkit L2 validation (proven config + gotchas): `domains/gpu-serving/specs/llmd-hyperpod.md` + `domains/gpu-serving/blueprints/llmd-hyperpod/results/progress.md`
- llm-d on vanilla EKS (EPP, InferencePool, Envoy): `domains/gpu-serving/blueprints/glm5-llmd/lessons.md`
- AWS Blog (disaggregation reference, `llm-d-aws` image, helmfile/gateway steps): https://aws.amazon.com/blogs/machine-learning/introducing-disaggregated-inference-on-aws-powered-by-llm-d/
- llm-d GitHub (helmfile guides, gateway provider): https://github.com/llm-d/llm-d
- AWS SageMaker docs: `managed-tier-checkpointing-setup.html` — `--tiered-storage-config`
- Local architecture note: `/Users/phi/Documents/workbench/aws/AWS_WorkDay/SageMaker Hyperpod - SMHP/HyperPod-Inference-Architecture.md`

> **Note**: Operational artifacts belong in `domains/gpu-serving/blueprints/llmd-hyperpod-lmcache/lessons.md` and `.../results/`.
