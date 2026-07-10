# NVIDIA Dynamo on SageMaker HyperPod - Managed L2 via LMCache Connector Spec

## Status: DRAFT (2026-07-09)

## Overview

Validate a small NVIDIA Dynamo deployment on SageMaker HyperPod EKS that reuses the HyperPod managed tiered storage component as the KV cache backend through vLLM's `LMCacheConnectorV1`.

This is the "adapter" path that the earlier `dynamo-hyperpod` spec did not cover. The existing spec validates Dynamo's own KVBM tiers against FSx. This spec validates whether Dynamo's vLLM backend can run with its documented LMCache integration and point that LMCache backend at the same HyperPod managed daemon used by the HyperPod Inference Operator:

```bash
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --connector lmcache
LMCACHE_REMOTE_URL=sagemaker-hyperpod://$(NODE_IP):9200
```

The goal is not a production benchmark. The goal is to prove or disprove that Dynamo orchestration can plug into HyperPod's managed L2 daemon path without using the HyperPod Inference Operator as the deployer.

## Integration Thesis

HyperPod's managed cache tier is exposed to inference pods through the LMCache protocol:

- L0: vLLM GPU prefix cache
- L1: LMCache in-process CPU memory
- L2: HyperPod managed tiered storage daemon on each GPU node, reachable at `sagemaker-hyperpod://<node-ip>:9200`
- L3: FSx Lustre through a normal POSIX mount

Dynamo's native KV path uses `DynamoConnector`/KVBM. HyperPod's managed L2 uses `LMCacheConnectorV1`. This spec treats direct KVBM-to-LMCache compatibility as the critical risk. The deployer must validate the actual connector path from logs and metrics rather than assuming that setting KVBM env vars is enough.

## Components

### 1. Compute

| Field | Value |
|---|---|
| Platform | SageMaker HyperPod with EKS orchestrator |
| Instance group | HyperPod GPU instance group with managed tiered storage enabled |
| Preferred target | `ml.g7.2xlarge` only if the live SageMaker HyperPod API accepts it |
| Approved fallback | `ml.g6e.xlarge` if `ml.g7.*` is unsupported or unavailable in both target regions |
| Current API caveat | Local AWS CLI `sagemaker create-cluster help` does not list `ml.g7.*` as an allowed HyperPod instance type, but does list `ml.g6e.*`; availability must be verified live |
| GPU | Single-GPU G7 or G6e family node; record exact GPU model, memory, and SM architecture during Stage 1 |
| Namespace | `dynamo-hp-lmcache` |
| Replicas | 1 for initial smoke, 2 only after L2 connection succeeds |
| Region | `us-east-2` primary, `us-west-2` fallback |

Enable the managed tiered storage component when creating or updating the HyperPod cluster. The AWS CLI flag is JSON, not a Boolean:

```bash
aws sagemaker update-cluster \
  --region us-east-2 \
  --cluster-name <hyperpod-cluster-name> \
  --tiered-storage-config '{"Mode":"Enable","InstanceMemoryAllocationPercentage":20}'
```

For a new cluster, pass the same `--tiered-storage-config` block to `aws sagemaker create-cluster`. Try `ml.g7.2xlarge` in `us-east-2` first, then `us-west-2`. If the live SageMaker API rejects `ml.g7.*` as unsupported or cannot place capacity in both regions, fall back to `ml.g6e.xlarge` in `us-east-2`, then `us-west-2`, and record the reason in `results/deployment-log-<date>.md`. Do not silently substitute any family other than `g6e`; another family requires an explicit spec update. The managed tiered storage daemon must be present on the HyperPod GPU node before this blueprint proceeds.

### 2. Model

| Field | Value |
|---|---|
| Model ID | `Qwen/Qwen3-0.6B` |
| Reason | Non-gated, small, already used by `dynamo-hyperpod`, fast startup |
| Precision | BF16 or engine default |
| Tensor parallelism | 1 |
| Max model length | 8192 for smoke; 16384 for cache pressure if stable |
| Engine | vLLM inside Dynamo worker/runtime |

This model is intentionally too small for a performance claim. It is a connector and managed-component validation target.

### 3. Serving Stack

| Component | Requirement |
|---|---|
| Dynamo | Full Dynamo graph if available: frontend/router/planner/workers. The fallback single-worker mode is allowed only for Stage 2 smoke and does not validate KVBM. |
| Worker image | Must include vLLM and LMCache. If `nvcr.io/nvidia/ai-dynamo/vllm-runtime:<tag>` lacks LMCache, build or overlay an image that installs the LMCache version compatible with the HyperPod daemon. |
| Connector | Prefer Dynamo's documented `--connector lmcache` path, which configures `LMCacheConnectorV1` with `kv_role=kv_both` for aggregated serving. Raw `--kv-transfer-config` is fallback only. |
| Service discovery | Dynamo default etcd or current Dynamo operator mechanism |
| Ingress | ClusterIP or internal NLB is enough for smoke tests |

Minimum Dynamo-native worker command:

```bash
DYN_SYSTEM_PORT=8081 \
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --connector lmcache \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80
```

Fallback raw vLLM command, only for isolating whether the HyperPod daemon accepts LMCache traffic from a non-operator pod:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-prefix-caching \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

Minimum LMCache/HyperPod env:

```yaml
env:
  - name: NODE_IP
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: LMCACHE_USE_EXPERIMENTAL
    value: "True"
  - name: LMCACHE_LOCAL_CPU
    value: "True"
  - name: LMCACHE_MAX_LOCAL_CPU_SIZE
    value: "8"
  - name: LMCACHE_CHUNK_SIZE
    value: "256"
  - name: LMCACHE_REMOTE_URL
    value: "sagemaker-hyperpod://$(NODE_IP):9200"
  - name: DYN_SYSTEM_PORT
    value: "8081"
```

Mount only the daemon IPC path:

```yaml
volumeMounts:
  - name: hp-tiered-cache
    mountPath: /dev/shm/ai_toolkit_cache
volumes:
  - name: hp-tiered-cache
    hostPath:
      path: /dev/shm/ai_toolkit_cache
      type: Directory
```

Do not mount host `/dev/shm` wholesale.

### 4. Optional FSx L3

FSx is optional for the first pass. If enabled, mount it separately from the HyperPod L2 daemon path:

```bash
LMCACHE_REMOTE_URL=file:///mnt/fsx/dynamo-hp-lmcache
```

or use the current LMCache multi-tier configuration format if the pinned LMCache build supports both the `sagemaker-hyperpod://` tier and a file tier. The deployer must not silently replace L2 with FSx and then call the L2 validation complete.

### 5. Observability

Required direct checks:

- Dynamo `/metrics` on `DYN_SYSTEM_PORT` exposes Dynamo, vLLM, and LMCache counters when `--connector lmcache` is used.
- vLLM `/metrics` exposes prefix cache counters if available in the selected runtime.
- Worker logs show `LMCacheConnectorV1` initialized.
- Worker logs show a connection attempt to `sagemaker-hyperpod://<node-ip>:9200`.
- HyperPod tiered storage daemon logs or metrics show a connection from the worker pod.
- Dynamo logs show whether the worker is participating in a Dynamo graph or is only standalone vLLM.
- The e2e smoke test writes a telemetry artifact under `domains/gpu-serving/blueprints/dynamo-hyperpod-lmcache/results/e2e-telemetry-<date>.json`.

AMP/AMG integration is optional for this smoke blueprint.

## Validation Stages

### Stage 0 - Carryover Audit

- [ ] Review `domains/gpu-serving/blueprints/dynamo-hyperpod/results/progress.md`: the previous `vllm-runtime:1.0.1` worker served Qwen3-0.6B, but KVBM env vars were not consumed because the full Dynamo stack was not active.
- [ ] Review `domains/gpu-serving/specs/llmd-hyperpod.md`: the known working HyperPod L2 path is LMCacheConnectorV1 plus `LMCACHE_REMOTE_URL=sagemaker-hyperpod://$(NODE_IP):9200`.
- [ ] Review NVIDIA Dynamo v0.8.1 LMCache integration docs: aggregated serving uses `--connector lmcache`; disaggregated serving uses LMCache only on the prefill worker through `PdConnector`/`MultiConnector` with NIXL.
- [ ] Verify whether `ml.g7.*` is accepted by the live SageMaker HyperPod API in `us-east-2` or `us-west-2`. The local AWS CLI model does not list `ml.g7.*`; if the service rejects it or capacity is unavailable, fall back to `ml.g6e.xlarge`.
- [ ] Review the current GPU card for the resolved SKU (`ml.g7.*` or `ml.g6e.xlarge`). Do not carry over hardware assumptions from other instance families or variants unless Stage 1 confirms the same properties.
- [ ] Record all applicable findings in the deployment log before applying manifests.

### Stage 1 - HyperPod Tiered Storage Discovery

| Check | Method | Pass Criteria |
|---|---|---|
| Tiered storage enabled | `aws sagemaker describe-cluster --cluster-name <name>` | `TieredStorageConfig.Mode` is `Enable` |
| Preferred G7 SKU checked | Live `create-cluster` or `update-cluster` validation in `us-east-2`, then `us-west-2` | Exact `ml.g7.*` result is recorded: accepted, unsupported, or capacity failure |
| Fallback G6e checked if needed | Live validation in `us-east-2`, then `us-west-2` | `ml.g6e.xlarge` is accepted and selected if G7 is unavailable |
| Selected SKU capacity available | Capacity/quota check after API acceptance | Selected SKU has usable capacity or the failed capacity reason is recorded |
| CPU architecture | `kubectl exec <pod> -- uname -m` | `x86_64`; Dynamo LMCache integration does not support ARM64 |
| GPU node is HyperPod-managed | `kubectl get nodes --show-labels` | Node has SageMaker/HyperPod instance-group labels |
| L2 daemon present | `kubectl get pods -A` and node-level port check | Daemon is running on the selected G7/G6e GPU node and port 9200 is listening |
| IPC path present | Debug pod on GPU node checks `/dev/shm/ai_toolkit_cache` | Directory exists |
| System add-ons healthy | `kubectl get pods -A` | HyperPod operator/system pods are Running or known non-blocking |
| GPU health | `nvidia-smi`, ECC/Xid checks | No uncorrected ECC, pending remaps, or Xid errors |
| Region pinned | Deployment log | Region is `us-east-2` or `us-west-2`; if fallback was used, reason is recorded |

Do not continue if the daemon is absent. Enabling `--tiered-storage-config` is a hard prerequisite for this spec.

### Stage 2 - Baseline Dynamo Worker on Selected G7/G6e SKU

Deploy a single worker without LMCache first.

| Check | Method | Pass Criteria |
|---|---|---|
| Image pulls on selected SKU | `kubectl get pod` / events | No `ImagePullBackOff` |
| CUDA works | `kubectl exec <pod> -- nvidia-smi` | GPU visible; exact GPU name and memory recorded |
| vLLM starts | Worker logs | Model loads, no CUDA/runtime mismatch |
| Health endpoint | `curl <pod-ip>:8000/health` | HTTP 200 |
| Completion endpoint | `/v1/chat/completions` | Valid response |
| Metrics endpoint | `/metrics` | vLLM metrics present |

This stage only validates the selected GPU runtime. It does not validate managed L2 or KVBM.

### Stage 3 - LMCache Connector to HyperPod L2

Add the LMCache connector, env vars, and `/dev/shm/ai_toolkit_cache` mount.

| Check | Method | Pass Criteria |
|---|---|---|
| Connector accepted | Worker starts with `--connector lmcache` | Dynamo configures `LMCacheConnectorV1`; no vLLM schema or import error |
| LMCache installed | `python -c 'import lmcache'` | Import succeeds; version recorded |
| NODE_IP expansion | Worker env/logs | `LMCACHE_REMOTE_URL` contains the host IP, not literal `$(NODE_IP)` |
| L2 connection | Worker logs and daemon logs | Connection to `sagemaker-hyperpod://<node-ip>:9200` succeeds |
| L1 active | LMCache logs/metrics | Local CPU tier active |
| Dynamo metrics endpoint | `curl <worker>:8081/metrics` | Dynamo/vLLM/LMCache metrics are exposed |
| L2 write | Send repeated long-prefix request | Daemon metrics/logs show put/write activity |
| L2 read | Repeat the prefix | LMCache/vLLM metrics show a hit or TTFT improves versus cold request |

If this stage fails because the daemon rejects non-operator pods, stop the L2 path and record the rejection. Do not fall back to FSx and report L2 success.

### Stage 4 - Dynamo Graph Participation

Validate that the LMCache-configured worker is actually part of a Dynamo-managed graph.

| Check | Method | Pass Criteria |
|---|---|---|
| Dynamo frontend/router running | `kubectl get pods` | Frontend/router pods Running |
| Worker registered | etcd or Dynamo control-plane query | Worker endpoint visible |
| Request path uses Dynamo | Send request through Dynamo frontend | Request reaches worker and returns output |
| Router logs include worker choice | Dynamo logs | Routing decision visible |
| L2 still active through Dynamo path | Repeat Stage 3 cache test via frontend | L2 hit/write evidence still present |

This stage separates "standalone vLLM with LMCache" from "Dynamo deployment using HyperPod L2".

### Stage 5 - End-to-End Smoke Test and Telemetry

Run a mechanical test through the same endpoint users will call. The test must prove serving, repeated-prefix behavior, connector telemetry, and daemon evidence in one artifact.

Minimum request shape:

- Use `/v1/chat/completions` through the Dynamo frontend service, not directly against the worker.
- Send one cold request with a deterministic shared prefix of at least 2048 tokens.
- Send the same request again with the same model, prefix, and sampling settings.
- Use `temperature: 0`, `max_tokens: 32`, and a fixed prompt template.
- Capture HTTP status, response body validity, TTFT, total latency, output token count, and error text if any.

Telemetry to capture before and after the two requests:

| Signal | Method | Required Evidence |
|---|---|---|
| vLLM prefix cache counters | `curl <worker>:8000/metrics` or `curl <worker>:8081/metrics` | Counter delta for prefix cache queries/hits, or explicit note that this runtime exposes different names |
| LMCache connector activity | Worker logs and `curl <worker>:8081/metrics` | `LMCacheConnectorV1` initialized plus get/put/hit/miss evidence |
| HyperPod L2 daemon activity | Daemon logs or daemon metrics | Connection plus write/read evidence from this worker pod or node IP |
| Dynamo request path | Frontend/router logs | Request routed through Dynamo frontend to the worker |
| GPU sanity | `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` | GPU memory nonzero during/after request, no Xid errors |
| Kubernetes health | `kubectl get pod -n dynamo-hp-lmcache -o wide` | Worker and frontend pods Running/Ready |

The artifact schema should be:

```json
{
  "region": "us-east-2",
  "preferred_instance_type": "ml.g7.2xlarge",
  "selected_instance_type": "ml.g7.2xlarge|ml.g6e.xlarge",
  "fallback_reason": "",
  "instance_type_api_accepted": true,
  "model": "Qwen/Qwen3-0.6B",
  "endpoint": "dynamo-frontend",
  "versions": {
    "dynamo": "",
    "vllm": "",
    "lmcache": "",
    "cuda": "",
    "hyperpod_inference_operator": ""
  },
  "requests": [
    {
      "name": "cold",
      "http_status": 200,
      "ttft_ms": 0,
      "e2e_ms": 0,
      "output_tokens": 0
    },
    {
      "name": "warm_same_prefix",
      "http_status": 200,
      "ttft_ms": 0,
      "e2e_ms": 0,
      "output_tokens": 0
    }
  ],
  "telemetry": {
    "vllm_prefix_cache": {"before": {}, "after": {}, "delta": {}},
    "lmcache": {"before": {}, "after": {}, "delta": {}, "log_evidence": []},
    "hyperpod_l2": {"log_evidence": [], "metric_delta": {}},
    "dynamo": {"route_log_evidence": []},
    "gpu": {"memory_used_mb": 0, "xid_errors": 0}
  },
  "result": "PASS|PARTIAL|FAIL",
  "notes": []
}
```

Pass criteria:

- Both requests return HTTP 200 with valid model output.
- Warm request TTFT is lower than cold request, or cache-hit counters/logs prove a hit even if TTFT is noisy.
- HyperPod L2 daemon evidence shows this pod connected and performed cache activity.
- Dynamo frontend/router logs prove the request path did not bypass Dynamo.
- The artifact is saved and referenced from the deployment log.

If the model is too small to produce a stable TTFT delta, cache-hit telemetry is the source of truth. Do not mark this stage complete from latency alone.

### Stage 6 - KVBM Compatibility Gate

This is the core unknown. Determine whether the Dynamo KVBM path can use the LMCache connector directly, needs an adapter, or must remain separate.

| Check | Method | Pass Criteria |
|---|---|---|
| KVBM active | Dynamo/KVBM logs or metrics | KVBM component initialized, not just env vars set |
| Connector ownership clear | Logs/config inspection | Worker is using either `LMCacheConnectorV1` or `DynamoConnector`, not both in conflict |
| Cache events visible to Dynamo | Router/KVBM/indexer logs | Dynamo sees cache state for routing decisions |
| Managed L2 still receives KV | L2 daemon logs/metrics | HyperPod daemon receives writes/reads while Dynamo graph is active |
| No scheduler regression | Small concurrency smoke | No request serialization, timeout spike, or crash under 4 concurrent repeated-prefix requests |

Acceptable outcomes:

- **PASS**: Dynamo graph, KVBM/cache index, and HyperPod L2 all participate with LMCache connector evidence.
- **PARTIAL**: Dynamo graph can run workers with `LMCacheConnectorV1`, but KVBM is bypassed. This is still useful, but it is not "KVBM using managed L2".
- **FAIL**: Dynamo KVBM requires `DynamoConnector` and cannot consume the LMCache-managed L2 path without a new adapter. Record this as the primary finding.

### Stage 7 - Two-Replica Same-Node Cache Sharing

Only run after Stage 3 succeeds. Use a multi-GPU `ml.g7.*` or `ml.g6e.*` SKU if two pods need separate GPUs. If staying on `ml.g7.2xlarge` or `ml.g6e.xlarge`, skip this stage because there is only one GPU.

| Check | Method | Pass Criteria |
|---|---|---|
| Two workers on one HyperPod node | `kubectl get pods -o wide` | Both pods on same node, one GPU each |
| Both connect to same L2 daemon | Logs | Both use same node IP and port 9200 |
| Pod A warms prefix | Direct pod or frontend request | Cache write observed |
| Pod B reads prefix | Route/force request to Pod B | Cache hit or TTFT improvement observed |
| Dynamo routing awareness | Router logs | Router chooses cached worker when cache state is visible |

This is the minimum proof that the managed L2 daemon provides cross-pod value outside the HyperPod Inference Operator.

### Stage 8 - Optional FSx L3

Only run after L2 behavior is understood.

| Check | Method | Pass Criteria |
|---|---|---|
| FSx PVC mounted | `kubectl exec -- touch /mnt/fsx/...` | Writable from worker pod |
| Separate cache path | Inspect config | Uses `/mnt/fsx/dynamo-hp-lmcache`, not another blueprint's path |
| LMCache writes files | `find /mnt/fsx/dynamo-hp-lmcache` | Cache files appear |
| Restart persistence | Restart worker and repeat prefix | Cache hit or lower TTFT after restart |

## Minimal G7 Example

Use this manifest only after Stage 1 confirms the selected SKU exists in SageMaker HyperPod and the node label is present. If falling back to `ml.g6e.xlarge`, use an instance group name such as `g6e-workers` and update the nodeSelector accordingly.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamo-hp-lmcache-worker
  namespace: dynamo-hp-lmcache
spec:
  replicas: 1
  selector:
    matchLabels:
      app: dynamo-hp-lmcache-worker
  template:
    metadata:
      labels:
        app: dynamo-hp-lmcache-worker
    spec:
      nodeSelector:
        sagemaker.amazonaws.com/instance-group-name: g7-workers
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: vllm
          image: <dynamo-vllm-lmcache-compatible-image>
          command: ["python3", "-m", "dynamo.vllm"]
          args:
            - "--model"
            - "Qwen/Qwen3-0.6B"
            - "--connector"
            - "lmcache"
            - "--host"
            - "0.0.0.0"
            - "--port"
            - "8000"
            - "--max-model-len"
            - "8192"
            - "--gpu-memory-utilization"
            - "0.80"
          env:
            - name: NODE_IP
              valueFrom:
                fieldRef:
                  fieldPath: status.hostIP
            - name: LMCACHE_USE_EXPERIMENTAL
              value: "True"
            - name: LMCACHE_LOCAL_CPU
              value: "True"
            - name: LMCACHE_MAX_LOCAL_CPU_SIZE
              value: "8"
            - name: LMCACHE_CHUNK_SIZE
              value: "256"
            - name: LMCACHE_REMOTE_URL
              value: "sagemaker-hyperpod://$(NODE_IP):9200"
            - name: DYN_SYSTEM_PORT
              value: "8081"
          ports:
            - containerPort: 8000
              name: http
            - containerPort: 8081
              name: metrics
          resources:
            requests:
              cpu: "4"
              memory: 24Gi
              nvidia.com/gpu: "1"
            limits:
              memory: 48Gi
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: hp-tiered-cache
              mountPath: /dev/shm/ai_toolkit_cache
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 15
            failureThreshold: 40
      volumes:
        - name: hp-tiered-cache
          hostPath:
            path: /dev/shm/ai_toolkit_cache
            type: Directory
```

## Success Criteria

| Criteria | Stage | Type |
|---|---|---|
| Live SageMaker HyperPod API accepts either preferred `ml.g7.*` or fallback `ml.g6e.xlarge` | 1 | Critical |
| HyperPod tiered storage is enabled with `--tiered-storage-config` | 1 | Critical |
| L2 daemon is running on the selected HyperPod GPU node | 1 | Critical |
| Qwen3-0.6B serves on the selected SKU with a CUDA-compatible image for the discovered GPU architecture | 2 | Critical |
| Worker starts with Dynamo `--connector lmcache`, configures `LMCacheConnectorV1`, and connects to `sagemaker-hyperpod://<node-ip>:9200` | 3 | Critical |
| HyperPod daemon receives KV writes/reads from a non-operator worker pod | 3 | Critical |
| Dynamo frontend/router can serve through the LMCache-configured worker | 4 | Critical |
| End-to-end telemetry artifact proves serving, cache activity, L2 daemon evidence, and Dynamo routing | 5 | Critical |
| KVBM compatibility outcome is explicitly classified as PASS/PARTIAL/FAIL | 6 | Critical |
| Two-replica same-node cache sharing works or is skipped for one-GPU selected SKU with reason | 7 | Important |
| FSx L3 is not confused with managed L2 in reporting | 8 | Important |

## Known Risks and Blockers

| Risk | Severity | Detail | Mitigation |
|---|---|---|---|
| `ml.g7.*` may not exist in SageMaker HyperPod | High | The local AWS CLI allowed-value list for `sagemaker create-cluster` does not include `ml.g7.*`. | Stage 1 must validate against the live API. If rejected or unavailable in both target regions, fall back to `ml.g6e.xlarge`. |
| KVBM may not support LMCache-managed L2 | High | Dynamo's native KVBM path uses Dynamo's connector stack, while HyperPod L2 is LMCache protocol. | Treat Stage 6 as a compatibility gate. Report PARTIAL if Dynamo runs but KVBM is bypassed. |
| HyperPod daemon may reject non-operator pods | High | The daemon may depend on labels, annotations, mounted IPC, or versions injected by the HyperPod Inference Operator. | Test a bare vLLM+LMCache pod first; add operator-equivalent labels only if discovered. |
| Image lacks LMCache | High | The earlier `vllm-runtime:1.0.1` image served but did not activate KVBM; it may also lack the right LMCache build. | Verify `import lmcache` before deployment; build an overlay image if needed. |
| Wrong Dynamo launch path | High | Raw vLLM can validate daemon connectivity but does not prove Dynamo LMCache integration. | Use `python -m dynamo.vllm --connector lmcache` for the passing path; raw vLLM is diagnostic only. |
| ARM64 unsupported | Medium | NVIDIA's Dynamo LMCache docs state LMCache integration currently supports x86 only. | Gate Stage 1 on `uname -m == x86_64`. |
| CUDA runtime mismatch | Medium | The exact selected GPU architecture must match the container CUDA/PyTorch/vLLM build. | Record `nvidia-smi` and use an image compatible with the discovered GPU architecture; verify tag existence before capacity use. |
| `NODE_IP` env ordering | Medium | `LMCACHE_REMOTE_URL` must expand after `NODE_IP` is defined. | Put `NODE_IP` before dependent env vars; verify runtime env. |
| Mounting host `/dev/shm` broadly | Medium | Broad host `/dev/shm` mounts can collide with other daemon files. | Mount only `/dev/shm/ai_toolkit_cache`. |
| LMCache sync I/O regression | Medium | LMCache can serialize vLLM scheduler work under cache pressure. | Keep initial concurrency low; benchmark pressure only after functionality passes. |
| Storage/interconnect assumptions unknown until discovery | Low | Do not assume EFA, GDS, or NVLink properties before checking the selected SKU. | Treat this as functional smoke only. Escalate to p5en for GDS performance if needed. |

## Non-Requirements

- Production throughput benchmarking.
- P/D disaggregation with NIXL.
- Multi-node EFA validation.
- Proving FSx L3 performance.
- Training checkpoint save/load through `amzn-sagemaker-checkpointing`.
- Replacing the HyperPod Inference Operator's managed serving path.
- Serving MLA/NSA models; this spec intentionally uses a simple standard-attention model.

## Deployment Notes

Before deployment, run:

```bash
export AWS_REGION=us-east-2
mdc get Qwen/Qwen3-0.6B --engine vllm
mdc prs Qwen/Qwen3-0.6B
gpu-infra card g7 || true
gpu-infra card g6e || true
```

If `us-east-2` cannot provide G7 HyperPod capacity, set `AWS_REGION=us-west-2` and record the fallback reason. If SageMaker rejects or cannot place `ml.g7.*` in both regions, fall back to `ml.g6e.xlarge` and record the fallback reason. Record the exact Dynamo, vLLM, LMCache, CUDA, region, selected instance type, GPU model, GPU architecture, and HyperPod Inference Operator versions in `domains/gpu-serving/blueprints/dynamo-hyperpod-lmcache/lessons.md`.

## References

- AWS SageMaker docs: `managed-tier-checkpointing-setup.html` - `--tiered-storage-config '{"Mode":"Enable"}'`
- Local architecture note: `/Users/phi/Documents/workbench/aws/AWS_WorkDay/SageMaker Hyperpod - SMHP/HyperPod-Inference-Architecture.md`
- Local EKS deconstruction note: `/Users/phi/Documents/workbench/aws/AWS_WorkDay/SageMaker Hyperpod - SMHP/SageMaker Hyperpod deconstructed on EKS.md`
- Local multi-engine note: `/Users/phi/Documents/workbench/aws/AWS_WorkDay/SageMaker Hyperpod - SMHP/inference-operator-multi-engine-architecture.md`
- Existing companion spec: `domains/gpu-serving/specs/dynamo-hyperpod.md`
- Existing L2 integration spec: `domains/gpu-serving/specs/llmd-hyperpod.md`
- Prior Dynamo HyperPod result: `domains/gpu-serving/blueprints/dynamo-hyperpod/results/progress.md`
- NVIDIA Dynamo docs: `https://docs.dynamo.nvidia.com/dynamo/v-0-8-1/components/kvbm/lm-cache-integration` - `--connector lmcache`, `DYN_SYSTEM_PORT=8081`, LMCache metrics, aggregated vs disaggregated connector modes
