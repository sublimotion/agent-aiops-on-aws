# llm-d on HyperPod — Validation Results

**Date**: 2026-03-31
**Clusters**:
- finetune-g5-cluster / finetune-eks (training, K8s 1.33) — L0 baseline validated
- llmd-inference-cluster / inference-eks-v132 (inference, K8s 1.32) — L2+L3 validated
**Model**: Qwen/Qwen3-0.6B on g5.4xlarge (A10G, 24GB)
**llm-d version**: modelservice v0.4.7, infra v1.4.0, GAIE inferencepool v1.4.0, vLLM v0.15.1

## Validation Summary

| Stage | Result | Notes |
|-------|--------|-------|
| 1. Infrastructure Discovery | PASS | HyperPod inference cluster with system nodes, GPU node, L2 daemon, FSx |
| 2. Gateway Stack | PASS | Istio 1.29.1 + Gateway API v1.2.1 + GAIE CRDs installed |
| 3. vLLM Baseline (L0) | PASS | Qwen3-0.6B serving, prefix cache active (59.5% hit rate) |
| 4. L2 Integration | PASS | LMCache → ai-toolkit daemon via sagemaker-hyperpod:// connector, shared memory IPC |
| 5. L3 FSx | PASS | FSx Lustre 1.2 TiB mounted via CSI PV/PVC, writable from vLLM pod at /mnt/fsx |
| 6. CRD Coexistence | PASS | Gateway API + GAIE + Istio + MPI/training + HyperPod inference CRDs coexist |
| 7. Observability | PARTIAL | PodMonitor/ServiceMonitor CRDs installed, vLLM /metrics exposed, no Prometheus operator to scrape |

## End-to-End Routing Verified

```
Client → Istio Gateway → EPP (ext-proc, port 9002) → vLLM (port 8000) → Response
```

- Gateway status: Programmed=True
- InferencePool: GA API (inference.networking.k8s.io/v1) with endpointPickerRef
- EPP detected vLLM pod via label selector, started metrics refresher
- Chat completions return valid responses through full gateway path

## Prefix Cache (L0) Working

```
prefix_cache_queries_total: 215
prefix_cache_hits_total: 128 (59.5% hit rate)
external_prefix_cache_queries_total: 215  (all queries also check L2)
external_prefix_cache_hits_total: 0  (expected: single replica, L0 satisfies before L2)
```

## L2 Integration — ai-toolkit Daemon Connected

**Config**:
```yaml
args:
  - "--kv-transfer-config"
  - '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"sagemaker_hyperpod_shared_memory_name":"ai_toolkit_cache"}}'
env:
  - LMCACHE_CONFIG_FILE=/etc/lmcache/config.yaml  # sets extra_config.sagemaker_hyperpod_shared_memory_name
  - LMCACHE_LOCAL_CPU=True  # L1 CPU offload
  - LMCACHE_REMOTE_URL=sagemaker-hyperpod://$(NODE_IP):9200
volumes:
  - /dev/shm (hostPath) — shares POSIX shm with ai-toolkit daemon
```

**Connection flow**:
1. LMCache discovers SageMakerHyperPodConnectorAdapter
2. Creates connector: `url=http://10.1.90.19:9200, bucket=lmcache, shared_memory=ai_toolkit_cache`
3. Opens shared memory: `ai_toolkit_cache (1024.00 MB)`
4. Health check recovers: `Connection initialized/re-established`

**Gotchas solved**:
- Default `shared_memory_name` in LMCache is `shared_memory`, but ai-toolkit uses `ai_toolkit_cache` → set via `kv_connector_extra_config` AND `LMCACHE_CONFIG_FILE` (the config.extra_config path is what the adapter reads)
- vLLM runs as uid 2000, ai-toolkit as uid 1000 → shared memory file is 0600 → need `chmod 666 /dev/shm/ai_toolkit_cache` (production: run both with same GID)
- `$(NODE_IP)` env var substitution requires `NODE_IP` to be defined BEFORE `LMCACHE_REMOTE_URL` in the env list (K8s dependent variable ordering)
- Must use `hostPath: /dev/shm` (not emptyDir with medium: Memory) to share POSIX shm with host processes

## HyperPod Inference Cluster Setup

**Cluster config**:
```
Name: llmd-inference-cluster
EKS: inference-eks-v132 (K8s 1.32)
GPU: ml.g5.4xlarge (InstanceGroup: gpu-workers)
System: ml.m5.2xlarge (RestrictedInstanceGroup: system-nodes)
FSx Lustre: 1.2 TiB (fs-0ea95350af6e402f6)
TieredStorage: Enabled (20% instance memory)
```

**Key findings**:
- `RestrictedInstanceGroups` requires EKS ≤ 1.32 (1.33 not supported)
- `TieredStorageConfig.Mode: Enable` deploys ai-toolkit DaemonSet on GPU nodes (port 9200)
- `FSxLustreConfig.SizeInGiB: 1200` (minimum) creates FSx file system automatically
- Inference operator addon (`amazon-sagemaker-hyperpod-inference`) installed but `EnableClusterInference` fails (EnableFailed) — appears to be a transient backend issue. Not required for self-managed llm-d.
- HyperPod dependencies helm chart from `sagemaker-hyperpod-cli` repo must be installed BEFORE cluster creation

## L3 FSx Lustre — HyperPod Managed

**FSx filesystem**: `fs-0ea95350af6e402f6` (1.2 TiB, HyperPod auto-provisioned via `FSxLustreConfig`)

**Integration**:
- FSx CSI driver (`fsx.csi.aws.com`) pre-installed on HyperPod inference cluster
- Static PV/PVC created pointing to HyperPod-managed FSx (`dnsname`, `mountname` from `aws fsx describe-file-systems`)
- Mounted at `/mnt/fsx` in vLLM pod (`10.1.239.185@tcp:/bukizb4v 1.2T`)
- Write verified from vLLM pod (uid 2000) after creating `/mnt/fsx/kvcache` with 777 permissions

**Gotchas**:
- FSx root dir is 755 owned by root:root — vLLM (uid 2000) cannot write directly. Need init container or privileged pod to `mkdir -p /mnt/fsx/kvcache && chmod 777 /mnt/fsx/kvcache`
- To use FSx as L3 cache backend, switch `LMCACHE_REMOTE_URL` from `sagemaker-hyperpod://` to `file:///mnt/fsx/kvcache`
- Both L2 (shared memory) and L3 (FSx) can coexist — L2 is same-node fast path, L3 is cross-node persistent cache

**PV/PVC manifest**: `manifests/fsx-pv-pvc.yaml`

## Inference Operator — EnableFailed (Non-blocking)

The `amazon-sagemaker-hyperpod-inference` addon (v1.0.1) installs the controller manager, KEDA, ALB controller into `hyperpod-inference-system`. The controller tries `EnableClusterInference` but gets `EnableFailed` after `Enabling` state.

Root causes explored:
1. **IRSA trust policy missing**: Fixed — added inference-eks-v132 OIDC provider to execution role trust policy
2. **Stale watcher state**: Previous cluster creation attempts left "watcher being deleted" state. Cleared after ~5 min.
3. **EnableFailed after Enabling**: Consistent across 5 retries. SageMaker backend issue — not a client-side problem.

**Resolution**: Not needed for llm-d validation. The operator enables SageMaker-managed inference endpoints (JumpStart). llm-d is self-managed via Helm.

## IRSA Configuration

```json
{
  "Principal": {"Federated": "arn:aws:iam::615299764834:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/8D82766265F2959DCA4BC9FE687E0DD6"},
  "Action": "sts:AssumeRoleWithWebIdentity",
  "Condition": {"StringLike": {"...sub": "system:serviceaccount:*:*"}}
}
```

## Helm Releases (inference cluster)

```
NAME                         CHART                                    VERSION  STATUS
infra-inference-scheduling   llm-d-infra/llm-d-infra                 v1.4.0   deployed
gaie-inference-scheduling    inferencepool (OCI)                      v1.4.0   deployed
ms-inference-scheduling      llm-d-modelservice/llm-d-modelservice   v0.4.7   deployed
```

## Pod Resource Usage (g5.4xlarge — 16 vCPU, 64 GiB, 1x A10G)

| Pod | CPU Req | CPU Lim | Mem Req | Mem Lim | GPU |
|-----|---------|---------|---------|---------|-----|
| vLLM decode | 2 | 6 | 8Gi | 24Gi | 1 |
| EPP | — | — | — | — | 0 |
| Istio Gateway | — | — | — | — | 0 |

## Key Lessons

1. **g5.4xlarge gives much better pod headroom**: 29 max-pods vs 14 on g5.2xlarge.
2. **HyperPod training cluster ≠ inference cluster**: RestrictedInstanceGroups (system nodes), TieredStorageConfig, and FSxLustreConfig are only available on inference-enabled clusters.
3. **EKS 1.32 required**: RestrictedInstanceGroups not supported on K8s 1.33.
4. **L2 shared memory name mismatch**: LMCache defaults to `shared_memory`, ai-toolkit uses `ai_toolkit_cache`. Must configure via both `kv_connector_extra_config` AND `LMCACHE_CONFIG_FILE` (the adapter reads `config.extra_config`, not the vLLM-level extra config).
5. **L2 permission issue**: ai-toolkit creates shm with 0600 (uid 1000). vLLM runs as uid 2000. Need to chmod or unify UIDs.
6. **Env var ordering matters**: K8s `$(VAR)` substitution only resolves vars defined earlier in the list.
7. **hostPath /dev/shm for L2**: Must use hostPath (not emptyDir) to share POSIX shared memory with the ai-toolkit daemon.
8. **HyperPod dependencies must be installed first**: Clone `sagemaker-hyperpod-cli` repo, install `HyperPodHelmChart` before cluster creation or system nodes won't schedule.
9. **Inference operator not required for llm-d**: The addon enables SageMaker-managed endpoints. Self-managed llm-d works without it.
10. **Qwen3-0.6B is ideal for validation**: Non-gated, <2GB, starts in <2 min on A10G.
11. **FSx root dir permissions**: HyperPod-managed FSx root is 755/root:root. vLLM (uid 2000) can't write. Create subdirectory with 777 via init container or privileged setup pod.
12. **FSx CSI PV uses static provisioning**: Use `volumeHandle` + `volumeAttributes.dnsname` + `volumeAttributes.mountname` from `aws fsx describe-file-systems`. No StorageClass needed.
