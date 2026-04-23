# NVIDIA Dynamo on SageMaker HyperPod EKS — KVBM + FSx Integration Spec

## Status: DRAFT (2026-03-31)

## Overview

Validate **NVIDIA Dynamo v1.0.1** running on SageMaker HyperPod EKS, using Dynamo's native KVBM for KV cache management with HyperPod's managed FSx Lustre as the disk-tier (G3) backend. This is **Option A**: Dynamo owns the full cache hierarchy (G1 GPU → G2 CPU → G3 FSx), with no dependency on HyperPod's L2 tiered storage daemon.

**Why this matters:**
- Dynamo adds disaggregated prefill/decode, KV-aware routing, and SLA-driven autoscaling — capabilities that llm-d lacks today
- KVBM's G3 disk tier is explicitly Lustre-aware (`ZEROFILL_FALLBACK`, `DISABLE_O_DIRECT`), making FSx a natural managed storage backend
- If Dynamo + KVBM + FSx works on HyperPod, customers get Dynamo's orchestration intelligence on managed infrastructure without needing to self-manage etcd, storage, or scaling
- This is complementary to `llmd-hyperpod` (which validates llm-d + LMCache + L2 daemon) — different orchestration layer, same managed compute/storage substrate

**Prior art:**
- `llmd-hyperpod` — llm-d EPP + LMCache on HyperPod EKS g5.4xlarge (inference-eks-v132, K8s 1.32). **Validated**: L2 daemon, L3 FSx, Gateway API, CRD coexistence. 21 PASS, 0 FAIL. **Different orchestration layer.**
- `nemotron-super` — Dynamo v0.9.1 on vanilla EKS p6-b200. **Validated**: vLLM runtime cold start, aggregated serving, KVBM. **Not on HyperPod.**
- `glm5-llmd` — llm-d on vanilla EKS p6-b200. **No HyperPod.**

**Model**: Qwen/Qwen3-0.6B — same model validated on `llmd-hyperpod` for direct comparison. Non-gated, <2 GB, starts in <2 min on A10G. Dense GQA, no MLA/NSA/Mamba blockers.

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │     Dynamo Frontend          │
                    │  OpenAI-compatible API :8000 │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │     Dynamo Router            │
                    │  KV-aware request routing    │
                    │  (reads KVBM cache state)    │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐
    │  vLLM Worker 1 │ │ vLLM Worker 2│ │ vLLM Worker 3│
    │  Node A (GPU)  │ │ Node B (GPU) │ │ Node C (GPU) │
    │                │ │              │ │              │
    │  G1: GPU HBM   │ │  G1: GPU HBM │ │  G1: GPU HBM │
    │  G2: CPU DRAM   │ │  G2: CPU DRAM│ │  G2: CPU DRAM│
    │  G3: FSx Lustre │ │  G3: FSx     │ │  G3: FSx     │
    └────────────────┘ └──────────────┘ └──────────────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   FSx Lustre (HyperPod L3)  │
                    │   1.2 TiB, auto-provisioned │
                    │   Shared cross-node G3 tier   │
                    └─────────────────────────────┘
```

**Key design decisions:**
- **No L2 daemon dependency** — KVBM manages G2 (CPU pinned DRAM) natively, no need for HyperPod's tiered storage daemon (ai-toolkit on port 9200). The daemon runs on GPU nodes but Dynamo doesn't interact with it.
- **FSx as G3** — KVBM's disk tier points at HyperPod's auto-provisioned FSx mount, giving cross-node shared KV cache with Lustre-aware I/O
- **Dynamo Router has full visibility** — unlike the hybrid Option B/C, KVBM reports cache state for all tiers to the Router
- **etcd for service discovery** — Dynamo v1.0+ removed NATS dependency, uses etcd (or file-based for single-node)
- **1 GPU per node** — g5.4xlarge has 1x A10G, so each worker = 1 node. Multi-replica requires scaling the gpu-workers instance group to 2-3 nodes.

---

## What We're Validating

### V1: Dynamo Operator on HyperPod EKS

Can the Dynamo Kubernetes Operator (`DynamoModel` CRD, `DynamoGraphDeploymentRequest`) deploy and manage workers on HyperPod EKS nodes? HyperPod has specific node labels, taints (`sagemaker.amazonaws.com/instance-group-name`), RestrictedInstanceGroups for system nodes, and managed add-ons (cert-manager, KEDA, ALB controller) that may conflict.

### V2: KVBM G3 Tier on FSx Lustre

Can KVBM's disk tier use HyperPod's auto-provisioned FSx Lustre (1.2 TiB, fs-0ea95350af6e402f6) as the G3 backend? Env vars:
```bash
DYN_KVBM_DISK_CACHE_DIR=/mnt/fsx/kv-cache/
DYN_KVBM_DISK_CACHE_GB=50          # conservative for 1.2 TiB filesystem
DYN_KVBM_DISK_ZEROFILL_FALLBACK=true    # Lustre lacks fallocate()
DYN_KVBM_DISK_DISABLE_O_DIRECT=true     # Lustre strict alignment
```

### V3: KV-Aware Routing with KVBM State

Does the Dynamo Router make correct routing decisions based on KVBM's view of G1/G2/G3 cache state? Send identical prefixed requests and verify the Router directs to the replica with the cached prefix.

### V4: Disaggregated Prefill/Decode (Stretch Goal)

Can Dynamo's P/D disaggregation work on HyperPod with g5.4xlarge? This is a stretch because:
- g5.4xlarge has only 1 GPU per node — P/D requires separate nodes
- PCIe Gen4 interconnect limits KV transfer bandwidth
- A10G (sm_86 Ampere) — verify NIXL compatibility
- Needs 2+ additional nodes beyond the routing test minimum

### V5: Planner Autoscaling (Stretch Goal)

Does Dynamo's Planner correctly scale worker replicas based on SLA targets, integrated with HyperPod's node management (Karpenter or instance group scaling)?

---

## Components

### 1. Compute — HyperPod EKS Cluster (Reused)

- **Platform**: SageMaker HyperPod with EKS orchestrator
- **Cluster**: `llmd-inference-cluster` / EKS `inference-eks-v132` (K8s 1.32) — same cluster as `llmd-hyperpod`
- **GPU nodes**: `ml.g5.4xlarge` — 1x A10G (24 GB VRAM), 16 vCPU, 64 GiB RAM
  - InstanceGroup: `gpu-workers` — **scale to 3 nodes** for multi-replica routing tests
- **System nodes**: `ml.m5.2xlarge` — RestrictedInstanceGroup
- **Namespace**: `dynamo-validation` (isolated from `llmd-validation` and existing workloads)
- **FSx**: Reuse existing auto-provisioned FSx (fs-0ea95350af6e402f6, 1.2 TiB)
- **Region**: us-east-1

> **Node scaling**: The llmd-hyperpod validation used 1 GPU node. Dynamo routing tests require 2-3 replicas on separate nodes. Scale the `gpu-workers` instance group via HyperPod `UpdateCluster` API or manually add nodes.

### 2. Model

| Property | Value |
|----------|-------|
| **Model ID** | `Qwen/Qwen3-0.6B` |
| **Parameters** | 0.6B (dense) |
| **Attention** | Standard GQA |
| **Precision** | BF16 (~1.2 GB) |
| **TP** | 1 (single A10G per replica) |
| **Max replicas** | 3 (one per g5.4xlarge node) |
| **KV cache headroom** | ~20 GB on A10G after model load |
| **Backend** | vLLM via Dynamo worker |

> **Model choice**: Qwen3-0.6B matches what was validated on `llmd-hyperpod` (progress.md lesson #10: non-gated, <2GB, starts in <2 min). The small footprint leaves ~20 GB for KV cache — set small G1/G2 limits to trigger G3 eviction during testing. For KVBM G2/G3 eviction tests, use `--gpu-memory-utilization=0.5` to artificially constrain G1.

### 3. NVIDIA Dynamo Stack

| Component | Image | Role |
|-----------|-------|------|
| **Frontend** | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1` | OpenAI-compatible API, request batching |
| **Router** | (bundled with Frontend) | KV-aware routing using KVBM cache state |
| **Planner** | (bundled or separate) | SLA-driven autoscaling decisions |
| **Workers** | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1` | vLLM inference with KVBM KV management |
| **etcd** | `quay.io/coreos/etcd:v3.5.x` | Service discovery (replaces NATS in v1.0+) |
| **Operator** | Dynamo Kubernetes Operator | Lifecycle management via `DynamoModel` CRD |

> **Image tag format**: nemotron-super lesson #3 — NGC Dynamo images use `0.9.1` (no `v` prefix). Verify `1.0.1` follows the same convention. If `1.0.1` returns MANIFEST_UNKNOWN, try without prefix. Also verify A10G (sm_86 Ampere) compatibility — v1.0.1 targets L40S/H100 primarily.

### 4. Storage

| Tier | KVBM Label | Medium | Config | Purpose |
|------|------------|--------|--------|---------|
| **G1** | GPU HBM | A10G VRAM (~20 GB available) | `--gpu-memory-utilization=0.5` (constrain to trigger eviction) | Active KV for inference |
| **G2** | CPU DRAM | Host pinned memory | `DYN_KVBM_CPU_CACHE_GB=8` (g5.4xlarge has 64 GiB total) | Warm cache, recently evicted KV |
| **G3** | Disk (FSx) | FSx Lustre PV | `DYN_KVBM_DISK_CACHE_DIR=/mnt/fsx/kv-cache/` | Cross-node shared persistent cache |
| **Model weights** | — | HuggingFace Hub download | `Qwen/Qwen3-0.6B` (non-gated, ~1.2 GB) | Model storage |

**FSx Lustre (reused from llmd-hyperpod):**
- Filesystem: `fs-0ea95350af6e402f6` (1.2 TiB, HyperPod auto-provisioned via `FSxLustreConfig`)
- DNS: `fs-0ea95350af6e402f6.fsx.us-east-1.amazonaws.com`
- Mount name: `bukizb4v`
- Mount: `/mnt/fsx` via existing FSx CSI PV/PVC (static provisioning)
- Reuse PV/PVC from `manifests/fsx-pv-pvc.yaml` (create new PVC in `dynamo-validation` namespace)

> **FSx permissions gotcha** (llmd-hyperpod lesson #11): FSx root dir is `755 root:root`. Dynamo workers won't be able to write to `/mnt/fsx/kv-cache/` unless the directory is created with appropriate permissions. Use a privileged init container: `mkdir -p /mnt/fsx/kv-cache && chmod 777 /mnt/fsx/kv-cache`. Same pattern used for llmd-hyperpod's `/mnt/fsx/kvcache`.

### 5. Networking

- **Client ingress**: NLB → Dynamo Frontend (port 8000)
- **Inter-component**: gRPC between Frontend/Router/Workers (intra-cluster)
- **Service discovery**: etcd (deployed in `dynamo-validation` namespace)
- **EFA**: Not available on g5.4xlarge (EFA requires g5.12xlarge+)
- **No NIXL for V1-V3**: Workers on separate nodes but no cross-node KV transfer — G3 FSx is the shared medium
- **NIXL for V4**: Would require g5.12xlarge+ for EFA, or TCP fallback with `kv_buffer_device: cpu`

### 6. Observability

- **Dynamo metrics**: Frontend exports `dynamo_frontend_*` metrics on configurable port
- **vLLM metrics**: Standard `vllm_*` on port 8000 per worker
- **KVBM metrics**: Cache hit rates per tier, eviction rates, transfer latencies
- **ADOT integration**: **NOT auto-installed** on our HyperPod cluster (llmd-hyperpod empirical correction). PodMonitor/ServiceMonitor CRDs are present but no Prometheus operator to scrape. Manual ADOT addon installation or port-forward for metrics validation.
- **Fallback**: Validate metrics via direct `curl <pod-ip>:<port>/metrics` and port-forward if ADOT not available

---

## Validation Stages

### Stage 1: Infrastructure Discovery

Reuse `llmd-hyperpod` smoke_test.sh infrastructure checks, plus Dynamo-specific checks. Apply empirical corrections from llmd-hyperpod deployment.

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| GPU nodes available (3 needed) | `kubectl get nodes -l sagemaker.amazonaws.com/instance-group-name=gpu-workers` | At least 3 GPU nodes Ready |
| FSx CSI driver installed | `kubectl get csidriver fsx.csi.aws.com` | CSI driver registered (validated on llmd-hyperpod) |
| FSx PV available | `kubectl get pv,pvc -A \| grep fsx` | Existing PV Bound |
| System node taints | `kubectl get nodes -l node-role.kubernetes.io/system=true -o jsonpath='{.items[*].spec.taints}'` | Document for etcd/Frontend tolerations |
| L2 daemon running (NOT used, check for conflicts) | `kubectl get pods -n aws-hyperpod \| grep ai-toolkit` | Running — verify port 9200 won't conflict with Dynamo |
| ai-toolkit shm file | `kubectl exec -n aws-hyperpod <daemon-pod> -- ls /dev/shm/ai_toolkit_cache` | Present — Dynamo must NOT mount `/dev/shm` as hostPath (would see this file) |
| No existing etcd in cluster | `kubectl get pods -A \| grep etcd` | No user-deployed etcd (EKS control plane etcd is inaccessible) |
| Existing llm-d resources | `kubectl get all -n llmd-validation` | Document — ensure no GPU resource conflict |
| ADOT collector status | `kubectl get pods -A \| grep -i adot\|otel` | **Expect: NOT FOUND** (empirically confirmed) |
| Gateway API CRDs present | `kubectl get crd \| grep gateway.networking.k8s.io` | Present from llm-d install — Dynamo CRDs must coexist |
| Inference Operator status | `kubectl get pods -n hyperpod-inference-system` | Document — EnableFailed is known/non-blocking |

### Stage 2: Dynamo Operator + etcd Deployment

Install the Dynamo Kubernetes Operator and etcd for service discovery.

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| Dynamo Operator CRDs installed | `kubectl get crd \| grep dynamo` | `dynamomodels` CRD present |
| Operator controller running | `kubectl get pods -n dynamo-system` | Controller pod Running |
| etcd deployed in dynamo-validation | `kubectl get pods -n dynamo-validation -l app=etcd` | etcd pod Running, healthy |
| etcd scheduled on GPU node | Check node placement | GPU nodes have abundant CPU/RAM (llmd-hyperpod: 16 vCPU, 64 GiB on g5.4xlarge). Add `tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]` |
| etcd reachable | `kubectl exec <etcd-pod> -- etcdctl endpoint health` | Healthy |
| No CRD conflicts with HyperPod | Check HyperPod operator logs | No admission webhook errors |
| No conflict with Gateway API CRDs | `kubectl get crd \| grep -E 'gateway\|dynamo\|inference'` | All CRD groups coexist: `gateway.networking.k8s.io`, `inference.networking.k8s.io`, `inference.sagemaker.aws.amazon.com`, `dynamo.*` |

> **etcd placement**: Deploy on GPU nodes, not system nodes. System nodes have RestrictedInstanceGroup taints. GPU nodes have abundant free CPU/RAM for lightweight services (same pattern used for Redis in ray-serve-ft blueprint).

### Stage 3: vLLM Workers via Dynamo (Baseline — G1 Only)

Deploy Qwen3-0.6B workers using Dynamo operator. No KVBM tiers beyond GPU HBM.

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| DynamoModel CR applied | `kubectl apply -f dynamo-model.yaml` | CR accepted |
| Worker pods scheduled on GPU nodes | `kubectl get pods -n dynamo-validation -o wide` | Running on gpu-workers nodes, 1 per node |
| Workers have correct nodeSelector | Check pod spec | `sagemaker.amazonaws.com/instance-group-name: gpu-workers` |
| Workers have GPU toleration | Check pod spec | `tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]` |
| Workers registered in etcd | `kubectl exec <etcd-pod> -- etcdctl get --prefix /dynamo/` | Worker endpoints listed |
| Frontend serves requests | `curl <frontend-svc>:8000/v1/chat/completions` | Valid response (Qwen3-0.6B) |
| Health endpoint | `curl <frontend-svc>:8000/health` | 200 OK |
| Prefix caching enabled | `curl <worker-ip>:8000/metrics \| grep prefix_cache` | Metric present |
| Dynamo Frontend metrics | `curl <frontend-svc>:<metrics-port>/metrics` | `dynamo_frontend_*` metrics present |
| Router KV-aware routing | Send 2 identical prefix requests, check Router logs | Second routes to same worker |
| A10G sm_86 compatibility | `kubectl exec <worker> -- nvidia-smi` | GPU recognized, CUDA works |

### Stage 4: KVBM G2 — CPU DRAM Cache

Enable KVBM's CPU tier to extend cache capacity. Use small G1 to trigger G2 eviction.

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| Constrain G1 | Set `--gpu-memory-utilization=0.5` | Workers restart, ~10 GB for KV cache |
| Set `DYN_KVBM_CPU_CACHE_GB=8` | Patch worker deployment env | Workers restart without crash |
| G2 tier active in metrics | Check KVBM metrics endpoint | `kvbm_cpu_cache_blocks` > 0 |
| KV eviction from G1 to G2 | Send many diverse long-context requests to fill G1 | Blocks move to CPU tier (visible in metrics) |
| G2 cache hit | Request with previously-evicted prefix | TTFT faster than full recompute |
| Router aware of G2 state | Check Router decision logs | Routes to replica with G2-cached prefix |

### Stage 5: KVBM G3 — FSx Lustre (Critical Integration)

Enable KVBM's disk tier pointing at HyperPod's managed FSx Lustre. This is the **key validation**.

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| Create `/mnt/fsx/kv-cache/` directory | Privileged init container or busybox pod: `mkdir -p /mnt/fsx/kv-cache && chmod 777 /mnt/fsx/kv-cache` | Directory writable by Dynamo worker UID |
| FSx PVC mounted in worker pods | Add PVC mount at `/mnt/fsx` (reuse fsx-pv-pvc.yaml pattern, new PVC in dynamo-validation ns) | Mount succeeds, writable |
| Set KVBM disk tier env vars | See env block below | Workers restart without crash |
| G3 tier active in metrics | Check KVBM metrics | `kvbm_disk_cache_blocks` > 0 |
| KV eviction G2 → G3 | Exhaust G1+G2 (send many requests), check FSx for files | Files appear in `/mnt/fsx/kv-cache/` |
| G3 cache hit | Request with G3-cached prefix | KV loaded from FSx, TTFT faster than recompute |
| Cross-node G3 sharing | Worker on Node A writes KV to FSx, worker on Node B reads same prefix | Node B cache hit from FSx (verify via TTFT delta) |
| Router routes to G3-cached replica | Check Router decision logs | Correct routing to replica with FSx-cached KV |
| No conflict with L2 daemon | Check L2 daemon logs (`kubectl logs -n aws-hyperpod <ai-toolkit-pod>`) | No errors about unexpected FSx activity in `/mnt/fsx/kv-cache/` |
| FSx file cleanup on eviction | Monitor `ls /mnt/fsx/kv-cache/` | Temp files removed when blocks evicted |
| No conflict with llmd-hyperpod cache dir | Verify Dynamo uses `/mnt/fsx/kv-cache/` (not `/mnt/fsx/kvcache/`) | Separate directory from llm-d's LMCache path |

**KVBM G3 env vars:**
```bash
DYN_KVBM_DISK_CACHE_DIR=/mnt/fsx/kv-cache/    # separate from llm-d's /mnt/fsx/kvcache/
DYN_KVBM_DISK_CACHE_GB=50                       # conservative for 1.2 TiB shared filesystem
DYN_KVBM_DISK_ZEROFILL_FALLBACK=true
DYN_KVBM_DISK_DISABLE_O_DIRECT=true
```

### Stage 6: Observability Integration

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| vLLM metrics accessible | `kubectl port-forward <worker>:8000` then `curl localhost:8000/metrics` | `vllm_*` metrics present |
| Dynamo Frontend metrics accessible | Port-forward Frontend pod | `dynamo_frontend_*` metrics present |
| KVBM tier metrics accessible | Check KVBM metrics endpoint (same port or separate) | Per-tier hit/miss/eviction rates visible |
| PodMonitor/ServiceMonitor created | `kubectl apply` monitors for Dynamo components | CRDs accepted (CRDs are present from llm-d install) |
| **ADOT not available** | `kubectl get pods -A \| grep adot` | Expect NOT FOUND — document as gap (same as llmd-hyperpod) |
| Metrics via manual scrape | Direct curl to pod metrics endpoints | All three metric sets (vLLM, Frontend, KVBM) accessible |

> **Observability gap**: ADOT is not auto-installed on our HyperPod inference cluster (llmd-hyperpod empirical finding). Metrics validation falls back to direct pod port-forward. Full AMP/AMG integration would require manual ADOT addon installation.

### Stage 7: Disaggregated P/D (Stretch Goal)

Only attempt if Stages 1-6 pass. Requires at least 4 nodes total (1 Frontend/Router, 1+ prefill, 1+ decode).

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| Scale gpu-workers to 4+ nodes | HyperPod UpdateCluster or manual | 4+ GPU nodes Ready |
| Configure prefill workers | Dynamo role=prefill, nodes 1-2 | Workers start in prefill mode |
| Configure decode workers | Dynamo role=decode, nodes 3-4 | Workers start in decode mode |
| KV transfer mechanism | Check if NIXL TCP fallback works (no EFA on g5.4xlarge) | KV transferred between nodes |
| TTFT vs monolithic | Compare TTFT at moderate concurrency | Document P/D overhead on PCIe/TCP path |
| Planner makes scaling decisions | Check Planner logs/metrics | Planner attempts to adjust prefill/decode ratio |

> **Limitation**: g5.4xlarge does NOT support EFA (requires g5.12xlarge+). NIXL must use TCP fallback with `kv_buffer_device: cpu`. Expect significant overhead vs NVLink/EFA paths. This is a functional validation, not a performance test.

---

## Known Risks and Blockers

| Risk | Severity | Detail | Mitigation |
|------|----------|--------|------------|
| **Dynamo Operator on HyperPod EKS** | HIGH | HyperPod's managed add-ons (cert-manager, KEDA, ALB controller, node lifecycle) may conflict with Dynamo Operator. No published HyperPod deployments. | Test operator install in isolated namespace first. If operator fails, deploy components manually via Helm/kubectl. |
| **NGC image A10G (sm_86) compatibility** | MEDIUM | v1.0.1 images primarily target L40S/H100/B200. A10G is Ampere (sm_86) — may lack optimized kernels or fail CUDA capability check. | Run `nvidia-smi` in container first. If incompatible, try v0.9.1 images (nemotron-super validated on B200 sm_90). Worst case: deploy Dynamo components manually with standard vLLM image. |
| **NGC image tag format** | LOW | nemotron-super lesson #3: tags use `0.9.1` not `v0.9.1`. Verify `1.0.1` follows same convention. | Try both `1.0.1` and `v1.0.1`. |
| **KVBM G3 on Lustre — fallocate/O_DIRECT** | MEDIUM | Despite Lustre-aware code paths, untested on HyperPod's auto-provisioned FSx specifically. | Pre-validate with simple file I/O test on FSx mount: `dd if=/dev/zero of=/mnt/fsx/kv-cache/test bs=1M count=100` before enabling KVBM. |
| **FSx permissions** | HIGH | HyperPod-managed FSx root is `755 root:root`. Dynamo workers (unknown UID) cannot write directly. | Create `/mnt/fsx/kv-cache/` with `chmod 777` via privileged pod before deploying workers. Same pattern as llmd-hyperpod (lesson #11). |
| **Tokenizer divergence (#7693)** | MEDIUM | Rust/Python tokenizer mismatch causes false prompt rejection. Known open bug in Dynamo v1.0.1. | Test with Qwen3-0.6B first. If hit, pin to Python tokenizer path. |
| **Structured output corruption (#7634)** | MEDIUM | Request migration corrupts structured-output after worker crash. | Avoid worker crash scenarios in validation. Document if encountered. |
| **Webhook blocking reconciliation (#7656)** | MEDIUM | DynamoGraphDeploymentScalingAdapter webhook blocks operator with short Helm names. | Use longer release names. If hit, deploy without scaling adapter. |
| **etcd on HyperPod nodes** | LOW | System nodes have RestrictedInstanceGroup taints. | Deploy etcd on GPU nodes with nvidia.com/gpu toleration (abundant free CPU/RAM). Same pattern as Redis in ray-serve-ft. |
| **Cross-pod G3 sharing semantics** | MEDIUM | KVBM's temp file naming (`mkostemp`) creates per-process files. Cross-pod sharing may require KVBM's NIXL remote transfer layer, not raw filesystem reads. | Test whether Worker B can read Worker A's KVBM files directly. If not, G3 sharing requires Router mediation — still validates FSx as durable tier. |
| **L2 daemon conflict** | LOW | ai-toolkit daemon manages `/dev/shm/ai_toolkit_cache` on GPU nodes. If Dynamo mounts `/dev/shm` as hostPath, it would see the daemon's shared memory. | Dynamo workers should NOT mount `/dev/shm` as hostPath. KVBM G2 uses pinned CPU DRAM (malloc), not POSIX shared memory. Verify no accidental hostPath in Dynamo manifests. |
| **No EFA on g5.4xlarge** | HIGH (Stage 7) | g5.4xlarge doesn't support EFA. NIXL requires TCP fallback with `kv_buffer_device: cpu`. | Stage 7 is stretch goal. Accept TCP-only path. For EFA validation, would need g5.12xlarge+ nodes. |
| **GPU resource contention with llmd-validation** | MEDIUM | If llmd-hyperpod pods are still running, they hold the GPU on their node. | Scale down llmd-validation replicas to 0 before running Dynamo tests, or scale gpu-workers to have dedicated nodes. |

---

## Empirical Corrections from llmd-hyperpod

These findings from the llmd-hyperpod deployment directly affect this spec:

| Finding | Impact on dynamo-hyperpod | Source |
|---------|---------------------------|--------|
| ADOT not auto-installed | Stage 6 observability is port-forward only | llmd-hyperpod empirical correction |
| FSx root is 755/root:root | Must create `/mnt/fsx/kv-cache/` with init container | llmd-hyperpod lesson #11 |
| FSx CSI uses static provisioning | Reuse PV pattern from `manifests/fsx-pv-pvc.yaml` | llmd-hyperpod lesson #12 |
| EKS 1.32 required for RestrictedInstanceGroups | Cluster is already 1.32 — no change needed | llmd-hyperpod lesson #3 |
| g5.4xlarge gives 29 max-pods | Sufficient headroom for Dynamo components | llmd-hyperpod lesson #1 |
| HyperPod Helm prereqs must be pre-installed | Already done on this cluster | llmd-hyperpod lesson #8 |
| EnableClusterInference: EnableFailed | Non-blocking — Dynamo doesn't use the inference operator | llmd-hyperpod progress.md |
| env var ordering: NODE_IP before dependent vars | Relevant if Dynamo uses `$(VAR)` substitution in env | llmd-hyperpod lesson #6 |

---

## Comparison: Dynamo vs llm-d on HyperPod

Both run on the same cluster (`llmd-inference-cluster`) with the same model (Qwen3-0.6B). This table captures what each validates:

| Capability | `llmd-hyperpod` (validated) | `dynamo-hyperpod` (this spec) |
|---|---|---|
| **Cache orchestration** | LMCache (vLLM plugin) | KVBM (Dynamo native) |
| **L2 daemon integration** | Yes — sagemaker-hyperpod:// connector | No — KVBM owns G2 (CPU pinned DRAM) |
| **FSx integration** | LMCache → `file:///mnt/fsx/kvcache` | KVBM G3 → `/mnt/fsx/kv-cache/` |
| **Routing intelligence** | EPP scorers (prefix + load) | Dynamo Router (KV-aware, full KVBM visibility) |
| **P/D disaggregation** | Not in scope | Stretch goal (Stage 7, TCP only) |
| **Autoscaling** | Not in scope (KEDA available) | Planner (native SLA-driven) — stretch goal |
| **Gateway** | Istio Gateway (Envoy) | Dynamo Frontend (built-in) |
| **Service discovery** | K8s native (EPP watches pods) | etcd |
| **Operator** | None (3 Helm charts) | Dynamo K8s Operator |
| **Managed infra used** | L2 daemon + FSx + health checks | FSx only (KVBM bypasses L2 daemon) |

---

## Success Criteria

| Criteria | Stage | Type |
|----------|-------|------|
| Dynamo Operator installs on HyperPod EKS without conflicts | 2 | **Critical** |
| vLLM workers serve inference via Dynamo Frontend | 3 | **Critical** |
| A10G (sm_86) works with NGC Dynamo images | 3 | **Critical** — gates all subsequent stages |
| KVBM G3 writes/reads files on FSx Lustre | 5 | **Critical** — validates the integration thesis |
| KV-aware Router routes correctly using KVBM state | 3, 4, 5 | **Critical** |
| Cross-node KV sharing via FSx (G3) | 5 | Important — validates shared cache value |
| KVBM G2 CPU tier works on g5.4xlarge | 4 | Baseline |
| Dynamo + KVBM metrics accessible | 6 | Important (port-forward acceptable given ADOT gap) |
| Gateway API + IEC + Dynamo CRDs coexist | 2 | Important |
| Disaggregated P/D over TCP (no EFA) | 7 | Stretch goal |
| Planner autoscaling decisions visible | 7 | Stretch goal |

---

## Non-Requirements

- **Performance benchmarking** — deployment validation only, not throughput/latency measurement
- L2 tiered storage daemon integration (Option A — KVBM owns G2)
- Multi-GPU inference (TP=1, single GPU per replica, model is small enough)
- Production TLS / auth / RBAC
- KEDA or external autoscaling (Planner is native)
- ModelExpress (requires NVLink)
- TensorRT-LLM or SGLang backends (vLLM only for comparison parity with llm-d)
- NIXL UCX/RDMA or EFA (g5.4xlarge doesn't support EFA)
- Full AMP/AMG integration (ADOT not available — use port-forward)

---

## Cost Considerations

| Resource | Estimated Cost | Notes |
|----------|---------------|-------|
| ml.g5.4xlarge x3 (4 hrs on-demand) | ~$24 | 3 nodes @ ~$2/hr each |
| ml.m5.2xlarge system nodes | $0 | Already running (shared with llmd-hyperpod) |
| EKS control plane | $0.10/hr | Already running |
| FSx Lustre (1.2 TiB, shared) | $0 | Already provisioned — reuse |
| etcd (on GPU nodes) | $0 | Runs on cluster nodes |
| **Total validation session** | ~$25-30 | Assuming llmd-validation scaled to 0 |

---

## References

- [NVIDIA Dynamo GitHub](https://github.com/ai-dynamo/dynamo) — v1.0.1, Apache 2.0
- [Dynamo Docs](https://docs.nvidia.com/dynamo/) — Installation, K8s operator, KVBM
- [KVBM disk storage source](https://github.com/ai-dynamo/dynamo/blob/main/lib/memory/src/disk.rs) — Lustre-aware fallback paths
- [KV Events API](https://github.com/ai-dynamo/dynamo/blob/main/docs/integrations/kv-events-custom-engines.md) — External cache state integration
- [llmd-hyperpod spec](./llmd-hyperpod.md) — Companion spec for comparison
- [llmd-hyperpod results](../blueprints/llmd-hyperpod/results/progress.md) — Empirical findings
- [nemotron-super lessons](../blueprints/nemotron-super/lessons.md) — Dynamo operational experience (B200)
- [NGC Dynamo Images](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo) — Pre-built containers

---

> **Note**: Operational artifacts belong in `blueprints/dynamo-hyperpod/lessons.md`
> and `blueprints/dynamo-hyperpod/results/`.
