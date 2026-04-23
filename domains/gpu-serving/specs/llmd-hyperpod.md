# llm-d on SageMaker HyperPod EKS — L2/L3 KV Cache Integration Spec

## Status: DRAFT (2026-03-31)

## Overview

Validate **llm-d (Path B) running on SageMaker HyperPod EKS infrastructure**, specifically testing whether llm-d's EPP routing and vLLM replicas can use HyperPod's managed L2 tiered storage daemon (port 9200) and L3 FSx Lustre for cross-replica KV cache sharing. This is the untested integration point identified in the HyperPod architecture analysis — llm-d has been proven on vanilla EKS (`glm5-llmd`) and the HyperPod Operator has been proven separately (`qwen3-32b-hyperpod`), but the combination has not been validated.

**Why this matters:**
- The HyperPod architecture doc marks L2 daemon compatibility with llm-d as **"untested"**
- If llm-d can use HyperPod's managed L2/L3 cache, customers get the best of both paths: llm-d's pluggable EPP routing + HyperPod's managed cache infrastructure + deep health checks + spare pool
- This validates the "shared infrastructure" thesis from the three-path architecture — that ~60% of HyperPod's managed components are reusable across all serving paths

**Prior art:**
- `glm5-llmd` — llm-d EPP + vLLM on vanilla EKS with p6-b200. Validated: InferencePool v1 GA, EPP scorer pipeline, Envoy Gateway integration. **No HyperPod infrastructure.**
- `qwen3-32b-hyperpod` — HyperPod Inference Operator (Path A) on g6e. Validated: L1+L2 managed tiered storage, prefix-aware routing, IEC CRD. **No llm-d.**
- AWS blog "Introducing disaggregated inference on AWS powered by llm-d" — P/D disaggregation on HyperPod p6-b200. Validated: NIXL/EFA, 50% throughput increase. **No L2/L3 cache integration testing.**

**Model**: Qwen3-32B-FP8 — same model as the prior HyperPod experiment for direct comparison. Small dense GQA, full KV cache compatibility, no MLA/NSA blockers. FP8 TP=1 enables multi-replica routing tests on a single node.

---

## What We're Validating

### L2 Integration: HyperPod Managed Tiered Storage + llm-d

The L2 daemon runs on each HyperPod GPU node at port 9200, using `/dev/shm/ai_toolkit_cache` for IPC with vLLM pods. Today, only the HyperPod Inference Operator configures this connection (injecting `--kv-transfer-config` with `LMCacheConnectorV1` and `LMCACHE_REMOTE_URL=sagemaker-hyperpod://$(NODE_IP):9200`).

**Hypothesis**: llm-d-managed vLLM pods can be manually configured with the same LMCache connector + env vars to use the L2 daemon, without the Inference Operator. The daemon is a per-node service — it doesn't know or care which orchestration layer spawned the vLLM pods.

**Test**: Deploy vLLM replicas via llm-d Helm charts (not IEC CRD) with:
- `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`
- `LMCACHE_LOCAL_CPU=True` (L1)
- `LMCACHE_REMOTE_URL=sagemaker-hyperpod://$(NODE_IP):9200` (L2)
- Mount `/dev/shm/ai_toolkit_cache` as hostPath volume

Verify cross-pod KV sharing via the L2 daemon by sending identical prefixed requests to different replicas and measuring TTFT improvement.

### L3 Integration: FSx Lustre as Cross-Node KV Cache

FSx Lustre is framework-agnostic (POSIX mount). The architecture doc confirms all three paths can use it as disk-tier storage. However, LMCache's serialization format on FSx has only been tested under the Operator path.

**Test**: Configure LMCache with `LMCACHE_REMOTE_URL=file:///mnt/fsx/kvcache` on llm-d-managed pods. Verify:
1. KV blocks written to FSx are readable by pods on different nodes
2. EPP's PrecisePrefixCacheScorer correctly routes to pods with FSx-cached prefixes
3. No serialization/deserialization errors when L2 daemon and FSx coexist

### llm-d EPP + HyperPod Infrastructure

Validate that the llm-d EPP (Envoy ext_proc) works alongside HyperPod's managed components:
- EPP pods schedule on HyperPod system nodes (alongside KEDA, cert-manager, ALB controller)
- EPP can discover vLLM pods running on HyperPod GPU nodes
- Prometheus metrics from EPP are scraped by HyperPod's ADOT collector → AMP
- Gateway API InferencePool CRDs coexist with the Inference Operator's IEC CRDs (no conflicts)

---

## Components

### 1. Compute — HyperPod EKS Cluster

- **Platform**: SageMaker HyperPod with EKS orchestrator
- **Cluster**: `finetune-g5-cluster` / EKS `finetune-eks`
- **Instance**: `ml.g5.2xlarge` — 1× A10G (24 GB VRAM), dedicated `llmd-validation` instance group
- **Existing workload**: `finetune-runner` on separate `g5-workers` instance group (g5.xlarge) — do not interfere
- **Auto node recovery**: enabled
- **Region**: us-east-1
- **Namespace**: `llmd-validation` (dedicated — isolates from existing workloads on the cluster)

### 2. Model

| Property | Value |
|----------|-------|
| **Model ID** | `meta-llama/Llama-3.2-3B-Instruct` |
| **Parameters** | 3B (dense) |
| **Attention** | Standard GQA |
| **Precision** | BF16 (~6 GB) |
| **TP** | 1 (single A10G per replica) |
| **Max replicas** | 1 (single GPU on g5.2xlarge) |
| **Serving** | vLLM via `ghcr.io/llm-d/llm-d-cuda:v0.5.1` (official llm-d CUDA container) |

> **Model choice**: Llama-3.2-3B-Instruct is the llm-d default for CPU/small-GPU configs. At ~6GB it leaves ~18GB KV cache headroom on A10G — enough to stress-test L2/L3 cache integration with meaningful prefix sizes. The goal is infrastructure validation, not model quality.

### 3. Networking

- **Gateway**: Istio (llm-d default) or Envoy Gateway — provisioned via llm-d helmfile
- **Ingress**: NLB via Gateway API (not ALB — llm-d uses Gateway API, not Ingress)
- **EPP**: ext_proc on port 9002
- **NIXL**: Not required (no P/D disaggregation — colocated replicas only)

### 4. Storage

- **Model weights**: S3 → PVC (reuse `s3://hyperpod-eks-test-bucket-495365983931/qwen3-32b/` from prior experiment)
- **L2 KV cache**: HyperPod managed tiered storage daemon (port 9200, `/dev/shm/ai_toolkit_cache`)
- **L3 KV cache**: FSx Lustre — PersistentVolume via FSx CSI driver, mounted at `/mnt/fsx`

### 5. Observability

- **AMP/AMG**: HyperPod-managed — ADOT scrapes vLLM metrics on port 8000 + EPP metrics on port 9090
- **Dashboards**: Pre-built GPU dashboards + custom llm-d routing dashboards (prefix cache hit rate, scorer decisions)

---

## Validation Stages

This is a deployment validation blueprint, not a benchmark. Each stage smoke-tests a specific integration point. Stages are additive — each builds on the previous one's validated state.

### Stage 1: HyperPod Infrastructure Discovery

Verify the HyperPod-managed components that llm-d will depend on.

| Check | Command / Method | Pass Criteria |
|-------|-----------------|---------------|
| L2 daemon running on GPU nodes | `kubectl get ds -n kube-system` or `ssh → ss -tlnp \| grep 9200` | Port 9200 listening on each GPU node |
| L2 daemon IPC path exists | `kubectl exec -it <gpu-pod> -- ls /dev/shm/ai_toolkit_cache` | Directory exists and is writable |
| Inference Operator running | `kubectl get pods -n hyperpod-inference-system` | Controller manager pod Running |
| ADOT collector running | `kubectl get pods -n kube-system -l app=adot` | Collector pod Running |
| FSx CSI driver installed | `kubectl get csidriver fsx.csi.aws.com` | CSI driver registered |
| FSx PV available | `kubectl get pv,pvc -A \| grep fsx` | PV Bound (or provision new one) |
| System node taints | `kubectl describe nodes -l node.kubernetes.io/instance-type=m5.2xlarge \| grep Taint` | Document taints for EPP tolerations |
| GPU node labels | `kubectl get nodes -l node.kubernetes.io/instance-type=ml.g6e.48xlarge --show-labels` | Document labels for vLLM pod affinity |

### Stage 2: llm-d Gateway Stack Deployment

Install the llm-d gateway provider (Istio or KGateway) on HyperPod EKS. Validate it coexists with HyperPod's managed add-ons.

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| Gateway API CRDs installed | `kubectl get crd \| grep gateway` | `gateways.gateway.networking.k8s.io` present |
| Istio control plane running | `kubectl get pods -n istio-system` | `istiod` Running |
| Gateway resource created | `kubectl get gateways` | Gateway `PROGRAMMED=True` |
| NLB provisioned | `kubectl get svc -n istio-system -l istio=gateway` | EXTERNAL-IP assigned |
| No conflicts with ALB controller | `kubectl get ingress -A` | Existing HyperPod ALB Ingresses unaffected |
| InferencePool CRD installed | `kubectl get crd inferencepools.inference.networking.k8s.io` | CRD exists alongside IEC CRDs |

### Stage 3: vLLM Deployment via llm-d (Baseline — L0 Only)

Deploy Qwen3-32B-FP8 replicas using llm-d Helm charts (not the Inference Operator). Validate basic serving.

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| Helmfile applies cleanly | `helmfile apply` | No errors, deployment created |
| vLLM pods scheduled on GPU nodes | `kubectl get pods -o wide` | Pods Running on g6e nodes |
| vLLM health endpoint | `curl <pod-ip>:8000/health` | 200 OK |
| Chat completions work | `curl <gateway>/v1/chat/completions` (via NLB) | Valid response with model output |
| Prefix caching enabled | `curl <pod-ip>:8000/metrics \| grep prefix_cache` | `prefix_cache_hit_total` metric present |
| EPP deployed and connected | `kubectl get pods -l app=epp` + `kubectl logs <epp-pod>` | EPP Running, gRPC stream established |
| EPP ext_proc routing works | Send 2 identical requests, check EPP logs for routing decision | Second request routes to same pod (prefix hit) |
| InferencePool selects vLLM pods | `kubectl describe inferencepool` | Endpoints list matches vLLM pod IPs |

### Stage 4: L2 Integration — Managed Tiered Storage Daemon

Connect llm-d vLLM pods to HyperPod's L2 daemon. This is the **critical untested integration**.

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| Add LMCache connector to vLLM args | Patch deployment with `--kv-transfer-config` | Pods restart successfully, no crash loop |
| Set `LMCACHE_LOCAL_CPU=True` | Patch env | L1 CPU offload active in vLLM logs |
| Set `LMCACHE_REMOTE_URL=sagemaker-hyperpod://$(NODE_IP):9200` | Patch env with fieldRef | vLLM connects to L2 daemon (check logs) |
| Mount `/dev/shm/ai_toolkit_cache` | Add hostPath volume + volumeMount | Mount succeeds, IPC path accessible |
| L2 daemon accepts connection | `kubectl logs <l2-daemon-pod>` or daemon metrics | No auth rejection, connection established |
| KV write to L2 | Send request, check L2 daemon metrics/logs for write | KV blocks stored in L2 |
| Cross-pod KV read via L2 | Send identical prefix to different replica, check TTFT | Second replica shows cache hit (faster TTFT) |
| LMCache version compatibility | Compare `pip show lmcache` in vLLM pod vs L2 daemon | Versions compatible (same major.minor) |

### Stage 5: L3 Integration — FSx Lustre Cross-Node Cache

Add FSx Lustre as the disk-tier KV cache backend.

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| FSx PVC mounted in vLLM pods | Add PVC volumeMount at `/mnt/fsx` | Mount succeeds, writable |
| LMCache writes to FSx | Set `LMCACHE_REMOTE_URL=file:///mnt/fsx/kvcache` or add as additional tier | Files appear in `/mnt/fsx/kvcache/` |
| KV blocks readable from different pod | Pod A writes, Pod B reads same prefix | No deserialization errors |
| L2 + L3 coexist | Enable both L2 daemon and L3 FSx | No conflicts, both tiers active |
| FSx file cleanup | Check that evicted KV blocks are removed | No unbounded disk growth |

### Stage 6: Observability Integration

Validate that llm-d metrics flow into HyperPod's managed observability stack.

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| vLLM metrics scraped by ADOT | Check AMP for `vllm_*` metrics | Metrics visible in Managed Prometheus |
| EPP metrics scraped by ADOT | Check AMP for `epp_*` or `inference_*` metrics | Metrics visible (may need ServiceMonitor) |
| GPU metrics from DCGM | Check AMP for `DCGM_*` metrics | Per-GPU utilization visible |
| Managed Grafana dashboards load | Open AMG console | Pre-built dashboards render with data |
| Custom Grafana panel for EPP | Add panel querying EPP prefix cache metrics | Cache hit/miss rates visible |

### Stage 7: CRD Coexistence Test

Validate that llm-d Gateway API resources and HyperPod Operator resources can coexist on the same cluster.

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| Deploy IEC CRD alongside InferencePool | `kubectl apply` both CRDs | No admission webhook conflicts |
| Operator doesn't interfere with llm-d pods | Check operator logs for warnings/errors | Operator ignores non-IEC pods |
| llm-d doesn't interfere with operator pods | Check EPP logs, Gateway status | EPP only manages its own InferencePool |
| Both endpoints respond independently | Curl both endpoints | Each returns correct model response |

---

## Smoke Test Script

A single script (`scripts/smoke_test.sh`) that runs through all validation stages and produces a pass/fail report:

```bash
#!/bin/bash
# Usage: ./smoke_test.sh [stage]
# Runs all stages if no argument, or a specific stage (1-7)

# Stage 1: Infrastructure discovery
# Stage 2: Gateway stack
# Stage 3: vLLM baseline
# Stage 4: L2 integration (critical)
# Stage 5: L3 integration
# Stage 6: Observability
# Stage 7: CRD coexistence
```

---

## Success Criteria

| Criteria | Stage | Type |
|----------|-------|------|
| L2 daemon accepts connections from non-operator pods | 4 | **Critical** — blocks the entire integration thesis |
| LMCache connector version compatibility | 4 | **Critical** — blocks L2/L3 functionality |
| Cross-pod KV sharing via L2 works | 4 | **Critical** — validates L2 integration value |
| L3 FSx writes/reads work from llm-d pods | 5 | Important — validates cross-node cache |
| EPP metrics flow to AMP | 6 | Important — validates observability reuse |
| Gateway API + IEC CRDs coexist | 7 | Important — validates multi-path deployment |
| llm-d chat completions via NLB | 3 | Baseline — must work for anything else to matter |
| No operator interference with llm-d pods | 7 | Safety — must not break existing operator workloads |

---

## Deployment Workflow

Execute stages sequentially. Each stage has a validation gate — do not proceed until the current stage passes.

### 0. Prerequisites

```bash
export REGION=us-east-1
# Discover the EKS cluster name from the HyperPod cluster
export HYPERPOD_CLUSTER_NAME="<your-hyperpod-cluster>"
export EKS_CLUSTER_NAME=$(aws --region $REGION sagemaker describe-cluster \
  --cluster-name $HYPERPOD_CLUSTER_NAME \
  --query 'Orchestrator.Eks.ClusterArn' --output text | cut -d'/' -f2)
aws eks update-kubeconfig --name $EKS_CLUSTER_NAME --region $REGION

# Use a dedicated namespace to avoid interfering with existing workloads
export LLMD_NS=llmd-validation
kubectl create namespace $LLMD_NS --dry-run=client -o yaml | kubectl apply -f -
```

Tools: `kubectl`, `helm`, `helmfile`, `aws` CLI, `curl`, `jq`.

> **Isolation**: All llm-d resources deploy into the `llmd-validation` namespace. The existing experiment in `default` or other namespaces is unaffected. Gateway API CRDs are cluster-scoped but additive (no overwrites). Istio control plane installs in `istio-system` — check if it's already present before installing.

### Stage 1 → Stage 2 → Stage 3: Foundation

1. Run infrastructure discovery checks (document L2 daemon state, taints, labels)
2. Install llm-d gateway provider:
   ```bash
   git clone https://github.com/llm-d/llm-d.git
   cd llm-d/guides/prereq/gateway-provider
   ./install-gateway-provider-dependencies.sh
   helmfile apply -f istio.helmfile.yaml
   ```
3. Customize `ms-inference-scheduling/values.yaml` for Qwen3-32B-FP8:
   - Image: `lmcache/vllm-openai:latest`
   - Model: `RedHatAI/Qwen3-32B-FP8-dynamic`
   - TP=1, `nvidia.com/gpu: "1"`, replicas: 2-4
   - `--enable-prefix-caching`, `--max-model-len 32000`
   - Node selector: g6e GPU nodes
4. `helmfile apply` — verify pods Running, health 200, chat completions via NLB

**Gate**: Basic chat completion works end-to-end through NLB → Istio → EPP → vLLM.

### Stage 4: L2 Integration (Critical Path)

1. Inspect L2 daemon: version, protocol, auth mechanism
2. Prepare overlay values or kustomize patch adding:
   - `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`
   - `LMCACHE_LOCAL_CPU=True`
   - `LMCACHE_REMOTE_URL=sagemaker-hyperpod://$(NODE_IP):9200` (using `fieldRef: status.hostIP`)
   - `hostPath` volume for `/dev/shm/ai_toolkit_cache`
3. Apply and watch for crash loops — if L2 daemon rejects, document the rejection mechanism
4. Smoke test: send identical prefix twice to different replicas, compare TTFT

**Gate**: L2 daemon accepts llm-d pods AND cross-pod KV sharing is confirmed.

### Stage 5: L3 Integration

1. Provision FSx Lustre PV (or reuse existing) and bind PVC
2. Add FSx mount to vLLM pods at `/mnt/fsx`
3. Configure LMCache FSx backend
4. Smoke test: write KV from pod A, read from pod B

**Gate**: KV blocks persist to FSx and are readable cross-pod.

### Stage 6–7: Observability + Coexistence

1. Verify metrics in AMP, create Grafana panel for EPP
2. Deploy IEC CRD for same model alongside llm-d InferencePool
3. Verify both paths serve independently without interference

---

## Known Risks and Blockers

| Risk | Severity | Detail | Mitigation |
|------|----------|--------|------------|
| **L2 daemon rejects non-operator pods** | HIGH | The L2 daemon may validate that connecting pods were created by the Inference Operator (e.g., checking labels or annotations). If so, llm-d pods would be rejected. | Test with a bare vLLM pod first (no llm-d) to isolate whether the daemon has auth. If rejected, check if specific labels/annotations can be added to llm-d pods. |
| **LMCacheConnectorV1 version mismatch** | MEDIUM | The L2 daemon runs a specific LMCache version. `llm-d-aws` may use a different version with incompatible serialization. | Pin container image tag to match the operator DLC's LMCache version. Check `kubectl exec` into L2 daemon for version. |
| **Operator always injects KV config** | MEDIUM | The operator injects `--kv-transfer-config` even when `enableL1Cache: false`. If llm-d pods also set this flag, there could be conflicts. | llm-d pods are not managed by the operator — no injection. Only relevant for Config D comparison. |
| **Gateway API CRD conflicts** | LOW | llm-d uses Gateway API InferencePool (v1). The operator uses IEC CRDs. Both may install Gateway API base CRDs. | Install Gateway API CRDs once before either deployment. Both can reference them. |
| **EPP on system nodes** | LOW | HyperPod system nodes may have taints that prevent non-operator pods. | Add tolerations matching HyperPod system node taints. |
| **g6e PCIe — no NIXL** | INFO | g6e has no NVLink. NIXL/EFA not relevant for this spec (colocated replicas, no P/D). | P/D disaggregation is out of scope. Focus is cache integration. |
| **Operator DLC vLLM version** | MEDIUM | Operator DLC uses vLLM 0.11.1 which may lack Qwen3 support. | `llm-d-aws` ships a newer vLLM. Operator baseline uses whatever DLC version is current. |

---

## Non-Requirements

- **Performance benchmarking** — this is deployment validation and smoke testing, not throughput/latency measurement
- P/D disaggregation (no NIXL needed — colocated replicas only)
- Expert parallelism (dense model, not MoE)
- Multi-node inference (fits on 1 GPU per replica)
- Scale-to-zero / KEDA autoscaling
- Fine-tuning or training
- Production-grade TLS / auth
- Managed llm-d operator integration (this is DIY llm-d on HP infra — the managed integration is the Operator v2 proposal)
- Comprehensive workload testing (only smoke tests to validate integration points)

---

## Cost Considerations

| Resource | Estimated Cost | Notes |
|----------|---------------|-------|
| ml.g6e.48xlarge (4 hrs on-demand) | ~$30-60 | Reuse existing cluster |
| EKS control plane | $0.10/hr | Already running |
| FSx Lustre (1.2 TiB, 4 hrs) | ~$2 | Minimal for cache validation |
| **Total benchmark session** | ~$35-65 | |

---

## References

- [HyperPod Architecture Deep Dive](../../docs/HyperPod-Inference-Architecture.md) — Three-path story, L2/L3 cache tiers, component placement
- [Managed llm-d Proposal](../../docs/Proposal-Managed-llm-d-HyperPod-Inference-Operator.md) — Operator v2 vision
- [AWS Blog: Introducing disaggregated inference on AWS powered by llm-d](https://aws.amazon.com/blogs/machine-learning/introducing-disaggregated-inference-on-aws-powered-by-llm-d/)
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Helmfile guides, gateway provider setup
- [glm5-llmd lessons](../blueprints/glm5-llmd/lessons.md) — EPP, InferencePool, Envoy Gateway operational findings
- [qwen3-32b-hyperpod (Kiro)](https://github.com/...) — Prior HyperPod operator experiment on same model/instance

---

> **Note**: Operational artifacts belong in `blueprints/llmd-hyperpod/lessons.md`
> and `blueprints/llmd-hyperpod/results/`.
