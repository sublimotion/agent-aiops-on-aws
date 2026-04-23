# HyperPod Inference vs Dynamo vs DIY EKS — Operational Overhead Comparison

**Date**: 2026-03-31 (updated with Dynamo findings)
**Status**: Draft inventory — collecting empirical evidence from deployed blueprints

## Three Paths Compared

We now have empirical data from three distinct approaches to GPU serving on AWS:

| Path | Framework | Cluster | KV Cache Strategy | Blueprint(s) |
|------|-----------|---------|-------------------|--------------|
| **A: HyperPod + llm-d** | llm-d + Gateway API + EPP | HyperPod managed | L2 daemon (shm) + L3 FSx via LMCache | llmd-hyperpod |
| **B: HyperPod + Dynamo** | NVIDIA Dynamo + vLLM | HyperPod managed | KVBM tiered (GPU→CPU→FSx) | dynamo-hyperpod |
| **C: DIY EKS** | vLLM/SGLang + various | Self-managed EKS | LMCache, HiCache, or none | glm5-llmd, glm5-lmcache, +10 others |

## Comparable Blueprint Pairs

| Comparison | HyperPod Blueprint | DIY EKS Blueprint | Serving Stack | Hardware |
|------------|--------------------|--------------------|---------------|----------|
| **llm-d (primary)** | llmd-hyperpod | glm5-llmd | llm-d + Gateway API + EPP | g5.4xlarge (A10G) vs p6-b200 (B200) |
| **Dynamo (new)** | dynamo-hyperpod | (no DIY equivalent) | NVIDIA Dynamo + vLLM | g5.4xlarge (A10G) |
| **LMCache/KV tiering** | llmd-hyperpod | glm5-lmcache | vLLM + LMCache | g5.4xlarge vs p6-b200 |
| **Ray Serve (secondary)** | (none yet) | ray-serve-ft | Ray Serve + KubeRay | — vs g5.xlarge |
| **Bare metal baseline** | (none) | devstral-sera | vLLM direct | — vs g7e.24xlarge |

## Artifact Inventory

### llmd-hyperpod (HyperPod Inference)
| Artifact | Path | What It Shows |
|----------|------|---------------|
| Helm values (vLLM + LMCache L0-L3) | `configs/ms-values-hyperpod.yaml` | KV cache config complexity |
| Helm values (GAIE + EPP) | `configs/gaie-values-hyperpod.yaml` | Gateway API routing config |
| FSx PV/PVC manifest | `manifests/fsx-pv-pvc.yaml` | Storage setup (2 resources) |
| HTTPRoute manifest | `manifests/httproute.yaml` | Ingress config |
| Smoke test script | `scripts/smoke_test.sh` | 7-stage validation (21 PASS, 0 FAIL) |
| Cache integration test | `scripts/test_cache_integration.sh` | L2/L3 validation |
| Validation results | `results/progress.md` | Full operational trace with 12 lessons |
| Architecture visual | `results/llmd-hyperpod-architecture.html` | Managed vs self-managed boundaries |
| Tiered cache explainer | `results/llmd-hyperpod-explainer.html` | L0-L3 architecture |
| **No Terraform** | — | Cluster provisioned via HyperPod CLI |

### glm5-llmd (llm-d on DIY EKS)
| Artifact | Path | What It Shows |
|----------|------|---------------|
| Terraform (EKS cluster) | `main.tf` (23 KB) | Full IaC for cluster + networking |
| Terraform outputs | `outputs.tf` | Cluster endpoints, SG IDs |
| Terraform variables | `variables.tf` | Configurable parameters |
| vLLM deployment | `manifests/glm5-deployment.yaml` | Pod spec with GPU requests |
| K8s service | `manifests/glm5-service.yaml` | Service exposure |
| EPP pod | `manifests/glm5-epp.yaml` | Endpoint picker config |
| HTTPRoute | `manifests/glm5-httproute.yaml` | Gateway API routing |
| InferencePool CRD | `manifests/glm5-inferencepool.yaml` | GA API pool config |
| Single vLLM baseline | `manifests/glm5-vllm-single.yaml` | Non-routed baseline |
| LMCache GDS config | `manifests/lmcache-config-gds.yaml` | GPUDirect Storage |
| Redis manifest | `manifests/redis.yaml` | Prefix cache state store |
| EPP scheduler config | `configs/epp-scheduler.yaml` | Scorer weights |
| Lessons learned | `lessons.md` (4.8 KB, 12 lessons) | Operational gotchas |

### glm5-lmcache (LMCache on DIY EKS)
| Artifact | Path | What It Shows |
|----------|------|---------------|
| Terraform (EKS + FSx) | `main.tf` | Cluster + FSx 1.2 TiB + NVMe |
| LMCache CPU config | `manifests/lmcache-config-cpu.yaml` | CPU DRAM offload |
| LMCache GDS config | `manifests/lmcache-config-gds.yaml` | GPUDirect Storage offload |
| LMCache POSIX config | `manifests/lmcache-config-posix.yaml` | POSIX file system offload |
| vLLM deployment | `manifests/glm5-deployment.yaml` | Pod spec |
| Lessons learned | `lessons.md` (8.5 KB, 15 lessons) | AL2023, nodeadm, HiCache |

### dynamo-hyperpod (Dynamo on HyperPod)
| Artifact | Path | What It Shows |
|----------|------|---------------|
| Namespace manifest | `manifests/namespace.yaml` | `dynamo-validation` namespace |
| FSx PV/PVC manifest | `manifests/fsx-pvc.yaml` | Second PV to same FSx (coexists with llmd-validation) |
| etcd manifest | `manifests/etcd.yaml` | Service discovery (replaces NATS in Dynamo v1.0+) |
| Dynamo vLLM worker | `manifests/dynamo-worker.yaml` | `vllm-runtime:1.0.1` with KVBM env vars + FSx mount |
| Smoke test script | `scripts/smoke_test.sh` | 7-stage validation (29 PASS, 0 FAIL, 5 SKIP) |
| Validation results | `results/progress.md` | Full operational trace with 6 findings |
| **No Terraform** | — | Same HyperPod cluster as llmd-hyperpod |
| **No Helm charts** | — | Manual manifests (Dynamo Operator not available for HyperPod) |

### Additional DIY EKS Blueprints (supporting evidence)
| Blueprint | Key Artifacts | Operational Lessons (count) |
|-----------|--------------|---------------------------|
| qwen3-next | `main.tf`, `lessons.md` (16 KB) | 15 lessons — HMA blocks KV offload, TP scaling deadlock |
| qwen3-next-sglang | `main.tf`, `lessons.md` (14 KB) | 12 lessons — Blackwell sm_120, MTP hurts PCIe |
| qwen3-next-g7e | `main.tf`, `lessons.md` (12 KB) | 10 lessons — capacity crisis, state lock |
| qwen3-next-custbench | `main.tf`, `lessons.md` (14 KB) | Customer config reproduction |
| kimi-k2.5 | `main.tf`, `lessons.md` (35 KB) | 20+ lessons — capacity blocks, MoE loading |
| nemotron-super | `main.tf`, `lessons.md` (9.3 KB) | 10 lessons — Mamba hybrid, disagg blocked |
| ray-serve-ft | `terraform/main.tf`, `lessons.md` (4.1 KB) | 8 lessons — stunnel TLS, NLB, GCS FT |
| ray-serve-video | `lessons.md` (6.1 KB) | 8 lessons — runtime_env isolation |
| devstral-sera | `lessons.md` (14 KB) | 12 lessons — NCCL broken, tool parsing |
| ministral-3b | `main.tf`, `lessons.md` (minimal) | Baseline deployment |

## Operational Overhead Dimensions

### 1. Cluster Provisioning

| Dimension | HyperPod + llm-d | HyperPod + Dynamo | DIY EKS | Evidence |
|-----------|------------------|-------------------|---------|----------|
| **IaC required** | 0 lines Terraform | 0 lines Terraform | 7-30 KB Terraform per blueprint | glm5-llmd: 23 KB main.tf |
| **Cluster creation** | `create-cluster` | Same cluster reused | `terraform apply` (EKS module) | dynamo reuses llmd cluster |
| **System node isolation** | RestrictedInstanceGroups | Same (inherited) | Manual taint/toleration | llmd-hyperpod: automatic |
| **K8s version constraint** | EKS <= 1.32 required | Same (inherited) | Any EKS version | llmd-hyperpod lesson #3 |
| **Prerequisites** | HyperPod Helm chart | Same + etcd deployment | None | Dynamo needs etcd for service discovery |
| **AMI selection** | Managed | Same (inherited) | Manual (AL2023 for B200) | glm5-lmcache: AL2 lacks ib_umad |
| **Node bootstrap** | Managed | Same (inherited) | Manual nodeadm MIME | glm5-lmcache: nodeadm vs bootstrap.sh |

### 2. GPU Infrastructure

| Dimension | HyperPod + llm-d | HyperPod + Dynamo | DIY EKS | Evidence |
|-----------|------------------|-------------------|---------|----------|
| **GPU driver** | Managed AMI | Same (inherited) | AMI-provided, verify manually | Both: AMI provides driver |
| **Fabric Manager** | Managed | Same (inherited) | Manual verification needed | glm5-lmcache: AL2 kernel missing ib_umad |
| **NVLink topology** | Auto-discovered | Same (inherited) | Manual verification | HyperPod auto-detects |
| **Health monitoring** | Managed DaemonSet + spare pool | Same (inherited) | None (nvidia-smi manual) | llmd-hyperpod: health-monitoring-agent |
| **GPU node scaling** | Karpenter + Training Plans | Same (inherited) | Manual kubectl scale | qwen3-next: must scale-to-0 |
| **NCCL compatibility** | Managed (version matched) | Same (inherited) | Manual (broken on sm_120) | devstral-sera: blocking issue |
| **Disk pressure risk** | Low (vLLM image ~3 GB) | **High (Dynamo image 12.3 GB)** | Low-Medium | dynamo-hyperpod: DiskPressure taint on g5.4xlarge |

### 3. Storage

| Dimension | HyperPod + llm-d | HyperPod + Dynamo | DIY EKS | Evidence |
|-----------|------------------|-------------------|---------|----------|
| **FSx provisioning** | Auto via FSxLustreConfig | Same FSx reused | Manual Terraform + SGs | llmd-hyperpod: auto |
| **FSx CSI driver** | Pre-installed | Same (inherited) | Manual installation | llmd-hyperpod: already present |
| **Lustre userspace** | Included in managed AMI | Same (inherited) | Missing on AL2023 AMI | nemotron-super: `chroot /host dnf install` |
| **FSx PV/PVC** | Manual (static provisioning) | Manual (second PV, same FS) | Manual (same) | dynamo: separate volumeHandle suffix |
| **FSx permissions** | Root dir 755/root (manual fix) | Init container `chmod 777` | Same issue if using FSx | llmd-hyperpod lesson #11 |
| **FSx cache coexistence** | `/mnt/fsx/kvcache` (LMCache) | `/mnt/fsx/kv-cache/` (KVBM) | Varies | Both dirs visible, no conflict |
| **Lustre compat flags** | N/A (LMCache handles) | `ZEROFILL_FALLBACK` + `DISABLE_O_DIRECT` | N/A | Lustre lacks fallocate + strict alignment |
| **NVMe setup** | Not applicable (g5 EBS) | Not applicable (g5 EBS) | Manual RAID0 in user data | glm5-lmcache: NVMe RAID0 |
| **Model staging** | HF download in pod | HF download in pod | NVMe pre-staging or FSx | qwen3-next: /mnt/nvme/models/ |

### 4. KV Cache / Tiered Storage

| Dimension | HyperPod + llm-d | HyperPod + Dynamo | DIY EKS | Evidence |
|-----------|------------------|-------------------|---------|----------|
| **Architecture** | LMCache (L0-L3) via llm-d | KVBM (G1 GPU→G2 CPU→G3 disk) | Varies | Fundamentally different approaches |
| **L2/G2 daemon** | Auto-deployed ai-toolkit DaemonSet | **None needed** (KVBM built into worker) | Not available | Dynamo: self-contained per-worker |
| **L2/G2 shared memory** | Managed (config gotchas: name, uid) | N/A (CPU pinned DRAM, no shm) | N/A | llmd-hyperpod: name/uid mismatch |
| **L2/G2 connection** | `sagemaker-hyperpod://` URL | Env var `DYN_KVBM_CPU_CACHE_GB` | N/A | Dynamo: simpler config |
| **L3/G3 FSx cache** | FSx auto-provisioned, manual mount | FSx reused, manual mount | FSx manual provision + mount | Similar PV/PVC effort |
| **Cache activation** | LMCache sidecar in Helm values | **KVBM not active in vllm-runtime** | Depends on framework | Dynamo needs full stack (Frontend+Router+Planner) |
| **Orchestration deps** | llm-d EPP + Gateway API + Istio | etcd for service discovery | None or Redis | Dynamo: lighter deps but full stack not tested |
| **LMCache compat** | Works (standard attn only) | N/A (uses KVBM instead) | Blocked on NSA/MLA | glm5-lmcache: LMCache PR #2629 |
| **Alternative: HiCache** | Not tested | N/A | Works with NSA/MLA | glm5-lmcache: 2.86x vs baseline |

### 5. Networking & Ingress

| Dimension | HyperPod + llm-d | HyperPod + Dynamo | DIY EKS | Evidence |
|-----------|------------------|-------------------|---------|----------|
| **Gateway API CRDs** | Installed via llm-d-infra Helm | Not needed (no Gateway API) | Manual installation | Dynamo uses internal routing |
| **Istio/Envoy** | Installed via llm-d-infra Helm | Not needed | Manual (skip-crds gotchas) | Dynamo: no service mesh required |
| **NLB** | Auto via Istio Gateway | Manual `type: LoadBalancer` svc | Manual service annotations | dynamo: NLB annotation on dynamo-frontend svc |
| **ext-proc timeout** | Configured in GAIE values | N/A | Manual (200ms too short) | Dynamo doesn't use ext-proc |
| **CRD coexistence** | Gateway API + IEC coexist | Gateway API + IEC coexist | Only Gateway API CRDs | dynamo-hyperpod stage 7: verified |
| **Service discovery** | Kubernetes native (via EPP) | etcd cluster (manual deploy) | Kubernetes native | Dynamo: etcd is an extra dependency |

### 6. Observability

| Dimension | HyperPod + llm-d | HyperPod + Dynamo | DIY EKS | Evidence |
|-----------|------------------|-------------------|---------|----------|
| **Metrics CRDs** | PodMonitor/ServiceMonitor pre-installed | Same (inherited) | Manual | llmd-hyperpod: CRDs present |
| **Prometheus scraping** | ADOT expected but NOT FOUND | Same gap (ADOT absent) | Manual Prometheus operator | Both HP paths: empirical correction |
| **vLLM /metrics** | Exposed, not scraped | Exposed, not scraped | Exposed, not scraped | All: same gap |
| **GPU metrics** | DCGM expected via ADOT | Same gap | nvidia-smi manual | All: partial |
| **Prefix cache metrics** | Via LMCache stats | vLLM native `prefix_cache_queries_total` | Varies | dynamo: confirmed exposed |

### 7. Operational Gotchas (Count by Category)

| Category | HyperPod + llm-d | HyperPod + Dynamo | DIY EKS |
|----------|------------------|-------------------|---------|
| **Permissions/UIDs** | 2 (shm uid, FSx root) | 1 (FSx root — init container) | 1 (FSx root) |
| **Config mismatches** | 2 (shm name, env var ordering) | 1 (NGC entrypoint needs command override) | 3+ (tool-call-parser, attn backend, fp8-gemm) |
| **Version constraints** | 1 (EKS <= 1.32) | 1 (EKS <= 1.32, inherited) | 3+ (AL2023, NCCL, SGLang tags) |
| **CRD/API issues** | 1 (EnableFailed non-blocking) | 0 (no CRDs needed) | 3 (EnvoyExtensionPolicy, GatewayClass, InferencePool) |
| **Capacity/scaling** | 0 | 1 (12.3 GB image causes DiskPressure on small instances) | 3 (scale-to-0, capacity blocks, termination delay) |
| **Framework compat** | 0 | 1 (KVBM needs full Dynamo stack, not just vllm-runtime) | 4+ (LMCache+NSA, Mamba+disagg, HMA+offload, MTP+PCIe) |
| **TOTAL** | ~6 | ~5 | ~17+ |

## Key Artifacts for Comparison Narrative

### "What you get for free" with HyperPod (both paths)
1. **Cluster provisioning**: No Terraform needed (saved 7-30 KB IaC per blueprint)
2. **System node isolation**: RestrictedInstanceGroups vs manual taints
3. **FSx Lustre**: Auto-provisioned 1.2 TiB with CSI driver pre-installed
4. **Health monitoring**: Automated node health checks + spare pool
5. **AMI management**: Correct kernel, drivers, Lustre client all managed
6. **Fabric Manager**: No AL2 vs AL2023 kernel module debugging

### "What you still have to do" — llm-d path
1. **HyperPod Helm prereqs**: Must install before cluster creation
2. **llm-d Helm charts**: 3 charts (infra, GAIE, modelservice) — same as DIY
3. **KV cache connector config**: sagemaker-hyperpod:// URL, shm name, env var ordering
4. **L2 daemon gotchas**: uid mismatch, shm name discovery
5. **Permission fixes**: chmod 666 on shm, chmod 777 on FSx subdir
6. **Observability**: ADOT not auto-installed (gap vs docs)

### "What you still have to do" — Dynamo path
1. **etcd deployment**: Manual (Dynamo needs etcd for service discovery)
2. **NGC entrypoint override**: `command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]` required
3. **Full Dynamo stack**: vllm-runtime alone doesn't activate KVBM — need Frontend + Router + Planner
4. **Disk management**: 12.3 GB NGC image causes DiskPressure on small instances
5. **FSx PV/PVC**: Second PV with unique volumeHandle suffix
6. **Observability**: Same ADOT gap as llm-d path

### Dynamo vs llm-d on HyperPod — Key Differences

| Dimension | llm-d | Dynamo | Winner |
|-----------|-------|--------|--------|
| **Setup complexity** | 3 Helm charts + GAIE + EPP + Istio | 4 kubectl manifests + etcd | Dynamo (simpler) |
| **KV cache architecture** | External: L2 daemon + LMCache sidecar | Internal: KVBM built into worker | Dynamo (self-contained) |
| **Routing** | Gateway API + EPP (production-grade) | Dynamo Router + Planner | llm-d (more mature) |
| **Model compatibility** | Limited (LMCache blocks NSA/MLA) | Broader (KVBM is model-agnostic) | Dynamo (in theory) |
| **HyperPod integration** | Deep (L2 daemon, IEC CRDs, Operator) | Shallow (just cluster + FSx) | Depends on use case |
| **Operational gotchas** | ~6 (shm, uid, config ordering) | ~5 (entrypoint, disk, KVBM activation) | Comparable |
| **Image size** | ~3 GB (vLLM) + ~1 GB (LMCache) | 12.3 GB (Dynamo vllm-runtime) | llm-d (smaller) |
| **Maturity** | GA (llm-d 1.3+, EPP v1.3.1) | Early (v1.0.1, KVBM needs full stack) | llm-d (more mature) |

### "What breaks more often" on DIY EKS
1. **AMI/kernel issues**: AL2 vs AL2023, ib_umad, Lustre userspace (3 blueprints hit this)
2. **Framework compatibility**: LMCache+NSA, Mamba+disagg, HMA+offload (4 blueprints)
3. **CRD management**: EnvoyExtensionPolicy, GatewayClass manual creation (2 blueprints)
4. **Capacity management**: scale-to-0 for config changes, capacity block limitations (3 blueprints)
5. **NCCL on Blackwell**: Critical blocker for distributed training (2 blueprints)

## Quantitative Comparison (Where Data Exists)

| Metric | HyperPod + llm-d | HyperPod + Dynamo | DIY EKS | Source |
|--------|------------------|-------------------|---------|--------|
| Terraform lines | 0 | 0 | 7,500-30,000 | Per blueprint main.tf |
| K8s manifests authored | 2-3 | 4 | 5-11 | Manifest count per blueprint |
| Helm charts required | 3 (infra + GAIE + modelservice) | 0 | 3+ | llm-d needs Helm, Dynamo is raw manifests |
| Lessons learned (gotchas) | 12 | 6 | 12-35 per blueprint | lessons.md / progress.md |
| Smoke test stages | 7 (21 PASS) | 7 (29 PASS, 5 SKIP) | Not standardized | smoke_test.sh |
| Cold start (small model) | ~2 min | ~4 min (12.3 GB image pull) | ~16 min (GLM-5 on B200) | progress.md |
| Container image size | ~3 GB | 12.3 GB | ~3 GB | kubectl describe node |
| KV cache hit rate | 59.5% (L0 prefix) | N/A (KVBM not active) | Varies | progress.md |

## Next Steps

- [ ] Deploy full Dynamo stack (Frontend + Router + Planner) to activate KVBM
- [ ] Benchmark Dynamo KVBM hit rate vs llm-d LMCache hit rate (apples-to-apples on same cluster)
- [ ] Deploy same model (e.g., Qwen3-0.6B) on DIY EKS for three-way comparison
- [ ] Time the end-to-end setup from scratch (cluster create → first inference) for all three paths
- [ ] Count kubectl/helm commands required for each path
- [ ] Measure Day 2 ops: config change, scaling, model swap, node failure recovery
- [ ] Compare cost (HyperPod pricing vs raw EKS + addons)
- [ ] Test observability parity (install ADOT on HyperPod, Prometheus on DIY)
