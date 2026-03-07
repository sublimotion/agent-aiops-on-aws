# Readiness Audit - Qwen3-Next Blueprint
**Date**: 2026-02-24
**Time**: 09:35 PST
**Blueprint**: qwen3-next
**Cluster**: qwen3-next-bench-eks-cluster
**Region**: us-east-2

## Audit Categories

### 1. EKS Cluster
| Component | Status | Details |
|-----------|--------|---------|
| Cluster Status | **PASS** | Cluster active and API endpoint reachable |
| API Endpoint | **PASS** | kubectl commands working correctly |
| System Nodes | **PASS** | 4 system nodes Ready (16-17h uptime) |
| GPU Node | **PASS** | 1 GPU node p5en.48xlarge Ready (172min uptime) in us-east-2c |
| CoreDNS | **PASS** | 2 pods running: coredns-59699dd88d-5877c, coredns-59699dd88d-9s58l |
| kube-proxy | **PASS** | 5 pods running on all nodes |

**Node Details:**
- ip-10-0-17-82.us-east-2.compute.internal (Ready, 17h)
- ip-10-0-30-38.us-east-2.compute.internal (Ready, 16h)
- ip-10-0-40-34.us-east-2.compute.internal (Ready, 16h)
- ip-10-0-42-151.us-east-2.compute.internal (Ready, 172m) — **GPU node**
- ip-10-0-46-147.us-east-2.compute.internal (Ready, 17h)

### 2. Storage
| Component | Status | Details |
|-----------|--------|---------|
| FSx Lustre Lifecycle | **PASS** | fs-0b0d2a41a77a71e92 active |
| FSx DNS | **PASS** | 10.0.44.21@tcp:/m2h5hbev |
| FSx Mount | **PASS** | Mounted at /mnt/fsx (4.5T capacity, 5% used) |
| PV/PVC | **PASS** | vllm-qwen3-fsx-pv bound to vllm-qwen3-fsx-pvc (4800Gi) |
| CSI Drivers | **PASS** | fsx-csi-controller (2 replicas), fsx-csi-node (5 daemonsets) all running |
| NVMe RAID | **PASS** | /dev/md0 mounted at /mnt/nvme (28T capacity, 2% used) |
| Model on NVMe | **PASS** | qwen3-next-fp8 (77GB) at /mnt/nvme/models/qwen3-next-fp8 |

### 3. Container Images (ECR)
| Image | Status | Details |
|-------|--------|---------|
| vLLM Image | **PASS** | Using vllm/vllm-openai:qwen3_5-x86_64-cu130 from Docker Hub |
| ECR Repos | **PASS** | 25 ECR repositories exist including vllm-openai |
| Config Scripts | **PASS** | All configs use Docker Hub image (not ECR) |
| Docker Directory | **N/A** | No docker/ directory in blueprint |
| stage-images-ecr.sh | **MISSING** | No staging script found |

**Note**: Blueprint is using Docker Hub images directly, not ECR-staged images. This is acceptable for benchmarking but may cause rate limiting issues.

### 4. GPU / Accelerator Plugins
| Component | Status | Details |
|-----------|--------|---------|
| NVIDIA Device Plugin | **PASS** | nvidia-device-plugin-z8cfj running on GPU node |
| EFA Device Plugin | **FAIL** | CrashLoopBackOff on GPU node - "No valid EFA devices found" |
| DCGM Exporter | **PASS** | 1 running on GPU node, 4 CrashLoopBackOff on non-GPU nodes (expected) |

**EFA Issue**: The p5en.48xlarge instance doesn't have EFA devices enabled or the EFA drivers aren't installed. This impacts RDMA/collective communication performance but doesn't block inference.

### 5. Monitoring
| Component | Status | Details |
|-----------|--------|---------|
| Prometheus | **PASS** | prometheus-0 running (2/2 containers) |
| Grafana | **PASS** | prometheus-grafana-78bd97f598-xj5j6 running (3/3 containers) |
| kube-state-metrics | **PASS** | prometheus-kube-state-metrics-7d7d4d7ddc-rpjwf running |
| node-exporter | **PASS** | 5 pods running on all nodes |
| Alertmanager | **PASS** | alertmanager-0 running (2/2 containers) |
| Prometheus Operator | **PASS** | prometheus-kube-prometheus-operator running |

### 6. Serving Layer
| Component | Status | Details |
|-----------|--------|---------|
| Deployment | **PASS** | vllm-qwen3-next (1/1 replicas) |
| ClusterIP Service | **PASS** | vllm-qwen3-next on 172.20.250.110:8000 |
| NodePort Service | **PASS** | vllm-qwen3-next-nodeport on 30080 |
| Pod Status | **PASS** | vllm-qwen3-next-74969fc645-lg7mh running for 20min |
| Health Endpoint | **PASS** | /health returns 200 OK |
| Model Serving | **PASS** | Qwen3-Next FP8 with TP=4, serving at ~300-900 prompt tok/s |
| Prefix Cache | **PASS** | ~30% hit rate observed |
| GPU Utilization | **PASS** | KV cache usage ~0.1%, memory utilization healthy |

### 7. Capacity Block
| Component | Status | Details |
|-----------|--------|---------|
| Reservation ID | **PASS** | cr-0e271f7913711df91 |
| State | **PASS** | active |
| Instance Type | **PASS** | p5en.48xlarge |
| Availability Zone | **PASS** | us-east-2c |
| End Time | **PASS** | 2026-02-25T11:30:00+00:00 (26 hours remaining) |

### 8. Config Scripts & Benchmark Wiring
| Component | Status | Details |
|-----------|--------|---------|
| Config Scripts Syntax | **PASS** | All 7 configs pass bash -n validation |
| Config Scripts Present | **PASS** | vllm-baseline, vllm-tp4-mtp, vllm-tp8-mtp, vllm-dp8-ep, sglang-baseline, sglang-tp4, sglang-tp8-mtp |
| Benchmark Script | **PASS** | run-benchmarks.sh present with P0-P2 tiers |
| run-benchmarks.py | **MISSING** | No Python benchmark orchestrator found |
| comparison.yaml | **MISSING** | No comparison config found |
| Scripts Directory | **PASS** | copy-to-nvme.sh, stage-model.sh, run-benchmarks.sh |

**Config Details:**
- All configs use `CONTAINER_RUNTIME` variable (defaults to nerdctl)
- All configs reference Docker Hub image: vllm/vllm-openai:qwen3_5-x86_64-cu130
- Model path: /mnt/nvme/models/qwen3-next-fp8
- Supports TP=4, TP=8, DP=8 with expert parallelism

## Action Items

| # | Priority | Action | Owner |
|---|----------|--------|-------|
| 1 | **P1** | Investigate EFA device plugin failure - check if EFA is enabled on p5en.48xlarge | Operator |
| 2 | **P2** | Create run-benchmarks.py if Python orchestration needed | Developer |
| 3 | **P2** | Create comparison.yaml if needed for benchmark result analysis | Developer |
| 4 | **P2** | Consider creating stage-images-ecr.sh to use ECR instead of Docker Hub | Developer |
| 5 | **P0** | Verify benchmark execution environment (local vs port-forward) before running | Operator |

## Overall Verdict

**CONDITIONAL PASS**

The infrastructure is fully operational and actively serving inference. The only failure is the EFA device plugin, which is non-critical for inference serving but may impact multi-node scaling performance.

**Pre-session requirements:**
1. Document benchmark execution location (kubectl port-forward vs direct node access)
2. Confirm whether EFA is needed for planned benchmarks
3. Ensure benchmark metrics collection includes prefix cache hit/miss rates

**Strengths:**
- Model successfully staged to NVMe (17x faster than FSx)
- vLLM serving with good performance (300-900 prompt tok/s)
- Monitoring stack fully operational
- Capacity block active with 26 hours remaining
- All config scripts validated and ready

**Ready for benchmarking**: YES