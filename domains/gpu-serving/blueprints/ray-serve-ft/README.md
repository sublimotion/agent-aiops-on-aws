# Ray Serve Fault Tolerance

End-to-end fault tolerance testing for Ray Serve on EKS with GCS persistence backed by Amazon ElastiCache Serverless (Redis).

## Architecture

```
                    ┌───────────────────────────────┐
                    │        Internal NLB            │
                    │   (yolo-ft-nlb, port 80)       │
                    │   health: /-/healthz :8000      │
                    │   cross-zone: enabled           │
                    └──────────┬─────────┬───────────┘
                               │         │
                  ┌────────────┘         └────────────┐
                  ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  EKS Cluster: qn-sglang-eks-cluster (us-west-2)                        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Namespace: ray-ft                                              │    │
│  │                                                                 │    │
│  │  ┌─────────────────────────────────┐                            │    │
│  │  │  Head Pod (m6i.xlarge)          │                            │    │
│  │  │  ┌───────────┐  ┌───────────┐  │                            │    │
│  │  │  │ ray-head  │  │  stunnel   │  │                            │    │
│  │  │  │ (CPU img) │  │ (TLS →    │──┼──────────┐                 │    │
│  │  │  │           │  │  :6380)   │  │          │                 │    │
│  │  │  │ • GCS     │  └───────────┘  │          │                 │    │
│  │  │  │ • Serve   │                 │          │                 │    │
│  │  │  │   Ctrl    │                 │          │                 │    │
│  │  │  │ • Proxy   │                 │          │                 │    │
│  │  │  │  (internal│                 │          │                 │    │
│  │  │  │   only)   │                 │          │                 │    │
│  │  │  │ • Dashboard│                │          │                 │    │
│  │  │  └───────────┘                 │          │                 │    │
│  │  └─────────────────────────────────┘          │                 │    │
│  │       ▲              ▲                        │                 │    │
│  │       │ GCS RPC      │ HTTP :8000             │                 │    │
│  │       ▼              ▼                        ▼                 │    │
│  │  ┌──────────────┐  ┌──────────────┐   ┌─────────────────┐     │    │
│  │  │ GPU Worker 1 │  │ GPU Worker 2 │   │  ElastiCache    │     │    │
│  │  │ (g5.xlarge)  │  │ (g5.xlarge)  │   │  Serverless     │     │    │
│  │  │              │  │              │   │  (Redis + TLS)  │     │    │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │   │                 │     │    │
│  │  │ │ray-worker│ │  │ │ray-worker│ │   │  GCS state      │     │    │
│  │  │ │  • YOLO  │ │  │ │  • YOLO  │ │   │  persistence    │     │    │
│  │  │ │  replica │ │  │ │  replica │ │   │  (multi-AZ HA)  │     │    │
│  │  │ │• Proxy ◄ │ │  │ │• Proxy ◄ │ │   └─────────────────┘     │    │
│  │  │ │  (NLB)   │ │  │ │  (NLB)   │ │                           │    │
│  │  │ │  (A10G)  │ │  │ │  (A10G)  │ │          ▲                │    │
│  │  │ └──────────┘ │  │ └──────────┘ │          │                │    │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │          │                │    │
│  │  │ │ stunnel  │─┼──┼─│ stunnel  │─┼──────────┘                │    │
│  │  │ │ (:6380)  │ │  │ │ (:6380)  │ │   TLS tunnel              │    │
│  │  │ └──────────┘ │  │ └──────────┘ │                            │    │
│  │  └──────────────┘  └──────────────┘                            │    │
│  │                                                                 │    │
│  │  Init containers:                                               │    │
│  │    • install-libgl (debian) → shared-libs volume for OpenCV     │    │
│  │    • wait-gcs-ready (KubeRay) → blocks until GCS is reachable  │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Nodegroups:                                                            │
│    • ray-ft-system: 2x m6i.xlarge (50GB EBS) — head + KubeRay         │
│    • ray-ft-gpu:    2x g5.xlarge  (100GB EBS) — YOLO workers          │
└─────────────────────────────────────────────────────────────────────────┘

Data flow:
  Client → NLB :80 → Worker Proxy (1 of 2) → local YOLO Replica (GPU)
  Ray GCS → stunnel :6380 → TLS → ElastiCache Serverless :6379

Fault tolerance (zero SPOF):
  Head dies   → KubeRay recreates pod → GCS restores from ElastiCache
                NLB routes to worker proxies (zero downtime)
  Worker dies → Ray reschedules replica → NLB routes to healthy proxies
  Redis HA    → ElastiCache Serverless (multi-AZ, automatic failover)
  Proxy dies  → NLB health checks detect, traffic shifts in ~10s
```

- **Model**: YOLOv8n object detection (lightweight, fast, deterministic)
- **Serving**: Ray Serve on KubeRay 1.3.0+
- **GCS FT**: ElastiCache Serverless Redis for state persistence via stunnel TLS proxy
- **Compute**: g5.xlarge (A10G) for GPU workers, m6i.xlarge for system nodes
- **Cluster**: Reuses `qn-sglang-eks-cluster` in us-west-2

## Test Scenarios

| Test | Fault | Key Metric |
|------|-------|------------|
| T1 | Kill one YOLO replica | Recovery time, error rate |
| T2 | Drain one GPU worker node | Migration time, latency spike |
| T3 | Kill head pod (GCS FT ON) | Worker serving continuity |
| T4 | Kill head pod (GCS FT OFF) | Total downtime (control) |
| T5 | Kill HTTP proxy on head | Proxy failover time |
| T6 | Block ElastiCache connectivity | GCS degraded mode |

## Quick Start

```bash
# 1. Deploy infrastructure
cd terraform && terraform init && terraform apply

# 2. Deploy Ray Serve
./scripts/deploy.sh

# 3. Run fault injection tests
python3 scripts/fault-inject.py --test all

# 4. Analyze results
ls results/
```

## Key Files

| File | Purpose |
|------|---------|
| `terraform/main.tf` | ElastiCache Serverless + EKS nodegroups |
| `k8s/ray-service.yaml` | RayService with GCS FT enabled |
| `k8s/ray-service-no-ft.yaml` | RayService without GCS FT (control) |
| `k8s/serve-nlb.yaml` | Internal NLB — routes to all 3 Serve proxies |
| `k8s/stunnel.yaml` | stunnel TLS proxy ConfigMap (→ ElastiCache) |
| `scripts/yolo_serve.py` | Ray Serve YOLO deployment |
| `scripts/deploy.sh` | KubeRay + RayService deployment |
| `scripts/traffic-gen.py` | Sustained traffic generator |
| `scripts/fault-inject.py` | Automated fault injection + metrics |
