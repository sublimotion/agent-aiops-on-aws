# GPU Serving Spec: Ray Serve Fault Tolerance

## Status: COMPLETE (2026-03-20)

## Overview
Validate Ray Serve end-to-end fault tolerance on EKS with GCS persistence backed by Amazon ElastiCache Serverless (Redis). Uses YOLOv8 object detection as the workload — lightweight (~25MB), fast inference (~10ms GPU), deterministic output for correctness validation after recovery. Test head node recovery, worker node draining, replica failover, and HTTP proxy resilience — confirming that serving continues uninterrupted through each failure mode.

## Components

### 1. Compute
- **Platform**: Amazon EKS (existing `qn-sglang-eks-cluster` in us-west-2, recreate nodegroups)
- **Instance Types**:
  - System nodes: m6i.xlarge (2-3x, KubeRay operator + head pod + spare)
  - GPU workers: g5.xlarge (2x, 1x A10G each) — cheapest GPU for YOLO inference, enough for replica spreading
- **Why g5 instead of g7e**: YOLO doesn't need 96GB VRAM or Blackwell. g5.xlarge ($1.006/hr) is 3.5x cheaper than g7e.12xlarge and sufficient for fault tolerance testing.
- **Scaling**: Fixed node count during fault tolerance testing (no autoscaler interference)
- **KubeRay Operator**: v1.3.0+ (required for GCS FT support)
- **Ray Version**: 2.38.0+ (improved Redis key schema, automatic namespace injection)

### 1a. GPU Pre-Flight
Minimal — single A10G per node, no multi-GPU or NCCL needed. Verify `nvidia-smi` shows A10G and CUDA available.

### 2. Model
- **Model**: Ultralytics YOLOv8n (nano, ~6.2MB weights) or YOLOv8s (small, ~22MB)
- **Why YOLO**: Lightweight (~10ms inference on GPU), deterministic output (bounding boxes), trivial to validate correctness after failover, no special serving engine needed
- **Format**: PyTorch (.pt) — downloaded on first run via `ultralytics` pip package
- **Serving**: Ray Serve native Python deployment (no vLLM needed)
- **Replicas**: 2-4 (spread across worker nodes via `max_replicas_per_node=1`)
- **GPU memory**: ~200MB VRAM — can run many replicas per GPU if needed
- **Input**: HTTP POST with image (base64 or multipart)
- **Output**: JSON with bounding boxes, classes, confidence scores

### 3. Networking
- **VPC**: Existing qn-sglang VPC in us-west-2
- **Access**: kubectl + port-forward for testing; no public ingress needed
- **Endpoints**: VPC endpoint for ElastiCache (Redis)
- **Ray Dashboard**: Port-forward to head pod :8265

### 4. Storage
- **Model Weights**: Downloaded at pod startup (~25MB, takes seconds) — no pre-staging needed
- **GCS State**: Amazon ElastiCache Serverless (Redis)
- **Test Images**: Bundled in container or fetched from public URL (COCO sample images)
- **Results**: Blueprint `results/` directory

### 5. Amazon ElastiCache Serverless (Redis)
- **Engine**: Redis OSS 7.x (ElastiCache Serverless)
- **Why serverless**: GCS metadata is tiny (few MB); pay-per-use cheaper than smallest dedicated node; built-in Multi-AZ HA; no instance management; automatic scaling
- **Encryption**: In-transit TLS + at-rest encryption (enabled by default on serverless)
- **Auth**: IAM authentication (preferred) or Redis AUTH token via Kubernetes Secret
- **Subnet Group**: Private subnets from existing VPC
- **Security Group**: Allow TCP 6379 from EKS worker/head node SG only
- **Cost**: ~$0.0034/hr storage + $0.016/ECPU (pennies for this workload)

## Architecture

```
                    ┌─────────────────────────────────┐
                    │        Amazon ElastiCache        │
                    │   Serverless Redis (GCS state)   │
                    │     Multi-AZ HA, auto-scale      │
                    └────────────┬────────────────────┘
                                 │ TCP 6379 (TLS)
                    ┌────────────┴────────────────────┐
                    │                                   │
        ┌───────────┴──────────┐          ┌────────────┴─────────┐
        │   Ray Head Pod       │          │   Ray Head Pod       │
        │   (GCS + Serve Ctrl) │          │   (after recovery)   │
        │   m6i.xlarge node    │          │   same or new pod    │
        └───────────┬──────────┘          └──────────────────────┘
                    │
       ┌────────────┼────────────┐
       │            │            │
  ┌────┴───┐  ┌────┴───┐  ┌────┴───┐
  │Worker 1│  │Worker 2│  │Worker 3│
  │YOLO rep│  │YOLO rep│  │(spare) │
  │g5 A10G │  │g5 A10G │  │        │
  └────────┘  └────────┘  └────────┘
```

## GCS Fault Tolerance Configuration

### RayService Annotations
```yaml
metadata:
  annotations:
    ray.io/ft-enabled: "true"
```

### Environment Variables (auto-injected by KubeRay)
| Variable | Head Pod | Worker Pod | Purpose |
|----------|----------|------------|---------|
| `RAY_REDIS_ADDRESS` | `<elasticache-endpoint>:6379` | `<elasticache-endpoint>:6379` | Redis connection |
| `RAY_gcs_rpc_server_reconnect_timeout_s` | 60 | 600 | Reconnect timeout before termination |
| `RAY_external_storage_namespace` | (auto: RayCluster UID) | (auto: RayCluster UID) | Redis key namespace |
| `RAY_REDIS_CA_CERT` | `/etc/ssl/certs/ca-certificates.crt` | same | TLS CA for ElastiCache |

### Ray Serve Deployment
```python
from ray import serve
from ultralytics import YOLO

@serve.deployment(
    num_replicas=2,
    max_replicas_per_node=1,
    health_check_period_s=10,
    health_check_timeout_s=30,
    ray_actor_options={"num_gpus": 1},
)
class YOLODetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # downloads ~6MB on first run

    def check_health(self):
        # Custom health check — run dummy inference
        import numpy as np
        result = self.model(np.zeros((640, 640, 3), dtype=np.uint8))
        if result is None:
            raise RuntimeError("Model inference failed")

    async def __call__(self, request):
        image = await request.body()
        results = self.model(image)
        return results[0].tojson()
```

## Test Scenarios

### T1: Replica Crash Recovery
1. Deploy Ray Serve with 2 replicas on separate nodes
2. Send sustained traffic (10 req/s)
3. Kill one replica process (`ray.kill(replica_actor)`)
4. **Expect**: Serve controller detects failure, restarts replica, traffic continues via healthy replica
5. **Measure**: Request error rate during recovery, time to full recovery

### T2: Worker Node Drain
1. Deploy with replicas spread across 2 worker nodes
2. Send sustained traffic
3. Drain one worker node (`kubectl drain --ignore-daemonsets`)
4. **Expect**: Replica migrates to remaining node, traffic disruption < 30s
5. **Measure**: P99 latency spike, error count during drain

### T3: Head Node Failure (GCS FT)
1. Deploy with GCS FT enabled (ElastiCache backing)
2. Send sustained traffic
3. Kill head pod (`kubectl delete pod <head-pod> --force`)
4. **Expect**: Workers continue serving during head recovery, GCS restores from Redis
5. **Measure**: Worker serving continuity (zero dropped requests on workers), head recovery time, total downtime

### T4: Head Node Failure WITHOUT GCS FT (control)
1. Deploy identical setup but `ray.io/ft-enabled: "false"`
2. Kill head pod
3. **Expect**: Full cluster restart, all requests fail during recovery
4. **Measure**: Total downtime (expected: minutes vs T3's seconds)

### T5: HTTP Proxy Failover
1. Deploy with multiple HTTP proxies (`num_cpus=0` proxy actors on each node)
2. Kill the HTTP proxy on the head node
3. **Expect**: Traffic routes through worker-node proxies, Serve controller restarts head proxy
4. **Measure**: Request success rate during proxy recovery

### T6: ElastiCache Connectivity Disruption
1. Temporarily block Redis SG ingress (simulate network partition to ElastiCache Serverless)
2. Restore after 30s
3. **Expect**: GCS operates degraded during partition, reconnects automatically
4. **Measure**: Redis reconnection time, impact on serving, whether workers continue serving

## Experiment Protocol

### Phase 0: Infrastructure (2 hrs)
1. Recreate EKS nodegroups on `qn-sglang-eks-cluster`
2. Deploy ElastiCache Redis in existing VPC
3. Install KubeRay operator v1.3.0+
4. Verify connectivity: EKS nodes → ElastiCache

### Phase 1: Baseline Deployment (30 min)
1. Deploy RayService with YOLOv8n (2 replicas, GCS FT enabled)
2. Verify detection works: POST a COCO image, confirm bounding boxes returned
3. Run baseline latency/throughput (expect ~10ms p50, ~500 req/s per replica)

### Phase 2: Fault Injection (2-3 hrs)
1. Run T1-T6 sequentially
2. Each test: 60s warm-up → inject fault → observe for 120s → record metrics
3. Traffic generator: `locust` or async Python client sending COCO images at 50 req/s

### Phase 3: Analysis (1 hr)
1. Compare T3 vs T4 (GCS FT value)
2. Measure recovery times across all scenarios
3. Document failure modes and edge cases

## Scripts to Build

| Script | Purpose | Priority |
|--------|---------|----------|
| `terraform/main.tf` | ElastiCache + EKS nodegroup recreation | P0 |
| `k8s/ray-service.yaml` | RayService manifest with GCS FT | P0 |
| `k8s/ray-service-no-ft.yaml` | Control: RayService without GCS FT | P0 |
| `scripts/deploy.sh` | KubeRay operator + RayService deployment | P0 |
| `scripts/fault-inject.py` | Automated fault injection + metric collection | P0 |
| `scripts/traffic-gen.py` | Sustained traffic generator (COCO images → YOLO endpoint) | P0 |
| `scripts/yolo_serve.py` | Ray Serve YOLO deployment application | P0 |
| `scripts/analyze.py` | Parse results, generate comparison report | P1 |

## Success Criteria
1. T3 (head node + GCS FT): Worker-side request success rate > 99% during head recovery
2. T3 recovery time < 90 seconds (head pod restart + GCS restore + controller reconcile)
3. T3 vs T4: Measurable improvement in availability (expect 10x less downtime)
4. T1 (replica crash): Zero sustained errors after controller restart (< 30s recovery)
5. T2 (node drain): Graceful migration with < 30s latency spike
6. All 6 test scenarios documented with metrics

## Non-Requirements
- Production-grade monitoring (Prometheus/Grafana) — manual observation sufficient
- LLM serving — YOLO is sufficient to test FT mechanisms; no vLLM/SGLang needed
- Autoscaling — fixed replica count to isolate fault tolerance behavior
- Public ingress / TLS termination — port-forward only
- Ray Tune / Ray Train fault tolerance — Serve only

## Security Requirements
- ElastiCache: TLS in-transit, encryption at rest, AUTH token
- ElastiCache SG: ingress only from EKS node SG on port 6379
- No public Redis endpoint
- EKS RBAC: KubeRay operator service account with minimal permissions

## Cost Considerations
- ElastiCache Serverless: ~$0.01/hr (GCS metadata is negligible)
- 2x g5.xlarge (A10G): ~$1.006/hr each ($2.01/hr total)
- 2-3x m6i.xlarge: ~$0.192/hr each ($0.58/hr total)
- **Total**: ~$2.60/hr — full experiment in half a day (~$13)
- Tear down GPU nodes immediately after Phase 2; ElastiCache Serverless costs pennies idle

## Known Limitations
- KubeRay GCS FT officially recommended for Ray Serve only (not Train/Tune)
- Detached actor state is not persisted — only actor reference survives head restart
- ElastiCache Serverless failover is transparent (multi-AZ built-in), but brief blips possible
- g5.xlarge availability is generally good in us-west-2 (on-demand)
- YOLOv8n download is ~6MB (seconds), but first inference triggers CUDA JIT (~5s cold start)

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes)
> belong in the blueprint directory, not in this spec.
