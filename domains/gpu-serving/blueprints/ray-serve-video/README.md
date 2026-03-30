# Ray Serve Video Pipeline

Multi-framework model composition on a single RayService cluster with Kafka ingestion and in-memory data passing. Proves that PyTorch and TensorFlow deployments coexist on GPU via `runtime_env` isolation, eliminating S3 serialization between pipeline stages.

## Architecture

```
                    Kafka (KRaft, single broker)
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     Single RayService Cluster                           │
│                                                                         │
│  KafkaIngress (CPU)  ─► FrameDecode (CPU)  ─► PTDetector (GPU, YOLO)   │
│  aiokafka consumer      S3 download + PIL      YOLOv8n detection        │
│                                                      │                  │
│                         In-memory via Ray Object Store (numpy ~10ms)     │
│                                                      │                  │
│                                                      ▼                  │
│                                              TFClassifier (GPU, TF)     │
│                                              MobileNetV2 classify       │
│                                                      │                  │
│                                                      ▼                  │
│                                              ResultWriter (CPU)         │
│                                              JSON logging               │
│                                                                         │
│  GCS FT: ElastiCache Serverless + stunnel sidecar                       │
└──────────────────────────────────────────────────────────────────────────┘

EKS: qn-sglang-eks-cluster (us-west-2)
GPU workers: 2x g5.xlarge (A10G 24GB)
System nodes: 2x m6i.xlarge
```

## Key Finding: 1.57x Faster with In-Memory

| Config | E2E p50 | PT→TF Overhead |
|--------|---------|----------------|
| A: In-memory (DeploymentHandle) | 159.3ms | ~10ms (shared memory) |
| B: S3 passthrough | 249.5ms | ~111ms (PUT 80ms + GET 32ms) |

At 30 fps, S3 passthrough adds 3.3s/s pipeline latency — untenable for real-time video. Use a single RayService cluster with `DeploymentHandle` composition.

## Prerequisites

- EKS cluster with KubeRay operator (reuses ray-serve-ft infrastructure)
- ElastiCache Serverless endpoint (reuses ray-serve-ft Terraform)
- S3 bucket with test images (COCO val2017)
- IAM: S3 read access on both GPU and system node roles
- IMDS hop limit 2 on all EKS nodes

## Quick Start

```bash
# 1. Deploy (Config A: in-memory pipeline)
./scripts/deploy-video.sh

# 2. Produce test messages
kubectl cp scripts/produce_test.py ray-video/<head-pod>:/tmp/produce_test.py
kubectl exec -n ray-video <head-pod> -c ray-head -- python /tmp/produce_test.py

# 3. Check results
kubectl logs -n ray-video -l ray-node=worker -c ray-worker | grep RESULT

# 4. Run benchmark (both configs)
python3 scripts/benchmark.py

# 5. Switch to Config B (S3 passthrough)
PIPELINE_FILE=video_pipeline_s3.py ./scripts/deploy-video.sh
```

## Key Files

| File | Purpose |
|------|---------|
| `k8s/ray-service-video.yaml` | RayService with 5 deployments, GCS FT, runtime_env isolation |
| `k8s/kafka.yaml` | Single-broker Kafka in KRaft mode (no Zookeeper) |
| `k8s/stunnel.yaml` | stunnel TLS proxy ConfigMap (→ ElastiCache) |
| `scripts/video_pipeline.py` | Config A: in-memory pipeline (DeploymentHandle) |
| `scripts/video_pipeline_s3.py` | Config B: S3 passthrough variant for benchmarking |
| `scripts/deploy-video.sh` | 8-step deployment orchestration |
| `scripts/produce_test.py` | Kafka test producer + S3 image upload |
| `scripts/benchmark.py` | Config A vs B latency comparison |
| `results/benchmark_results.md` | Full benchmark data |
| `results/benchmark-visual-report.html` | Interactive Chart.js report |
| `lessons.md` | Deployment lessons (protobuf, cuDNN, IMDS fixes) |

## Critical runtime_env Dependencies

### TFClassifier (GPU)
```yaml
pip:
  - "numpy<2"           # Ray pyarrow requires numpy 1.x
  - "protobuf<5"        # TF installs protobuf 7.x which breaks Ray
  - "nvidia-cudnn-cu12==9.3.0.75"  # Ray image has cuDNN 9.2.1, TF needs 9.3.0
  - "tensorflow==2.16.2"
env_vars:
  TF_FORCE_GPU_ALLOW_GROWTH: "true"
```

### PTDetector (GPU)
```yaml
pip:
  - "numpy<2"
  - torch
  - torchvision
  - ultralytics
```

## Related

- [ray-serve-ft](../ray-serve-ft/) — Base FT infrastructure (ElastiCache, stunnel, KubeRay)
- [Spec](../../specs/ray-serve-video.md) — Full requirements and test scenarios
- [Ray Serve model composition docs](https://docs.ray.io/en/latest/serve/model-composition.html)
