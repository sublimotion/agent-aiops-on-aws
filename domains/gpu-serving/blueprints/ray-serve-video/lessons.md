---
model: "YOLOv8n + MobileNetV2"
engine: "ray-serve"
hardware: "g5.xlarge"
gpu_arch: "A10G"
deployment_date: "2026-03-27"

outcome: "success"
failure_categories:
  - dependency_conflict
  - driver
  - other

cards_used:
  mdc: []
  gpu_infra: []

card_helped: null

benchmark:
  throughput_toks_s: null
  ttft_p50_ms: null
  ttft_p99_ms: null
  concurrent_users: null
  gpu_util_pct: null

ralph_iterations: 3

mdc_learn_commands: []
gpu_infra_learn_commands:
  - 'gpu-infra learn -c platform "EKS pods need IMDS hop limit 2 for IAM credentials — aws ec2 modify-instance-metadata-options --http-put-response-hop-limit 2"'
---

# Lessons Learned: Ray Serve Video Pipeline

## Summary

Deployed a 5-deployment Ray Serve video pipeline on EKS (2x g5.xlarge A10G workers, 2x m6i.xlarge system nodes) proving that PyTorch (YOLOv8n) and TensorFlow (MobileNetV2) can coexist on GPU within a single RayService cluster via `runtime_env` isolation. Kafka ingestion feeds COCO val2017 images through KafkaIngress → FrameDecode → PTDetector → TFClassifier → ResultWriter, with all inter-stage data passing in-memory via Ray object store. Benchmark showed Config A (in-memory) is **1.57x faster** than Config B (S3 passthrough) at E2E p50, with S3 adding 111ms overhead per frame.

## What Worked

- **runtime_env isolation for PT + TF on GPU**: Each deployment gets its own virtualenv under `/tmp/ray/session_*/runtime_resources/pip/<hash>/virtualenv/`. PyTorch and TensorFlow install cleanly into separate environments sharing the same base CUDA 12.5 image. No CUDA context conflicts — each deployment is a separate Ray actor (process).
- **In-memory data passing via DeploymentHandle**: PTDetector passes numpy arrays directly to TFClassifier through Ray's object store. Handoff p50 is ~10ms (shared memory, near zero-copy) vs ~111ms via S3 round-trip.
- **Dedicated KafkaIngress on CPU**: The CPU-only ingress deployment decouples Kafka consumer scaling from GPU model scaling. No GPU cycles wasted on polling. Health check monitors consumer task liveness.
- **Reuse of ray-serve-ft infrastructure**: ElastiCache Serverless + stunnel TLS, KubeRay 1.3.0, existing EKS cluster — all reused without modification. Only added Kafka and the video pipeline namespace.
- **Single-broker Kafka in KRaft mode**: bitnami/kafka:3.7.2 with KRaft (no Zookeeper) works well for POC. Init job creates the topic automatically.

## Failures and Fixes

### dependency_conflict: protobuf 7.x breaks Ray Serve config parsing

TensorFlow 2.16.2 installs `protobuf==7.34.1`, which is incompatible with Ray's `FieldDescriptor.label` attribute (Ray needs protobuf <5). Error: `AttributeError: 'google._upb._message.FieldDescriptor' object has no attribute 'label'`. The error surfaces during `ServeDeploySchema` construction on the head node, not inside the TF deployment itself — making it non-obvious.

**Fix**: Pin `protobuf<5` in TFClassifier's `runtime_env.pip` list. Ray GitHub issue #45351 confirms this is a known incompatibility.

### driver: cuDNN 9.2.1 vs 9.3.0 mismatch for TF GPU

Ray's `rayproject/ray:2.44.1-py312-cu125` image ships cuDNN 9.2.1, but TensorFlow 2.16.2 is compiled against cuDNN 9.3.0. Error: `Loaded runtime CuDNN library: 9.2.1 but source was compiled with: 9.3.0`. TF falls back to CPU-only mode silently unless `TF_FORCE_GPU_ALLOW_GROWTH=true` is set, which surfaces the error.

**Fix**: Add `nvidia-cudnn-cu12==9.3.0.75` to TFClassifier's `runtime_env.pip` list. This pip package provides the matching cuDNN shared libraries that override the system-installed version within the virtualenv.

### other: IMDS hop limit blocks pod S3 access

EKS nodes defaulted to `HttpPutResponseHopLimit=1`. Pods running inside containers need hop limit 2 to reach the instance metadata service for IAM credentials. Error: `botocore.exceptions.NoCredentialsError: Unable to locate credentials`.

**Fix**: `aws ec2 modify-instance-metadata-options --http-put-response-hop-limit 2` on all cluster instances. Added to `deploy-video.sh` step 4.

### other: System node IAM role missing S3 access

FrameDecode runs on the head pod (system node) and needs S3 read access for downloading source images. The system node IAM role had no S3 policy attached.

**Fix**: Attached `AmazonS3ReadOnlyAccess` to both GPU and system node IAM roles. For the S3-passthrough benchmark variant, also created a scoped `ray-video-s3-write` policy for PutObject/GetObject/DeleteObject on the intermediate bucket.

### other: GPU resource exhaustion from failed replicas

Old TFClassifier replicas that failed during initialization (due to protobuf/cuDNN issues) held GPU allocations. The cluster showed `2.0/2.0 GPU` used with no healthy deployments.

**Fix**: Full `kubectl delete rayservice` and redeploy to release stuck GPU allocations. RayService in-place config updates do not garbage-collect failed replicas from previous configs.

## Benchmark Results

### Config A vs Config B Comparison

| Metric | Config A (in-memory) | Config B (S3 passthrough) | Delta |
|--------|---------------------|--------------------------|-------|
| E2E p50 | 159.3ms | 249.5ms | +90.2ms (+57%) |
| E2E p95 | 330.0ms | 470.2ms | +140.2ms (+42%) |
| PT→TF handoff p50 | 10.2ms | 2.6ms + 111.4ms S3 | +101ms |
| S3 overhead p50 | 0ms | 111.4ms (PUT 79.8 + GET 31.6) | — |
| At 30 fps | 0ms/s | ~3,342ms/s | Untenable |

### Scale Projections

| Frame Rate | S3 Overhead | Verdict |
|-----------|------------|---------|
| 1 fps | +111ms/s | Manageable |
| 10 fps | +1.1s/s | Pipeline falls behind |
| 30 fps | +3.3s/s | Untenable |
| 30 fps × 10 cameras | ~1,050 S3 PUTs/s | Near S3 throttle (3,500/s) |

Full results: `results/benchmark_results.md`, visual report: `results/benchmark-visual-report.html`.

## Card Accuracy

No deployment cards were consulted (multi-framework Ray Serve pipeline, not a single-model LLM deployment). The protobuf<5 pin and nvidia-cudnn-cu12 version pinning are novel findings specific to TensorFlow on Ray Serve CUDA images. The IMDS hop limit finding applies broadly to any EKS pod needing IAM credentials.
