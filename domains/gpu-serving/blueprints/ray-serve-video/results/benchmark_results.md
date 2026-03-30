# Ray Serve Video Pipeline Benchmark Results

**Date**: 2026-03-27
**Cluster**: qn-sglang-eks-cluster (us-west-2)
**Workers**: 2x g5.xlarge (A10G 24GB)
**Images**: 10 COCO val2017 images, 20 messages per config
**Pipeline**: KafkaIngress → FrameDecode → PTDetector (YOLOv8n) → TFClassifier (MobileNetV2) → ResultWriter

## Config A: In-Memory (DeploymentHandle)

Data flows through Ray object store. PT passes numpy arrays directly to TF.

| Stage | p50 | p95 | mean |
|-------|-----|-----|------|
| E2E Latency | 159.3ms | 330.0ms | 184.7ms |
| S3 Decode | 31.9ms | 56.7ms | 32.7ms |
| PT Detect (GPU) | 9.4ms | 13.7ms | 9.6ms |
| TF Classify | 90.1ms | 218.8ms | 108.8ms |
| PT→TF Handoff | 10.2ms | 62.2ms | 13.2ms |

## Config B: S3 Passthrough

PT writes full frame to S3 as numpy array. TF reads from S3. Simulates two-cluster architecture.

| Stage | p50 | p95 | mean |
|-------|-----|-----|------|
| E2E Latency | 249.5ms | 470.2ms | 263.9ms |
| S3 Decode | 31.4ms | 74.0ms | 33.0ms |
| PT Detect (GPU) | 10.0ms | 74.7ms | 19.4ms |
| TF Classify | 107.2ms | 139.8ms | 108.4ms |
| PT→TF Handoff | 2.6ms | 4.5ms | 2.7ms |
| **S3 Write (PT→S3)** | **79.8ms** | **142.2ms** | **84.7ms** |
| **S3 Read (S3→TF)** | **31.6ms** | **64.8ms** | **31.8ms** |
| **S3 Total (W+R)** | **108.4ms** | **185.7ms** | **116.4ms** |

## Comparison

| Metric | Config A | Config B | Delta |
|--------|----------|----------|-------|
| E2E p50 | 159.3ms | 249.5ms | **+90.2ms (+57%)** |
| E2E p95 | 330.0ms | 470.2ms | +140.2ms (+42%) |
| S3 overhead p50 | 0ms | 111.4ms | **+111.4ms per frame** |
| At 10 fps | 0ms/s | ~1,114ms/s | Adds ~1.1s pipeline latency/sec |
| At 30 fps | 0ms/s | ~3,342ms/s | S3 becomes bottleneck (3,500 PUT/s soft limit) |

## Key Findings

1. **Config A is 1.57x faster (E2E p50)** than Config B due to eliminating S3 serialization
2. **S3 overhead is 111ms per frame** (79.8ms write + 31.6ms read) — this is the cost of the two-cluster architecture
3. **PT→TF handoff via DeploymentHandle is ~10ms** (in-memory numpy through Ray object store) vs **~111ms via S3**
4. At production frame rates (30 fps), S3 passthrough adds ~3.3 seconds of pipeline latency per second — making it untenable for real-time processing
5. S3 PUT has ~3,500 req/s soft limit — at 30 fps with 3.5 detections/frame, Config B would hit S3 throttling at ~30 cameras

## Architecture Recommendation

**Use a single RayService cluster** with PT and TF deployed as separate actors sharing GPU via `runtime_env` isolation. This eliminates S3 serialization overhead and enables in-memory data passing through Ray's object store.

For the customer's specific setup:
- Consolidate PT and TF Ray Serve clusters into one
- Use `runtime_env` per deployment for pip isolation (torch vs tensorflow)
- Pin `protobuf<5` in TF runtime_env (TF installs protobuf 7.x which breaks Ray)
- Pin `nvidia-cudnn-cu12==9.3.0.75` for TF GPU on Ray's CUDA 12.5 image
- Move Kafka consumer to a dedicated CPU deployment (KafkaIngress pattern)
