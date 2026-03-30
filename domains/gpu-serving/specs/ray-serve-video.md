# GPU Serving Spec: Ray Serve Video Pipeline

## Status: COMPLETE

## Overview
Event-driven video processing pipeline on Ray Serve with Kafka ingestion and multi-framework model composition. Single RayService cluster runs both PyTorch and TensorFlow models via `runtime_env` isolation, chained through `DeploymentHandle` with **in-memory data passing** (Ray object store). Extends the ray-serve-ft blueprint (ElastiCache GCS FT, stunnel TLS, KubeRay).

## Architecture

```
Camera/Source ──► S3 (video segments) ──► S3 Event Notification ──► Kafka Topic
                                                                        │
                    ┌───────────────────────────────────────────────────┘
                    ▼
         ┌──────────────────────────────────────────────────────────────────────────┐
         │                     Single RayService Cluster                            │
         │                                                                          │
         │  KafkaIngress (CPU)                                                      │
         │  AIOKafkaConsumer                                                        │
         │  num_cpus=0.1                                                            │
         │       │                                                                  │
         │       │ DeploymentHandle (S3 key)                                        │
         │       ▼                                                                  │
         │  FrameDecode (CPU)              ┌──────────────────────────────────┐     │
         │  ffmpeg/OpenCV                  │  In-memory via Ray Object Store  │     │
         │  runtime_env: opencv-python     │  numpy arrays, no S3 round-trip │     │
         │       │                         └──────────────────────────────────┘     │
         │       │ DeploymentHandle (numpy frames)                                  │
         │       ▼                                                                  │
         │  PTModel (GPU)                                                           │
         │  e.g. YOLOv8 object detection                                            │
         │  runtime_env: torch, ultralytics                                         │
         │       │                                                                  │
         │       │ DeploymentHandle (detections + frames)                            │
         │       ▼                                                                  │
         │  TFModel (GPU)                                                           │
         │  e.g. EfficientNet classification                                        │
         │  runtime_env: tensorflow                                                 │
         │       │                                                                  │
         │       │ DeploymentHandle (results)                                        │
         │       ▼                                                                  │
         │  ResultWriter (CPU)                                                      │
         │  S3 / DynamoDB / downstream Kafka                                        │
         │                                                                          │
         └──────────────────────────────────────────────────────────────────────────┘
```

### Why Single Cluster with Model Composition
- **In-memory data passing**: Intermediate data (decoded frames, feature maps, detections) flows through Ray's object store — zero-copy shared memory for numpy arrays. No S3 serialization between pipeline stages.
- **Single lifecycle**: One RayService CRD, one GCS FT domain, one monitoring stack.
- **Process isolation is natural**: Each Ray Serve deployment is a separate actor (process). PT and TF never share a process — no in-process CUDA context conflict.
- **`runtime_env` per deployment**: PT deployments install `torch`, TF deployments install `tensorflow`. Both use the same CUDA runtime from the base container image. Modern PT and TF both work on CUDA 12.x.

### State Persistence (from ray-serve-ft)
```
ElastiCache Serverless (Redis) ◄──► stunnel sidecar ◄──► Ray GCS
```

## Components

### 1. Compute
- **Platform**: Amazon EKS (existing `qn-sglang-eks-cluster` in us-west-2)
- **Instance Types**:
  - System nodes: m6i.xlarge (2-3x, head pod + KafkaIngress + FrameDecode + ResultWriter)
  - GPU workers: g5.xlarge (2-4x, 1x A10G each) — PT and TF models share the GPU pool
- **Scaling**: Autoscaling enabled for model deployments (`target_ongoing_requests`-based)
- **KubeRay Operator**: v1.3.0+ (GCS FT support)
- **Ray Version**: 2.38.0+
- **Base container image**: NVIDIA CUDA 12.x runtime — neither PT nor TF pre-installed (installed via `runtime_env`)
  - Alternative: Bake both PT and TF into the image to eliminate cold-start pip install (~2GB torch + ~600MB tensorflow)

### 1a. GPU Pre-Flight
Minimal — single A10G per node. Verify `nvidia-smi` shows A10G and CUDA available.

### 2. Kafka Infrastructure
- **Broker**: Amazon MSK Serverless or self-managed Kafka on EKS
- **Topics**:
  - `video-segments` — S3 keys for new video segments (partitioned by camera/source ID)
  - `inference-results` — output topic for downstream consumers (optional)
- **Message format**: JSON with S3 key, source ID, timestamp, segment metadata
- **Message size**: <1KB (S3 references, not raw video bytes)
- **Partitioning**: By source/camera ID — preserves frame ordering per stream
- **Retention**: 24h (segments are in S3; Kafka is the notification channel)

### 3. Ray Serve Deployments

#### KafkaIngress (CPU-only)
```python
@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.1},
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class KafkaIngress:
    def __init__(self, decode_handle):
        self.decode = decode_handle
        self.consumer = AIOKafkaConsumer(
            "video-segments",
            bootstrap_servers=os.environ["KAFKA_BROKERS"],
            group_id="ray-video-pipeline",
            enable_auto_commit=False,
        )
        self._loop = asyncio.get_event_loop()
        self._task = self._loop.create_task(self._consume())
        self._healthy = True

    async def _consume(self):
        await self.consumer.start()
        try:
            async for msg in self.consumer:
                await self.decode.decode.remote(msg.value)
                await self.consumer.commit()
        except Exception:
            self._healthy = False
            raise

    def check_health(self):
        if not self._healthy or self._task.done():
            raise RuntimeError("Kafka consumer failed")

    async def __call__(self, request):
        """Health/status endpoint for NLB health checks."""
        return {"status": "consuming", "healthy": self._healthy}
```

#### FrameDecode (CPU)
```python
@serve.deployment(
    num_replicas=2,
    max_replicas_per_node=2,
    ray_actor_options={
        "num_cpus": 1,
        "runtime_env": {"pip": ["opencv-python-headless", "boto3"]},
    },
    autoscaling_config={"target_ongoing_requests": 5},
)
class FrameDecode:
    def __init__(self, pt_model_handle):
        import cv2, boto3
        self.cv2 = cv2
        self.s3 = boto3.client("s3")
        self.pt_model = pt_model_handle

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def decode(self, messages: List[bytes]):
        """Download segments from S3, extract frames, pass to PT model."""
        frames = []
        for msg in messages:
            meta = json.loads(msg)
            obj = self.s3.get_object(Bucket=meta["bucket"], Key=meta["key"])
            video_bytes = obj["Body"].read()
            # Extract frames with OpenCV
            ...
            frames.append(frame_array)  # numpy array
        # Pass in-memory numpy arrays — zero-copy via Ray object store
        results = await self.pt_model.infer.remote(frames)
        return results
```

#### PTModel (GPU — PyTorch)
```python
@serve.deployment(
    num_replicas=2,
    max_replicas_per_node=1,
    ray_actor_options={
        "num_gpus": 1,
        "runtime_env": {"pip": ["torch", "torchvision", "ultralytics"]},
    },
    autoscaling_config={
        "target_ongoing_requests": 2,
        "min_replicas": 1,
        "max_replicas": 4,
    },
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class PTModel:
    def __init__(self, tf_model_handle):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")
        self.tf_model = tf_model_handle

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.05)
    async def infer(self, frame_batches: List):
        detections = self.model(frame_batches)
        # Pass detections + cropped regions to TF model in-memory
        crops = self._extract_crops(frame_batches, detections)
        classifications = await self.tf_model.classify.remote(crops)
        return self._merge_results(detections, classifications)
```

#### TFModel (GPU — TensorFlow)
```python
@serve.deployment(
    num_replicas=2,
    max_replicas_per_node=1,
    ray_actor_options={
        "num_gpus": 1,
        "runtime_env": {"pip": ["tensorflow"]},
    },
    autoscaling_config={
        "target_ongoing_requests": 2,
        "min_replicas": 1,
        "max_replicas": 4,
    },
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class TFModel:
    def __init__(self, writer_handle):
        import tensorflow as tf
        self.model = tf.saved_model.load("efficientnet_v2")
        self.writer = writer_handle

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.05)
    async def classify(self, crops: List):
        # TF inference on cropped detections from PT model
        predictions = self.model(crops)
        await self.writer.write.remote(predictions)
        return predictions
```

#### ResultWriter (CPU)
```python
@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 0.5},
    autoscaling_config={"target_ongoing_requests": 10},
)
class ResultWriter:
    def __init__(self):
        import boto3
        self.s3 = boto3.client("s3")
        # Optional: Kafka producer for downstream topic

    async def write(self, results):
        """Write inference results to S3/DynamoDB/Kafka."""
        ...
```

#### Application Composition
```python
# Bind the full pipeline — data flows in-memory via DeploymentHandle
writer = ResultWriter.bind()
tf_model = TFModel.bind(writer)
pt_model = PTModel.bind(tf_model)
decode = FrameDecode.bind(pt_model)
app = KafkaIngress.bind(decode)
```

### 4. Networking
- **VPC**: Existing qn-sglang VPC in us-west-2
- **MSK connectivity**: VPC endpoint or same VPC as EKS
- **NLB**: Internal NLB targeting worker proxies (reuse ray-serve-ft pattern)
- **Kafka security**: SASL/SCRAM or IAM auth for MSK

### 5. Storage
- **Video segments**: S3 bucket with lifecycle policy (30-day expiry)
- **Model weights**: Downloaded at pod startup (YOLO ~25MB, EfficientNet ~50MB)
- **GCS state**: ElastiCache Serverless (reuse from ray-serve-ft)
- **Results**: S3 or DynamoDB for inference output

## Design Decisions

### Why single cluster with `runtime_env` (not separate RayService per framework)
- **In-memory pipeline**: `DeploymentHandle` passes numpy arrays through Ray object store — zero-copy. Separate clusters would require S3 or HTTP serialization between PT and TF stages, adding latency and complexity for every frame.
- **Process isolation is sufficient**: Each deployment is a separate Ray actor (process). PT and TF never share a process, so no CUDA context conflict.
- **Same CUDA version**: Modern PyTorch (2.x) and TensorFlow (2.15+) both support CUDA 12.x. No system library conflict.
- **Single GCS FT domain**: One ElastiCache connection, one stunnel setup, one recovery domain.

**When to split into separate RayService CRDs instead:**
- PT and TF need genuinely different CUDA versions (rare)
- Models are independent (no chaining — separate inputs, separate outputs)
- Different teams need independent upgrade/rollback cadences
- Container image size becomes prohibitive with both frameworks

### Why dedicated KafkaIngress (not consumer per model)
Embedding a Kafka consumer in every model deployment is an **anti-pattern**:
- Couples consumer scaling (CPU, by message volume) to model scaling (GPU, by compute)
- Wastes GPU while consumer polls with no messages
- Consumer group rebalancing churn when GPU replicas autoscale
- Conflated health checks (Kafka issue → GPU replica marked unhealthy)

### Why S3 references in Kafka (not raw video)
- Kafka message size limit (default 1MB, video frames are 2-6MB uncompressed)
- S3 provides durability, Kafka provides notification
- Multiple consumers can read same segment without re-transmission
- Partition by camera/source ID for frame ordering

### Cold start mitigation
`runtime_env` pip installs at replica startup add latency:
- `torch` (~2GB): 30-60s on fast network
- `tensorflow` (~600MB): 15-30s
- **Mitigation**: Bake both frameworks into the base container image. They coexist as installed packages — `runtime_env` is only needed if versions differ across deployments. Pre-baked image eliminates cold-start entirely.

## Backpressure Strategy

| Layer | Mechanism |
|-------|-----------|
| Kafka → KafkaIngress | `aiokafka` buffer backpressure (stops fetching when buffer full) |
| KafkaIngress → FrameDecode | `DeploymentHandle` queue; `max_queued_requests` returns `BackPressureError` |
| FrameDecode → PTModel | `max_ongoing_requests` on GPU deployment; excess queued |
| PTModel → TFModel | `DeploymentHandle` queue; same backpressure semantics |
| TFModel → ResultWriter | Async dispatch with bounded queue |
| Cluster-level | Ray Serve autoscaler adds replicas on sustained `target_ongoing_requests` breach |
| External (future) | Kafka consumer lag metric → KEDA HPA → scale EKS GPU node group |

## Test Scenarios

### T1: End-to-End Pipeline
1. Produce 100 video segment references to Kafka
2. Verify all 100 flow through FrameDecode → PTModel → TFModel → ResultWriter
3. **Measure**: End-to-end latency (Kafka publish → result written), throughput (segments/sec)

### T2: Consumer Failover
1. Send sustained traffic to Kafka
2. Kill KafkaIngress replica
3. **Expect**: Ray `check_health()` detects failure, restarts replica, resumes from last committed offset
4. **Measure**: Message loss (expect zero), recovery time

### T3: GPU Model Scaling
1. Ramp Kafka throughput from 10 to 100 segments/sec
2. **Expect**: PTModel and TFModel replicas scale up independently via autoscaler
3. **Measure**: Autoscale reaction time, throughput plateau, GPU utilization per framework

### T4: Head Node Failure (GCS FT)
1. Full pipeline running with sustained Kafka traffic
2. Kill head pod
3. **Expect**: Workers continue inference, consumer resumes after head recovery
4. **Measure**: Message processing gap, recovery time (expect ~3 min from ray-serve-ft baseline)

### T5: Backpressure
1. Produce messages faster than GPU pipeline can process (saturate)
2. **Expect**: Consumer pauses, Kafka lag grows, no OOM or dropped messages
3. **Measure**: Max Kafka lag, memory stability, recovery time when load decreases

### T6: Framework Isolation
1. Crash a PTModel replica (e.g. OOM or segfault)
2. **Expect**: TFModel replicas unaffected, PTModel restarts independently
3. **Measure**: TFModel request success rate during PTModel recovery (expect 100%)

### T7: runtime_env Cold Start (if not pre-baked)
1. Scale PTModel from 0 to 2 replicas
2. **Measure**: Time from autoscaler decision to replica ready (pip install + model load)
3. **Baseline**: Compare pre-baked image vs `runtime_env` pip install

## Experiment Protocol

### Phase 0: Infrastructure (1 hr)
1. Reuse ray-serve-ft EKS cluster + ElastiCache
2. Deploy MSK Serverless (or Kafka on EKS via Strimzi)
3. Build base container image with CUDA 12.x + both PT and TF (or test `runtime_env` cold start)
4. Verify EKS → Kafka connectivity

### Phase 1: Pipeline Deployment (1.5 hrs)
1. Deploy composed Ray Serve application (KafkaIngress → FrameDecode → PTModel → TFModel → ResultWriter)
2. Verify `.bind()` composition works with cross-framework `runtime_env`
3. Produce test messages, verify end-to-end flow with in-memory data passing
4. Baseline latency and throughput

### Phase 2: Fault Injection (3 hrs)
1. Run T1-T7 sequentially
2. Each test: 60s warm-up → inject fault → observe 120s → record metrics

### Phase 3: Analysis (1 hr)
1. Compare with ray-serve-ft baselines (T4 should match)
2. Quantify in-memory vs S3 round-trip savings
3. Document framework isolation behavior (T6)
4. Validate backpressure propagation across the full chain

## Success Criteria
1. End-to-end pipeline processes video segments with <5s latency (Kafka → result)
2. Zero message loss during consumer failover (T2)
3. GPU autoscaling reacts within 30s of sustained load increase (T3)
4. Head node failure recovery matches ray-serve-ft baseline (~3 min) (T4)
5. No OOM or dropped messages under backpressure (T5)
6. PTModel crash does NOT affect TFModel replicas (T6)
7. In-memory data passing confirmed (no S3 round-trip between PT and TF stages)
8. Consumer health check triggers restart within 30s of Kafka disconnection

## Non-Requirements
- Production-grade Kafka cluster (MSK Serverless or minimal Strimzi sufficient)
- Real camera feeds (synthetic S3 segments with COCO images)
- Multi-region Kafka replication
- Schema registry (Avro/Protobuf) — plain JSON sufficient for validation
- KEDA-based autoscaling (future work)
- LLM inference — vision models only for this spec

## Security Requirements
- MSK: SASL/SCRAM or IAM auth, TLS in-transit
- ElastiCache: TLS via stunnel (existing pattern)
- S3: Server-side encryption, VPC endpoint access
- No public Kafka endpoints

## Cost Considerations
- MSK Serverless: ~$0.10/hr + $0.10/GB throughput (minimal for testing)
- Reuse ray-serve-ft compute: ~$2.60/hr base (2x g5.xlarge + 2x m6i.xlarge)
- Additional GPU workers for TF: +$1.01/hr per g5.xlarge
- **Total**: ~$3.60/hr (4x g5 + 2x m6i) — full experiment in half a day (~$18)
- Tear down MSK + GPU nodes after Phase 2

## Known Limitations
- `ray.data.read_kafka()` is bounded (batch) only — cannot be used for streaming
- Ray Serve autoscaler uses HTTP request metrics, not Kafka consumer lag
- `aiokafka` required (not `confluent_kafka`) to avoid blocking Ray's event loop
- Consumer group rebalance pauses message processing briefly on replica changes
- `runtime_env` does not isolate system-level libraries (CUDA/cuDNN) — both frameworks must be compatible with the base image's CUDA version
- Ray docs caveat: "does not guarantee compatibility between tasks and actors with conflicting runtime environments" — PT + TF on same CUDA 12.x is well-trodden, but test in Phase 1
- `runtime_env` pip install cold start can be 30-60s for torch — mitigate by pre-baking into container image

## References
- [Ray Serve + Kafka integration pattern](https://medium.com/data-science/integrate-distributed-ray-serve-deployment-with-kafka-181403f4e194)
- [slanj/ray-serve-kafka](https://github.com/slanj/ray-serve-kafka)
- [Ray Serve model composition docs](https://docs.ray.io/en/latest/serve/model-composition.html)
- [ray-serve-ft blueprint](../blueprints/ray-serve-ft/) — base FT infrastructure

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes)
> belong in the blueprint directory, not in this spec.
