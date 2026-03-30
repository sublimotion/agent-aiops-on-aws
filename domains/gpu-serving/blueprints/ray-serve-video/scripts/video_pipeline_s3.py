"""
Ray Serve Video Pipeline — S3 passthrough variant for benchmarking.

Same pipeline as video_pipeline.py but PTDetector writes crops to S3
and passes S3 keys (not numpy arrays) to TFClassifier. TFClassifier
reads crops from S3. Simulates two-cluster architecture overhead.

Data path: KafkaIngress → FrameDecode → PTDetector -[S3]→ TFClassifier → ResultWriter
"""

import asyncio
import io
import json
import logging
import os
import time
import uuid
from typing import List

import numpy as np
from ray import serve

logger = logging.getLogger("ray.serve")

_SHARED_DEPS = ["numpy<2", "pillow", "opencv-python-headless", "boto3"]
_S3_INTERMEDIATE_BUCKET = os.environ.get("S3_INTERMEDIATE_BUCKET", "ray-video-poc-intermediate")

# ---------------------------------------------------------------------------
# ResultWriter — same as Config A
# ---------------------------------------------------------------------------

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5},
)
class ResultWriter:
    def __init__(self):
        self._count = 0

    async def write(self, result: dict) -> dict:
        self._count += 1
        result["t_result_written"] = time.time()
        result["result_index"] = self._count
        result["config"] = "B_s3_passthrough"
        logger.info(f"RESULT|{json.dumps(result)}")
        return result

    async def __call__(self, request):
        return {"status": "ok", "results_written": self._count}


# ---------------------------------------------------------------------------
# TFClassifier — reads crops from S3 instead of receiving in-memory
# ---------------------------------------------------------------------------

@serve.deployment(
    num_replicas=1,
    max_replicas_per_node=1,
    health_check_period_s=10,
    health_check_timeout_s=60,
    ray_actor_options={
        "num_gpus": 1,
        "runtime_env": {
            "pip": _SHARED_DEPS + ["protobuf<5", "nvidia-cudnn-cu12==9.3.0.75", "tensorflow==2.16.2"],
            "env_vars": {
                "PIP_NO_CACHE_DIR": "1",
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            },
        },
    },
)
class TFClassifier:
    def __init__(self, writer):
        self._writer = writer
        import boto3
        self._s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))

        import tensorflow as tf
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        self._model = tf.keras.applications.MobileNetV2(weights="imagenet")
        self._preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        self._decode_preds = tf.keras.applications.mobilenet_v2.decode_predictions
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        self._model.predict(dummy, verbose=0)
        logger.info("TFClassifier ready (MobileNetV2, S3 mode)")

    def check_health(self):
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        pred = self._model.predict(dummy, verbose=0)
        if pred is None:
            raise RuntimeError("TF model inference failed")

    async def classify(self, frame_key: str, detections: list, meta: dict) -> dict:
        import tensorflow as tf

        meta["t_tf_start"] = time.time()

        # Read full frame from S3 (simulates cross-cluster data transfer)
        t_s3_read_start = time.time()
        obj = self._s3.get_object(Bucket=_S3_INTERMEDIATE_BUCKET, Key=frame_key)
        frame_bytes = obj["Body"].read()
        image_np = np.load(io.BytesIO(frame_bytes), allow_pickle=False)
        meta["t_s3_read_duration"] = time.time() - t_s3_read_start
        meta["s3_bytes_read"] = len(frame_bytes)

        # Run classification on full frame (resize to 224x224)
        crops = [image_np]  # classify the whole frame

        classifications = []
        if crops:
            processed = []
            for crop in crops:
                img = tf.image.resize(crop, (224, 224))
                processed.append(img.numpy())
            batch = np.array(processed, dtype=np.float32)
            batch = self._preprocess(batch)
            preds = self._model.predict(batch, verbose=0)
            decoded = self._decode_preds(preds, top=1)
            # Full-frame classification applies to all detections
            frame_label = decoded[0][0][1] if decoded and decoded[0] else "unknown"
            frame_conf = float(decoded[0][0][2]) if decoded and decoded[0] else 0.0
            for det in detections:
                det["tf_class"] = frame_label
                det["tf_confidence"] = frame_conf
                classifications.append(det)
        else:
            classifications = detections

        meta["t_tf_end"] = time.time()
        meta["detections"] = classifications
        meta["detection_count"] = len(classifications)

        result = await self._writer.write.remote(meta)
        return result


# ---------------------------------------------------------------------------
# PTDetector — writes crops to S3 instead of passing in-memory
# ---------------------------------------------------------------------------

@serve.deployment(
    num_replicas=1,
    max_replicas_per_node=1,
    health_check_period_s=10,
    health_check_timeout_s=60,
    ray_actor_options={
        "num_gpus": 1,
        "runtime_env": {
            "pip": _SHARED_DEPS + ["torch", "torchvision", "ultralytics"],
            "env_vars": {"PIP_NO_CACHE_DIR": "1"},
        },
    },
)
class PTDetector:
    def __init__(self, tf_classifier):
        self._tf = tf_classifier
        import boto3
        self._s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))

        from ultralytics import YOLO
        self._model = YOLO("yolov8n.pt")
        self._model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        logger.info("PTDetector ready (YOLOv8n, S3 mode)")

    def check_health(self):
        result = self._model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
        if result is None:
            raise RuntimeError("PT model inference failed")

    async def detect(self, image_np: np.ndarray, meta: dict) -> dict:
        meta["t_pt_start"] = time.time()

        results = self._model(image_np, verbose=False)
        boxes = results[0].boxes

        detections = []
        crops = []
        h, w = image_np.shape[:2]

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < 0.3:
                    continue
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = image_np[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
                    detections.append({
                        "yolo_class": cls_name,
                        "yolo_confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    })

        meta["t_pt_end"] = time.time()
        meta["yolo_detection_count"] = len(detections)

        # Write full frame + detections to S3 (simulates cross-cluster serialization)
        # In a two-cluster architecture, PT results must be serialized to S3
        # for the TF cluster to read — even when there are no detections.
        t_s3_write_start = time.time()
        batch_id = uuid.uuid4().hex[:8]
        frame_key = f"intermediate/{batch_id}/frame.npy"
        buf = io.BytesIO()
        np.save(buf, image_np)
        buf.seek(0)
        self._s3.put_object(
            Bucket=_S3_INTERMEDIATE_BUCKET,
            Key=frame_key,
            Body=buf.getvalue(),
        )
        meta["t_s3_write_duration"] = time.time() - t_s3_write_start
        meta["s3_bytes_written"] = buf.tell()
        meta["t_pt_to_tf_start"] = time.time()

        # Pass S3 key (not numpy) to TFClassifier
        result = await self._tf.classify.remote(frame_key, detections, meta)
        return result


# ---------------------------------------------------------------------------
# FrameDecode — same as Config A
# ---------------------------------------------------------------------------

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1},
)
class FrameDecode:
    def __init__(self, pt_detector):
        self._pt = pt_detector
        import boto3
        self._s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
        logger.info("FrameDecode ready (S3 mode)")

    async def decode(self, message: dict) -> dict:
        from PIL import Image

        meta = {
            "s3_bucket": message["s3_bucket"],
            "s3_key": message["s3_key"],
            "t_kafka_publish": message.get("t_kafka_publish", 0),
            "t_decode_start": time.time(),
        }

        obj = self._s3.get_object(Bucket=message["s3_bucket"], Key=message["s3_key"])
        img_bytes = obj["Body"].read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(img)

        meta["t_decode_end"] = time.time()
        meta["image_shape"] = list(image_np.shape)

        result = await self._pt.detect.remote(image_np, meta)
        return result


# ---------------------------------------------------------------------------
# KafkaIngress — same as Config A
# ---------------------------------------------------------------------------

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1},
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class KafkaIngress:
    def __init__(self, decode):
        self._decode = decode
        self._healthy = True
        self._consuming = False
        self._consumed_count = 0
        self._error_count = 0
        self._last_error = None
        self._task = None

        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._consume_loop())
        logger.info("KafkaIngress starting consumer (S3 mode)...")

    async def _consume_loop(self):
        from aiokafka import AIOKafkaConsumer

        broker = os.environ.get("KAFKA_BROKERS", "kafka-broker.ray-video.svc.cluster.local:9092")
        topic = os.environ.get("KAFKA_TOPIC", "video-segments")

        for attempt in range(30):
            try:
                consumer = AIOKafkaConsumer(
                    topic,
                    bootstrap_servers=broker,
                    group_id="ray-video-pipeline-s3",
                    enable_auto_commit=False,
                    auto_offset_reset="earliest",
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                )
                await consumer.start()
                self._consuming = True
                logger.info(f"KafkaIngress connected to {broker}, topic={topic}")
                break
            except Exception as e:
                logger.warning(f"Kafka connection attempt {attempt+1}/30 failed: {e}")
                await asyncio.sleep(2)
        else:
            self._healthy = False
            raise RuntimeError(f"Failed to connect to Kafka at {broker} after 30 attempts")

        try:
            async for msg in consumer:
                try:
                    message = msg.value
                    message["t_kafka_consumed"] = time.time()
                    await self._decode.decode.remote(message)
                    await consumer.commit()
                    self._consumed_count += 1
                except Exception as e:
                    self._error_count += 1
                    self._last_error = str(e)
                    logger.error(f"Error processing message: {e}")
        except Exception as e:
            self._healthy = False
            self._consuming = False
            self._last_error = str(e)
            logger.error(f"Consumer loop failed: {e}")
            raise

    def check_health(self):
        if not self._healthy:
            raise RuntimeError(f"Consumer unhealthy: {self._last_error}")
        if self._task and self._task.done() and self._task.exception():
            raise RuntimeError(f"Consumer task failed: {self._task.exception()}")

    async def __call__(self, request):
        return {
            "status": "consuming" if self._consuming else "starting",
            "healthy": self._healthy,
            "consumed_count": self._consumed_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
        }


# ---------------------------------------------------------------------------
# Application composition
# ---------------------------------------------------------------------------

writer = ResultWriter.bind()
tf_model = TFClassifier.bind(writer)
pt_model = PTDetector.bind(tf_model)
decode = FrameDecode.bind(pt_model)
app = KafkaIngress.bind(decode)
