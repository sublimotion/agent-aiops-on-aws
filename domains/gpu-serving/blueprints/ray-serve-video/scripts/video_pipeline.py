"""
Ray Serve Video Pipeline — Multi-framework model composition with Kafka ingestion.

5-deployment pipeline:
  KafkaIngress (CPU) → FrameDecode (CPU) → PTDetector (GPU/PT) → TFClassifier (GPU/TF) → ResultWriter (CPU)

Data flows in-memory via Ray object store (DeploymentHandle). No S3 round-trip between stages.
"""

import asyncio
import io
import json
import logging
import os
import time
from typing import List

import numpy as np
from ray import serve

logger = logging.getLogger("ray.serve")

_SHARED_DEPS = ["numpy<2", "pillow", "opencv-python-headless", "boto3"]

# ---------------------------------------------------------------------------
# ResultWriter — logs final results (CPU-only)
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
        logger.info(f"RESULT|{json.dumps(result)}")
        return result

    async def __call__(self, request):
        return {"status": "ok", "results_written": self._count}


# ---------------------------------------------------------------------------
# TFClassifier — MobileNetV2 classification (GPU, TensorFlow)
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
        import tensorflow as tf
        # Use system cuDNN — force GPU memory growth
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        self._model = tf.keras.applications.MobileNetV2(weights="imagenet")
        self._preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        self._decode_preds = tf.keras.applications.mobilenet_v2.decode_predictions
        # Warm up
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        self._model.predict(dummy, verbose=0)
        logger.info("TFClassifier ready (MobileNetV2)")

    def check_health(self):
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        pred = self._model.predict(dummy, verbose=0)
        if pred is None:
            raise RuntimeError("TF model inference failed")

    async def classify(self, crops: List[np.ndarray], detections: list, meta: dict) -> dict:
        import tensorflow as tf

        meta["t_tf_start"] = time.time()

        classifications = []
        if crops:
            # Resize and preprocess all crops
            processed = []
            for crop in crops:
                img = tf.image.resize(crop, (224, 224))
                processed.append(img.numpy())
            batch = np.array(processed, dtype=np.float32)
            batch = self._preprocess(batch)
            preds = self._model.predict(batch, verbose=0)
            decoded = self._decode_preds(preds, top=1)
            for i, det in enumerate(detections):
                label, desc, conf = decoded[i][0]
                det["tf_class"] = desc
                det["tf_confidence"] = float(conf)
                classifications.append(det)
        else:
            classifications = detections

        meta["t_tf_end"] = time.time()
        meta["detections"] = classifications
        meta["detection_count"] = len(classifications)

        result = await self._writer.write.remote(meta)
        return result


# ---------------------------------------------------------------------------
# PTDetector — YOLOv8n detection (GPU, PyTorch)
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
        from ultralytics import YOLO
        self._model = YOLO("yolov8n.pt")
        # Warm up
        self._model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        logger.info("PTDetector ready (YOLOv8n)")

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
                # Clamp and crop
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
        meta["t_pt_to_tf_start"] = time.time()
        meta["yolo_detection_count"] = len(detections)

        # Pass crops + detections IN-MEMORY to TFClassifier
        result = await self._tf.classify.remote(crops, detections, meta)
        return result


# ---------------------------------------------------------------------------
# FrameDecode — downloads from S3, decodes image (CPU-only)
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
        logger.info("FrameDecode ready")

    async def decode(self, message: dict) -> dict:
        from PIL import Image

        meta = {
            "s3_bucket": message["s3_bucket"],
            "s3_key": message["s3_key"],
            "t_kafka_publish": message.get("t_kafka_publish", 0),
            "t_decode_start": time.time(),
        }

        # Download from S3
        obj = self._s3.get_object(Bucket=message["s3_bucket"], Key=message["s3_key"])
        img_bytes = obj["Body"].read()

        # Decode to numpy
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(img)

        meta["t_decode_end"] = time.time()
        meta["image_shape"] = list(image_np.shape)

        # Pass numpy array IN-MEMORY to PTDetector
        result = await self._pt.detect.remote(image_np, meta)
        return result


# ---------------------------------------------------------------------------
# KafkaIngress — async consumer, dispatches to pipeline (CPU-only)
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

        # Start consumer in background
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._consume_loop())
        logger.info("KafkaIngress starting consumer...")

    async def _consume_loop(self):
        from aiokafka import AIOKafkaConsumer

        broker = os.environ.get("KAFKA_BROKERS", "kafka-broker.ray-video.svc.cluster.local:9092")
        topic = os.environ.get("KAFKA_TOPIC", "video-segments")

        # Retry connection — Kafka may not be ready yet
        for attempt in range(30):
            try:
                consumer = AIOKafkaConsumer(
                    topic,
                    bootstrap_servers=broker,
                    group_id="ray-video-pipeline",
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
