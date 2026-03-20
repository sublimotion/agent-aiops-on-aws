"""Ray Serve YOLO Deployment for Fault Tolerance Testing."""

import io
import logging
import numpy as np
from PIL import Image
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("ray.serve")


@serve.deployment(
    num_replicas=2,
    max_replicas_per_node=1,
    health_check_period_s=10,
    health_check_timeout_s=30,
    ray_actor_options={"num_gpus": 1},
)
class YOLODetector:
    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO("yolov8n.pt")
        # Warm up with a dummy inference to trigger CUDA JIT
        self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        logger.info("YOLOv8n loaded and warmed up")

    def check_health(self):
        """Custom health check — run dummy inference."""
        result = self.model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
        if result is None:
            raise RuntimeError("Model inference failed")

    async def __call__(self, request: Request) -> JSONResponse:
        content_type = request.headers.get("content-type", "")

        if "multipart" in content_type:
            form = await request.form()
            image_file = form["image"]
            image_bytes = await image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
        elif "json" in content_type:
            import base64

            body = await request.json()
            image_bytes = base64.b64decode(body["image"])
            image = Image.open(io.BytesIO(image_bytes))
        else:
            # Raw image bytes
            image_bytes = await request.body()
            image = Image.open(io.BytesIO(image_bytes))

        image_np = np.array(image)
        results = self.model(image_np, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(
                    {
                        "class": r.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),
                    }
                )

        return JSONResponse(
            {
                "detections": detections,
                "count": len(detections),
                "image_size": list(image_np.shape[:2]),
            }
        )


app = YOLODetector.bind()
