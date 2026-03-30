#!/usr/bin/env python3
"""
Produce test messages to Kafka for the video pipeline.
Generates synthetic images, uploads to S3, publishes S3 keys to Kafka.

Usage (from within the cluster or with port-forward):
    python produce_test.py --num-messages 10
    python produce_test.py --num-messages 50 --bucket my-bucket
"""

import argparse
import io
import json
import logging
import time

import boto3
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Colors for synthetic images
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0),
]


def generate_test_image(index: int, width: int = 640, height: int = 480) -> bytes:
    """Generate a synthetic test image with colored shapes (gives YOLO something to detect)."""
    img = Image.new("RGB", (width, height), (200, 200, 200))
    draw = ImageDraw.Draw(img)

    # Draw some rectangles (crude object-like shapes)
    for i in range(3):
        color = COLORS[(index + i) % len(COLORS)]
        x1 = (i * 180 + 30) % (width - 100)
        y1 = (index * 50 + i * 100) % (height - 100)
        x2 = x1 + 80 + (index * 10) % 60
        y2 = y1 + 80 + (i * 15) % 50
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)

    # Add text label
    draw.text((10, 10), f"test-{index:03d}", fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def ensure_bucket(s3_client, bucket: str, region: str):
    """Create S3 bucket if it doesn't exist."""
    try:
        s3_client.head_bucket(Bucket=bucket)
        logger.info(f"Bucket {bucket} exists")
    except Exception:
        logger.info(f"Creating bucket {bucket}...")
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket)
        else:
            s3_client.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )


def main():
    parser = argparse.ArgumentParser(description="Produce test messages for video pipeline")
    parser.add_argument("--num-messages", type=int, default=10, help="Number of messages to produce")
    parser.add_argument("--bucket", type=str, default=None, help="S3 bucket name")
    parser.add_argument("--region", type=str, default="us-west-2")
    parser.add_argument("--bootstrap-servers", type=str,
                        default="kafka-broker.ray-video.svc.cluster.local:9092")
    parser.add_argument("--topic", type=str, default="video-segments")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between messages")
    args = parser.parse_args()

    # Determine bucket name
    if args.bucket is None:
        sts = boto3.client("sts", region_name=args.region)
        account_id = sts.get_caller_identity()["Account"]
        args.bucket = f"ray-video-poc-{account_id}"

    # Upload test images to S3
    s3 = boto3.client("s3", region_name=args.region)
    ensure_bucket(s3, args.bucket, args.region)

    logger.info(f"Generating and uploading {args.num_messages} test images to s3://{args.bucket}/test-segments/")
    for i in range(args.num_messages):
        img_bytes = generate_test_image(i)
        key = f"test-segments/{i:03d}.jpg"
        s3.put_object(Bucket=args.bucket, Key=key, Body=img_bytes, ContentType="image/jpeg")
    logger.info(f"Uploaded {args.num_messages} images")

    # Publish to Kafka
    from kafka import KafkaProducer

    logger.info(f"Connecting to Kafka at {args.bootstrap_servers}...")
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    logger.info(f"Publishing {args.num_messages} messages to topic '{args.topic}'...")
    for i in range(args.num_messages):
        message = {
            "s3_bucket": args.bucket,
            "s3_key": f"test-segments/{i:03d}.jpg",
            "source_id": "test-camera-1",
            "t_kafka_publish": time.time(),
            "message_index": i,
        }
        producer.send(args.topic, value=message)
        logger.info(f"  [{i+1}/{args.num_messages}] Published: {message['s3_key']}")
        if args.delay > 0 and i < args.num_messages - 1:
            time.sleep(args.delay)

    producer.flush()
    producer.close()
    logger.info(f"Done. {args.num_messages} messages published to '{args.topic}'")


if __name__ == "__main__":
    main()
