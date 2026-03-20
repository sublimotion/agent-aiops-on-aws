#!/usr/bin/env python3
"""
Traffic generator for Ray Serve YOLO fault tolerance testing.

Sends a steady stream of detection requests and records per-request metrics:
  - Latency (ms)
  - HTTP status code
  - Timestamp
  - Detection count (for correctness validation)

Usage:
  # Start port-forward first:
  kubectl port-forward -n ray-ft svc/yolo-ft-serve-svc 8000:8000

  # Run traffic at 50 req/s for 5 minutes:
  python3 traffic-gen.py --rps 50 --duration 300 --output results/traffic.jsonl

  # Continuous mode (for fault injection tests):
  python3 traffic-gen.py --rps 50 --duration 0 --output results/traffic.jsonl
"""

import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
from PIL import Image


def generate_test_image() -> str:
    """Generate a small test image as base64 string."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Add a bright rectangle (detectable by YOLO as a potential object)
    img[100:300, 200:400] = [255, 0, 0]
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    image_b64: str,
    request_id: int,
) -> dict:
    """Send one detection request (JSON/base64) and return metrics."""
    start = time.monotonic()
    record = {
        "request_id": request_id,
        "timestamp": time.time(),
        "status": 0,
        "latency_ms": 0,
        "detections": -1,
        "error": None,
    }

    try:
        payload = {"image": image_b64}
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            record["status"] = resp.status
            record["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
            if resp.status == 200:
                body = await resp.json()
                record["detections"] = body.get("count", -1)
            else:
                record["error"] = (await resp.text())[:200]
    except asyncio.TimeoutError:
        record["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        record["error"] = "timeout"
    except aiohttp.ClientError as e:
        record["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        record["error"] = str(e)[:200]

    return record


async def run_traffic(
    url: str,
    rps: float,
    duration: float,
    output_path: str,
):
    """Generate steady traffic at target RPS."""
    image_b64 = generate_test_image()
    interval = 1.0 / rps
    request_id = 0
    start_time = time.monotonic()

    # Rolling stats
    total = 0
    successes = 0
    errors = 0
    latencies = []

    connector = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector) as session:
        out_file = open(output_path, "a")
        print(f"Traffic gen started: {rps} req/s → {url}")
        print(f"Output: {output_path}")
        print(f"Duration: {'infinite' if duration == 0 else f'{duration}s'}")
        print("-" * 60)

        try:
            while True:
                elapsed = time.monotonic() - start_time
                if duration > 0 and elapsed >= duration:
                    break

                request_id += 1
                asyncio.create_task(
                    _record_request(
                        session, url, image_b64, request_id, out_file,
                        latencies,
                    )
                )
                total += 1

                # Print rolling stats every 5 seconds
                if total % max(1, int(rps * 5)) == 0:
                    recent = latencies[-100:] if latencies else [0]
                    ok = sum(1 for l in recent if l > 0)
                    p50 = sorted(recent)[len(recent) // 2] if recent else 0
                    print(
                        f"  [{elapsed:.0f}s] sent={total} "
                        f"ok={ok}/{len(recent)} "
                        f"p50={p50:.0f}ms"
                    )

                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            # Wait for in-flight requests
            await asyncio.sleep(2)
            out_file.close()

    # Final summary
    elapsed = time.monotonic() - start_time
    ok_latencies = [l for l in latencies if l > 0]
    err_count = len(latencies) - len(ok_latencies)
    if ok_latencies:
        sorted_lat = sorted(ok_latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
    else:
        p50 = p99 = 0

    print(f"\n{'=' * 60}")
    print(f"Traffic Summary ({elapsed:.0f}s)")
    print(f"  Total requests: {len(latencies)}")
    print(f"  Successes:      {len(ok_latencies)}")
    print(f"  Errors:         {err_count}")
    print(f"  Error rate:     {100 * err_count / max(1, len(latencies)):.1f}%")
    print(f"  Latency p50:    {p50:.0f}ms")
    print(f"  Latency p99:    {p99:.0f}ms")
    print(f"  Actual RPS:     {len(latencies) / elapsed:.1f}")


async def _record_request(session, url, image_b64, request_id, out_file, latencies):
    """Send request and write result to file."""
    record = await send_request(session, url, image_b64, request_id)
    latencies.append(record["latency_ms"] if record["error"] is None else -1)
    out_file.write(json.dumps(record) + "\n")
    out_file.flush()


def main():
    parser = argparse.ArgumentParser(description="YOLO Traffic Generator")
    parser.add_argument("--url", default="http://localhost:8000/", help="Serve endpoint")
    parser.add_argument("--rps", type=float, default=50, help="Requests per second")
    parser.add_argument("--duration", type=float, default=300, help="Duration in seconds (0=infinite)")
    parser.add_argument("--output", default="results/traffic.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_traffic(args.url, args.rps, args.duration, args.output))


if __name__ == "__main__":
    main()
