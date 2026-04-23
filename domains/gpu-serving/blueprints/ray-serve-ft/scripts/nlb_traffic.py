#!/usr/bin/env python3
"""
Traffic generator that runs INSIDE the cluster against the NLB ClusterIP.
Fault injection happens from outside (local machine) in parallel.

Usage:
  kubectl exec -n ray-ft <head-pod> -c ray-head -- \
    python3 /tmp/nlb_traffic.py --url http://172.20.X.X:80/ --duration 240 --rps 50
"""

import argparse
import asyncio
import base64
import io
import json
import sys
import time


def make_test_image():
    """Generate a valid 8x8 RGB PNG using pure Python (no PIL/numpy needed)."""
    import struct, zlib
    width, height = 8, 8
    # 8x8 red pixels: each row = filter byte (0) + RGB * width
    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00" + (b"\xff\x00\x00" * width)

    def png_chunk(chunk_type, data):
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    idat = zlib.compress(raw_rows)

    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", ihdr)
    png += png_chunk(b"IDAT", idat)
    png += png_chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


async def send_request(session, url, image_b64, request_id):
    import aiohttp
    start = time.monotonic()
    record = {
        "request_id": request_id,
        "timestamp": time.time(),
        "status": 0,
        "latency_ms": 0,
        "error": None,
    }
    try:
        payload = {"image": image_b64}
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            record["status"] = resp.status
            record["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
            if resp.status != 200:
                record["error"] = (await resp.text())[:200]
    except asyncio.TimeoutError:
        record["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        record["error"] = "timeout"
    except Exception as e:
        record["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        record["error"] = str(e)[:200]
    return record


async def run_traffic(url, rps, duration):
    import aiohttp
    image_b64 = make_test_image()
    interval = 1.0 / rps
    request_id = 0
    start_time = time.monotonic()
    tasks = []

    connector = aiohttp.TCPConnector(limit=200, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"TRAFFIC_START url={url} rps={rps} duration={duration}", flush=True)

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= duration:
                break

            request_id += 1
            task = asyncio.create_task(send_request(session, url, image_b64, request_id))
            tasks.append(task)

            # Print heartbeat every 5s
            if request_id % max(1, int(rps * 5)) == 0:
                done = [t for t in tasks if t.done()]
                results = [t.result() for t in done]
                errors = sum(1 for r in results if r["error"])
                ok = len(results) - errors
                print(f"HEARTBEAT t={elapsed:.0f}s sent={request_id} ok={ok} err={errors}", flush=True)

            await asyncio.sleep(interval)

        # Wait for stragglers
        print("DRAINING...", flush=True)
        await asyncio.sleep(5)

    records = []
    for t in tasks:
        if t.done():
            try:
                records.append(t.result())
            except Exception:
                pass

    # Compute stats
    total = len(records)
    errors = [r for r in records if r.get("error") or r.get("status", 200) != 200]
    ok = [r for r in records if not r.get("error") and r.get("status") == 200]
    ok_latencies = sorted([r["latency_ms"] for r in ok]) if ok else [0]

    error_window_s = 0
    if len(errors) >= 2:
        error_window_s = errors[-1]["timestamp"] - errors[0]["timestamp"]

    summary = {
        "total": total,
        "successes": len(ok),
        "errors": len(errors),
        "error_rate_pct": round(100 * len(errors) / max(1, total), 2),
        "latency_p50_ms": ok_latencies[len(ok_latencies) // 2],
        "latency_p99_ms": ok_latencies[int(len(ok_latencies) * 0.99)],
        "error_window_s": round(error_window_s, 1),
    }

    print(f"SUMMARY {json.dumps(summary)}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--rps", type=float, default=50)
    parser.add_argument("--duration", type=float, default=240)
    args = parser.parse_args()

    asyncio.run(run_traffic(args.url, args.rps, args.duration))


if __name__ == "__main__":
    main()
