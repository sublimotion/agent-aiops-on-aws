#!/usr/bin/env python3
"""
NLB fault injection test runner for Ray Serve FT.

Runs traffic through the NLB ClusterIP (worker-only targets) while injecting
faults, to validate zero-downtime claims. Must be run from a pod inside the
cluster (e.g. via kubectl exec on head pod).

Usage (from local machine):
  # Copy script to head pod
  kubectl cp scripts/nlb_test.py ray-ft/$(kubectl get pods -n ray-ft -l ray-node=head -o name | head -1 | cut -d/ -f2):/tmp/nlb_test.py -c ray-head

  # Run from head pod
  kubectl exec -n ray-ft <head-pod> -c ray-head -- python3 /tmp/nlb_test.py --test T5
  kubectl exec -n ray-ft <head-pod> -c ray-head -- python3 /tmp/nlb_test.py --test all
"""

import argparse
import asyncio
import base64
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# Generate a small test image as base64
def make_test_image():
    try:
        import numpy as np
        from PIL import Image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img[100:300, 200:400] = [255, 0, 0]
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Fallback: tiny 1x1 JPEG
        raw = bytes([
            0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,
            0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,
            0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
            0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
            0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,
            0x24,0x2E,0x27,0x20,0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,
            0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,0x39,0x3D,0x38,0x32,
            0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
            0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,
            0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
            0x09,0x0A,0x0B,0xFF,0xC4,0x00,0xB5,0x10,0x00,0x02,0x01,0x03,
            0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D,
            0xFF,0xDA,0x00,0x08,0x01,0x01,0x00,0x00,0x3F,0x00,0x7B,0x94,
            0x11,0x00,0x00,0x00,0x00,0xFF,0xD9
        ])
        return base64.b64encode(raw).decode()


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


async def run_traffic(url, rps, duration, inject_fn=None, inject_at=60):
    import aiohttp
    image_b64 = make_test_image()
    interval = 1.0 / rps
    records = []
    request_id = 0
    start_time = time.monotonic()
    injected = False

    connector = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        print(f"  Traffic: {rps} req/s for {duration}s -> {url}")
        print(f"  Fault injection at t={inject_at}s")

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= duration:
                break

            # Inject fault at the right time
            if not injected and elapsed >= inject_at and inject_fn:
                print(f"  [{elapsed:.0f}s] INJECTING FAULT")
                inject_fn()
                injected = True

            request_id += 1
            task = asyncio.create_task(send_request(session, url, image_b64, request_id))
            tasks.append(task)

            # Print rolling stats every 5s
            if request_id % max(1, int(rps * 5)) == 0:
                done = [t for t in tasks if t.done()]
                results = [t.result() for t in done]
                errors = sum(1 for r in results if r["error"])
                ok = len(results) - errors
                print(f"  [{elapsed:.0f}s] sent={request_id} ok={ok} err={errors}")

            await asyncio.sleep(interval)

        # Wait for stragglers
        await asyncio.sleep(3)
        for t in tasks:
            if t.done():
                records.append(t.result())

    return records


def analyze(records, test_name, fault_desc):
    total = len(records)
    errors = [r for r in records if r.get("error") or r.get("status", 200) != 200]
    ok = [r for r in records if not r.get("error") and r.get("status") == 200]
    ok_latencies = sorted([r["latency_ms"] for r in ok]) if ok else [0]

    error_window_s = 0
    if len(errors) >= 2:
        error_window_s = errors[-1]["timestamp"] - errors[0]["timestamp"]

    result = {
        "test": test_name,
        "fault": fault_desc,
        "total": total,
        "successes": len(ok),
        "errors": len(errors),
        "error_rate_pct": round(100 * len(errors) / max(1, total), 2),
        "latency_p50_ms": ok_latencies[len(ok_latencies) // 2] if ok_latencies else 0,
        "latency_p99_ms": ok_latencies[int(len(ok_latencies) * 0.99)] if ok_latencies else 0,
        "error_window_s": round(error_window_s, 1),
        "via": "NLB (worker-only targets)",
    }

    print(f"\n  === {test_name} Results ===")
    print(f"  Total: {total}, OK: {len(ok)}, Errors: {len(errors)}")
    print(f"  Error rate: {result['error_rate_pct']}%")
    print(f"  Error window: {result['error_window_s']}s")
    print(f"  P50: {result['latency_p50_ms']}ms, P99: {result['latency_p99_ms']}ms")
    return result


def kubectl(*args):
    cmd = ["kubectl", "-n", "ray-ft"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def get_nlb_url():
    r = kubectl("get", "svc", "yolo-ft-nlb", "-o", "jsonpath={.spec.clusterIP}")
    ip = r.stdout.strip()
    if not ip:
        # Fallback: if running inside cluster, try env or arg
        raise RuntimeError("Cannot get NLB ClusterIP. Pass --url explicitly.")
    return f"http://{ip}:80/"


def get_head_pod():
    r = kubectl("get", "pods", "-l", "ray-node=head", "-o", "jsonpath={.items[0].metadata.name}")
    return r.stdout.strip()


def get_worker_pods():
    r = kubectl("get", "pods", "-l", "ray-node=worker", "-o",
                "jsonpath={range .items[*]}{.metadata.name}{\\n}{end}")
    return [p for p in r.stdout.strip().split("\n") if p]


def get_worker_nodes():
    r = kubectl("get", "pods", "-l", "ray-node=worker", "-o",
                "jsonpath={range .items[*]}{.spec.nodeName}{\\n}{end}")
    return list(set(n for n in r.stdout.strip().split("\n") if n))


def run_test(test_name, url, rps=50, duration=240, inject_at=60, inject_fn=None, fault_desc=""):
    print(f"\n{'='*60}")
    print(f"{test_name}: {fault_desc}")
    print(f"{'='*60}")
    records = asyncio.run(run_traffic(url, rps, duration, inject_fn, inject_at))
    return analyze(records, test_name, fault_desc)


def test_t1(url):
    workers = get_worker_pods()
    target = workers[0] if workers else None
    if not target:
        print("ERROR: no worker pods")
        return None

    def inject():
        print(f"    Killing ray worker process in {target}")
        kubectl("exec", target, "-c", "ray-worker", "--", "pkill", "-f", "ray::SERVE_REPLICA")

    return run_test("T1", url, inject_fn=inject, fault_desc=f"Kill YOLO replica in {target}")


def test_t2(url):
    nodes = get_worker_nodes()
    target = nodes[0] if nodes else None
    if not target:
        print("ERROR: no worker nodes")
        return None

    def inject():
        print(f"    Draining node {target}")
        subprocess.run(["kubectl", "drain", target, "--ignore-daemonsets",
                        "--delete-emptydir-data", "--force", "--grace-period=30",
                        "--timeout=120s"], check=False)

    result = run_test("T2", url, duration=300, inject_fn=inject, fault_desc=f"Drain node {target}")

    # Uncordon
    print(f"  Uncordoning {target}")
    subprocess.run(["kubectl", "uncordon", target], check=False)
    return result


def test_t3(url):
    head = get_head_pod()
    if not head:
        print("ERROR: no head pod")
        return None

    def inject():
        print(f"    Force-deleting head pod {head}")
        subprocess.run(["kubectl", "-n", "ray-ft", "delete", "pod", head,
                        "--force", "--grace-period=0"], check=False)

    return run_test("T3", url, duration=300, inject_at=60,
                    inject_fn=inject, fault_desc=f"Kill head pod {head} (GCS FT ON)")


def test_t5(url):
    head = get_head_pod()
    if not head:
        print("ERROR: no head pod")
        return None

    def inject():
        print(f"    Killing HTTP proxy in head pod {head}")
        kubectl("exec", head, "-c", "ray-head", "--", "pkill", "-f", "ProxyActor")

    return run_test("T5", url, inject_fn=inject, fault_desc=f"Kill head proxy in {head}")


TESTS = {"T1": test_t1, "T2": test_t2, "T3": test_t3, "T5": test_t5}


def main():
    parser = argparse.ArgumentParser(description="NLB Fault Injection Tests")
    parser.add_argument("--test", action="append", default=[])
    parser.add_argument("--url", default=None, help="NLB URL (e.g. http://localhost:8080/)")
    parser.add_argument("--output", default="/tmp/nlb_ft_results.json")
    args = parser.parse_args()

    if not args.test:
        args.test = ["all"]
    tests = list(TESTS.keys()) if "all" in args.test else [t.upper() for t in args.test]

    url = args.url if args.url else get_nlb_url()
    if not url.endswith("/"):
        url += "/"
    print(f"NLB URL: {url}")

    results = []
    for t in tests:
        if t in TESTS:
            r = TESTS[t](url)
            if r:
                results.append(r)
                # Write intermediate results
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
            # Wait between tests for recovery
            if t != tests[-1]:
                print("\n  Waiting 120s for cluster recovery before next test...")
                time.sleep(120)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Test':<6} {'ErrRate':>8} {'ErrWin':>8} {'P50ms':>8} {'P99ms':>8} {'Via'}")
    print("-" * 55)
    for r in results:
        print(f"{r['test']:<6} {r['error_rate_pct']:>7.1f}% {r['error_window_s']:>7.1f}s "
              f"{r['latency_p50_ms']:>7.0f} {r['latency_p99_ms']:>7.0f} {r['via']}")

    print(f"\nResults: {args.output}")


if __name__ == "__main__":
    main()
