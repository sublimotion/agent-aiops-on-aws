#!/usr/bin/env python3
"""
P2 Concurrency Sweep — find throughput ceiling and SLO violation point.

Sends concurrent requests at increasing concurrency levels, measuring
aggregate throughput (tok/s), TTFT, and ITL.
"""

import asyncio
import json
import time
import sys
import statistics
from datetime import datetime

try:
    import aiohttp
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "-q"])
    import aiohttp


API_URL = sys.argv[1] if len(sys.argv) > 1 else "http://10.2.20.79:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen3-235B-A22B-FP8"
OUTPUT = sys.argv[3] if len(sys.argv) > 3 else "/mnt/nvme/results/qwen3-235b-vllm-tp4/concurrency_sweep.json"

# Concurrency levels to test
CONCURRENCY_LEVELS = [1, 4, 8, 16, 32, 64, 128, 256, 512]

# Each request: ~1K input tokens, 256 output tokens (non-thinking for consistent measurement)
SYSTEM_MSG = "/no_think\nYou are a helpful coding assistant. Analyze the following code and suggest improvements."
# ~1K tokens of filler context
CODE_CONTEXT = """
def process_data(data):
    results = []
    for item in data:
        if item.get('type') == 'A':
            value = item['value'] * 2 + item.get('offset', 0)
            if value > 100:
                results.append({'id': item['id'], 'value': value, 'status': 'high'})
            else:
                results.append({'id': item['id'], 'value': value, 'status': 'normal'})
        elif item.get('type') == 'B':
            value = sum(item.get('values', []))
            results.append({'id': item['id'], 'value': value, 'status': 'aggregated'})
        else:
            results.append({'id': item['id'], 'value': 0, 'status': 'unknown'})
    return sorted(results, key=lambda x: x['value'], reverse=True)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.errors = []

    def validate(self, item):
        if not isinstance(item, dict):
            self.errors.append(f"Invalid type: {type(item)}")
            return False
        if 'id' not in item:
            self.errors.append("Missing id field")
            return False
        return True

    def transform(self, item):
        if item['type'] == 'A':
            return {**item, 'value': item['value'] * self.config.get('multiplier', 1)}
        return item

    def process_batch(self, batch):
        valid = [i for i in batch if self.validate(i)]
        transformed = [self.transform(i) for i in valid]
        return process_data(transformed)
""" * 3  # ~1K tokens


async def send_request(session, request_id):
    """Send one request, return timing data."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": f"Review this code:\n```python\n{CODE_CONTEXT}\n```\nSuggest 3 improvements."},
        ],
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time = None
    token_times = []
    output_tokens = 0
    error = None

    try:
        async with session.post(
            f"{API_URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "id": request_id, "success": False, "error": f"HTTP {resp.status}: {body[:200]}",
                    "ttft_ms": 0, "total_ms": 0, "output_tokens": 0, "tps": 0, "itl_ms": [],
                }

            async for line in resp.content:
                text = line.decode("utf-8").strip()
                if not text.startswith("data: "):
                    continue
                data_str = text[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content") or delta.get("reasoning_content") or ""
                    if content:
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        token_times.append(now)
                        output_tokens += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

    except asyncio.TimeoutError:
        error = "timeout"
    except Exception as e:
        error = str(e)[:200]

    end = time.perf_counter()
    ttft = (first_token_time - start) * 1000 if first_token_time else 0
    total = (end - start) * 1000

    # Inter-token latencies
    itl_ms = []
    for i in range(1, len(token_times)):
        itl_ms.append((token_times[i] - token_times[i - 1]) * 1000)

    tps = output_tokens / (total / 1000) if total > 0 and output_tokens > 0 else 0

    return {
        "id": request_id,
        "success": error is None and output_tokens > 0,
        "error": error,
        "ttft_ms": round(ttft, 1),
        "total_ms": round(total, 1),
        "output_tokens": output_tokens,
        "tps": round(tps, 1),
        "itl_ms": itl_ms,
    }


def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


async def run_sweep_level(concurrency):
    """Run all requests at a given concurrency level."""
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_request(session, i) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)
    return results


async def main():
    print(f"P2 Concurrency Sweep — {MODEL}")
    print(f"API: {API_URL}")
    print(f"Levels: {CONCURRENCY_LEVELS}")
    print(f"Output: {OUTPUT}")
    print()

    all_results = []

    for conc in CONCURRENCY_LEVELS:
        print(f"  c={conc:>4} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        results = await run_sweep_level(conc)
        wall_time = time.perf_counter() - t0

        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]

        if successes:
            ttfts = [r["ttft_ms"] for r in successes]
            tps_list = [r["tps"] for r in successes]
            total_output = sum(r["output_tokens"] for r in successes)
            aggregate_tps = total_output / wall_time if wall_time > 0 else 0
            all_itl = []
            for r in successes:
                all_itl.extend(r["itl_ms"])

            level_result = {
                "concurrency": conc,
                "ok": len(successes),
                "fail": len(failures),
                "wall_time_s": round(wall_time, 1),
                "total_output_tokens": total_output,
                "aggregate_tps": round(aggregate_tps, 1),
                "avg_tps_per_request": round(statistics.mean(tps_list), 1),
                "ttft_p50_ms": round(percentile(ttfts, 50), 1),
                "ttft_p95_ms": round(percentile(ttfts, 95), 1),
                "ttft_p99_ms": round(percentile(ttfts, 99), 1),
                "itl_p50_ms": round(percentile(all_itl, 50), 1) if all_itl else 0,
                "itl_p95_ms": round(percentile(all_itl, 95), 1) if all_itl else 0,
                "itl_p99_ms": round(percentile(all_itl, 99), 1) if all_itl else 0,
                "errors": [r["error"] for r in failures] if failures else [],
            }
            all_results.append(level_result)

            slo_ttft = "SLO!" if level_result["ttft_p99_ms"] > 2000 else ""
            slo_itl = "SLO!" if level_result["itl_p99_ms"] > 100 else ""

            print(
                f"ok={len(successes)}/{conc} "
                f"agg_tps={aggregate_tps:.0f} "
                f"ttft_p50={level_result['ttft_p50_ms']:.0f}ms "
                f"ttft_p99={level_result['ttft_p99_ms']:.0f}ms {slo_ttft} "
                f"itl_p50={level_result['itl_p50_ms']:.1f}ms "
                f"itl_p99={level_result['itl_p99_ms']:.1f}ms {slo_itl}"
            )
        else:
            print(f"FAIL all {conc} requests failed: {failures[0]['error'][:100]}")
            all_results.append({
                "concurrency": conc,
                "ok": 0, "fail": conc,
                "errors": [r["error"] for r in failures[:3]],
            })

    # Save results
    output_data = {
        "config": "vllm-tp4-prefix",
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "instance_type": "p6-b300.48xlarge",
        "gpu_count": 4,
        "sweep_type": "concurrency",
        "results": all_results,
    }

    with open(OUTPUT, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {OUTPUT}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Conc':>6} {'OK':>5} {'Agg TPS':>8} {'Avg TPS':>8} {'TTFT p50':>9} {'TTFT p99':>9} {'ITL p50':>8} {'ITL p99':>8}")
    print("-" * 90)
    for r in all_results:
        if r.get("aggregate_tps"):
            print(
                f"{r['concurrency']:>6} {r['ok']:>5} {r['aggregate_tps']:>8.0f} {r['avg_tps_per_request']:>8.1f} "
                f"{r['ttft_p50_ms']:>8.0f}ms {r['ttft_p99_ms']:>8.0f}ms "
                f"{r['itl_p50_ms']:>7.1f}ms {r['itl_p99_ms']:>7.1f}ms"
            )
        else:
            print(f"{r['concurrency']:>6} {'FAIL':>5}")
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
