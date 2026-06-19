#!/usr/bin/env python3
"""
1h burn-in soak: sustained load at ~85% of the c=32 ceiling, sliced into
5-minute windows. Records drift, error count, and per-window throughput
for the stability block of the enriched artifact.
"""
import concurrent.futures, json, random, statistics, sys, time, urllib.request
from pathlib import Path

ENDPOINT = "http://localhost:8000/v1/embeddings"
MODEL = "Qwen/Qwen3-Embedding-8B"
DURATION_S = 60 * 60      # 1 hour
SLICE_S = 5 * 60          # 5-minute slices
WARMUP_S = 10 * 60        # 10-minute warmup excluded from drift math
CONCURRENCY = 28          # ~85% of the c=32 peak

BASE_CHUNK = "Document retrieval at enterprise scale requires dense embeddings. "
# Mixed 2-10K char prompts matching rag-qa shape
PROMPTS = [(BASE_CHUNK * (ln // len(BASE_CHUNK) + 1))[:ln]
           for ln in [2048, 4096, 6144, 8192, 10240]]


def call_once(_):
    body = json.dumps({"model": MODEL, "input": random.choice(PROMPTS)}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            _ = json.load(r)
        return (time.time() - t0) * 1000, None
    except Exception as e:
        return (time.time() - t0) * 1000, str(e)[:150]


def run_slice(slice_idx, executor):
    """Saturate at CONCURRENCY for SLICE_S seconds, record per-slice stats."""
    end = time.time() + SLICE_S
    lats, errs = [], []
    in_flight = {}
    for _ in range(CONCURRENCY):
        fut = executor.submit(call_once, None)
        in_flight[fut] = time.time()
    while time.time() < end or in_flight:
        done = concurrent.futures.wait(list(in_flight), timeout=1,
                                        return_when=concurrent.futures.FIRST_COMPLETED).done
        for fut in done:
            in_flight.pop(fut, None)
            try:
                lat, err = fut.result()
                if err:
                    errs.append(err)
                else:
                    lats.append(lat)
            except Exception as e:
                errs.append(str(e)[:100])
            if time.time() < end:
                in_flight[executor.submit(call_once, None)] = time.time()
    s = sorted(lats)
    return {
        "slice_idx": slice_idx,
        "start_ts": int(time.time() - SLICE_S),
        "duration_s": SLICE_S,
        "completed": len(lats),
        "failed": len(errs),
        "output_throughput": len(lats) / SLICE_S,
        "error_rate": len(errs) / max(len(lats) + len(errs), 1),
        "latency_p50": statistics.median(lats) if lats else None,
        "latency_p99": s[int(0.99 * len(s))] if s else None,
        "errors_sample": errs[:3],
    }


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results" / "burn-in"
    results_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    print(f"Burn-in: {DURATION_S}s at concurrency={CONCURRENCY}, {SLICE_S}s slices",
          file=sys.stderr)
    print(f"Warmup window: first {WARMUP_S}s (2 slices)", file=sys.stderr)

    num_slices = DURATION_S // SLICE_S
    summary = {
        "workload": "burn-in",
        "duration_s": DURATION_S,
        "slice_duration_s": SLICE_S,
        "warmup_s": WARMUP_S,
        "concurrency": CONCURRENCY,
        "slices": [],
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY * 2) as ex:
        for i in range(num_slices):
            slice_data = run_slice(i, ex)
            summary["slices"].append(slice_data)
            warm = "(warmup)" if (i + 1) * SLICE_S <= WARMUP_S else ""
            print(f"slice {i+1}/{num_slices} {warm}: {slice_data['output_throughput']:.1f} req/s, "
                  f"p50={slice_data['latency_p50']:.0f}ms, p99={slice_data['latency_p99']:.0f}ms, "
                  f"err={slice_data['failed']}", file=sys.stderr, flush=True)
            # Write partial progress on each slice so monitoring can check
            (results_dir / "burn-in-progress.json").write_text(json.dumps(summary, indent=2))

    # Drift math: hour-1 baseline = avg of first 4 post-warmup slices; final = last slice
    warmup_slices = WARMUP_S // SLICE_S  # 2
    post_warmup = summary["slices"][warmup_slices:]
    baseline = post_warmup[:4] if len(post_warmup) >= 4 else post_warmup
    base_tp = statistics.mean(s["output_throughput"] for s in baseline)
    final_tp = summary["slices"][-1]["output_throughput"]
    drift_pct = ((final_tp - base_tp) / base_tp * 100) if base_tp else 0
    unrecoverable = sum(s["failed"] for s in summary["slices"])

    summary["stability"] = {
        "hour_1_throughput": base_tp,
        "final_throughput": final_tp,
        "throughput_drift_pct": drift_pct,
        "unrecoverable_errors": unrecoverable,
        "drift_gate_passed": abs(drift_pct) <= 2.0 and unrecoverable == 0,
    }

    final_path = results_dir / "burn-in-final.json"
    final_path.write_text(json.dumps(summary, indent=2))
    print(f"\nDone. drift={drift_pct:.2f}%  errors={unrecoverable}  "
          f"gate={'PASS' if summary['stability']['drift_gate_passed'] else 'FAIL'}",
          file=sys.stderr)
    print(f"Report: {final_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
