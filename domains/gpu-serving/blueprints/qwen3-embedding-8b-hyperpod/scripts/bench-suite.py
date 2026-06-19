#!/usr/bin/env python3
"""
Embedding benchmark suite for Qwen3-Embedding-8B on HyperPod.
Runs workloads #2, #3, #4 from the spec:
  - long-context: context_lengths sweep [1024, 2048, 4096, 8192]
  - rag-qa: retrieved-context distribution 2K-10K chars
  - production-mix: length distribution from ShareGPT-style traces

Each workload emits its own JSON to results/.
"""
import argparse, concurrent.futures, json, random, statistics, sys, time, urllib.request
from pathlib import Path

ENDPOINT = "http://localhost:8000/v1/embeddings"
MODEL = "Qwen/Qwen3-Embedding-8B"

BASE_CHUNK = "Enterprise knowledge retrieval requires dense embeddings that capture semantic meaning across documents. "


def call_once(text):
    body = json.dumps({"model": MODEL, "input": text}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            _ = json.load(r)
        return (time.time() - t0) * 1000, None
    except Exception as e:
        return (time.time() - t0) * 1000, str(e)[:200]


def stats(lats):
    if not lats:
        return {}
    s = sorted(lats)
    return {
        "mean": statistics.mean(lats),
        "p50": statistics.median(lats),
        "p90": s[int(0.90 * len(s))] if len(s) > 1 else s[0],
        "p95": s[int(0.95 * len(s))] if len(s) > 1 else s[0],
        "p99": s[int(0.99 * len(s))] if len(s) > 1 else s[0],
        "max": max(lats),
        "min": min(lats),
    }


def sweep(prompts, concurrency, total_requests):
    """Hit the endpoint with `total_requests` prompts drawn from `prompts`, at given concurrency."""
    random.seed(42)
    picks = [random.choice(prompts) for _ in range(total_requests)]
    lats, errs = [], []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for lat, err in ex.map(call_once, picks):
            if err:
                errs.append(err)
            else:
                lats.append(lat)
    elapsed = time.time() - t0
    return {
        "concurrency": concurrency,
        "n_requests": total_requests,
        "n_success": len(lats),
        "n_errors": len(errs),
        "duration_s": elapsed,
        "req_per_s": len(lats) / elapsed if elapsed else 0,
        "latency_ms": stats(lats),
        "errors_sample": errs[:3],
    }


def workload_long_context(results_dir):
    """Workload #2: context_lengths [1024, 2048, 4096, 8192]."""
    print("\n=== Workload #2: long-context sweep ===", file=sys.stderr)
    out = {"workload": "long-context-sweep", "contexts": []}
    for target_chars in [4000, 8000, 16000, 32000]:  # ~1K, 2K, 4K, 8K tokens
        prompt = (BASE_CHUNK * (target_chars // len(BASE_CHUNK) + 1))[:target_chars]
        est_tokens = target_chars // 4
        print(f"  context ~{est_tokens} tokens ({target_chars} chars)", file=sys.stderr)
        levels = []
        for c in [1, 4, 16]:
            r = sweep([prompt], c, total_requests=c * 10)
            levels.append(r)
            print(f"    c={c}  req/s={r['req_per_s']:.2f}  p50={r['latency_ms'].get('p50',0):.0f}ms  p99={r['latency_ms'].get('p99',0):.0f}ms  errors={r['n_errors']}", file=sys.stderr)
        out["contexts"].append({
            "approx_tokens": est_tokens,
            "char_len": target_chars,
            "levels": levels,
        })
    path = Path(results_dir) / "workload-long-context.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  → {path}", file=sys.stderr)


def workload_rag_qa(results_dir):
    """Workload #3: RAG Q&A — 2-10K chars mixed."""
    print("\n=== Workload #3: rag-qa ===", file=sys.stderr)
    # Mixed prompt lengths typical of retrieved context chunks
    prompts = []
    for length in [2048, 4096, 6144, 8192, 10240]:
        prompts.append((BASE_CHUNK * (length // len(BASE_CHUNK) + 1))[:length])
    out = {"workload": "rag-qa", "levels": []}
    for c in [1, 2, 4, 8, 16, 32]:
        r = sweep(prompts, c, total_requests=c * 8)
        out["levels"].append(r)
        print(f"  c={c}  req/s={r['req_per_s']:.2f}  p50={r['latency_ms'].get('p50',0):.0f}ms  p99={r['latency_ms'].get('p99',0):.0f}ms  errors={r['n_errors']}", file=sys.stderr)
    path = Path(results_dir) / "workload-rag-qa.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  → {path}", file=sys.stderr)


def workload_production_mix(results_dir):
    """Workload #4: Production mix — variable length distribution, ShareGPT-like."""
    print("\n=== Workload #4: production-mix ===", file=sys.stderr)
    # Synthetic distribution: 40% short (256-512), 40% medium (1K-2K), 20% long (4K-8K)
    random.seed(17)
    prompts = []
    for _ in range(40):
        ln = random.randint(256, 512)
        prompts.append((BASE_CHUNK * (ln // len(BASE_CHUNK) + 1))[:ln])
    for _ in range(40):
        ln = random.randint(1024, 2048)
        prompts.append((BASE_CHUNK * (ln // len(BASE_CHUNK) + 1))[:ln])
    for _ in range(20):
        ln = random.randint(4096, 8192)
        prompts.append((BASE_CHUNK * (ln // len(BASE_CHUNK) + 1))[:ln])
    out = {"workload": "production-mix", "levels": [],
           "distribution_note": "40/40/20 short/medium/long chars (256-512 / 1K-2K / 4K-8K)"}
    for c in [1, 4, 16, 32]:
        r = sweep(prompts, c, total_requests=c * 8)
        out["levels"].append(r)
        print(f"  c={c}  req/s={r['req_per_s']:.2f}  p50={r['latency_ms'].get('p50',0):.0f}ms  p99={r['latency_ms'].get('p99',0):.0f}ms  errors={r['n_errors']}", file=sys.stderr)
    path = Path(results_dir) / "workload-production-mix.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  → {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", nargs="+",
                        default=["long-context", "rag-qa", "production-mix"],
                        choices=["long-context", "rag-qa", "production-mix"])
    parser.add_argument("--results-dir", default=".")
    args = parser.parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    # Warmup
    print("[warmup]", file=sys.stderr)
    for _ in range(5):
        call_once("warmup " * 20)
    dispatch = {
        "long-context": workload_long_context,
        "rag-qa": workload_rag_qa,
        "production-mix": workload_production_mix,
    }
    for w in args.workloads:
        dispatch[w](args.results_dir)
    print("\n[done]", file=sys.stderr)


if __name__ == "__main__":
    main()
