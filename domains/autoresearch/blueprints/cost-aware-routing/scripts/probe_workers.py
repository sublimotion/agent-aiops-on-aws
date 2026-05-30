"""
Three-stage Bedrock probe for the cost-aware-routing 9-worker pool.

Stage A — ping (verify each model_id invokes successfully, capture latency).
Stage B — token usage probe (5 calls per worker on a real MATH500 question,
          capture input/output tokens to validate $/query assumptions).
Stage C — TPM burst probe for the 3 most expensive workers (Opus 4.7,
          Sonnet 4.6, Qwen3-Coder-480B): fire 32 concurrent calls and time
          the burst, infer effective TPM ceiling.

Output: results/preflight/worker_probe.json
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import statistics
import time

import boto3
from botocore.config import Config

REGION = "us-west-2"

POOL = [
    {"ord": 0, "name": "gpt-oss-120b", "model_id": "openai.gpt-oss-120b-1:0", "assumed_per_query": 0.001, "api": "openai"},
    {"ord": 1, "name": "gemma-3-27b-it", "model_id": "google.gemma-3-27b-it", "assumed_per_query": 0.002, "api": "messages"},
    {"ord": 2, "name": "haiku-4-5", "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "assumed_per_query": 0.005, "api": "messages"},
    {"ord": 3, "name": "deepseek-v3.2", "model_id": "deepseek.v3.2", "assumed_per_query": 0.008, "api": "messages"},
    {"ord": 4, "name": "qwen3-32b", "model_id": "qwen.qwen3-32b-v1:0", "assumed_per_query": 0.010, "api": "messages"},
    {"ord": 5, "name": "mistral-large-3", "model_id": "mistral.mistral-large-3-675b-instruct", "assumed_per_query": 0.012, "api": "messages"},
    {"ord": 6, "name": "qwen3-coder-480b", "model_id": "qwen.qwen3-coder-480b-a35b-v1:0", "assumed_per_query": 0.020, "api": "messages"},
    {"ord": 7, "name": "sonnet-4-6", "model_id": "us.anthropic.claude-sonnet-4-6", "assumed_per_query": 0.060, "api": "messages"},
    {"ord": 8, "name": "opus-4-7", "model_id": "us.anthropic.claude-opus-4-7", "assumed_per_query": 0.300, "api": "messages"},
]

MATH_PING_Q = "What is the value of 7 * 8?"
MATH_PROBE_Q = (
    "Convert the point (0,3) in rectangular coordinates to polar coordinates. "
    "Enter your answer in the form (r, theta), where r > 0 and 0 <= theta < 2*pi. "
    "Show your reasoning briefly."
)


def make_client():
    cfg = Config(retries={"max_attempts": 3, "mode": "standard"}, read_timeout=60)
    return boto3.client("bedrock-runtime", region_name=REGION, config=cfg)


def invoke(client, worker, prompt, max_tokens=256, temperature=0.7):
    """Returns (text, input_tokens, output_tokens, elapsed_s, error).

    Opus 4.7 deprecates `temperature` — pass empty inferenceConfig for it.
    """
    t0 = time.time()
    if "opus-4-7" in worker["model_id"]:
        cfg = {"maxTokens": max_tokens}
    else:
        cfg = {"maxTokens": max_tokens, "temperature": temperature}
    try:
        resp = client.converse(
            modelId=worker["model_id"],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=cfg,
        )
        dur = time.time() - t0
        out = resp["output"]["message"]["content"]
        text = out[0].get("text", "") if out else ""
        usage = resp.get("usage", {})
        return {
            "text": text[:200],
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "elapsed_s": round(dur, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "text": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_s": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


def stage_a_ping(client):
    print("\n=== Stage A: ping all 9 workers ===")
    results = []
    for w in POOL:
        r = invoke(client, w, MATH_PING_Q, max_tokens=32, temperature=0)
        ok = r["error"] is None
        print(f"  ord={w['ord']} {w['name']:20s} {'OK' if ok else 'FAIL'} "
              f"in={r['input_tokens']} out={r['output_tokens']} t={r['elapsed_s']}s "
              f"{r['error'] or ''}")
        results.append({"worker": w, "result": r})
    return results


def stage_b_token_probe(client, n=5):
    print(f"\n=== Stage B: token usage probe (n={n} per worker, MATH question) ===")
    rows = []
    for w in POOL:
        runs = []
        for i in range(n):
            r = invoke(client, w, MATH_PROBE_Q, max_tokens=512, temperature=0.7)
            runs.append(r)
        oks = [r for r in runs if r["error"] is None]
        if oks:
            in_tok = statistics.mean(r["input_tokens"] for r in oks)
            out_tok = statistics.mean(r["output_tokens"] for r in oks)
            lat = statistics.mean(r["elapsed_s"] for r in oks)
        else:
            in_tok = out_tok = lat = 0
        first_err = next((r["error"] for r in runs if r["error"]), None)
        print(f"  ord={w['ord']} {w['name']:20s} "
              f"avg_in={in_tok:.0f} avg_out={out_tok:.0f} avg_t={lat:.2f}s "
              f"errs={n - len(oks)}/{n} {first_err or ''}")
        rows.append({
            "worker": w,
            "n": n,
            "n_ok": len(oks),
            "avg_input_tokens": round(in_tok, 1),
            "avg_output_tokens": round(out_tok, 1),
            "avg_latency_s": round(lat, 2),
            "first_error": first_err,
        })
    return rows


def stage_c_burst(client, ords=(6, 7, 8), concurrency=32):
    print(f"\n=== Stage C: burst probe ({concurrency}-way concurrent, ords={ords}) ===")
    rows = []
    for ord_idx in ords:
        w = POOL[ord_idx]
        print(f"\n  -- {w['name']} ({w['model_id']}) --")
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [
                ex.submit(invoke, client, w, MATH_PROBE_Q, 256, 0.7)
                for _ in range(concurrency)
            ]
            results = [f.result() for f in cf.as_completed(futures)]
        wall = time.time() - t0
        oks = [r for r in results if r["error"] is None]
        errs = [r["error"] for r in results if r["error"] is not None]
        total_in = sum(r["input_tokens"] for r in oks)
        total_out = sum(r["output_tokens"] for r in oks)
        total_tok = total_in + total_out
        print(f"    {len(oks)}/{concurrency} ok, wall={wall:.1f}s, "
              f"total_tokens={total_tok}, effective_TPM={int(60 * total_tok / wall) if wall else 0}")
        if errs:
            uniq = list({e.split(":")[0] for e in errs})
            print(f"    errors: {len(errs)} ({uniq})")
        rows.append({
            "worker": w,
            "concurrency": concurrency,
            "wall_s": round(wall, 2),
            "n_ok": len(oks),
            "n_err": len(errs),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "effective_tpm": int(60 * total_tok / wall) if wall else 0,
            "first_error": errs[0] if errs else None,
            "all_error_types": list({e.split(":")[0] for e in errs}),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="domains/autoresearch/blueprints/cost-aware-routing/results/preflight/worker_probe.json",
    )
    ap.add_argument("--burst", type=int, default=32)
    args = ap.parse_args()

    client = make_client()
    a = stage_a_ping(client)
    b = stage_b_token_probe(client, n=5)
    c = stage_c_burst(client, ords=(6, 7, 8), concurrency=args.burst)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"stage_a_ping": a, "stage_b_tokens": b, "stage_c_burst": c}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
