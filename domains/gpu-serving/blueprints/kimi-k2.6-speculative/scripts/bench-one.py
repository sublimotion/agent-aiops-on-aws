#!/usr/bin/env python3
"""bench-one.py — Single concurrency benchmark against /generate (SGLang) or /v1/completions (vLLM).

Usage:
  python3 bench-one.py <port> <concurrency> <input_len> <output_len> <reqs_per_conc> <out_file> [endpoint]

endpoint: sglang (default) | vllm
"""
import json, sys, time, asyncio, aiohttp, statistics

port = int(sys.argv[1])
conc = int(sys.argv[2])
ilen = int(sys.argv[3])
olen = int(sys.argv[4])
reqs = int(sys.argv[5])
out_file = sys.argv[6]
endpoint = sys.argv[7] if len(sys.argv) > 7 else "sglang"

prompt = " ".join(["hello"] * ilen)


async def one_sglang(session, idx):
    body = {"text": prompt, "sampling_params": {"max_new_tokens": olen, "temperature": 0.0}}
    t0 = time.time()
    async with session.post(f"http://localhost:{port}/generate", json=body) as r:
        data = await r.json()
    t1 = time.time()
    meta = data.get("meta_info", {})
    tokens = meta.get("completion_tokens") or olen
    accept_rate = meta.get("spec_accept_rate")
    accept_len = meta.get("spec_accept_length")
    return t1 - t0, tokens, accept_rate, accept_len


async def one_vllm(session, idx):
    body = {"model": "kimi-k26-fp8", "prompt": prompt, "max_tokens": olen, "temperature": 0.0}
    t0 = time.time()
    async with session.post(f"http://localhost:{port}/v1/completions", json=body) as r:
        data = await r.json()
    t1 = time.time()
    tokens = data.get("usage", {}).get("completion_tokens") or olen
    return t1 - t0, tokens, None, None


one = one_sglang if endpoint == "sglang" else one_vllm


async def main():
    connector = aiohttp.TCPConnector(limit=conc * 2)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        sem = asyncio.Semaphore(conc)

        async def gated(i):
            async with sem:
                return await one(session, i)

        total_reqs = reqs * max(1, conc // 4) if conc >= 64 else reqs * conc
        total_reqs = max(total_reqs, 4)
        t_start = time.time()
        results = await asyncio.gather(*[gated(i) for i in range(total_reqs)], return_exceptions=True)
        t_end = time.time()

    ok = [r for r in results if not isinstance(r, Exception)]
    errs = len(results) - len(ok)
    if not ok:
        json.dump({"concurrency": conc, "error": "all requests failed", "errors": errs}, open(out_file, "w"))
        return
    lats = [r[0] for r in ok]
    total_tokens = sum(r[1] for r in ok)
    agg = total_tokens / (t_end - t_start)
    per_req = statistics.mean([r[1] / r[0] for r in ok])

    accept_rates = [r[2] for r in ok if r[2] is not None]
    accept_lens = [r[3] for r in ok if r[3] is not None]

    out = {
        "concurrency": conc,
        "requests_ok": len(ok),
        "requests_err": errs,
        "input_len": ilen,
        "output_len_target": olen,
        "total_tokens": total_tokens,
        "duration_s": round(t_end - t_start, 3),
        "agg_tok_per_s": round(agg, 1),
        "per_req_tok_per_s": round(per_req, 1),
        "p50_latency_s": round(statistics.median(lats), 3),
        "p90_latency_s": round(sorted(lats)[int(len(lats) * 0.9)], 3) if len(lats) > 10 else round(max(lats), 3),
        "p99_latency_s": round(sorted(lats)[int(len(lats) * 0.99)], 3) if len(lats) > 10 else round(max(lats), 3),
    }
    if accept_rates:
        out["spec_accept_rate_mean"] = round(statistics.mean(accept_rates), 3)
    if accept_lens:
        out["spec_accept_length_mean"] = round(statistics.mean(accept_lens), 3)
    json.dump(out, open(out_file, "w"), indent=2)
    print(json.dumps(out, indent=2))


asyncio.run(main())
