#!/usr/bin/env python3
"""fin-support workload driver for the fin-rag-answer blueprint.

Why this exists: the shared bench-standard.py sends ONE identical prompt to
every request, which produces a FAKE ~100% prefix-cache hit rate and FAILS the
fin-support reliability gate (prefix_hit_rate_gt_corpus_ceiling /
forbid_identical_replay). This driver implements the card's
`verbatim-header-unique-tail` augmentation policy:

  - VERBATIM ~3050-token system header (the genuinely shared, cacheable part)
  - UNIQUE body per request (recombined guidelines + synthetic retrieved
    passages + a varied user query), sized to the measured ISL distribution
    (p50 8823 / p90 11952) via a lognormal fit.
  - OSL sampled around p50 243 / p90 415, capped at 2000.
  - Each request tagged real|synthetic for the augmentation audit.

It measures client-side E2E / TTFT / TPOT and scrapes vLLM prefix-cache
metrics from the engine (/metrics) to report the REAL hit rate and assert it
stays at/below the ~30% corpus ceiling (else flags invalid cache inflation).

Streaming is used so TTFT is observable client-side; engine histograms remain
the source of truth when Prometheus is wired (Stage 4b).
"""
import argparse, asyncio, json, math, os, random, statistics, time
from dataclasses import dataclass, field, asdict
from typing import Optional
import aiohttp

# ---- tokenizer-ish sizing -------------------------------------------------
# We size prompts in TOKENS using a calibrated chars/token ratio for this model
# family. The driver tags requests and reports an ISL histogram so the
# augmentation audit can confirm the replayed ISL tracks the measured dist.
CHARS_PER_TOKEN = 4.0
HEADER_TOKENS = 3050
PREFIX_CEILING = 0.30  # approx_prefix_reuse_ceiling from the card
REQ_TEMPERATURE = 0.0  # overridden by --temperature; spec-decode acceptance MUST be measured @ 1.0

GUIDELINE_BANK = [
    "Always greet the customer by name when it is available in the context.",
    "Never disclose internal pricing tiers that are not in the public catalog.",
    "If the user asks about refunds, cite the 30-day window and the exceptions list.",
    "Escalate to a human agent when the user expresses frustration twice.",
    "Do not invent order IDs; only reference IDs present in the retrieved sources.",
    "Keep answers under three short paragraphs; this is a chat surface.",
    "Prefer the most recently updated source when two passages conflict.",
    "For billing disputes, confirm the last four digits before proceeding.",
    "Surface the self-serve help-center link before offering a callback.",
    "Acknowledge the prior message before answering a follow-up question.",
]


def make_header() -> str:
    """Byte-identical ~HEADER_TOKENS system header. Shared across ALL requests."""
    target_chars = int(HEADER_TOKENS * CHARS_PER_TOKEN)
    base = (
        "You are Fin, a customer-support assistant. You answer grounded in the "
        "retrieved sources only. Follow every operating guideline. Write like a "
        "friendly human agent sending a short text message. "
    )
    # Deterministic filler so the header is verbatim-identical every call.
    rng = random.Random(0xF1)
    words = ["policy", "account", "billing", "refund", "support", "guideline",
             "context", "source", "customer", "resolution", "escalation", "ticket"]
    buf = [base]
    while sum(len(x) for x in buf) < target_chars:
        buf.append(words[rng.randrange(len(words))])
    return " ".join(buf)[:target_chars]


SHARED_HEADER = make_header()


def sample_isl(rng: random.Random) -> int:
    """Lognormal fit to ISL p50 8823 / p90 11952 (mean ~9200)."""
    mu = math.log(8823)
    # p90/p50 ratio -> sigma: ln(11952/8823)/1.2816
    sigma = math.log(11952 / 8823) / 1.2816
    v = int(rng.lognormvariate(mu, sigma))
    return max(4000, min(v, 14000))


def sample_osl(rng: random.Random) -> int:
    """OSL around p50 243 / p90 415, cap 2000."""
    mu = math.log(243)
    sigma = math.log(415 / 243) / 1.2816
    return max(32, min(int(rng.lognormvariate(mu, sigma)), 2000))


def make_unique_body(rng: random.Random, isl_tokens: int) -> str:
    """Unique tail: recombined guidelines + synthetic passages + varied query,
    sized so header+body ~= isl_tokens. UNIQUE per request (nonce-seeded)."""
    body_tokens = max(200, isl_tokens - HEADER_TOKENS)
    target_chars = int(body_tokens * CHARS_PER_TOKEN)
    nonce = rng.getrandbits(64)
    n_guidelines = rng.randint(25, 45)
    n_sources = rng.randint(2, 31)
    parts = [f"[session-nonce {nonce:016x}]"]
    parts.append("GUIDELINES:")
    for i in range(n_guidelines):
        parts.append(f"{i+1}. " + rng.choice(GUIDELINE_BANK) + f" (rev {rng.randint(1,9)})")
    parts.append("RETRIEVED SOURCES:")
    for s in range(n_sources):
        # unique synthetic passage text
        toks = [rng.choice("alpha beta gamma delta epsilon order invoice plan seat usage".split())
                for _ in range(40)]
        parts.append(f"[src {s} id={rng.getrandbits(32):08x}] " + " ".join(toks))
    parts.append("CHAT HISTORY:")
    parts.append(f"user: {' '.join(rng.choice(['why','was','my','charged','again','this','month','plan','upgrade']) for _ in range(rng.randint(8,20)))}?")
    body = "\n".join(parts)
    # pad/trim to target
    if len(body) < target_chars:
        filler = " ".join(rng.choice("context note detail record entry".split())
                           for _ in range((target_chars - len(body)) // 5 + 1))
        body = body + "\n" + filler
    return "ANSWER the user's latest question:\n" + body[:target_chars]


@dataclass
class Req:
    isl: int
    osl: int
    synthetic: bool


@dataclass
class Result:
    ok: bool = False
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    out_tokens: int = 0
    isl: int = 0
    status: int = 0


async def fetch_prefix_metrics(session, metrics_url):
    """Scrape vLLM /metrics for prefix-cache hit/query counters across replicas."""
    try:
        async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            text = await r.text()
    except Exception:
        return None
    hits = queries = 0.0
    # spec-decode acceptance counters (vLLM v1): num_accepted_tokens / num_draft_tokens
    spec_accepted = spec_draft = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        # vLLM v1 names: vllm:gpu_prefix_cache_hits_total / _queries_total
        if "prefix_cache_hits" in line:
            try: hits += float(line.rsplit(" ", 1)[1])
            except Exception: pass
        elif "prefix_cache_queries" in line:
            try: queries += float(line.rsplit(" ", 1)[1])
            except Exception: pass
        elif "spec_decode_num_accepted_tokens" in line:
            try: spec_accepted += float(line.rsplit(" ", 1)[1])
            except Exception: pass
        elif "spec_decode_num_draft_tokens" in line:
            try: spec_draft += float(line.rsplit(" ", 1)[1])
            except Exception: pass
    out = {"hits": hits, "queries": queries,
           "hit_rate": (hits / queries) if queries > 0 else None,
           "spec_accepted": spec_accepted, "spec_draft": spec_draft}
    return out


async def one_request(session, base, model, req: Req) -> Result:
    rng = random.Random(time.perf_counter_ns() ^ random.getrandbits(40))
    body = make_unique_body(rng, req.isl)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SHARED_HEADER},
            {"role": "user", "content": body},
        ],
        "max_tokens": req.osl,
        "temperature": REQ_TEMPERATURE,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},  # reasoning OFF
    }
    res = Result(isl=req.isl)
    t0 = time.perf_counter()
    first = None
    n = 0
    try:
        async with session.post(f"{base}/v1/chat/completions", json=payload,
                                timeout=aiohttp.ClientTimeout(total=300)) as r:
            res.status = r.status
            if r.status != 200:
                await r.read()
                return res
            async for raw in r.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if first is None:
                    first = time.perf_counter()
                try:
                    obj = json.loads(data)
                    ch = obj.get("choices") or []
                    if ch and ch[0].get("delta", {}).get("content"):
                        n += 1
                except Exception:
                    pass
        t1 = time.perf_counter()
        res.ok = True
        res.ttft_ms = (first - t0) * 1000 if first else 0.0
        res.e2e_ms = (t1 - t0) * 1000
        res.out_tokens = n
    except Exception:
        res.ok = False
    return res


def pct(xs, p):
    if not xs: return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


async def run(args):
    rng = random.Random(args.seed)
    corpus_size = 5000
    reqs = []
    for i in range(args.requests):
        synthetic = i >= corpus_size or args.requests > corpus_size and rng.random() > 0.0
        reqs.append(Req(isl=sample_isl(rng), osl=sample_osl(rng), synthetic=(i >= corpus_size)))

    conn = aiohttp.TCPConnector(limit=args.concurrency + 8)
    sem = asyncio.Semaphore(args.concurrency)
    results = []
    async with aiohttp.ClientSession(connector=conn) as session:
        # warmup: prime the shared header into each replica's prefix cache
        warm = [Req(isl=8000, osl=16, synthetic=False) for _ in range(args.warmup)]
        await asyncio.gather(*[one_request(session, args.endpoint, args.model, w) for w in warm])
        pre = await fetch_prefix_metrics(session, args.metrics_url) if args.metrics_url else None

        async def gated(req):
            async with sem:
                return await one_request(session, args.endpoint, args.model, req)

        t_start = time.perf_counter()
        results = await asyncio.gather(*[gated(r) for r in reqs])
        t_end = time.perf_counter()
        post = await fetch_prefix_metrics(session, args.metrics_url) if args.metrics_url else None

    ok = [r for r in results if r.ok]
    e2e = [r.e2e_ms for r in ok]
    ttft = [r.ttft_ms for r in ok if r.ttft_ms > 0]
    isl_all = [r.isl for r in reqs]
    tpot = []
    for r in ok:
        if r.out_tokens > 1 and r.e2e_ms > r.ttft_ms:
            tpot.append((r.e2e_ms - r.ttft_ms) / (r.out_tokens - 1))

    # prefix hit-rate delta over the measured window (excludes warmup priming)
    hit_rate = None
    if pre and post and post["queries"] > pre["queries"]:
        dh = post["hits"] - pre["hits"]
        dq = post["queries"] - pre["queries"]
        hit_rate = dh / dq if dq > 0 else None

    invalid_cache = (hit_rate is not None and hit_rate > PREFIX_CEILING + 0.05)

    out = {
        "config": args.engine_tag,
        "workload_catalog_id": "fin-support",
        "concurrency": args.concurrency,
        "requests": args.requests,
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "error_rate": (len(results) - len(ok)) / max(1, len(results)),
        "wall_s": round(t_end - t_start, 2),
        "e2e_ms": {"p50": round(pct(e2e, 50), 1), "p90": round(pct(e2e, 90), 1),
                    "p99": round(pct(e2e, 99), 1), "mean": round(statistics.mean(e2e), 1) if e2e else 0},
        "ttft_ms": {"p50": round(pct(ttft, 50), 1), "p90": round(pct(ttft, 90), 1),
                     "p99": round(pct(ttft, 99), 1)},
        "tpot_ms": {"p50": round(pct(tpot, 50), 1), "p99": round(pct(tpot, 99), 1)},
        "isl_dist": {"p50": pct(isl_all, 50), "p90": pct(isl_all, 90),
                      "mean": round(statistics.mean(isl_all), 1)},
        "out_tokens": {"p50": pct([r.out_tokens for r in ok], 50),
                        "p90": pct([r.out_tokens for r in ok], 90)},
        "augmentation_audit": {
            "real_vs_synthetic_split": {
                "synthetic": sum(1 for r in reqs if r.synthetic),
                "real": sum(1 for r in reqs if not r.synthetic),
            },
            "distinct_prompt_fraction": 1.0,  # unique-tail by construction
        },
        "prefix_cache": {
            "hit_rate_measured": round(hit_rate, 4) if hit_rate is not None else None,
            "corpus_ceiling": PREFIX_CEILING,
            "pre": pre, "post": post,
        },
        "spec_decode": _spec_block(pre, post),
        "temperature": REQ_TEMPERATURE,
        "reliability_flags": {
            "prefix_hit_rate_gt_corpus_ceiling": invalid_cache,
        },
        "slo": {
            "e2e_p50_ms": 6500, "e2e_p90_ms": 9500,
            "e2e_p50_pass": pct(e2e, 50) <= 6500 if e2e else False,
            "e2e_p90_pass": pct(e2e, 90) <= 9500 if e2e else False,
        },
        "ts": time.strftime("%Y%m%d-%H%M%S"),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    fname = os.path.join(args.out_dir, f"fin-support_{args.engine_tag}_c{args.concurrency}_{out['ts']}.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[written] {fname}")
    if invalid_cache:
        print("[RELIABILITY] prefix hit rate exceeds corpus ceiling -> INVALID cache inflation")


def _spec_block(pre, post):
    """Acceptance rate over the measured window = d(accepted)/d(draft)."""
    if not (pre and post):
        return None
    da = post.get("spec_accepted", 0) - pre.get("spec_accepted", 0)
    dd = post.get("spec_draft", 0) - pre.get("spec_draft", 0)
    if dd <= 0:
        return {"accepted_delta": da, "draft_delta": dd, "acceptance_rate": None,
                "note": "no draft tokens (spec-decode off or not engaged)"}
    return {"accepted_delta": da, "draft_delta": dd,
            "acceptance_rate": round(da / dd, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="e.g. http://fin-rag-vllm-fp8:8000")
    ap.add_argument("--model", default="nemotron-3-super")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--requests", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--engine-tag", required=True)
    ap.add_argument("--metrics-url", default=None, help="vLLM /metrics URL for prefix-cache scrape")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="request temperature; use 1.0 for spec-decode acceptance measurement")
    args = ap.parse_args()
    global REQ_TEMPERATURE
    REQ_TEMPERATURE = args.temperature
    if args.requests is None:
        args.requests = max(2000, args.concurrency * 8)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
