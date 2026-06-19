#!/usr/bin/env python3
"""Gate 1 — restored-replica token-id equality.

Run a deterministic completion (temp=0, top_p=1, top_k=-1, seed=42) against:
  (a) a freshly-warmed Ministral-3B pod (the "control")
  (b) each of the 4 restored replicas

Compare SHA256(first 64 token IDs) — must all match.

Usage:
    python gate1-token-equality.py --control-url http://control:8000 \
        --restore-urls http://r1:8000,http://r2:8000,http://r3:8000,http://r4:8000 \
        --model ministral-3b --output results/e1/gate1.json

Iter 5b note: in cluster, the URLs come from per-pod ClusterIP services or
direct pod-IP routing. Use `kubectl port-forward` only as a last-resort —
prefer ad-hoc Services or `kubectl exec` inside a curl pod.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import List

import urllib.request

PROMPT = "The capital of France is"
MAX_TOKENS = 64

def first_64_tokens(url: str, model: str) -> List[int]:
    body = json.dumps({
        "model": model,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 42,
        "logprobs": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    # vLLM returns choices[0].logprobs.tokens with str repr; for token IDs
    # we use the OpenAI-compatible token_ids field if present, else the
    # logprobs.tokens fallback (string tokens hashed).
    choice = data["choices"][0]
    lp = choice.get("logprobs") or {}
    if "token_ids" in lp:
        return list(lp["token_ids"])[:MAX_TOKENS]
    if "tokens" in lp:
        # Fall back to string tokens — still deterministic.
        return [hash(t) for t in lp["tokens"]][:MAX_TOKENS]
    # Last resort: hash the text.
    text = choice.get("text", "")
    return [ord(c) for c in text[:MAX_TOKENS]]

def digest(tokens: List[int]) -> str:
    h = hashlib.sha256()
    for t in tokens:
        h.update(str(int(t)).encode() + b",")
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-url", required=True)
    ap.add_argument("--restore-urls", required=True, help="comma-separated")
    ap.add_argument("--model", default="ministral-3b")
    ap.add_argument("--output", default="results/e1/gate1.json")
    args = ap.parse_args()

    control_tokens = first_64_tokens(args.control_url, args.model)
    control_digest = digest(control_tokens)
    print(f"control digest: {control_digest}")

    results = {
        "control_url": args.control_url,
        "control_digest": control_digest,
        "control_tokens": control_tokens,
        "restores": [],
        "all_match": True,
    }
    for url in args.restore_urls.split(","):
        url = url.strip()
        if not url:
            continue
        toks = first_64_tokens(url, args.model)
        d = digest(toks)
        match = d == control_digest
        results["restores"].append({"url": url, "digest": d, "tokens": toks, "match": match})
        if not match:
            results["all_match"] = False
        print(f"{url}: digest={d} match={match}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    return 0 if results["all_match"] else 1

if __name__ == "__main__":
    sys.exit(main())
