"""
Iter-0 histogram gate — brand-bias mitigation diagnostic.

Goal: before burning p4de spot hours, verify the routing prompt
(metadata header + balanced 9-shot examples) leads to a roughly
uniform pick distribution across the 9-worker pool on a held-out
question mix.

Pass criterion (per plan addendum §4):
  every worker picked between 5-20% of the time on 256 questions
  → 13-51 picks out of 256

If Opus (ord_8) > 25% (>64 picks), regenerate few-shot examples or
shuffle ordinal mapping and re-run.

Important: this script uses **Qwen3-32B as a proxy for the base
trainer model** (Qwen2.5-7B-Instruct). The 7B base will be tested
properly once p4de spot is up. Treat this as a directional smoke
gate, not a final calibration. A clean pass here predicts a clean
pass on Qwen2.5-7B; a fail here means we definitely have a prompt
problem.

Cost: 256 questions × Qwen3-32B (~$0.001/q) ≈ $0.30.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import random
import re
import time

import boto3
from botocore.config import Config

from few_shot import render_chat_messages, render_few_shot_block
from worker_pool import POOL, build_metadata_prompt

REGION = "us-west-2"
PROXY_MODEL_ID = "qwen.qwen3-32b-v1:0"

# Held-out question mix: 64 each of math, factual, code, reasoning.
# Drawn from common public benchmarks; intentionally diverse to exercise
# every worker's qualitative strength so balanced routing is plausible.
HELDOUT_QUESTIONS = [
    # --- math / arithmetic ---
    "What is 47 * 53?",
    "Compute 1 + 2 + 3 + ... + 100.",
    "What is sqrt(144)?",
    "Solve for x: 2x + 7 = 19.",
    "What is the derivative of x^3 + 2x with respect to x?",
    "If a circle has radius 5, what is its area? Use pi = 3.14159.",
    "What is 17 mod 5?",
    "Convert 0.625 to a fraction in lowest terms.",
    # --- factual / MMLU-style ---
    "Who wrote 'Pride and Prejudice'?",
    "What is the chemical symbol for gold?",
    "In what year did World War II end?",
    "What is the capital of Brazil?",
    "Which planet is closest to the Sun?",
    "What is the largest ocean on Earth?",
    "Who painted the Mona Lisa?",
    "What is the speed of light in m/s (approximately)?",
    # --- code / short ---
    "Write a Python function `is_palindrome(s: str) -> bool`.",
    "Write a one-line Python list comprehension that returns squares of 1..10.",
    "Write a SQL query to select all users from a table 'users' where age > 30.",
    "Implement Python `def reverse_string(s: str) -> str:` without using slicing.",
    "Write a regex that matches a US phone number in (XXX) XXX-XXXX format.",
    "Write a Python function to compute the nth Fibonacci number iteratively.",
    "Implement bubble sort in Python.",
    "Write a Python function that checks if a string is valid JSON.",
    # --- reasoning / hard ---
    "Prove that the square root of 2 is irrational.",
    "Explain why 0.999... = 1.",
    "Find all integer solutions to x^2 + y^2 = 25.",
    "If a clock shows 3:15, what is the angle between the hour and minute hands?",
    "A box contains 5 red and 3 blue balls. What is the probability of drawing 2 reds without replacement?",
    "Solve the recurrence T(n) = 2T(n/2) + n with T(1) = 1.",
    "What is the time complexity of merge sort? Justify your answer.",
    "Find the limit of (sin x) / x as x approaches 0.",
    # --- multilingual ---
    "Translate 'The quick brown fox jumps over the lazy dog' into Spanish.",
    "Translate the following French sentence into English: 'Le chat dort sur le canapé.'",
    "What is 'thank you' in Japanese (romaji)?",
    "Translate 'Where is the train station?' into German.",
    # --- structured output ---
    "List the planets in our solar system in order from the Sun. Reply as a JSON array.",
    "Reply with a JSON object containing the keys 'name' and 'year' for the discovery of penicillin.",
    "Reply as a YAML object with keys 'red', 'green', 'blue' giving the RGB hex for tomato red.",
    # --- tool-style ---
    "Given a function `lookup(symbol)` that returns a stock price, write code to fetch the price of AAPL.",
    "Given two arrays of integers, return their intersection sorted ascending.",
    "Given a binary tree, write a function to compute its maximum depth.",
    # --- fact + reasoning ---
    "Why does ice float on water?",
    "Explain the difference between mitosis and meiosis in two sentences.",
    "What is the function of mitochondria?",
    "Explain why the sky is blue in two sentences.",
    "What causes lightning?",
    # --- frontier-hard reasoning ---
    "Find all positive integer solutions (a, b, c) to a! + b! = c! with a, b, c <= 10.",
    "Prove that there are infinitely many primes congruent to 3 mod 4.",
    "Show that any planar graph can be 4-colored. (You may cite the theorem; provide a proof sketch.)",
    "What is the asymptotic density of squarefree integers? Derive it.",
    # --- short reasoning ---
    "If today is Tuesday, what day will it be in 100 days?",
    "A pizza is cut into 8 slices. If 3 friends share equally, how much does each get? Reduce the fraction.",
    "If a shirt is on sale at 20% off and originally costs $50, what is the sale price?",
    "John is twice as old as Mary. In 5 years, he will be 1.5 times as old. How old is John now?",
    # --- code review / debug ---
    "Find the bug: `def avg(xs): return sum(xs) / len(xs)` — what fails?",
    "What's wrong with this Python: `x = [1,2,3]; y = x; y.append(4); print(x)`?",
    "Why does `for i in range(len(arr)): arr.append(i)` cause an infinite loop?",
    # --- vocabulary / language ---
    "What is the past participle of 'go'?",
    "Define 'ephemeral' in one sentence.",
    "What part of speech is 'quickly'?",
    "Spell 'occasion' correctly.",
    # --- factoid bursts ---
    "What is the boiling point of water in Celsius?",
    "How many sides does a hexagon have?",
    "What is the molecular formula for water?",
    "How many bones are in the human body?",
    "What is the smallest prime number?",
    "What is the longest river in the world?",
]

assert len(HELDOUT_QUESTIONS) >= 64, f"need at least 64 questions, have {len(HELDOUT_QUESTIONS)}"


PICK_RE = re.compile(r"PICK\s+ord[_\s]*(\d)", re.IGNORECASE)


def make_client():
    cfg = Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=60)
    return boto3.client("bedrock-runtime", region_name=REGION, config=cfg)


def make_router_messages(question: str) -> tuple[str, list[dict]]:
    """Build (system_text, messages_list) for a single routing decision."""
    system = build_metadata_prompt() + "\n\n" + render_few_shot_block().split(
        "\nNow answer the actual question:\n"
    )[0]
    # ^ system holds metadata + few-shot block (no trailing user prompt)
    msgs = render_chat_messages()  # 9-shot as user/assistant turn pairs
    msgs.append({"role": "user", "content": question})
    return system, msgs


def route_one(client, question: str) -> dict:
    system, msgs = make_router_messages(question)
    bedrock_msgs = [
        {"role": m["role"], "content": [{"text": m["content"]}]} for m in msgs
    ]
    t0 = time.time()
    try:
        resp = client.converse(
            modelId=PROXY_MODEL_ID,
            system=[{"text": system}],
            messages=bedrock_msgs,
            inferenceConfig={"maxTokens": 64, "temperature": 0.7},
        )
        dur = time.time() - t0
        text = resp["output"]["message"]["content"][0].get("text", "")
        usage = resp.get("usage", {})
        m = PICK_RE.search(text)
        pick = int(m.group(1)) if m else None
        return {
            "question": question,
            "pick": pick,
            "raw": text[:300],
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "elapsed_s": round(dur, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "question": question,
            "pick": None,
            "raw": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_s": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


def run_gate(n: int = 256, workers: int = 16, seed: int = 17) -> dict:
    rng = random.Random(seed)
    questions = [rng.choice(HELDOUT_QUESTIONS) for _ in range(n)]
    client = make_client()
    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(route_one, client, q): i for i, q in enumerate(questions)}
        for done, fut in enumerate(cf.as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            if done % 32 == 0 or done == n:
                print(f"  [{done}/{n}] last_pick={r['pick']} err={r['error']}", flush=True)

    picks = collections.Counter(r["pick"] for r in rows)
    parse_failures = picks.get(None, 0)
    valid = n - parse_failures
    pct = {ord_: 100 * picks.get(ord_, 0) / n for ord_ in range(9)}
    pct_valid = {ord_: 100 * picks.get(ord_, 0) / max(valid, 1) for ord_ in range(9)}

    # Pass criterion: every ord between 5% and 20% of total (not valid).
    pass_per_ord = {ord_: 5.0 <= pct[ord_] <= 20.0 for ord_ in range(9)}
    overall_pass = all(pass_per_ord.values()) and parse_failures < 0.10 * n

    summary = {
        "n": n,
        "valid": valid,
        "parse_failures": parse_failures,
        "pick_counts": dict(picks),
        "pick_pct_total": pct,
        "pick_pct_valid": pct_valid,
        "per_ord_pass": pass_per_ord,
        "overall_pass": overall_pass,
        "proxy_model": PROXY_MODEL_ID,
        "seed": seed,
    }
    return {"summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument(
        "--output",
        default="domains/autoresearch/blueprints/cost-aware-routing/results/preflight/iter0_histogram.json",
    )
    args = ap.parse_args()

    result = run_gate(args.n, args.workers, args.seed)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    s = result["summary"]
    print("\n=== ITER-0 HISTOGRAM GATE ===")
    print(f"  n={s['n']}, valid={s['valid']} ({s['parse_failures']} parse failures)")
    print(f"  proxy model: {s['proxy_model']}")
    print()
    print(f"  ord  worker            picks   pct_total  pct_valid  pass?")
    for ord_ in range(9):
        w = POOL[ord_]
        cnt = s["pick_counts"].get(ord_, 0)
        ok = "PASS" if s["per_ord_pass"][ord_] else "FAIL"
        print(f"  ord_{ord_}  {w.name:18s} {cnt:>5}    {s['pick_pct_total'][ord_]:>5.1f}%    "
              f"{s['pick_pct_valid'][ord_]:>5.1f}%    {ok}")
    print()
    print(f"  OVERALL: {'PASS' if s['overall_pass'] else 'FAIL'}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
