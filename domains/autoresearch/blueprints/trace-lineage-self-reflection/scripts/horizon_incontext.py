#!/usr/bin/env python3
"""Forced-in-context long-horizon driver — faithful to arXiv:2509.09677.

Reproduces the drift condition our file-based tasks could NOT: the model must
carry running state IN CONTEXT across turns, one operation per turn, with NO
tools and NO file offload (files would BE its memory and prevent forgetting).

Task (dict_sum): K accounts start at `start`. Each turn we reveal ONE operation
(acct += delta) and ask the model to output the FULL current balance of every
account. It must accumulate across turns from context alone. We grade the final
turn's balances against ground truth → execution_accuracy. Because success ≈ p^H
and no per-step tool corrects it, accuracy decays with horizon (the drift).

Self-conditioning dial (--inject-error-rate): with probability p, after the
model answers, we CORRUPT one balance in the running truth we echo back — i.e.
inject an error into the context the model conditions on next turn. 2509.09677
shows this accelerates collapse; it's also exactly the "stale fact in context"
the reliability layer targets.

Deterministic ops (LCG on --seed). Bedrock Converse (bearer-token auth).

Usage:
  horizon_incontext.py --seed 0 --k 4 --turns 40 [--inject-error-rate 0.0] \
     [--model us.anthropic.claude-sonnet-4-6] [--json out.json]
"""
import argparse
import json
import re
import sys

import boto3


def lcg(seed):
    x = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    while True:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        yield x


def build_ops(seed, k, turns):
    rng = lcg(seed + 1)
    keys = [f"acct_{i}" for i in range(k)]
    ops = []
    for _ in range(turns):
        key = keys[next(rng) % k]
        delta = (next(rng) % 41) - 20
        ops.append((key, delta))
    return keys, ops


def parse_balances(text, keys):
    """Extract 'acct_i: N' pairs from the model's reply. Word-boundary anchored
    so `acct_1` does not swallow `acct_12`; longest keys first for safety."""
    got = {}
    for k in sorted(keys, key=len, reverse=True):
        m = re.search(rf"\b{re.escape(k)}\b\s*[:=]\s*(-?\d+)", text)
        if m:
            got[k] = int(m.group(1))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--inject-error-rate", type=float, default=0.0,
                    help="self-conditioning: prob. of corrupting an echoed balance per turn")
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-6")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    keys, ops = build_ops(args.seed, args.k, args.turns)
    truth = {k: args.start for k in keys}
    client = boto3.client("bedrock-runtime", region_name=args.region)
    inj = lcg(args.seed + 7)  # separate deterministic stream for injection decisions

    sys_prompt = (
        "You are tracking account balances across many turns. Each turn I give you "
        "ONE operation (an account and a signed delta). Apply it to your running "
        "totals and reply with the CURRENT balance of EVERY account, one per line, "
        f"in the exact form 'acct_i: N'. You have {args.k} accounts, each starting "
        f"at {args.start}. Do not use tools; track the balances yourself. Reply with "
        "ONLY the balance lines, nothing else."
    )
    messages = []
    per_turn = []
    last_got = {}

    for t, (key, delta) in enumerate(ops):
        truth[key] += delta
        user_msg = f"Operation {t+1}: {key} {'+' if delta>=0 else ''}{delta}"
        messages.append({"role": "user", "content": [{"text": user_msg}]})
        try:
            r = client.converse(
                modelId=args.model,
                system=[{"text": sys_prompt}],
                messages=messages,
                inferenceConfig={"maxTokens": 400, "temperature": 0},
            )
            reply = r["output"]["message"]["content"][0]["text"]
        except Exception as e:
            print(f"converse failed at turn {t+1}: {e}", file=sys.stderr)
            break
        messages.append({"role": "assistant", "content": [{"text": reply}]})
        got = parse_balances(reply, keys)
        last_got = got or last_got
        # accuracy this turn = accounts matching truth
        correct = sum(1 for k in keys if got.get(k) == truth[k])
        per_turn.append({"turn": t + 1, "correct": correct, "k": args.k})

        # self-conditioning: corrupt one echoed balance the model will condition on
        if args.inject_error_rate > 0 and (next(inj) % 1000) / 1000.0 < args.inject_error_rate:
            bad_key = keys[next(inj) % args.k]
            # replace that line in the assistant message we keep in context
            corrupted = re.sub(rf"({re.escape(bad_key)}\s*[:=]\s*)-?\d+",
                               rf"\g<1>{truth[bad_key] + 7}", reply)
            messages[-1]["content"][0]["text"] = corrupted

    final_correct = sum(1 for k in keys if last_got.get(k) == truth[k])
    verdict = {
        "seed": args.seed, "k": args.k, "turns": args.turns,
        "inject_error_rate": args.inject_error_rate, "model": args.model,
        "final_execution_accuracy": round(final_correct / args.k, 4) if args.k else 0.0,
        "final_correct": final_correct,
        "final_got": last_got, "final_truth": truth,
        "accuracy_curve": [round(p["correct"] / p["k"], 3) for p in per_turn],
    }
    out = json.dumps(verdict, indent=2)
    if args.json:
        with open(args.json, "w") as f:
            f.write(out)
    print(out)


if __name__ == "__main__":
    main()
