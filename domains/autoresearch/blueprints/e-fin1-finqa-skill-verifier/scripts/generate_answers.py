#!/usr/bin/env python3
"""Stage 1: generate agent answers over FinQA dev, score against exe_ans.

A mid-tier model (default haiku) reads question + table + gold supporting
text and produces a numeric answer + reasoning. We then exact-match the
stated answer against qa.exe_ans. This produces the labelled set the two
verifiers are scored against.

Output JSONL per example:
  {id, question, exe_ans, agent_answer_raw, agent_number, reasoning,
   match (bool), input_tokens, output_tokens, cost_usd, latency_ms}
"""
import argparse
import json
import sys

from finqa_common import (build_context, call_bedrock, cost_usd, exact_match,
                          extract_json)

PROMPT = """You are a financial analyst answering a numeric question from a 10-K filing.

{context}

QUESTION: {question}

Work through the calculation step by step, then give your final numeric answer.
Respond with ONLY a JSON object:
{{
  "reasoning": "<your step-by-step calculation>",
  "answer": "<the final numeric answer, e.g. 127.4 or 12.5%>"
}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    data = json.load(open(args.data))
    if args.limit:
        data = data[:args.limit]

    n_match = 0
    total_cost = 0.0
    with open(args.out, "w") as out:
        for i, ex in enumerate(data):
            qa = ex["qa"]
            ctx = build_context(ex)
            prompt = PROMPT.format(context=ctx, question=qa["question"])
            try:
                r = call_bedrock(prompt, model_key=args.model,
                                 temperature=args.temperature, max_tokens=1024)
            except Exception as e:
                print(f"[{i}] ERROR: {e}", file=sys.stderr)
                continue
            parsed = extract_json(r["text"]) or {}
            answer_raw = parsed.get("answer", r["text"][:200])
            reasoning = parsed.get("reasoning", "")
            match = exact_match(answer_raw, qa["exe_ans"])
            c = cost_usd(r["input_tokens"], r["output_tokens"], args.model)
            total_cost += c
            if match:
                n_match += 1
            rec = {
                "id": ex.get("id", f"idx_{i}"),
                "question": qa["question"],
                "exe_ans": qa["exe_ans"],
                "agent_answer_raw": answer_raw,
                "reasoning": reasoning,
                "match": match,
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "cost_usd": c,
                "latency_ms": r["latency_ms"],
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"[{i+1}/{len(data)}] match={match} ans={answer_raw!r} "
                  f"gold={qa['exe_ans']!r} (in={r['input_tokens']} out={r['output_tokens']})")

    n = len(data)
    print(f"\n== generation done ==")
    print(f"accuracy: {n_match}/{n} = {n_match/n:.3f}")
    print(f"total generation cost: ${total_cost:.4f} (${total_cost/n:.5f}/example)")


if __name__ == "__main__":
    main()
