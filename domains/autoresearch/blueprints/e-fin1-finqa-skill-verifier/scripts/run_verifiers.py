#!/usr/bin/env python3
"""Stage 2-3: run confirmatory + adversarial verifiers over labelled answers.

For each agent answer (from generate_answers.py output), runs:
  - Verifier A (confirmatory): 1 call, 1-5 rating.
  - Verifier B (adversarial v009 analog): 4-call temperature ensemble
    (1 @ t=0.0, 3 @ t=0.3). 4/4 unanimous likely_correct => confident-pass.
    No confirmatory v001 gate (verifier-reward T10b: v009-only is the ceiling).

The verifier sees the SAME context the agent saw, plus the agent's answer +
reasoning. The verifier does NOT see exe_ans (that's the ground truth we score
against).

Output JSONL per example merges the generation record with:
  {conf_rating, conf_verdict, conf_*_tokens, conf_cost,
   adv_verdicts (list of 4), adv_confident (bool), adv_*_tokens, adv_cost}
"""
import argparse
import json
import sys

from finqa_common import (build_context, call_bedrock, cost_usd, extract_json)

# Load data file globally for context rebuild (keyed by id)
ADV_TEMPS = [0.0, 0.3, 0.3, 0.3]  # 1 @ t=0, 3 @ t=0.3


def load_rubric(path):
    return open(path).read()


def build_verify_prompt(rubric, context, question, agent_answer, reasoning):
    return f"""{rubric}

---
QUESTION: {question}

FINANCIAL CONTEXT:
{context}

ANALYST'S REASONING: {reasoning}

ANALYST'S FINAL ANSWER: {agent_answer}
---

Now perform your evaluation and respond with ONLY the JSON object."""


def run_confirmatory(rubric, ctx, rec, model):
    prompt = build_verify_prompt(rubric, ctx, rec["question"],
                                 rec["agent_answer_raw"], rec["reasoning"])
    r = call_bedrock(prompt, model_key=model, temperature=0.0, max_tokens=512)
    parsed = extract_json(r["text"]) or {}
    return {
        "conf_rating": parsed.get("rating"),
        "conf_verdict": parsed.get("verdict"),
        "conf_input_tokens": r["input_tokens"],
        "conf_output_tokens": r["output_tokens"],
        "conf_cost": cost_usd(r["input_tokens"], r["output_tokens"], model),
    }


def run_adversarial(rubric, ctx, rec, model):
    prompt = build_verify_prompt(rubric, ctx, rec["question"],
                                 rec["agent_answer_raw"], rec["reasoning"])
    verdicts = []
    in_tok = out_tok = 0
    for t in ADV_TEMPS:
        r = call_bedrock(prompt, model_key=model, temperature=t, max_tokens=768)
        parsed = extract_json(r["text"]) or {}
        verdicts.append(parsed.get("verdict", "parse_error"))
        in_tok += r["input_tokens"]
        out_tok += r["output_tokens"]
    confident = all(v == "likely_correct" for v in verdicts)
    return {
        "adv_verdicts": verdicts,
        "adv_confident": confident,
        "adv_input_tokens": in_tok,
        "adv_output_tokens": out_tok,
        "adv_cost": cost_usd(in_tok, out_tok, model),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="generate_answers.py output")
    ap.add_argument("--data", required=True, help="sampled dev json (for context)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf-rubric",
                    default="../skills/finqa-verifier/versions/confirmatory.md")
    ap.add_argument("--adv-rubric",
                    default="../skills/finqa-verifier/versions/adversarial.md")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    conf_rubric = load_rubric(args.conf_rubric)
    adv_rubric = load_rubric(args.adv_rubric)

    data = json.load(open(args.data))
    by_id = {ex.get("id", f"idx_{i}"): ex for i, ex in enumerate(data)}

    recs = [json.loads(l) for l in open(args.answers) if l.strip()]
    if args.limit:
        recs = recs[:args.limit]

    total_conf_cost = total_adv_cost = 0.0
    with open(args.out, "w") as out:
        for i, rec in enumerate(recs):
            ex = by_id.get(rec["id"])
            if ex is None:
                print(f"[{i}] missing context for {rec['id']}", file=sys.stderr)
                continue
            ctx = build_context(ex)
            try:
                conf = run_confirmatory(conf_rubric, ctx, rec, args.model)
                adv = run_adversarial(adv_rubric, ctx, rec, args.model)
            except Exception as e:
                print(f"[{i}] ERROR: {e}", file=sys.stderr)
                continue
            merged = {**rec, **conf, **adv}
            total_conf_cost += conf["conf_cost"]
            total_adv_cost += adv["adv_cost"]
            out.write(json.dumps(merged) + "\n")
            out.flush()
            print(f"[{i+1}/{len(recs)}] match={rec['match']} "
                  f"conf={conf['conf_verdict']}({conf['conf_rating']}) "
                  f"adv={adv['adv_verdicts']} confident={adv['adv_confident']}")

    n = len(recs)
    print(f"\n== verification done (n={n}) ==")
    print(f"confirmatory cost: ${total_conf_cost:.4f} (${total_conf_cost/n:.5f}/eval)")
    print(f"adversarial cost:  ${total_adv_cost:.4f} (${total_adv_cost/n:.5f}/eval)")


if __name__ == "__main__":
    main()
