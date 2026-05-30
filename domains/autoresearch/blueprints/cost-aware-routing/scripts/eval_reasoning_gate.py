"""Reproduce the vLLM Semantic Router reasoning-gate baseline.

Per "When to Reason: Semantic Router for vLLM" (arxiv 2510.08731):
  Binary classifier: reasoning-required vs not.
  Route reasoning-required to a strong reasoning model; rest to a cheap default.

Adaptation to our 9-worker Bedrock pool:
  reasoning-required -> Opus 4.7 (ord_8) — strongest reasoning worker
  not-reasoning      -> Gemma-3-27B (ord_0) — cheapest worker
Optional alt: route non-reasoning to Qwen-Coder-480B (ord_3) since it's
  strong on the broad set.

We construct the binary label two ways:
  (A) Project our 5-class category to binary:
      {math, reasoning} -> reasoning-required
      {code, factual, open-domain} -> not
  (B) Use per-question difficulty signal: a question is "reasoning-required"
      if Opus is correct AND ≥3 cheap-tier workers (ord 0-3) are wrong.
      This is closer to what the paper means.

We compare both projections + the existing 5-class classifier-router
+ always-static baselines on the same 96-question held-out eval split
that classifier_router_eval used.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, "domains/autoresearch/blueprints/cost-aware-routing/scripts")
from router_policy import cost_aware_reward
from eval_classifier_router import (
    CATEGORIES,
    load_eval_split,
    load_full_rollouts,
    classify_questions,
    build_train_quality_table,
    build_eval_lookup,
)


# --- Reasoning-gate label projections -----------------------------------

REASONING_CATEGORIES = {"math", "reasoning"}

def project_5class_to_binary(category: str) -> str:
    """5-class category -> {reasoning, not}."""
    return "reasoning" if category in REASONING_CATEGORIES else "not"


def question_needs_reasoning_oracle(rollouts_by_qid_ord: dict, qid: str) -> bool:
    """Per-question oracle 'needs reasoning' label.

    True if Opus (ord_8) is correct AND ≥3 of {Gemma, gpt-oss, qwen3-32b,
    qwen-coder} are wrong.
    """
    cheap_ords = [0, 1, 2, 3]
    cheap_wrong = sum(
        1 for o in cheap_ords
        if not rollouts_by_qid_ord.get((qid, o), {"is_correct": True})["is_correct"]
    )
    opus_correct = rollouts_by_qid_ord.get((qid, 8), {"is_correct": False})["is_correct"]
    return opus_correct and cheap_wrong >= 3


# --- Routing pick functions ---------------------------------------------

def make_reasoning_gate_pick(label_fn, reasoning_ord: int = 8, default_ord: int = 0):
    """Returns a pick function: question/qid -> worker_id.

    label_fn(qid) -> 'reasoning' or 'not'.
    """
    def pick(qid: str, alpha: float) -> int:
        return reasoning_ord if label_fn(qid) == "reasoning" else default_ord
    return pick


# --- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="domains/autoresearch/blueprints/cost-aware-routing/data/augmented_baseline_500q.jsonl")
    ap.add_argument("--classifier", default="domains/autoresearch/blueprints/cost-aware-routing/artifacts/classifier")
    ap.add_argument("--output", default="domains/autoresearch/blueprints/cost-aware-routing/results/runs/reasoning_gate_eval.json")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    eval_rows = load_eval_split(args.data, seed=args.seed)
    eval_qids = {r["id"] for r in eval_rows}
    print(f"Eval split: {len(eval_rows)} questions")
    from collections import Counter
    print(f"  by category: {dict(Counter(r['category'] for r in eval_rows))}")

    all_rollouts = load_full_rollouts()
    eval_lookup = build_eval_lookup(eval_qids, all_rollouts)
    print(f"Eval lookup: {len(eval_lookup)} (qid, ord) pairs")

    # Classify the eval Qs with our 5-class ModernBERT
    pred_categories = classify_questions(eval_rows, args.classifier, "cpu")

    # Construct binary labels (two methods)
    binary_5c: dict[str, str] = {qid: project_5class_to_binary(pred_categories[qid]) for qid in pred_categories}

    # Per-question oracle reasoning labels (using actual rollout outcomes)
    binary_oracle: dict[str, str] = {}
    for r in eval_rows:
        is_reasoning = question_needs_reasoning_oracle(eval_lookup, r["id"])
        binary_oracle[r["id"]] = "reasoning" if is_reasoning else "not"

    # Counts
    n_5c_reasoning = sum(1 for v in binary_5c.values() if v == "reasoning")
    n_oracle_reasoning = sum(1 for v in binary_oracle.values() if v == "reasoning")
    print(f"\n5-class projection: {n_5c_reasoning}/{len(binary_5c)} flagged as reasoning")
    print(f"Per-Q oracle:        {n_oracle_reasoning}/{len(binary_oracle)} flagged as reasoning")
    agree_5c_oracle = sum(1 for q in binary_oracle if binary_oracle[q] == binary_5c.get(q))
    print(f"5-class vs oracle agreement: {agree_5c_oracle}/{len(binary_oracle)} = "
          f"{agree_5c_oracle/len(binary_oracle):.1%}")

    # Build evaluation function (mirror of the one in eval_classifier_router)
    def evaluate(eval_rows, pick_fn, alpha):
        rewards = []; n_correct = 0; total_cost = 0.0
        pick_dist: dict[int, int] = defaultdict(int)
        for r in eval_rows:
            qid = r["id"]
            ord_ = pick_fn(qid, alpha)
            if ord_ is None:
                continue
            outcome = eval_lookup.get((qid, ord_))
            if outcome is None:
                continue
            rew = cost_aware_reward(outcome["is_correct"], outcome["cost_usd"], alpha)
            rewards.append(rew)
            n_correct += int(outcome["is_correct"])
            total_cost += outcome["cost_usd"]
            pick_dist[ord_] += 1
        n = len(rewards)
        return {
            "mean_reward": sum(rewards) / max(n, 1),
            "accuracy": n_correct / max(n, 1),
            "avg_cost_usd": total_cost / max(n, 1),
            "n": n,
            "pick_distribution": dict(pick_dist),
        }

    # The two reasoning-gate variants:
    #   (A) "vsr_5c_opus_gemma": 5-class projection, opus on reasoning, gemma on rest
    #   (B) "vsr_oracle_opus_gemma": oracle binary label, opus on reasoning, gemma on rest
    #   (C) "vsr_5c_opus_qwc":   5-class, opus on reasoning, qwen-coder on rest
    #   (D) "vsr_oracle_opus_qwc": oracle, opus on reasoning, qwen-coder on rest

    pick_5c_opus_gemma = make_reasoning_gate_pick(lambda q: binary_5c.get(q, "not"), 8, 0)
    pick_oracle_opus_gemma = make_reasoning_gate_pick(lambda q: binary_oracle.get(q, "not"), 8, 0)
    pick_5c_opus_qwc = make_reasoning_gate_pick(lambda q: binary_5c.get(q, "not"), 8, 3)
    pick_oracle_opus_qwc = make_reasoning_gate_pick(lambda q: binary_oracle.get(q, "not"), 8, 3)

    # Compare against best-static baselines from the prior eval
    def always_qwc_pick(qid, alpha): return 3
    def always_opus_pick(qid, alpha): return 8
    def always_gemma_pick(qid, alpha): return 0

    policies = {
        "vsr_5c_opus_gemma":     pick_5c_opus_gemma,
        "vsr_5c_opus_qwen-coder":pick_5c_opus_qwc,
        "vsr_oracle_opus_gemma": pick_oracle_opus_gemma,
        "vsr_oracle_opus_qwen-coder": pick_oracle_opus_qwc,
        "always_qwen_coder_480b":always_qwc_pick,
        "always_opus_4_7":       always_opus_pick,
        "always_gemma_3_27b":    always_gemma_pick,
    }

    out = {"n_eval": len(eval_rows),
           "binary_labels_summary": {
               "n_5c_reasoning": n_5c_reasoning,
               "n_oracle_reasoning": n_oracle_reasoning,
               "5c_vs_oracle_agreement": agree_5c_oracle / len(binary_oracle),
           },
           "alphas": {}}

    print(f"\n{'alpha':>5} {'policy':<28} {'reward':>9} {'acc':>7} {'$/q':>10} {'gap_vs_qwc':>12}")
    print("-" * 90)

    for alpha in [0.5, 1.0, 1.7, 3.0]:
        out["alphas"][alpha] = {}
        qwc_reward = None
        for name, pick_fn in policies.items():
            res = evaluate(eval_rows, pick_fn, alpha)
            out["alphas"][alpha][name] = {
                "mean_reward": round(res["mean_reward"], 4),
                "accuracy": round(res["accuracy"], 4),
                "avg_cost_usd": round(res["avg_cost_usd"], 6),
                "pick_distribution": res["pick_distribution"],
            }
            if name == "always_qwen_coder_480b":
                qwc_reward = res["mean_reward"]
            gap = ""
            if qwc_reward is not None and name != "always_qwen_coder_480b":
                gap = f"{(res['mean_reward'] - qwc_reward):+.3f}"
            print(f"{alpha:>5.1f} {name:<28} {res['mean_reward']:>+9.3f} "
                  f"{res['accuracy']:>6.1%} ${res['avg_cost_usd']:>9.5f} {gap:>12}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
