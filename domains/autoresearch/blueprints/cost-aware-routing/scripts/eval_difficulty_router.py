"""Evaluate the per-question difficulty classifier as a routing signal.

Pipeline:
  question -> difficulty classifier -> {easy, hard}
           -> difficulty-gate policy: hard -> Opus, easy -> Qwen-Coder (or Gemma)

Compares against:
  - always-Qwen-Coder (best static)
  - per-question oracle (true difficulty from baselines)
  - vSR-style 5-class projection reasoning-gate
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

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, "domains/autoresearch/blueprints/cost-aware-routing/scripts")
from router_policy import cost_aware_reward
from eval_classifier_router import (
    load_eval_split,
    load_full_rollouts,
    build_eval_lookup,
)


def classify_difficulty(eval_rows, classifier_dir, labels):
    device = torch.device("cpu")
    print(f"Loading {classifier_dir}...")
    tok = AutoTokenizer.from_pretrained(classifier_dir)
    model = AutoModelForSequenceClassification.from_pretrained(classifier_dir).to(device)
    model.eval()
    out = {}
    with torch.no_grad():
        for r in eval_rows:
            enc = tok(r["question"], truncation=True, max_length=512,
                      padding="max_length", return_tensors="pt").to(device)
            logits = model(**enc).logits
            out[r["id"]] = labels[int(logits.argmax(-1).item())]
    return out


def true_difficulty_from_rollouts(eval_qids, all_rollouts):
    """ground-truth difficulty: hard if n_correct < 6 across 9 workers."""
    by_qid_correct: dict[str, int] = defaultdict(int)
    for r in all_rollouts:
        if r["qid"] in eval_qids:
            by_qid_correct[r["qid"]] += int(r["is_correct"])
    return {qid: ("hard" if n < 6 else "easy") for qid, n in by_qid_correct.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="domains/autoresearch/blueprints/cost-aware-routing/data/augmented_baseline_500q.jsonl")
    ap.add_argument("--difficulty-classifier",
                    default="domains/autoresearch/blueprints/cost-aware-routing/artifacts/difficulty_classifier_binary")
    ap.add_argument("--output", default="domains/autoresearch/blueprints/cost-aware-routing/results/runs/difficulty_router_eval.json")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    eval_rows = load_eval_split(args.data, seed=args.seed)
    eval_qids = {r["id"] for r in eval_rows}
    print(f"Eval: {len(eval_rows)} questions")

    all_rollouts = load_full_rollouts()
    eval_lookup = build_eval_lookup(eval_qids, all_rollouts)

    pred_difficulty = classify_difficulty(eval_rows, args.difficulty_classifier, ["easy", "hard"])
    true_difficulty = true_difficulty_from_rollouts(eval_qids, all_rollouts)

    n_pred_hard = sum(1 for v in pred_difficulty.values() if v == "hard")
    n_true_hard = sum(1 for v in true_difficulty.values() if v == "hard")
    agree = sum(1 for q in true_difficulty if pred_difficulty.get(q) == true_difficulty[q])
    print(f"Predicted hard: {n_pred_hard}/{len(pred_difficulty)}")
    print(f"True hard:      {n_true_hard}/{len(true_difficulty)}")
    print(f"Pred vs true agreement: {agree}/{len(true_difficulty)} = {agree/len(true_difficulty):.1%}")

    def diff_router_pick(label_fn, hard_ord, easy_ord):
        def pick(qid, alpha): return hard_ord if label_fn(qid) == "hard" else easy_ord
        return pick

    policies = {
        # Trained difficulty classifier -> opus on hard, qwen-coder on easy
        "diff_classifier_opus_qwc": diff_router_pick(lambda q: pred_difficulty.get(q, "easy"), 8, 3),
        "diff_classifier_opus_gemma": diff_router_pick(lambda q: pred_difficulty.get(q, "easy"), 8, 0),
        "diff_classifier_sonnet_qwc": diff_router_pick(lambda q: pred_difficulty.get(q, "easy"), 7, 3),
        # Oracle difficulty
        "diff_oracle_opus_qwc": diff_router_pick(lambda q: true_difficulty.get(q, "easy"), 8, 3),
        "diff_oracle_opus_gemma": diff_router_pick(lambda q: true_difficulty.get(q, "easy"), 8, 0),
        "diff_oracle_sonnet_qwc": diff_router_pick(lambda q: true_difficulty.get(q, "easy"), 7, 3),
        # Static comparisons
        "always_qwen_coder_480b": lambda q, a: 3,
        "always_opus_4_7": lambda q, a: 8,
        "always_gemma_3_27b": lambda q, a: 0,
    }

    def evaluate(eval_rows, pick_fn, alpha):
        rewards = []; n_correct = 0; total_cost = 0.0
        pick_dist: dict[int, int] = defaultdict(int)
        per_cat: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_correct": 0, "reward_sum": 0.0})
        for r in eval_rows:
            ord_ = pick_fn(r["id"], alpha)
            outcome = eval_lookup.get((r["id"], ord_))
            if outcome is None:
                continue
            rew = cost_aware_reward(outcome["is_correct"], outcome["cost_usd"], alpha)
            rewards.append(rew)
            n_correct += int(outcome["is_correct"])
            total_cost += outcome["cost_usd"]
            pick_dist[ord_] += 1
            pc = per_cat[r["category"]]
            pc["n"] += 1
            pc["n_correct"] += int(outcome["is_correct"])
            pc["reward_sum"] += rew
        n = len(rewards)
        return {
            "mean_reward": sum(rewards) / max(n, 1),
            "accuracy": n_correct / max(n, 1),
            "avg_cost_usd": total_cost / max(n, 1),
            "n": n,
            "pick_distribution": dict(pick_dist),
            "by_category": {c: {"mean_reward": v["reward_sum"] / max(v["n"], 1),
                                "accuracy": v["n_correct"] / max(v["n"], 1), "n": v["n"]}
                            for c, v in per_cat.items()},
        }

    out = {
        "n_eval": len(eval_rows),
        "difficulty_classifier_accuracy": agree / len(true_difficulty),
        "n_pred_hard": n_pred_hard,
        "n_true_hard": n_true_hard,
        "alphas": {},
    }
    print(f"\n{'alpha':>5} {'policy':<28} {'reward':>9} {'acc':>7} {'$/q':>10} {'gap_qwc':>9}")
    print("-" * 80)
    for alpha in [0.5, 1.0, 1.7, 3.0]:
        out["alphas"][alpha] = {}
        qwc = None
        for name, pf in policies.items():
            res = evaluate(eval_rows, pf, alpha)
            out["alphas"][alpha][name] = {k: v for k, v in res.items() if k != "by_category"}
            out["alphas"][alpha][name]["by_category"] = {c: {"mean_reward": round(v["mean_reward"], 4),
                                                              "accuracy": round(v["accuracy"], 4), "n": v["n"]}
                                                          for c, v in res["by_category"].items()}
            for k in ("mean_reward", "accuracy", "avg_cost_usd"):
                out["alphas"][alpha][name][k] = round(out["alphas"][alpha][name][k], 4)
            if name == "always_qwen_coder_480b":
                qwc = res["mean_reward"]
            gap = ""
            if qwc is not None and name != "always_qwen_coder_480b":
                gap = f"{res['mean_reward'] - qwc:+.3f}"
            print(f"{alpha:>5.1f} {name:<28} {res['mean_reward']:>+9.3f} {res['accuracy']:>6.1%} ${res['avg_cost_usd']:>9.5f} {gap:>9}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
