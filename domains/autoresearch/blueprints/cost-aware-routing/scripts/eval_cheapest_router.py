"""Evaluate the cheapest-correct-worker classifier as a router.

Pipeline:
  question -> 9-way classifier -> predicted worker_id (direct routing)
           -> look up reward from baseline rollouts

Compare to:
  - always-Qwen-Coder (best static)
  - per-question oracle (the cheapest_correct label itself, applied perfectly)
  - difficulty-router (Phase 2A) and category-router (Phase 1)
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


def classify_workers(eval_rows, classifier_dir):
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
            out[r["id"]] = int(logits.argmax(-1).item())
    return out


def true_cheapest_correct(eval_qids, all_rollouts):
    """For each eval qid: cheapest worker that's correct, else ord_8."""
    by_qid: dict[str, dict[int, bool]] = defaultdict(dict)
    for r in all_rollouts:
        if r["qid"] in eval_qids:
            by_qid[r["qid"]][r["ord"]] = bool(r["is_correct"])
    out = {}
    for qid, by_ord in by_qid.items():
        for w in range(9):
            if by_ord.get(w, False):
                out[qid] = w
                break
        else:
            out[qid] = 8
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="domains/autoresearch/blueprints/cost-aware-routing/data/augmented_baseline_500q.jsonl")
    ap.add_argument("--classifier",
                    default="domains/autoresearch/blueprints/cost-aware-routing/artifacts/cheapest_correct_classifier")
    ap.add_argument("--output", default="domains/autoresearch/blueprints/cost-aware-routing/results/runs/cheapest_router_eval.json")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    eval_rows = load_eval_split(args.data, seed=args.seed)
    eval_qids = {r["id"] for r in eval_rows}
    print(f"Eval: {len(eval_rows)} questions")

    all_rollouts = load_full_rollouts()
    eval_lookup = build_eval_lookup(eval_qids, all_rollouts)

    pred_workers = classify_workers(eval_rows, args.classifier)
    true_workers = true_cheapest_correct(eval_qids, all_rollouts)

    # Classifier accuracy on the cheapest-correct target
    n_match = sum(1 for q in true_workers if pred_workers.get(q) == true_workers[q])
    print(f"Pred vs true cheapest-correct: {n_match}/{len(true_workers)} = {n_match/len(true_workers):.1%}")

    # Per-class breakdown
    from collections import Counter
    pred_dist = Counter(pred_workers.values())
    true_dist = Counter(true_workers.values())
    NAMES = ["gemma","gpt-oss","qwen3-32b","qwen-coder","mistral","deepseek","haiku","sonnet","opus"]
    print(f"Predicted: {dict((NAMES[w], n) for w, n in sorted(pred_dist.items()))}")
    print(f"True:      {dict((NAMES[w], n) for w, n in sorted(true_dist.items()))}")

    policies = {
        "cheapest_classifier": lambda q, a: pred_workers.get(q, 3),
        "cheapest_oracle":     lambda q, a: true_workers.get(q, 3),
        "always_qwen_coder_480b": lambda q, a: 3,
        "always_opus_4_7":     lambda q, a: 8,
        "always_gemma_3_27b":  lambda q, a: 0,
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

    out = {"n_eval": len(eval_rows),
           "classifier_accuracy_on_cheapest": n_match / len(true_workers),
           "alphas": {}}

    print(f"\n{'alpha':>5} {'policy':<25} {'reward':>9} {'acc':>7} {'$/q':>10} {'gap_qwc':>9}")
    print("-" * 80)
    for alpha in [0.5, 1.0, 1.7, 3.0]:
        out["alphas"][alpha] = {}
        qwc = None
        for name, pf in policies.items():
            res = evaluate(eval_rows, pf, alpha)
            res_save = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "by_category"}
            res_save["by_category"] = {c: {"mean_reward": round(v["mean_reward"], 4),
                                            "accuracy": round(v["accuracy"], 4), "n": v["n"]}
                                       for c, v in res["by_category"].items()}
            out["alphas"][alpha][name] = res_save
            if name == "always_qwen_coder_480b":
                qwc = res["mean_reward"]
            gap = ""
            if qwc is not None and name != "always_qwen_coder_480b":
                gap = f"{res['mean_reward'] - qwc:+.3f}"
            print(f"{alpha:>5.1f} {name:<25} {res['mean_reward']:>+9.3f} {res['accuracy']:>6.1%} ${res['avg_cost_usd']:>9.5f} {gap:>9}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
