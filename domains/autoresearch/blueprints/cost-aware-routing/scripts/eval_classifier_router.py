"""Evaluate the classifier-router on the held-out eval split.

Pipeline:
  question -> ModernBERT classifier -> category prediction
           -> RouterPolicy.pick(category, alpha) -> worker_id
           -> look up that worker's outcome on this question (from baselines)
           -> compute reward

Compares classifier-router against:
  - best static (always-Qwen-Coder-480B)
  - per-question oracle (best worker per category, given true category)
  - random uniform routing

Output: results/runs/classifier_router_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# OpenMP duplicate-init workaround
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, "domains/autoresearch/blueprints/cost-aware-routing/scripts")
from router_policy import QualityTable, RouterPolicy, cost_aware_reward

CATEGORIES = ["math", "code", "factual", "reasoning", "open-domain"]


def load_eval_split(data_path: str, seed: int = 17, eval_frac: float = 0.20) -> list[dict]:
    """Reproduce the same train/eval split the classifier used.

    The classifier saw the train rows; eval rows are unseen.
    """
    rows = []
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_eval = int(len(rows) * eval_frac)
    return rows[:n_eval]


def load_full_rollouts(eval_questions_by_text: dict[str, str] | None = None) -> list[dict]:
    """All worker rollouts: existing 130-q baselines + augmented 350-q.

    Existing math500/aime25 baselines use raw question text as id (or no id);
    the augmented file uses hashed `math_<sha>` ids. We bridge by mapping
    the existing rollouts to the augmented hashed ids when the question
    text matches.
    """
    import hashlib
    def short_id(category: str, text: str) -> str:
        return f"{category}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"

    rollouts: list[dict] = []
    src_to_cat = {"math500": "math", "aime25": "math", "wildchat": "open-domain"}
    for path, cor_key in [
        ("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_math500.json", "is_correct"),
        ("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json", "is_correct"),
        ("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json", "acceptable"),
    ]:
        d = json.load(open(path))
        source = "math500" if "math500" in path else ("aime25" if "aime25" in path else "wildchat")
        category = src_to_cat[source]
        for r in d["rollouts"]:
            text = r["question"]
            # Use the same hashed id scheme as the augmented assembly,
            # so existing rollouts and augmented ids share a key.
            qid = short_id(category, text)
            rollouts.append({
                "ord": r["ord"],
                "qid": qid,
                "source": source,
                "category": category,
                "is_correct": bool(r[cor_key]),
                "cost_usd": r["cost_usd"],
                "question_text": text,
            })

    aug = json.load(open("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_augmented.json"))
    for r in aug["rollouts"]:
        rollouts.append({
            "ord": r["ord"],
            "qid": r["id"],
            "source": r["source"],
            "category": r["category"],
            "is_correct": bool(r["is_correct"]),
            "cost_usd": r["cost_usd"],
            "question_text": r.get("question", ""),
        })
    return rollouts


def build_train_quality_table(eval_qids: set[str], all_rollouts: list[dict]) -> QualityTable:
    """Build the policy's quality table EXCLUDING eval questions.

    Critical for fair eval: the policy can't use eval data to fit the table.
    """
    table = QualityTable()
    for r in all_rollouts:
        if r["qid"] in eval_qids:
            continue
        table.add_observation(r["category"], r["ord"], r["is_correct"], r["cost_usd"])
    return table


def build_eval_lookup(eval_qids: set[str], all_rollouts: list[dict]) -> dict[tuple[str, int], dict]:
    """Per-question per-worker lookup of (is_correct, cost_usd) for eval Qs."""
    out: dict = {}
    for r in all_rollouts:
        if r["qid"] not in eval_qids:
            continue
        out[(r["qid"], r["ord"])] = {"is_correct": r["is_correct"], "cost_usd": r["cost_usd"]}
    return out


def classify_questions(eval_rows: list[dict], classifier_dir: str, device_str: str = "cpu") -> dict[str, str]:
    """Returns qid -> predicted category."""
    device = torch.device(device_str)
    print(f"Loading classifier from {classifier_dir} on {device}...")
    tok = AutoTokenizer.from_pretrained(classifier_dir)
    model = AutoModelForSequenceClassification.from_pretrained(classifier_dir).to(device)
    model.eval()

    out: dict[str, str] = {}
    with torch.no_grad():
        for r in eval_rows:
            enc = tok(
                r["question"], truncation=True, max_length=512,
                padding="max_length", return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            pred_idx = int(logits.argmax(-1).item())
            out[r["id"]] = CATEGORIES[pred_idx]
    return out


def evaluate(eval_rows: list[dict], pred_categories: dict[str, str],
             eval_lookup: dict[tuple, dict],
             policy_pick_fn, alpha: float) -> dict:
    """Evaluate a routing policy on the eval split. policy_pick_fn(qid, alpha) -> worker_id."""
    rewards = []
    n_correct = 0
    total_cost = 0.0
    pick_dist: dict[int, int] = defaultdict(int)
    by_category: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_correct": 0, "reward_sum": 0.0, "cost_sum": 0.0})
    routing_correctness: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_correct": 0})

    for r in eval_rows:
        qid = r["id"]
        true_cat = r["category"]
        pred_cat = pred_categories.get(qid, true_cat)  # fall back to true if not classified
        ord_ = policy_pick_fn(qid, alpha)
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
        bc = by_category[true_cat]
        bc["n"] += 1
        bc["n_correct"] += int(outcome["is_correct"])
        bc["reward_sum"] += rew
        bc["cost_sum"] += outcome["cost_usd"]
        rc = routing_correctness[true_cat]
        rc["n"] += 1
        rc["n_correct"] += int(pred_cat == true_cat)

    n = len(rewards)
    return {
        "mean_reward": sum(rewards) / max(n, 1),
        "accuracy": n_correct / max(n, 1),
        "avg_cost_usd": total_cost / max(n, 1),
        "n": n,
        "pick_distribution": dict(pick_dist),
        "by_category": {c: {**v,
                            "mean_reward": v["reward_sum"] / max(v["n"], 1),
                            "accuracy": v["n_correct"] / max(v["n"], 1),
                            "avg_cost": v["cost_sum"] / max(v["n"], 1)}
                        for c, v in by_category.items()},
        "classification_accuracy": {c: round(rc["n_correct"] / max(rc["n"], 1), 4)
                                     for c, rc in routing_correctness.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="domains/autoresearch/blueprints/cost-aware-routing/data/augmented_baseline_500q.jsonl")
    ap.add_argument("--classifier", default="domains/autoresearch/blueprints/cost-aware-routing/artifacts/classifier")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", default="domains/autoresearch/blueprints/cost-aware-routing/results/runs/classifier_router_eval.json")
    args = ap.parse_args()

    eval_rows = load_eval_split(args.data, seed=args.seed)
    eval_qids = {r["id"] for r in eval_rows}
    print(f"Eval split: {len(eval_rows)} questions")
    from collections import Counter
    print(f"  by category: {dict(Counter(r['category'] for r in eval_rows))}")

    all_rollouts = load_full_rollouts()
    print(f"All rollouts: {len(all_rollouts)}")

    # Build quality table EXCLUDING eval Qs (fair eval)
    table = build_train_quality_table(eval_qids, all_rollouts)
    print(f"Quality table: {len(table.cells)} categories")

    eval_lookup = build_eval_lookup(eval_qids, all_rollouts)
    print(f"Eval lookup: {len(eval_lookup)} (question, worker) pairs (should be ~{len(eval_rows)*9})")

    # Classify the eval questions
    pred_categories = classify_questions(eval_rows, args.classifier, args.device)
    cls_correct = sum(1 for r in eval_rows if pred_categories.get(r["id"]) == r["category"])
    print(f"Classifier accuracy on eval: {cls_correct}/{len(eval_rows)} = {cls_correct/len(eval_rows):.1%}")

    # Build the routing policies
    policy = RouterPolicy(table, fallback_category="open-domain")

    # Pick functions
    def classifier_pick(qid: str, alpha: float) -> int:
        cat = pred_categories.get(qid, "open-domain")
        return policy.pick(cat, alpha)

    def oracle_per_q_pick(qid: str, alpha: float) -> int:
        # Per-question oracle: pick the worker that maximizes ACTUAL reward on this question
        best_w = None
        best_r = -1e9
        for w in range(9):
            o = eval_lookup.get((qid, w))
            if o is None:
                continue
            r = cost_aware_reward(o["is_correct"], o["cost_usd"], alpha)
            if r > best_r:
                best_r = r
                best_w = w
        return best_w if best_w is not None else 0

    def oracle_per_cat_pick(qid: str, alpha: float) -> int:
        # Use TRUE category, then RouterPolicy on the table
        true_cat = next((r["category"] for r in eval_rows if r["id"] == qid), "open-domain")
        return policy.pick(true_cat, alpha)

    def always_qwen_coder_pick(qid: str, alpha: float) -> int:
        return 3  # ord_3 = Qwen-Coder-480B

    def always_opus_pick(qid: str, alpha: float) -> int:
        return 8

    def always_gemma_pick(qid: str, alpha: float) -> int:
        return 0

    rng = random.Random(args.seed)
    def random_pick(qid: str, alpha: float) -> int:
        return rng.randrange(9)

    policies = {
        "classifier_router": classifier_pick,
        "oracle_per_q": oracle_per_q_pick,
        "oracle_per_category": oracle_per_cat_pick,
        "always_qwen_coder_480b": always_qwen_coder_pick,
        "always_opus_4_7": always_opus_pick,
        "always_gemma_3_27b": always_gemma_pick,
        "random_uniform": random_pick,
    }

    out = {"n_eval": len(eval_rows), "classifier_accuracy_overall": cls_correct / len(eval_rows),
           "alphas": {}}

    print(f"\n{'alpha':>6} {'policy':<25} {'reward':>9} {'acc':>7} {'$/q':>10} {'gap_vs_static':>14}")
    print("-" * 90)

    for alpha in [0.5, 1.0, 1.7, 3.0]:
        out["alphas"][alpha] = {}
        baseline_reward = None
        for policy_name, pick_fn in policies.items():
            res = evaluate(eval_rows, pred_categories, eval_lookup, pick_fn, alpha)
            out["alphas"][alpha][policy_name] = {
                "mean_reward": round(res["mean_reward"], 4),
                "accuracy": round(res["accuracy"], 4),
                "avg_cost_usd": round(res["avg_cost_usd"], 6),
                "n": res["n"],
                "pick_distribution": res["pick_distribution"],
                "by_category": {c: {"mean_reward": round(v["mean_reward"], 4),
                                    "accuracy": round(v["accuracy"], 4),
                                    "avg_cost": round(v["avg_cost"], 6),
                                    "n": v["n"]}
                                for c, v in res["by_category"].items()},
            }
            if policy_name == "always_qwen_coder_480b":
                baseline_reward = res["mean_reward"]
            gap = ""
            if baseline_reward is not None and policy_name != "always_qwen_coder_480b":
                gap = f"{(res['mean_reward'] - baseline_reward):+.3f}"
            print(f"{alpha:>6.1f} {policy_name:<25} {res['mean_reward']:>+9.3f} "
                  f"{res['accuracy']:>6.1%} ${res['avg_cost_usd']:>9.5f} {gap:>14}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
