"""Simulate the redesigned classifier-router on existing baseline data.

Uses:
  - Haiku's measured difficulty classification per question (from
    results/preflight/difficulty_classifier_probe.json) — represents the
    "signal extractor" output.
  - Per-(predicted_difficulty, worker) quality table computed from the
    SAME baselines, leave-one-out style — represents the "closed-form
    policy."

For each alpha, the policy picks worker = argmax_w E[r | predicted_difficulty, w, alpha].

This validates that the proposed architecture actually delivers the gap
between best-static (+0.65) and oracle (+0.77). If the closed-form
policy + 68% Haiku classifier hits +0.70 at alpha=1.0, the redesign is
solid. If it falls flat at ~+0.65, the redesign is wishful.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


_MIN_REF = 0.00035
_MAX_REF = 0.02100


def cn(c: float) -> float:
    return max(0.0, min(1.0, (c - _MIN_REF) / (_MAX_REF - _MIN_REF)))


def reward(is_correct: bool, cost_usd: float, alpha: float, floor: float = -1.0) -> float:
    if not is_correct:
        return 0.0
    return max(1.0 - alpha * cn(cost_usd), floor)


def load_data():
    """Returns: list of {qid, source, true_difficulty, haiku_predicted_difficulty,
    by_ord: {ord: {is_correct, cost_usd}}}.
    """
    sources = [
        ("math500", "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_math500.json", "is_correct"),
        ("aime25", "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json", "is_correct"),
        ("wildchat", "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json", "acceptable"),
    ]
    by_qid: dict[tuple[str, str], dict] = {}
    for src, path, key in sources:
        d = json.load(open(path))
        for r in d["rollouts"]:
            qid = r.get("id") or r.get("question", "")[:60]
            entry = by_qid.setdefault((src, qid), {
                "source": src, "qid": qid,
                "true_difficulty": "hard" if src == "aime25" else "easy",
                "by_ord": {},
            })
            entry["by_ord"][r["ord"]] = {
                "is_correct": bool(r[key]),
                "cost_usd": r["cost_usd"],
            }

    # Attach Haiku's predicted difficulty
    classifier_probe = json.load(open(
        "domains/autoresearch/blueprints/cost-aware-routing/results/preflight/difficulty_classifier_probe.json"
    ))
    haiku_pred = {(r["source"], r["qid"]): r["verdict"] for r in classifier_probe["rows"]}
    for key, entry in by_qid.items():
        entry["haiku_predicted"] = haiku_pred.get(key)

    return list(by_qid.values())


def build_quality_table(data: list[dict], difficulty_label: str) -> dict[str, dict[int, dict]]:
    """For each (difficulty_label_value, worker), keep per-rollout
    (is_correct, cost) tuples + aggregate stats. The per-rollout list is
    needed to compute E[r] correctly: mean(reward(c_i)) is NOT the same
    as reward(mean(c)) under our floored cost-aware reward.
    """
    accumulator: dict[tuple, dict] = defaultdict(lambda: {"correct": 0, "n": 0, "cost_sum": 0.0, "rollouts": []})
    for q in data:
        diff = q.get(difficulty_label)
        if diff is None:
            continue
        for ord_, info in q["by_ord"].items():
            key = (diff, ord_)
            accumulator[key]["correct"] += int(info["is_correct"])
            accumulator[key]["n"] += 1
            accumulator[key]["cost_sum"] += info["cost_usd"]
            accumulator[key]["rollouts"].append((info["is_correct"], info["cost_usd"]))

    table: dict[str, dict[int, dict]] = {}
    for (diff, ord_), s in accumulator.items():
        table.setdefault(diff, {})[ord_] = {
            "p_correct": s["correct"] / max(s["n"], 1),
            "avg_cost": s["cost_sum"] / max(s["n"], 1),
            "rollouts": s["rollouts"],
        }
    return table


def policy_pick(table: dict[int, dict], alpha: float) -> int:
    """argmax_w E[r | worker w] = mean over per-rollout reward(c_i)."""
    best_ord = None
    best_er = -1e9
    for ord_, cell in table.items():
        rollouts = cell["rollouts"]
        if not rollouts:
            continue
        er = sum(reward(c, cost, alpha) for c, cost in rollouts) / len(rollouts)
        if er > best_er:
            best_er = er
            best_ord = ord_
    return best_ord


def evaluate(data: list[dict], policy_fn, alpha: float) -> dict:
    """For each question, route to worker = policy_fn(question, alpha), look up the
    actual reward from the question's by_ord table."""
    rewards = []
    n_correct = 0
    total_cost = 0.0
    pick_dist: dict[int, int] = defaultdict(int)
    by_source: dict[str, dict] = defaultdict(lambda: {"n": 0, "n_correct": 0, "reward_sum": 0.0, "cost_sum": 0.0})

    for q in data:
        ord_ = policy_fn(q, alpha)
        if ord_ is None:
            continue
        cell = q["by_ord"].get(ord_)
        if cell is None:
            continue
        r = reward(cell["is_correct"], cell["cost_usd"], alpha)
        rewards.append(r)
        n_correct += int(cell["is_correct"])
        total_cost += cell["cost_usd"]
        pick_dist[ord_] += 1
        s = by_source[q["source"]]
        s["n"] += 1
        s["n_correct"] += int(cell["is_correct"])
        s["reward_sum"] += r
        s["cost_sum"] += cell["cost_usd"]

    return {
        "mean_reward": sum(rewards) / max(len(rewards), 1),
        "accuracy": n_correct / max(len(rewards), 1),
        "avg_cost_usd": total_cost / max(len(rewards), 1),
        "n": len(rewards),
        "pick_dist": dict(pick_dist),
        "by_source": {s: {**v, "mean_reward": v["reward_sum"] / max(v["n"], 1),
                          "accuracy": v["n_correct"] / max(v["n"], 1),
                          "avg_cost": v["cost_sum"] / max(v["n"], 1)}
                      for s, v in by_source.items()},
    }


def main():
    data = load_data()
    print(f"Loaded {len(data)} questions")
    haiku_count = sum(1 for q in data if q["haiku_predicted"])
    print(f"Haiku predictions available: {haiku_count}/{len(data)}")

    # Build three quality tables:
    #   - TRUE difficulty (oracle 2-class)
    #   - HAIKU predicted difficulty (real 2-class classifier)
    #   - SOURCE (oracle 3-class — math500/aime25/wildchat)
    table_true = build_quality_table(data, "true_difficulty")
    table_haiku = build_quality_table(data, "haiku_predicted")
    table_source = build_quality_table(data, "source")

    print()
    print("Quality table (TRUE difficulty):")
    for diff, t in table_true.items():
        for ord_ in sorted(t):
            c = t[ord_]
            print(f"  {diff:5s} ord_{ord_}  p={c['p_correct']:.2f}  cost=${c['avg_cost']:.5f}")

    print()
    print("Quality table (HAIKU predicted):")
    for diff, t in table_haiku.items():
        for ord_ in sorted(t):
            c = t[ord_]
            print(f"  {diff:5s} ord_{ord_}  p={c['p_correct']:.2f}  cost=${c['avg_cost']:.5f}")

    print()
    print("=" * 100)
    print(f"{'alpha':>6} {'src_3class':>12} {'diff_2class':>13} {'haiku_2class':>13} {'static':>9} {'q_oracle':>10}")
    print("=" * 100)

    oracle_data = json.load(open(
        "domains/autoresearch/blueprints/cost-aware-routing/results/runs/oracle_alpha_sweep.json"
    ))

    rows_per_alpha = []
    for alpha in [0.5, 1.0, 1.7, 3.0]:
        # Oracle classifier (2-class) = use TRUE difficulty
        def diff_2class_pick(q, a):
            return policy_pick(table_true[q["true_difficulty"]], a)
        # Haiku classifier (2-class)
        def haiku_pick(q, a):
            pred = q.get("haiku_predicted") or "easy"
            tbl = table_haiku.get(pred)
            return policy_pick(tbl, a) if tbl else None
        # Source classifier (3-class oracle)
        def src_3class_pick(q, a):
            return policy_pick(table_source[q["source"]], a)

        diff_eval = evaluate(data, diff_2class_pick, alpha)
        haiku_eval = evaluate(data, haiku_pick, alpha)
        src_eval = evaluate(data, src_3class_pick, alpha)
        static = oracle_data[str(alpha)]["best_static_e_r"]
        oracle_full = oracle_data[str(alpha)]["oracle_e_r"]

        print(f"{alpha:>6.1f} "
              f"{src_eval['mean_reward']:>+12.3f} "
              f"{diff_eval['mean_reward']:>+13.3f} "
              f"{haiku_eval['mean_reward']:>+13.3f} "
              f"{static:>+9.3f} "
              f"{oracle_full:>+10.3f}")

        rows_per_alpha.append({
            "alpha": alpha,
            "src_3class_mean_reward": round(src_eval["mean_reward"], 4),
            "diff_2class_mean_reward": round(diff_eval["mean_reward"], 4),
            "haiku_2class_mean_reward": round(haiku_eval["mean_reward"], 4),
            "best_static": static,
            "oracle_full_per_q": oracle_full,
            "src_3class_pick_dist": src_eval["pick_dist"],
            "diff_2class_pick_dist": diff_eval["pick_dist"],
            "haiku_2class_pick_dist": haiku_eval["pick_dist"],
        })

    out = {
        "table_true_difficulty": {d: {str(o): t for o, t in tab.items()} for d, tab in table_true.items()},
        "table_haiku_predicted": {d: {str(o): t for o, t in tab.items()} for d, tab in table_haiku.items()},
        "results_by_alpha": rows_per_alpha,
    }
    out_path = "domains/autoresearch/blueprints/cost-aware-routing/results/runs/sim_classifier_router.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    # Compare classifiers: gap captured vs oracle-full
    print()
    print("=== GAP-CAPTURE % (vs per-question oracle) ===")
    print(f"{'alpha':>6} {'src_3':>7} {'diff_2':>7} {'haiku':>7}  oracle_gap")
    for row in rows_per_alpha:
        oracle_gap = row["oracle_full_per_q"] - row["best_static"]
        if oracle_gap <= 0:
            continue
        s3 = (row["src_3class_mean_reward"] - row["best_static"]) / oracle_gap * 100
        d2 = (row["diff_2class_mean_reward"] - row["best_static"]) / oracle_gap * 100
        h2 = (row["haiku_2class_mean_reward"] - row["best_static"]) / oracle_gap * 100
        print(f"{row['alpha']:>6.1f} {s3:>6.0f}% {d2:>6.0f}% {h2:>6.0f}%   "
              f"{oracle_gap:+.3f}")


if __name__ == "__main__":
    main()
