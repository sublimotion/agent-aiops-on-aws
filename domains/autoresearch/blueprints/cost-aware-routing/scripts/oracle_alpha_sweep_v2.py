"""Compute oracle / best-static / per-category-classifier for the
augmented 480-question dataset across 5 categories.

Categories: math, code, factual, reasoning, open-domain.

Outputs:
  results/runs/oracle_alpha_sweep_v2.json
"""
from __future__ import annotations

import json
from collections import defaultdict


_MIN_REF = 0.00035
_MAX_REF = 0.02100


def cn(c: float) -> float:
    return max(0.0, min(1.0, (c - _MIN_REF) / (_MAX_REF - _MIN_REF)))


def reward(is_correct: bool, cost_usd: float, alpha: float, floor: float = -1.0) -> float:
    if not is_correct:
        return 0.0
    return max(1.0 - alpha * cn(cost_usd), floor)


NAMES = ["gemma", "gpt-oss", "qwen3-32b", "qwen-coder", "mistral",
         "deepseek", "haiku", "sonnet", "opus"]


def load_all_rollouts() -> list[dict]:
    """Combine existing math/aime/wildchat baselines with the new augmented one."""
    rollouts: list[dict] = []

    # Existing 130-q baselines: tag with category from source
    src_to_cat = {"math500": "math", "aime25": "math", "wildchat": "open-domain"}
    for path, cor_key in [
        ("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_math500.json", "is_correct"),
        ("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json", "is_correct"),
        ("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json", "acceptable"),
    ]:
        d = json.load(open(path))
        for r in d["rollouts"]:
            qid = r.get("id") or r["question"][:60]
            # Map source name from path
            source = "math500" if "math500" in path else ("aime25" if "aime25" in path else "wildchat")
            rollouts.append({
                "ord": r["ord"],
                "qid": qid,
                "source": source,
                "category": src_to_cat[source],
                "is_correct": bool(r[cor_key]),
                "cost_usd": r["cost_usd"],
            })

    # Augmented 350-q baseline: already has 'category' field
    aug = json.load(open("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_augmented.json"))
    for r in aug["rollouts"]:
        rollouts.append({
            "ord": r["ord"],
            "qid": r["id"],
            "source": r["source"],
            "category": r["category"],
            "is_correct": bool(r["is_correct"]),
            "cost_usd": r["cost_usd"],
        })

    return rollouts


def expected_reward_per_rollout(rollouts: list[dict], alpha: float) -> float:
    if not rollouts:
        return 0.0
    return sum(reward(r["is_correct"], r["cost_usd"], alpha) for r in rollouts) / len(rollouts)


def main():
    rollouts = load_all_rollouts()
    n_questions = len(set((r["category"], r["qid"]) for r in rollouts))
    print(f"Total rollouts: {len(rollouts)}, unique questions: {n_questions}")

    # Group by (category, ord)
    by_cat_ord: dict[tuple, list] = defaultdict(list)
    for r in rollouts:
        by_cat_ord[(r["category"], r["ord"])].append(r)

    cats = sorted({r["category"] for r in rollouts})
    print(f"Categories: {cats}")
    n_per_cat = {c: len(set(r["qid"] for r in rollouts if r["category"] == c)) for c in cats}
    print(f"Questions per category: {n_per_cat}")

    out: dict = {"categories": cats, "n_per_category": n_per_cat}

    print()
    print(f"{'alpha':>5}  per-category oracle picks                                                                  oracle  best-static  gap")
    print("-" * 160)

    for alpha in [0.3, 0.5, 1.0, 1.7, 3.0, 5.0]:
        # Per-(category, ord) E[r]
        per_cat_er: dict[str, dict[int, float]] = {}
        for c in cats:
            ers = {}
            for ord_ in range(9):
                rs = by_cat_ord.get((c, ord_), [])
                ers[ord_] = expected_reward_per_rollout(rs, alpha) if rs else 0.0
            per_cat_er[c] = ers

        # Per-category best ord
        best_per_cat = {c: max(per_cat_er[c], key=per_cat_er[c].get) for c in cats}
        best_e_r_per_cat = {c: per_cat_er[c][best_per_cat[c]] for c in cats}

        # Oracle E[r] = sum over questions of (best_ord on that category's E[r])
        # Weighted by question count
        total = sum(n_per_cat.values())
        oracle_er = sum(best_e_r_per_cat[c] * n_per_cat[c] for c in cats) / total

        # Best-static: pick one ord that maximizes weighted E[r] across all categories
        best_static_er = -10.0
        best_static_ord = None
        for ord_ in range(9):
            er_static = sum(per_cat_er[c][ord_] * n_per_cat[c] for c in cats) / total
            if er_static > best_static_er:
                best_static_er = er_static
                best_static_ord = ord_

        gap = oracle_er - best_static_er

        picks_str = " ".join(f"{c[:5]}=ord_{best_per_cat[c]}({NAMES[best_per_cat[c]]})" for c in cats)
        print(f"{alpha:>5.1f}  {picks_str:90s}  {oracle_er:+.3f}  ord_{best_static_ord}({NAMES[best_static_ord]}):{best_static_er:+.3f}  {gap:+.3f}")

        out[str(alpha)] = {
            "per_category_e_r": {c: {str(o): round(er, 4) for o, er in ers.items()} for c, ers in per_cat_er.items()},
            "per_category_best": {c: {"ord": best_per_cat[c], "name": NAMES[best_per_cat[c]],
                                      "e_r": round(best_e_r_per_cat[c], 4)} for c in cats},
            "oracle_e_r": round(oracle_er, 4),
            "best_static_ord": best_static_ord,
            "best_static_name": NAMES[best_static_ord],
            "best_static_e_r": round(best_static_er, 4),
            "oracle_gap": round(gap, 4),
        }

    out_path = "domains/autoresearch/blueprints/cost-aware-routing/results/runs/oracle_alpha_sweep_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    # Print per-(category, alpha) full table for the most interesting alphas
    print()
    print("Full per-(category, ord) E[r] table at alpha=1.0:")
    print(f"  {'cat':<12} " + " ".join(f"ord_{i}".rjust(7) for i in range(9)))
    for c in cats:
        ers = out["1.0"]["per_category_e_r"][c]
        cells = []
        best_o = out["1.0"]["per_category_best"][c]["ord"]
        for i in range(9):
            v = ers.get(str(i), 0)
            if i == best_o:
                cells.append(f"[{v:+5.2f}]")
            else:
                cells.append(f" {v:+5.2f} ")
        print(f"  {c:<12} " + " ".join(cells))


if __name__ == "__main__":
    main()
