"""
Oracle router simulator — sanity-check the cross-dataset Pareto target.

Uses the actual rollout records (one per (worker, question) pair) from
the three baseline runs (MATH500, AIME25, WildChat) to compute the
performance of an "oracle" router that knows the dataset tag of each
question and routes to the best worker for that tag.

Compares against the best static policy (Always-X) on the same records.

Output: results/baselines/oracle_router.json — per-strategy
        (accuracy, $/query, $/correct) and a delta table.

This validates that the projected 84.7% / $0.00579 oracle vs 84.0% /
$0.01421 Always-Opus is correct on the actual measured data, not just on
per-worker means.
"""
from __future__ import annotations

import argparse
import json
import pathlib

DEFAULT_MATH = "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_math500.json"
DEFAULT_AIME = "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json"
DEFAULT_WILDCHAT = "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json"


def load_rollouts(path: str, correctness_field: str, source_tag: str) -> list[dict]:
    """Returns list of {ord, question_id, correct, cost, ...} records."""
    raw = json.load(open(path))
    out = []
    for r in raw["rollouts"]:
        out.append({
            "source": source_tag,
            "ord": r["ord"],
            "name": r["name"],
            "question_id": r.get("id") or r.get("question", "")[:60],  # math files use question text as id surrogate
            "correct": bool(r[correctness_field]),
            "cost_usd": r["cost_usd"],
        })
    return out


def group_by_question(records: list[dict]) -> dict[str, dict[int, dict]]:
    """qid -> ord -> record"""
    out: dict[str, dict[int, dict]] = {}
    for r in records:
        out.setdefault(r["question_id"], {})[r["ord"]] = r
    return out


def best_worker_for_source(records: list[dict]) -> tuple[int, float]:
    """Pick the ord with highest correct-rate on these records."""
    by_ord = {}
    for r in records:
        s = by_ord.setdefault(r["ord"], {"n": 0, "n_correct": 0})
        s["n"] += 1
        if r["correct"]:
            s["n_correct"] += 1
    accs = [(ord_, s["n_correct"] / s["n"]) for ord_, s in by_ord.items()]
    accs.sort(key=lambda x: -x[1])
    return accs[0][0], accs[0][1]


# Cost-normalization anchors used by the cost-aware reward:
# min cost = $0.00035 (Gemma reference), max = $0.02100 (Opus reference)
_MIN_REF = 0.00035
_MAX_REF = 0.02100


def cost_normalized(cost_usd: float) -> float:
    z = (cost_usd - _MIN_REF) / (_MAX_REF - _MIN_REF)
    return max(0.0, min(1.0, z))


def best_worker_for_source_at_alpha(records: list[dict], alpha: float) -> tuple[int, float]:
    """Pick the ord with highest *expected reward* at this alpha.
    expected_reward = P(correct) * max(1 - alpha * cost_normalized(avg_cost), -1)
    """
    by_ord: dict[int, dict] = {}
    for r in records:
        s = by_ord.setdefault(r["ord"], {"n": 0, "n_correct": 0, "cost": 0.0})
        s["n"] += 1
        s["cost"] += r["cost_usd"]
        if r["correct"]:
            s["n_correct"] += 1
    rewards = []
    for ord_, s in by_ord.items():
        p_correct = s["n_correct"] / s["n"]
        avg_cost = s["cost"] / s["n"]
        cn = cost_normalized(avg_cost)
        e_reward = p_correct * max(1 - alpha * cn, -1)
        rewards.append((ord_, e_reward))
    rewards.sort(key=lambda x: -x[1])
    return rewards[0][0], rewards[0][1]


def simulate_oracle(grouped: dict[str, dict[int, dict]], best_ord_per_source: dict[str, int]) -> dict:
    """For each question, route to the best worker for its source's task type.
    Returns aggregate accuracy and cost."""
    n = 0
    n_correct = 0
    total_cost = 0.0
    by_source: dict[str, dict] = {}
    for qid, ords in grouped.items():
        # Need to know the source. It's the same for all ord-records of the same qid.
        any_rec = next(iter(ords.values()))
        src = any_rec["source"]
        target_ord = best_ord_per_source[src]
        rec = ords.get(target_ord)
        if not rec:
            continue  # missing record; skip
        n += 1
        n_correct += int(rec["correct"])
        total_cost += rec["cost_usd"]
        s = by_source.setdefault(src, {"n": 0, "n_correct": 0, "cost": 0.0, "ord": target_ord})
        s["n"] += 1
        s["n_correct"] += int(rec["correct"])
        s["cost"] += rec["cost_usd"]
    return {
        "strategy": "oracle",
        "n": n,
        "n_correct": n_correct,
        "accuracy": round(n_correct / max(n, 1), 4),
        "avg_cost_usd": round(total_cost / max(n, 1), 6),
        "per_source": {
            s: {
                "ord": v["ord"],
                "n": v["n"],
                "n_correct": v["n_correct"],
                "accuracy": round(v["n_correct"] / max(v["n"], 1), 4),
                "avg_cost_usd": round(v["cost"] / max(v["n"], 1), 6),
            } for s, v in by_source.items()
        },
    }


def simulate_always_x(grouped: dict[str, dict[int, dict]], target_ord: int) -> dict:
    """Always-X: pick a single ord regardless of source."""
    n = 0
    n_correct = 0
    total_cost = 0.0
    for qid, ords in grouped.items():
        rec = ords.get(target_ord)
        if not rec:
            continue
        n += 1
        n_correct += int(rec["correct"])
        total_cost += rec["cost_usd"]
    return {
        "strategy": f"always_ord_{target_ord}",
        "n": n,
        "n_correct": n_correct,
        "accuracy": round(n_correct / max(n, 1), 4),
        "avg_cost_usd": round(total_cost / max(n, 1), 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math", default=DEFAULT_MATH)
    ap.add_argument("--aime", default=DEFAULT_AIME)
    ap.add_argument("--wildchat", default=DEFAULT_WILDCHAT)
    ap.add_argument(
        "--output",
        default="domains/autoresearch/blueprints/cost-aware-routing/results/baselines/oracle_router.json",
    )
    args = ap.parse_args()

    math_records = load_rollouts(args.math, "is_correct", "math500")
    aime_records = load_rollouts(args.aime, "is_correct", "aime25")
    # WildChat uses 'acceptable' as the correctness field
    wc_records = load_rollouts(args.wildchat, "acceptable", "wildchat")

    print("Per-source rollout counts:")
    print(f"  math500: {len(math_records)} records ({len(math_records)//9} questions × 9 ords)")
    print(f"  aime25:  {len(aime_records)} records")
    print(f"  wildchat:{len(wc_records)} records")

    all_records = math_records + aime_records + wc_records
    grouped = group_by_question(all_records)

    # Determine best worker per source
    best_per_source = {}
    for src, recs in [("math500", math_records), ("aime25", aime_records), ("wildchat", wc_records)]:
        ord_, acc = best_worker_for_source(recs)
        best_per_source[src] = ord_
        print(f"  best for {src}: ord_{ord_} ({acc:.1%})")

    # Oracle
    oracle = simulate_oracle(grouped, best_per_source)

    # All Always-X
    always = []
    for ord_ in range(9):
        always.append(simulate_always_x(grouped, ord_))

    # Random routing baseline (uniform over 9 workers)
    n_total = len(grouped)
    rand_acc = sum(a["n_correct"] for a in always) / sum(a["n"] for a in always) if always else 0
    rand_cost = sum(a["avg_cost_usd"] for a in always) / 9
    random_baseline = {
        "strategy": "uniform_random",
        "n": n_total,
        "accuracy": round(rand_acc, 4),
        "avg_cost_usd": round(rand_cost, 6),
    }

    # Output
    summary = {
        "best_per_source": best_per_source,
        "oracle": oracle,
        "always_x": always,
        "random_uniform": random_baseline,
    }
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 80)
    print(f"{'strategy':25s} {'n':>4s} {'accuracy':>9s} {'$/query':>10s} {'$/correct':>10s}")
    print("=" * 80)
    print(f"{'oracle (per-task best)':25s} {oracle['n']:>4d} {oracle['accuracy']:>8.1%} "
          f"${oracle['avg_cost_usd']:>9.5f} ${oracle['avg_cost_usd']/max(oracle['accuracy'], 0.01):>9.5f}")
    print(f"{'random_uniform':25s} {random_baseline['n']:>4d} {random_baseline['accuracy']:>8.1%} "
          f"${random_baseline['avg_cost_usd']:>9.5f}")
    for a in sorted(always, key=lambda x: x["avg_cost_usd"]):
        per_correct = a["avg_cost_usd"] / max(a["accuracy"], 0.01)
        print(f"  {a['strategy']:23s} {a['n']:>4d} {a['accuracy']:>8.1%} ${a['avg_cost_usd']:>9.5f} ${per_correct:>9.5f}")
    print()

    # Headline
    best_static = max(always, key=lambda a: a["accuracy"])
    print(f"Best static policy: {best_static['strategy']} "
          f"({best_static['accuracy']:.1%} acc, ${best_static['avg_cost_usd']:.5f}/q)")
    print(f"Oracle router:      {oracle['accuracy']:.1%} acc, ${oracle['avg_cost_usd']:.5f}/q")
    delta_acc = (oracle["accuracy"] - best_static["accuracy"]) * 100
    cost_ratio = oracle["avg_cost_usd"] / best_static["avg_cost_usd"]
    print(f"Oracle vs best-static: {delta_acc:+.1f}pp accuracy, "
          f"cost ratio {cost_ratio:.2f}× ({(1-cost_ratio)*100:+.0f}%)")

    print(f"\nPer-source oracle picks:")
    for src, det in oracle["per_source"].items():
        print(f"  {src:10s} ord_{det['ord']}: {det['accuracy']:.1%} at ${det['avg_cost_usd']:.5f}/q ({det['n']} questions)")

    # Per-alpha oracle: best worker by expected reward, picked per source
    print()
    print("=" * 80)
    print("Per-alpha oracle (best worker per dataset by expected reward at each alpha):")
    print(f"{'alpha':>6s} {'oracle picks':30s} {'oracle E[reward]':>18s} {'oracle acc':>10s} {'oracle $/q':>11s}")
    alpha_oracles = {}
    for alpha in [0.5, 1.0, 1.7, 3.0, 5.0]:
        picks = {}
        for src, recs in [("math500", math_records), ("aime25", aime_records), ("wildchat", wc_records)]:
            ord_, _ = best_worker_for_source_at_alpha(recs, alpha)
            picks[src] = ord_
        # Now simulate this routing on the actual data
        n = 0; n_corr = 0; total_cost = 0.0; total_reward = 0.0
        for qid, ords in grouped.items():
            any_rec = next(iter(ords.values()))
            src = any_rec["source"]
            target = picks[src]
            rec = ords.get(target)
            if not rec:
                continue
            n += 1
            n_corr += int(rec["correct"])
            total_cost += rec["cost_usd"]
            cn = cost_normalized(rec["cost_usd"])
            r = max(1 - alpha * cn, -1) if rec["correct"] else 0
            total_reward += r
        picks_str = " ".join(f"{s[:4]}=ord_{o}" for s, o in picks.items())
        e_r = total_reward / max(n, 1)
        acc = n_corr / max(n, 1)
        cost = total_cost / max(n, 1)
        print(f"  {alpha:>4.1f}  {picks_str:30s}  {e_r:>+17.3f}  {acc:>9.1%} ${cost:>10.5f}")
        alpha_oracles[alpha] = {
            "picks": picks, "expected_reward": round(e_r, 4),
            "accuracy": round(acc, 4), "avg_cost_usd": round(cost, 6),
        }

    summary["per_alpha_oracle"] = alpha_oracles
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
