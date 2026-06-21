#!/usr/bin/env python3
"""
E_harness2 analysis — Pass@1 per layer, the three deltas with bootstrap CIs,
per-failure-type breakdown, JIT state-size confound check, and RQ4 transfer.

Pairs by task_id so deltas are computed on the SAME tasks (paired bootstrap),
which is the right estimator for a within-task layered ablation.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import dbbench_common as C

# deterministic bootstrap (no Math.random equivalent needed; use a seeded RNG)
import random


def load(tag, results_dir):
    path = os.path.join(results_dir, f"{tag}.jsonl")
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
            out[r["task_id"]] = r
        except Exception:  # noqa: BLE001
            pass
    return out


def passat1(d):
    if not d:
        return None
    return sum(r["is_correct"] for r in d.values()) / len(d)


def paired_delta_ci(a, b, n_boot=5000, seed=42):
    """Paired bootstrap CI for Pass@1(b) - Pass@1(a) over shared task_ids."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    av = [a[i]["is_correct"] for i in ids]
    bv = [b[i]["is_correct"] for i in ids]
    obs = (sum(bv) - sum(av)) / len(ids)
    rng = random.Random(seed)
    deltas = []
    n = len(ids)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(av[j] for j in idx) / n
        db = sum(bv[j] for j in idx) / n
        deltas.append(db - da)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot)]
    return {"delta": obs, "ci": [lo, hi], "n": n}


def by_type(d):
    agg = defaultdict(lambda: [0, 0])
    for r in d.values():
        agg[r["type"]][0] += int(r["is_correct"])
        agg[r["type"]][1] += 1
    return {k: (c, n, c / n) for k, (c, n) in sorted(agg.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{C.ROOT}/results")
    ap.add_argument("--out", default=f"{C.ROOT}/results/analysis.json")
    args = ap.parse_args()

    models = ["haiku", "sonnet"]
    report = {"models": {}, "transfer": {}}

    for m in models:
        layers = {
            "L0": load(f"L0_{m}", args.results),
            "L1": load(f"L1_{m}", args.results),
            "L2": load(f"L2_{m}", args.results),
            "L3": load(f"L3_{m}_v-haiku", args.results) or load(f"L3_{m}", args.results),
        }
        present = {k: v for k, v in layers.items() if v}
        if not present:
            continue
        p = {k: passat1(v) for k, v in present.items()}
        deltas = {}
        if "L0" in present and "L1" in present:
            deltas["L0->L1"] = paired_delta_ci(layers["L0"], layers["L1"])
        if "L1" in present and "L2" in present:
            deltas["L1->L2"] = paired_delta_ci(layers["L1"], layers["L2"])
        if "L2" in present and "L3" in present:
            deltas["L2->L3"] = paired_delta_ci(layers["L2"], layers["L3"])
        # JIT confound: state size vs correctness for L2/L3
        jit_state = {}
        for L in ("L2", "L3"):
            if L in present:
                rs = list(present[L].values())
                sizes = [r.get("jit_state_chars", 0) for r in rs]
                jit_state[L] = {
                    "max_chars": max(sizes) if sizes else 0,
                    "mean_chars": sum(sizes) / len(sizes) if sizes else 0,
                    "n_interventions": max((r.get("jit_notes_total", 0) for r in rs), default=0),
                }
        report["models"][m] = {
            "n_tasks": {k: len(v) for k, v in present.items()},
            "pass_at_1": p,
            "deltas": deltas,
            "by_type": {k: by_type(v) for k, v in present.items()},
            "jit_state": jit_state,
        }

    # RQ4 transfer: does the L2->L3 sign/magnitude port across models?
    for m in models:
        md = report["models"].get(m, {})
        d = md.get("deltas", {}).get("L2->L3")
        if d:
            report["transfer"][m] = d

    json.dump(report, open(args.out, "w"), indent=2)

    # pretty print
    print("=" * 70)
    print("E_harness2 — DBBench layered ablation (JIT vs offline, self vs external)")
    print("=" * 70)
    for m, md in report["models"].items():
        print(f"\n### worker model: {m}")
        print("  Pass@1:", {k: f"{v:.3f}" for k, v in md["pass_at_1"].items()})
        print("  n:", md["n_tasks"])
        for name, dd in md["deltas"].items():
            if dd:
                lo, hi = dd["ci"]
                sig = "" if (lo <= 0 <= hi) else "  *(CI excludes 0)*"
                print(f"  Δ {name}: {dd['delta']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}] (n={dd['n']}){sig}")
        if md.get("jit_state"):
            print("  JIT state:", md["jit_state"])
    print("\n### RQ4 transfer (L2->L3 across models):")
    for m, dd in report["transfer"].items():
        lo, hi = dd["ci"]
        print(f"  {m}: Δ={dd['delta']:+.3f} CI[{lo:+.3f},{hi:+.3f}]")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
