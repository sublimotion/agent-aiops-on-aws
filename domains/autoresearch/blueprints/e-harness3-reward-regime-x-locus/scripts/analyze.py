#!/usr/bin/env python3
"""
E_harness3 analysis — the reward-regime × authoring-locus MATRIX.

Cells (Pass@1 per worker model):
                self-author     external-author
  verifiable    A (=H2 L2)      C (=H2 L3)        DBBench, SQL-exec reward
  withheld      B               D                 DBBench, reward blinded to author
  consensus     E               F                 FinanceBench, LLM-judge reward

Primary outputs:
  * per-cell Pass@1.
  * the locus gap (external - self) per regime:
        verifiable: C - A    (paired by DBBench task_id)
        withheld:   D - B    (paired by DBBench task_id)
        consensus:  F - E    (paired by FinanceBench task_id)
    with paired-bootstrap 95% CIs.
  * MONOTONICITY of the locus gap across verifiable -> withheld -> consensus
    (the core hypothesis: gap grows as verifiable reward weakens).
  * RQ4: judge-confound flag for E/F from results/judge_gate.json.

DBBench cells (A/B/C/D) pair on the SAME 120 task_ids; consensus cells (E/F) pair
on FinanceBench ids. The locus gap is a within-task paired estimate in every
regime, so the three gaps are directly comparable.
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict


def load(tag, results_dir):
    # tolerate the optional _v-haiku suffix on external cells
    paths = glob.glob(os.path.join(results_dir, f"{tag}.jsonl")) or \
        glob.glob(os.path.join(results_dir, f"{tag}_v-*.jsonl"))
    out = {}
    for p in paths:
        for line in open(p):
            try:
                r = json.loads(line)
                out[r["task_id"]] = r
            except Exception:  # noqa: BLE001
                pass
    return out


def passat1(d):
    return (sum(int(r["is_correct"]) for r in d.values()) / len(d)) if d else None


def paired_delta_ci(self_d, ext_d, n_boot=5000, seed=42):
    """Paired bootstrap CI for Pass@1(external) - Pass@1(self) on shared ids."""
    ids = sorted(set(self_d) & set(ext_d))
    if not ids:
        return None
    sv = [int(self_d[i]["is_correct"]) for i in ids]
    ev = [int(ext_d[i]["is_correct"]) for i in ids]
    obs = (sum(ev) - sum(sv)) / len(ids)
    rng = random.Random(seed)
    n = len(ids)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append((sum(ev[j] for j in idx) - sum(sv[j] for j in idx)) / n)
    deltas.sort()
    return {"delta": obs, "ci": [deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot)]],
            "n": n, "self_p": sum(sv) / n, "ext_p": sum(ev) / n}


def diff_of_gaps_ci(self_a, ext_a, self_b, ext_b, n_boot=5000, seed=7):
    """Bootstrap CI for (gap_b - gap_a) where each gap is a paired locus diff.
    Used for monotonic-step significance (e.g. withheld-gap minus verifiable-gap).
    Resamples each regime's shared-id pool independently (regimes are different
    task sets / different domains)."""
    ids_a = sorted(set(self_a) & set(ext_a))
    ids_b = sorted(set(self_b) & set(ext_b))
    if not ids_a or not ids_b:
        return None

    def gap(ids, sd, ed, idx):
        n = len(ids)
        sv = sum(int(sd[ids[j]]["is_correct"]) for j in idx)
        ev = sum(int(ed[ids[j]]["is_correct"]) for j in idx)
        return (ev - sv) / n
    rng = random.Random(seed)
    na, nb = len(ids_a), len(ids_b)
    obs = gap(ids_b, self_b, ext_b, range(nb)) - gap(ids_a, self_a, ext_a, range(na))
    diffs = []
    for _ in range(n_boot):
        ia = [rng.randrange(na) for _ in range(na)]
        ib = [rng.randrange(nb) for _ in range(nb)]
        diffs.append(gap(ids_b, self_b, ext_b, ib) - gap(ids_a, self_a, ext_a, ia))
    diffs.sort()
    return {"diff": obs, "ci": [diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    R = args.results

    # cell -> (self_tag, ext_tag) per model
    REGIMES = [
        ("verifiable", "A", "C"),
        ("withheld", "B", "D"),
        ("consensus", "E", "F"),
    ]
    models = ["haiku", "sonnet"]
    report = {"models": {}, "monotonicity": {}, "judge_gate": None}

    # pull judge gate verdict (RQ4)
    jg_path = os.path.join(R, "judge_gate.json")
    if os.path.exists(jg_path):
        jg = json.load(open(jg_path))
        report["judge_gate"] = {k: jg.get(k) for k in
                                ("verdict", "auc", "near_miss_reject_rate",
                                 "temperature_stability", "note")}

    for m in models:
        cells = {}
        gaps = {}
        for regime, s, e in REGIMES:
            sd = load(f"{s}_{m}", R)
            ed = load(f"{e}_{m}", R)
            cells[s] = passat1(sd)
            cells[e] = passat1(ed)
            g = paired_delta_ci(sd, ed)
            if g:
                gaps[regime] = g
        report["models"][m] = {"pass_at_1": cells, "locus_gap_external_minus_self": gaps}

    # monotonicity: locus gap across verifiable -> withheld -> consensus
    for m in models:
        gaps = report["models"][m]["locus_gap_external_minus_self"]
        seq = [(rg, gaps[rg]["delta"]) for rg, _, _ in REGIMES if rg in gaps]
        report["monotonicity"][m] = {
            "ordered_gaps": seq,
            "monotonic_nondecreasing": all(seq[i + 1][1] >= seq[i][1] - 1e-9
                                           for i in range(len(seq) - 1)) if len(seq) > 1 else None,
        }
        # step CIs
        loaded = {}
        for regime, s, e in REGIMES:
            loaded[regime] = (load(f"{s}_{m}", R), load(f"{e}_{m}", R))
        steps = {}
        if "verifiable" in gaps and "withheld" in gaps:
            steps["withheld_minus_verifiable"] = diff_of_gaps_ci(
                *loaded["verifiable"], *loaded["withheld"])
        if "withheld" in gaps and "consensus" in gaps:
            steps["consensus_minus_withheld"] = diff_of_gaps_ci(
                *loaded["withheld"], *loaded["consensus"])
        if "verifiable" in gaps and "consensus" in gaps:
            steps["consensus_minus_verifiable"] = diff_of_gaps_ci(
                *loaded["verifiable"], *loaded["consensus"])
        report["monotonicity"][m]["steps"] = steps

    out = args.out or os.path.join(R, "matrix.json")
    json.dump(report, open(out, "w"), indent=2)

    # pretty
    print("=" * 74)
    print("E_harness3 — reward-regime × authoring-locus MATRIX")
    print("=" * 74)
    for m in models:
        md = report["models"][m]
        print(f"\n### worker: {m}")
        p = md["pass_at_1"]
        print(f"  Pass@1  | self  | ext   |  (external - self)")
        for regime, s, e in REGIMES:
            g = md["locus_gap_external_minus_self"].get(regime)
            if not g:
                print(f"  {regime:10s}|  {p.get(s)}  |  {p.get(e)}  |  (missing)")
                continue
            lo, hi = g["ci"]
            sig = "" if lo <= 0 <= hi else "  *(CI excludes 0)*"
            print(f"  {regime:10s}| {g['self_p']:.3f} | {g['ext_p']:.3f} | "
                  f"{g['delta']:+.3f} [{lo:+.3f},{hi:+.3f}] (n={g['n']}){sig}")
        mono = report["monotonicity"][m]
        print(f"  monotonic (gap grows verifiable->withheld->consensus)? "
              f"{mono['monotonic_nondecreasing']}")
        print(f"  ordered gaps: {[(r, round(v,3)) for r,v in mono['ordered_gaps']]}")
        for name, st in mono["steps"].items():
            if st:
                lo, hi = st["ci"]
                sig = "" if lo <= 0 <= hi else "  *(CI excludes 0)*"
                print(f"    step {name}: {st['diff']:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
    if report["judge_gate"]:
        jg = report["judge_gate"]
        print(f"\n### RQ4 judge gate: {jg['verdict']}  AUC={jg['auc']}  "
              f"near-miss-reject={jg.get('near_miss_reject_rate')}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
