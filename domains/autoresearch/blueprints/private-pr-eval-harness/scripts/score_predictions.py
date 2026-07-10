#!/usr/bin/env python3
"""Phase 3/4 (pilot) — score a cell's predictions against held-out tests + report.

For each prediction: seal a fresh workspace at base_commit, write the candidate
patch, run_eval (apply patch -> inject held-out tests from fix commit -> pytest).
Aggregate per-tier pass rate and print the contamination-gap framing vs a supplied
SWE-bench-Lite baseline for the same model family.

Usage:
  python3 score_predictions.py --pred results/pred-opus.jsonl --cell opus \
      --task-file results/tasks.jsonl --tmp /tmp/score-opus \
      --lite-baseline 0.583 --out results/scored-opus.json
"""
import argparse, json, subprocess, pathlib, sys
from collections import defaultdict

HERE = pathlib.Path(__file__).parent


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def task_by_pr(task_file):
    return {json.loads(l)["pr_number"]: json.loads(l) for l in open(task_file)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--task-file", default="results/tasks.jsonl")
    ap.add_argument("--tmp", default="/tmp/score")
    ap.add_argument("--lite-baseline", type=float, default=None,
                    help="published SWE-bench-Lite pass rate for this model family")
    ap.add_argument("--out", default="results/scored.json")
    args = ap.parse_args()

    tasks = task_by_pr(args.task_file)
    preds = [json.loads(l) for l in open(args.pred)]
    results = []
    for p in preds:
        pr = p["pr"]
        t = tasks.get(pr)
        rec = {"pr": pr, "tier": p.get("tier"), "gen_ok": p.get("gen_ok"),
               "patch_len": p.get("patch_len", 0)}
        if not t or not p.get("patch", "").strip():
            rec.update(passed=False, note="empty_patch")
            results.append(rec); print(rec); continue
        ws = f"{args.tmp}-{pr}"
        patchf = f"{args.tmp}-{pr}.diff"
        pathlib.Path(patchf).write_text(p["patch"])
        s = run(["python3", str(HERE / "seal_workspace.py"),
                 "--repo", t["repo"], "--base", t["base_commit"], "--dir", ws])
        if "PASS" not in s.stdout:
            rec.update(passed=False, note="seal_failed"); results.append(rec); print(rec); continue
        e = run(["python3", str(HERE / "run_eval.py"), "--task-file", args.task_file,
                 "--pr", str(pr), "--patch", patchf, "--dir", ws, "--mode", "venv"])
        try:
            rec["passed"] = json.loads(e.stdout.strip().splitlines()[-1])["passed"]
        except Exception:
            rec.update(passed=False, note="eval_error", tail=(e.stdout + e.stderr)[-200:])
        results.append(rec); print(rec)

    n = len(results)
    npass = sum(1 for r in results if r.get("passed"))
    by_tier = defaultdict(lambda: [0, 0])
    for r in results:
        by_tier[r.get("tier")][0] += 1
        by_tier[r.get("tier")][1] += 1 if r.get("passed") else 0
    summary = {"cell": args.cell, "n": n, "passed": npass,
               "pass_rate": npass / n if n else 0,
               "by_tier": {k: {"n": v[0], "passed": v[1]} for k, v in by_tier.items()},
               "lite_baseline": args.lite_baseline, "results": results}
    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))

    print(f"\n=== {args.cell}: pydantic OOD pass rate = {npass}/{n} = "
          f"{summary['pass_rate']:.1%} ===")
    for tier in ("low", "medium", "high"):
        if tier in by_tier:
            v = by_tier[tier]; print(f"  {tier:<7} {v[1]}/{v[0]}")
    if args.lite_baseline is not None:
        gap = summary["pass_rate"] - args.lite_baseline
        print(f"  SWE-bench-Lite baseline: {args.lite_baseline:.1%}  "
              f"=> contamination gap: {gap:+.1%} "
              f"({'OOD HARDER' if gap < 0 else 'OOD not harder'})")
    print(f"[score] wrote {args.out}")


if __name__ == "__main__":
    main()
