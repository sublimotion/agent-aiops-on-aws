#!/usr/bin/env python3
"""Phase 2 exit — validate the seal->inject->score pipeline across N tasks.

For each task: fetch the PR diff, split into source-half (gold candidate) and
test-half (ground truth, injected by run_eval), seal a fresh workspace at
base_commit, apply the gold SOURCE patch, and confirm held-out tests PASS.

A healthy pipeline shows gold_pass=True for most tasks. Tasks where gold fails to
apply or gold-patched tests still fail flag either a mining artifact (wrong base,
test needs uninstalled dep) or a genuinely hard-to-isolate PR — logged, not hidden.

This does NOT run any agent — it validates ground truth. Agent cells (generation)
run separately via `fe agent launch`.

Usage:
  python3 validate_batch.py --task-file results/tasks.jsonl --n 10 \
      --tmp /tmp/ppe --out results/validate-batch.json
"""
import argparse, json, re, subprocess, sys, os, pathlib

HERE = pathlib.Path(__file__).parent


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def pr_diff(repo, pr):
    r = run(["gh", "api", f"repos/{repo}/pulls/{pr}",
             "-H", "Accept: application/vnd.github.v3.diff"])
    return r.stdout if r.returncode == 0 else ""


def split_source(diff):
    parts = re.split(r"(?=^diff --git )", diff, flags=re.M)
    src = [p for p in parts
           if p.startswith("diff --git") and "/tests/" not in p.splitlines()[0]
           and "/test_" not in p.splitlines()[0]]
    return "".join(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", default="results/tasks.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tmp", default="/tmp/ppe-batch")
    ap.add_argument("--out", default="results/validate-batch.json")
    ap.add_argument("--tiers", default="low,medium,high",
                    help="sample across these tiers proportionally")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.task_file)]
    # spread across tiers so validation isn't all-easy
    want = args.tiers.split(",")
    picked, by = [], {t: [r for r in tasks if r.get("complexity_tier") == t] for t in want}
    i = 0
    while len(picked) < args.n and any(by.values()):
        t = want[i % len(want)]
        if by[t]:
            picked.append(by[t].pop(0))
        i += 1

    results = []
    for r in picked:
        pr, repo, base = r["pr_number"], r["repo"], r["base_commit"]
        ws = f"{args.tmp}-{pr}"
        rec = {"pr": pr, "tier": r.get("complexity_tier"), "n_files": r["n_files"],
               "net_lines": r["net_lines"]}
        diff = pr_diff(repo, pr)
        src = split_source(diff)
        if not src.strip():
            rec.update(gold_pass=False, note="no source hunks in diff")
            results.append(rec); print(rec); continue
        patchf = f"{args.tmp}-{pr}.diff"
        pathlib.Path(patchf).write_text(src)
        # seal
        s = run(["python3", str(HERE / "seal_workspace.py"),
                 "--repo", repo, "--base", base, "--dir", ws])
        if "PASS" not in s.stdout:
            rec.update(gold_pass=False, note="seal/probe failed", seal=s.stdout[-200:])
            results.append(rec); print(rec); continue
        # eval gold
        e = run(["python3", str(HERE / "run_eval.py"), "--task-file", args.task_file,
                 "--pr", str(pr), "--patch", patchf, "--dir", ws, "--mode", "venv"])
        try:
            verdict = json.loads(e.stdout.strip().splitlines()[-1])
            rec["gold_pass"] = verdict["passed"]
        except Exception:
            rec.update(gold_pass=False, note="eval parse error",
                       tail=(e.stdout + e.stderr)[-300:])
        results.append(rec)
        print(rec)

    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    npass = sum(1 for r in results if r.get("gold_pass"))
    print(f"\n[validate] gold_pass {npass}/{len(results)} "
          f"— healthy pipeline expects most True")
    print(f"[validate] wrote {args.out}")


if __name__ == "__main__":
    main()
