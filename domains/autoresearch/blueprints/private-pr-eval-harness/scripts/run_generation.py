#!/usr/bin/env python3
"""Phase 3 (pilot) — generate candidate patches with Claude Code headless (Bedrock).

For each task: seal a fresh workspace at base_commit, run `claude -p` headless inside
it with tools enabled, capture `git diff` (SOURCE files only) as the candidate patch.
This is the generation half; run_eval.py scores each patch against held-out tests.

Bedrock model is selected by env (CLAUDE_CODE_USE_BEDROCK=1 + ANTHROPIC_MODEL), so a
cell = one model. Run once per model (Opus, Sonnet) to get 2 cells.

Usage:
  ANTHROPIC_MODEL=<opus-arn> python3 run_generation.py --cell opus \
      --task-file results/tasks.jsonl --n 10 --tmp /tmp/gen --out results/pred-opus.jsonl
"""
import argparse, json, subprocess, os, pathlib, sys

HERE = pathlib.Path(__file__).parent


def run(cmd, cwd=None, timeout=900, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def seal(repo, base, ws):
    r = run(["python3", str(HERE / "seal_workspace.py"),
             "--repo", repo, "--base", base, "--dir", ws])
    return "PASS" in r.stdout


def source_diff(ws, source_files):
    """git diff restricted to the PR's source files = the candidate patch.
    (Test files are ground truth injected at eval time, never the agent's job.)"""
    r = run(["git", "diff", "--", *source_files], cwd=ws)
    if r.returncode != 0 or not r.stdout.strip():
        # fall back to full diff minus tests
        r = run(["git", "diff"], cwd=ws)
    return r.stdout


def generate_one(task, ws, max_turns):
    prompt = (task["prompt"] +
              "\n\nMake the minimal code change that fixes this. Edit the source "
              "files directly. Do not write new test files.")
    env = dict(os.environ)  # inherits CLAUDE_CODE_USE_BEDROCK + ANTHROPIC_MODEL
    r = run(["claude", "-p", prompt, "--output-format", "json",
             "--max-turns", str(max_turns),
             "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep"],
            cwd=ws, timeout=1200, env=env)
    patch = source_diff(ws, task["source_files"])
    # claude -p exits 0 even at max_turns; success = a non-empty candidate patch was
    # produced (an empty diff means the agent explored but never edited — a real fail)
    ok = bool(patch.strip())
    note = "" if ok else (f"rc={r.returncode} no_edit; " + (r.stderr or "")[-200:])
    return ok, patch, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--task-file", default="results/tasks.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tmp", default="/tmp/gen")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--out", default="results/pred.jsonl")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.task_file)][:args.n]
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    model = os.environ.get("ANTHROPIC_MODEL", "unknown")
    print(f"[gen] cell={args.cell} model={model.split('/')[-1]} n={len(tasks)}")

    with outp.open("w") as fh:
        for i, t in enumerate(tasks, 1):
            ws = f"{args.tmp}-{args.cell}-{t['pr_number']}"
            if not seal(t["repo"], t["base_commit"], ws):
                rec = {"pr": t["pr_number"], "cell": args.cell,
                       "patch": "", "note": "seal_failed"}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"[gen] {i}/{len(tasks)} #{t['pr_number']} SEAL FAILED"); continue
            try:
                ok, patch, err = generate_one(t, ws, args.max_turns)
            except subprocess.TimeoutExpired:
                ok, patch, err = False, "", "timeout"
            rec = {"pr": t["pr_number"], "cell": args.cell,
                   "tier": t.get("complexity_tier"),
                   "patch": patch, "patch_len": len(patch),
                   "gen_ok": ok, "note": "" if ok else err}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[gen] {i}/{len(tasks)} #{t['pr_number']} tier={t.get('complexity_tier')} "
                  f"gen_ok={ok} patch_len={len(patch)}")

    print(f"[gen] wrote {outp}")


if __name__ == "__main__":
    main()
