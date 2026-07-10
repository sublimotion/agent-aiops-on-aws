#!/usr/bin/env python3
"""Phase 2 — evaluate a candidate patch against a task's held-out tests.

Scoring model (SWE-bench-style, no LLM judge):
  1. fresh checkout at base_commit (sealed workspace)
  2. apply the agent's candidate patch (the fix under test)
  3. OVERWRITE the held-out test files with their POST-FIX versions (fetched from
     the merge commit) — the agent never saw these; they are ground truth
  4. install the package + run exactly those test files
  5. PASS iff all held-out tests pass

Modes:
  --mode venv    : uv venv on this box (pilot mechanism check; fast, no Docker)
  --mode docker  : build/run in a container (production path, separate CPU box)

Usage:
  python3 run_eval.py --task-file results/tasks.jsonl --pr 13363 \
      --patch /tmp/cand.diff --dir /tmp/ppe-ws-13363 --mode venv
  # empty/omitted --patch = evaluate the base (should FAIL held-out tests => sane)
"""
import argparse, json, subprocess, sys, pathlib, os


def run(cmd, cwd=None, check=False, env=None, timeout=1800):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          env=env, timeout=timeout)


def load_task(task_file, pr):
    for l in open(task_file):
        r = json.loads(l)
        if r["pr_number"] == pr:
            return r
    sys.exit(f"pr {pr} not in {task_file}")


def fetch_postfix_tests(repo, merge_commit, test_paths, dest):
    """Get the ground-truth (post-fix) test files from the merge commit via gh API.
    These OVERWRITE whatever the agent left, so the agent can't game the tests."""
    for tp in test_paths:
        r = run(["gh", "api",
                 f"repos/{repo}/contents/{tp}?ref={merge_commit}",
                 "-H", "Accept: application/vnd.github.raw"])
        if r.returncode != 0:
            print(f"[eval] WARN could not fetch held-out {tp}@{merge_commit[:8]}")
            continue
        p = pathlib.Path(dest) / tp
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(r.stdout)
    print(f"[eval] injected {len(test_paths)} held-out test file(s) from fix commit")


def apply_patch(wsdir, patch):
    if not patch or not os.path.exists(patch) or os.path.getsize(patch) == 0:
        print("[eval] no patch (evaluating base tree) — held-out tests should FAIL")
        return True
    for args in (["git", "apply", "--whitespace=nowarn", patch],
                 ["git", "apply", "-3", patch],
                 ["patch", "-p1", "-i", patch]):
        r = run(args, cwd=wsdir)
        if r.returncode == 0:
            print(f"[eval] patch applied via: {' '.join(args[:2])}")
            return True
    print(f"[eval] PATCH FAILED TO APPLY -> counts as fail\n{r.stderr[:400]}")
    return False


def eval_venv(wsdir, test_paths):
    """Isolated uv venv per workspace: create venv, editable-install pydantic +
    test extras, pytest the held-out files. Returns (passed, summary)."""
    venv = os.path.join(wsdir, ".eval-venv")
    py = os.path.join(venv, "bin", "python")
    mk = run(["uv", "venv", "--python", "3.12", venv], cwd=wsdir, timeout=300)
    if mk.returncode != 0:
        return False, f"venv create failed:\n{mk.stderr[-600:]}"
    # editable install of pydantic + its DECLARED test dep-groups (PEP-735) rather
    # than hand-guessed extras — avoids the "version-specific deps" trap (carryover
    # agent-harness L158). uv resolves dependency-groups from pyproject/uv.lock.
    inst = run(["uv", "pip", "install", "--python", py, "-e", ".",
                "--group", "dev", "--group", "testing-extra", "setuptools"],
               cwd=wsdir, timeout=1800)
    if inst.returncode != 0:
        # fallback: some repos don't use dependency-groups → common extras
        inst = run(["uv", "pip", "install", "--python", py, "-e", ".",
                    "pytest", "dirty-equals", "pytest-mock", "hypothesis",
                    "email-validator", "jsonschema", "cloudpickle", "pytz",
                    "tzdata", "setuptools"], cwd=wsdir, timeout=1800)
        if inst.returncode != 0:
            return False, f"install failed:\n{inst.stderr[-800:]}"
    # neutralize repo-specific addopts (pydantic uses pytest-benchmark flags in
    # pyproject) so held-out tests run standalone regardless of the repo's config
    res = run([py, "-m", "pytest", "-x", "-q", "-o", "addopts=",
               "-p", "no:cacheprovider", *test_paths], cwd=wsdir, timeout=1200)
    tail = (res.stdout + res.stderr)[-1400:]
    passed = res.returncode == 0
    return passed, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", default="results/tasks.jsonl")
    ap.add_argument("--pr", type=int, required=True)
    ap.add_argument("--patch", default="")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", choices=["venv", "docker"], default="venv")
    args = ap.parse_args()

    task = load_task(args.task_file, args.pr)
    if not apply_patch(args.dir, args.patch):
        print(json.dumps({"pr": args.pr, "passed": False, "reason": "patch_apply_failed"}))
        return
    fetch_postfix_tests(task["repo"], task["merge_commit"],
                        task["held_out_tests"], args.dir)

    if args.mode == "docker":
        sys.exit("[eval] docker mode is the production path — run on the separate CPU box "
                 "(see spec Execution topology). Not wired in the pilot.")
    passed, summary = eval_venv(args.dir, task["held_out_tests"])
    print(f"[eval] pr={args.pr} tier={task.get('complexity_tier')} "
          f"PASSED={passed}\n--- pytest tail ---\n{summary}")
    print(json.dumps({"pr": args.pr, "passed": passed,
                      "tier": task.get("complexity_tier")}))


if __name__ == "__main__":
    main()
