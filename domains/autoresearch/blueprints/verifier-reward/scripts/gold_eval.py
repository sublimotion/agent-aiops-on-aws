#!/usr/bin/env python3
"""
Gold evaluation of agent-generated patches using Docker (Python 3.11).

For each diff in results/diffs/opencode_{model}/, clones the repo at
base_commit, applies test_patch + agent diff, installs deps, and runs
the gold test command inside a Docker container.

Usage:
  python3 gold_eval.py --model haiku
  python3 gold_eval.py --model sonnet --resume
  python3 gold_eval.py --model opus --issue django__django-10914
  python3 gold_eval.py --model all   # run all models
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SUBSET_SEED = 42
SUBSET_SIZE = 50
DOCKER_IMAGE = "python:3.11-bookworm"
EVAL_TIMEOUT = 300  # 5 min per issue
REPO_CACHE_VOL = "swebench-repo-cache"

REPO_SETUP = {
    "django/django": "pip install -e . -q 2>/dev/null",
    "pytest-dev/pytest": "pip install -e '.[testing]' -q 2>/dev/null",
    "sympy/sympy": "pip install -e . -q 2>/dev/null",
    "scikit-learn/scikit-learn": "pip install -e . -q --no-build-isolation 2>/dev/null",
    "matplotlib/matplotlib": "pip install -e '.[dev]' -q 2>/dev/null",
    "pallets/flask": "pip install -e '.[dev]' -q 2>/dev/null",
    "pydata/xarray": "pip install -e . -q 2>/dev/null",
    "astropy/astropy": "pip install -e '.[test]' -q 2>/dev/null",
    "sphinx-doc/sphinx": "pip install -e '.[test]' -q 2>/dev/null",
    "mwaskom/seaborn": "pip install -e '.[dev]' -q 2>/dev/null",
    "pylint-dev/pylint": "pip install -e . -q 2>/dev/null",
    "psf/requests": "pip install -e '.[dev]' -q 2>/dev/null",
}

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@dataclass
class Issue:
    instance_id: str
    repo: str
    base_commit: str
    test_patch: str
    test_cmd: str


def _build_test_cmd(row: dict) -> str:
    repo = row["repo"]
    fail_to_pass = row.get("FAIL_TO_PASS", "")
    if isinstance(fail_to_pass, str):
        try:
            tests = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            tests = [fail_to_pass] if fail_to_pass else []
    else:
        tests = fail_to_pass or []

    if not tests:
        return "python -m pytest"

    if "django" in repo:
        test_modules = set()
        for t in tests:
            if "(" in t:
                module_path = t.split("(")[1].rstrip(")")
                parts = module_path.split(".")
                if parts:
                    test_modules.add(parts[0])
            else:
                test_modules.add(t.split(".")[0])
        return f"python3 tests/runtests.py {' '.join(sorted(test_modules))}"

    test_paths = set()
    for t in tests:
        if "::" in t:
            test_paths.add(t.split("::")[0])
        elif "(" in t:
            module_path = t.split("(")[1].rstrip(")")
            test_paths.add(module_path.replace(".", "/") + ".py")
        else:
            test_paths.add(t)
    if test_paths:
        return f"python3 -m pytest {' '.join(sorted(test_paths))} -x"
    return "python3 -m pytest"


def load_subset() -> dict[str, Issue]:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    all_issues = []
    for row in ds:
        all_issues.append(Issue(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            test_patch=row.get("test_patch", ""),
            test_cmd=_build_test_cmd(row),
        ))

    rng = random.Random(SUBSET_SEED)
    by_repo = {}
    for issue in all_issues:
        by_repo.setdefault(issue.repo, []).append(issue)

    selected = []
    repos = sorted(by_repo.keys())
    rng.shuffle(repos)
    idx = {r: 0 for r in repos}
    while len(selected) < SUBSET_SIZE:
        for repo in repos:
            issues = by_repo[repo]
            if idx[repo] < len(issues) and len(selected) < SUBSET_SIZE:
                selected.append(issues[idx[repo]])
                idx[repo] += 1

    return {i.instance_id: i for i in selected}


def run_gold_eval_docker(issue: Issue, diff_path: str) -> dict:
    """Run gold eval for a single issue inside Docker."""
    setup_cmd = REPO_SETUP.get(issue.repo, "pip install -e . -q 2>/dev/null")

    # Write test patch to a temp file to mount into container
    import tempfile
    test_patch_file = tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False)
    test_patch_file.write(issue.test_patch or "")
    test_patch_file.close()

    repo_cache_name = issue.repo.replace("/", "__")
    script = f"""#!/bin/bash

# Use cached repo clone if available, otherwise clone fresh
CACHE_DIR="/repo-cache/{repo_cache_name}"
if [ -d "$CACHE_DIR/.git" ]; then
    cp -a "$CACHE_DIR" /workspace
    cd /workspace
    git fetch origin 2>/dev/null
else
    git clone https://github.com/{issue.repo}.git /workspace 2>&1 | tail -1
    cd /workspace
    # Cache for reuse
    cp -a /workspace "$CACHE_DIR" 2>/dev/null || true
fi
git checkout -f {issue.base_commit} 2>&1
if [ $? -ne 0 ]; then
    echo "CHECKOUT_FAILED"
    exit 1
fi

# Apply test patch (mounted at /mnt/test.patch)
if [ -s /mnt/test.patch ]; then
    git apply --allow-empty /mnt/test.patch 2>&1 || echo "TEST_PATCH_WARN"
fi

# Apply agent diff (use patch -p1 which tolerates truncated hunks)
{{ cat /mnt/agent.patch; printf "\\n"; }} > /tmp/agent.patch
patch -p1 --no-backup-if-mismatch < /tmp/agent.patch 2>&1
if [ $? -ne 0 ]; then
    echo "AGENT_PATCH_FAILED"
fi

# Install deps
{setup_cmd}

# Run tests
echo "===GOLD_TEST_START==="
{issue.test_cmd} 2>&1 || true
echo "===GOLD_TEST_END==="
"""

    start = time.monotonic()
    try:
        proc = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network", "host",
                "--memory", "4g",
                "--cpus", "2",
                "-v", f"{REPO_CACHE_VOL}:/repo-cache",
                "-v", f"{os.path.abspath(diff_path)}:/mnt/agent.patch:ro",
                "-v", f"{test_patch_file.name}:/mnt/test.patch:ro",
                DOCKER_IMAGE,
                "bash", "-c", script,
            ],
            capture_output=True, text=True,
            timeout=EVAL_TIMEOUT,
        )
        output = proc.stdout + "\n" + proc.stderr
        elapsed = time.monotonic() - start

        # Cleanup temp file
        os.unlink(test_patch_file.name)

        # Parse test results
        checkout_failed = "CHECKOUT_FAILED" in output
        patch_failed = "AGENT_PATCH_FAILED" in output
        if "===GOLD_TEST_START===" in output:
            test_output = output.split("===GOLD_TEST_START===")[1].split("===GOLD_TEST_END===")[0].lower()
        else:
            test_output = output.lower()

        passed = False
        if not patch_failed:
            if "passed" in test_output:
                before_passed = test_output.split("passed")[0]
                if "failed" not in before_passed and "error" not in before_passed:
                    passed = True
            if "\nok" in test_output or test_output.rstrip().endswith("ok"):
                if "fail" not in test_output and "error" not in test_output:
                    passed = True

        return {
            "passed": passed,
            "patch_applied": not patch_failed,
            "elapsed_s": round(elapsed, 1),
            "exit_code": proc.returncode,
            "error": None,
        }

    except subprocess.TimeoutExpired:
        os.unlink(test_patch_file.name)
        return {"passed": False, "patch_applied": None, "elapsed_s": EVAL_TIMEOUT, "exit_code": -1, "error": "timeout"}
    except Exception as e:
        os.unlink(test_patch_file.name)
        return {"passed": False, "patch_applied": None, "elapsed_s": 0, "exit_code": -1, "error": str(e)[:200]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="haiku|sonnet|opus|all")
    parser.add_argument("--issue", type=str, help="Single issue ID")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    models = ["haiku", "sonnet", "opus"] if args.model == "all" else [args.model]

    # Pull image and create cache volume
    log.info(f"Ensuring Docker image {DOCKER_IMAGE} is available...")
    subprocess.run(["docker", "pull", DOCKER_IMAGE], capture_output=True, timeout=300)
    subprocess.run(["docker", "volume", "create", REPO_CACHE_VOL], capture_output=True)

    # Load dataset
    log.info("Loading SWE-bench Lite subset...")
    issues = load_subset()

    for model in models:
        diffs_dir = RESULTS_DIR / "diffs" / f"opencode_{model}"
        if not diffs_dir.exists():
            log.warning(f"No diffs for {model}, skipping")
            continue

        output_file = RESULTS_DIR / f"gold_{model}_opencode.jsonl"

        # Load completed
        completed = set()
        if args.resume and output_file.exists():
            for line in output_file.read_text().strip().split("\n"):
                if line:
                    try:
                        completed.add(json.loads(line)["instance_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
            log.info(f"[{model}] Resuming: {len(completed)} already done")

        diff_files = sorted(diffs_dir.glob("*.diff"))
        if args.issue:
            diff_files = [d for d in diff_files if d.stem == args.issue]
        if args.limit:
            diff_files = diff_files[:args.limit]

        total = len(diff_files)
        passed = 0
        evaluated = 0

        log.info(f"\n{'='*60}")
        log.info(f"Gold eval: {model.upper()} × OpenCode ({total} diffs)")
        log.info(f"{'='*60}")

        for idx, diff_file in enumerate(diff_files):
            instance_id = diff_file.stem
            if instance_id in completed:
                log.info(f"[{idx+1}/{total}] SKIP {instance_id} (done)")
                continue

            if instance_id not in issues:
                log.warning(f"[{idx+1}/{total}] {instance_id} not in subset, skipping")
                continue

            issue = issues[instance_id]
            log.info(f"[{idx+1}/{total}] {instance_id} ({issue.repo})")

            result = run_gold_eval_docker(issue, str(diff_file))
            evaluated += 1
            if result["passed"]:
                passed += 1

            status = "PASS" if result["passed"] else "FAIL"
            log.info(f"  {status} | patch={result['patch_applied']} time={result['elapsed_s']}s")
            if result["error"]:
                log.warning(f"  ERROR: {result['error']}")

            row = {"instance_id": instance_id, "model": model, **result}
            with open(output_file, "a") as f:
                f.write(json.dumps(row) + "\n")

        log.info(f"\n{model.upper()}: {passed}/{evaluated} passed ({100*passed/max(evaluated,1):.0f}%)")

    log.info("\nDone!")


if __name__ == "__main__":
    main()
