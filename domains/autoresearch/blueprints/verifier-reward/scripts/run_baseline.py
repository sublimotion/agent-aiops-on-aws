#!/usr/bin/env python3
"""
Phase 1: Claude Model Baselines on SWE-bench Lite 50-issue subset.

Runs OpenCode (or Claude Code) with Claude models via Bedrock against the same
50 issues used in agent-harness experiments. Captures diffs, behavioral metrics,
and optionally runs gold tests.

Usage:
  # Run Haiku × OpenCode on all 50 issues
  python3 run_baseline.py --model haiku --harness opencode

  # Run on a single issue (for testing)
  python3 run_baseline.py --model haiku --harness opencode --issue django__django-10914

  # Run with gold eval (requires Python 3.11 for some repos)
  python3 run_baseline.py --model haiku --harness opencode --gold-eval

  # Resume from a partial run
  python3 run_baseline.py --model haiku --harness opencode --resume

Environment:
  AWS_REGION          - Bedrock region (default: us-east-1)
  OPENCODE_BIN        - Path to opencode binary (default: ~/.opencode/bin/opencode)
  WORKSPACE_BASE      - Base directory for repo clones (default: /tmp/swebench-workspaces)
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBSET_SEED = 42
SUBSET_SIZE = 50

BEDROCK_MODELS = {
    "haiku": "amazon-bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "amazon-bedrock/anthropic.claude-sonnet-4-6",
    "opus": "amazon-bedrock/anthropic.claude-opus-4-6-v1",
}

BEDROCK_MODEL_IDS = {
    "haiku": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "anthropic.claude-sonnet-4-6",
    "opus": "anthropic.claude-opus-4-6-v1",
}

# Per-repo setup commands (from harness_eval.py)
_REPO_SETUP = {
    "django/django": "pip install -e . -q",
    "pytest-dev/pytest": "pip install -e '.[testing]' -q",
    "sympy/sympy": "pip install -e . -q",
    "scikit-learn/scikit-learn": "pip install -e . -q --no-build-isolation",
    "matplotlib/matplotlib": "pip install -e . -q",
    "pallets/flask": "pip install -e '.[dev]' -q",
    "pydata/xarray": "pip install -e . -q",
    "astropy/astropy": "pip install -e . -q",
    "sphinx-doc/sphinx": "pip install -e '.[test]' -q",
    "mwaskom/seaborn": "pip install -e '.[dev]' -q",
    "pylint-dev/pylint": "pip install -e . -q",
    "psf/requests": "pip install -e '.[dev]' -q",
}

OPENCODE_BIN = os.environ.get("OPENCODE_BIN", os.path.expanduser("~/.opencode/bin/opencode"))
WORKSPACE_BASE = os.environ.get("WORKSPACE_BASE", "/tmp/swebench-workspaces")

# ---------------------------------------------------------------------------
# Prompt Variants (Phase 2b: Adversarial Self-Critique)
# ---------------------------------------------------------------------------

PROMPT_VARIANTS = {
    "control": "",  # Phase 1 baseline — no addition
    "self-critique": (
        "\n\nAfter writing your fix, review it critically: assume the patch is wrong "
        "and try to find a bug. If you find one, fix it before finishing."
    ),
    "self-critique-strong": (
        "\n\nIMPORTANT: Before you finish, you MUST do a self-review. "
        "Assume your patch is incorrect. Try to construct an input that would make "
        "the patched code fail. If you find a plausible failure, fix the patch. "
        "Only finish when you cannot break your own fix."
    ),
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    test_cmd: str
    hints: str = ""


@dataclass
class RunResult:
    instance_id: str
    model: str
    harness: str
    fix_generated: bool = False
    tests_pass: Optional[bool] = None  # None = not evaluated
    turns_used: int = 0
    tokens_total: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_read: int = 0
    tokens_cache_write: int = 0
    cost_usd: float = 0.0
    total_latency_s: float = 0.0
    has_edit: bool = False
    tool_calls: int = 0
    error: Optional[str] = None
    patch_diff: Optional[str] = None


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

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


def load_subset(seed: int = SUBSET_SEED, size: int = SUBSET_SIZE) -> list[Issue]:
    """Load a deterministic subset of SWE-bench Lite."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install: pip install datasets")
        sys.exit(1)

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    all_issues = []
    for row in ds:
        all_issues.append(Issue(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            test_patch=row.get("test_patch", ""),
            test_cmd=_build_test_cmd(row),
            hints=row.get("hints_text", ""),
        ))

    rng = random.Random(seed)
    by_repo = {}
    for issue in all_issues:
        by_repo.setdefault(issue.repo, []).append(issue)

    selected = []
    repos = sorted(by_repo.keys())
    rng.shuffle(repos)
    idx = {r: 0 for r in repos}
    while len(selected) < size:
        for repo in repos:
            issues = by_repo[repo]
            if idx[repo] < len(issues) and len(selected) < size:
                selected.append(issues[idx[repo]])
                idx[repo] += 1

    log.info(f"Selected {len(selected)} issues across {len(set(i.repo for i in selected))} repos")
    return selected


# ---------------------------------------------------------------------------
# Workspace Setup
# ---------------------------------------------------------------------------

def setup_workspace(issue: Issue, base_dir: str) -> str:
    workspace = os.path.join(base_dir, issue.instance_id)
    repo_cache = os.path.join(base_dir, "_repo_cache", issue.repo.replace("/", "__"))

    if not os.path.isdir(repo_cache):
        os.makedirs(os.path.dirname(repo_cache), exist_ok=True)
        log.info(f"[{issue.instance_id}] Cloning {issue.repo}...")
        proc = subprocess.run(
            f"git clone --depth 1000 https://github.com/{issue.repo}.git {repo_cache}",
            shell=True, capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Clone failed: {proc.stderr[:500]}")

    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    shutil.copytree(repo_cache, workspace, symlinks=True)

    proc = subprocess.run(
        f"git checkout -f {issue.base_commit}",
        shell=True, capture_output=True, text=True,
        timeout=30, cwd=workspace,
    )
    if proc.returncode != 0:
        subprocess.run(
            f"git fetch --unshallow origin && git checkout -f {issue.base_commit}",
            shell=True, capture_output=True, text=True,
            timeout=120, cwd=workspace,
        )

    # Apply test patch (so gold tests exist in the workspace)
    if issue.test_patch:
        proc = subprocess.run(
            "git apply --allow-empty",
            shell=True, input=issue.test_patch, capture_output=True, text=True,
            timeout=10, cwd=workspace,
        )
        if proc.returncode == 0:
            subprocess.run(
                "git add -A && git commit -m 'apply test patch' --allow-empty",
                shell=True, capture_output=True, text=True,
                timeout=10, cwd=workspace,
            )

    return workspace


# ---------------------------------------------------------------------------
# OpenCode Runner
# ---------------------------------------------------------------------------

def run_opencode(issue: Issue, workspace: str, model_key: str,
                  prompt_variant: str = "control") -> RunResult:
    """Run OpenCode with a Bedrock Claude model on a single issue."""
    result = RunResult(
        instance_id=issue.instance_id,
        model=model_key,
        harness="opencode",
    )
    start = time.monotonic()

    bedrock_model = BEDROCK_MODELS[model_key]
    events_file = f"/tmp/opencode_events_{issue.instance_id}.json"

    variant_suffix = PROMPT_VARIANTS.get(prompt_variant, "")
    prompt = (
        f"Fix this bug in the repo at {workspace}.\n\n"
        f"{issue.problem_statement[:8000]}\n\n"
        "IMPORTANT: Edit the source file immediately after reading the relevant code. "
        "Do not read more than 3 files before making your edit. "
        "Use the edit tool, not bash, to modify files."
        f"{variant_suffix}"
    )

    env = os.environ.copy()
    env["AWS_REGION"] = env.get("AWS_REGION", "us-east-1")

    try:
        proc = subprocess.run(
            [
                OPENCODE_BIN, "run",
                "--model", bedrock_model,
                "--format", "json",
                "--dir", workspace,
                prompt,
            ],
            capture_output=True, text=True,
            timeout=600,  # 10 min per issue
            env=env,
        )

        # Parse JSON events from stdout
        for line in proc.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue

            evt_type = evt.get("type", "")

            if evt_type == "step_finish":
                part = evt.get("part", {})
                tokens = part.get("tokens", {})
                result.tokens_total += tokens.get("total", 0)
                result.tokens_input += tokens.get("input", 0)
                result.tokens_output += tokens.get("output", 0)
                cache = tokens.get("cache", {})
                result.tokens_cache_read += cache.get("read", 0)
                result.tokens_cache_write += cache.get("write", 0)
                result.cost_usd += part.get("cost", 0)
                result.turns_used += 1

            elif evt_type == "tool_use":
                part = evt.get("part", {})
                tool_name = part.get("tool", "")
                result.tool_calls += 1
                if tool_name == "edit":
                    result.has_edit = True

            elif evt_type == "error":
                error_data = evt.get("error", {})
                if isinstance(error_data, dict):
                    result.error = error_data.get("data", {}).get("message", "unknown")[:500]
                else:
                    result.error = str(error_data)[:500]

        # Save raw events for later analysis
        events_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results", "events",
        )
        os.makedirs(events_dir, exist_ok=True)
        variant_tag = f"_{prompt_variant}" if prompt_variant != "control" else ""
        with open(os.path.join(events_dir, f"opencode_{model_key}{variant_tag}_{issue.instance_id}.jsonl"), "w") as f:
            f.write(proc.stdout)

    except subprocess.TimeoutExpired:
        result.error = "timeout (600s)"
    except Exception as e:
        result.error = str(e)[:500]

    # Capture diff
    try:
        diff_proc = subprocess.run(
            "git diff HEAD", shell=True, capture_output=True, text=True,
            timeout=10, cwd=workspace,
        )
        diff = diff_proc.stdout.strip()
        if diff:
            result.fix_generated = True
            result.patch_diff = diff[:50000]

            # Save diff to file
            variant_tag = f"_{prompt_variant}" if prompt_variant != "control" else ""
            diffs_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "results", "diffs", f"opencode_{model_key}{variant_tag}",
            )
            os.makedirs(diffs_dir, exist_ok=True)
            with open(os.path.join(diffs_dir, f"{issue.instance_id}.diff"), "w") as f:
                f.write(diff)
    except Exception:
        pass

    result.total_latency_s = time.monotonic() - start
    return result


# ---------------------------------------------------------------------------
# Gold Evaluation
# ---------------------------------------------------------------------------

def run_gold_eval(issue: Issue, workspace: str) -> Optional[bool]:
    """Run the gold test command. Returns True/False/None (if eval fails)."""
    try:
        venv_activate = os.path.join(workspace, ".venv", "bin", "activate")
        if os.path.exists(venv_activate):
            cmd = f"source {venv_activate} && {issue.test_cmd}"
        else:
            cmd = issue.test_cmd

        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=120, cwd=workspace, executable="/bin/bash",
        )
        output = (proc.stdout + "\n" + proc.stderr).lower()

        if "passed" in output:
            before_passed = output.split("passed")[0]
            if "failed" not in before_passed and "error" not in before_passed:
                return True
        if "\nok" in output or output.rstrip().endswith("ok"):
            if "fail" not in output and "error" not in output:
                return True
        return False

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Claude model baselines on SWE-bench Lite")
    parser.add_argument("--model", choices=["haiku", "sonnet", "opus"], default="haiku")
    parser.add_argument("--harness", choices=["opencode", "claude-code"], default="opencode")
    parser.add_argument("--prompt-variant", choices=list(PROMPT_VARIANTS.keys()), default="control",
                        help="Prompt variant for Phase 2b adversarial self-critique experiment")
    parser.add_argument("--issue", type=str, help="Run a single issue by instance_id")
    parser.add_argument("--gold-eval", action="store_true", help="Run gold test evaluation")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed issues")
    parser.add_argument("--workspace-base", type=str, default=WORKSPACE_BASE)
    parser.add_argument("--limit", type=int, help="Limit to first N issues")
    args = parser.parse_args()

    # Output file
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    variant_tag = f"_{args.prompt_variant}" if args.prompt_variant != "control" else ""
    output_file = os.path.join(results_dir, f"baseline_{args.model}_{args.harness}{variant_tag}.jsonl")

    # Load completed issues for resume
    completed = set()
    if args.resume and os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    completed.add(row["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        log.info(f"Resuming: {len(completed)} issues already completed")

    # Load issues
    log.info("Loading SWE-bench Lite subset...")
    issues = load_subset()

    if args.issue:
        issues = [i for i in issues if i.instance_id == args.issue]
        if not issues:
            log.error(f"Issue {args.issue} not in 50-issue subset")
            sys.exit(1)

    if args.limit:
        issues = issues[:args.limit]

    # Run
    os.makedirs(args.workspace_base, exist_ok=True)
    total = len(issues)
    passed = 0
    fixed = 0
    total_cost = 0.0

    for idx, issue in enumerate(issues):
        if issue.instance_id in completed:
            log.info(f"[{idx+1}/{total}] SKIP {issue.instance_id} (already done)")
            continue

        log.info(f"[{idx+1}/{total}] {issue.instance_id} ({issue.repo})")

        try:
            workspace = setup_workspace(issue, args.workspace_base)
        except Exception as e:
            log.error(f"  Workspace setup failed: {e}")
            result = RunResult(
                instance_id=issue.instance_id,
                model=args.model,
                harness=args.harness,
                error=f"workspace_setup: {e}",
            )
            with open(output_file, "a") as f:
                row = asdict(result)
                row.pop("patch_diff", None)  # Don't inline huge diffs in JSONL
                f.write(json.dumps(row) + "\n")
            continue

        if args.harness == "opencode":
            result = run_opencode(issue, workspace, args.model, args.prompt_variant)
        else:
            log.error(f"Harness {args.harness} not yet implemented")
            continue

        # Gold eval
        if args.gold_eval and result.fix_generated:
            result.tests_pass = run_gold_eval(issue, workspace)

        # Stats
        if result.fix_generated:
            fixed += 1
        if result.tests_pass:
            passed += 1
        total_cost += result.cost_usd

        # Log result
        status = "PASS" if result.tests_pass else ("FIX" if result.fix_generated else "FAIL")
        log.info(
            f"  {status} | turns={result.turns_used} tools={result.tool_calls} "
            f"edit={result.has_edit} tokens={result.tokens_total} "
            f"cost=${result.cost_usd:.4f} time={result.total_latency_s:.1f}s"
        )
        if result.error:
            log.warning(f"  ERROR: {result.error}")

        # Write result (without patch_diff in summary JSONL)
        with open(output_file, "a") as f:
            row = asdict(result)
            row.pop("patch_diff", None)
            f.write(json.dumps(row) + "\n")

        # Cleanup workspace to save disk
        try:
            shutil.rmtree(workspace)
        except Exception:
            pass

    # Summary
    done = len(issues) - len(completed)
    log.info(f"\n{'='*60}")
    log.info(f"Model: {args.model} | Harness: {args.harness} | Variant: {args.prompt_variant}")
    log.info(f"Issues: {done} | Fixed: {fixed} ({100*fixed/max(done,1):.0f}%) | Cost: ${total_cost:.2f}")
    if args.gold_eval:
        log.info(f"Passed: {passed} ({100*passed/max(done,1):.0f}%)")
    log.info(f"Results: {output_file}")


if __name__ == "__main__":
    main()
