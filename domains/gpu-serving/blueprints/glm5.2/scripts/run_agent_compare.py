#!/usr/bin/env python3
"""
GLM-5.2-FP8 agent-capability comparison on SWE-bench Lite.

3 harnesses (OpenCode, Codex, Claude Code) x N issues, all driving the
self-hosted GLM-5.2-FP8 SGLang endpoint (localhost:30000). Codex + Claude Code
go via the in-pod LiteLLM proxy (localhost:4000); OpenCode hits SGLang directly.

Forked from verifier-reward/run_baseline.py — identical trace schema.

Usage (run IN the runner pod):
  python3 run_agent_compare.py --harness all --model glm52 \
      --issues-file /cfg/matched-issues-46.txt --gold-eval \
      --issue django__django-10914 --issue sympy__sympy-11400 --issue psf__requests-2317

Environment:
  OPENCODE_BIN   - opencode binary (default: search PATH / ~/.opencode/bin/opencode)
  CODEX_BIN      - codex binary (default: codex)
  CLAUDE_BIN     - claude binary (default: claude)
  WORKSPACE_BASE - repo clone base (default: /tmp/swebench-workspaces)
  RESULTS_DIR    - trace output dir (default: /mnt/nvme/results/agent-compare)
  LITELLM_BASE   - LiteLLM base (default: http://127.0.0.1:4000)
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBSET_SEED = 42
SUBSET_SIZE = 50

# Model registry. Each entry knows how each harness should address GLM-5.2.
MODELS = {
    "glm52": {
        # OpenCode talks to SGLang chat-completions directly via the vllm provider.
        "opencode": "vllm/GLM-5.2-FP8",
        # Codex + Claude Code go via LiteLLM, which exposes model "glm-5.2".
        "litellm_model": "glm-5.2",
    },
}

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


def _find_opencode() -> str:
    cand = os.environ.get("OPENCODE_BIN")
    if cand and os.path.exists(cand):
        return cand
    for p in (
        os.path.expanduser("~/.opencode/bin/opencode"),
        "/root/.opencode/bin/opencode",
        shutil.which("opencode"),
    ):
        if p and os.path.exists(p):
            return p
    return "opencode"


OPENCODE_BIN = _find_opencode()
CODEX_BIN = os.environ.get("CODEX_BIN", shutil.which("codex") or "codex")
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", shutil.which("claude") or "claude")
WORKSPACE_BASE = os.environ.get("WORKSPACE_BASE", "/tmp/swebench-workspaces")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/mnt/nvme/results/agent-compare")
LITELLM_BASE = os.environ.get("LITELLM_BASE", "http://127.0.0.1:4000")

PROMPT_TAIL = (
    "IMPORTANT: Edit the source file immediately after reading the relevant code. "
    "Do not read more than 3 files before making your edit. "
    "Use the edit tool, not bash, to modify files."
)

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
    tests_pass: Optional[bool] = None
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


def load_subset(seed: int = SUBSET_SEED, size: int = SUBSET_SIZE,
                issues_file: Optional[str] = None) -> list[Issue]:
    """Load SWE-bench Lite issues.

    If issues_file is given, restrict to those instance_ids (full dataset scan,
    no seeded subsampling). Otherwise use the deterministic 50-issue subset.
    """
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

    if issues_file:
        wanted = set()
        with open(issues_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    wanted.add(line)
        selected = [i for i in all_issues if i.instance_id in wanted]
        missing = wanted - {i.instance_id for i in selected}
        if missing:
            log.warning(f"{len(missing)} issues from file not found in dataset: {sorted(missing)[:5]}")
        log.info(f"Selected {len(selected)} issues from {issues_file}")
        return selected

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
            f"git clone https://github.com/{issue.repo}.git {repo_cache}",
            shell=True, capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Clone failed: {proc.stderr[:500]}")

    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    shutil.copytree(repo_cache, workspace, symlinks=True)

    proc = subprocess.run(
        f"git checkout -f {issue.base_commit}",
        shell=True, capture_output=True, text=True, timeout=60, cwd=workspace,
    )
    if proc.returncode != 0:
        subprocess.run(
            f"git fetch --unshallow origin && git checkout -f {issue.base_commit}",
            shell=True, capture_output=True, text=True, timeout=300, cwd=workspace,
        )

    if issue.test_patch:
        proc = subprocess.run(
            "git apply --allow-empty",
            shell=True, input=issue.test_patch, capture_output=True, text=True,
            timeout=10, cwd=workspace,
        )
        if proc.returncode == 0:
            subprocess.run(
                "git add -A && git commit -m 'apply test patch' --allow-empty",
                shell=True, capture_output=True, text=True, timeout=10, cwd=workspace,
            )

    return workspace


# ---------------------------------------------------------------------------
# Trace + diff persistence (shared)
# ---------------------------------------------------------------------------

def _save_trace(harness: str, model_key: str, issue: Issue, raw: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{harness}_{model_key}_{issue.instance_id}.jsonl")
    with open(path, "w") as f:
        f.write(raw or "")


def _capture_diff(issue: Issue, workspace: str, result: RunResult, harness: str, model_key: str):
    try:
        diff_proc = subprocess.run(
            "git diff HEAD", shell=True, capture_output=True, text=True,
            timeout=10, cwd=workspace,
        )
        diff = diff_proc.stdout.strip()
        if diff:
            result.fix_generated = True
            result.patch_diff = diff[:50000]
            diffs_dir = os.path.join(RESULTS_DIR, "diffs", f"{harness}_{model_key}")
            os.makedirs(diffs_dir, exist_ok=True)
            with open(os.path.join(diffs_dir, f"{issue.instance_id}.diff"), "w") as f:
                f.write(diff)
    except Exception:
        pass


def _prompt(issue: Issue, workspace: str) -> str:
    return (
        f"Fix this bug in the repo at {workspace}.\n\n"
        f"{issue.problem_statement[:8000]}\n\n"
        f"{PROMPT_TAIL}"
    )


# ---------------------------------------------------------------------------
# OpenCode Runner (direct to SGLang)
# ---------------------------------------------------------------------------

def run_opencode(issue: Issue, workspace: str, model_key: str) -> RunResult:
    result = RunResult(instance_id=issue.instance_id, model=model_key, harness="opencode")
    start = time.monotonic()
    oc_model = MODELS[model_key]["opencode"]
    prompt = _prompt(issue, workspace)

    env = os.environ.copy()
    raw = ""
    try:
        proc = subprocess.run(
            [OPENCODE_BIN, "run", "--model", oc_model, "--format", "json",
             "--dir", workspace, prompt],
            capture_output=True, text=True, timeout=900, env=env,
        )
        raw = proc.stdout
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
                result.tool_calls += 1
                if part.get("tool", "") == "edit":
                    result.has_edit = True
            elif evt_type == "error":
                error_data = evt.get("error", {})
                if isinstance(error_data, dict):
                    result.error = error_data.get("data", {}).get("message", "unknown")[:500]
                else:
                    result.error = str(error_data)[:500]
        if proc.returncode != 0 and not result.error:
            result.error = f"exit={proc.returncode}: {proc.stderr[:400]}"
    except subprocess.TimeoutExpired:
        result.error = "timeout (900s)"
    except Exception as e:
        result.error = str(e)[:500]

    _save_trace("opencode", model_key, issue, raw)
    _capture_diff(issue, workspace, result, "opencode", model_key)
    result.total_latency_s = time.monotonic() - start
    return result


# ---------------------------------------------------------------------------
# Codex Runner (via LiteLLM /v1/responses)
# ---------------------------------------------------------------------------

def run_codex(issue: Issue, workspace: str, model_key: str) -> RunResult:
    result = RunResult(instance_id=issue.instance_id, model=model_key, harness="codex")
    start = time.monotonic()
    litellm_model = MODELS[model_key]["litellm_model"]
    prompt = _prompt(issue, workspace)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "dummy")
    raw = ""
    try:
        with open(os.devnull) as devnull:
            proc = subprocess.run(
                [CODEX_BIN, "exec",
                 "--model", litellm_model,
                 "--config", "model_provider=glm52",
                 "--cd", workspace,
                 "--skip-git-repo-check",
                 prompt],
                stdin=devnull, capture_output=True, text=True,
                timeout=900, env=env,
            )
        raw = proc.stdout + "\n---STDERR---\n" + proc.stderr
        # Codex exec emits human-readable logs; parse tool/turn signals heuristically.
        out = proc.stdout
        # Count tool invocations: codex logs "exec" / "apply_patch" / "tool" lines.
        result.tool_calls = len(re.findall(r"(?im)^\s*(tool|exec|apply_patch|function_call)\b", out))
        # Turns: count assistant message / "codex" turn markers.
        result.turns_used = len(re.findall(r"(?im)^\s*(codex|assistant)\b", out)) or (1 if out.strip() else 0)
        if re.search(r"apply_patch|edit_file|\bwrite\b", out, re.I):
            result.has_edit = True
        if proc.returncode != 0:
            result.error = f"exit={proc.returncode}: {proc.stderr[:400]}"
    except subprocess.TimeoutExpired:
        result.error = "timeout (900s)"
    except FileNotFoundError:
        result.error = f"codex binary not found: {CODEX_BIN}"
    except Exception as e:
        result.error = str(e)[:500]

    _save_trace("codex", model_key, issue, raw)
    _capture_diff(issue, workspace, result, "codex", model_key)
    # has_edit fallback: if a diff was produced, an edit happened.
    if result.fix_generated:
        result.has_edit = True
    result.total_latency_s = time.monotonic() - start
    return result


# ---------------------------------------------------------------------------
# Claude Code Runner (via LiteLLM /v1/messages)
# ---------------------------------------------------------------------------

CLAUDE_SETTINGS_PATH = "/tmp/claude-glm52-settings.json"


def _write_claude_settings():
    settings = {
        "env": {
            "CLAUDE_CODE_USE_BEDROCK": "0",
            "ANTHROPIC_BASE_URL": LITELLM_BASE,
            "ANTHROPIC_API_KEY": "dummy",
            "ANTHROPIC_MODEL": "glm-5.2",
            "ANTHROPIC_SMALL_FAST_MODEL": "glm-5.2",
        }
    }
    with open(CLAUDE_SETTINGS_PATH, "w") as f:
        json.dump(settings, f)
    return CLAUDE_SETTINGS_PATH


def run_claude_code(issue: Issue, workspace: str, model_key: str) -> RunResult:
    result = RunResult(instance_id=issue.instance_id, model=model_key, harness="claude-code")
    start = time.monotonic()
    settings = _write_claude_settings()
    prompt = _prompt(issue, workspace)

    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = "dummy"
    env["IS_SANDBOX"] = "1"  # allow --dangerously-skip-permissions under root (in-pod runner)
    raw = ""
    try:
        with open(os.devnull) as devnull:
            proc = subprocess.run(
                [CLAUDE_BIN, "-p", prompt,
                 "--settings", settings,
                 "--output-format", "json",
                 "--max-turns", "15",
                 "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep",
                 "--dangerously-skip-permissions"],
                stdin=devnull, capture_output=True, text=True,
                timeout=900, env=env, cwd=workspace,
            )
        raw = proc.stdout + "\n---STDERR---\n" + proc.stderr
        parsed = None
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError:
            # try last JSON object on stdout
            for line in reversed(proc.stdout.strip().split("\n")):
                try:
                    parsed = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if isinstance(parsed, dict):
            result.turns_used = parsed.get("num_turns", 0)
            usage = parsed.get("usage", {}) or {}
            result.tokens_input += usage.get("input_tokens", 0)
            result.tokens_output += usage.get("output_tokens", 0)
            result.tokens_cache_read += usage.get("cache_read_input_tokens", 0)
            result.tokens_cache_write += usage.get("cache_creation_input_tokens", 0)
            result.tokens_total += (result.tokens_input + result.tokens_output)
            result.cost_usd += parsed.get("total_cost_usd", 0.0)
            if parsed.get("is_error"):
                result.error = (str(parsed.get("result", ""))[:500]) or "is_error"
        # Detect thinking-block 400 known limitation
        if re.search(r"thinking.*content|invalid.*thinking|400", raw, re.I) and "thinking" in raw.lower():
            if not result.error:
                result.error = "possible thinking-block 400 (LiteLLM Anthropic adapter)"
        if proc.returncode != 0 and not result.error:
            result.error = f"exit={proc.returncode}: {proc.stderr[:400]}"
    except subprocess.TimeoutExpired:
        result.error = "timeout (900s)"
    except FileNotFoundError:
        result.error = f"claude binary not found: {CLAUDE_BIN}"
    except Exception as e:
        result.error = str(e)[:500]

    _save_trace("claude-code", model_key, issue, raw)
    _capture_diff(issue, workspace, result, "claude-code", model_key)
    if result.fix_generated:
        result.has_edit = True
        # crude tool_calls floor: at least the edit
        if result.tool_calls == 0:
            result.tool_calls = 1
    result.total_latency_s = time.monotonic() - start
    return result


HARNESS_RUNNERS = {
    "opencode": run_opencode,
    "codex": run_codex,
    "claude-code": run_claude_code,
}


# ---------------------------------------------------------------------------
# Gold Evaluation
# ---------------------------------------------------------------------------

def run_gold_eval(issue: Issue, workspace: str) -> Optional[bool]:
    try:
        venv_activate = os.path.join(workspace, ".venv", "bin", "activate")
        if os.path.exists(venv_activate):
            cmd = f"source {venv_activate} && {issue.test_cmd}"
        else:
            cmd = issue.test_cmd
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=180, cwd=workspace, executable="/bin/bash",
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
    parser = argparse.ArgumentParser(description="GLM-5.2 agent-harness comparison on SWE-bench Lite")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="glm52")
    parser.add_argument("--harness", choices=["opencode", "codex", "claude-code", "all"], default="all")
    parser.add_argument("--issue", action="append", default=[],
                        help="Run a single issue by instance_id (repeatable)")
    parser.add_argument("--issues-file", type=str, help="Restrict to instance_ids in this file")
    parser.add_argument("--gold-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workspace-base", type=str, default=WORKSPACE_BASE)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    harnesses = ["opencode", "codex", "claude-code"] if args.harness == "all" else [args.harness]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Per-harness output JSONL (resume-aware)
    out_files = {h: os.path.join(RESULTS_DIR, f"summary_{h}_{args.model}.jsonl") for h in harnesses}
    completed = {h: set() for h in harnesses}
    if args.resume:
        for h, of in out_files.items():
            if os.path.exists(of):
                with open(of) as f:
                    for line in f:
                        try:
                            completed[h].add(json.loads(line)["instance_id"])
                        except (json.JSONDecodeError, KeyError):
                            pass

    log.info("Loading SWE-bench Lite...")
    issues = load_subset(issues_file=args.issues_file)

    if args.issue:
        wanted = set(args.issue)
        issues = [i for i in issues if i.instance_id in wanted]
        missing = wanted - {i.instance_id for i in issues}
        if missing:
            log.error(f"Issues not found: {sorted(missing)}")
            sys.exit(1)
    if args.limit:
        issues = issues[:args.limit]

    log.info(f"Harnesses: {harnesses} | Issues: {len(issues)}")

    os.makedirs(args.workspace_base, exist_ok=True)
    all_results = []

    for idx, issue in enumerate(issues):
        log.info(f"[{idx+1}/{len(issues)}] {issue.instance_id} ({issue.repo})")
        for h in harnesses:
            if issue.instance_id in completed[h]:
                log.info(f"  SKIP {h} (already done)")
                continue
            # Fresh workspace per harness (each harness mutates the tree)
            try:
                workspace = setup_workspace(issue, args.workspace_base)
            except Exception as e:
                log.error(f"  Workspace setup failed: {e}")
                result = RunResult(instance_id=issue.instance_id, model=args.model,
                                   harness=h, error=f"workspace_setup: {e}")
                _write_result(out_files[h], result)
                all_results.append(result)
                continue

            log.info(f"  -> {h}")
            runner = HARNESS_RUNNERS[h]
            result = runner(issue, workspace, args.model)

            if args.gold_eval and result.fix_generated:
                result.tests_pass = run_gold_eval(issue, workspace)

            status = "PASS" if result.tests_pass else ("FIX" if result.fix_generated else "FAIL")
            log.info(f"     {status} | turns={result.turns_used} tools={result.tool_calls} "
                     f"edit={result.has_edit} time={result.total_latency_s:.0f}s "
                     f"{'ERR:'+result.error if result.error else ''}")

            _write_result(out_files[h], result)
            all_results.append(result)

            try:
                shutil.rmtree(workspace)
            except Exception:
                pass

    # Summary JSON
    summary_path = os.path.join(RESULTS_DIR, f"pilot_summary_{args.model}.json")
    summary = {
        "model": args.model,
        "harnesses": harnesses,
        "issues": [i.instance_id for i in issues],
        "results": [
            {k: v for k, v in asdict(r).items() if k != "patch_diff"}
            for r in all_results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSummary written: {summary_path}")
    _print_table(all_results)


def _write_result(out_file: str, result: RunResult):
    with open(out_file, "a") as f:
        row = asdict(result)
        row.pop("patch_diff", None)
        f.write(json.dumps(row) + "\n")


def _print_table(results: list):
    log.info("\n" + "=" * 100)
    hdr = f"{'instance_id':<28}{'harness':<13}{'fix':<5}{'edit':<6}{'tools':<7}{'turns':<7}{'pass':<6}{'error'}"
    log.info(hdr)
    log.info("-" * 100)
    for r in results:
        log.info(
            f"{r.instance_id:<28}{r.harness:<13}{str(r.fix_generated):<5}{str(r.has_edit):<6}"
            f"{r.tool_calls:<7}{r.turns_used:<7}{str(r.tests_pass):<6}{(r.error or '')[:30]}"
        )


if __name__ == "__main__":
    main()
