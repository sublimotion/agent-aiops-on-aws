#!/usr/bin/env python3
"""
Phase 1: Multi-Model × Multi-Harness Evaluation for Agent Swarm Experiment.

Runs 4 models × 3 harnesses against the same SWE-bench Lite 50-issue subset
to measure how model capability affects harness spread, turn budget, and precision.

Reuses adapter scripts and evaluation infrastructure from agent-harness blueprint.

Usage:
  # Run one model × one harness
  python3 swarm_eval.py --model devstral-small-2 --harness opencode \
      --endpoint http://localhost:8000

  # Run full matrix (all models × all harnesses)
  python3 swarm_eval.py --run-all --endpoint http://localhost:8000

  # Generate comparative leaderboard from results
  python3 swarm_eval.py --report results/swarm_phase1_*.jsonl

  # Check what's installed
  python3 swarm_eval.py --check
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Import shared infrastructure from agent-harness
# On g7e: harness_eval.py is at /mnt/nvme/agent-harness-scripts/
# Locally: it's in the agent-harness blueprint
_REMOTE_SCRIPTS = Path("/mnt/nvme/agent-harness-scripts")
_LOCAL_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "agent-harness" / "scripts"
HARNESS_SCRIPTS = _REMOTE_SCRIPTS if _REMOTE_SCRIPTS.exists() else _LOCAL_SCRIPTS
sys.path.insert(0, str(HARNESS_SCRIPTS))
from harness_eval import Issue, load_subset, setup_workspace, _get_git_diff

# Adapter scripts directory (on g7e, deployed separately)
REMOTE_ADAPTERS = Path("/mnt/nvme/agent-harness/adapters")
LOCAL_ADAPTERS = _LOCAL_SCRIPTS / "adapters"

# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

MODELS = {
    "devstral-small-2": {
        "name": "Devstral Small 2 24B FP8",
        "hf_id": "mistralai/Devstral-Small-2501",
        "weights_path": "/mnt/nvme/models/devstral-small-2-fp8",
        "tp": 1,
        "context": 65536,
        "vllm_args": [
            "--tool-call-parser", "mistral",
            "--enable-auto-tool-choice",
            "--enable-prefix-caching",
        ],
        "chat_template": "mistral",
        "role": "baseline",
    },
    "qwen25-coder-32b": {
        "name": "Qwen 2.5 Coder 32B Instruct",
        "hf_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "weights_path": "/mnt/nvme/models/qwen25-coder-32b",
        "tp": 1,
        "context": 65536,
        "vllm_args": [
            "--tool-call-parser", "hermes",
            "--enable-auto-tool-choice",
            "--enable-prefix-caching",
        ],
        "chat_template": "qwen",
        "role": "finetuning-control",
    },
    "swesmith-qwen25-32b": {
        "name": "SWE-agent-LM 32B (SERA-finetuned)",
        "hf_id": "SWE-bench/SWE-agent-LM-32B",
        "weights_path": "/mnt/nvme/models/swe-agent-lm-32b",
        "tp": 1,
        "context": 65536,
        "vllm_args": [
            "--tool-call-parser", "hermes",
            "--enable-auto-tool-choice",
            "--enable-prefix-caching",
        ],
        "chat_template": "qwen",
        "role": "finetuning-after",
    },
    "qwen35-397b": {
        "name": "Qwen3.5-397B-A17B FP8",
        "hf_id": "Qwen/Qwen3.5-72B-A22B",  # placeholder — actual is on NVMe
        "weights_path": "/mnt/nvme/models/qwen3-next-fp8",
        "tp": 4,
        "context": 65536,
        "vllm_args": [
            "--tool-call-parser", "hermes",
            "--enable-auto-tool-choice",
            "--enable-prefix-caching",
        ],
        "chat_template": "qwen",
        "role": "scale-frontier",
    },
}

# Phase 1 harnesses (subset of the 7 already tested)
HARNESSES = ["opencode", "claude-code", "sera"]

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SwarmResult:
    """Result for one issue under one model × harness."""
    instance_id: str
    model: str
    harness: str
    tests_pass: bool = False
    fix_generated: bool = False
    turns_used: int = 0
    tokens_consumed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_ms: float = 0
    first_edit_turn: int = 0
    error: Optional[str] = None
    # Trajectory data (for verifier training)
    patch_diff: Optional[str] = None

# ---------------------------------------------------------------------------
# vLLM Server Management
# ---------------------------------------------------------------------------

def get_vllm_launch_cmd(model_id: str, port: int = 8000) -> list[str]:
    """Generate vLLM launch command for a model."""
    model = MODELS[model_id]
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model["weights_path"],
        "--served-model-name", model_id,
        "--tensor-parallel-size", str(model["tp"]),
        "--max-model-len", str(model["context"]),
        "--port", str(port),
        "--dtype", "auto",
        "--trust-remote-code",
    ]
    cmd.extend(model["vllm_args"])
    return cmd


def check_vllm_health(endpoint: str) -> Optional[str]:
    """Check if vLLM is serving and return the model name."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{endpoint}/v1/models", timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            return models[0] if models else None
    except Exception:
        return None


def wait_for_vllm(endpoint: str, timeout: int = 300) -> bool:
    """Wait for vLLM to become healthy."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        model = check_vllm_health(endpoint)
        if model:
            log.info(f"vLLM healthy — serving {model}")
            return True
        time.sleep(5)
    return False

# ---------------------------------------------------------------------------
# Harness Runners
# ---------------------------------------------------------------------------

def get_adapter_path(harness_id: str) -> Optional[Path]:
    """Find adapter script, preferring remote (deployed) over local."""
    script_name = f"run_{harness_id.replace('-', '_')}.sh"
    for base in [REMOTE_ADAPTERS, LOCAL_ADAPTERS]:
        path = base / script_name
        if path.exists():
            return path
    return None


def run_adapter(
    harness_id: str,
    model_id: str,
    issue: Issue,
    workspace: str,
    endpoint: str,
) -> SwarmResult:
    """Run an adapter script and parse results."""
    result = SwarmResult(
        instance_id=issue.instance_id,
        model=model_id,
        harness=harness_id,
    )
    start = time.monotonic()

    adapter = get_adapter_path(harness_id)
    if not adapter:
        result.error = f"Adapter not found: {harness_id}"
        return result

    env = os.environ.copy()
    env.update({
        "WORKSPACE": workspace,
        "ENDPOINT": endpoint,
        "MODEL": model_id,
        "ISSUE_ID": issue.instance_id,
        "PROBLEM_STATEMENT": issue.problem_statement[:10000],
        "TEST_CMD": issue.test_cmd,
        "REPO": issue.repo,
    })

    try:
        proc = subprocess.run(
            ["bash", str(adapter)],
            capture_output=True, text=True,
            timeout=600,
            cwd=workspace,
            env=env,
        )

        # Parse JSON from last non-empty line
        lines = [l.strip() for l in proc.stdout.strip().split("\n") if l.strip()]
        if lines:
            try:
                output = json.loads(lines[-1])
                result.tests_pass = output.get("pass", False)
                result.turns_used = output.get("turns", 0)
                result.tokens_consumed = output.get("tokens", 0)
                result.input_tokens = output.get("input_tokens", 0)
                result.output_tokens = output.get("output_tokens", 0)
                result.fix_generated = output.get("fix_generated", False)
                result.first_edit_turn = output.get("first_edit_turn", 0)
            except json.JSONDecodeError:
                result.error = f"Parse error: {lines[-1][:200]}"
        else:
            result.error = "No adapter output"

        if proc.returncode != 0 and not result.error:
            result.error = f"Exit {proc.returncode}: {proc.stderr[:300]}"

    except subprocess.TimeoutExpired:
        result.error = "Timeout (600s)"
    except Exception as e:
        result.error = str(e)[:500]

    result.total_latency_ms = (time.monotonic() - start) * 1000

    # Capture patch diff for verifier training
    diff = _get_git_diff(workspace)
    result.patch_diff = diff if diff else None
    if not result.fix_generated:
        result.fix_generated = bool(diff)

    return result


async def run_sera(
    model_id: str,
    issue: Issue,
    workspace: str,
    endpoint: str,
    output_dir: str = "results",
) -> SwarmResult:
    """Run SERA baseline (builtin agent loop) with Phase 1 config D."""
    from harness_eval import run_instrumented_loop, PHASE1_CONFIGS
    import aiohttp

    result = SwarmResult(
        instance_id=issue.instance_id,
        model=model_id,
        harness="sera",
    )
    start = time.monotonic()
    config = dict(PHASE1_CONFIGS["D"])  # 30 turns, strict
    config["_output_dir"] = output_dir  # enable trajectory saving

    try:
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            eval_result = await run_instrumented_loop(
                session, endpoint, model_id, issue, workspace,
                config, f"swarm_{model_id}_sera",
            )
            result.tests_pass = eval_result.tests_pass
            result.fix_generated = eval_result.fix_generated
            result.turns_used = eval_result.turns_used
            result.patch_diff = eval_result.patch_diff
            result.tokens_consumed = sum(
                tm.get("context_tokens", 0) for tm in eval_result.turn_metrics
            )
            # Find first edit turn
            for tm in eval_result.turn_metrics:
                if tm.get("edit_applied"):
                    result.first_edit_turn = tm.get("turn_number", 0)
                    break
    except Exception as e:
        result.error = str(e)[:500]

    result.total_latency_ms = (time.monotonic() - start) * 1000
    return result

# ---------------------------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------------------------

async def run_model_harness(
    model_id: str,
    harness_id: str,
    issues: list[Issue],
    endpoint: str,
    workspace_dir: str,
    output_path: str,
) -> dict:
    """Run one model × harness pair against all issues."""
    model = MODELS[model_id]
    log.info(f"=== {model['name']} × {harness_id} ({model['role']}) ===")

    results = []
    # Resume from partial results if they exist
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    results.append(SwarmResult(**json.loads(line)))
        if results:
            log.info(f"  Resuming from {len(results)} completed issues")
    completed_ids = {r.instance_id for r in results}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for i, issue in enumerate(issues):
        if issue.instance_id in completed_ids:
            log.info(f"  [{i+1}/{len(issues)}] {issue.instance_id} — skipped (already done)")
            continue

        log.info(f"  [{i+1}/{len(issues)}] {issue.instance_id}")
        workspace = None
        try:
            workspace = setup_workspace(issue, workspace_dir)
        except Exception as e:
            log.warning(f"  Workspace setup failed for {issue.instance_id}: {e}")
            result = SwarmResult(
                instance_id=issue.instance_id,
                model=model_id,
                harness=harness_id,
                error=f"workspace_setup: {str(e)[:400]}",
            )
            results.append(result)
            with open(output_path, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")
            continue

        try:
            if harness_id == "sera":
                result = await run_sera(model_id, issue, workspace, endpoint, output_dir=os.path.dirname(output_path) or "results")
            else:
                result = run_adapter(harness_id, model_id, issue, workspace, endpoint)
            results.append(result)
        except Exception as e:
            result = SwarmResult(
                instance_id=issue.instance_id,
                model=model_id,
                harness=harness_id,
                error=str(e)[:500],
            )
            results.append(result)
        finally:
            if workspace and os.path.exists(workspace):
                import shutil
                shutil.rmtree(workspace, ignore_errors=True)

        # Write incrementally (append)
        with open(output_path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

        # Log progress
        if (i + 1) % 10 == 0:
            n_pass = sum(1 for r in results if r.tests_pass)
            n_fix = sum(1 for r in results if r.fix_generated)
            log.info(f"  Progress: {n_pass} pass, {n_fix} fix after {i+1} issues")

    # Compute summary
    n = len(results)
    n_pass = sum(1 for r in results if r.tests_pass)
    n_fix = sum(1 for r in results if r.fix_generated)
    n_err = sum(1 for r in results if r.error)
    avg_turns = sum(r.turns_used for r in results) / max(n, 1)
    avg_tokens = sum(r.tokens_consumed for r in results) / max(n, 1)
    precision = n_pass / max(n_fix, 1) if n_fix else 0
    first_edits = [r.first_edit_turn for r in results if r.first_edit_turn > 0]
    avg_first_edit = sum(first_edits) / max(len(first_edits), 1) if first_edits else 0
    parkinsons = avg_first_edit / 30 if avg_first_edit else 0  # fraction of 30-turn budget

    summary = {
        "model": model_id,
        "model_name": model["name"],
        "model_role": model["role"],
        "harness": harness_id,
        "pass_rate": n_pass / max(n, 1),
        "passes": n_pass,
        "fixes": n_fix,
        "fix_rate": n_fix / max(n, 1),
        "precision": precision,
        "errors": n_err,
        "total": n,
        "avg_turns": avg_turns,
        "avg_tokens": avg_tokens,
        "avg_first_edit": avg_first_edit,
        "parkinsons_ratio": parkinsons,
    }

    log.info(
        f"=== {model['name']} × {harness_id} DONE ===\n"
        f"  Pass rate: {n_pass}/{n} ({100*summary['pass_rate']:.1f}%)\n"
        f"  Fix rate: {n_fix}/{n} ({100*summary['fix_rate']:.1f}%)\n"
        f"  Precision: {100*precision:.1f}%\n"
        f"  Avg turns: {avg_turns:.1f}\n"
        f"  Avg first edit: {avg_first_edit:.1f} (Parkinson's: {100*parkinsons:.0f}%)\n"
        f"  Output: {output_path}"
    )

    return summary

# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def generate_leaderboard(result_files: list[str]):
    """Generate 2D leaderboard from Phase 1 result files."""
    summaries = []

    for filepath in result_files:
        results = []
        with open(filepath) as f:
            for line in f:
                results.append(json.loads(line))

        if not results:
            continue

        model = results[0].get("model", "unknown")
        harness = results[0].get("harness", "unknown")
        n = len(results)
        n_pass = sum(1 for r in results if r.get("tests_pass"))
        n_fix = sum(1 for r in results if r.get("fix_generated"))
        precision = n_pass / max(n_fix, 1) if n_fix else 0
        avg_turns = sum(r.get("turns_used", 0) for r in results) / max(n, 1)
        avg_tokens = sum(r.get("tokens_consumed", 0) for r in results) / max(n, 1)
        first_edits = [r.get("first_edit_turn", 0) for r in results if r.get("first_edit_turn", 0) > 0]
        avg_first_edit = sum(first_edits) / max(len(first_edits), 1) if first_edits else 0

        summaries.append({
            "model": model,
            "harness": harness,
            "pass_rate": n_pass / max(n, 1),
            "passes": n_pass,
            "fixes": n_fix,
            "fix_rate": n_fix / max(n, 1),
            "precision": precision,
            "total": n,
            "avg_turns": avg_turns,
            "avg_tokens": avg_tokens,
            "avg_first_edit": avg_first_edit,
        })

    # Sort by model role order, then pass rate
    role_order = {"baseline": 0, "finetuning-control": 1, "finetuning-after": 2, "scale-frontier": 3}
    summaries.sort(key=lambda s: (
        role_order.get(MODELS.get(s["model"], {}).get("role", ""), 99),
        -s["pass_rate"],
    ))

    # Print 2D table
    print("\n" + "=" * 100)
    print("PHASE 1: MODEL × HARNESS LEADERBOARD")
    print("=" * 100)
    print(f"{'Model':<30} {'Harness':<14} {'Pass':<10} {'Fix':<10} {'Prec':<8} {'Turns':<8} {'1st Edit':<10} {'Tok/Issue':<10}")
    print("-" * 100)

    for s in summaries:
        model_name = MODELS.get(s["model"], {}).get("name", s["model"])[:28]
        print(
            f"{model_name:<30} {s['harness']:<14} "
            f"{s['passes']:>2}/{s['total']:<3} ({100*s['pass_rate']:>4.1f}%) "
            f"{s['fixes']:>2}/{s['total']:<3} ({100*s['fix_rate']:>4.1f}%) "
            f"{100*s['precision']:>5.1f}%  "
            f"{s['avg_turns']:>5.1f}  "
            f"{s['avg_first_edit']:>5.1f}     "
            f"{s['avg_tokens']:>8.0f}"
        )

    # Compute per-model aggregates
    print("\n" + "-" * 100)
    print("PER-MODEL SUMMARY (best harness)")
    print("-" * 100)

    models_seen = {}
    for s in summaries:
        m = s["model"]
        if m not in models_seen or s["pass_rate"] > models_seen[m]["pass_rate"]:
            models_seen[m] = s

    # Harness spread per model
    model_harness_rates = {}
    for s in summaries:
        model_harness_rates.setdefault(s["model"], []).append(s["pass_rate"])

    for m, best in sorted(models_seen.items(), key=lambda x: -x[1]["pass_rate"]):
        rates = model_harness_rates[m]
        spread = (max(rates) - min(rates)) * 100
        model_name = MODELS.get(m, {}).get("name", m)[:28]
        role = MODELS.get(m, {}).get("role", "")
        print(
            f"  {model_name:<30} Best: {best['harness']:<12} "
            f"{best['passes']}/{best['total']} ({100*best['pass_rate']:.1f}%)  "
            f"Spread: {spread:.0f}pp  Role: {role}"
        )

    # Bitter lesson metrics
    print("\n" + "-" * 100)
    print("BITTER LESSON VALIDATION")
    print("-" * 100)

    devstral = models_seen.get("devstral-small-2")
    swesmith = models_seen.get("swesmith-qwen25-32b")
    qwen35 = models_seen.get("qwen35-397b")
    base32 = models_seen.get("qwen25-coder-32b")

    if devstral and swesmith:
        delta = (swesmith["pass_rate"] - devstral["pass_rate"]) * 100
        print(f"  Finetuning axis: Devstral 24B ({100*devstral['pass_rate']:.1f}%) → "
              f"SWE-smith 32B ({100*swesmith['pass_rate']:.1f}%) = {delta:+.0f}pp")
    if base32 and swesmith:
        delta = (swesmith["pass_rate"] - base32["pass_rate"]) * 100
        print(f"  SERA effect: Base 32B ({100*base32['pass_rate']:.1f}%) → "
              f"SWE-smith 32B ({100*swesmith['pass_rate']:.1f}%) = {delta:+.0f}pp")
    if devstral and qwen35:
        delta = (qwen35["pass_rate"] - devstral["pass_rate"]) * 100
        print(f"  Scale axis: Devstral 24B ({100*devstral['pass_rate']:.1f}%) → "
              f"Qwen3.5 397B ({100*qwen35['pass_rate']:.1f}%) = {delta:+.0f}pp")
    if swesmith and qwen35:
        delta = abs(swesmith["pass_rate"] - qwen35["pass_rate"]) * 100
        converged = "YES" if delta < 5 else "NO"
        print(f"  Convergence: SWE-smith 32B vs Qwen3.5 397B = {delta:.0f}pp gap → "
              f"Bitter lesson converged: {converged}")

    print("=" * 100 + "\n")


# ---------------------------------------------------------------------------
# Model Download
# ---------------------------------------------------------------------------

def download_model(model_id: str):
    """Download model weights from HuggingFace."""
    model = MODELS[model_id]
    dest = model["weights_path"]

    if os.path.exists(dest) and os.listdir(dest):
        log.info(f"Model already exists: {dest}")
        return True

    hf_id = model["hf_id"]
    log.info(f"Downloading {hf_id} to {dest}...")
    os.makedirs(dest, exist_ok=True)

    try:
        proc = subprocess.run(
            ["huggingface-cli", "download", hf_id, "--local-dir", dest],
            capture_output=True, text=True, timeout=3600,
        )
        if proc.returncode != 0:
            log.error(f"Download failed: {proc.stderr[:500]}")
            return False
        log.info(f"Downloaded {hf_id} to {dest}")
        return True
    except Exception as e:
        log.error(f"Download error: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Phase 1: Multi-Model × Multi-Harness Evaluation")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="vLLM endpoint")
    parser.add_argument("--model", help="Model to evaluate (from MODELS registry)")
    parser.add_argument("--harness", help="Harness to use")
    parser.add_argument("--run-all", action="store_true", help="Run full matrix")
    parser.add_argument("--report", nargs="+", help="Generate leaderboard from result files")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--workspace-dir", default="/mnt/nvme/swarm-workspaces", help="Workspace dir")
    parser.add_argument("--download", help="Download a model's weights")
    parser.add_argument("--download-all", action="store_true", help="Download all missing models")
    parser.add_argument("--check", action="store_true", help="Check model weights and harness status")
    parser.add_argument("--vllm-cmd", help="Print vLLM launch command for a model")
    args = parser.parse_args()

    if args.report:
        generate_leaderboard(args.report)
        return

    if args.vllm_cmd:
        if args.vllm_cmd not in MODELS:
            print(f"Unknown model: {args.vllm_cmd}. Choose from: {list(MODELS.keys())}")
            sys.exit(1)
        cmd = get_vllm_launch_cmd(args.vllm_cmd)
        print(" ".join(cmd))
        return

    if args.check:
        print("\n=== MODEL STATUS ===")
        for mid, m in MODELS.items():
            exists = os.path.exists(m["weights_path"]) and os.listdir(m["weights_path"]) if os.path.exists(m["weights_path"]) else False
            status = "READY" if exists else "MISSING"
            tp = f"TP{m['tp']}"
            print(f"  {mid:<25} {status:<10} {tp:<5} {m['name']}")
            if not exists:
                print(f"    Download: huggingface-cli download {m['hf_id']} --local-dir {m['weights_path']}")

        print("\n=== HARNESS STATUS ===")
        for hid in HARNESSES:
            adapter = get_adapter_path(hid)
            status = "READY" if adapter else "MISSING"
            print(f"  {hid:<15} {status:<10} {adapter or 'not found'}")

        print("\n=== vLLM STATUS ===")
        serving = check_vllm_health(args.endpoint)
        if serving:
            print(f"  Serving: {serving} at {args.endpoint}")
        else:
            print(f"  Not running at {args.endpoint}")
        return

    if args.download:
        download_model(args.download)
        return

    if args.download_all:
        for mid in MODELS:
            download_model(mid)
        return

    if args.run_all:
        # Run full matrix: all models × all harnesses
        # Models must be served one at a time (swap vLLM between models)
        all_summaries = []
        for model_id in MODELS:
            # Check if this model is currently being served
            serving = check_vllm_health(args.endpoint)
            if serving != model_id:
                log.warning(
                    f"vLLM is serving '{serving}', not '{model_id}'. "
                    f"Please restart vLLM with:\n  {' '.join(get_vllm_launch_cmd(model_id))}"
                )
                log.info("Skipping this model. Run --run-all again after swapping models.")
                continue

            for harness_id in HARNESSES:
                output = os.path.join(
                    args.output_dir,
                    f"swarm_phase1_{model_id}_{harness_id}.jsonl",
                )
                # Skip if already completed
                if os.path.exists(output):
                    with open(output) as f:
                        n_lines = sum(1 for _ in f)
                    if n_lines >= 50:
                        log.info(f"Skipping {model_id} × {harness_id} — already complete ({n_lines} results)")
                        continue

                summary = await run_model_harness(
                    model_id, harness_id, load_subset(),
                    args.endpoint, args.workspace_dir, output,
                )
                all_summaries.append(summary)

        if all_summaries:
            # Print summary
            all_summaries.sort(key=lambda s: -s["pass_rate"])
            print("\n=== COMPLETED CONFIGURATIONS ===")
            for s in all_summaries:
                print(f"  {s['model_name']:<30} × {s['harness']:<12}: "
                      f"{s['passes']}/{s['total']} ({100*s['pass_rate']:.1f}%)")
        return

    if args.model and args.harness:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}. Choose from: {list(MODELS.keys())}")
            sys.exit(1)
        if args.harness not in HARNESSES:
            print(f"Unknown harness: {args.harness}. Choose from: {HARNESSES}")
            sys.exit(1)

        output = os.path.join(
            args.output_dir,
            f"swarm_phase1_{args.model}_{args.harness}.jsonl",
        )
        await run_model_harness(
            args.model, args.harness, load_subset(),
            args.endpoint, args.workspace_dir, output,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
