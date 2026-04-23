#!/usr/bin/env python3
"""
Autoresearch iteration driver: runs the full rubric iteration loop unattended.

Phases:
  1. Smoke test new rubric versions
  2. Sweep new + top versions on dev set (sonnet patches)
  3. Error analysis on sweep results
  4. Evaluate best version on holdout set (haiku + opus patches)
  5. Update progress

Usage:
  python3 run_iteration.py --iteration 2
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
VERSIONS_DIR = SKILL_DIR / "versions"
BLUEPRINT_DIR = SKILL_DIR.parent.parent
RESULTS_DIR = BLUEPRINT_DIR / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / "iteration_log.txt", mode="a"),
    ],
)
log = logging.getLogger(__name__)


def run_cmd(cmd: list[str], description: str, timeout: int = 1800) -> tuple[int, str]:
    """Run a command and return (exit_code, output)."""
    log.info(f">>> {description}")
    log.info(f"    cmd: {' '.join(cmd)}")
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(SCRIPT_DIR)
        )
        elapsed = time.monotonic() - start
        output = result.stdout + result.stderr
        log.info(f"    exit={result.returncode} elapsed={elapsed:.1f}s")
        if result.returncode != 0:
            log.warning(f"    stderr: {result.stderr[-500:]}")
        return result.returncode, output
    except subprocess.TimeoutExpired:
        log.error(f"    TIMEOUT after {timeout}s")
        return -1, "TIMEOUT"


def smoke_test(version_name: str, model: str = "haiku", patch_source: str = "sonnet") -> bool:
    """Run smoke test on a rubric version. Returns True if passed."""
    version_file = VERSIONS_DIR / f"{version_name}.md"
    if not version_file.exists():
        log.error(f"Version file not found: {version_file}")
        return False

    code, output = run_cmd(
        [sys.executable, "smoke_test.py",
         "--rubric", str(version_file),
         "--model", model,
         "--patch-source", patch_source],
        f"Smoke test {version_name} × {model}",
    )
    passed = code == 0
    log.info(f"    Smoke test {'PASSED' if passed else 'FAILED'}")
    return passed


def sweep(versions: list[str], verifier_model: str, patch_source: str,
          output_file: str, temperature: float = 0.0) -> bool:
    """Run sweep on specified versions. Returns True if completed."""
    version_filter = ",".join(versions)
    code, output = run_cmd(
        [sys.executable, "sweep_versions.py",
         "--versions", version_filter,
         "--verifier-model", verifier_model,
         "--patch-source", patch_source,
         "--temperature", str(temperature),
         "--output", output_file],
        f"Sweep {version_filter} × {verifier_model} × {patch_source}",
        timeout=3600,
    )
    return code == 0


def error_analysis(sweep_file: str, output_file: str) -> dict:
    """Run error analysis on sweep results. Returns summary."""
    code, output = run_cmd(
        [sys.executable, "analyze_errors.py", sweep_file, "--output", output_file],
        f"Error analysis on {Path(sweep_file).name}",
    )
    # Parse summary from output file
    summary = {}
    try:
        first_line = Path(output_file).read_text().strip().split("\n")[0]
        summary = json.loads(first_line).get("summary", {})
    except Exception:
        pass
    return summary


def parse_sweep_summaries(sweep_file: str) -> list[dict]:
    """Extract sweep_complete events from a JSONL file."""
    summaries = []
    for line in Path(sweep_file).read_text().strip().split("\n"):
        if not line:
            continue
        row = json.loads(line)
        if row.get("event") == "sweep_complete":
            summaries.append(row)
    return summaries


def pick_best_version(summaries: list[dict], metric: str = "f05") -> str:
    """Pick the best version from sweep summaries by a given metric."""
    best = None
    best_val = -1
    for s in summaries:
        val = s["metrics"].get(metric, 0)
        if val > best_val:
            best_val = val
            best = s["config"]["skill_version"]
    return best


def update_progress(iteration: int, results: dict):
    """Append iteration results to progress log."""
    progress_file = RESULTS_DIR / "iteration_results.jsonl"
    entry = {
        "iteration": iteration,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **results,
    }
    with open(progress_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info(f"Results logged to {progress_file}")


def main():
    parser = argparse.ArgumentParser(description="Run autoresearch iteration")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke tests")
    parser.add_argument("--verifier-model", default="haiku")
    args = parser.parse_args()

    iteration = args.iteration
    verifier = args.verifier_model
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    log.info(f"\n{'='*70}")
    log.info(f"AUTORESEARCH ITERATION {iteration}")
    log.info(f"Started: {timestamp}")
    log.info(f"Verifier model: {verifier}")
    log.info(f"{'='*70}\n")

    # ── Phase 1: Identify versions to test ──────────────────────────
    # New versions (created since last sweep) + top performers from prior sweep
    all_versions = sorted(VERSIONS_DIR.glob("v*.md"))
    version_names = [v.stem for v in all_versions]
    log.info(f"Available versions: {version_names}")

    # For iteration 2: new versions are v006, v007; carry forward v001, v004 from Phase 2b
    # For future iterations: this would be data-driven from prior sweep results
    new_versions = ["v006_reformat_aware", "v007_strict_logic"]
    carry_forward = ["v001_baseline", "v004_cot"]

    # Validate all versions exist
    test_versions = []
    for v in new_versions + carry_forward:
        if (VERSIONS_DIR / f"{v}.md").exists():
            test_versions.append(v)
        else:
            log.warning(f"Version {v} not found, skipping")

    log.info(f"Versions to test: {test_versions}")

    # ── Phase 2: Smoke test new versions ────────────────────────────
    if not args.skip_smoke:
        log.info("\n--- Phase 2: Smoke Tests ---")
        smoke_passed = []
        smoke_failed = []

        for v in new_versions:
            if v not in test_versions:
                continue
            if smoke_test(v, model=verifier):
                smoke_passed.append(v)
            else:
                smoke_failed.append(v)
                log.warning(f"  {v} FAILED smoke test — will still include in sweep for comparison")

        log.info(f"  Smoke results: {len(smoke_passed)} passed, {len(smoke_failed)} failed")
        if smoke_failed:
            log.info(f"  Failed: {smoke_failed}")
    else:
        log.info("Skipping smoke tests (--skip-smoke)")

    # ── Phase 3: Dev set sweep ──────────────────────────────────────
    log.info("\n--- Phase 3: Dev Set Sweep (sonnet patches) ---")
    dev_sweep_file = str(RESULTS_DIR / f"sweep_iter{iteration}_dev.jsonl")

    # Extract version short names for filter (v001, v004, v006, v007)
    version_short = [v.split("_")[0] for v in test_versions]

    sweep_ok = sweep(
        versions=version_short,
        verifier_model=verifier,
        patch_source="sonnet",
        output_file=dev_sweep_file,
    )

    if not sweep_ok:
        log.error("Dev sweep failed!")
        update_progress(iteration, {"status": "FAILED", "phase": "dev_sweep"})
        return

    # Parse results
    dev_summaries = parse_sweep_summaries(dev_sweep_file)
    log.info("\n--- Dev Set Results ---")
    for s in dev_summaries:
        m = s["metrics"]
        log.info(
            f"  {s['config']['skill_version']:25s} "
            f"prec={m['precision']:.2f} rec={m['recall']:.2f} "
            f"f05={m.get('f05', 'N/A')} "
            f"conf_err={m.get('confident_error_rate', 'N/A')} "
            f"lift={m['lift_over_random_pp']:+.1f}pp "
            f"cost=${m['total_cost_usd']:.4f}"
        )

    # ── Phase 4: Error analysis ─────────────────────────────────────
    log.info("\n--- Phase 4: Error Analysis ---")
    errors_file = str(RESULTS_DIR / f"errors_iter{iteration}_dev.jsonl")
    error_summary = error_analysis(dev_sweep_file, errors_file)
    log.info(f"  Errors: {error_summary}")

    # ── Phase 5: Pick best version and evaluate on holdout ──────────
    log.info("\n--- Phase 5: Holdout Evaluation ---")
    best_version = pick_best_version(dev_summaries, metric="precision")
    if not best_version:
        best_version = pick_best_version(dev_summaries, metric="f1")
    log.info(f"  Best version (by precision): {best_version}")

    if best_version:
        best_short = best_version.split("_")[0]

        # Holdout: haiku patches
        holdout_haiku_file = str(RESULTS_DIR / f"sweep_iter{iteration}_holdout_haiku.jsonl")
        sweep(
            versions=[best_short],
            verifier_model=verifier,
            patch_source="haiku",
            output_file=holdout_haiku_file,
        )

        # Holdout: opus patches
        holdout_opus_file = str(RESULTS_DIR / f"sweep_iter{iteration}_holdout_opus.jsonl")
        sweep(
            versions=[best_short],
            verifier_model=verifier,
            patch_source="opus",
            output_file=holdout_opus_file,
        )

        # Parse holdout results
        holdout_summaries = (
            parse_sweep_summaries(holdout_haiku_file) +
            parse_sweep_summaries(holdout_opus_file)
        )

        log.info("\n--- Holdout Results ---")
        for s in holdout_summaries:
            m = s["metrics"]
            log.info(
                f"  {s['config']['skill_version']} × {s['config']['patch_source']:8s} "
                f"prec={m['precision']:.2f} rec={m['recall']:.2f} "
                f"f05={m.get('f05', 'N/A')} "
                f"conf_err={m.get('confident_error_rate', 'N/A')} "
                f"lift={m['lift_over_random_pp']:+.1f}pp"
            )

        # Compute holdout delta
        dev_best = next((s for s in dev_summaries if s["config"]["skill_version"] == best_version), None)
        if dev_best and holdout_summaries:
            dev_prec = dev_best["metrics"]["precision"]
            holdout_precs = [s["metrics"]["precision"] for s in holdout_summaries]
            avg_holdout_prec = sum(holdout_precs) / len(holdout_precs)
            holdout_delta = abs(dev_prec - avg_holdout_prec)
            overfit = holdout_delta > 0.05
            log.info(f"\n  Holdout delta: {holdout_delta:.2f} ({'OVERFIT WARNING' if overfit else 'OK'})")

    # ── Phase 6: Cross-model sweep (Sonnet as verifier) ─────────────
    log.info("\n--- Phase 6: Cross-Model Sweep (Sonnet as verifier) ---")
    if best_version:
        best_short = best_version.split("_")[0]
        cross_model_file = str(RESULTS_DIR / f"sweep_iter{iteration}_sonnet_verifier.jsonl")
        sweep(
            versions=[best_short],
            verifier_model="sonnet",
            patch_source="sonnet",
            output_file=cross_model_file,
        )

        cross_summaries = parse_sweep_summaries(cross_model_file)
        log.info("\n--- Sonnet-as-Verifier Results ---")
        for s in cross_summaries:
            m = s["metrics"]
            log.info(
                f"  {s['config']['skill_version']} × sonnet-verifier "
                f"prec={m['precision']:.2f} rec={m['recall']:.2f} "
                f"f05={m.get('f05', 'N/A')} "
                f"conf_err={m.get('confident_error_rate', 'N/A')} "
                f"lift={m['lift_over_random_pp']:+.1f}pp "
                f"cost=${m['total_cost_usd']:.4f}"
            )

    # ── Summary ─────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info(f"ITERATION {iteration} COMPLETE")
    log.info(f"{'='*70}")

    iteration_results = {
        "status": "COMPLETE",
        "best_version": best_version,
        "dev_summaries": [{
            "version": s["config"]["skill_version"],
            "precision": s["metrics"]["precision"],
            "recall": s["metrics"]["recall"],
            "f05": s["metrics"].get("f05"),
            "confident_error_rate": s["metrics"].get("confident_error_rate"),
            "lift_pp": s["metrics"]["lift_over_random_pp"],
            "cost": s["metrics"]["total_cost_usd"],
        } for s in dev_summaries],
        "error_summary": error_summary,
    }

    if best_version and holdout_summaries:
        iteration_results["holdout_summaries"] = [{
            "version": s["config"]["skill_version"],
            "patch_source": s["config"]["patch_source"],
            "precision": s["metrics"]["precision"],
            "recall": s["metrics"]["recall"],
            "lift_pp": s["metrics"]["lift_over_random_pp"],
        } for s in holdout_summaries]

    if best_version and cross_summaries:
        iteration_results["cross_model_summaries"] = [{
            "version": s["config"]["skill_version"],
            "verifier": "sonnet",
            "precision": s["metrics"]["precision"],
            "recall": s["metrics"]["recall"],
            "lift_pp": s["metrics"]["lift_over_random_pp"],
            "cost": s["metrics"]["total_cost_usd"],
        } for s in cross_summaries]

    update_progress(iteration, iteration_results)

    log.info("\nAll results written. Check:")
    log.info(f"  Dev sweep:    {dev_sweep_file}")
    log.info(f"  Error report: {errors_file}")
    if best_version:
        log.info(f"  Holdout:      {holdout_haiku_file}, {holdout_opus_file}")
        log.info(f"  Cross-model:  {cross_model_file}")
    log.info(f"  Summary:      {RESULTS_DIR / 'iteration_results.jsonl'}")


if __name__ == "__main__":
    main()
