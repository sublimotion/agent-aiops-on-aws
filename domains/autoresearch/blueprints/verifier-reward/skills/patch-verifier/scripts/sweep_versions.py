#!/usr/bin/env python3
"""
Sweep all rubric versions × patches with telemetry.

Usage:
  python3 sweep_versions.py \
    --versions-dir versions/ \
    --verifier-model haiku \
    --patch-source sonnet \
    --temperature 0.0 \
    --output results/sweep_phase2b.jsonl

  # Specific versions only:
  python3 sweep_versions.py \
    --versions v001,v003,v004 \
    --verifier-model haiku,sonnet \
    --patch-source sonnet \
    --output results/sweep_phase2c.jsonl
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

# Add parent dirs to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
BLUEPRINT_DIR = SKILL_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BLUEPRINT_DIR / "scripts"))

from telemetry import TelemetryLogger, InvocationEvent
from verify_patch import verify_patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SUBSET_SEED = 42
SUBSET_SIZE = 50
RESULTS_DIR = BLUEPRINT_DIR / "results"


def load_gold_labels(patch_source: str) -> dict:
    """Load gold eval labels for a patch source."""
    gold_file = RESULTS_DIR / f"gold_{patch_source}_opencode.jsonl"
    labels = {}
    if gold_file.exists():
        for line in gold_file.read_text().strip().split("\n"):
            if line:
                row = json.loads(line)
                labels[row["instance_id"]] = {
                    "passed": row["passed"],
                    "patch_applied": row.get("patch_applied"),
                }
    return labels


def load_issue_metadata() -> dict:
    """Load problem statements from SWE-bench dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install: pip install datasets")
        sys.exit(1)

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    metadata = {}
    for row in ds:
        metadata[row["instance_id"]] = {
            "problem_statement": row["problem_statement"],
            "repo": row["repo"],
        }

    # Filter to our 50-issue subset
    all_issues = list(metadata.keys())
    by_repo = {}
    for iid in all_issues:
        repo = metadata[iid]["repo"]
        by_repo.setdefault(repo, []).append(iid)

    rng = random.Random(SUBSET_SEED)
    repos = sorted(by_repo.keys())
    rng.shuffle(repos)
    selected = []
    idx = {r: 0 for r in repos}
    while len(selected) < SUBSET_SIZE:
        for repo in repos:
            issues = by_repo[repo]
            if idx[repo] < len(issues) and len(selected) < SUBSET_SIZE:
                selected.append(issues[idx[repo]])
                idx[repo] += 1

    return {iid: metadata[iid] for iid in selected}


def get_version_files(versions_dir: str, version_filter: str = None) -> list[Path]:
    """Get rubric version files, optionally filtered."""
    versions_path = Path(versions_dir)
    all_versions = sorted(versions_path.glob("v*.md"))

    if version_filter:
        names = set(version_filter.split(","))
        all_versions = [v for v in all_versions if v.stem.split("_")[0] in names]

    return all_versions


def main():
    parser = argparse.ArgumentParser(description="Sweep rubric versions")
    parser.add_argument("--versions-dir", default=str(SKILL_DIR / "versions"))
    parser.add_argument("--versions", type=str, help="Comma-separated version IDs (e.g. v001,v003)")
    parser.add_argument("--verifier-model", default="haiku", help="Comma-separated models")
    parser.add_argument("--patch-source", default="sonnet", help="Comma-separated sources")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, help="Limit patches per version")
    parser.add_argument("--dry-run", action="store_true", help="Show config without running")
    args = parser.parse_args()

    verifier_models = args.verifier_model.split(",")
    patch_sources = args.patch_source.split(",")
    version_files = get_version_files(args.versions_dir, args.versions)

    if not version_files:
        log.error("No version files found")
        sys.exit(1)

    # Compute sweep size and cost estimate
    total_cells = len(version_files) * len(verifier_models) * len(patch_sources)
    log.info(f"Sweep: {len(version_files)} versions × {len(verifier_models)} models × {len(patch_sources)} sources = {total_cells} cells")

    # Load metadata
    log.info("Loading issue metadata...")
    issue_meta = load_issue_metadata()

    for patch_source in patch_sources:
        diffs_dir = RESULTS_DIR / "diffs" / f"opencode_{patch_source}"
        diff_count = len(list(diffs_dir.glob("*.diff"))) if diffs_dir.exists() else 0
        gold_labels = load_gold_labels(patch_source)
        log.info(f"  {patch_source}: {diff_count} diffs, {len(gold_labels)} gold labels")

    if args.dry_run:
        for vf in version_files:
            log.info(f"  Version: {vf.stem}")
        est_calls = total_cells * 50  # ~50 patches per source
        log.info(f"Estimated calls: ~{est_calls}, cost: ~${est_calls * 0.001:.2f} (haiku)")
        return

    # Run sweep
    for version_file in version_files:
        version_name = version_file.stem

        for verifier_model in verifier_models:
            for patch_source in patch_sources:
                run_id = f"sweep_{version_name}_{verifier_model}_{patch_source}_t{args.temperature}"
                log.info(f"\n{'='*60}")
                log.info(f"Run: {run_id}")
                log.info(f"{'='*60}")

                telemetry = TelemetryLogger(args.output, run_id)
                version_hash = telemetry.version_hash(str(version_file))

                diffs_dir = RESULTS_DIR / "diffs" / f"opencode_{patch_source}"
                if not diffs_dir.exists():
                    log.warning(f"No diffs dir for {patch_source}, skipping")
                    continue

                gold_labels = load_gold_labels(patch_source)
                diff_files = sorted(diffs_dir.glob("*.diff"))

                if args.limit:
                    diff_files = diff_files[:args.limit]

                for idx, diff_file in enumerate(diff_files):
                    instance_id = diff_file.stem

                    if instance_id not in issue_meta:
                        continue

                    problem = issue_meta[instance_id]["problem_statement"]
                    diff_content = diff_file.read_text()
                    gold = gold_labels.get(instance_id, {})

                    log.info(f"  [{idx+1}/{len(diff_files)}] {instance_id}")

                    # Create telemetry event
                    event = telemetry.new_event(
                        skill_version=version_name,
                        version_hash=version_hash,
                        instance_id=instance_id,
                        patch_source=patch_source,
                        verifier_model=verifier_model,
                        temperature=args.temperature,
                        gold_passed=gold.get("passed"),
                        gold_patch_applied=gold.get("patch_applied"),
                    )

                    # Call verification
                    result = verify_patch(
                        rubric_path=str(version_file),
                        problem_statement=problem,
                        diff_content=diff_content,
                        model_key=verifier_model,
                        temperature=args.temperature,
                    )

                    # Populate event
                    event.invoked = True
                    event.input_tokens = result["input_tokens"]
                    event.output_tokens = result["output_tokens"]
                    event.latency_ms = result["latency_ms"]
                    event.cost_usd = result["cost_usd"]
                    event.parse_success = result["parse_success"]
                    event.error = result["error"]
                    event.problem_statement_tokens = result["problem_tokens"]
                    event.diff_tokens = result["diff_tokens"]

                    if result["parsed"]:
                        parsed = result["parsed"]
                        event.scores = parsed.get("scores", {})
                        event.overall_score = parsed.get("overall_score", 0.0)
                        event.confidence = parsed.get("confidence", 0.0)
                        event.verdict = parsed.get("verdict", "")
                        event.reasoning = parsed.get("reasoning", "")[:200]

                    # Log
                    telemetry.log_event(event)

                    status = event.verdict if event.parse_success else f"ERROR: {event.error}"
                    gold_str = "PASS" if gold.get("passed") else "FAIL"
                    log.info(f"    {status} (gold={gold_str}) score={event.overall_score:.2f} ${event.cost_usd:.4f}")

                    # Check circuit breaker
                    halt, reason = telemetry.check_circuit_breaker()
                    if halt:
                        log.error(f"  CIRCUIT BREAKER: {reason}")
                        break

                # Emit sweep summary
                config = {
                    "skill_version": version_name,
                    "verifier_model": verifier_model,
                    "patch_source": patch_source,
                    "temperature": args.temperature,
                }
                summary = telemetry.log_sweep_summary(config)
                if summary:
                    metrics = summary["metrics"]
                    log.info(
                        f"  SUMMARY: precision={metrics['precision']:.2f} "
                        f"recall={metrics['recall']:.2f} "
                        f"lift={metrics['lift_over_random_pp']:+.1f}pp "
                        f"cost=${metrics['total_cost_usd']:.4f}"
                    )

    log.info("\nSweep complete!")


if __name__ == "__main__":
    main()
