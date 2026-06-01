#!/usr/bin/env python3
"""validate-serving-config.py — CLI gate over a blueprint's serving config.

Reads a benchmark.yaml sidecar (and optionally a model-deployment-card JSON for
facts the sidecar lacks, e.g. moe_intermediate_size), runs the deterministic rule
registry, and exits non-zero on any hard-rule FAIL. Wire into pre-deploy / Stage 0.

Usage:
  validate-serving-config.py --sidecar path/to/benchmark.yaml [--card card.json]
  validate-serving-config.py --sidecar ... --warnings-as-errors
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml  # noqa: E402
from compiler import validate_sidecar  # noqa: E402
from corpus import load_corpus  # noqa: E402


def _emit(report) -> None:
    cfg = report.config
    print(f"=== serving-config validation: {cfg.model.name} on {cfg.hardware.instance_type} "
          f"[{cfg.engine.name} TP{cfg.engine.tensor_parallel}] ===")
    if not report.findings:
        print("  no applicable rules fired — config is clean.")
        return
    for f in report.failures:
        print(f"  ✗ FAIL [{f.rule}] {f.reason}")
        print(f"        source: {f.source}")
        if f.fix:
            print(f"        fix:    {f.fix}")
    for f in report.warnings:
        print(f"  ! WARN [{f.rule}] {f.reason}")
        if f.fix:
            print(f"        fix:    {f.fix}")
    for f in report.infos:
        print(f"  i INFO [{f.rule}] {f.reason}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a serving config against the rule registry")
    ap.add_argument("--sidecar", required=True, type=Path)
    ap.add_argument("--card", type=Path, help="model-deployment-card JSON (optional facts)")
    ap.add_argument("--corpus-root", type=Path,
                    help="repo root to harvest blueprint lessons.md field notes; "
                         "prior recorded failures for this model/engine surface as findings")
    ap.add_argument("--warnings-as-errors", action="store_true")
    ap.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = ap.parse_args()

    sidecar = yaml.safe_load(open(args.sidecar))
    card = json.load(open(args.card)) if args.card else None
    corpus = load_corpus(args.corpus_root) if args.corpus_root else None
    report = validate_sidecar(sidecar, card=card, source=str(args.sidecar), corpus=corpus)

    if args.json:
        print(json.dumps({
            "ok": report.ok,
            "findings": [vars(f) for f in report.findings],
        }, indent=2))
    else:
        _emit(report)

    if not report.ok:
        print(f"\nBLOCKED: {len(report.failures)} hard-rule violation(s). Fix the "
              f"sidecar before deploying.", file=sys.stderr)
        return 2
    if args.warnings_as_errors and report.warnings:
        print(f"\nBLOCKED (--warnings-as-errors): {len(report.warnings)} warning(s).",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
