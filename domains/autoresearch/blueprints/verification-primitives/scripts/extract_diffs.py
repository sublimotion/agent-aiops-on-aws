#!/usr/bin/env python3
"""
Extract patch diffs from experiment JSONL results into individual .diff files
for gold evaluation.

Usage:
    python3 extract_diffs.py --input results/full_control.jsonl --output-dir results/diffs/control
    python3 extract_diffs.py --input results/full_B_checkpoint.jsonl --output-dir results/diffs/B_checkpoint
"""

import argparse
import json
import os
from pathlib import Path


def extract(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    with_diff = 0

    with open(input_path) as f:
        for line in f:
            r = json.loads(line)
            total += 1
            diff = r.get("patch_diff", "")
            if diff.strip():
                with_diff += 1
                diff_path = os.path.join(output_dir, f"{r['instance_id']}.diff")
                # Ensure trailing newline — git apply requires it
                if not diff.endswith("\n"):
                    diff += "\n"
                Path(diff_path).write_text(diff)

    print(f"Extracted {with_diff}/{total} diffs to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    extract(args.input, args.output_dir)


if __name__ == "__main__":
    main()
