#!/usr/bin/env python3
"""
Production ensemble verifier: v001∩v009(2+/3 lc from 3 temp=0.3 runs).

Runs v001_baseline once (temp=0.0) and v009_adversarial 3 times
(all temp=0.3). Patch passes only if:
  - v001 says "likely_correct"
  - v009 says "likely_correct" in >= 2 of 3 runs

This achieves precision=1.00, recall=0.33, F0.5=0.71 on holdout.
Cost: ~$0.038 per patch (4 Haiku calls).

Usage:
  python3 verify_patch_ensemble.py \
    --problem "bug description" \
    --diff path/to/file.diff

  # Or pipe problem statement:
  echo "bug description" | python3 verify_patch_ensemble.py --diff path/to/file.diff
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VERSIONS_DIR = SCRIPT_DIR.parent / "versions"
sys.path.insert(0, str(SCRIPT_DIR))

from verify_patch import verify_patch

V001_RUBRIC = str(VERSIONS_DIR / "v001_baseline.md")
V009_RUBRIC = str(VERSIONS_DIR / "v009_adversarial.md")

# Ensemble parameters
V009_RUNS = 3  # 3x temp=0.3
V009_THRESHOLD = 2  # need >= this many "likely_correct" from v009
DEFAULT_MODEL = "haiku"


def run_ensemble(
    problem_statement: str,
    diff_content: str,
    model_key: str = DEFAULT_MODEL,
    threshold: int = V009_THRESHOLD,
    max_workers: int = 5,
) -> dict:
    """Run the v001∩v009 ensemble and return combined result."""

    calls = [
        ("v001", V001_RUBRIC, 0.0),
        ("v009_r1", V009_RUBRIC, 0.3),
        ("v009_r2", V009_RUBRIC, 0.3),
        ("v009_r3", V009_RUBRIC, 0.3),
    ]

    results = {}

    def run_one(name, rubric, temp):
        r = verify_patch(
            rubric_path=rubric,
            problem_statement=problem_statement,
            diff_content=diff_content,
            model_key=model_key,
            temperature=temp,
        )
        return name, r

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_one, n, r, t) for n, r, t in calls]
        for f in as_completed(futures):
            name, result = f.result()
            results[name] = result

    # Extract verdicts
    v001_result = results["v001"]
    v001_verdict = (v001_result.get("parsed") or {}).get("verdict", "")
    v001_pass = v001_verdict == "likely_correct"

    v009_names = [n for n in results if n.startswith("v009")]
    v009_lc_count = sum(
        1 for n in v009_names
        if (results[n].get("parsed") or {}).get("verdict") == "likely_correct"
    )
    v009_pass = v009_lc_count >= threshold

    ensemble_pass = v001_pass and v009_pass

    # Aggregate costs
    total_cost = sum(r.get("cost_usd", 0) for r in results.values())
    total_input = sum(r.get("input_tokens", 0) for r in results.values())
    total_output = sum(r.get("output_tokens", 0) for r in results.values())
    total_latency = max(r.get("latency_ms", 0) for r in results.values())  # parallel
    errors = [
        f"{n}: {r['error']}" for n, r in results.items() if r.get("error")
    ]

    return {
        "ensemble_verdict": "likely_correct" if ensemble_pass else "likely_incorrect",
        "ensemble_pass": ensemble_pass,
        "v001_verdict": v001_verdict,
        "v001_score": (v001_result.get("parsed") or {}).get("overall_score"),
        "v009_likely_correct_count": v009_lc_count,
        "v009_threshold": threshold,
        "v009_verdicts": {
            n: (results[n].get("parsed") or {}).get("verdict", "error")
            for n in sorted(v009_names)
        },
        "total_cost_usd": round(total_cost, 6),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "wall_latency_ms": total_latency,
        "errors": errors if errors else None,
        "details": {
            n: {
                "verdict": (r.get("parsed") or {}).get("verdict"),
                "score": (r.get("parsed") or {}).get("overall_score"),
                "reasoning": (r.get("parsed") or {}).get("reasoning", "")[:200],
                "cost_usd": r.get("cost_usd", 0),
            }
            for n, r in sorted(results.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble patch verification (v001∩v009)"
    )
    parser.add_argument("--problem", type=str, help="Problem statement text")
    parser.add_argument("--diff", required=True, help="Path to diff file")
    parser.add_argument(
        "--model", choices=["haiku", "sonnet", "opus"], default=DEFAULT_MODEL
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=V009_THRESHOLD,
        help=f"Min v009 likely_correct count (default: {V009_THRESHOLD})",
    )
    parser.add_argument("--compact", action="store_true", help="Compact output")
    args = parser.parse_args()

    problem = args.problem if args.problem else sys.stdin.read()
    diff_content = Path(args.diff).read_text()

    result = run_ensemble(
        problem_statement=problem,
        diff_content=diff_content,
        model_key=args.model,
        threshold=args.threshold,
    )

    if args.compact:
        print(json.dumps({
            "verdict": result["ensemble_verdict"],
            "v001": result["v001_verdict"],
            "v009_lc": f"{result['v009_likely_correct_count']}/{V009_RUNS}",
            "cost": f"${result['total_cost_usd']:.4f}",
        }))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
