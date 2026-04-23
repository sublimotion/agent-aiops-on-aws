#!/usr/bin/env python3
"""
Build the Phase 3 joint feature matrix by merging all 5 signal sources.

Sources:
1. Behavioral features (tiny-judge features.csv) — 282 instances × 37 features
2. v009 rubric scores (run_v009_lite.py output) — 300 instances × per-run details
3. Debate verdicts (phase3_2round.jsonl) — 285 instances × confidence scores
4. SVG consensus (svg_results_production_run1.jsonl) — 300 instances × line_recall
5. Gold labels (gold_lite_combined.jsonl) — 300 instances × pass/fail

Output: results/combined_features.csv — 1 row per instance with all signal features
"""

import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit  # sigmoid for Platt scaling

BASE = Path(__file__).resolve().parent.parent
TINY_JUDGE = BASE.parent / "tiny-judge"
DEBATE = BASE.parent / "debate-verification"
SVG_DATA = BASE / "data" / "phase0"
VP_SWEBENCH = BASE.parent / "verification-primitives-swebench"

ENEW_PATH = BASE / "results" / "enew1_enew2_features.csv"
OUTPUT_PATH = BASE / "results" / "combined_features.csv"
SUMMARY_PATH = BASE / "results" / "combined_feature_summary.json"


def load_behavioral_features():
    """Load the tiny-judge feature matrix (behavioral + v009 from VP overlap)."""
    path = TINY_JUDGE / "results" / "features.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  Behavioral: {len(df)} instances × {len(df.columns)} features")
    return df


def load_v009_results():
    """Load v009 results from the full 300-instance run."""
    path = BASE / "results" / "v009_lite_300.jsonl"
    if not path.exists():
        print(f"  WARNING: {path} not found — run run_v009_lite.py first")
        return {}

    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            iid = r["instance_id"]

            # Extract per-run scores
            details = r.get("details", {})
            scores = []
            logic_scores = []
            completeness_scores = []
            confidences = []
            for run_key in ["r0", "r1", "r2", "r3"]:
                d = details.get(run_key, {})
                s = d.get("score")
                if s is not None:
                    scores.append(float(s))
                lc = d.get("logic_correctness")
                if lc is not None:
                    logic_scores.append(float(lc))
                comp = d.get("completeness")
                if comp is not None:
                    completeness_scores.append(float(comp))
                conf = d.get("confidence")
                if conf is not None:
                    confidences.append(float(conf))

            results[iid] = {
                "v009_lc_count": r.get("lc_count", 0),
                "v009_unanimous": 1 if r.get("v009_unanimous") else 0,
                "v009_verdict_lc": 1 if r.get("v009_verdict") == "likely_correct" else 0,
                "v009_mean_score": np.mean(scores) if scores else None,
                "v009_std_score": np.std(scores) if len(scores) > 1 else 0.0,
                "v009_mean_logic": np.mean(logic_scores) if logic_scores else None,
                "v009_mean_completeness": np.mean(completeness_scores) if completeness_scores else None,
                "v009_mean_confidence": np.mean(confidences) if confidences else None,
                "v009_min_score": min(scores) if scores else None,
                "v009_max_score": max(scores) if scores else None,
            }

    print(f"  v009: {len(results)} instances")
    return results


def load_debate_results():
    """Load debate verdicts and confidence scores."""
    path = DEBATE / "results" / "phase3_2round.jsonl"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return {}

    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            iid = r["instance_id"]

            verdict = r.get("verdict", "UNKNOWN")
            judge_conf = r.get("judge_confidence")
            advocate_conf = r.get("advocate_confidence")
            bugs_found = r.get("challenger_bugs_found", 0)

            # Parse challenger assessment
            challenger_assessment = r.get("challenger_assessment", "")
            challenger_is_broken = 1 if "broken" in str(challenger_assessment).lower() else 0

            # Derive numeric debate score:
            # CORRECT → judge_confidence (positive), INCORRECT → 1 - judge_confidence (negative)
            if verdict == "CORRECT":
                debate_score = judge_conf if judge_conf is not None else 0.7
            elif verdict == "INCORRECT":
                debate_score = (1 - judge_conf) if judge_conf is not None else 0.3
            elif verdict == "UNCERTAIN":
                debate_score = 0.5
            else:
                debate_score = 0.5

            # Agreement: advocate and judge agree
            advocate_judge_agree = 1 if (verdict == "CORRECT" and advocate_conf and advocate_conf > 0.7) else 0

            results[iid] = {
                "debate_verdict_correct": 1 if verdict == "CORRECT" else 0,
                "debate_verdict_incorrect": 1 if verdict == "INCORRECT" else 0,
                "debate_verdict_uncertain": 1 if verdict == "UNCERTAIN" else 0,
                "debate_score": debate_score,
                "debate_judge_confidence": judge_conf,
                "debate_advocate_confidence": advocate_conf,
                "debate_bugs_found": bugs_found,
                "debate_challenger_broken": challenger_is_broken,
                "debate_advocate_judge_agree": advocate_judge_agree,
            }

    print(f"  Debate: {len(results)} instances")
    return results


def load_svg_results():
    """Load SVG consensus results with Platt-scaled probabilities."""
    path = SVG_DATA / "svg_results_production_run1.jsonl"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return {}

    # Platt scaling parameters from svg-ece-measurement
    PLATT_A = 1.232
    PLATT_B = 0.347

    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            iid = r["instance_id"]
            line_recall = r.get("line_recall", 0.0)
            accepted = 1 if r.get("accepted") else 0
            fix_generated = 1 if r.get("fix_generated") else 0

            # Platt-scaled probability
            platt_prob = float(expit(PLATT_A * line_recall + PLATT_B))

            results[iid] = {
                "svg_line_recall": line_recall,
                "svg_accepted": accepted,
                "svg_fix_generated": fix_generated,
                "svg_platt_prob": platt_prob,
            }

    print(f"  SVG: {len(results)} instances")
    return results


def load_enew_features():
    """Load E_new1 + E_new2 features (Read:Edit ratio, recovery breadth)."""
    if not ENEW_PATH.exists():
        print(f"  WARNING: {ENEW_PATH} not found — run extract_enew1_enew2.py first")
        return {}

    df = pd.read_csv(ENEW_PATH)
    results = {}
    for _, row in df.iterrows():
        iid = row["instance_id"]
        results[iid] = {col: (None if pd.isna(row[col]) else row[col])
                        for col in df.columns if col != "instance_id"}

    print(f"  E_new1/E_new2: {len(results)} instances × {len(df.columns)-1} features")
    return results


def load_gold_labels():
    """Load gold pass/fail labels."""
    path = DEBATE / "results" / "gold_lite_combined.jsonl"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return {}

    labels = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            labels[r["instance_id"]] = 1 if r.get("passed") else 0

    print(f"  Gold labels: {len(labels)} instances")
    return labels


def main():
    print("Loading all signal sources...\n")

    behavioral_df = load_behavioral_features()
    v009 = load_v009_results()
    debate = load_debate_results()
    svg = load_svg_results()
    enew = load_enew_features()
    gold = load_gold_labels()

    # Get union of all instance IDs (only those with gold labels)
    all_ids = sorted(gold.keys())
    print(f"\nTotal instances with gold labels: {len(all_ids)}")

    # Build combined feature matrix
    rows = []
    for iid in all_ids:
        row = {"instance_id": iid, "gold_pass": gold[iid]}

        # Behavioral features (from tiny-judge)
        if not behavioral_df.empty and iid in behavioral_df["instance_id"].values:
            brow = behavioral_df[behavioral_df["instance_id"] == iid].iloc[0]
            # Copy all behavioral features except instance_id and gold_pass
            for col in behavioral_df.columns:
                if col not in ("instance_id", "gold_pass"):
                    val = brow[col]
                    row[f"beh_{col}"] = None if pd.isna(val) else val

        # v009 features (from full run)
        v009_data = v009.get(iid, {})
        for k, v in v009_data.items():
            row[k] = v

        # Debate features
        debate_data = debate.get(iid, {})
        for k, v in debate_data.items():
            row[k] = v

        # SVG features
        svg_data = svg.get(iid, {})
        for k, v in svg_data.items():
            row[k] = v

        # E_new1/E_new2 features
        enew_data = enew.get(iid, {})
        for k, v in enew_data.items():
            row[k] = v

        rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    print(f"\nCombined matrix: {len(df)} instances × {len(df.columns)} columns")

    # Coverage stats
    print("\nFeature coverage:")
    coverage = {}
    for col in df.columns:
        if col in ("instance_id", "gold_pass"):
            continue
        non_null = df[col].notna().sum()
        pct = non_null / len(df) * 100
        coverage[col] = {"non_null": int(non_null), "pct": round(pct, 1)}
        if pct < 100:
            print(f"  {col}: {non_null}/{len(df)} ({pct:.1f}%)")

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")

    # Summary
    summary = {
        "n_instances": len(df),
        "n_features": len(df.columns) - 2,  # minus instance_id, gold_pass
        "class_balance": {
            "pass": int(df["gold_pass"].sum()),
            "fail": int((1 - df["gold_pass"]).sum()),
        },
        "signal_coverage": {
            "behavioral": int(df.filter(like="beh_").notna().any(axis=1).sum()),
            "v009": int(df["v009_mean_score"].notna().sum()) if "v009_mean_score" in df else 0,
            "debate": int(df["debate_score"].notna().sum()) if "debate_score" in df else 0,
            "svg": int(df["svg_line_recall"].notna().sum()) if "svg_line_recall" in df else 0,
        },
        "feature_coverage": coverage,
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {SUMMARY_PATH}")

    return df


if __name__ == "__main__":
    main()
