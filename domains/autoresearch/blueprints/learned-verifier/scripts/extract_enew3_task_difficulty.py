#!/usr/bin/env python3
"""
E_new3: Task-Conditioned Behavioral Features.

Calls Haiku to assess task difficulty for each of 300 SWE-bench Lite issues,
then creates conditioned features (behavioral_signal / difficulty).

Resolves the cross-sectional vs longitudinal paradox (Simpson's Paradox):
- stellaraccident (longitudinal): fewer reads = degraded agent
- CoderForge (cross-sectional): more reads = failing agent (hard tasks)
- Resolution: normalize by task difficulty

Outputs:
  - results/enew3_task_difficulty.jsonl (raw LLM assessments)
  - results/enew3_conditioned_features.csv (conditioned feature matrix)
"""

import json
import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DEBATE = BASE.parent / "debate-verification"
SWEBENCH_PATH = DEBATE / "results" / "swebench_lite.jsonl"
OUTPUT_RAW = BASE / "results" / "enew3_task_difficulty.jsonl"
OUTPUT_CSV = BASE / "results" / "enew3_conditioned_features.csv"

# Haiku via Bedrock
try:
    import boto3
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    HAS_BEDROCK = True
except Exception:
    HAS_BEDROCK = False

HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

DIFFICULTY_PROMPT = """You are an expert software engineer assessing the difficulty of a GitHub issue for an AI coding agent to solve.

Given the issue description below, rate the following dimensions on a scale of 1-5 (1=easiest, 5=hardest):

1. **complexity**: How complex is the underlying bug/feature? (1=typo/off-by-one, 5=deep architectural issue)
2. **domain_knowledge**: How much domain-specific knowledge is needed? (1=basic Python, 5=deep framework internals)
3. **files_scope**: How many files likely need modification? (1=single file, 5=many files across modules)
4. **modification_scope**: How extensive are the changes? (1=one-liner, 5=significant refactoring)
5. **testing_clarity**: How clear is what a correct fix looks like from the description? (1=very clear, 5=ambiguous)
6. **debugging_difficulty**: How hard is it to locate the bug? (1=traceback points to it, 5=requires deep investigation)
7. **edge_case_risk**: How many edge cases might the fix miss? (1=none, 5=many subtle edge cases)
8. **regression_risk**: How likely is it that a fix breaks something else? (1=isolated, 5=highly coupled code)

Respond with ONLY a JSON object, no explanation. Example format:
{{"complexity": 3, "domain_knowledge": 2, "files_scope": 1, "modification_scope": 2, "testing_clarity": 3, "debugging_difficulty": 2, "edge_case_risk": 2, "regression_risk": 1}}

Issue:
{problem_statement}"""


def call_haiku(problem_statement, instance_id):
    """Call Haiku to assess task difficulty."""
    prompt = DIFFICULTY_PROMPT.format(
        problem_statement=problem_statement[:4000]  # cap to avoid huge inputs
    )

    try:
        resp = bedrock.invoke_model(
            modelId=HAIKU_MODEL,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.0,
            }),
        )
        body = json.loads(resp["body"].read())
        text = body["content"][0]["text"].strip()

        # Parse JSON from response
        # Handle potential markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        scores = json.loads(text)

        # Validate
        dims = ["complexity", "domain_knowledge", "files_scope", "modification_scope",
                "testing_clarity", "debugging_difficulty", "edge_case_risk", "regression_risk"]
        for d in dims:
            if d not in scores or not isinstance(scores[d], (int, float)):
                scores[d] = 3  # default mid-point

        # Compute composite difficulty (mean of all dimensions)
        scores["difficulty_mean"] = np.mean([scores[d] for d in dims])

        input_tokens = body.get("usage", {}).get("input_tokens", 0)
        output_tokens = body.get("usage", {}).get("output_tokens", 0)
        scores["_input_tokens"] = input_tokens
        scores["_output_tokens"] = output_tokens

        return scores

    except Exception as e:
        print(f"  ERROR {instance_id}: {e}")
        return None


def main():
    print("Loading problem statements...")
    problems = {}
    with open(SWEBENCH_PATH) as f:
        for line in f:
            r = json.loads(line.strip())
            problems[r["instance_id"]] = r["problem_statement"]
    print(f"  {len(problems)} problem statements")

    # Load existing results for resume
    existing = {}
    if OUTPUT_RAW.exists():
        with open(OUTPUT_RAW) as f:
            for line in f:
                r = json.loads(line.strip())
                existing[r["instance_id"]] = r
        print(f"  Resuming: {len(existing)} already assessed")

    if not HAS_BEDROCK:
        print("ERROR: boto3/bedrock not available")
        return

    # Assess remaining issues
    todo = [iid for iid in sorted(problems.keys()) if iid not in existing]
    print(f"  {len(todo)} issues to assess")

    total_cost = 0
    with open(OUTPUT_RAW, "a") as out:
        for i, iid in enumerate(todo):
            scores = call_haiku(problems[iid], iid)
            if scores is None:
                continue

            row = {"instance_id": iid, **scores}
            out.write(json.dumps(row) + "\n")
            existing[iid] = row

            # Cost: Haiku input $0.80/MTok, output $4.00/MTok
            cost = (scores.get("_input_tokens", 0) * 0.80 + scores.get("_output_tokens", 0) * 4.00) / 1_000_000
            total_cost += cost

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(todo)}] cost so far: ${total_cost:.2f}")

    print(f"  Total assessment cost: ${total_cost:.2f}")
    print(f"  Results: {OUTPUT_RAW}")

    # Now build conditioned features
    print("\nBuilding conditioned features...")

    # Load behavioral + E_new features
    combined = pd.read_csv(BASE / "results" / "combined_features.csv")

    # Load difficulty scores
    difficulty = {}
    with open(OUTPUT_RAW) as f:
        for line in f:
            r = json.loads(line.strip())
            difficulty[r["instance_id"]] = r

    print(f"  {len(difficulty)} difficulty assessments")

    # Build conditioned features
    dims = ["complexity", "domain_knowledge", "files_scope", "modification_scope",
            "testing_clarity", "debugging_difficulty", "edge_case_risk", "regression_risk"]

    rows = []
    for _, row in combined.iterrows():
        iid = row["instance_id"]
        diff = difficulty.get(iid, {})

        if not diff:
            rows.append({"instance_id": iid})
            continue

        beta = diff.get("difficulty_mean", 3.0)
        out = {"instance_id": iid}

        # Raw difficulty dimensions
        for d in dims:
            out[f"task_{d}"] = diff.get(d)
        out["task_difficulty_mean"] = beta

        # Conditioned features: normalize behavioral signals by difficulty
        # Higher difficulty → more reads/cost is expected → conditioned value is lower
        if beta > 0:
            if pd.notna(row.get("enew1_read_edit_ratio")):
                out["cond_read_edit_ratio"] = row["enew1_read_edit_ratio"] / beta
            if pd.notna(row.get("beh_total_cost_usd")):
                out["cond_cost"] = row["beh_total_cost_usd"] / beta
            if pd.notna(row.get("beh_loop_count")):
                out["cond_loop_count"] = row["beh_loop_count"] / beta
            if pd.notna(row.get("beh_tokens_per_edit")):
                out["cond_tokens_per_edit"] = row["beh_tokens_per_edit"] / beta
            if pd.notna(row.get("beh_action_pct_edit")):
                out["cond_action_pct_edit"] = row["beh_action_pct_edit"] * beta
            if pd.notna(row.get("enew1_n_reads")):
                out["cond_n_reads"] = row["enew1_n_reads"] / beta
            if pd.notna(row.get("enew2_total_errors")):
                out["cond_total_errors"] = row["enew2_total_errors"] / beta

        rows.append(out)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"  Conditioned features: {len(df_out)} × {len(df_out.columns)} → {OUTPUT_CSV}")

    # Quick stats
    for col in ["task_difficulty_mean", "cond_read_edit_ratio", "cond_cost"]:
        if col in df_out.columns:
            vals = df_out[col].dropna()
            print(f"  {col}: mean={vals.mean():.3f}, std={vals.std():.3f}")


if __name__ == "__main__":
    main()
