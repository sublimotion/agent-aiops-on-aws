#!/usr/bin/env python3
"""
Phase 1: Extract feature matrix for tiny-judge from VP SWE-bench data.

Sources:
- pivot-analysis/results/decision_sequences.json (pre-extracted per-issue data)
- pivot-analysis/data/telemetry/*.jsonl (VP tool invocations)
- pivot-analysis/data/telemetry/*_claude_code.json (CC session summaries)
- pivot-analysis/data/sessions/*.jsonl (full turn-by-turn sessions)
- verification-primitives-swebench/results/eval_report*.json (gold labels)
- verifier-reward/results/*v009*.jsonl (v009 labels where available)

Outputs:
- results/features.csv — feature matrix (1 row per instance)
- results/feature_summary.json — feature distributions + stats
"""

import json
import csv
import glob
import re
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

BASE = Path(__file__).resolve().parent.parent
PIVOT = BASE.parent / "pivot-analysis"
VP_SWEBENCH = BASE.parent / "verification-primitives-swebench"
VERIFIER = BASE.parent / "verifier-reward"


def load_decision_sequences():
    """Load pre-extracted decision sequences from pivot analysis."""
    path = PIVOT / "results" / "decision_sequences.json"
    with open(path) as f:
        return {d["instance_id"]: d for d in json.load(f)}


def load_v009_labels():
    """Load v009 verdicts from verifier-reward results. Use sweep_holdout_v009_opus as primary."""
    v009 = {}

    # Load all v009 files, prefer opus, accumulate
    for pattern in ["sweep_holdout_v009_opus.jsonl", "sweep_holdout_v009_haiku.jsonl",
                    "sweep_iter6_v009.jsonl", "iter45_decomposed_v009.jsonl"]:
        path = VERIFIER / "results" / pattern
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                iid = r.get("instance_id", "")
                if iid and iid not in v009:
                    verdict = r.get("verdict", r.get("v009_verdict", ""))
                    score = r.get("overall_score", r.get("v009_score", r.get("score", None)))
                    if verdict or score is not None:
                        v009[iid] = {
                            "verdict": verdict,
                            "score": float(score) if score is not None else None,
                        }

    return v009


def load_telemetry_details():
    """Load detailed per-issue VP tool telemetry for additional features."""
    telemetry_dir = PIVOT / "data" / "telemetry"
    details = {}

    for fp in telemetry_dir.glob("*.jsonl"):
        iid = fp.stem
        invocations = []
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        invocations.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not invocations:
            continue

        # Extract additional features
        test_pass_total = 0
        test_fail_total = 0
        test_error_count = 0
        generate_count = 0
        run_count = 0
        review_scores = []

        for inv in invocations:
            tool = inv.get("tool", "")
            outputs = inv.get("outputs", {})

            if tool == "generate_tests":
                generate_count += 1
            elif tool == "run_tests":
                run_count += 1
                test_pass_total += outputs.get("passed", 0)
                test_fail_total += outputs.get("failed", 0)
                if outputs.get("error"):
                    test_error_count += 1
            elif tool == "adversarial_review":
                score = outputs.get("overall_score")
                if score is not None:
                    review_scores.append(float(score))

        details[iid] = {
            "generate_count": generate_count,
            "run_count": run_count,
            "test_pass_total": test_pass_total,
            "test_fail_total": test_fail_total,
            "test_error_count": test_error_count,
            "review_scores": review_scores,
            "total_tool_cost": sum(inv.get("cost_usd", 0) for inv in invocations),
        }

    return details


def extract_session_features(session_data):
    """Extract behavioral features from session turn-by-turn data."""
    if not session_data:
        return {}

    total_actions = session_data.get("total_actions", 0)
    if total_actions == 0:
        return {}

    num_edits = session_data.get("num_edits", 0)
    num_explores = session_data.get("num_explores", 0)
    num_bashes = session_data.get("num_bashes", 0)

    action_seq = session_data.get("action_type_sequence", [])

    # Compute loop count (repeated consecutive same-action sequences)
    loop_count = 0
    if len(action_seq) >= 2:
        for i in range(1, len(action_seq)):
            if action_seq[i] == action_seq[i - 1]:
                loop_count += 1

    # Compute action transition entropy
    transitions = defaultdict(int)
    for i in range(1, len(action_seq)):
        transitions[(action_seq[i - 1], action_seq[i])] += 1
    total_trans = sum(transitions.values())
    entropy = 0.0
    if total_trans > 0:
        for count in transitions.values():
            p = count / total_trans
            if p > 0:
                entropy -= p * math.log2(p)

    return {
        "total_actions": total_actions,
        "num_edits": num_edits,
        "num_explores": num_explores,
        "num_bashes": num_bashes,
        "action_pct_edit": num_edits / total_actions,
        "action_pct_search": num_explores / total_actions,
        "action_pct_bash": num_bashes / total_actions,
        "first_edit_pct": session_data.get("first_edit_pct"),
        "first_edit_action": session_data.get("first_edit_action"),
        "loop_count": loop_count,
        "action_entropy": entropy,
    }


def extract_token_features(session_path):
    """Extract token-level telemetry from session JSONL.

    Returns dict with universal token metrics — the actual resource signals
    behind beh_total_cost_usd. Cost is just tokens * price; these are the
    model-agnostic primitives.
    """
    if not session_path or not Path(session_path).exists():
        return {}

    total_input = 0
    total_output = 0
    total_cache_create = 0
    total_cache_read = 0
    total_thinking_chars = 0
    n_thinking_blocks = 0
    n_edits = 0
    n_turns = 0

    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = event.get("message", {})
            usage = msg.get("usage", {})

            if usage:
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
                total_cache_create += usage.get("cache_creation_input_tokens", 0)
                total_cache_read += usage.get("cache_read_input_tokens", 0)
                n_turns += 1

            # Count thinking blocks and chars (extended thinking / chain-of-thought)
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "thinking":
                            text = block.get("thinking", "")
                            total_thinking_chars += len(text)
                            n_thinking_blocks += 1
                        elif block.get("type") == "tool_use" and block.get("name") in ("Edit", "Write"):
                            n_edits += 1

    if n_turns == 0:
        return {}

    # Estimate thinking tokens (~4 chars per token for English text)
    est_thinking_tokens = total_thinking_chars // 4

    total_tokens = total_input + total_output + total_cache_create + total_cache_read
    # Context window approximation: Sonnet 4.6 has 200K context
    max_context = 200_000

    features = {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cache_create_tokens": total_cache_create,
        "total_cache_read_tokens": total_cache_read,
        "total_thinking_tokens": est_thinking_tokens,
        "total_tokens": total_tokens,
        "n_thinking_blocks": n_thinking_blocks,
    }

    # Derived ratios
    if total_tokens > 0:
        features["token_efficiency_ratio"] = total_output / total_tokens
        features["cache_hit_ratio"] = total_cache_read / total_tokens
    else:
        features["token_efficiency_ratio"] = None
        features["cache_hit_ratio"] = None

    features["token_budget_utilization"] = total_tokens / max_context if max_context > 0 else None

    if n_edits > 0:
        features["tokens_per_edit"] = total_tokens / n_edits
    else:
        features["tokens_per_edit"] = float(total_tokens) if total_tokens > 0 else None

    if total_output > 0:
        features["thinking_to_output_ratio"] = est_thinking_tokens / total_output
    else:
        features["thinking_to_output_ratio"] = None

    return features


def compute_context_growth_rate(session_path):
    """Estimate context growth from session JSONL by tracking message sizes."""
    if not session_path or not Path(session_path).exists():
        return None

    msg_sizes = []
    cumulative = 0

    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = event.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, list):
                size = sum(len(json.dumps(b)) for b in content)
            elif isinstance(content, str):
                size = len(content)
            else:
                continue

            cumulative += size
            msg_sizes.append(cumulative)

    if len(msg_sizes) < 3:
        return None

    # Context growth rate = slope of cumulative size over message index
    x = np.arange(len(msg_sizes))
    y = np.array(msg_sizes, dtype=float)
    # Linear regression slope
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def main():
    print("Loading decision sequences...")
    ds = load_decision_sequences()
    print(f"  {len(ds)} instances")

    print("Loading v009 labels...")
    v009 = load_v009_labels()
    print(f"  {len(v009)} v009 labels loaded")

    print("Loading telemetry details...")
    telemetry = load_telemetry_details()
    print(f"  {len(telemetry)} instances with tool details")

    # Find session files for context growth
    session_dir = PIVOT / "data" / "sessions"
    session_files = {}
    if session_dir.exists():
        for f in session_dir.glob("*.jsonl"):
            parts = f.stem.split("__")
            if len(parts) >= 3:
                iid = "__".join(parts[:2])
                session_files[iid] = f

    print(f"  {len(session_files)} session files")

    # Build feature matrix
    print("\nExtracting features...")
    features = []
    feature_names = None

    for iid, d in sorted(ds.items()):
        row = {"instance_id": iid}

        # Gold outcome (target)
        outcome = d.get("gold_outcome", "unknown")
        if outcome == "unknown":
            continue  # Skip unevaluated instances
        row["gold_pass"] = 1 if outcome == "pass" else 0

        # Patch features
        row["fix_generated"] = 1 if d.get("fix_generated") else 0
        row["patch_len"] = d.get("patch_len", 0)
        row["diff_size_chars"] = d.get("patch_len", 0)

        # Count files modified from patch
        patch = ""
        pred_path = VP_SWEBENCH / "results" / "predictions_lite.jsonl"
        # We already have patch_len, estimate files_modified from it
        # More accurate: parse the diff
        row["files_modified"] = None  # Will fill below

        # CC session features
        row["num_turns"] = d.get("num_turns", 0)
        row["total_cost_usd"] = d.get("total_cost_usd", 0)
        row["elapsed_s"] = d.get("elapsed_s", 0)

        # VP tool features
        row["tool_count"] = d.get("num_tool_invocations", 0)
        row["tool_used"] = 1 if d.get("num_tool_invocations", 0) > 0 else 0

        # Composition pattern (one-hot)
        pattern = d.get("composition_pattern", "ignore")
        for p in ["ignore", "generate_run", "gen_run_iterate", "full_pipeline", "generate_only", "other"]:
            row[f"comp_{p}"] = 1 if pattern == p else 0

        row["adversarial_review_used"] = 1 if "adversarial_review" in d.get("tools_used", []) else 0
        row["revised_after_failure"] = 1 if d.get("revised_after_failure") else 0
        row["submitted_despite_failure"] = 1 if d.get("submitted_despite_failure") else 0

        # v009 features
        v009_data = v009.get(iid, {})
        # Also check inline review from VP telemetry
        review_verdict = d.get("review_verdict", "")
        review_score = d.get("review_score")

        if v009_data.get("verdict"):
            row["v009_verdict"] = 1 if v009_data["verdict"] in ("likely_correct", "correct") else 0
            row["v009_confidence"] = v009_data.get("score")
        elif review_verdict:
            row["v009_verdict"] = 1 if review_verdict in ("likely_correct", "correct") else 0
            row["v009_confidence"] = review_score if review_score and review_score >= 0 else None
        else:
            row["v009_verdict"] = None
            row["v009_confidence"] = None

        # Telemetry detail features
        td = telemetry.get(iid, {})
        row["generate_count"] = td.get("generate_count", 0)
        row["run_count"] = td.get("run_count", 0)
        row["test_pass_total"] = td.get("test_pass_total", 0)
        row["test_fail_total"] = td.get("test_fail_total", 0)
        row["test_error_count"] = td.get("test_error_count", 0)
        review_scores = td.get("review_scores", [])
        row["review_score_mean"] = float(np.mean(review_scores)) if review_scores else None
        row["review_score_max"] = float(np.max(review_scores)) if review_scores else None

        # Session behavioral features
        session = d.get("session", {})
        sf = extract_session_features(session) if session else {}
        row["total_actions"] = sf.get("total_actions", row["num_turns"])
        row["action_pct_edit"] = sf.get("action_pct_edit")
        row["action_pct_search"] = sf.get("action_pct_search")
        row["action_pct_bash"] = sf.get("action_pct_bash")
        row["first_edit_pct"] = sf.get("first_edit_pct")
        row["first_edit_action"] = sf.get("first_edit_action")
        row["loop_count"] = sf.get("loop_count", 0)
        row["action_entropy"] = sf.get("action_entropy")

        # Parkinson ratio
        if sf.get("first_edit_pct") is not None:
            row["parkinson_ratio"] = sf["first_edit_pct"]
        else:
            row["parkinson_ratio"] = None

        # Token-level telemetry (the real signal behind cost)
        if iid in session_files:
            tf = extract_token_features(session_files[iid])
            for key in ["total_input_tokens", "total_output_tokens",
                        "total_cache_create_tokens", "total_cache_read_tokens",
                        "total_thinking_tokens", "total_tokens",
                        "n_thinking_blocks", "token_efficiency_ratio",
                        "cache_hit_ratio", "token_budget_utilization",
                        "tokens_per_edit", "thinking_to_output_ratio"]:
                row[key] = tf.get(key)
        else:
            for key in ["total_input_tokens", "total_output_tokens",
                        "total_cache_create_tokens", "total_cache_read_tokens",
                        "total_thinking_tokens", "total_tokens",
                        "n_thinking_blocks", "token_efficiency_ratio",
                        "cache_hit_ratio", "token_budget_utilization",
                        "tokens_per_edit", "thinking_to_output_ratio"]:
                row[key] = None

        features.append(row)

    # Fill files_modified from predictions
    print("Counting files modified per patch...")
    pred_path = VP_SWEBENCH / "results" / "predictions_lite.jsonl"
    patches = {}
    with open(pred_path) as f:
        for line in f:
            r = json.loads(line.strip())
            patch = r.get("model_patch", "")
            files = set(re.findall(r'diff --git a/(\S+)', patch))
            patches[r["instance_id"]] = len(files)

    for row in features:
        row["files_modified"] = patches.get(row["instance_id"], 0)

    # Compute context growth rate (sample — expensive for all 300)
    print("Computing context growth rates...")
    for row in features:
        iid = row["instance_id"]
        if iid in session_files:
            rate = compute_context_growth_rate(session_files[iid])
            row["context_growth_rate"] = rate
        else:
            row["context_growth_rate"] = None

    # Save CSV
    if not features:
        print("ERROR: No features extracted!")
        return

    out_path = BASE / "results" / "features.csv"
    fieldnames = list(features[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)

    print(f"\nFeature matrix: {len(features)} instances × {len(fieldnames)} features -> {out_path}")

    # Summary stats
    summary = {
        "n_instances": len(features),
        "n_features": len(fieldnames) - 2,  # minus instance_id and gold_pass
        "class_balance": {
            "pass": sum(1 for r in features if r["gold_pass"] == 1),
            "fail": sum(1 for r in features if r["gold_pass"] == 0),
        },
        "feature_coverage": {},
    }

    for fname in fieldnames:
        if fname in ("instance_id", "gold_pass"):
            continue
        non_null = sum(1 for r in features if r[fname] is not None)
        summary["feature_coverage"][fname] = {
            "non_null": non_null,
            "coverage_pct": non_null / len(features),
        }

    # v009 coverage
    v009_count = sum(1 for r in features if r["v009_verdict"] is not None)
    summary["v009_coverage"] = v009_count

    summary_path = BASE / "results" / "feature_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary: {summary_path}")
    print(f"  Class balance: {summary['class_balance']}")
    print(f"  v009 coverage: {v009_count}/{len(features)}")
    print(f"  Features with 100% coverage: "
          f"{sum(1 for v in summary['feature_coverage'].values() if v['coverage_pct'] == 1.0)}")
    print(f"  Features with <50% coverage: "
          f"{sum(1 for v in summary['feature_coverage'].values() if v['coverage_pct'] < 0.5)}")


if __name__ == "__main__":
    main()
