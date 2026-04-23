#!/usr/bin/env python3
"""
Extract decision sequences from VP SWE-bench production experiment data.

Combines:
1. Per-issue tool telemetry JSONL (data/telemetry/*.jsonl) — VP tool invocations
2. Claude Code session summaries (data/telemetry/*_claude_code.json) — num_turns, cost
3. Gold labels (eval_report.json + eval_report_errors_v2.json) — pass/fail outcomes
4. Predictions (predictions_lite.jsonl) — patch diffs (fix generated or not)
5. Session JSONL files (data/sessions/) — full turn-by-turn action sequences

Produces: results/decision_sequences.json with per-issue pivot data.
"""

import json
import os
import sys
import re
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
VP_BASE = BASE.parent / "verification-primitives"
VP_SWEBENCH = BASE.parent / "verification-primitives-swebench"


def load_gold_labels():
    """Load gold pass/fail labels from eval reports."""
    labels = {}
    report_path = VP_SWEBENCH / "results" / "eval_report.json"
    with open(report_path) as f:
        report = json.load(f)
    for iid in report.get("resolved_ids", []):
        labels[iid] = "pass"
    for iid in report.get("completed_ids", []):
        if iid not in labels:
            labels[iid] = "fail"

    errors_path = VP_SWEBENCH / "results" / "eval_report_errors_v2.json"
    if errors_path.exists():
        with open(errors_path) as f:
            err_report = json.load(f)
        for iid in err_report.get("resolved_ids", []):
            labels[iid] = "pass"
        for iid in err_report.get("completed_ids", []):
            if iid not in labels:
                labels[iid] = "fail"
    return labels


def load_predictions():
    """Load predictions to determine which issues got patches."""
    predictions = {}
    pred_path = VP_SWEBENCH / "results" / "predictions_lite.jsonl"
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                iid = r["instance_id"]
                patch = r.get("model_patch", "")
                predictions[iid] = {
                    "fix_generated": bool(patch and patch.strip()),
                    "patch_len": len(patch) if patch else 0,
                }
    return predictions


def load_tool_telemetry():
    """Load per-issue VP tool telemetry from JSONL files."""
    telemetry_dir = BASE / "data" / "telemetry"
    records = {}

    for f in telemetry_dir.glob("*.jsonl"):
        instance_id = f.stem
        invocations = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        invocations.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        tools_used = list(set(inv["tool"] for inv in invocations))
        has_generate = "generate_tests" in tools_used
        has_run = "run_tests" in tools_used
        has_review = "adversarial_review" in tools_used

        if not tools_used:
            pattern = "ignore"
        elif has_generate and has_run and has_review:
            pattern = "full_pipeline"
        elif has_generate and has_run:
            run_count = sum(1 for t in invocations if t["tool"] == "run_tests")
            pattern = "gen_run_iterate" if run_count > 1 else "generate_run"
        elif has_generate:
            pattern = "generate_only"
        else:
            pattern = "other"

        # Check for revision after test failure
        revised_after_failure = False
        submitted_despite_failure = False
        for i, inv in enumerate(invocations):
            if inv["tool"] == "run_tests":
                outputs = inv.get("outputs", {})
                passed = outputs.get("passed", 0)
                total = outputs.get("total", 0)
                error = outputs.get("error", "")
                if passed == 0 and (total > 0 or error):
                    remaining = invocations[i + 1:]
                    if any(t["tool"] in ("generate_tests", "run_tests") for t in remaining):
                        revised_after_failure = True
                    else:
                        submitted_despite_failure = True

        review_verdict = ""
        review_score = -1.0
        for inv in invocations:
            if inv["tool"] == "adversarial_review":
                outputs = inv.get("outputs", {})
                review_verdict = outputs.get("verdict", "")
                review_score = outputs.get("overall_score", -1.0)

        records[instance_id] = {
            "tool_invocations_raw": invocations,
            "tools_used": tools_used,
            "num_tool_invocations": len(invocations),
            "composition_pattern": pattern,
            "revised_after_failure": revised_after_failure,
            "submitted_despite_failure": submitted_despite_failure,
            "review_verdict": review_verdict,
            "review_score": review_score,
            "verification_cost_usd": sum(inv.get("cost_usd", 0) for inv in invocations),
        }

    return records


def load_cc_summaries():
    """Load Claude Code session summaries."""
    summaries = {}
    for search_dir in [BASE / "data" / "telemetry", BASE / "data" / "cc_summaries"]:
        if not search_dir.exists():
            continue
        for f in search_dir.glob("*_claude_code.json"):
            instance_id = f.stem.replace("_claude_code", "")
            with open(f) as fh:
                data = json.load(fh)
            output = data.get("output", {})
            summaries[instance_id] = {
                "num_turns": output.get("num_turns", 0),
                "duration_ms": output.get("duration_ms", 0),
                "total_cost_usd": output.get("total_cost_usd", 0),
                "session_id": output.get("session_id", ""),
                "stop_reason": output.get("stop_reason", ""),
                "elapsed_s": data.get("elapsed_s", 0),
            }
    return summaries


def classify_action(tool_name):
    """Classify a Claude Code tool call into an action type."""
    tool_lower = tool_name.lower()
    if tool_lower in ("edit", "write", "notebookedit"):
        return "EDIT"
    if tool_lower in ("read", "glob", "grep"):
        return "EXPLORE"
    if tool_lower == "bash":
        return "BASH"
    if tool_lower == "agent":
        return "AGENT"
    return "OTHER"


def extract_from_session_jsonl(session_path):
    """Extract turn-by-turn action sequence from a Claude Code session JSONL."""
    actions = []
    action_counter = 0
    first_edit_action = None
    first_bash_action = None

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
            role = msg.get("role", event.get("type", ""))

            if role == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            action_counter += 1
                            tool_name = block.get("name", "unknown")
                            action_type = classify_action(tool_name)
                            actions.append({
                                "seq": action_counter,
                                "action": action_type,
                                "tool": tool_name,
                            })
                            if action_type == "EDIT" and first_edit_action is None:
                                first_edit_action = action_counter
                            if action_type == "BASH" and first_bash_action is None:
                                first_bash_action = action_counter

    total_actions = action_counter
    return {
        "total_actions": total_actions,
        "first_edit_action": first_edit_action,
        "first_bash_action": first_bash_action,
        "first_edit_pct": first_edit_action / total_actions if first_edit_action and total_actions > 0 else None,
        "num_edits": sum(1 for a in actions if a["action"] == "EDIT"),
        "num_explores": sum(1 for a in actions if a["action"] == "EXPLORE"),
        "num_bashes": sum(1 for a in actions if a["action"] == "BASH"),
        "action_type_sequence": [a["action"] for a in actions],
    }


def main():
    print("Loading gold labels...")
    labels = load_gold_labels()
    print(f"  {len(labels)} labels ({sum(1 for v in labels.values() if v == 'pass')} pass)")

    print("Loading predictions...")
    predictions = load_predictions()
    print(f"  {len(predictions)} predictions")

    print("Loading tool telemetry...")
    telemetry = load_tool_telemetry()
    print(f"  {len(telemetry)} issues with VP tool data")

    print("Loading CC summaries...")
    cc_summaries = load_cc_summaries()
    print(f"  {len(cc_summaries)} CC summaries")

    session_dir = BASE / "data" / "sessions"
    session_files = {}
    if session_dir.exists():
        for f in session_dir.glob("*.jsonl"):
            parts = f.stem.split("__")
            if len(parts) >= 3:
                instance_id = "__".join(parts[:2])
                session_files[instance_id] = f
        print(f"  {len(session_files)} session files")

    all_instance_ids = set(predictions.keys())
    print(f"\nTotal instances: {len(all_instance_ids)}")

    results = []
    for instance_id in sorted(all_instance_ids):
        pred = predictions.get(instance_id, {})
        cc = cc_summaries.get(instance_id, {})
        tool = telemetry.get(instance_id, {})

        entry = {
            "instance_id": instance_id,
            "gold_outcome": labels.get(instance_id, "unknown"),
            "fix_generated": pred.get("fix_generated", False),
            "patch_len": pred.get("patch_len", 0),

            # CC session summary
            "num_turns": cc.get("num_turns", 0),
            "duration_ms": cc.get("duration_ms", 0),
            "total_cost_usd": cc.get("total_cost_usd", 0),
            "elapsed_s": cc.get("elapsed_s", 0),

            # VP tool telemetry
            "tools_used": tool.get("tools_used", []),
            "num_tool_invocations": tool.get("num_tool_invocations", 0),
            "composition_pattern": tool.get("composition_pattern", "ignore"),
            "revised_after_failure": tool.get("revised_after_failure", False),
            "submitted_despite_failure": tool.get("submitted_despite_failure", False),
            "review_verdict": tool.get("review_verdict", ""),
            "review_score": tool.get("review_score", -1.0),
            "verification_cost_usd": tool.get("verification_cost_usd", 0),
        }

        # Session turn-by-turn data
        if instance_id in session_files:
            entry["session"] = extract_from_session_jsonl(session_files[instance_id])
        else:
            entry["session"] = None

        results.append(entry)

    # Save
    out_path = BASE / "results" / "decision_sequences.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExtracted {len(results)} decision sequences -> {out_path}")

    # Summary
    outcomes = defaultdict(int)
    patterns = defaultdict(lambda: defaultdict(int))
    for r in results:
        outcome = r["gold_outcome"]
        outcomes[outcome] += 1
        patterns[r["composition_pattern"]][outcome] += 1

    print(f"\nOutcome distribution: {dict(outcomes)}")
    print(f"\nPattern × Outcome:")
    for pat in sorted(patterns.keys()):
        counts = patterns[pat]
        total = counts.get("pass", 0) + counts.get("fail", 0)
        pass_rate = counts.get("pass", 0) / total if total > 0 else 0
        print(f"  {pat:25s}: n={total:3d}, pass={counts.get('pass', 0):3d}, "
              f"fail={counts.get('fail', 0):3d}, rate={pass_rate:.1%}")

    with_sessions = sum(1 for r in results if r["session"] is not None)
    print(f"\nSession data: {with_sessions}/{len(results)} issues")

    edit_pcts = [r["session"]["first_edit_pct"] for r in results
                 if r["session"] and r["session"]["first_edit_pct"] is not None]
    if edit_pcts:
        import numpy as np
        print(f"First edit timing: median={np.median(edit_pcts):.1%}, mean={np.mean(edit_pcts):.1%} (n={len(edit_pcts)})")


if __name__ == "__main__":
    main()
