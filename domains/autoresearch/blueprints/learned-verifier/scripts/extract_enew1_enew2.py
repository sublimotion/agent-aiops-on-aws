#!/usr/bin/env python3
"""
E_new1 + E_new2: Extract Read:Edit ratio and Recovery Breadth features
from Claude Code session JSONLs.

E_new1 features (stellaraccident-inspired):
  - read_edit_ratio: file Read calls / (Edit + Write) calls
  - edits_without_read_pct: % of Edit/Write calls where the target file
    was NOT Read in the preceding N=5 tool calls
  - write_vs_edit_ratio: Write calls / (Write + Edit) calls
    (Write = full file overwrite = precision loss signal)

E_new2 features (Claw-Eval-inspired):
  - recovery_breadth: |tool_types_recovered| / |tool_types_errored|
  - retry_without_change_rate: same (tool, args_hash) repeated / total calls
  - error_rate_early: error fraction in first 30% of trajectory
  - error_rate_late: error fraction in last 30% of trajectory

Sources:
  - pivot-analysis/data/sessions/*.jsonl (Claude Code turn-by-turn)

Output:
  - results/enew1_enew2_features.csv
"""

import json
import hashlib
import csv
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
PIVOT = BASE.parent / "pivot-analysis"
SESSION_DIR = PIVOT / "data" / "sessions"
OUTPUT = BASE / "results" / "enew1_enew2_features.csv"


def parse_session(session_path):
    """Parse a session JSONL into ordered lists of tool calls and results."""
    tool_calls = []  # (idx, name, input_dict, tool_use_id)
    tool_results = {}  # tool_use_id -> is_error

    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = ev.get("message", {})
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue

                if block.get("type") == "tool_use":
                    tool_calls.append({
                        "idx": len(tool_calls),
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                        "id": block.get("id", ""),
                    })
                elif block.get("type") == "tool_result":
                    tool_results[block.get("tool_use_id", "")] = {
                        "is_error": block.get("is_error", False),
                    }

    # Attach error status to each tool call
    for tc in tool_calls:
        result = tool_results.get(tc["id"], {})
        tc["is_error"] = result.get("is_error", False)

    return tool_calls


def get_file_path(tc):
    """Extract the target file path from a tool call."""
    inp = tc.get("input", {})
    return inp.get("file_path", inp.get("path", ""))


def args_hash(tc):
    """Hash the (tool_name, input) for retry detection."""
    key = json.dumps({"name": tc["name"], "input": tc["input"]}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def extract_enew1(tool_calls):
    """E_new1: Read:Edit ratio features."""
    read_tools = {"Read"}
    write_tools = {"Edit", "Write"}

    n_reads = sum(1 for tc in tool_calls if tc["name"] in read_tools)
    n_edits = sum(1 for tc in tool_calls if tc["name"] == "Edit")
    n_writes = sum(1 for tc in tool_calls if tc["name"] == "Write")
    n_mutations = n_edits + n_writes

    # Read:Edit ratio
    if n_mutations > 0:
        read_edit_ratio = n_reads / n_mutations
    else:
        read_edit_ratio = None  # no edits → ratio undefined

    # Edits-without-prior-read %
    # For each Edit/Write, check if the target file was Read in preceding N=5 calls
    LOOKBACK = 5
    edits_without_read = 0
    total_edits_with_file = 0

    for i, tc in enumerate(tool_calls):
        if tc["name"] not in write_tools:
            continue
        target = get_file_path(tc)
        if not target:
            continue

        total_edits_with_file += 1
        # Check preceding N calls for a Read of the same file
        found_read = False
        for j in range(max(0, i - LOOKBACK), i):
            prev = tool_calls[j]
            if prev["name"] in read_tools and get_file_path(prev):
                # Fuzzy match: same file if paths end the same way
                prev_path = get_file_path(prev)
                if prev_path == target or prev_path.endswith(target.split("/")[-1]):
                    found_read = True
                    break
        if not found_read:
            edits_without_read += 1

    if total_edits_with_file > 0:
        edits_without_read_pct = edits_without_read / total_edits_with_file
    else:
        edits_without_read_pct = None

    # Write vs Edit ratio (full overwrite vs surgical edit)
    if n_mutations > 0:
        write_vs_edit_ratio = n_writes / n_mutations
    else:
        write_vs_edit_ratio = None

    return {
        "enew1_read_edit_ratio": read_edit_ratio,
        "enew1_edits_without_read_pct": edits_without_read_pct,
        "enew1_write_vs_edit_ratio": write_vs_edit_ratio,
        "enew1_n_reads": n_reads,
        "enew1_n_edits": n_edits,
        "enew1_n_writes": n_writes,
    }


def extract_enew2(tool_calls):
    """E_new2: Recovery breadth features."""
    if not tool_calls:
        return {
            "enew2_recovery_breadth": None,
            "enew2_retry_without_change_rate": None,
            "enew2_error_rate_early": None,
            "enew2_error_rate_late": None,
            "enew2_total_errors": 0,
        }

    # Recovery breadth: for each tool type that had an error,
    # did a subsequent successful call of the same type occur?
    errored_types = set()
    recovered_types = set()
    type_had_success_after_error = defaultdict(bool)

    # Track per tool type: did it error, and was there a success after the error?
    type_first_error_idx = {}
    type_last_success_idx = {}

    for tc in tool_calls:
        name = tc["name"]
        idx = tc["idx"]
        if tc["is_error"]:
            errored_types.add(name)
            if name not in type_first_error_idx:
                type_first_error_idx[name] = idx
        else:
            type_last_success_idx[name] = idx

    for t in errored_types:
        first_err = type_first_error_idx.get(t, float("inf"))
        last_ok = type_last_success_idx.get(t, -1)
        if last_ok > first_err:
            recovered_types.add(t)

    if errored_types:
        recovery_breadth = len(recovered_types) / len(errored_types)
    else:
        recovery_breadth = 1.0  # no errors = fully recovered (trivially)

    # Retry-without-change rate: same (tool, args) hash repeated consecutively
    n_retries = 0
    prev_hash = None
    for tc in tool_calls:
        h = args_hash(tc)
        if h == prev_hash:
            n_retries += 1
        prev_hash = h

    retry_rate = n_retries / len(tool_calls) if tool_calls else 0

    # Error rate by phase
    n = len(tool_calls)
    early_cutoff = int(n * 0.3)
    late_start = int(n * 0.7)

    early_calls = tool_calls[:early_cutoff] if early_cutoff > 0 else []
    late_calls = tool_calls[late_start:] if late_start < n else []

    early_errors = sum(1 for tc in early_calls if tc["is_error"])
    late_errors = sum(1 for tc in late_calls if tc["is_error"])

    error_rate_early = early_errors / len(early_calls) if early_calls else 0
    error_rate_late = late_errors / len(late_calls) if late_calls else 0

    total_errors = sum(1 for tc in tool_calls if tc["is_error"])

    return {
        "enew2_recovery_breadth": recovery_breadth,
        "enew2_retry_without_change_rate": retry_rate,
        "enew2_error_rate_early": error_rate_early,
        "enew2_error_rate_late": error_rate_late,
        "enew2_total_errors": total_errors,
    }


def main():
    print("Scanning session files...")
    session_files = {}
    for f in SESSION_DIR.glob("*.jsonl"):
        # Format: {instance_id}__{session_uuid}.jsonl
        parts = f.stem.split("__")
        if len(parts) >= 3:
            iid = "__".join(parts[:2])
            session_files[iid] = f

    print(f"  Found {len(session_files)} session files")

    rows = []
    for iid in sorted(session_files.keys()):
        sp = session_files[iid]
        tool_calls = parse_session(sp)

        if not tool_calls:
            continue

        row = {"instance_id": iid}
        row.update(extract_enew1(tool_calls))
        row.update(extract_enew2(tool_calls))
        rows.append(row)

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nExtracted {len(rows)} instances × {len(fieldnames)} features → {OUTPUT}")

    # Quick stats
    import numpy as np
    for feat in ["enew1_read_edit_ratio", "enew1_edits_without_read_pct",
                 "enew1_write_vs_edit_ratio", "enew2_recovery_breadth",
                 "enew2_retry_without_change_rate"]:
        vals = [r[feat] for r in rows if r[feat] is not None]
        if vals:
            arr = np.array(vals)
            print(f"  {feat}: mean={arr.mean():.3f}, median={np.median(arr):.3f}, "
                  f"std={arr.std():.3f}, n={len(vals)}")


if __name__ == "__main__":
    main()
