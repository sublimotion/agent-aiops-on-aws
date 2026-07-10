#!/usr/bin/env python3
"""Offline drift-vs-outcome analysis on existing SWE-bench traces.

The free retrospective gate before any live experiment: on 300 real Claude Code
SWE-bench sessions with known pass/fail labels (pivot-analysis blueprint), does
the lineage drift signal correlate with failure? If failing traces carry more
drift flags than passing ones, the detector has predictive validity and the
reflect experiment is worth running. If not, stop before spending on live runs.

This is retrospective CORRELATION, not causal proof — only a live reflect arm
shows feedback *fixes* drift. But it's the correct, zero-cost go/no-go.

For each session: run lineage.py's detectors → count drift flags + trace shape
features; join to the telemetry `success` label; report drift-rate by outcome +
a couple of association stats. Also reports coverage caveats (Bash-only editing).

Usage:
  offline_drift_analysis.py \
    --sessions .../pivot-analysis/data/sessions \
    --telemetry .../pivot-analysis/data/telemetry \
    --lineage .../trace-effectiveness/lineage.py \
    [--json out.json]
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter


def load_lineage(path):
    """Import lineage.py as a module so we reuse its detectors verbatim."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("lineage", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_UUID_RE = re.compile(
    r"__[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\.jsonl$")


def issue_id_from_session(fname):
    """Session files are `[run-prefix-]<issue_id>__<uuid>.jsonl`. The SWE-bench
    issue id is `<org>__<repo>-<num>` (contains one `__`). Strip the trailing
    `__<uuid>.jsonl`, then strip any leading run-prefix before the org token."""
    base = os.path.basename(fname)
    stem = _UUID_RE.sub("", base)   # remove trailing __<uuid>.jsonl
    # stem is now `[prefix-]<org>__<repo>-<num>`. Drop a leading prefix that ends
    # right before the canonical `<org>__<repo>-<num>` tail.
    m = re.search(r"([A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*__[A-Za-z0-9._-]+-\d+)$", stem)
    return m.group(1) if m else stem


def telemetry_label(telemetry_dir, issue_id):
    """Find the success label for an issue. Files are `<issue_id>_claude_code.json`."""
    cand = os.path.join(telemetry_dir, f"{issue_id}_claude_code.json")
    if os.path.exists(cand):
        try:
            return json.load(open(cand)).get("success")
        except Exception:
            return None
    # fallback: fuzzy match on issue id substring
    for f in glob.glob(os.path.join(telemetry_dir, "*_claude_code.json")):
        if issue_id in os.path.basename(f):
            try:
                return json.load(open(f)).get("success")
            except Exception:
                return None
    return None


def bash_edit_signal(session_path):
    """Count Bash commands that mutate files (sed -i / patch / tee / redirect / >).
    SWE-bench agents often edit via Bash, invisible to the file-event detector —
    we surface this as a coverage caveat, not a drift signal."""
    n_bash_edit = 0
    for line in open(session_path, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "assistant":
            continue
        m = d.get("message", {})
        for b in (m.get("content") or []) if isinstance(m, dict) else []:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Bash":
                cmd = (b.get("input") or {}).get("command", "")
                if re.search(r"\bsed\s+-i|\bpatch\b|\btee\b|>>?\s|\bcat\s*>|\bapply", cmd):
                    n_bash_edit += 1
    return n_bash_edit


def analyze(sessions_dir, telemetry_dir, lineage_mod):
    rows = []
    for sp in sorted(glob.glob(os.path.join(sessions_dir, "*.jsonl"))):
        issue = issue_id_from_session(sp)
        label = telemetry_label(telemetry_dir, issue)
        events = lineage_mod.load_session(sp)
        per_file = lineage_mod.build_lineage(events)
        ref_flags = lineage_mod.detect_drift(per_file)
        val_flags = lineage_mod.detect_value_drift(events)
        n_edit = sum(1 for e in events if e["action"] == "edit")
        n_write = sum(1 for e in events if e["action"] == "write")
        n_read = sum(1 for e in events if e["action"] == "read")
        rows.append({
            "issue": issue,
            "success": label,               # True/False/None
            "n_events": len([e for e in events if e["action"] != "search"]),
            "n_files": len(per_file),
            "n_read": n_read, "n_write": n_write, "n_edit": n_edit,
            "ref_drift": len(ref_flags),
            "val_drift": len(val_flags),
            "any_drift": len(ref_flags) + len(val_flags) > 0,
            "bash_edits": bash_edit_signal(sp),
        })
    return rows


def summarize(rows):
    labeled = [r for r in rows if r["success"] in (True, False)]
    passed = [r for r in labeled if r["success"]]
    failed = [r for r in labeled if not r["success"]]

    def mean(xs, k):
        xs = [r[k] for r in xs]
        return sum(xs) / len(xs) if xs else 0.0

    def rate(xs, k):
        xs = [r for r in xs if r[k]]
        return len(xs)

    out = {}
    out["n_sessions"] = len(rows)
    out["n_labeled"] = len(labeled)
    out["n_pass"] = len(passed)
    out["n_fail"] = len(failed)
    out["base_fail_rate"] = round(len(failed) / len(labeled), 4) if labeled else None

    # File-event coverage caveat: how many sessions edited via file tools at all?
    file_editors = [r for r in labeled if r["n_edit"] + r["n_write"] > 0]
    out["pct_sessions_with_file_edits"] = round(100 * len(file_editors) / len(labeled), 1) if labeled else 0
    out["pct_sessions_bash_only_edits"] = round(
        100 * len([r for r in labeled if r["n_edit"] + r["n_write"] == 0 and r["bash_edits"] > 0]) / len(labeled), 1
    ) if labeled else 0

    # The core question: drift rate by outcome.
    def drift_rate(xs):
        return round(100 * rate(xs, "any_drift") / len(xs), 1) if xs else 0.0
    out["drift_rate_pass_pct"] = drift_rate(passed)
    out["drift_rate_fail_pct"] = drift_rate(failed)
    out["mean_ref_drift_pass"] = round(mean(passed, "ref_drift"), 3)
    out["mean_ref_drift_fail"] = round(mean(failed, "ref_drift"), 3)
    out["mean_val_drift_pass"] = round(mean(passed, "val_drift"), 3)
    out["mean_val_drift_fail"] = round(mean(failed, "val_drift"), 3)

    # 2x2 association: drift present vs outcome (Fisher-ish via odds ratio).
    a = len([r for r in failed if r["any_drift"]])      # fail & drift
    b = len([r for r in failed if not r["any_drift"]])  # fail & no drift
    c = len([r for r in passed if r["any_drift"]])      # pass & drift
    d = len([r for r in passed if not r["any_drift"]])  # pass & no drift
    out["contingency"] = {"fail_drift": a, "fail_nodrift": b, "pass_drift": c, "pass_nodrift": d}
    # odds ratio (drift → fail); +0.5 continuity correction
    orr = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    out["odds_ratio_drift_to_fail"] = round(orr, 3)
    # P(fail | drift) vs P(fail | no drift)
    out["p_fail_given_drift"] = round(a / (a + c), 4) if (a + c) else None
    out["p_fail_given_nodrift"] = round(b / (b + d), 4) if (b + d) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--lineage", required=True, help="path to lineage.py")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    lin = load_lineage(args.lineage)
    rows = analyze(args.sessions, args.telemetry, lin)
    summary = summarize(rows)

    print(json.dumps(summary, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2)
        print(f"\nWrote per-session rows to {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
