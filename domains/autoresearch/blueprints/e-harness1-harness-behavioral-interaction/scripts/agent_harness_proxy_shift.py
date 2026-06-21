#!/usr/bin/env python3
"""
Partition (b): distribution shift across the 3 harnesses that actually ran
(SERA / LangGraph / Aider), per the agent-harness phase2 eval.

IMPORTANT LIMITATION: these phase2 files carry ONLY trajectory metadata
(turns_used, edit_count, repeat_count, first_edit_turn, diff_size, fix_generated)
— they have NO Phase-3 RF features (cost / tokens_per_edit / loop_count) and NO
gold pass/fail labels. So this is a DISTRIBUTION-SHIFT report only; no RF re-fit
and no AUC is possible on partition (b). The RF interaction analysis lives in
analyze_harness_interaction.py (partition (a), the VP composition axis).

Proxy mapping to the 4 spec behavioral features:
  loop_count   -> repeat_count (SERA only; absent for LangGraph/Aider)
  tool/edit %  -> edit_count, fix_generated rate
  (cost / tokens_per_edit unavailable)
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
AH = HERE.parent.parent / "agent-harness" / "results"

HARNESSES = ["sera", "langgraph", "aider"]


def desc(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "mean": round(float(a.mean()), 4),
            "std": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
            "var": round(float(a.var(ddof=1)) if len(a) > 1 else 0.0, 4),
            "median": round(float(np.median(a)), 4)}


def main():
    out = {"_meta": {
        "note": "Distribution-only. No RF features, no gold labels in this corpus. "
                "Devstral-24B across SERA/LangGraph/Aider (50-issue subset, seed 42). "
                "Aider produced 0 fixes (cannot drive Devstral).",
        "source": "agent-harness/results/phase2_{sera,langgraph,aider}.jsonl",
    }}
    for h in HARNESSES:
        rows = [json.loads(l) for l in (AH / f"phase2_{h}.jsonl").read_text().splitlines() if l.strip()]
        n = len(rows)
        fix = sum(bool(r.get("fix_generated")) for r in rows)
        out[h] = {
            "n": n,
            "fix_rate": round(fix / n, 4) if n else None,
            "edit_rate": round(sum(r.get("edit_count", 0) > 0 for r in rows) / n, 4) if n else None,
            "turns_used": desc([r.get("turns_used") for r in rows]),
            "edit_count": desc([r.get("edit_count") for r in rows]),
            "first_edit_turn": desc([r.get("first_edit_turn") for r in rows if r.get("first_edit_turn") is not None]),
            "diff_size": desc([r.get("diff_size") for r in rows]),
            "repeat_count_loop_proxy": desc([r.get("repeat_count") for r in rows if "repeat_count" in r]),
            "elapsed_seconds": desc([r.get("elapsed_seconds") for r in rows]),
        }
    (RESULTS / "agent_harness_proxy_shift.json").write_text(json.dumps(out, indent=2))
    print("Wrote", RESULTS / "agent_harness_proxy_shift.json")


if __name__ == "__main__":
    main()
