#!/usr/bin/env python3
"""drift_trajectory_report.py — aggregate round summaries into the full drift trajectory.

Produces:
  drift_trajectory.md   — human-readable report with the trajectory table
  drift_trajectory.json — canonical machine-readable (same as verifier_recalibrate appends to)
  phase2_gate.json      — decision on whether Phase 2 entry criteria are met

Phase 2 entry criteria (all must hold):
  - Verifier-gold agreement on drift_audit >= 0.85 (SFT-ready precision) for 3 CONSECUTIVE rounds
  - Model gold pass rate on drift_audit non-decreasing across last 3 rounds
  - Verifier ECE on drift_audit <= 0.1 in the most recent round (RL-ready)
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True, help="Parent of round_N/ subdirs")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    runs = Path(args.runs_dir)
    out = Path(args.output_dir) if args.output_dir else runs
    out.mkdir(parents=True, exist_ok=True)

    traj_path = runs / "drift_trajectory.json"
    if not traj_path.exists():
        print(f"[drift_trajectory_report] no trajectory at {traj_path}; has any round completed?")
        return

    with open(traj_path) as f:
        trajectory = json.load(f)
    trajectory.sort(key=lambda t: t["round"])

    # Phase 2 gate
    gate = {"n_rounds": len(trajectory), "entry_ready": False, "reasons": []}
    if len(trajectory) < 3:
        gate["reasons"].append(f"only {len(trajectory)} round(s) done; need 3 consecutive")
    else:
        last3 = trajectory[-3:]
        agreements = [t["verifier_agreement_on_drift"] for t in last3]
        gold_rates = [t["model_gold_pass_on_drift"] for t in last3]
        ece = trajectory[-1]["verifier_ece_on_drift"]
        if not all(a >= 0.85 for a in agreements):
            gate["reasons"].append(f"agreement in last 3 rounds {[round(a,3) for a in agreements]} not all >= 0.85")
        if not (gold_rates[-1] >= gold_rates[-2] >= gold_rates[-3]):
            gate["reasons"].append(f"model gold pass not monotone in last 3: {[round(g,3) for g in gold_rates]}")
        if ece > 0.1:
            gate["reasons"].append(f"latest ECE {ece:.3f} > 0.1 (RL-ready threshold)")
        gate["entry_ready"] = not gate["reasons"]
        gate["last3_agreement"] = agreements
        gate["last3_model_gold"] = gold_rates
        gate["latest_ece"] = ece

    with open(out / "phase2_gate.json", "w") as f:
        json.dump(gate, f, indent=2)

    # Markdown report
    lines = [
        "# Drift Trajectory Report",
        "",
        f"**Rounds completed**: {len(trajectory)}",
        f"**Phase 2 entry gate**: {'PASS' if gate['entry_ready'] else 'NOT READY'}",
        "",
        "## Per-round metrics on drift_audit_300 (identical inputs, different models)",
        "",
        "| Round | Gen | Model gold | Verifier ECE | Verifier agreement | Verifier P@R>=0.30 |",
        "|------:|-----|-----------:|-------------:|-------------------:|-------------------:|",
    ]
    for t in trajectory:
        m = t["drift_audit_metrics"]
        lines.append(
            f"| {t['round']} | {t['gen_id']} | "
            f"{m['gold_pass_rate']:.3f} | {m['ece']:.3f} | {m['agreement']:.3f} | {m['p_at_r30']:.3f} |"
        )
    lines += ["", "## Phase 2 entry decision", "", "```json",
              json.dumps(gate, indent=2), "```", ""]
    (out / "drift_trajectory.md").write_text("\n".join(lines))
    print(f"[drift_trajectory_report] wrote {out / 'drift_trajectory.md'}")
    print(f"[drift_trajectory_report] Phase 2 gate: {'PASS' if gate['entry_ready'] else 'NOT READY'}")


if __name__ == "__main__":
    main()
