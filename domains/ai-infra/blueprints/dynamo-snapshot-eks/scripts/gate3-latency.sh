#!/usr/bin/env bash
# Gate 3 — restore latency.
# Requires: results/e1/e1-result.json with restore_seconds_to_all_ready and
# results/e1/b1-baseline/summary.json with seconds_to_first_token.p50.
#
# Pass criterion: snapshot variant p50 pod-create-to-first-token <= 30s
# AND <= 0.5x of the baseline cold-start p50.
set -euo pipefail
E1_RESULT=${1:-results/e1/e1-result.json}
B1_SUMMARY=${2:-results/e1/b1-baseline/summary.json}

python3 - <<EOF
import json
e1=json.load(open("$E1_RESULT"))
b1=json.load(open("$B1_SUMMARY"))
restore_p50 = e1.get("restore_seconds_to_all_ready") or 0
b1_p50 = b1["seconds_to_first_token"]["p50"]
abs_pass = restore_p50 <= 30.0
rel_pass = restore_p50 <= 0.5 * b1_p50
overall = abs_pass and rel_pass
out={
  "restore_seconds_p50": restore_p50,
  "baseline_seconds_p50": b1_p50,
  "absolute_cap_30s_pass": abs_pass,
  "half_baseline_pass": rel_pass,
  "overall_pass": overall,
}
print(json.dumps(out, indent=2))
import sys; sys.exit(0 if overall else 1)
EOF
