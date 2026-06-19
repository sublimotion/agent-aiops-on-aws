#!/bin/bash
# run_round.sh — single round of the continuous-calibration RLVR loop.
# Runs on p4de for GPU work; shells out to m7i SSH for Docker gold eval.
#
# Usage:
#   run_round.sh N            # full round
#   run_round.sh N --skip-train   # resume from after-train checkpoint
#   run_round.sh N --skip-generate  # resume after generation
#
# Assumes:
#   - Running on p4de instance
#   - splits/ directory populated by build_splits.py
#   - Gen0 adapter at $WORK_DIR/checkpoints/gen0/
#   - m7i.16xlarge available at $M7I_HOST for Docker eval
#     (or set --self-host to run Docker on same node; slower but works)
set -euo pipefail

ROUND="${1:?Usage: $0 <round-num> [--skip-train|--skip-generate]}"
shift || true
EXTRA_ARGS="$*"

WORK_DIR="${WORK_DIR:-/mnt/nvme/self-coding-agent-loop}"
BLUEPRINT="$WORK_DIR/agent-aiops-on-aws/domains/autoresearch/blueprints/self-coding-agent-loop"
SPLITS="$BLUEPRINT/data/splits"
RUNS="$WORK_DIR/runs"
BUCKET="${EXPERIMENT_BUCKET:-s3://agent-aiops-artifacts/self-coding-agent-loop}"
M7I_HOST="${M7I_HOST:-}"   # e.g. ubuntu@10.0.32.42 — set if Docker eval is remote
SELF_HOST_DOCKER="${SELF_HOST_DOCKER:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-27B}"

RDIR="$RUNS/round_${ROUND}"
mkdir -p "$RDIR"
cd "$WORK_DIR"
source venv/bin/activate

log() { echo "[$(date -u +%H:%M:%S)] round_$ROUND: $*" | tee -a "$RDIR/round.log"; }
snap() { aws s3 sync "$RUNS/" "$BUCKET/runs/" --only-show-errors 2>&1 | tail -3 || true; }

log "======= Round $ROUND begin ======="

# Step 1-3: GPU work — SFT + generate patches on control + drift_audit
log "Step 1-3: SFT + generate control + drift_audit predictions"
python3 "$BLUEPRINT/scripts/round_runner.py" \
    --round "$ROUND" \
    --base-model "$BASE_MODEL" \
    --gen0-adapter "$WORK_DIR/checkpoints/gen0" \
    --splits-dir "$SPLITS" \
    --output-root "$RUNS" \
    $EXTRA_ARGS 2>&1 | tee "$RDIR/round_runner.log"
snap

# Step 4: Docker gold eval
ctrl_preds="$RDIR/control_predictions.jsonl"
drift_preds="$RDIR/drift_audit_predictions.jsonl"
ctrl_gold="$RDIR/control_gold_results.jsonl"
drift_gold="$RDIR/drift_audit_gold_results.jsonl"

if [[ -n "$M7I_HOST" ]]; then
    log "Step 4: Docker gold eval on $M7I_HOST"
    # Ship predictions + script to m7i, run eval, pull results back
    rsync -az "$ctrl_preds" "$drift_preds" \
        "$BLUEPRINT/scripts/docker_gold_eval.py" \
        "$M7I_HOST:/tmp/round_${ROUND}/"
    ssh "$M7I_HOST" "cd /tmp/round_${ROUND} && \
        python3 docker_gold_eval.py --predictions control_predictions.jsonl \
            --output control_gold_results.jsonl --run-id round_${ROUND}_ctrl && \
        python3 docker_gold_eval.py --predictions drift_audit_predictions.jsonl \
            --output drift_audit_gold_results.jsonl --run-id round_${ROUND}_drift"
    rsync -az "$M7I_HOST:/tmp/round_${ROUND}/*_gold_results.jsonl" "$RDIR/"
    rsync -az "$M7I_HOST:/tmp/round_${ROUND}/*.summary.json" "$RDIR/"
elif [[ "$SELF_HOST_DOCKER" == "1" ]]; then
    log "Step 4: Docker gold eval on this node (slower; m7i recommended)"
    python3 "$BLUEPRINT/scripts/docker_gold_eval.py" \
        --predictions "$ctrl_preds" --output "$ctrl_gold" \
        --run-id "round_${ROUND}_ctrl" 2>&1 | tee "$RDIR/docker_ctrl.log"
    python3 "$BLUEPRINT/scripts/docker_gold_eval.py" \
        --predictions "$drift_preds" --output "$drift_gold" \
        --run-id "round_${ROUND}_drift" 2>&1 | tee "$RDIR/docker_drift.log"
else
    log "Step 4 SKIPPED: no M7I_HOST set, SELF_HOST_DOCKER=0"
    log "       run gold eval manually then re-run this script with --skip-train --skip-generate"
    log "       OR set M7I_HOST and re-run"
    snap
    exit 0
fi
snap

# Step 5: verifier recalibration
log "Step 5: verifier recalibration on cumulative labels"
prior_rf="$RUNS/v1b_bootstrap/rf_recalibrated.pkl"
if [[ "$ROUND" -gt 1 ]]; then
    prior_rf="$RUNS/round_$((ROUND-1))/rf.pkl"
fi
python3 "$BLUEPRINT/scripts/verifier_recalibrate.py" \
    --round "$ROUND" \
    --control-predictions "$ctrl_preds" \
    --control-gold "$ctrl_gold" \
    --drift-predictions "$drift_preds" \
    --drift-gold "$drift_gold" \
    --prior-rf "$prior_rf" \
    --output-dir "$RDIR" 2>&1 | tee "$RDIR/verifier_recalibrate.log"
snap

# Step 6: round summary
python3 - <<PY
import json
from pathlib import Path
rdir = Path("$RDIR")
summary = {"round": $ROUND}
for name in ["round_runner_summary.json", "verifier_metrics.json"]:
    p = rdir / name
    if p.exists():
        summary[name.replace(".json","")] = json.loads(p.read_text())
# Pull gold rates
for kind in ["control", "drift_audit"]:
    sp = rdir / f"{kind}_gold_results.jsonl.summary.json"
    if sp.exists():
        summary[f"{kind}_gold_summary"] = json.loads(sp.read_text())
with open(rdir / "round_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps({"round": $ROUND,
    "control_gold_pass_rate": summary.get("control_gold_summary",{}).get("gold_pass_rate"),
    "drift_audit_gold_pass_rate": summary.get("drift_audit_gold_summary",{}).get("gold_pass_rate"),
    "verifier_ece_on_drift": summary.get("verifier_metrics",{}).get("new_verifier_on_drift_audit",{}).get("ece"),
    "verifier_agreement_on_drift": summary.get("verifier_metrics",{}).get("new_verifier_on_drift_audit",{}).get("agreement"),
}, indent=2))
PY

snap
log "======= Round $ROUND complete ======="
