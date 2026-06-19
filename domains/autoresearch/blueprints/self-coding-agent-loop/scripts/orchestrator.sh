#!/bin/bash
# orchestrator.sh — runs the full 5-round rolling-rounds experiment end-to-end.
# Keeps the p4de GPU busy by overlapping Round N+1 training with Round N Docker eval on m7i.
#
# Also gates each round's progression on prior results: if Round N's model_gold
# on round_N_control doesn't improve by >= min_improvement vs prior round,
# stop early (plateau — activate Loop 3/4 instead).
#
# Runs Phase 2 entry check after each round. If Phase 2 gate passes, we could in
# principle shift to reduced-label-density mode — but for now just log and stop.
#
# Usage (on p4de):
#   bash orchestrator.sh
# Env vars:
#   MAX_ROUNDS         (default 5)
#   MIN_IMPROVEMENT    (default 0.01 — 1pp on model_gold round-over-round)
#   M7I_HOST           (required — ubuntu@<m7i-ip>)
#   EXPERIMENT_BUCKET  (default s3://agent-aiops-artifacts/self-coding-agent-loop)
#   BASE_MODEL         (default Qwen/Qwen3.5-27B)

set -euo pipefail

MAX_ROUNDS="${MAX_ROUNDS:-5}"
MIN_IMPROVEMENT="${MIN_IMPROVEMENT:-0.01}"
M7I_HOST="${M7I_HOST:?set M7I_HOST=ubuntu@<m7i-ip>}"
EXPERIMENT_BUCKET="${EXPERIMENT_BUCKET:-s3://agent-aiops-artifacts/self-coding-agent-loop}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-27B}"

WORK_DIR=/mnt/nvme/self-coding-agent-loop
BLUEPRINT="$WORK_DIR/agent-aiops-on-aws/domains/autoresearch/blueprints/self-coding-agent-loop"
RUNS="$WORK_DIR/runs"
SPLITS="$BLUEPRINT/data/splits"
ORCH_LOG="$RUNS/orchestrator.log"

mkdir -p "$RUNS"
cd "$WORK_DIR"
source venv/bin/activate
export PYTHONUNBUFFERED=1 EXPERIMENT_BUCKET BASE_MODEL M7I_HOST WORK_DIR

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] orchestrator: $*" | tee -a "$ORCH_LOG"; }
snap_all() { aws s3 sync "$RUNS/" "$EXPERIMENT_BUCKET/runs/" --only-show-errors || true; }

# ---- Helper: run docker gold eval on m7i (asynchronous; returns immediately) ----
ship_eval_to_m7i_async() {
    local round=$1
    local kind=$2  # "control" or "drift_audit"
    local preds="$RUNS/round_${round}/${kind}_predictions.jsonl"
    local remote_dir="/tmp/round_${round}_${kind}"
    local remote_log="/tmp/round_${round}_${kind}/run.log"
    log "shipping ${kind} predictions for round $round to m7i"
    ssh $SSH_OPTS "$M7I_HOST" "mkdir -p $remote_dir"
    rsync -az -e "ssh $SSH_OPTS" "$preds" "$BLUEPRINT/scripts/docker_gold_eval.py" "$M7I_HOST:$remote_dir/"
    # Launch Docker eval in tmux on m7i — returns immediately
    # Uses m7i's existing /home/ubuntu/swebench-env virtualenv (swebench 4.1.0)
    ssh $SSH_OPTS "$M7I_HOST" "tmux new -d -s round_${round}_${kind} 'cd $remote_dir && \
        /home/ubuntu/swebench-env/bin/python docker_gold_eval.py --predictions ${kind}_predictions.jsonl \
            --output ${kind}_gold_results.jsonl \
            --run-id round_${round}_${kind} 2>&1 | tee $remote_log'"
    log "m7i started tmux session round_${round}_${kind}"
}

# ---- Helper: wait for m7i tmux session to finish, then pull results ----
wait_for_eval_and_pull() {
    local round=$1
    local kind=$2
    local remote_dir="/tmp/round_${round}_${kind}"
    log "waiting for m7i eval round ${round} ${kind}"
    while ssh $SSH_OPTS "$M7I_HOST" "tmux has-session -t round_${round}_${kind} 2>/dev/null"; do
        sleep 300  # poll every 5 min
        log "  still running m7i round ${round} ${kind}..."
    done
    log "m7i eval round ${round} ${kind} done — pulling results"
    rsync -az -e "ssh $SSH_OPTS" "$M7I_HOST:$remote_dir/${kind}_gold_results.jsonl" "$RUNS/round_${round}/" 2>/dev/null || true
    rsync -az -e "ssh $SSH_OPTS" "$M7I_HOST:$remote_dir/${kind}_gold_results.jsonl.summary.json" "$RUNS/round_${round}/" 2>/dev/null || true
}

# ---- Helper: read gold pass rate from summary ----
get_gold_rate() {
    local round=$1
    local kind=$2  # "control" or "drift_audit"
    local f="$RUNS/round_${round}/${kind}_gold_results.jsonl.summary.json"
    python3 -c "import json, sys; p=sys.argv[1]; import pathlib
f=pathlib.Path(p)
if not f.exists(): print('nan'); sys.exit(0)
print(json.loads(f.read_text()).get('gold_pass_rate', 'nan'))" "$f"
}

# ---- Helper: check Phase 2 entry gate ----
check_phase2_gate() {
    python3 "$BLUEPRINT/scripts/drift_trajectory_report.py" \
        --runs-dir "$RUNS" 2>&1 | tail -3
    # Read result
    python3 -c "import json; print(json.loads(open('$RUNS/phase2_gate.json').read()).get('entry_ready', False))" 2>/dev/null || echo False
}

log "=========================================="
log "orchestrator start: max_rounds=$MAX_ROUNDS, min_improvement=$MIN_IMPROVEMENT"
log "=========================================="

# Pre-flight: check prerequisites
test -f "$WORK_DIR/checkpoints/gen0/adapter_model.safetensors" \
    || { log "FATAL: Gen0 adapter missing"; exit 1; }
test -f "$SPLITS/round_1_train.jsonl" \
    || { log "FATAL: splits missing"; exit 1; }
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -i /home/ubuntu/.ssh/g7e-bench.pem"
ssh $SSH_OPTS "$M7I_HOST" "echo m7i-ok" \
    || { log "FATAL: m7i unreachable at $M7I_HOST"; exit 1; }
log "pre-flight ok"

# ========================================================================
# V1b_bootstrap happens once, before any round (retrains RF on 200 labeled)
# ========================================================================
if [[ ! -f "$RUNS/v1b_bootstrap/rf_recalibrated.pkl" ]]; then
    log "V1b_bootstrap: retraining RF on v1b_bootstrap_200 pool"
    python3 "$BLUEPRINT/scripts/v1b_bootstrap.py" \
        --pool "$SPLITS/v1b_bootstrap_200.jsonl" \
        --output-dir "$RUNS/v1b_bootstrap" \
        --n-labels 200 2>&1 | tee -a "$ORCH_LOG"
    snap_all
fi

# ========================================================================
# Main round loop
# ========================================================================
prior_gold_rate=""
for round in $(seq 1 "$MAX_ROUNDS"); do
    RDIR="$RUNS/round_${round}"
    mkdir -p "$RDIR"
    log "========== ROUND $round =========="

    # Step A: GPU work — SFT + generate on control + drift_audit
    if [[ ! -f "$RDIR/drift_audit_predictions.jsonl" ]]; then
        # Auto-detect: if adapter already exists, skip SFT (resume from saved Gen-N)
        SKIP_FLAGS=""
        if [[ -f "$RDIR/adapter/adapter_model.safetensors" ]]; then
            SKIP_FLAGS="--skip-train"
            log "round $round step A: adapter exists, skipping SFT, running generation only"
        else
            log "round $round step A: SFT + generate (GPU-bound, ~13-17hr)"
        fi
        python3 "$BLUEPRINT/scripts/round_runner.py" \
            --round "$round" \
            --base-model "$BASE_MODEL" \
            --gen0-adapter "$WORK_DIR/checkpoints/gen0" \
            --splits-dir "$SPLITS" \
            --output-root "$RUNS" \
            --nebius-parquet "$WORK_DIR/data/nebius/trajectories.parquet" \
            $SKIP_FLAGS 2>&1 | tee -a "$RDIR/round_runner.log" | tail -40
        snap_all
    else
        log "round $round step A: predictions already exist, skipping"
    fi

    # Step B: ship both eval jobs to m7i (async)
    ship_eval_to_m7i_async "$round" "control"
    ship_eval_to_m7i_async "$round" "drift_audit"

    # Step C: KEEP GPU BUSY — start Round N+1 SFT in parallel while m7i evals
    # Only do this if we haven't hit max_rounds yet
    next_round=$((round + 1))
    NEXT_RDIR="$RUNS/round_${next_round}"
    if [[ "$next_round" -le "$MAX_ROUNDS" ]] && [[ ! -f "$NEXT_RDIR/drift_audit_predictions.jsonl" ]]; then
        log "round $round step C: pre-running round $next_round SFT to keep GPU busy during eval"
        # Note: we can SFT on round_$next_round_train right now — it uses the Round $round adapter.
        # But we CAN'T generate on round_$next_round_control yet because we want Gen_round
        # (after gold eval) — eh, actually we can: round_N's adapter = after round N's SFT.
        # We want Gen_{round+1} = after round+1's SFT, which starts from round's adapter.
        # So: round N+1 SFT depends on round N's adapter (available now, no need to wait for eval).
        # It does NOT depend on round N's gold eval. Run it.
        python3 "$BLUEPRINT/scripts/round_runner.py" \
            --round "$next_round" \
            --base-model "$BASE_MODEL" \
            --gen0-adapter "$WORK_DIR/checkpoints/gen0" \
            --splits-dir "$SPLITS" \
            --output-root "$RUNS" \
            --nebius-parquet "$WORK_DIR/data/nebius/trajectories.parquet" 2>&1 | tee -a "$NEXT_RDIR/round_runner.log" | tail -20
        snap_all
    fi

    # Step D: wait for this round's m7i evals to finish (likely already done by now)
    wait_for_eval_and_pull "$round" "control"
    wait_for_eval_and_pull "$round" "drift_audit"
    snap_all

    # Step E: verifier recalibration
    log "round $round step E: verifier recalibrate"
    prior_rf="$RUNS/v1b_bootstrap/rf_recalibrated.pkl"
    if [[ "$round" -gt 1 ]]; then
        prior_rf="$RUNS/round_$((round-1))/rf.pkl"
    fi
    python3 "$BLUEPRINT/scripts/verifier_recalibrate.py" \
        --round "$round" \
        --control-predictions "$RDIR/control_predictions.jsonl" \
        --control-gold "$RDIR/control_gold_results.jsonl" \
        --drift-predictions "$RDIR/drift_audit_predictions.jsonl" \
        --drift-gold "$RDIR/drift_audit_gold_results.jsonl" \
        --prior-rf "$prior_rf" \
        --output-dir "$RDIR" 2>&1 | tee -a "$ORCH_LOG" | tail -20

    # Step F: per-round summary + decision gate
    ctrl_gold=$(get_gold_rate "$round" "control")
    drift_gold=$(get_gold_rate "$round" "drift_audit")
    log "round $round: control_gold=$ctrl_gold, drift_gold=$drift_gold"
    snap_all

    # Plateau check: if round >= 2 and ctrl_gold didn't improve by min_improvement, stop
    if [[ -n "$prior_gold_rate" ]] && [[ "$prior_gold_rate" != "nan" ]]; then
        delta=$(python3 -c "print($ctrl_gold - $prior_gold_rate)" 2>/dev/null || echo 0)
        below_thresh=$(python3 -c "print(1 if $delta < $MIN_IMPROVEMENT else 0)" 2>/dev/null || echo 0)
        if [[ "$below_thresh" == "1" ]]; then
            log "round $round: PLATEAU detected (delta=$delta < $MIN_IMPROVEMENT). Stopping early."
            log "    Recommend: activate Loop 3 (harness) or Loop 4 (model selection) before more rounds"
            break
        fi
    fi
    prior_gold_rate="$ctrl_gold"

    # Phase 2 gate check
    if [[ "$round" -ge 3 ]]; then
        ready=$(check_phase2_gate)
        log "phase 2 gate after round $round: $ready"
        if [[ "$ready" == "True" ]]; then
            log "PHASE 2 GATE PASSED — verifier is RL-ready and stable for 3 rounds"
            log "    Not auto-transitioning. Stop after current round; review drift trajectory."
            break
        fi
    fi
done

log "=========================================="
log "orchestrator: generating final drift trajectory report"
python3 "$BLUEPRINT/scripts/drift_trajectory_report.py" --runs-dir "$RUNS" 2>&1 | tail -10
snap_all
log "orchestrator DONE"
log "=========================================="
