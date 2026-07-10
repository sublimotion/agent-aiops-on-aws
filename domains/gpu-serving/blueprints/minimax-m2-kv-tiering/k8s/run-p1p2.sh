#!/usr/bin/env bash
# =============================================================================
# MiniMax-M2 B200 — P1 (KV-tiering) + P2 (DP/TP sweep) RE-RUN, 2026-06-28.
# P3 (v0.23.0) already succeeded in the prior session — not repeated.
# Includes a BOOT SMOKE-TEST gate: confirm a serving pod from the PATCHED
# gen-serving-manifest.sh actually boots (the tokenizer-dep fix) BEFORE committing
# to the full sweeps. Detached, self-terminating, scale-to-0 on exit.
# =============================================================================
set -uo pipefail
CTX="qwen3-next-bench-eks-cluster"
NODEGROUP="ai-infra-use2-b200-spot-maz"; REGION="us-east-2"
K=(kubectl --context "$CTX")
KVT_BP="/Users/phi/Documents/workbench/agent-aiops-on-aws/domains/gpu-serving/blueprints/minimax-m2-kv-tiering"
M2_BP="/Users/phi/Documents/workbench/agent-aiops-on-aws/domains/gpu-serving/blueprints/minimax-m2"
R="${KVT_BP}/results"; LOG="${R}/p1p2.log"; PHASE_STATUS="${R}/PHASE-STATUS"
PREFLIGHT_OK=0; mkdir -p "$R"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
pstat(){ echo "$*" > "$PHASE_STATUS"; }
scale0(){ [ "$PREFLIGHT_OK" = "1" ] || { log "scale0 SKIPPED (preflight failed)"; return; }
  for i in 1 2 3 4 5; do aws eks update-nodegroup-config --cluster-name "$CTX" --nodegroup-name "$NODEGROUP" \
    --region "$REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0 >/dev/null 2>&1 \
    && { log "scale-to-zero accepted ($i)"; return; }; sleep 10; done
  log "SCALE-TO-ZERO FAILED 5x — MANUAL ACTION"; pstat "MANUAL_ACTION_REQUIRED"; }
trap 'rc=$?; log "EXIT rc=$rc"; scale0' EXIT
trap 'log "SIGINT"; exit 130' INT
trap 'log "SIGTERM"; exit 143' TERM
( sleep $((420*60)); log "WALL CAP hit"; kill -TERM -$$ 2>/dev/null ) & WATCHDOG=$!

log "=== P1+P2 RE-RUN START ==="
ACTUAL=$(kubectl config view -o jsonpath="{.contexts[?(@.name=='${CTX}')].context.cluster}" 2>/dev/null)
echo "$ACTUAL" | grep -q "$CTX" || { log "PREFLIGHT FAIL: context->cluster mismatch. ABORT."; exit 1; }

# wait for node Ready, label by selector
log "waiting for node Ready..."
for i in $(seq 1 40); do
  "${K[@]}" get nodes -l ai-infra/role=b200-spot 2>/dev/null | grep -q Ready && break; sleep 30; done
"${K[@]}" label nodes -l ai-infra/role=b200-spot blueprint=minimax-m2 nvidia.com/gpu.present=true --overwrite >/dev/null 2>&1
"${K[@]}" get nodes -l blueprint=minimax-m2 2>/dev/null | grep -q Ready \
  || { log "PREFLIGHT FAIL: no Ready blueprint=minimax-m2 node. ABORT."; exit 1; }
PREFLIGHT_OK=1; log "PREFLIGHT OK — scaledown ARMED."

# stage weights (fresh node)
"${K[@]}" delete job stage-minimax-m2 --ignore-not-found >/dev/null 2>&1; sleep 3
"${K[@]}" apply -f "${KVT_BP}/k8s/stage-model.yaml" >>"$LOG" 2>&1
log "staging weights (~25min)..."
for i in $(seq 1 50); do
  st=$("${K[@]}" get job stage-minimax-m2 -o jsonpath='{.status.conditions[*].type}' 2>/dev/null)
  case "$st" in *Complete*) log "staging done"; break;; *Failed*) log "FATAL staging failed"; exit 1;; esac; sleep 60; done
"${K[@]}" apply -f "${KVT_BP}/k8s/observability.yaml" >>"$LOG" 2>&1; log "observability applied"

# ── BOOT SMOKE GATE: prove the PATCHED gen-manifest boots (the tokenizer-dep fix) before full sweeps ──
log "=== BOOT SMOKE: tp4 gpu-only via patched gen-serving-manifest.sh ==="
bash "${M2_BP}/k8s/gen-serving-manifest.sh" tp4 gpu-only > /tmp/smoke-tp4.yaml 2>>"$LOG"
"${K[@]}" apply -f /tmp/smoke-tp4.yaml >>"$LOG" 2>&1
SMOKE_OK=0; SMOKE_POD=$("${K[@]}" get -f /tmp/smoke-tp4.yaml -o jsonpath='{.metadata.name}' 2>/dev/null)
log "smoke pod=${SMOKE_POD}; waiting for boot (pip install + ~35min cold start)..."
for i in $(seq 1 80); do
  ph=$("${K[@]}" get pod "$SMOKE_POD" -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$ph" != "Running" ] && [ "$ph" != "Pending" ]; then
    log "BOOT SMOKE FAILED phase=$ph — fix did NOT work. Capturing logs, aborting (trap scales down)."
    "${K[@]}" logs "$SMOKE_POD" --tail=30 2>&1 | tail -30 | tee -a "$LOG"
    pstat "ABORT boot-smoke-failed"; exit 1
  fi
  code=$("${K[@]}" exec "$SMOKE_POD" -- curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)
  log "smoke phase=$ph health=$code"
  [ "$code" = "200" ] && { SMOKE_OK=1; log "BOOT SMOKE PASS — patched gen-manifest boots. Proceeding to sweeps."; break; }
  sleep 40
done
[ "$SMOKE_OK" = "1" ] || { log "smoke never healthy; abort"; pstat "ABORT smoke-timeout"; exit 1; }
# tear the smoke pod down to free GPUs for the sweep (which relaunches per shape/arm)
"${K[@]}" delete pod "$SMOKE_POD" --ignore-not-found --wait=true >>"$LOG" 2>&1; sleep 5

# ── PHASE 1: KV-tiering distinct-prefix sweep ──
pstat "P1 RUNNING kv-tiering sweep"; log "=== PHASE 1: KV-tiering distinct-prefix sweep ==="
SKIP_SCALEDOWN=1 EXPECT_CONTEXT="$CTX" bash "${KVT_BP}/k8s/run-tiering-sweep.sh" >>"$LOG" 2>&1 \
  && log "PHASE 1 complete" || log "PHASE 1 ended rc=$? (zero-runs guard may have fired — check P1 STATUS)"

# ── PHASE 2: DP/TP parallelism + nvme arm ──
pstat "P2 RUNNING DP/TP sweep"; log "=== PHASE 2: DP/TP parallelism sweep ==="
SKIP_SCALEDOWN=1 EXPECT_CONTEXT="$CTX" SHAPES="tp2dp2 tp4dp2 tp2dp4" KV_ARMS="gpu-only nvme-tiering" \
  bash "${M2_BP}/k8s/run-pareto-sweep.sh" >>"$LOG" 2>&1 \
  && log "PHASE 2 complete" || log "PHASE 2 ended rc=$? (check P2 STATUS)"

pstat "P1+P2 DONE — scaling to 0"; log "=== P1+P2 DONE ==="
kill "$WATCHDOG" 2>/dev/null
