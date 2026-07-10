#!/usr/bin/env bash
# =============================================================================
# MiniMax-M2 B200 — 3-PHASE unattended gap-closure session on ONE node.
# Detached, self-terminating, scale-to-0 on exit. Operator does NOT babysit.
#
#   P1: KV-tiering distinct-prefix sweep   (this blueprint — the new question)
#   P2: DP/TP parallelism sweep            (minimax-m2 gap — TP2+DP2/TP4+DP2/TP2+DP4 + nvme arm)
#   P3: v0.23.0 patched-image correctness  (the open engine A/B; pip-installs the tokenizer deps)
#
# Phases are SELF-CONTAINED + value-ordered: a spot reclaim mid-P3 still banks P1+P2.
# Weights are staged ONCE (shared NVMe). The EXIT trap scales the nodegroup to 0 on
# completion / 420-min cap / Ctrl-C / any fatal error.
# =============================================================================
set -uo pipefail   # NOT -e: a failed phase must continue + still scale down

CTX="qwen3-next-bench-eks-cluster"
NODEGROUP="ai-infra-use2-b200-spot"
REGION="us-east-2"
K=(kubectl --context "$CTX")
KVT_BP="/Users/phi/Documents/workbench/agent-aiops-on-aws/domains/gpu-serving/blueprints/minimax-m2-kv-tiering"
M2_BP="/Users/phi/Documents/workbench/agent-aiops-on-aws/domains/gpu-serving/blueprints/minimax-m2"
R="${KVT_BP}/results"
LOG="${R}/all-phases.log"
PHASE_STATUS="${R}/PHASE-STATUS"
PREFLIGHT_OK=0
mkdir -p "$R"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
pstat(){ echo "$*" > "$PHASE_STATUS"; }

scale0(){
  [ "$PREFLIGHT_OK" = "1" ] || { log "scale0 SKIPPED (preflight never passed — not our node to touch)"; return; }
  for i in 1 2 3 4 5; do
    aws eks update-nodegroup-config --cluster-name "$CTX" --nodegroup-name "$NODEGROUP" \
      --region "$REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0 >/dev/null 2>&1 \
      && { log "scale-to-zero accepted (attempt $i)"; return; }
    sleep 10
  done
  log "SCALE-TO-ZERO FAILED 5x — MANUAL ACTION REQUIRED"; pstat "MANUAL_ACTION_REQUIRED scale-to-zero failed"
}
trap 'rc=$?; log "EXIT rc=$rc"; scale0' EXIT
trap 'log "SIGINT"; exit 130' INT
trap 'log "SIGTERM (cap/operator)"; exit 143' TERM

# Internal wall-clock watchdog (defense in depth if `timeout` wrapper absent)
( sleep $((420*60)); log "WALL-CLOCK CAP 420m hit — killing pgid"; kill -TERM -$$ 2>/dev/null ) &
WATCHDOG=$!

# ── PREFLIGHT: pinned context maps to the cluster + node Ready (the wrong-cluster guard) ──
log "=== 3-PHASE SESSION START ==="
ACTUAL=$(kubectl config view -o jsonpath="{.contexts[?(@.name=='${CTX}')].context.cluster}" 2>/dev/null)
echo "$ACTUAL" | grep -q "$CTX" || { log "PREFLIGHT FAIL: context '$CTX' -> cluster '$ACTUAL'. ABORT (no scaledown)."; exit 1; }
"${K[@]}" get nodes -l blueprint=minimax-m2 2>/dev/null | grep -q Ready \
  || { log "PREFLIGHT FAIL: no Ready node labeled blueprint=minimax-m2. ABORT (no scaledown)."; exit 1; }
PREFLIGHT_OK=1
log "PREFLIGHT OK — scaledown ARMED."

# ── Wait for staged weights (job started separately; gate on Complete) ──
log "waiting for weight staging..."
for i in $(seq 1 50); do
  st=$("${K[@]}" get job stage-minimax-m2 -o jsonpath='{.status.conditions[*].type}' 2>/dev/null)
  case "$st" in *Complete*) log "staging complete"; break;; *Failed*) log "FATAL staging failed"; exit 1;; esac
  sleep 60
done

# ── Observability (cache-hit harvest depends on it — the load-bearing pod) ──
"${K[@]}" apply -f "${KVT_BP}/k8s/observability.yaml" >>"$LOG" 2>&1
log "observability applied"

# =============================================================================
# PHASE 1 — KV-tiering distinct-prefix sweep (the NEW question)
# =============================================================================
pstat "P1 RUNNING kv-tiering distinct-prefix sweep"
log "=== PHASE 1: KV-tiering distinct-prefix sweep ==="
# run-tiering-sweep.sh has its OWN trap/preflight; we disarm its scaledown by NOT letting it
# scale down between phases — it scales down only on ITS exit, but we want the node to persist.
# Solution: run it with SKIP_SCALEDOWN so the node stays up for P2/P3 (its trap honors the flag).
SKIP_SCALEDOWN=1 EXPECT_CONTEXT="$CTX" bash "${KVT_BP}/k8s/run-tiering-sweep.sh" >>"$LOG" 2>&1 \
  && log "PHASE 1 complete" || log "PHASE 1 ended (rc=$? — continuing to P2)"

# =============================================================================
# PHASE 2 — DP/TP parallelism sweep (minimax-m2 gap)
# =============================================================================
pstat "P2 RUNNING DP/TP parallelism sweep"
log "=== PHASE 2: DP/TP parallelism + nvme arm (minimax-m2 gap) ==="
# reuse the ORIGINAL minimax-m2 sweep, restricted to the UNRUN shapes + nvme arm.
SKIP_SCALEDOWN=1 EXPECT_CONTEXT="$CTX" \
  SHAPES="tp2dp2 tp4dp2 tp2dp4" KV_ARMS="gpu-only nvme-tiering" \
  bash "${M2_BP}/k8s/run-pareto-sweep.sh" >>"$LOG" 2>&1 \
  && log "PHASE 2 complete" || log "PHASE 2 ended (rc=$? — continuing to P3)"

# =============================================================================
# PHASE 3 — v0.23.0 patched-image correctness A/B
# =============================================================================
pstat "P3 RUNNING v0.23.0 patched correctness A/B"
log "=== PHASE 3: v0.23.0 patched (sentencepiece+tiktoken) correctness gate ==="
"${K[@]}" delete pod vllm-minimax-m2-v23 --ignore-not-found --wait=true >>"$LOG" 2>&1
# free GPUs: tear down any serving pod from P2
"${K[@]}" delete pods -l app=vllm-minimax-m2 --ignore-not-found --wait=true >>"$LOG" 2>&1; sleep 5
"${K[@]}" apply -f "${M2_BP}/k8s/vllm-v23.yaml" >>"$LOG" 2>&1
log "v0.23.0 patched pod applied; waiting for boot (pip install + ~35min cold start)..."
V23_OK=0
for i in $(seq 1 90); do
  ph=$("${K[@]}" get pod vllm-minimax-m2-v23 -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$ph" != "Running" ] && [ "$ph" != "Pending" ]; then
    log "v0.23.0 boot FAILED phase=$ph"; "${K[@]}" logs vllm-minimax-m2-v23 --tail=30 2>&1 | tail -30 | tee -a "$LOG"
    echo "VERDICT: v0.23.0 patched STILL fails to boot (phase=$ph) — see log" >> "${R}/v23-RESULT.txt"; break
  fi
  code=$("${K[@]}" exec vllm-minimax-m2-v23 -- curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)
  log "v23 boot phase=$ph health=$code"
  [ "$code" = "200" ] && { V23_OK=1; log "v0.23.0 patched HEALTHY"; break; }
  sleep 40
done
if [ "$V23_OK" = "1" ]; then
  # correctness smoke: tool call + garbage screen
  TC=$("${K[@]}" exec vllm-minimax-m2-v23 -- curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"MiniMax-M2","messages":[{"role":"user","content":"Weather in Paris? Call the tool, do not overthink."}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":2000}' 2>/dev/null | python3 -c "import sys,json;c=json.load(sys.stdin)['choices'][0];print('PASS' if c.get('finish_reason')=='tool_calls' and c['message'].get('tool_calls') else 'FAIL')" 2>/dev/null)
  log "v0.23.0 patched correctness smoke: tool_call=$TC"
  echo "VERDICT: v0.23.0 patched (sentencepiece+tiktoken) BOOTS + correctness tool_call=$TC @ $(date -u +%FT%TZ)" >> "${R}/v23-RESULT.txt"
  # quick knee bench at the canonical points (reuse the kv-tiering bench-runner)
  for SC in shared-prefix 100k-90pct-reuse; do for C in 64 128; do
    "${K[@]}" exec bench-runner -- python3 /scripts/bench.py --port 8000 --model MiniMax-M2 \
      --scenario $SC --conc $C --prefix-tokens 90000 --suffix-tokens 10000 --reuse-target 0.90 \
      --out-tokens 512 --out /tmp/v23-$SC-c$C.json >>"$LOG" 2>&1 || log "  v23 bench $SC c=$C failed"
    "${K[@]}" cp bench-runner:/tmp/v23-$SC-c$C.json "${R}/v23-$SC-c$C.json" >>"$LOG" 2>&1
    line=$(python3 -c "import json;d=json.load(open('${R}/v23-$SC-c$C.json'));print(f'  v23 $SC c=$C tok/s={d.get(\"agg_decode_tok_s\")} ttft_p95={d.get(\"ttft_p95_s\")} cache_hit={d.get(\"client_cache_hit_fraction\")} qpass={d.get(\"quality_pass\")}')" 2>/dev/null || echo "  v23 $SC c=$C parse-fail")
    log "$line"; echo "$line" >> "${R}/v23-RESULT.txt"
  done; done
  log "PHASE 3 complete (v0.23.0 vs 0.19.1rc1 A/B data in v23-RESULT.txt + v23-*.json)"
else
  log "PHASE 3: v0.23.0 did not become healthy — verdict recorded"
fi

pstat "ALL PHASES DONE — scaling to 0"
log "=== ALL PHASES DONE ==="
kill "$WATCHDOG" 2>/dev/null
# EXIT trap fires scale0
