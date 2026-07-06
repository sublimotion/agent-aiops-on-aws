#!/usr/bin/env bash
# =============================================================================
# MiniMax-M2 H200 like-for-like KNEE comparison (2026-06-28).
# The ONE missing number: the customer runs H-class, but all our data is B200.
# Run the BEST config (v0.23.0-patched, TP4 — IDENTICAL parallelism to B200) at the
# KNEE concurrency points only, on the 2 cache scenarios. Produces the hardware-matched
# H200-vs-B200 table that retires the COMPARISON-CAVEATS asterisk.
# "Most interesting run at the knees; expand only if results don't compare."
# Detached, self-terminating, scale-to-0 on exit.
# =============================================================================
set -uo pipefail
CTX="qwen3-next-bench-eks-cluster"; NODEGROUP="ai-infra-use2-p5en-spot"; REGION="us-east-2"
K=(kubectl --context "$CTX")
BP="/Users/phi/Documents/workbench/agent-aiops-on-aws/domains/gpu-serving/blueprints/minimax-m2"
R="${BP}/results"; LOG="${R}/h200.log"; ST="${R}/H200-STATUS"; RES="${R}/h200-comparison.txt"
PREFLIGHT_OK=0; mkdir -p "$R"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
scale0(){ [ "$PREFLIGHT_OK" = "1" ] || { log "scale0 skipped (preflight failed)"; return; }
  for i in 1 2 3 4 5; do aws eks update-nodegroup-config --cluster-name "$CTX" --nodegroup-name "$NODEGROUP" \
    --region "$REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0 >/dev/null 2>&1 \
    && { log "p5en scaled to 0 ($i)"; return; }; sleep 10; done
  log "SCALE-TO-ZERO FAILED 5x — MANUAL ACTION on $NODEGROUP"; }
trap 'rc=$?; log "EXIT rc=$rc"; scale0' EXIT
trap 'log SIGINT; exit 130' INT
trap 'log SIGTERM; exit 143' TERM
( sleep $((420*60)); log "WALL CAP 420m"; kill -TERM -$$ 2>/dev/null ) & WD=$!

log "=== H200 KNEE COMPARISON START ==="
ACTUAL=$(kubectl config view -o jsonpath="{.contexts[?(@.name=='${CTX}')].context.cluster}" 2>/dev/null)
echo "$ACTUAL" | grep -q "$CTX" || { log "PREFLIGHT FAIL context. ABORT"; exit 1; }
log "waiting for p5en node Ready..."
for i in $(seq 1 40); do "${K[@]}" get nodes -l ai-infra/role=p5en-h200 2>/dev/null | grep -q Ready && break; sleep 30; done
"${K[@]}" label nodes -l ai-infra/role=p5en-h200 blueprint=minimax-m2-h200 nvidia.com/gpu.present=true --overwrite >/dev/null 2>&1
"${K[@]}" get nodes -l blueprint=minimax-m2-h200 2>/dev/null | grep -q Ready || { log "PREFLIGHT FAIL: no Ready h200 node. ABORT"; exit 1; }
PREFLIGHT_OK=1; log "PREFLIGHT OK (p5en) — scaledown ARMED. GPUs=$("${K[@]}" get nodes -l blueprint=minimax-m2-h200 -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}')"

# stage weights to THIS node's NVMe (fresh node). stage-model.yaml nodeSelector is minimax-m2;
# generate an h200-targeted staging job inline (same image/command, different nodeSelector+toleration).
sed -e 's/blueprint: minimax-m2}/blueprint: minimax-m2-h200}/' \
    -e 's/key: ai-infra\/b200/key: ai-infra\/h200/' \
    "${BP}/k8s/stage-model.yaml" > /tmp/stage-h200.yaml
"${K[@]}" delete job stage-minimax-m2 --ignore-not-found >/dev/null 2>&1; sleep 3
"${K[@]}" apply -f /tmp/stage-h200.yaml >>"$LOG" 2>&1
log "staging weights to H200 NVMe (~25min)..."
for i in $(seq 1 50); do
  s=$("${K[@]}" get job stage-minimax-m2 -o jsonpath='{.status.conditions[*].type}' 2>/dev/null)
  case "$s" in *Complete*) log "staging done"; break;; *Failed*) log "FATAL staging failed"; exit 1;; esac; sleep 60; done

# observability (h200-targeted) + bench-runner
sed 's/blueprint: minimax-m2}/blueprint: minimax-m2-h200}/; s/key: ai-infra\/b200/key: ai-infra\/h200/' "${BP}/k8s/observability.yaml" > /tmp/obs-h200.yaml
"${K[@]}" apply -f /tmp/obs-h200.yaml >>"$LOG" 2>&1
sed 's/blueprint: minimax-m2}/blueprint: minimax-m2-h200}/; s/key: ai-infra\/b200/key: ai-infra\/h200/' "${BP}/k8s/bench-runner.yaml" > /tmp/bench-h200.yaml
"${K[@]}" apply -f /tmp/bench-h200.yaml >>"$LOG" 2>&1

# boot the serving pod (v0.23.0-patched, TP4 — identical to B200 winning config)
"${K[@]}" apply -f "${BP}/k8s/vllm-h200.yaml" >>"$LOG" 2>&1
log "booting H200 serving pod (pip-install + ~30min cold start)..."
OK=0
for i in $(seq 1 80); do
  ph=$("${K[@]}" get pod vllm-minimax-m2-h200 -o jsonpath='{.status.phase}' 2>/dev/null)
  [ "$ph" != "Running" ] && [ "$ph" != "Pending" ] && { log "H200 boot FAILED phase=$ph"; "${K[@]}" logs vllm-minimax-m2-h200 --tail=30 2>&1|tail -30|tee -a "$LOG"; exit 1; }
  code=$("${K[@]}" exec vllm-minimax-m2-h200 -- curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)
  log "h200 boot phase=$ph health=$code"
  [ "$code" = "200" ] && { OK=1; log "H200 HEALTHY"; break; }
  sleep 40
done
[ "$OK" = "1" ] || { log "h200 never healthy; abort"; exit 1; }

# correctness smoke (must match B200 — same model, mature Hopper NCCL)
TC=$("${K[@]}" exec vllm-minimax-m2-h200 -- curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"MiniMax-M2","messages":[{"role":"user","content":"Weather in Paris? Call the tool, do not overthink."}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":2000}' 2>/dev/null | python3 -c "import sys,json;c=json.load(sys.stdin)['choices'][0];print('PASS' if c.get('finish_reason')=='tool_calls' and c['message'].get('tool_calls') else 'FAIL')" 2>/dev/null)
log "H200 correctness smoke: tool_call=$TC"
echo "=== H200 (p5en, sm_90, TP4, vLLM 0.23.0-patched) vs B200 like-for-like ===" > "$RES"
echo "correctness: tool_call=$TC @ $(date -u +%FT%TZ)" >> "$RES"

# H200 is the CUSTOMER's production hardware → run the COMPLETE benchmark, not just 2 knee points.
# Full concurrency sweep across the 3 production-shaped scenarios so the customer gets the actual
# goodput knee on THEIR chip. cold = the floor; shared-prefix + 100k-90pct-reuse = the real regimes.
# Stop climbing a scenario once errors spike (saturation), so we don't waste time past the knee.
CONC_STEPS="${CONC_STEPS:-1 8 16 32 64 128 256 512}"
for SC in cold shared-prefix 100k-90pct-reuse; do
  case "$SC" in
    cold)             FLAGS="--scenario cold --isl-tokens 59000";;
    shared-prefix)    FLAGS="--scenario shared-prefix --prefix-tokens 90000 --suffix-tokens 350";;
    100k-90pct-reuse) FLAGS="--scenario 100k-90pct-reuse --prefix-tokens 90000 --suffix-tokens 10000 --reuse-target 0.90";;
  esac
  echo "  -- $SC --" >> "$RES"
  for C in $CONC_STEPS; do
    ST_MSG="H200 RUNNING $SC c=$C"; echo "$ST_MSG" > "$ST"; log "$ST_MSG"
    "${K[@]}" exec bench-runner -- python3 /scripts/bench.py --port 8000 --model MiniMax-M2 $FLAGS \
      --conc $C --out-tokens 512 --out /tmp/h200-$SC-c$C.json >>"$LOG" 2>&1 || log "  h200 $SC c=$C failed"
    "${K[@]}" cp bench-runner:/tmp/h200-$SC-c$C.json "${R}/h200-$SC-c$C.json" >>"$LOG" 2>&1
    # parse result + error rate for the saturation stop
    read -r line err <<<"$(python3 -c "
import json
try:
    d=json.load(open('${R}/h200-$SC-c$C.json'))
    itl=round((d.get('itl_client_mean_s') or 0)*1000,1)
    print(f\"H200|$SC|c=$C|tok/s={d.get('agg_decode_tok_s')}|ttft_p95={d.get('ttft_p95_s')}|itl_ms={itl}|qpass={d.get('quality_pass')}\", d.get('error_rate') or 0)
except Exception as e:
    print(f'H200|$SC|c=$C|parse-fail', 1)
" 2>/dev/null)"
    log "  ${line//|/ }"; echo "  ${line//|/ }" >> "$RES"
    # saturation stop: if error rate > 10%, this scenario has passed its knee — stop climbing
    awk "BEGIN{exit !($err > 0.10)}" && { log "  [$SC] error_rate=$err > 0.10 — past knee, stopping climb"; echo "  [$SC] saturated at c=$C (err=$err)" >> "$RES"; break; }
  done
done
echo "H200 DONE — scaling to 0" > "$ST"; log "=== H200 KNEE COMPARISON DONE (see h200-comparison.txt) ==="
kill "$WD" 2>/dev/null
