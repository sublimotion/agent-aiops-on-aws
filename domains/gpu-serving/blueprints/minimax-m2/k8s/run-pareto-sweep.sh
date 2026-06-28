#!/usr/bin/env bash
# =============================================================================
# MiniMax-M2 B200 — DETACHED, UNATTENDED Pareto-map sweep runner.
#
# This is a DETERMINISTIC bash runner (NOT an agent loop). It is meant to be
# launched with nohup/run_in_background on the workstation and survive
# disconnection. The OPERATOR DOES NOT BABYSIT IT.
#
# THE SINGLE MOST IMPORTANT PROPERTY: on ANY exit path — normal completion, the
# 420-min wall-clock cap, Ctrl-C, or any fatal error — the EXIT trap scales the
# B200 nodegroup to desiredSize=0. An idle B200 left billing is the worst outcome.
#
# Grid (Pareto MAP — no SLO gate, no hill-climb; run the whole grid, report frontier):
#   parallelism shapes : tp4 tp4ep4 tp2dp2 tp4dp2 tp2dp4   (TP in {2,4} — TP8 INVALID per FP8 block-128)
#   kv arms            : gpu-only cpu-offload nvme-tiering
#   scenarios          : cold  shared-prefix(90k)
#   concurrency        : 1 4 8 16 32 64 128 (stop a config early once error rate spikes)
#
# Usage:
#   ./run-pareto-sweep.sh            # full grid, detached
#   VALIDATE_ONLY=1 ./run-pareto-sweep.sh   # ONE config (tp4 gpu-only) @ c=1,8 to prove plumbing
#   SHAPES="tp4" KV_ARMS="gpu-only" ./run-pareto-sweep.sh   # subset override
# =============================================================================
set -uo pipefail   # NOT -e: a failed config must NOT kill the run (it must continue + still scale down)

# ── Config ──────────────────────────────────────────────────────────────────
CLUSTER="qwen3-next-bench-eks-cluster"
NODEGROUP="ai-infra-use2-b200-spot"
REGION="us-east-2"
EXPECT_CONTEXT="${EXPECT_CONTEXT:-qwen3-next-bench-eks-cluster}"   # kubectl context that maps to CLUSTER. PINNED (restored 2026-06-27 after usw2 flip).
# We pass --context EXPLICITLY to every kubectl call so a context drift in another shell
# can NEVER redirect this run at a different cluster (the 2026-06-27 wrong-cluster incident).
KCTX=(kubectl --context "$EXPECT_CONTEXT")
# Scaledown SAFETY INTERLOCK: only scale the nodegroup to 0 if we VERIFIED we were operating
# the intended cluster (preflight passed) AND actually drove >=1 benchmark run. A run that
# never touched the target must NOT terminate the target's node on exit.
PREFLIGHT_OK=0
BP="/Users/phi/Documents/workbench/agent-aiops-on-aws/domains/gpu-serving/blueprints/minimax-m2"
K8S="${BP}/k8s"
RESULTS="${BP}/results"
DATE="$(date +%Y-%m-%d)"
LOG="${RESULTS}/sweep.log"
STATUS="${RESULTS}/STATUS"
PARETO="${RESULTS}/pareto-${DATE}.json"
TRAJ="${RESULTS}/optimization-trajectory-${DATE}.json"
HARD_TIMEOUT_MIN=420
PROM_POD="observability-bench"
RUNNER_POD="bench-runner"

SHAPES="${SHAPES:-tp4 tp4ep4 tp2dp2 tp4dp2 tp2dp4}"
KV_ARMS="${KV_ARMS:-gpu-only cpu-offload nvme-tiering}"
# 100k-90pct-reuse is the customer's HEADLINE production scenario (Req 1): total ISL≈100k =
# 90k byte-identical shared prefix + ~10k unique suffix. Listed last so cold/shared-prefix
# warm the engine first; it is the scenario we push concurrency hardest on.
SCENARIOS="${SCENARIOS:-cold shared-prefix 100k-90pct-reuse}"
# Concurrency PRESSURE-TEST past 128 (Req 2). Base ladder doubles to 512; PUSH_BEYOND then
# keeps doubling (1024, 2048, ...) on the high-reuse scenario until a saturation signal fires.
CONC_STEPS="${CONC_STEPS:-1 4 8 16 32 64 128 256 512}"
PUSH_BEYOND_LAST="${PUSH_BEYOND_LAST:-1}"      # keep doubling past the last step until saturation
PUSH_BEYOND_MAX="${PUSH_BEYOND_MAX:-8192}"     # hard ceiling on the doubling (safety)
REUSE_TARGET="${REUSE_TARGET:-0.90}"           # the customer's stated >90% reuse claim (measured)
# Saturation signals (ANY fires -> stop climbing this config; record the knee cause):
KV_UTIL_STOP="${KV_UTIL_STOP:-0.97}"           # KV pool full -> preemptions imminent
ERROR_RATE_STOP="${ERROR_RATE_STOP:-0.10}"     # stop climbing a config once error rate exceeds this
ERROR_RATE_SAT="${ERROR_RATE_SAT:-0.02}"       # softer saturation signal per the workload card
THROUGHPUT_PLATEAU_FRAC="${THROUGHPUT_PLATEAU_FRAC:-1.05}"  # tok/s must rise >5% on a doubling, else plateau

mkdir -p "$RESULTS"

# ── Logging ───────────────────────────────────────────────────────────────────
log()    { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
status() { echo "$*" > "$STATUS"; }

# ── THE TRAP: scale nodegroup to 0 on EVERY exit path. Bulletproof + idempotent. ──
SCALED_DOWN=0
scale_to_zero() {
  local reason="$1"
  if [ "$SCALED_DOWN" = "1" ]; then return 0; fi
  SCALED_DOWN=1
  if [ "$PREFLIGHT_OK" != "1" ]; then
    log "TRAP(${reason}): PREFLIGHT did not pass (never confirmed operating ${CLUSTER}) -> NOT scaling down. Node left as-is; verify manually."
    status "EXIT_NO_SCALEDOWN reason=${reason} preflight=failed $(date -u +%FT%TZ)"
    return 0
  fi
  log "TRAP(${reason}): scaling nodegroup ${NODEGROUP} -> desired=0 (no idle B200)."
  status "SCALING_DOWN reason=${reason} $(date -u +%FT%TZ)"
  # Retry up to 5x — this MUST succeed; an unscaled B200 bills ~$18/hr.
  local i
  for i in 1 2 3 4 5; do
    if aws eks update-nodegroup-config \
         --cluster-name "$CLUSTER" --nodegroup-name "$NODEGROUP" --region "$REGION" \
         --scaling-config minSize=0,maxSize=1,desiredSize=0 >>"$LOG" 2>&1; then
      log "scale-to-zero accepted (attempt $i)."
      status "DONE reason=${reason} scaled_down=yes $(date -u +%FT%TZ)"
      return 0
    fi
    log "scale-to-zero attempt $i FAILED; retrying in 15s..."
    sleep 15
  done
  log "FATAL: scale-to-zero FAILED 5x. OPERATOR MUST MANUALLY SCALE ${NODEGROUP} TO 0 IN ${REGION}."
  status "ERROR reason=${reason} scaled_down=FAILED MANUAL_ACTION_REQUIRED $(date -u +%FT%TZ)"
}
on_exit() { rc=$?; scale_to_zero "EXIT(rc=${rc})"; }
trap on_exit EXIT
trap 'log "SIGINT received"; exit 130' INT
trap 'log "SIGTERM received (wall-clock cap or operator)"; exit 143' TERM

# Internal wall-clock watchdog (defense in depth). If the launcher could not wrap us in an
# external `timeout` (WALL_CAP_MIN set), a background timer SIGTERMs THIS process at the cap
# -> TERM trap -> EXIT trap -> scaledown. Guarantees scaledown with no coreutils `timeout`.
if [ -n "${WALL_CAP_MIN:-}" ]; then
  MAIN_PID=$$
  ( sleep "$(( WALL_CAP_MIN * 60 ))"; echo "[watchdog] WALL_CAP_MIN=${WALL_CAP_MIN} reached; SIGTERM ${MAIN_PID}" >> "$LOG"; kill -TERM "$MAIN_PID" 2>/dev/null ) &
  WATCHDOG_PID=$!
  trap 'kill "$WATCHDOG_PID" 2>/dev/null; on_exit' EXIT
  log "internal wall-clock watchdog armed: ${WALL_CAP_MIN}min -> SIGTERM self (pid ${MAIN_PID})"
fi

# ── Prometheus harvest helper. Runs the query INSIDE the prometheus container ──
# (Prometheus on this node is on the host net at :9090; exec into the obs pod to query.)
prom_query() {
  local expr="$1"
  "${KCTX[@]}" exec "$PROM_POD" -c prometheus -- \
    wget -qO- "http://localhost:9090/api/v1/query?query=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$expr")" 2>/dev/null \
    | python3 -c "import sys,json
try:
  d=json.load(sys.stdin); r=d['data']['result']
  print(r[0]['value'][1] if r else 'null')
except Exception: print('null')"
}

# Harvest the full Prometheus signal set over a [window] for a finished run.
# This is the CROSS-CHECK path for the bench.py client-side numbers (Req 2+3): KV residency
# MAX (the customer's "KV cache spikes"), prefix-cache gauges, AND the offload-tier counters so
# the offload/nvme arms can be reconciled against bench.py's kv_tier breakdown.
harvest_prom() {
  local win="$1" outfile="$2"
  # vLLM 0.19.1rc1 metric names verified live (NOT the sidecar's gpu_prefix_cache_hit_rate,
  # which this build does not export; compute hit-rate from counters instead).
  local ttft_p50 ttft_p95 ttft_p99 itl_p95 kv_max preempt qdepth running prefix_hit
  local ph_total pq_total eph_total epq_total ext_hit src_gpu src_ext recomputed
  ttft_p50=$(prom_query "histogram_quantile(0.50,sum(rate(vllm:time_to_first_token_seconds_bucket[${win}]))by(le))")
  ttft_p95=$(prom_query "histogram_quantile(0.95,sum(rate(vllm:time_to_first_token_seconds_bucket[${win}]))by(le))")
  ttft_p99=$(prom_query "histogram_quantile(0.99,sum(rate(vllm:time_to_first_token_seconds_bucket[${win}]))by(le))")
  itl_p95=$(prom_query "histogram_quantile(0.95,sum(rate(vllm:inter_token_latency_seconds_bucket[${win}]))by(le))")
  # KV residency MAX over the window — the customer flagged "KV cache spikes"; this is that signal.
  kv_max=$(prom_query "max_over_time(vllm:kv_cache_usage_perc[${win}])")
  preempt=$(prom_query "increase(vllm:num_preemptions_total[${win}])")
  qdepth=$(prom_query "max_over_time(vllm:num_requests_waiting[${win}])")
  running=$(prom_query "max_over_time(vllm:num_requests_running[${win}])")
  # GPU-HBM (local APC) prefix-cache: raw Δ counters + derived hit-rate.
  ph_total=$(prom_query "increase(vllm:prefix_cache_hits_total[${win}])")
  pq_total=$(prom_query "increase(vllm:prefix_cache_queries_total[${win}])")
  prefix_hit=$(prom_query "increase(vllm:prefix_cache_hits_total[${win}])/clamp_min(increase(vllm:prefix_cache_queries_total[${win}]),1)")
  # OFFLOAD-TIER (KV connector / external) counters — proves CPU/NVMe tier service under reuse (Req 3).
  eph_total=$(prom_query "increase(vllm:external_prefix_cache_hits_total[${win}])")
  epq_total=$(prom_query "increase(vllm:external_prefix_cache_queries_total[${win}])")
  ext_hit=$(prom_query "increase(vllm:external_prefix_cache_hits_total[${win}])/clamp_min(increase(vllm:external_prefix_cache_queries_total[${win}]),1)")
  # prompt-token source split (local_cache_hit=GPU HBM, external_kv_transfer=offload tier, recomputed=offload COST).
  src_gpu=$(prom_query "increase(vllm:prompt_tokens_by_source_total{source=\"local_cache_hit\"}[${win}])")
  src_ext=$(prom_query "increase(vllm:prompt_tokens_by_source_total{source=\"external_kv_transfer\"}[${win}])")
  recomputed=$(prom_query "increase(vllm:prompt_tokens_recomputed_total[${win}])")

  # ── DCGM hardware telemetry (Req 3): answers "WHICH resource bounds us at the knee?" ──
  # Prometheus already scrapes the DCGM exporter (:9400). Confirmed-available non-PROF gauges
  # (FB_USED=KV-capacity ceiling, POWER=compute-bound proxy near TDP, UTIL, TEMP, XID=invalidator).
  local dcgm_power dcgm_temp dcgm_fb dcgm_util dcgm_xid
  dcgm_power=$(prom_query "max_over_time(DCGM_FI_DEV_POWER_USAGE[${win}])")
  dcgm_temp=$(prom_query  "max_over_time(DCGM_FI_DEV_GPU_TEMP[${win}])")
  dcgm_fb=$(prom_query    "max_over_time(DCGM_FI_DEV_FB_USED[${win}])")
  dcgm_util=$(prom_query  "avg_over_time(DCGM_FI_DEV_GPU_UTIL[${win}])")
  dcgm_xid=$(prom_query   "max_over_time(DCGM_FI_DEV_XID_ERRORS[${win}])")
  # PROF roofline classifiers — present ONLY if `dcgmi profile --resume` succeeded at preflight.
  # If unavailable they return 'null'; we record prof_counters_available so the bottleneck-class
  # confidence is explicit (card caveat: fall back to FB+POWER+vLLM-histogram inference).
  local dcgm_dram dcgm_tensor dcgm_sm dcgm_nvtx dcgm_nvrx prof_avail
  dcgm_dram=$(prom_query   "avg_over_time(DCGM_FI_PROF_DRAM_ACTIVE[${win}])")
  dcgm_tensor=$(prom_query "avg_over_time(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE[${win}])")
  dcgm_sm=$(prom_query     "avg_over_time(DCGM_FI_PROF_SM_ACTIVE[${win}])")
  dcgm_nvtx=$(prom_query   "increase(DCGM_FI_PROF_NVLINK_TX_BYTES[${win}])")
  dcgm_nvrx=$(prom_query   "increase(DCGM_FI_PROF_NVLINK_RX_BYTES[${win}])")
  if [ "$dcgm_dram" = "null" ] && [ "$dcgm_tensor" = "null" ]; then prof_avail="false"; else prof_avail="true"; fi

  cat > "$outfile" <<JSON
{
  "ttft_p50_s": ${ttft_p50}, "ttft_p95_s": ${ttft_p95}, "ttft_p99_s": ${ttft_p99},
  "itl_p95_s": ${itl_p95}, "kv_utilization_max": ${kv_max},
  "preemptions": ${preempt}, "queue_depth_max": ${qdepth}, "running_max": ${running},
  "prefix_hit_rate": ${prefix_hit},
  "prefix_cache_hits_total_delta": ${ph_total}, "prefix_cache_queries_total_delta": ${pq_total},
  "external_prefix_hit_rate": ${ext_hit},
  "external_prefix_cache_hits_total_delta": ${eph_total}, "external_prefix_cache_queries_total_delta": ${epq_total},
  "tokens_gpu_hbm_hit": ${src_gpu}, "tokens_offload_tier_hit": ${src_ext}, "tokens_recomputed": ${recomputed},
  "kv_transfer_bytes_latency_exposed": false,
  "dcgm": {
    "prof_counters_available": ${prof_avail},
    "power_usage_w_max": ${dcgm_power}, "gpu_temp_c_max": ${dcgm_temp},
    "fb_used_mib_max": ${dcgm_fb}, "gpu_util_avg": ${dcgm_util}, "xid_errors_max": ${dcgm_xid},
    "prof_dram_active_avg": ${dcgm_dram}, "prof_pipe_tensor_active_avg": ${dcgm_tensor},
    "prof_sm_active_avg": ${dcgm_sm},
    "prof_nvlink_tx_bytes": ${dcgm_nvtx}, "prof_nvlink_rx_bytes": ${dcgm_nvrx}
  }
}
JSON
}

# Read enable_chunked_prefill straight off the live engine (offload-regression assert).
chunked_prefill_active() {
  "${KCTX[@]}" exec "$1" -- python3 -c "
import urllib.request,re
m=urllib.request.urlopen('http://localhost:8000/metrics').read().decode()
for l in m.splitlines():
  if l.startswith('vllm:cache_config_info'):
    print('true' if 'enable_chunked_prefill=\"True\"' in l else 'false'); break
else: print('unknown')
" 2>/dev/null || echo "unknown"
}

# ── DCGM PROF-counter enable attempt (Req 3) ──────────────────────────────────
# The roofline classifiers (DRAM_ACTIVE=HBM-BW-bound, PIPE_TENSOR_ACTIVE=compute-bound,
# NVLINK_TX/RX=transfer) were NOT enabled on the first node (lessons stage-4b; kimi-k2.6-nvfp4
# L11). Try to resume the DCGM profiling module inside the privileged dcgm-exporter container.
# IF it fails we DO NOT abort — we fall back to FB_USED+POWER+vLLM-histogram inference and
# record PROF as unavailable in every harvested point (prof_counters_available=false).
PROF_ENABLED="unknown"
try_enable_dcgm_prof() {
  # dcgmi lives in the dcgm-exporter container of the observability-bench pod.
  if "${KCTX[@]}" exec "$PROM_POD" -c dcgm-exporter -- bash -c \
       'command -v dcgmi >/dev/null 2>&1 && (dcgmi profile --resume 2>&1 || dcgmi profile -r 2>&1) || true' >>"$LOG" 2>&1; then
    # Confirm via Prometheus that a PROF series now exists.
    local probe; probe=$(prom_query "DCGM_FI_PROF_DRAM_ACTIVE")
    if [ "$probe" != "null" ]; then PROF_ENABLED="true"; else PROF_ENABLED="false"; fi
  else
    PROF_ENABLED="false"
  fi
  log "DCGM PROF counters: prof_counters_available=${PROF_ENABLED} (false => roofline by FB+POWER+vLLM-histogram inference)"
}

# ── Pareto / trajectory accumulators ──────────────────────────────────────────
# DURABLE per-point ledger. EVERY emitted pareto point is appended here as one JSON
# line (append-only, no stdin/heredoc games). The headline pareto-<date>.json is
# REBUILT from this ledger at the end (finalize_pareto). This is the same pattern the
# trajectory already uses (.traj-nodes.jsonl) and is why the trajectory survived while
# the old incremental pareto did not.
#
# ROOT CAUSE of the empty pareto (2026-06-27): the old append_point did
#   python3 - "$PARETO" <<'PY' ... obj=json.load(sys.stdin) ... PY
# A `<<'PY'` heredoc REDIRECTS stdin to the heredoc body, so json.load(sys.stdin)
# read the (already-consumed) script text -> "Expecting value: line 1 column 1" ->
# the point was NEVER appended -> pareto stayed []. (Those JSONDecodeErrors in the log
# were append_point dying, NOT the tar|json.load harvest step.) Fix: pass the point via
# a FILE argument, never stdin, and rebuild from the ledger so a single bad append can
# never zero out the headline file.
echo "[]" > "$PARETO"
PARETO_POINTS="${RESULTS}/.pareto-points.jsonl"; : > "$PARETO_POINTS"
append_point() {  # $1 = path to a file containing ONE pareto-point JSON object
  local pointfile="$1"
  python3 - "$PARETO" "$PARETO_POINTS" "$pointfile" <<'PY'
import json,sys
pareto_f,ledger_f,pointfile=sys.argv[1],sys.argv[2],sys.argv[3]
# Defensive load of the point: empty/missing/malformed -> skip + note, never crash.
try:
    with open(pointfile) as fh:
        obj=json.load(fh)
except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
    obj={"_harvest_error": f"unreadable pareto point {pointfile}: {e}", "client": None, "prometheus": None}
# Append to the durable ledger (one JSON object per line).
with open(ledger_f,"a") as fh:
    fh.write(json.dumps(obj)+"\n")
# Mirror into the headline array too (defensive read of the current array).
try:
    with open(pareto_f) as fh:
        arr=json.load(fh)
    if not isinstance(arr,list): arr=[]
except (FileNotFoundError, json.JSONDecodeError, ValueError):
    arr=[]
arr.append(obj)
json.dump(arr,open(pareto_f,"w"),indent=2)
PY
}
# Rebuild the headline pareto-<date>.json from the durable per-point ledger. Called at
# the end so the deliverable reflects EVERY point that was emitted, even if an
# incremental mirror-append ever failed. Defensive: bad lines are recorded, not fatal.
finalize_pareto() {
  python3 - "$PARETO" "$PARETO_POINTS" <<'PY'
import json,sys
out,ledger_f=sys.argv[1],sys.argv[2]
pts=[]
try:
    lines=open(ledger_f).read().splitlines()
except FileNotFoundError:
    lines=[]
for n,l in enumerate(lines,1):
    if not l.strip(): continue
    try:
        pts.append(json.loads(l))
    except json.JSONDecodeError as e:
        pts.append({"_harvest_error": f"unparseable ledger line {n}: {e}"})
json.dump(pts,open(out,"w"),indent=2)
print(f"pareto: {len(pts)} points -> {out}")
PY
}
TRAJ_NODES="${RESULTS}/.traj-nodes.jsonl"; : > "$TRAJ_NODES"
finalize_trajectory() {
  python3 - "$TRAJ" "$TRAJ_NODES" <<'PY'
import json,sys
out,nodes_f=sys.argv[1],sys.argv[2]
nodes=[]
try:
    lines=open(nodes_f).read().splitlines()
except FileNotFoundError:
    lines=[]
for n,l in enumerate(lines,1):
    if not l.strip(): continue
    try:
        nodes.append(json.loads(l))
    except json.JSONDecodeError as e:
        nodes.append({"_harvest_error": f"unparseable traj line {n}: {e}", "status":"harvest-error"})
doc={"objective":"pareto_map: output_tokens_per_sec vs concurrency (TTFT/ITL reported, not gated)",
     "regime":"prefill-heavy huge-context MoE | B200 SM100 NVSwitch | KV-capacity/reuse + prefill-latency",
     "guardrail":"lightweight garbage screen: no repetition loop / empty / broken tool-parse; error_rate<=0.01",
     "nodes":nodes}
json.dump(doc,open(out,"w"),indent=2)
print(f"trajectory: {len(nodes)} nodes -> {out}")
PY
}

# ── Serving-pod lifecycle for a (shape, arm) ──────────────────────────────────
BASELINE_POD="vllm-minimax-m2-baseline"   # already running TP4 gpu-only
wait_ready() {  # $1=pod  $2=timeout_s
  local pod="$1" to="${2:-3000}" t=0
  while [ $t -lt $to ]; do
    if "${KCTX[@]}" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null | grep -q true; then
      return 0
    fi
    local ph; ph=$("${KCTX[@]}" get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null)
    if [ "$ph" = "Failed" ] || [ "$ph" = "Succeeded" ]; then
      log "  pod ${pod} entered phase=${ph} before ready"; return 1
    fi
    sleep 15; t=$((t+15))
  done
  log "  pod ${pod} not ready after ${to}s"; return 1
}

ensure_serving_pod() {  # $1=shape $2=arm -> echoes pod name on stdout, returns 0/1
  local shape="$1" arm="$2" pod manifest
  if [ "$shape" = "tp4" ] && [ "$arm" = "gpu-only" ]; then
    # Reuse the already-running, already-correctness-gated baseline.
    if "${KCTX[@]}" get pod "$BASELINE_POD" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null | grep -q true; then
      echo "$BASELINE_POD"; return 0
    fi
  fi
  pod="vllm-${shape}-${arm}"
  manifest="/tmp/${pod}.yaml"
  bash "${K8S}/gen-serving-manifest.sh" "$shape" "$arm" > "$manifest" 2>>"$LOG" || { log "  gen manifest FAILED ${shape}/${arm}"; return 1; }
  # tear down any non-baseline serving pods to free GPUs before launching the new shape
  "${KCTX[@]}" delete pod -l app=vllm-minimax-m2 --field-selector "metadata.name!=${BASELINE_POD}" --ignore-not-found --wait=true >>"$LOG" 2>&1
  # if the new shape needs >4 GPUs (dp2/dp4 use 8) we must also evict the baseline
  case "$shape" in tp4dp2|tp2dp4) "${KCTX[@]}" delete pod "$BASELINE_POD" --ignore-not-found --wait=true >>"$LOG" 2>&1 ;; esac
  log "  launching ${pod} (boot budget ~40min)..."
  "${KCTX[@]}" apply -f "$manifest" >>"$LOG" 2>&1 || { log "  apply FAILED ${pod}"; return 1; }
  if wait_ready "$pod" 3000; then echo "$pod"; return 0; fi
  log "  ${pod} FAILED to become ready — capturing last logs"
  "${KCTX[@]}" logs "$pod" --tail=30 >>"$LOG" 2>&1
  "${KCTX[@]}" delete pod "$pod" --ignore-not-found >>"$LOG" 2>&1
  return 1
}

# ── Run one config (shape × arm × scenario): the concurrency sweep ─────────────
run_config() {
  local shape="$1" arm="$2" scenario="$3" pod="$4"
  local cfg="${shape}-${arm}-${scenario}"
  local outdir="/mnt/nvme/bench-results/${cfg}"
  "${KCTX[@]}" exec "$RUNNER_POD" -- mkdir -p "$outdir" >>"$LOG" 2>&1

  # offload-regression assert: chunked prefill must stay TRUE when a KV connector engages
  local chunked; chunked=$(chunked_prefill_active "$pod")
  log "  [${cfg}] chunked_prefill_active=${chunked}"
  if [ "$arm" != "gpu-only" ] && [ "$chunked" != "true" ]; then
    log "  [${cfg}] WARNING: KV connector engaged but chunked_prefill=${chunked} (known regression mechanism — flagged)"
  fi

  local scen_flags
  case "$scenario" in
    shared-prefix)
      scen_flags="--scenario shared-prefix --prefix-tokens 90000 --suffix-tokens 350" ;;
    100k-90pct-reuse)
      # total ISL≈100k = 90k byte-identical shared prefix + ~10k unique suffix → 90% cacheable.
      # --reuse-target makes bench.py MEASURE whether observed cached/prompt actually reaches 0.90.
      scen_flags="--scenario 100k-90pct-reuse --prefix-tokens 90000 --suffix-tokens 10000 --reuse-target ${REUSE_TARGET}" ;;
    *)
      scen_flags="--scenario cold --isl-tokens 59000" ;;
  esac

  # Build the concurrency ladder. push_beyond_last (Req 2): on the high-reuse scenario, keep
  # DOUBLING past the last fixed step (1024, 2048, ...) until a saturation signal fires —
  # a 90%-cached workload has only ~10k fresh tokens/req so its knee may be FAR past 128.
  local -a CONC_LIST=($CONC_STEPS)
  local prev_tput="" knee_cause=""
  local i=0
  while [ $i -lt ${#CONC_LIST[@]} ]; do
    local C="${CONC_LIST[$i]}"
    i=$((i+1))
    status "RUNNING cfg=${cfg} conc=${C} | $(_progress_str)"
    log "  [${cfg}] concurrency=${C}"
    local t0 t1 win local_json prom_json qpass
    t0=$(date +%s)
    # run bench.py; exit code 3 = quality breach. capture rc without killing the run.
    "${KCTX[@]}" exec "$RUNNER_POD" -- python3 /scripts/bench.py \
      --port 8000 --model MiniMax-M2 --conc "$C" $scen_flags \
      --out-tokens 512 --out "${outdir}/c${C}.json" >>"$LOG" 2>&1
    local rc=$?
    t1=$(date +%s); win="$((t1-t0+5))s"
    qpass="true"; [ "$rc" = "3" ] && qpass="false"

    # pull the client-side json + harvest Prometheus over the run window
    prom_json="/tmp/${cfg}-c${C}.prom.json"
    harvest_prom "$win" "$prom_json"
    "${KCTX[@]}" cp "${RUNNER_POD}:${outdir}/c${C}.json" "/tmp/${cfg}-c${C}.client.json" >>"$LOG" 2>&1

    # error-rate read for the stop rule
    local err_rate
    err_rate=$(python3 -c "import json;print(json.load(open('/tmp/${cfg}-c${C}.client.json')).get('error_rate') or 0)" 2>/dev/null || echo 1)

    # merge client + prom into one pareto point, write it to a FILE, then append.
    # We write to a file (NOT a pipe into a heredoc-script) because a `<<'PY'` heredoc
    # owns stdin — piping JSON into append_point's stdin silently lost every point
    # (the 2026-06-27 empty-pareto bug). The defensive json.load below + append_point's
    # own defensive read mean a missing/empty/corrupt client or prom file degrades to a
    # null-bearing point with a note, never a crash and never a dropped point.
    local pointfile="/tmp/${cfg}-c${C}.point.json"
    python3 - "$shape" "$arm" "$scenario" "$C" "$pod" "$chunked" "$qpass" \
      "/tmp/${cfg}-c${C}.client.json" "$prom_json" "$pointfile" <<'PY'
import json,sys,os
shape,arm,scen,C,pod,chunked,qpass,cf,pf,out=sys.argv[1:11]
def load(p):
    try:
        with open(p) as fh: return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        return {"_harvest_error": f"unreadable {os.path.basename(p)}: {e}"}
client=load(cf); prom=load(pf)
pt={"config":f"{shape}-{arm}", "shape":shape, "kv_arm":arm, "scenario":scen,
    "concurrency":int(C), "pod":pod, "chunked_prefill_active":chunked,
    "quality_pass":(qpass=="true"),
    "client":client, "prometheus":prom}
json.dump(pt, open(out,"w"))
PY
    append_point "$pointfile"

    # trajectory node (lineage: parent = baseline; lever_delta = the shape/arm change)
    python3 - "$shape" "$arm" "$scenario" "$C" "$qpass" "$err_rate" \
      "/tmp/${cfg}-c${C}.client.json" "$prom_json" >> "$TRAJ_NODES" <<'PY'
import json,sys,os
shape,arm,scen,C,qpass,err,cf,pf=sys.argv[1:9]
def load(p):
    try:
        with open(p) as fh: return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}
client=load(cf); prom=load(pf)
status="kept"
if qpass!="true": status="quality_breach"
elif float(err or 0) > 0.10: status="dead-end"
node={"id":f"{shape}-{arm}-{scen}-c{C}","parent":"tp4-gpu-only-baseline",
      "lever_delta":[f"shape={shape}",f"kv_arm={arm}",f"scenario={scen}",f"conc={C}"],
      "confidence":"code-confirmed",
      "objective_value":client.get("agg_decode_tok_s"),
      "guardrail_value":f"quality_pass={qpass} err={err}",
      "regime":"prefill-heavy huge-context MoE | B200 SM100 | KV-reuse",
      "status":status,
      "ttft_p95_s":prom.get("ttft_p95_s"),"itl_p95_s":prom.get("itl_p95_s"),
      "prefix_hit_rate":prom.get("prefix_hit_rate")}
print(json.dumps(node))
PY

    DONE_RUNS=$((DONE_RUNS+1))
    log "  [${cfg}] c=${C} done: tok/s=$(python3 -c "import json;print(json.load(open('/tmp/${cfg}-c${C}.client.json')).get('agg_decode_tok_s'))" 2>/dev/null) err=${err_rate} ttft_p95=$(python3 -c "import json;print(json.load(open('${prom_json}')).get('ttft_p95_s'))" 2>/dev/null) qpass=${qpass}"

    # ── SATURATION signals (workload card): stop climbing this config when ANY fires.
    # The Pareto map still KEEPS the saturated point; we just stop pushing concurrency higher.
    # Signals: error_rate>2% (and the harder >10% dead-end stop), KV residency>97%,
    # preemptions>0, or throughput plateau (tok/s failed to rise >5% on the last doubling).
    local cur_tput kv_max preempt
    cur_tput=$(python3 -c "import json;print(json.load(open('/tmp/${cfg}-c${C}.client.json')).get('agg_decode_tok_s') or 0)" 2>/dev/null || echo 0)
    kv_max=$(python3 -c "import json;v=json.load(open('${prom_json}')).get('kv_utilization_max');print(v if v not in (None,'null') else 0)" 2>/dev/null || echo 0)
    preempt=$(python3 -c "import json;v=json.load(open('${prom_json}')).get('preemptions');print(v if v not in (None,'null') else 0)" 2>/dev/null || echo 0)

    knee_cause=$(python3 - "$err_rate" "$kv_max" "$preempt" "$cur_tput" "$prev_tput" \
        "$ERROR_RATE_SAT" "$ERROR_RATE_STOP" "$KV_UTIL_STOP" "$THROUGHPUT_PLATEAU_FRAC" <<'PY'
import sys
err,kv,pre,cur,prev,esat,estop,kvstop,platf=sys.argv[1:10]
def f(x):
    try: return float(x)
    except: return 0.0
err,kv,pre,cur=f(err),f(kv),f(pre),f(cur)
esat,estop,kvstop,platf=f(esat),f(estop),f(kvstop),f(platf)
prev=f(prev) if prev not in ("","None") else None
causes=[]
if err>estop: causes.append(f"error_rate>{estop}")
elif err>esat: causes.append(f"error_rate>{esat}")
if kvstop and kv>kvstop: causes.append(f"kv_utilization>{kvstop}")
if pre>0: causes.append("preemptions>0")
# throughput plateau: tok/s failed to rise >5% vs the previous step (only meaningful once we have a prev)
if prev is not None and prev>0 and cur < prev*platf: causes.append("throughput_plateau")
print(",".join(causes))
PY
)
    prev_tput="$cur_tput"

    if [ -n "$knee_cause" ]; then
      log "  [${cfg}] SATURATION at c=${C}: ${knee_cause} (tok/s=${cur_tput} kv=${kv_max} preempt=${preempt}) — knee found, stopping climb"
      status "KNEE cfg=${cfg} conc=${C} cause=${knee_cause} | $(_progress_str)"
      break
    fi

    # push_beyond_last (Req 2): if we've consumed the fixed ladder, no signal fired, and this is
    # the high-reuse scenario, append the next doubling and KEEP GOING until saturation / ceiling.
    if [ $i -ge ${#CONC_LIST[@]} ] && [ "$PUSH_BEYOND_LAST" = "1" ] && [ "$scenario" = "100k-90pct-reuse" ]; then
      local next=$(( C * 2 ))
      if [ $next -le "$PUSH_BEYOND_MAX" ]; then
        CONC_LIST+=("$next")
        log "  [${cfg}] no saturation at c=${C}; push_beyond -> appending c=${next} (ceiling ${PUSH_BEYOND_MAX})"
      else
        log "  [${cfg}] push_beyond hit ceiling ${PUSH_BEYOND_MAX} at c=${C} — stopping climb (no saturation reached)"
      fi
    fi
  done
}

# ── Progress / ETA bookkeeping ────────────────────────────────────────────────
TOTAL_CONFIGS=0
for s in $SHAPES; do for a in $KV_ARMS; do for sc in $SCENARIOS; do TOTAL_CONFIGS=$((TOTAL_CONFIGS+1)); done; done; done
DONE_CONFIGS=0; DONE_RUNS=0; START_EPOCH=$(date +%s)
_progress_str() {
  local el=$(( ($(date +%s)-START_EPOCH)/60 ))
  echo "config ${DONE_CONFIGS}/${TOTAL_CONFIGS} runs=${DONE_RUNS} elapsed=${el}m cap=${HARD_TIMEOUT_MIN}m"
}

# =============================================================================
# MAIN
# =============================================================================
log "=== MiniMax-M2 B200 Pareto sweep START (date=${DATE}) ==="
log "grid: shapes=[${SHAPES}] arms=[${KV_ARMS}] scenarios=[${SCENARIOS}] conc=[${CONC_STEPS}]"
log "push_beyond_last=${PUSH_BEYOND_LAST} (ceiling=${PUSH_BEYOND_MAX}) reuse_target=${REUSE_TARGET} | sat-signals: err>${ERROR_RATE_SAT}/${ERROR_RATE_STOP} kv>${KV_UTIL_STOP} preempt>0 plateau<${THROUGHPUT_PLATEAU_FRAC}x"
log "total configs=${TOTAL_CONFIGS} | hard cap=${HARD_TIMEOUT_MIN}min | nodegroup=${NODEGROUP}"
status "STARTING $(date -u +%FT%TZ) | $(_progress_str)"

# ── PREFLIGHT (the 2026-06-27 wrong-cluster guard) ────────────────────────────
# Confirm the pinned context maps to the intended CLUSTER *and* the B200 node is present.
# scale_to_zero stays DISARMED (PREFLIGHT_OK=0) until BOTH pass — so a misdirected or
# never-started run can never terminate the target's node on exit.
log "PREFLIGHT: binding to context '${EXPECT_CONTEXT}' and verifying cluster=${CLUSTER}..."
ACTUAL_CLUSTER="$(kubectl config view -o jsonpath="{.contexts[?(@.name=='${EXPECT_CONTEXT}')].context.cluster}" 2>/dev/null)"
if ! echo "$ACTUAL_CLUSTER" | grep -q "$CLUSTER"; then
  log "PREFLIGHT FAIL: context '${EXPECT_CONTEXT}' maps to cluster '${ACTUAL_CLUSTER}', expected to contain '${CLUSTER}'. ABORT (no scaledown)."
  status "ABORT preflight=cluster-mismatch $(date -u +%FT%TZ)"
  exit 1
fi
# Server reachable AND the labeled B200 node is Ready on THIS context.
if ! "${KCTX[@]}" get nodes -l blueprint=minimax-m2 2>/dev/null | grep -q "Ready"; then
  log "PREFLIGHT FAIL: no Ready node with label blueprint=minimax-m2 on context '${EXPECT_CONTEXT}'. ABORT (no scaledown)."
  status "ABORT preflight=no-node $(date -u +%FT%TZ)"
  exit 1
fi
PREFLIGHT_OK=1   # ARM the scaledown — we have CONFIRMED we are driving the intended cluster+node.
log "PREFLIGHT OK: context->cluster verified, B200 node Ready. Scaledown ARMED."

# Ensure the bench-runner pod + ConfigMap exist (idempotent).
"${KCTX[@]}" apply -f "${K8S}/bench-runner.yaml" >>"$LOG" 2>&1
log "waiting for bench-runner pod ready (pip install aiohttp)..."
wait_ready "$RUNNER_POD" 600 || { log "bench-runner never became ready — aborting (trap will scale down)"; exit 1; }

# Attempt to enable DCGM PROF roofline counters (Req 3). Non-fatal — falls back to inference.
try_enable_dcgm_prof

if [ "${VALIDATE_ONLY:-0}" = "1" ]; then
  log "=== VALIDATE_ONLY: tp4 gpu-only, scenario=cold, conc=1 and 8 ==="
  SHAPES="tp4"; KV_ARMS="gpu-only"; SCENARIOS="cold"; CONC_STEPS="1 8"
  TOTAL_CONFIGS=1
fi

for shape in $SHAPES; do
  for arm in $KV_ARMS; do
    DONE_CONFIGS=$((DONE_CONFIGS+1))
    log "--- CONFIG ${DONE_CONFIGS}/${TOTAL_CONFIGS}: shape=${shape} kv_arm=${arm} ---"
    status "BRINGING_UP shape=${shape} arm=${arm} | $(_progress_str)"
    pod=$(ensure_serving_pod "$shape" "$arm")
    if [ -z "$pod" ]; then
      log "  shape=${shape}/${arm} pod did not come up — recording config as dead-end, skipping scenarios"
      echo "{\"id\":\"${shape}-${arm}-NOBOOT\",\"parent\":\"tp4-gpu-only-baseline\",\"lever_delta\":[\"shape=${shape}\",\"kv_arm=${arm}\"],\"confidence\":\"code-confirmed\",\"objective_value\":null,\"guardrail_value\":\"pod failed to boot\",\"regime\":\"B200 SM100\",\"status\":\"dead-end\"}" >> "$TRAJ_NODES"
      continue
    fi
    log "  serving pod ready: ${pod}"
    for scenario in $SCENARIOS; do
      run_config "$shape" "$arm" "$scenario" "$pod"
    done
  done
done

finalize_pareto       # rebuild headline pareto-<date>.json from the durable per-point ledger
finalize_trajectory
log "=== SWEEP COMPLETE: ${DONE_RUNS} runs, ${DONE_CONFIGS} configs. pareto=${PARETO} traj=${TRAJ} ==="
status "COMPLETE runs=${DONE_RUNS} configs=${DONE_CONFIGS} $(date -u +%FT%TZ)"
# EXIT trap fires here -> scale_to_zero
