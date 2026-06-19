#!/usr/bin/env bash
# Stage 6a baseline sweep driver. Run from the workstation; execs into bench-runner.
# Usage: ./run-sweep.sh <engine> <port>   e.g. ./run-sweep.sh sglang 30000  /  ./run-sweep.sh vllm 8000
set -euo pipefail
ENGINE="${1:?engine: sglang|vllm}"
PORT="${2:?port}"
VARIANT="${3:-baseline}"                  # e.g. baseline | fp8kv | hicache | eagle3
LEVELS=(64 128 256 512 1024)              # knee is ~512-1024; >1536 times out (baseline c1536 had 1642 errs)
PROM="http://localhost:9090"
TAG="${ENGINE}-${VARIANT}"
OUTDIR="/mnt/nvme/bench-results/${TAG}"

kubectl exec bench-runner -- mkdir -p "$OUTDIR"
echo "=== ${TAG} concurrency sweep @ 31404-ctx / 1024-out ==="
for C in "${LEVELS[@]}"; do
  echo "--- concurrency=$C ---"
  T0=$(date +%s)
  kubectl exec bench-runner -- python3 /scripts/bench.py \
    --engine "$ENGINE" --port "$PORT" --conc "$C" --total $((C*2)) \
    --prefix-tokens 23250 --suffix-tokens 8154 --out-tokens 1024 \
    --out "${OUTDIR}/c${C}.json" || { echo "level $C failed (likely OOM/saturation) — stopping sweep"; break; }
  T1=$(date +%s)
  # Pull TTFT/TPOT percentiles + KV + DCGM from Prometheus over the run window
  W=$((T1-T0))s
  kubectl exec bench-runner -- python3 - "$ENGINE" "$W" "${OUTDIR}/c${C}.prom.json" <<'PY'
import sys,urllib.request,urllib.parse,json
eng,win,out=sys.argv[1],sys.argv[2],sys.argv[3]
P="http://localhost:9090/api/v1/query"
def q(expr):
    try:
        r=json.load(urllib.request.urlopen(f"{P}?query="+urllib.parse.quote(expr),timeout=10))
        res=r["data"]["result"]; return float(res[0]["value"][1]) if res else None
    except Exception as e: return f"err:{e}"
# SGLang KV/queue gauges are the reliable knee signal here (DCGM PROF not exported — L8b)
m={
 "ttft_p50":f'histogram_quantile(0.50,sum(rate({eng}:time_to_first_token_seconds_bucket[{win}]))by(le))',
 "ttft_p99":f'histogram_quantile(0.99,sum(rate({eng}:time_to_first_token_seconds_bucket[{win}]))by(le))',
 "kv_token_usage":f'max_over_time({eng}:token_usage[{win}])' if eng=="sglang" else f'max_over_time(vllm:gpu_cache_usage_perc[{win}])',
 "max_running":f'max_over_time({eng}:num_running_reqs[{win}])' if eng=="sglang" else f'max_over_time(vllm:num_requests_running[{win}])',
 "max_queue":f'max_over_time({eng}:num_queue_reqs[{win}])' if eng=="sglang" else f'max_over_time(vllm:num_requests_waiting[{win}])',
 "power_w":f'avg(avg_over_time(DCGM_FI_DEV_POWER_USAGE[{win}]))',
}
res={k:q(v) for k,v in m.items()}
json.dump(res,open(out,"w"),indent=2); print(json.dumps(res,indent=2))
PY
done
echo "=== sweep done; results in bench-runner:${OUTDIR} ==="
kubectl exec bench-runner -- sh -c "cat ${OUTDIR}/c*.json"
