#!/bin/sh
# Sample DCGM prof metrics from localhost:9400 every INTERVAL secs for DURATION secs.
# Emits one CSV row per sample: ts,gpu,gr_engine,dram,tensor,power,fb_used
# Run from a hostNetwork pod on the B200 node. Aggregate offline.
URL="${URL:-http://localhost:9400/metrics}"
INTERVAL="${INTERVAL:-3}"
DURATION="${DURATION:-300}"
OUT="${OUT:-/tmp/dcgm-sample.csv}"
echo "ts,gpu,gr_engine_active,dram_active,tensor_active,power_w,fb_used_mib" > "$OUT"
end=$(( $(date +%s) + DURATION ))
while [ "$(date +%s)" -lt "$end" ]; do
  ts=$(date +%s)
  curl -s "$URL" | gawk -v ts="$ts" '
    function gpu(s){ match(s,/gpu="([0-9]+)"/,m); return m[1] }
    /^DCGM_FI_PROF_GR_ENGINE_ACTIVE/   {gr[gpu($0)]=$NF}
    /^DCGM_FI_PROF_DRAM_ACTIVE/        {dr[gpu($0)]=$NF}
    /^DCGM_FI_PROF_PIPE_TENSOR_ACTIVE/ {te[gpu($0)]=$NF}
    /^DCGM_FI_DEV_POWER_USAGE/         {pw[gpu($0)]=$NF}
    /^DCGM_FI_DEV_FB_USED/             {fb[gpu($0)]=$NF}
    END{for(g=0;g<8;g++) printf "%s,%d,%s,%s,%s,%s,%s\n", ts,g,gr[g],dr[g],te[g],pw[g],fb[g]}
  ' >> "$OUT"
  sleep "$INTERVAL"
done
echo "[dcgm-sampler] wrote $OUT ($(wc -l < "$OUT") rows)"
