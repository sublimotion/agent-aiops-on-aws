#!/usr/bin/env bash
# Round 2 benchmarks: prefix caching enabled, long context up to 512K.
# Workloads: W0 sharegpt, T1 prefix-warm vs cold, T3 long-context (64K-512K), 1M single-stream.
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${MODEL_NAME:-DeepSeek-V4-Flash}"
TOKENIZER_PATH=/mnt/nvme/models/DeepSeek-V4-Flash
RESULT_DIR="/results/r2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"
export RESULT_DIR

cd "$(dirname "$0")"
source ./benchmark-helpers.sh

echo "=== DSv4 Flash Round 2 — $(date) ==="
echo "RESULT_DIR=$RESULT_DIR"

# Health check
curl -sf "${VLLM_URL}/health" >/dev/null || { echo "[error] vLLM unhealthy"; exit 1; }

# smoke
RESPONSE=$(curl -s "${VLLM_URL}/v1/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":10,\"temperature\":0}")
echo "$RESPONSE" | python3 -c "import sys,json; print('Smoke:', json.load(sys.stdin)['choices'][0]['text'][:120])"

capture_metrics pre "$VLLM_URL"
capture_kv_metrics pre "$VLLM_URL"

run_vbench() {
  local label="$1"; shift
  echo
  echo ">>> [${label}] $@"
  vllm bench serve \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER_PATH" \
    --base-url "$VLLM_URL" \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "${label}.json" \
    "$@" 2>&1 | tee "${RESULT_DIR}/${label}.log" | grep -E "Successful|Failed|Output token|Median TTFT|P99 TTFT|Median TPOT|Median ITL|Total token" || echo "[fail] $label"
}

##############################################
# W0 — ShareGPT-style real distribution
##############################################
echo
echo "================================ W0: ShareGPT real distribution ================================"
for QPS in 1.0 4.0 8.0; do
  run_vbench "w0_sharegpt_qps${QPS}" \
    --dataset-name sharegpt \
    --dataset-path /sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 200 --request-rate "$QPS"
  sleep 20
done

##############################################
# T1 — Prefix caching cold vs warm
# Use generated-shared-prefix dataset to maximize prefix reuse signal.
##############################################
echo
echo "================================ T1: Prefix caching cold vs warm ================================"
# First request = cold; subsequent should hit warm prefix cache
for CTX_LBL in 4k 16k 32k; do
  case "$CTX_LBL" in
    4k)  GSP_SYS=3500;  GSP_Q=512;  GSP_OUT=256 ;;
    16k) GSP_SYS=15500; GSP_Q=512;  GSP_OUT=256 ;;
    32k) GSP_SYS=31500; GSP_Q=512;  GSP_OUT=256 ;;
  esac
  echo
  echo "=== T1 ${CTX_LBL}: shared system prompt ${GSP_SYS} tokens, question ${GSP_Q} ==="
  run_vbench "t1_gsp_${CTX_LBL}" \
    --dataset-name random \
    --random-input-len "$((GSP_SYS + GSP_Q))" --random-output-len "$GSP_OUT" \
    --random-prefix-len "$GSP_SYS" \
    --num-prompts 50 --request-rate 1.0
  sleep 15
done

##############################################
# T3 — Long-context scaling (rag-1m-context style)
# Note: server max-model-len=524288, so we sweep up to 512K here.
# 1M single-stream test runs in r3 if possible (needs config change).
##############################################
echo
echo "================================ T3: Long-context sweep ================================"
for CTX in 65536 131072 262144 524288; do
  CTX_LBL=$(( CTX / 1024 ))k
  N_PROMPTS=12
  if [ "$CTX" -ge 262144 ]; then N_PROMPTS=6; fi
  if [ "$CTX" -eq 524288 ]; then N_PROMPTS=3; fi
  echo
  echo "=== T3 ${CTX_LBL}: ${CTX} input tokens ==="
  run_vbench "t3_ctx${CTX_LBL}" \
    --dataset-name random \
    --random-input-len $((CTX - 1024)) --random-output-len 1024 \
    --num-prompts "$N_PROMPTS" --request-rate 0.1 \
    --max-concurrency 2
  sleep 30
done

# Capture mid-run metrics (long-context KV pressure)
capture_metrics mid_t3 "$VLLM_URL"
capture_kv_metrics mid_t3 "$VLLM_URL"

##############################################
# Single-stream sanity at very long context (within 512K)
##############################################
echo
echo "================================ Single-stream long-ctx ================================"
for CTX_TOKENS in 131072 262144 400000; do
  CTX_LBL=$(( CTX_TOKENS / 1024 ))k
  echo
  echo "=== single-stream ${CTX_LBL} ==="
  run_vbench "single_${CTX_LBL}" \
    --dataset-name random \
    --random-input-len $((CTX_TOKENS - 256)) --random-output-len 256 \
    --num-prompts 1 --request-rate 1.0 --max-concurrency 1
done

capture_metrics post "$VLLM_URL"
capture_kv_metrics post "$VLLM_URL"

echo
echo "=== R2 sweep complete — results in $RESULT_DIR ==="
ls -la "$RESULT_DIR"
echo
echo "=== Summary ==="
for f in "$RESULT_DIR"/*.json; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f") ---"
  python3 -c "
import json,sys
try:
  d=json.load(open('$f'))
  for k in ['request_throughput','output_throughput','total_token_throughput','mean_ttft_ms','median_ttft_ms','p99_ttft_ms','median_tpot_ms','p99_tpot_ms','median_itl_ms','p99_itl_ms']:
    if k in d: print(f'  {k}: {d[k]:.1f}')
except Exception as e:
  print('  (err)', e)
"
done
