#!/usr/bin/env bash
# Orchestrate the DeepSeek V4 Flash benchmark sweep — T0 baseline + W0-W6 workloads.
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${MODEL_NAME:-DeepSeek-V4-Flash}"
RESULT_DIR="${RESULT_DIR:-/results/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULT_DIR"
export RESULT_DIR

cd "$(dirname "$0")"
source ./benchmark-helpers.sh

echo "=== DSv4 Flash benchmark — $(date) ==="
echo "VLLM_URL=$VLLM_URL"
echo "MODEL=$MODEL"
echo "RESULT_DIR=$RESULT_DIR"

# Health check
if ! curl -sf "${VLLM_URL}/health" >/dev/null; then
  echo "[error] vLLM health check failed at $VLLM_URL"
  exit 1
fi

# Quick precision smoke test (per SGLang #25662)
echo
echo "=== Precision smoke test ==="
RESPONSE=$(curl -s "${VLLM_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":10,\"temperature\":0}")
echo "$RESPONSE" > "${RESULT_DIR}/smoke_test.json"
echo "Smoke output:" && echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'][:200])"

# Pre-run metrics
echo
echo "=== Pre-run metrics ==="
capture_metrics pre "$VLLM_URL"
capture_kv_metrics pre "$VLLM_URL"

# Helper for vllm bench serve calls
# vllm bench needs a tokenizer path; --tokenizer points at the model files on the server
# (we use the same hostPath /mnt/nvme on the bench-runner since it landed on the B300)
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/nvme/models/DeepSeek-V4-Flash}"

run_vbench() {
  local label="$1"; shift
  local outfile="${RESULT_DIR}/${label}.json"
  echo
  echo ">>> [${label}] $@"
  vllm bench serve \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER_PATH" \
    --base-url "$VLLM_URL" \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "${label}.json" \
    "$@" 2>&1 | tee "${RESULT_DIR}/${label}.log" | tail -40 || echo "[fail] $label (exit $?)"
}

# T0: random QPS sweep at 2K input / 512 output
echo
echo "=== T0: P1v-a QPS sweep (random 2K/512) ==="
for QPS in 1.0 2.0 4.0 8.0; do
  run_vbench "p1va_qps${QPS}" \
    --dataset-name random --random-input-len 2048 --random-output-len 512 \
    --num-prompts 80 --request-rate "$QPS"
  sleep 30
done

# T0: context scaling at QPS=1
echo
echo "=== T0: P1v-b context scaling (random ctx, fixed 512 out) ==="
for CTX in 1024 4096 16384; do
  run_vbench "p1vb_ctx${CTX}" \
    --dataset-name random --random-input-len "$CTX" --random-output-len 512 \
    --num-prompts 50 --request-rate 1.0
  sleep 30
done

# T0: 32K context (within max_model_len=32768; uses small num-prompts to fit KV)
run_vbench "p1vb_ctx32K" \
  --dataset-name random --random-input-len 30000 --random-output-len 512 \
  --num-prompts 20 --request-rate 0.5
sleep 30

# T1 candidate: shared-prefix (prefix caching benefit)
echo
echo "=== T1: P1v-c prefix-sharing test (generated-shared-prefix) ==="
# Note: V4 Flash startup config has --no-enable-prefix-caching; this measures cold-cache TTFT only.
# To validate the prefix-cache path requires a server restart with --enable-prefix-caching.
run_vbench "p1vc_shared_prefix_8k_cold" \
  --dataset-name random --random-input-len 8192 --random-output-len 256 \
  --num-prompts 30 --request-rate 1.0 \
  --random-prefix-len 7000   # NOTE: not all vLLM versions support this; falls back to random if unknown

# Post-run metrics
echo
echo "=== Post-run metrics ==="
capture_metrics post "$VLLM_URL"
capture_kv_metrics post "$VLLM_URL"

echo
echo "=== T0 sweep complete — results in $RESULT_DIR ==="
ls -la "$RESULT_DIR"
echo
echo "=== Summary ==="
for f in "$RESULT_DIR"/*.json; do
  [ -f "$f" ] || continue
  echo
  echo "--- $(basename "$f") ---"
  python3 -c "
import json,sys
try:
  d=json.load(open('$f'))
  for k in ['request_throughput','output_throughput','mean_ttft_ms','median_ttft_ms','p99_ttft_ms','mean_tpot_ms','median_tpot_ms','p99_tpot_ms','mean_itl_ms','p99_itl_ms']:
    if k in d: print(f'  {k}: {d[k]:.1f}')
except Exception as e:
  print('  (parse error:', e, ')')
"
done
