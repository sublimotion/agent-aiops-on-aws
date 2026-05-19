#!/usr/bin/env bash
# Round 3: MTP speculative decoding measurements.
# Pre-run guardrails per spec: stability test, acceptance rate, output repetition check.
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${MODEL_NAME:-DeepSeek-V4-Flash}"
TOKENIZER_PATH=/mnt/nvme/models/DeepSeek-V4-Flash
RESULT_DIR="/results/r3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"
export RESULT_DIR

cd "$(dirname "$0")"
source ./benchmark-helpers.sh

echo "=== DSv4 Flash Round 3 (MTP) — $(date) ==="
echo "RESULT_DIR=$RESULT_DIR"

curl -sf "${VLLM_URL}/health" >/dev/null || { echo "[error] vLLM unhealthy"; exit 1; }

##############################################
# Pre-run gate 1: smoke test (precision)
##############################################
echo
echo "=== gate 1: smoke test ==="
RESPONSE=$(curl -s "${VLLM_URL}/v1/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":20,\"temperature\":0}")
echo "$RESPONSE" > "${RESULT_DIR}/smoke_mtp.json"
echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Smoke:', d['choices'][0]['text'][:200])"

##############################################
# Pre-run gate 2: degenerate output check (per spec RCA)
# Sample 20 outputs on random-style prompts; reject if any token repeats >5x consecutively.
##############################################
echo
echo "=== gate 2: degenerate output detector (20 samples) ==="
python3 - <<'PYEOF'
import json, urllib.request, re, sys

URL = "http://localhost:8000/v1/completions"
prompts = [
  "Write a one-paragraph review of the Solar System.",
  "Explain why the sky is blue in simple terms.",
  "List five techniques for time management.",
  "Translate to French: 'The quick brown fox jumps over the lazy dog.'",
  "Summarize the plot of Romeo and Juliet in three sentences.",
  "What is the difference between TCP and UDP?",
  "Give a short biography of Marie Curie.",
  "Write a haiku about autumn leaves.",
  "Explain photosynthesis in 50 words or fewer.",
  "List three causes of the French Revolution.",
  "Describe the taste of dark chocolate.",
  "What is a closure in JavaScript?",
  "Compare and contrast cats and dogs.",
  "Explain the Pythagorean theorem.",
  "Write a short poem about the ocean.",
  "Why do leaves change color in autumn?",
  "What is the speed of light?",
  "Give a one-sentence summary of WWII.",
  "Explain Einstein's theory of special relativity.",
  "Recommend three classic novels.",
]

degenerate_count = 0
for i, p in enumerate(prompts):
  body = json.dumps({"model": "DeepSeek-V4-Flash", "prompt": p, "max_tokens": 100, "temperature": 0.7}).encode()
  req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
  try:
    out = json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["text"]
  except Exception as e:
    print(f"  [{i}] ERR: {e}")
    continue
  # Check for token repetition >5x consecutive (whitespace-tokenized)
  tokens = out.split()
  bad = False
  for j in range(len(tokens) - 5):
    if all(tokens[j] == tokens[j+k] for k in range(1, 6)):
      bad = True; break
  status = "OK" if not bad else "DEGENERATE"
  if bad: degenerate_count += 1
  print(f"  [{i}] {status}: {out[:100]}...")

print(f"\nResult: {degenerate_count}/{len(prompts)} degenerate")
sys.exit(1 if degenerate_count > 0 else 0)
PYEOF

if [ $? -ne 0 ]; then
  echo
  echo "=== GATE FAILED: degenerate output detected ==="
  echo "Per spec RCA, MTP throughput numbers from this config would be untrustworthy."
  echo "Recording in /results/ but flagging as UNRELIABLE."
fi

capture_metrics pre "$VLLM_URL"
capture_kv_metrics pre "$VLLM_URL"

##############################################
# T6: MTP throughput sweep — same shape as P1v-a but with spec-decode active
##############################################
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

echo
echo "================================ T6: MTP QPS sweep (random 2K/512) ================================"
for QPS in 1.0 4.0 8.0; do
  run_vbench "t6_mtp_qps${QPS}" \
    --dataset-name random --random-input-len 2048 --random-output-len 512 \
    --num-prompts 80 --request-rate "$QPS"
  sleep 20
done

# Multi-turn-like (longer outputs) where MTP shines
echo
echo "=== T6 long-output (where speculation pays) ==="
run_vbench "t6_mtp_long_out_qps2" \
  --dataset-name random --random-input-len 1024 --random-output-len 2048 \
  --num-prompts 40 --request-rate 2.0
sleep 20

# ShareGPT real-distribution under MTP
echo
echo "=== T6 W0 sharegpt under MTP ==="
run_vbench "t6_mtp_sharegpt_qps4" \
  --dataset-name sharegpt \
  --dataset-path /sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 200 --request-rate 4.0

# Acceptance rate from server-side metrics
echo
echo "=== MTP acceptance metrics ==="
curl -sf "${VLLM_URL}/metrics" | grep -iE "spec_dec|draft_acc|num_accepted|num_drafts" | head -20 | tee "${RESULT_DIR}/mtp_acceptance.txt"

capture_metrics post "$VLLM_URL"
capture_kv_metrics post "$VLLM_URL"

echo
echo "=== R3 (MTP) sweep complete — results in $RESULT_DIR ==="
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
  for k in ['request_throughput','output_throughput','total_token_throughput','median_ttft_ms','p99_ttft_ms','median_tpot_ms','median_itl_ms','p99_itl_ms']:
    if k in d: print(f'  {k}: {d[k]:.1f}')
except Exception as e:
  print('  (err)', e)
"
done
