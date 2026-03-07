#!/usr/bin/env bash
# Benchmark runner for customer MTP config vs baseline on g7e.48xlarge.
# Uses the customer's production vLLM args:
#   - MTP speculative decoding (2 tokens, qwen3_next_mtp method)
#   - FP8 KV cache
#   - max-model-len 32768 (32K)
#   - max-num-seqs 512
#   - Torch compilation passes (fi_allreduce_fusion, noop)
#   - chunked prefill disabled (required for MTP)
#
# Phases:
#   c0   — Smoke test: QPS 0.5, 1024/512 random
#   c1   — QPS sweep: 1, 2, 4, 8, 16 at 1024/512
#   c2   — High concurrency: QPS 4, 8 at 1024/512 (max-num-seqs stress)
#   c3   — Context: 4K, 16K, 32K at QPS 2 (capped at customer's 32K limit)
#   c4   — Customer-comparable: 1000 concurrent, 10K in / 1K out (matches dashboard)
#   all  — Run c0, c1, c2, c3, c4 sequentially
#
# Usage:
#   # Launch customer MTP config first:
#   ./configs/vllm-customer-mtp.sh
#   # Then run benchmarks:
#   ./scripts/run-benchmarks-customer.sh all
#
# Environment:
#   VLLM_URL      — vLLM base URL (default: http://localhost:8000)
#   MODEL_NAME    — Served model name (default: qwen3-next)
#   MODEL_PATH    — Model path for bench tokenizer (default: /mnt/nvme/models/qwen3-next-fp8)
#   RESULT_DIR    — Output directory (default: ./results/customer-session-YYYYMMDD-HHMM)
#   RUNS          — Runs per config (default: 3)
#   WARMUP        — Warmup requests (default: 30)
#   COOLDOWN      — Seconds between configs (default: 60)
#   NUM_PROMPTS   — Requests per run (default: 100)
#   DRY_RUN       — Set to 1 to print commands without executing

set -euo pipefail

# --- Configuration ---
PHASE="${1:-c0}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL_NAME="${MODEL_NAME:-qwen3-next}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"
RESULT_DIR="${RESULT_DIR:-./results/customer-session-$(date +%Y%m%d-%H%M)}"
RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-30}"
COOLDOWN="${COOLDOWN:-60}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$RESULT_DIR"

# --- Helpers (shared with run-benchmarks.sh) ---

log() { echo "$(date +%H:%M:%S) | $*"; }

cooldown() {
  if [ "$COOLDOWN" -gt 0 ]; then
    log "Cooldown ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  fi
}

run_bench() {
  local label="$1" base_url="$2" dataset="$3" input_len="$4" output_len="$5" qps="$6"
  shift 6
  local extra_args=("$@")

  local out_dir="${RESULT_DIR}/${label}"
  mkdir -p "$out_dir"

  local cmd=(
    vllm bench serve
    --model "$MODEL_NAME"
    --tokenizer "$MODEL_PATH"
    --backend openai-chat
    --base-url "$base_url"
    --endpoint /v1/chat/completions
    --num-prompts "$NUM_PROMPTS"
    --num-warmups "$WARMUP"
    --request-rate "$qps"
    --metric-percentiles 50,90,99
    --percentile-metrics ttft,tpot,itl,e2el
    --temperature 0.7
    --top-p 0.8
    --top-k 20
    --save-result
    --save-detailed
    --result-dir "$out_dir"
  )

  case "$dataset" in
    random)
      cmd+=(--dataset-name random --random-input-len "$input_len" --random-output-len "$output_len")
      ;;
  esac

  if [ ${#extra_args[@]} -gt 0 ]; then
    cmd+=("${extra_args[@]}")
  fi

  log ">>> [$label] dataset=$dataset in=$input_len out=$output_len qps=$qps"
  if [ "$DRY_RUN" = "1" ]; then
    echo "  DRY_RUN: ${cmd[*]}"
    return
  fi

  for run in $(seq 1 "$RUNS"); do
    log "  Run $run/$RUNS"
    "${cmd[@]}" 2>&1 | tee "${out_dir}/run_${run}.log"
    sleep 5
  done
}

wait_healthy() {
  local url="$1" max_wait="${2:-600}"
  log "Waiting for server at $url (max ${max_wait}s)..."
  local elapsed=0
  while [ $elapsed -lt "$max_wait" ]; do
    if curl -sf "${url}/health" > /dev/null 2>&1; then
      log "Server healthy at $url"
      return 0
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  log "ERROR: Server not healthy after ${max_wait}s"
  return 1
}

capture_metrics() {
  local url="$1" label="$2"
  curl -sf "${url}/metrics" > "${RESULT_DIR}/${label}_metrics.txt" 2>/dev/null || true
}

# --- Phase Functions ---

run_c0() {
  log "============================================================"
  log "  PHASE: C0 — Smoke Test (Customer MTP Config)"
  log "  Config: vLLM tp4 + MTP spec(2) + FP8 KV + 32K context"
  log "  QPS 0.5, 1024 in / 512 out, $RUNS runs"
  log "============================================================"
  log ""

  wait_healthy "$VLLM_URL"

  run_bench "c0_mtp_smoke_1024_512_qps0.5" "$VLLM_URL" random 1024 512 0.5
  capture_metrics "$VLLM_URL" "c0_smoke"

  log ""
  log "========== C0 Smoke Test complete =========="
}

run_c1() {
  log "============================================================"
  log "  PHASE: C1 — QPS Sweep (Customer MTP Config)"
  log "  QPS 1, 2, 4, 8, 16 at 1024/512, $RUNS runs each"
  log "  Note: QPS 16 tests MTP throughput under pressure"
  log "============================================================"
  log ""

  wait_healthy "$VLLM_URL"

  for qps in 1.0 2.0 4.0 8.0 16.0; do
    run_bench "c1_mtp_qps${qps}_1024_512" "$VLLM_URL" random 1024 512 "$qps"
    sleep 30
  done
  capture_metrics "$VLLM_URL" "c1_final"
  cooldown

  log ""
  log "========== C1 QPS Sweep complete =========="
}

run_c2() {
  log "============================================================"
  log "  PHASE: C2 — High Concurrency (Customer MTP Config)"
  log "  QPS 4, 8 at 1024/512 with max-num-seqs=512 stress"
  log "  Higher NUM_PROMPTS to stress concurrent batch sizes"
  log "============================================================"
  log ""

  wait_healthy "$VLLM_URL"

  # Use more prompts to test sustained high concurrency
  local saved_prompts="$NUM_PROMPTS"
  NUM_PROMPTS=200

  run_bench "c2_mtp_highconc_qps4_1024_512" "$VLLM_URL" random 1024 512 4.0
  cooldown

  run_bench "c2_mtp_highconc_qps8_1024_512" "$VLLM_URL" random 1024 512 8.0
  capture_metrics "$VLLM_URL" "c2_final"

  NUM_PROMPTS="$saved_prompts"
  cooldown

  log ""
  log "========== C2 High Concurrency complete =========="
}

run_c3() {
  log "============================================================"
  log "  PHASE: C3 — Context Scaling (Customer MTP Config)"
  log "  Context 4K, 16K, 32K at QPS 2, $RUNS runs each"
  log "  Note: Customer max-model-len=32768 caps context"
  log "============================================================"
  log ""

  wait_healthy "$VLLM_URL"

  # 4K context: 4096 in / 256 out
  run_bench "c3_mtp_ctx4k_qps2" "$VLLM_URL" random 4096 256 2.0
  cooldown

  # 16K context: 16384 in / 512 out
  run_bench "c3_mtp_ctx16k_qps2" "$VLLM_URL" random 16384 512 2.0
  cooldown

  # 32K context: 24576 in / 512 out (max within 32K model len budget)
  run_bench "c3_mtp_ctx32k_qps2" "$VLLM_URL" random 24576 512 2.0
  capture_metrics "$VLLM_URL" "c3_final"

  log ""
  log "========== C3 Context Scaling complete =========="
}

run_c4() {
  log "============================================================"
  log "  PHASE: C4 — Customer-Comparable Run"
  log "  Matches customer dashboard: 1000 concurrent, 10K/1K tokens"
  log "  Customer result: 3612.8 tok/s, TTFT 1.47s mean, 19.07s E2E"
  log "============================================================"
  log ""

  wait_healthy "$VLLM_URL"

  local saved_prompts="$NUM_PROMPTS"
  local saved_warmup="$WARMUP"
  NUM_PROMPTS=1000
  WARMUP=50

  # Exact match of customer benchmark:
  #   1000 concurrent requests, 10K input / 1K output, high request rate
  # Using QPS inf (all at once) to simulate 1000 concurrent
  run_bench "c4_customer_match_10k_1k_conc1000" "$VLLM_URL" random 10000 1000 inf
  capture_metrics "$VLLM_URL" "c4_customer_match"

  NUM_PROMPTS="$saved_prompts"
  WARMUP="$saved_warmup"

  log ""
  log "========== C4 Customer-Comparable Run complete =========="
  log "Compare against customer baseline:"
  log "  Throughput: 3612.8 tok/s"
  log "  Mean TTFT:  1.47s (P50: 0.94s, P95: 4.34s)"
  log "  Mean E2E:   19.07s (P95: 24.59s)"
  log "  Requests/min: 182.7"
}

# --- Main ---

log "Qwen3-Next g7e Benchmark Runner — Customer MTP Config"
log "Instance: g7e.48xlarge (8x RTX PRO Server 6000, Blackwell GB202)"
log "Config: MTP spec(2), FP8 KV, 32K ctx, max-num-seqs=512"
log "Phase: $PHASE | Runs: $RUNS | Warmup: $WARMUP | Cooldown: ${COOLDOWN}s"
log "Results: $RESULT_DIR"
log ""

case "$PHASE" in
  c0)  run_c0 ;;
  c1)  run_c1 ;;
  c2)  run_c2 ;;
  c3)  run_c3 ;;
  c4)  run_c4 ;;
  all)
    run_c0
    run_c1
    run_c2
    run_c3
    run_c4
    ;;
  *)
    echo "Usage: $0 {c0|c1|c2|c3|c4|all}"
    echo ""
    echo "Phases:"
    echo "  c0  — Smoke test: QPS 0.5, 1024/512 random"
    echo "  c1  — QPS sweep: 1, 2, 4, 8, 16 at 1024/512"
    echo "  c2  — High concurrency: QPS 4, 8 at 1024/512 (200 prompts)"
    echo "  c3  — Context: 4K, 16K, 32K at QPS 2"
    echo "  c4  — Customer-comparable: 1000 concurrent, 10K/1K tokens"
    echo "  all — Run all phases sequentially"
    exit 1
    ;;
esac

log ""
log "All benchmarks for phase '$PHASE' complete."
log "Results saved to: $RESULT_DIR"
