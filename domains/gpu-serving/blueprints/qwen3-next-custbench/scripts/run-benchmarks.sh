#!/usr/bin/env bash
# Benchmark runner for Qwen3-Next customer config comparison on p5en.48xlarge
# Tests customer's exact config (A) vs optimized (B) vs optimized-no-MTP (C)
#
# Usage:
#   ./scripts/run-benchmarks.sh [phase]
#
# Phases:
#   t1        — T1: Customer reproduction (Config A, 1000 concurrent)
#   t2        — T2: Optimized head-to-head (Config B, same workload)
#   t3        — T3: MTP isolation (Config C vs B)
#   t4        — T4: Load scaling (Config B at 10, 100, 1000 concurrent)
#   all       — Run all phases sequentially (requires server restarts between)
#
# Prerequisites:
#   - Build customer image: docker build -f docker/Dockerfile.vllm-customer -t qwen3-next-custbench:latest .
#   - Model staged to /mnt/nvme/models/qwen3-next-fp8
#   - vllm bench serve available in PATH
#
# Environment:
#   VLLM_URL      — Base URL (default: http://localhost:8000)
#   MODEL_NAME    — Served model name (default: qwen3-next)
#   MODEL_PATH    — Model path for tokenizer (default: /mnt/nvme/models/qwen3-next-fp8)
#   RESULT_DIR    — Output directory (default: ./results/session-YYYYMMDD-HHMM)
#   RUNS          — Runs per config (default: 3)
#   WARMUP        — Warmup requests (default: 30)
#   COOLDOWN      — Seconds between benchmarks (default: 60)
#   DRY_RUN       — Set to 1 to print commands without executing

set -euo pipefail

# --- Configuration ---
PHASE="${1:-t1}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL_NAME="${MODEL_NAME:-qwen3-next}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"
RESULT_DIR="${RESULT_DIR:-./results/session-$(date +%Y%m%d-%H%M)}"
RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-30}"
COOLDOWN="${COOLDOWN:-60}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$RESULT_DIR"

# --- Helpers ---

log() { echo "$(date +%H:%M:%S) | $*"; }

cooldown() {
  if [ "$COOLDOWN" -gt 0 ]; then
    log "Cooldown ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  fi
}

prompt_restart() {
  local config_name="$1"
  log ""
  log ">>> RESTART REQUIRED: $config_name"
  log "    Press Enter when server is healthy, or Ctrl+C to skip."
  if [ "$DRY_RUN" != "1" ]; then
    read -r -p "Continue? " || true
  fi
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

# Run a single benchmark invocation.
# Args: label num_prompts input_len output_len qps
run_bench() {
  local label="$1" num_prompts="$2" input_len="$3" output_len="$4" qps="$5"

  local out_dir="${RESULT_DIR}/${label}"
  mkdir -p "$out_dir"

  local cmd=(
    vllm bench serve
    --model "$MODEL_NAME"
    --tokenizer "$MODEL_PATH"
    --backend openai-chat
    --base-url "$VLLM_URL"
    --endpoint /v1/chat/completions
    --num-prompts "$num_prompts"
    --num-warmups "$WARMUP"
    --request-rate "$qps"
    --metric-percentiles 25,50,75,90,95,99
    --percentile-metrics ttft,tpot,itl,e2el
    --temperature 0.7
    --top-p 0.8
    --top-k 20
    --dataset-name random
    --random-input-len "$input_len"
    --random-output-len "$output_len"
    --save-result
    --save-detailed
    --result-dir "$out_dir"
  )

  log ">>> [$label] prompts=$num_prompts in=$input_len out=$output_len qps=$qps"
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

capture_metrics() {
  local label="$1"
  curl -sf "${VLLM_URL}/metrics" > "${RESULT_DIR}/${label}_metrics.txt" 2>/dev/null || true
}

# --- Phase Functions ---

run_t1() {
  log "============================================================"
  log "  T1: Customer Reproduction (Config A)"
  log "  Config: configs/vllm-customer-baseline.sh"
  log "  Workload: 1000 requests, 10K input, 1K output"
  log "============================================================"
  log ""
  log "NOTE: Start server with configs/vllm-customer-baseline.sh"
  wait_healthy "$VLLM_URL"

  # Customer's exact workload: 1000 concurrent, 10K input, 1K output
  # QPS inf (all at once) to match concurrent=1000 behavior
  run_bench "t1_customer_1000c_10k_1k" 1000 10000 1000 inf
  capture_metrics "t1_customer"
  cooldown

  log "========== T1 complete =========="
}

run_t2() {
  log "============================================================"
  log "  T2: Optimized Head-to-Head (Config B)"
  log "  Config: configs/vllm-optimized.sh"
  log "  Workload: same as T1 (1000 requests, 10K input, 1K output)"
  log "============================================================"
  log ""
  log "NOTE: Start server with configs/vllm-optimized.sh"
  prompt_restart "vLLM optimized (configs/vllm-optimized.sh)"
  wait_healthy "$VLLM_URL"

  # Same workload as customer
  run_bench "t2_optimized_1000c_10k_1k" 1000 10000 1000 inf
  capture_metrics "t2_optimized"
  cooldown

  log "========== T2 complete =========="
}

run_t3() {
  log "============================================================"
  log "  T3: MTP Isolation (Config C — optimized without MTP)"
  log "  Config: configs/vllm-optimized-nomtp.sh"
  log "  Workload: same as T1"
  log "============================================================"
  log ""
  log "NOTE: Start server with configs/vllm-optimized-nomtp.sh"
  prompt_restart "vLLM optimized no-MTP (configs/vllm-optimized-nomtp.sh)"
  wait_healthy "$VLLM_URL"

  run_bench "t3_opt_nomtp_1000c_10k_1k" 1000 10000 1000 inf
  capture_metrics "t3_opt_nomtp"
  cooldown

  log "========== T3 complete =========="
}

run_t4() {
  log "============================================================"
  log "  T4: Load Scaling (Config B at varying concurrency)"
  log "  Config: configs/vllm-optimized.sh"
  log "  Concurrency: 10, 100, 1000"
  log "============================================================"
  log ""
  log "NOTE: Ensure optimized config is running (configs/vllm-optimized.sh)"
  wait_healthy "$VLLM_URL"

  # Low load — latency floor
  run_bench "t4_opt_10c_10k_1k" 100 10000 1000 0.5
  cooldown

  # Moderate load — realistic production
  run_bench "t4_opt_100c_10k_1k" 100 10000 1000 5.0
  cooldown

  # High load — customer's concurrency level
  run_bench "t4_opt_1000c_10k_1k" 1000 10000 1000 inf
  capture_metrics "t4_scaling"
  cooldown

  log "========== T4 complete =========="
}

# --- Main ---

log "Qwen3-Next Customer Benchmark Runner"
log "Phase: $PHASE | Runs: $RUNS | Warmup: $WARMUP | Cooldown: ${COOLDOWN}s"
log "Results: $RESULT_DIR"
log ""

case "$PHASE" in
  t1)  run_t1 ;;
  t2)  run_t2 ;;
  t3)  run_t3 ;;
  t4)  run_t4 ;;
  all)
    run_t1
    run_t2
    run_t3
    run_t4
    ;;
  *)
    echo "Usage: $0 {t1|t2|t3|t4|all}"
    echo ""
    echo "Phases:"
    echo "  t1  — Customer reproduction (Config A, 1000 concurrent)"
    echo "  t2  — Optimized head-to-head (Config B, same workload)"
    echo "  t3  — MTP isolation (Config C vs B)"
    echo "  t4  — Load scaling (Config B at 10, 100, 1000 concurrent)"
    echo "  all — Run all phases sequentially"
    exit 1
    ;;
esac

log ""
log "All benchmarks for phase '$PHASE' complete."
log "Results saved to: $RESULT_DIR"
