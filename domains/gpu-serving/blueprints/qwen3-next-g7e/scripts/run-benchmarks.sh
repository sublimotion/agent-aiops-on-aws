#!/usr/bin/env bash
# Benchmark runner for Qwen3-Next on g7e.48xlarge (8x RTX PRO Server 6000).
# Includes both g7e-native phases (G0-G2) and customer reproduction phases (C1-C4).
#
# Phases — G-series (g7e baseline, optimized config):
#   g0   — Smoke test: QPS 0.5, 1024/512 random (3 runs)
#   g1   — QPS sweep: 1, 2, 4, 8 at 1024/512 (3 runs each)
#   g2   — Context: 4K, 32K, 64K at QPS 0.5 (3 runs each)
#   gall — Run g0, g1, g2 sequentially
#
# Phases — C-series (customer config reproduction on g7e):
#   c1   — Customer reproduction: Config A (customer baseline), 1000 concurrent, 10K/1K
#   c2   — Optimized head-to-head: Config B (our optimized) vs C1, same workload
#   c3   — MTP isolation: Config B without MTP, same workload
#   c4   — Load scaling: Config B at 0.5, 5.0, inf QPS with 10K/1K
#   call — Run c1, c2, c3, c4 sequentially
#
#   all  — Run all phases (g0-g2 then c1-c4)
#
# Prerequisites:
#   - vLLM serving on the GPU node
#     G-series: configs/vllm-baseline.sh (optimized config)
#     C-series: configs/vllm-customer-baseline.sh (customer config for c1),
#               configs/vllm-baseline.sh (optimized for c2/c4),
#               restart without MTP for c3
#   - vllm bench serve available in PATH
#   - Results written to RESULT_DIR
#
# Environment:
#   VLLM_URL      — vLLM base URL (default: http://localhost:8000)
#   MODEL_NAME    — Served model name (default: qwen3-next)
#   MODEL_PATH    — Model path for bench tokenizer (default: /mnt/nvme/models/qwen3-next-fp8)
#   RESULT_DIR    — Output directory (default: ./results/session-YYYYMMDD-HHMM)
#   RUNS          — Runs per config (default: 3)
#   WARMUP        — Warmup requests (default: 30)
#   COOLDOWN      — Seconds between configs (default: 60)
#   NUM_PROMPTS   — Requests per run for G-series (default: 100)
#   DRY_RUN       — Set to 1 to print commands without executing

set -euo pipefail

# --- Configuration ---
PHASE="${1:-g0}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL_NAME="${MODEL_NAME:-qwen3-next}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"
RESULT_DIR="${RESULT_DIR:-./results/session-$(date +%Y%m%d-%H%M)}"
RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-30}"
COOLDOWN="${COOLDOWN:-60}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
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
  log "    Stop current server, start the new config, then press Enter."
  log "    Ctrl+C to abort."
  if [ "$DRY_RUN" != "1" ]; then
    read -r -p "Continue? " || true
  fi
}

# Run a single vllm bench serve invocation.
# Args: label base_url dataset_name input_len output_len qps [extra_args...]
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
    --metric-percentiles 25,50,75,90,95,99
    --percentile-metrics ttft,tpot,itl,e2el
    --temperature 0.7
    --top-p 0.8
    --top-k 20
    --save-result
    --save-detailed
    --result-dir "$out_dir"
  )

  # Dataset-specific args
  case "$dataset" in
    random)
      cmd+=(--dataset-name random --random-input-len "$input_len" --random-output-len "$output_len")
      ;;
  esac

  # Extra args
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

# Customer benchmark variant: takes explicit num_prompts instead of global NUM_PROMPTS.
# Args: label num_prompts input_len output_len qps
run_bench_custom() {
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

# Wait for server health
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

# Capture Prometheus metrics snapshot
capture_metrics() {
  local url="$1" label="$2"
  curl -sf "${url}/metrics" > "${RESULT_DIR}/${label}_metrics.txt" 2>/dev/null || true
}

# =============================================
# G-series: g7e baseline benchmarks
# =============================================

run_g0() {
  log "============================================================"
  log "  PHASE: G0 — Smoke Test"
  log "  Config: vLLM tp4 baseline (configs/vllm-baseline.sh)"
  log "  QPS 0.5, 1024 in / 512 out, $RUNS runs"
  log "============================================================"
  log ""

  wait_healthy "$VLLM_URL"

  run_bench "g0_smoke_1024_512_qps0.5" "$VLLM_URL" random 1024 512 0.5
  capture_metrics "$VLLM_URL" "g0_smoke"

  log ""
  log "========== G0 Smoke Test complete =========="
  log "Results: $RESULT_DIR/g0_*"
}

run_g1() {
  log "============================================================"
  log "  PHASE: G1 — QPS Sweep"
  log "  Config: vLLM tp4 baseline"
  log "  QPS 1, 2, 4, 8 at 1024/512, $RUNS runs each"
  log "============================================================"
  log ""

  wait_healthy "$VLLM_URL"

  for qps in 1.0 2.0 4.0 8.0; do
    run_bench "g1_qps${qps}_1024_512" "$VLLM_URL" random 1024 512 "$qps"
    sleep 30
  done
  capture_metrics "$VLLM_URL" "g1_final"
  cooldown

  log ""
  log "========== G1 QPS Sweep complete =========="
  log "Results: $RESULT_DIR/g1_*"
}

run_g2() {
  log "============================================================"
  log "  PHASE: G2 — Context Scaling"
  log "  Config: vLLM tp4 baseline"
  log "  Context 4K, 32K, 64K at QPS 0.5, $RUNS runs each"
  log "  NOTE: 64K may fail on g7e (96 GB VRAM). If so, reduce max_model_len."
  log "============================================================"
  log ""

  wait_healthy "$VLLM_URL"

  # 4K context: 4096 in / 256 out
  run_bench "g2_ctx4k_qps0.5" "$VLLM_URL" random 4096 256 0.5
  cooldown

  # 32K context: 16384 in / 512 out
  run_bench "g2_ctx32k_qps0.5" "$VLLM_URL" random 16384 512 0.5
  cooldown

  # 64K context: 32768 in / 512 out
  run_bench "g2_ctx64k_qps0.5" "$VLLM_URL" random 32768 512 0.5
  capture_metrics "$VLLM_URL" "g2_final"

  log ""
  log "========== G2 Context Scaling complete =========="
  log "Results: $RESULT_DIR/g2_*"
}

# =============================================
# C-series: customer config reproduction on g7e
# Adapted from custbench (p5en) benchmark runner.
# Same workloads, adapted for g7e constraints:
#   - 96 GB VRAM (vs 141 GB on p5en) — customer's max_model_len=32768 fits
#   - Customer config uses MTP (works on customer's vLLM nightly image)
#   - 10K input / 1K output at 1000 concurrent matches customer's reported workload
# =============================================

run_c1() {
  log "============================================================"
  log "  PHASE: C1 — Customer Reproduction (Config A) on g7e"
  log "  Config: configs/vllm-customer-baseline.sh"
  log "  Image:  qwen3-next-customer:latest (Dockerfile.vllm-customer)"
  log "  Workload: 1000 requests, 10K input, 1K output, QPS inf"
  log "  Customer reports on p5en: 3,612 tok/s, TTFT p50 940ms"
  log "============================================================"
  log ""
  log "NOTE: Start server with configs/vllm-customer-baseline.sh"
  wait_healthy "$VLLM_URL"

  # Customer's exact workload: 1000 prompts fired at once (QPS inf)
  run_bench_custom "c1_customer_1000c_10k_1k" 1000 10000 1000 inf
  capture_metrics "$VLLM_URL" "c1_customer"
  cooldown

  log "========== C1 Customer Reproduction complete =========="
}

run_c2() {
  log "============================================================"
  log "  PHASE: C2 — Optimized Head-to-Head (Config B) on g7e"
  log "  Config: configs/vllm-baseline.sh (g7e optimized)"
  log "  Workload: same as C1 (1000 requests, 10K input, 1K output)"
  log "============================================================"
  log ""
  prompt_restart "vLLM optimized (configs/vllm-baseline.sh)"
  wait_healthy "$VLLM_URL"

  run_bench_custom "c2_optimized_1000c_10k_1k" 1000 10000 1000 inf
  capture_metrics "$VLLM_URL" "c2_optimized"
  cooldown

  log "========== C2 Optimized Head-to-Head complete =========="
}

run_c3() {
  log "============================================================"
  log "  PHASE: C3 — MTP Isolation (optimized without MTP) on g7e"
  log "  Config: configs/vllm-baseline.sh without --speculative-config"
  log "  Workload: same as C1"
  log "  NOTE: MTP is blocked on vLLM 0.16+. This phase tests the"
  log "        optimized config without MTP to isolate its contribution"
  log "        vs the customer's nightly build (which supports MTP)."
  log "============================================================"
  log ""
  prompt_restart "vLLM optimized without MTP (remove --speculative-config from vllm-baseline.sh)"
  wait_healthy "$VLLM_URL"

  run_bench_custom "c3_opt_nomtp_1000c_10k_1k" 1000 10000 1000 inf
  capture_metrics "$VLLM_URL" "c3_opt_nomtp"
  cooldown

  log "========== C3 MTP Isolation complete =========="
}

run_c4() {
  log "============================================================"
  log "  PHASE: C4 — Load Scaling (Config B at varying rates) on g7e"
  log "  Config: configs/vllm-baseline.sh (g7e optimized)"
  log "  Rates: 0.5 QPS (low), 5.0 QPS (moderate), inf (high/customer)"
  log "============================================================"
  log ""
  log "NOTE: Ensure optimized config is running (configs/vllm-baseline.sh)"
  wait_healthy "$VLLM_URL"

  # Low load (0.5 QPS) — latency floor measurement
  run_bench_custom "c4_opt_low_qps0.5" 100 10000 1000 0.5
  cooldown

  # Moderate load (5.0 QPS) — realistic production
  run_bench_custom "c4_opt_moderate_qps5.0" 200 10000 1000 5.0
  cooldown

  # High load (inf) — all 1000 at once, matches customer workload
  run_bench_custom "c4_opt_high_inf" 1000 10000 1000 inf
  capture_metrics "$VLLM_URL" "c4_scaling"
  cooldown

  log "========== C4 Load Scaling complete =========="
}

# --- Main ---

log "Qwen3-Next g7e Benchmark Runner"
log "Instance: g7e.48xlarge (8x RTX PRO Server 6000, Blackwell GB202)"
log "Phase: $PHASE | Runs: $RUNS | Warmup: $WARMUP | Cooldown: ${COOLDOWN}s"
log "Results: $RESULT_DIR"
log ""

case "$PHASE" in
  g0)   run_g0 ;;
  g1)   run_g1 ;;
  g2)   run_g2 ;;
  gall)
    run_g0
    run_g1
    run_g2
    ;;
  c1)   run_c1 ;;
  c2)   run_c2 ;;
  c3)   run_c3 ;;
  c4)   run_c4 ;;
  call)
    run_c1
    run_c2
    run_c3
    run_c4
    ;;
  all)
    run_g0
    run_g1
    run_g2
    run_c1
    run_c2
    run_c3
    run_c4
    ;;
  *)
    echo "Usage: $0 {g0|g1|g2|gall|c1|c2|c3|c4|call|all}"
    echo ""
    echo "G-series (g7e baseline, optimized config):"
    echo "  g0   — Smoke test: QPS 0.5, 1024/512 random"
    echo "  g1   — QPS sweep: 1, 2, 4, 8 at 1024/512"
    echo "  g2   — Context: 4K, 32K, 64K at QPS 0.5"
    echo "  gall — Run g0, g1, g2 sequentially"
    echo ""
    echo "C-series (customer config reproduction on g7e):"
    echo "  c1   — Customer reproduction (Config A, 1000 concurrent, 10K/1K)"
    echo "  c2   — Optimized head-to-head (Config B, same workload)"
    echo "  c3   — MTP isolation (Config B without MTP)"
    echo "  c4   — Load scaling (Config B at 0.5/5.0/inf QPS)"
    echo "  call — Run c1, c2, c3, c4 sequentially"
    echo ""
    echo "  all  — Run all phases (g0-g2, then c1-c4)"
    exit 1
    ;;
esac

log ""
log "All benchmarks for phase '$PHASE' complete."
log "Results saved to: $RESULT_DIR"
