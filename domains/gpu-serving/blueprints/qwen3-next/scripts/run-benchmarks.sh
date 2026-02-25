#!/usr/bin/env bash
# Benchmark runner for Qwen3-Next on p5en.48xlarge (8x H200).
# Uses `vllm bench serve` (built into vLLM >= 0.15).
#
# Phases are grouped by framework to minimize server restarts (~390s each).
#
# Usage:
#   ./scripts/run-benchmarks.sh [phase]
#
# Phases (framework-grouped):
#   vllm      — All vLLM configs: tp4 baseline, tp4+MTP, dp8-ep
#   sglang    — All SGLang configs: tp8 baseline, tp8+MTP, tp4
#   winner    — Winner-only: context scaling (P1b), QPS sweep (P1c), CPU offload (P2)
#   longctx   — P2b: CPU offload + extended context (126K–252K) benchmarks
#   all       — Run vllm, sglang, winner, longctx sequentially
#
# Mapping to spec tiers:
#   vllm    → P0a + P1a-1 (tp4 no-MTP) + P1a-2 (tp4+MTP) + P1d (dp8-ep)
#   sglang  → P0b (tp8) + P1a-3 (tp8 no-MTP) + P1a-4 (tp8+MTP) + P0c (tp4)
#   winner  → P1b (context scaling) + P1c (QPS sweep) + P2 (CPU offload)
#   longctx → P2b (CPU offload + extended context: 126K–252K)
#
# Prerequisites:
#   - vLLM or SGLang serving on the GPU node (use configs/*.sh to launch)
#   - vllm bench serve available in PATH (pip install vllm, or exec into container)
#   - Results written to RESULT_DIR (default: ./results/session-$(date))
#
# Environment:
#   VLLM_URL      — vLLM base URL (default: http://localhost:8000)
#   SGLANG_URL    — SGLang base URL (default: http://localhost:30000)
#   WINNER_URL    — Winner engine URL for phase 3 (default: VLLM_URL)
#   MODEL_NAME    — Served model name (default: qwen3-next)
#   MODEL_PATH    — Model path for bench tokenizer (default: /mnt/nvme/models/qwen3-next-fp8)
#   RESULT_DIR    — Output directory (default: ./results/session-YYYYMMDD-HHMM)
#   RUNS          — Runs per config (default: 5)
#   WARMUP        — Warmup requests (default: 30)
#   COOLDOWN      — Seconds between configs (default: 60)
#   NUM_PROMPTS   — Requests per run (default: 100)
#   SLO_MAX_QPS   — SLO-max QPS from P1c (default: 3.0, used by dp8-ep and P2)
#   DRY_RUN       — Set to 1 to print commands without executing

set -euo pipefail

# --- Configuration ---
PHASE="${1:-vllm}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
WINNER_URL="${WINNER_URL:-$VLLM_URL}"
MODEL_NAME="${MODEL_NAME:-qwen3-next}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"
RESULT_DIR="${RESULT_DIR:-./results/session-$(date +%Y%m%d-%H%M)}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-30}"
COOLDOWN="${COOLDOWN:-60}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
SLO_MAX_QPS="${SLO_MAX_QPS:-3.0}"
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
    --metric-percentiles 50,90,99
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
    generated-shared-prefix)
      cmd+=(--dataset-name generated-shared-prefix
            --gsp-system-prompt-len "$input_len"
            --gsp-question-len 128
            --gsp-output-len "$output_len")
      ;;
    prefix_repetition)
      cmd+=(--dataset-name prefix_repetition
            --prefix-repetition-prefix-len "$input_len"
            --prefix-repetition-suffix-len 128
            --prefix-repetition-num-prefixes 2
            --prefix-repetition-output-len "$output_len")
      ;;
  esac

  # Extra args (e.g. --max-tokens)
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
    # Brief pause between runs
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

# --- Phase Functions ---

run_vllm() {
  log "============================================================"
  log "  PHASE: vLLM (all configs)"
  log "  Configs: tp4 baseline → tp4+MTP → dp8-ep"
  log "  Restarts: 2 (MTP, dp8-ep)"
  log "============================================================"
  log ""

  # --- Config 1: vLLM tp4 baseline (no restart needed) ---
  # Covers: P0a smoke test + P1a-1 no-MTP QPS sweep
  log "--- vLLM tp4 baseline (P0a + P1a-1) ---"
  log "NOTE: Start vLLM with configs/vllm-baseline.sh"
  wait_healthy "$VLLM_URL"

  # P0a: smoke test at low QPS
  run_bench "p0a_vllm_tp4_32k_qps0.5" "$VLLM_URL" random 1024 512 0.5
  capture_metrics "$VLLM_URL" "p0a_vllm_tp4"

  # P1a-1: no-MTP at multiple QPS (same server, no restart)
  for qps in 0.5 2.0; do
    run_bench "p1a1_vllm_tp4_noMTP_32k_qps${qps}" "$VLLM_URL" random 1024 512 "$qps"
  done
  capture_metrics "$VLLM_URL" "p1a1_vllm_tp4_noMTP"
  cooldown

  # --- Config 2: vLLM tp4+MTP (restart required) ---
  # Covers: P1a-2 MTP QPS sweep
  log "--- vLLM tp4+MTP (P1a-2) ---"
  prompt_restart "vLLM tp4+MTP (configs/vllm-tp4-mtp.sh)"
  if curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; then
    wait_healthy "$VLLM_URL"
    for qps in 0.5 2.0; do
      run_bench "p1a2_vllm_tp4_MTP_32k_qps${qps}" "$VLLM_URL" random 1024 512 "$qps"
    done
    capture_metrics "$VLLM_URL" "p1a2_vllm_tp4_MTP"
    cooldown
  else
    log "SKIP: vLLM tp4+MTP not running"
  fi

  # --- Config 3: vLLM dp8-ep (restart required) ---
  # Covers: P1d DP+EP throughput mode
  log "--- vLLM dp8-ep (P1d) ---"
  prompt_restart "vLLM dp8-ep (configs/vllm-dp8-ep.sh)"
  if curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; then
    wait_healthy "$VLLM_URL"
    local slo_2x
    slo_2x=$(echo "$SLO_MAX_QPS * 2" | bc)
    run_bench "p1d_dp8ep_32k_qps${SLO_MAX_QPS}" "$VLLM_URL" random 1024 512 "$SLO_MAX_QPS"
    sleep 30
    run_bench "p1d_dp8ep_32k_qps${slo_2x}" "$VLLM_URL" random 1024 512 "$slo_2x"
    capture_metrics "$VLLM_URL" "p1d_dp8ep_final"
    cooldown
  else
    log "SKIP: vLLM dp8-ep not running"
  fi

  log ""
  log "========== vLLM phase complete =========="
  log "Results: $RESULT_DIR/p0a_*, $RESULT_DIR/p1a1_*, $RESULT_DIR/p1a2_*, $RESULT_DIR/p1d_*"
}

run_sglang() {
  log "============================================================"
  log "  PHASE: SGLang (all configs)"
  log "  Configs: tp8 baseline → tp8+MTP → tp4"
  log "  Restarts: 2 (MTP, tp4)"
  log "============================================================"
  log ""

  # --- Config 1: SGLang tp8 baseline (framework switch) ---
  # Covers: P0b smoke test + P1a-3 no-MTP QPS sweep
  log "--- SGLang tp8 baseline (P0b + P1a-3) ---"
  log "NOTE: Start SGLang with configs/sglang-baseline.sh"
  if ! curl -sf "${SGLANG_URL}/health" > /dev/null 2>&1; then
    prompt_restart "SGLang tp8 baseline (configs/sglang-baseline.sh)"
  fi
  if curl -sf "${SGLANG_URL}/health" > /dev/null 2>&1; then
    wait_healthy "$SGLANG_URL"

    # P0b: smoke test
    run_bench "p0b_sglang_tp8_32k_qps0.5" "$SGLANG_URL" random 1024 512 0.5
    capture_metrics "$SGLANG_URL" "p0b_sglang_tp8"

    # P1a-3: no-MTP at multiple QPS (same server, no restart)
    for qps in 0.5 2.0; do
      run_bench "p1a3_sglang_tp8_noMTP_32k_qps${qps}" "$SGLANG_URL" random 1024 512 "$qps"
    done
    capture_metrics "$SGLANG_URL" "p1a3_sglang_tp8_noMTP"
    cooldown
  else
    log "SKIP: SGLang tp8 not running at $SGLANG_URL"
  fi

  # --- Config 2: SGLang tp8+MTP (restart required) ---
  # Covers: P1a-4 MTP QPS sweep
  log "--- SGLang tp8+MTP (P1a-4) ---"
  prompt_restart "SGLang tp8+MTP (configs/sglang-tp8-mtp.sh)"
  if curl -sf "${SGLANG_URL}/health" > /dev/null 2>&1; then
    wait_healthy "$SGLANG_URL"
    for qps in 0.5 2.0; do
      run_bench "p1a4_sglang_tp8_MTP_32k_qps${qps}" "$SGLANG_URL" random 1024 512 "$qps"
    done
    capture_metrics "$SGLANG_URL" "p1a4_sglang_tp8_MTP"
    cooldown
  else
    log "SKIP: SGLang tp8+MTP not running"
  fi

  # --- Config 3: SGLang tp4 (restart required) ---
  # Covers: P0c tp4 comparison
  log "--- SGLang tp4 (P0c) ---"
  prompt_restart "SGLang tp4 (configs/sglang-tp4.sh)"
  if curl -sf "${SGLANG_URL}/health" > /dev/null 2>&1; then
    wait_healthy "$SGLANG_URL"
    run_bench "p0c_sglang_tp4_32k_qps0.5" "$SGLANG_URL" random 1024 512 0.5
    capture_metrics "$SGLANG_URL" "p0c_sglang_tp4"
    cooldown
  else
    log "SKIP: SGLang tp4 not running"
  fi

  log ""
  log "========== SGLang phase complete =========="
  log "Results: $RESULT_DIR/p0b_*, $RESULT_DIR/p1a3_*, $RESULT_DIR/p1a4_*, $RESULT_DIR/p0c_*"
}

run_winner() {
  log "============================================================"
  log "  PHASE: Winner-only (context, QPS sweep, CPU offload)"
  log "  Uses WINNER_URL=$WINNER_URL"
  log "  Configs: best config → cpu-light"
  log "  Restarts: 1 (cpu-light)"
  log "============================================================"
  log ""

  # --- P1b: Context Scaling (no restart — use winning config already running) ---
  log "--- P1b: Context Scaling ---"
  log "NOTE: Ensure winning engine+config is running at $WINNER_URL"
  wait_healthy "$WINNER_URL"

  # 1b-1: Synthetic at 32K, 64K, 128K
  for ctx in 1024,512,32k 4096,512,64k 8192,512,128k; do
    IFS=',' read -r in_len out_len ctx_label <<< "$ctx"
    run_bench "p1b1_synthetic_${ctx_label}_qps0.5" "$WINNER_URL" random "$in_len" "$out_len" 0.5
  done
  cooldown

  # 1b-2: RAG workload (long input, short output)
  for ctx in 4096,256,32k 16384,256,64k 65536,256,128k; do
    IFS=',' read -r in_len out_len ctx_label <<< "$ctx"
    run_bench "p1b2_rag_${ctx_label}_qps0.5" "$WINNER_URL" random "$in_len" "$out_len" 0.5
  done
  cooldown

  # 1b-3: Prefix-sharing workload
  for qps in 0.5 2.0; do
    run_bench "p1b3_prefix_32k_qps${qps}" "$WINNER_URL" generated-shared-prefix 4096 256 "$qps"
  done
  capture_metrics "$WINNER_URL" "p1b_final"
  cooldown

  # --- P1c: QPS Sweep (same server, no restart) ---
  log "--- P1c: QPS Sweep / Pareto Frontier ---"

  # 1c-1: Synthetic at 32K, sweep QPS
  for qps in 0.5 1.0 2.0 3.0 5.0 8.0; do
    run_bench "p1c1_synthetic_32k_qps${qps}" "$WINNER_URL" random 1024 512 "$qps"
    sleep 30
  done
  capture_metrics "$WINNER_URL" "p1c1_final"
  cooldown

  # 1c-2: Agentic workload (512 in / 512 out)
  for qps in 0.5 2.0 4.0; do
    run_bench "p1c2_agentic_32k_qps${qps}" "$WINNER_URL" random 512 512 "$qps"
    sleep 30
  done
  capture_metrics "$WINNER_URL" "p1c2_final"
  cooldown

  log "P1b + P1c complete."
  log "Find SLO-max QPS: highest QPS where TTFT p99 < 300ms."
  log ""

  # --- P2: CPU KV Offload (restart required) ---
  log "--- P2: CPU KV Offload ---"
  log "NOTE: vLLM only. Add --cpu-offload-gb 10 --gpu-memory-utilization 0.88"
  prompt_restart "vLLM cpu-light config"
  if curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; then
    wait_healthy "$VLLM_URL"
    local slo_15x
    slo_15x=$(echo "$SLO_MAX_QPS * 1.5" | bc)

    for ctx in 1024,512,32k 8192,512,128k; do
      IFS=',' read -r in_len out_len ctx_label <<< "$ctx"
      for qps in "$SLO_MAX_QPS" "$slo_15x"; do
        run_bench "p2a_cpulight_${ctx_label}_qps${qps}" "$VLLM_URL" random "$in_len" "$out_len" "$qps"
        sleep 30
      done
    done
    capture_metrics "$VLLM_URL" "p2_cpulight_final"
  else
    log "SKIP: vLLM cpu-light not running"
  fi

  log ""
  log "========== Winner phase complete =========="
  log "Results: $RESULT_DIR/p1b*, $RESULT_DIR/p1c*, $RESULT_DIR/p2*"
}

run_longctx() {
  log "============================================================"
  log "  PHASE: Long Context Extended (P2b)"
  log "  Config: vllm-tp4-cpuoffload.sh (262K ctx, no cpu-offload)"
  log "  Note: --cpu-offload-gb blocked on vLLM V1 engine (vllm#18298)"
  log "  Uses VLLM_URL=$VLLM_URL"
  log "  No restarts within phase"
  log "============================================================"
  log ""

  log "NOTE: Start vLLM with configs/vllm-tp4-cpuoffload.sh"
  wait_healthy "$VLLM_URL"

  # P2b-1: 126K context, QPS 0.5 — baseline comparison with P1b result
  run_bench "p2b1_extctx_126k_qps0.5" "$VLLM_URL" random 65536 256 0.5
  cooldown

  # P2b-2: 252K context, QPS 0.5 — full native context
  run_bench "p2b2_extctx_252k_qps0.5" "$VLLM_URL" random 131072 256 0.5
  cooldown

  # P2b-3: 126K context, QPS 2.0 — concurrency stress with long context
  run_bench "p2b3_extctx_126k_qps2.0" "$VLLM_URL" random 65536 256 2.0
  cooldown

  # P2b-4: 252K context, QPS 0.5 with prefix sharing
  # 200K prefix + 128 suffix + 256 output — test prefix cache at extreme context
  run_bench "p2b4_extctx_252k_prefix_qps0.5" "$VLLM_URL" prefix_repetition 200000 256 0.5
  capture_metrics "$VLLM_URL" "p2b_final"

  log ""
  log "========== Long Context phase complete =========="
  log "Results: $RESULT_DIR/p2b*"
}

# --- Main ---

log "Qwen3-Next Benchmark Runner"
log "Phase: $PHASE | Runs: $RUNS | Warmup: $WARMUP | Cooldown: ${COOLDOWN}s"
log "Results: $RESULT_DIR"
log ""

case "$PHASE" in
  vllm)    run_vllm ;;
  sglang)  run_sglang ;;
  winner)  run_winner ;;
  longctx) run_longctx ;;
  all)
    run_vllm
    run_sglang
    run_winner
    run_longctx
    ;;
  *)
    echo "Usage: $0 {vllm|sglang|winner|longctx|all}"
    echo ""
    echo "Phases:"
    echo "  vllm    — All vLLM configs (tp4 baseline, tp4+MTP, dp8-ep)"
    echo "  sglang  — All SGLang configs (tp8 baseline, tp8+MTP, tp4)"
    echo "  winner  — Winner engine: context scaling, QPS sweep, CPU offload"
    echo "  longctx — P2b: CPU offload + extended context (126K–252K)"
    echo "  all     — Run all phases sequentially"
    exit 1
    ;;
esac

log ""
log "All benchmarks for phase '$PHASE' complete."
log "Results saved to: $RESULT_DIR"
