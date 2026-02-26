#!/usr/bin/env bash
# Benchmark runner for Qwen3-Next customer config comparison on p5en.48xlarge
# Tests customer's exact config (A) vs optimized (B/C/D) vs 2-replica (E)
#
# Usage:
#   ./scripts/run-benchmarks.sh [phase]
#
# Phases:
#   t1        — T1: Customer reproduction (Config A, 1000 concurrent)
#   t2        — T2: Optimized head-to-head (Config B, same workload)
#   t3        — T3: MTP isolation (Config C vs B)
#   t4        — T4: Load scaling (Config B at 0.5/5.0/inf qps)
#   t5        — T5: Simulated memory-constrained KV offload (gpu-mem=0.30)
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
# Args: label num_prompts input_len output_len qps [dataset]
# dataset: "random" (default) or "generated-shared-prefix"
run_bench() {
  local label="$1" num_prompts="$2" input_len="$3" output_len="$4" qps="$5"

  local out_dir="${RESULT_DIR}/${label}"
  mkdir -p "$out_dir"

  local dataset="${6:-random}"

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
      # Shared system prompt (input_len) + unique 128-token question per request
      cmd+=(--dataset-name generated-shared-prefix
            --gsp-system-prompt-len "$input_len"
            --gsp-question-len 128
            --gsp-output-len "$output_len")
      ;;
  esac

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
  local url="${2:-$VLLM_URL}"
  curl -sf "${url}/metrics" > "${RESULT_DIR}/${label}_metrics.txt" 2>/dev/null || true
}

# Capture KV cache specific metrics (prefix cache hits, GPU/CPU cache usage, offload stats)
# Captures both vLLM standard metrics and Dynamo KVBM per-tier metrics.
# Also snapshots FSx cache directory size for disk tier validation.
capture_kv_metrics() {
  local label="$1"
  local url="${2:-$VLLM_URL}"
  local out="${RESULT_DIR}/${label}_kv_metrics.txt"

  log "Capturing KV cache metrics for $label..."
  {
    echo "=== KV Cache Metrics: ${label} ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
    echo ""

    # 1. vLLM standard metrics (prefix cache, preemptions, GPU/CPU cache usage)
    echo "--- vLLM Metrics ---"
    curl -sf "${url}/metrics" 2>/dev/null | grep -iE \
      'prefix_cache|gpu_cache_usage|cpu_cache_usage|num_preemption|swap_space|running_requests|waiting_requests|cache_config' \
      || echo "No vLLM cache metrics found"

    echo ""

    # 2. Dynamo KVBM per-tier metrics (if running Dynamo)
    echo "--- Dynamo KVBM Tier Metrics ---"
    curl -sf "${url}/metrics" 2>/dev/null | grep -iE \
      'kvbm|device_pool|cpu_pool|disk_pool|tier.*hit|tier.*miss|offload|evict|nixl|transfer_bytes|transfer_latency' \
      || echo "No Dynamo KVBM metrics found (expected if not running Dynamo)"

    echo ""

    # 3. FSx disk cache size (validates disk tier is being used)
    echo "--- FSx Cache Directory ---"
    local fsx_cache="${FSX_EFA_CACHE_DIR:-/mnt/fsx-efa/kv-cache/qwen3-custbench}"
    if [ -d "$fsx_cache" ]; then
      du -sh "$fsx_cache" 2>/dev/null || echo "Cannot read $fsx_cache"
      ls -la "$fsx_cache" 2>/dev/null | head -20
    else
      echo "FSx cache dir not found at $fsx_cache"
    fi

    echo ""

    # 4. GPU memory snapshot (nvidia-smi, confirms actual VRAM usage)
    echo "--- GPU Memory ---"
    nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free \
      --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

    echo ""
    echo "=== End KV Cache Metrics ==="
  } > "$out"

  # Full metrics dump for post-hoc analysis
  curl -sf "${url}/metrics" > "${RESULT_DIR}/${label}_full_metrics.txt" 2>/dev/null || true
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

run_t2b() {
  log "============================================================"
  log "  T2b: Prefix Sharing — Optimized (Config B)"
  log "  Config: configs/vllm-optimized.sh (keep running from T2)"
  log "  Workload: 1000 requests, 8K shared prefix + 128 unique, 1K output"
  log "  Tests: prefix caching effectiveness with realistic shared prompts"
  log "============================================================"
  log ""
  log "NOTE: Keep optimized config running from T2."
  wait_healthy "$VLLM_URL"

  # 8K shared system prompt + 128 unique question per request = ~8.1K input
  # All 1000 requests share the same prefix — prefix caching should dramatically
  # reduce TTFT after the first request computes the prefix KV cache.
  # Compare against T2 (random, no prefix sharing) to quantify the gain.
  capture_kv_metrics "t2b_opt_prefix_pre"
  run_bench "t2b_opt_prefix_1000c_8k_1k" 1000 8000 1000 inf generated-shared-prefix
  capture_kv_metrics "t2b_opt_prefix_post"
  capture_metrics "t2b_prefix"
  cooldown

  # Config A: customer baseline (no prefix caching) for contrast
  log "NOTE: Now start customer baseline (configs/vllm-customer-baseline.sh)"
  prompt_restart "vLLM customer baseline for prefix test (configs/vllm-customer-baseline.sh)"
  wait_healthy "$VLLM_URL"

  capture_kv_metrics "t2b_customer_prefix_pre"
  run_bench "t2b_customer_prefix_1000c_8k_1k" 1000 8000 1000 inf generated-shared-prefix
  capture_kv_metrics "t2b_customer_prefix_post"
  capture_metrics "t2b_customer_prefix"
  cooldown

  log "========== T2b complete =========="
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
  log "  T4: Load Scaling (Config B at varying request rates)"
  log "  Config: configs/vllm-optimized.sh"
  log "  Rates: 0.5 qps (~10 concurrent), 5.0 qps (~100), inf (~512)"
  log "============================================================"
  log ""
  log "NOTE: Ensure optimized config is running (configs/vllm-optimized.sh)"
  wait_healthy "$VLLM_URL"

  # Low load (0.5 qps) — ~10 peak concurrent, latency floor
  run_bench "t4_opt_low_qps0.5" 100 10000 1000 0.5
  cooldown

  # Moderate load (5.0 qps) — ~100 peak concurrent, realistic production
  run_bench "t4_opt_moderate_qps5.0" 200 10000 1000 5.0
  cooldown

  # High load (inf) — all 1000 at once, ~512 concurrent (max-num-seqs cap)
  run_bench "t4_opt_high_inf" 1000 10000 1000 inf
  capture_metrics "t4_scaling"
  cooldown

  log "========== T4 complete =========="
}

run_t5() {
  log "============================================================"
  log "  T5: Simulated Memory-Constrained KV Cache Offloading"
  log "  Simulates smaller-GPU scenarios (e.g. g7e) by constraining"
  log "  --gpu-memory-utilization to 0.30, leaving ~22 GB KV/GPU."
  log "  Tests whether CPU offload and prefix caching mitigate the"
  log "  KV cache pressure under high concurrency."
  log "============================================================"
  log ""

  # T5a: Constrained HBM, no offload, random data (baseline)
  log "--- T5a: gpu-mem-util=0.30, no offload, random ---"
  log "NOTE: Start server with Config B flags BUT --gpu-memory-utilization 0.30"
  log "      (edit configs/vllm-optimized.sh temporarily or pass override)"
  prompt_restart "vLLM optimized, gpu-mem-util=0.30, NO cpu-offload"
  wait_healthy "$VLLM_URL"

  capture_kv_metrics "t5a_pre"
  run_bench "t5a_constrained_nooffload_1000c_10k_1k" 1000 10000 1000 inf
  capture_kv_metrics "t5a_post"
  capture_metrics "t5a_constrained_nooffload"
  cooldown

  # T5b: Constrained HBM + CPU offload, random data
  log "--- T5b: gpu-mem-util=0.30, --cpu-offload-gb 64, random ---"
  log "NOTE: Start server with Config B flags + --gpu-memory-utilization 0.30 + --cpu-offload-gb 64"
  log "      If --cpu-offload-gb fails at startup, skip T5b and note incompatibility."
  prompt_restart "vLLM optimized, gpu-mem-util=0.30, cpu-offload-gb=64"
  wait_healthy "$VLLM_URL"

  capture_kv_metrics "t5b_pre"
  run_bench "t5b_constrained_cpuoffload_1000c_10k_1k" 1000 10000 1000 inf
  capture_kv_metrics "t5b_post"
  capture_metrics "t5b_constrained_cpuoffload"
  cooldown

  # T5c: Constrained HBM, no offload, shared prefix (prefix caching saves the day?)
  log "--- T5c: gpu-mem-util=0.30, no offload, 8K shared prefix ---"
  log "NOTE: Start server with Config B flags + --gpu-memory-utilization 0.30 (no offload)"
  prompt_restart "vLLM optimized, gpu-mem-util=0.30, NO cpu-offload (prefix workload)"
  wait_healthy "$VLLM_URL"

  capture_kv_metrics "t5c_pre"
  run_bench "t5c_constrained_prefix_1000c_8k_1k" 1000 8000 1000 inf generated-shared-prefix
  capture_kv_metrics "t5c_post"
  capture_metrics "t5c_constrained_prefix"
  cooldown

  # T5d: Constrained HBM + Dynamo KVBM tiered offload to FSx
  log "--- T5d: gpu-mem-util=0.30, Dynamo KVBM → CPU DRAM → FSx Lustre, random ---"
  log "NOTE: Start server with configs/vllm-constrained-dynamo-fsx.sh"
  log "      Requires Dynamo KVBM image (dynamo-kvbm-qwen3:latest)."
  log "      Requires FSx mounted at /mnt/fsx."
  log "      Uses dynamo-run (not vllm serve) with 4-tier KV hierarchy:"
  log "        Tier 0: GPU VRAM (~22 GB) → Tier 1: CPU (128 GB) → Tier 2: FSx (500 GB)"
  prompt_restart "Dynamo KVBM constrained + FSx (configs/vllm-constrained-dynamo-fsx.sh)"
  wait_healthy "$VLLM_URL" 900

  capture_kv_metrics "t5d_pre"
  run_bench "t5d_constrained_dynamo_fsx_1000c_10k_1k" 1000 10000 1000 inf
  capture_kv_metrics "t5d_post"
  capture_metrics "t5d_constrained_dynamo_fsx"
  cooldown

  log "========== T5 complete =========="
  log "Compare: T5a (baseline) vs T5b (CPU offload) vs T5c (prefix caching) vs T5d (Dynamo FSx)"
}

run_t6() {
  log "============================================================"
  log "  T6: 2x Replica + CPU Offload (Config E)"
  log "  Config: configs/vllm-2replica-cpuoffload.sh"
  log "  Workload: 1000 requests, 10K input, 1K output"
  log "  Two replicas on ports 8000/8001, round-robin"
  log "============================================================"
  log ""
  log "NOTE: Start BOTH replicas with configs/vllm-2replica-cpuoffload.sh"
  prompt_restart "vLLM 2x replica + CPU offload (configs/vllm-2replica-cpuoffload.sh)"
  wait_healthy "$VLLM_URL"
  wait_healthy "http://localhost:8001"

  # Split 1000 requests across 2 replicas (500 each, round-robin)
  # Run against each port separately for clean metrics, then combine
  run_bench "t6_2rep_r0_500c_10k_1k" 500 10000 1000 inf
  VLLM_URL_ORIG="$VLLM_URL"
  VLLM_URL="http://localhost:8001"
  run_bench "t6_2rep_r1_500c_10k_1k" 500 10000 1000 inf
  VLLM_URL="$VLLM_URL_ORIG"
  capture_metrics "t6_2rep_r0"
  VLLM_URL="http://localhost:8001" capture_metrics "t6_2rep_r1"
  cooldown

  log "========== T6 complete =========="
}

run_t7() {
  log "============================================================"
  log "  T7: Stress Test — 1500 Concurrent (Config E)"
  log "  Config: configs/vllm-2replica-cpuoffload.sh (already running)"
  log "  Workload: 1500 requests, 10K input, 1K output"
  log "  Split 750/750 across two replicas"
  log "============================================================"
  log ""
  log "NOTE: Ensure both replicas from T6 are still running."
  wait_healthy "$VLLM_URL"
  wait_healthy "http://localhost:8001"

  run_bench "t7_2rep_r0_750c_10k_1k" 750 10000 1000 inf
  VLLM_URL_ORIG="$VLLM_URL"
  VLLM_URL="http://localhost:8001"
  run_bench "t7_2rep_r1_750c_10k_1k" 750 10000 1000 inf
  VLLM_URL="$VLLM_URL_ORIG"
  capture_metrics "t7_2rep_r0"
  VLLM_URL="http://localhost:8001" capture_metrics "t7_2rep_r1"
  cooldown

  log "========== T7 complete =========="
}

# --- Main ---

log "Qwen3-Next Customer Benchmark Runner"
log "Phase: $PHASE | Runs: $RUNS | Warmup: $WARMUP | Cooldown: ${COOLDOWN}s"
log "Results: $RESULT_DIR"
log ""

case "$PHASE" in
  t1)  run_t1 ;;
  t2)  run_t2 ;;
  t2b) run_t2b ;;
  t3)  run_t3 ;;
  t4)  run_t4 ;;
  t5)  run_t5 ;;
  t6)  run_t6 ;;
  t7)  run_t7 ;;
  all)
    run_t1
    run_t2
    run_t2b
    run_t3
    run_t4
    run_t5
    run_t6
    run_t7
    ;;
  *)
    echo "Usage: $0 {t1|t2|t2b|t3|t4|t5|t6|t7|all}"
    echo ""
    echo "Phases:"
    echo "  t1  — Customer reproduction (Config A, 1000 concurrent)"
    echo "  t2  — Optimized head-to-head (Config B, same workload)"
    echo "  t2b — Prefix sharing (Config B vs A, 8K shared prefix)"
    echo "  t3  — MTP isolation (Config C vs B)"
    echo "  t4  — Load scaling (Config B at 0.5/5.0/inf qps)"
    echo "  t5  — Simulated memory-constrained KV offload (gpu-mem=0.30)"
    echo "  t6  — 2x replica + CPU offload (Config E, 1000 concurrent)"
    echo "  t7  — Stress test 1500 concurrent (Config E, 2x replica)"
    echo "  all — Run all phases sequentially"
    exit 1
    ;;
esac

log ""
log "All benchmarks for phase '$PHASE' complete."
log "Results saved to: $RESULT_DIR"
