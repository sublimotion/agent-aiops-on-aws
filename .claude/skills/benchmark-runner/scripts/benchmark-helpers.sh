#!/usr/bin/env bash
# Shared benchmark helper functions for LLM serving benchmarks.
# Source this file from your blueprint's run-benchmarks.sh.
#
# Usage:
#   source "$(dirname "$0")/../../.claude/skills/benchmark-runner/scripts/benchmark-helpers.sh"
#
# Required variables (set before sourcing):
#   MODEL_NAME  — Served model name (e.g., "qwen3-next")
#   MODEL_PATH  — Local model path for tokenizer (e.g., /mnt/nvme/models/qwen3-next-fp8)
#
# Optional variables (with defaults):
#   VLLM_URL    — Base URL (default: http://localhost:8000)
#   RESULT_DIR  — Output directory (default: ./results/session-YYYYMMDD-HHMM)
#   RUNS        — Runs per config (default: 3)
#   WARMUP      — Warmup requests (default: 30)
#   COOLDOWN    — Seconds between benchmarks (default: 60)
#   DRY_RUN     — Set to 1 to print commands without executing (default: 0)

set -euo pipefail

# --- Defaults ---
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
RESULT_DIR="${RESULT_DIR:-./results/session-$(date +%Y%m%d-%H%M)}"
RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-30}"
COOLDOWN_SECS="${COOLDOWN:-60}"
DRY_RUN="${DRY_RUN:-0}"

# --- Core Helpers ---

log() { echo "$(date +%H:%M:%S) | $*"; }

cooldown() {
  if [ "$COOLDOWN_SECS" -gt 0 ]; then
    log "Cooldown ${COOLDOWN_SECS}s..."
    sleep "$COOLDOWN_SECS"
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

# --- Metrics Capture ---

# Capture raw Prometheus metrics snapshot.
# Args: label [url]
capture_metrics() {
  local label="$1"
  local url="${2:-$VLLM_URL}"
  local out="${RESULT_DIR}/${label}_metrics.txt"
  curl -sf "${url}/metrics" > "$out" 2>/dev/null || true
  log "Metrics captured: $out"
}

# Capture KV cache specific metrics (prefix cache, GPU/CPU cache, offload stats).
# Includes vLLM standard metrics, Dynamo KVBM per-tier metrics, FSx cache size, and GPU memory.
# Args: label [url]
capture_kv_metrics() {
  local label="$1"
  local url="${2:-$VLLM_URL}"
  local out="${RESULT_DIR}/${label}_kv_metrics.txt"

  log "Capturing KV cache metrics for $label..."
  {
    echo "=== KV Cache Metrics: ${label} ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
    echo ""

    # 1. vLLM standard metrics
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

    # 3. FSx disk cache size (validates disk tier usage)
    echo "--- FSx Cache Directory ---"
    local fsx_cache="${FSX_CACHE_DIR:-/mnt/fsx/kv-cache}"
    if [ -d "$fsx_cache" ]; then
      du -sh "$fsx_cache" 2>/dev/null || echo "Cannot read $fsx_cache"
      ls -la "$fsx_cache" 2>/dev/null | head -20
    else
      echo "FSx cache dir not found at $fsx_cache (expected if not using FSx tier)"
    fi

    echo ""

    # 4. GPU memory snapshot
    echo "--- GPU Memory ---"
    nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free \
      --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

    echo ""
    echo "=== End KV Cache Metrics ==="
  } > "$out"

  # Full metrics dump for post-hoc analysis
  curl -sf "${url}/metrics" > "${RESULT_DIR}/${label}_full_metrics.txt" 2>/dev/null || true
}

# --- Benchmark Execution ---

# Build dataset-specific vllm bench serve flags.
# Args: dataset input_len output_len [extra_args...]
# Returns: array of flags via _DATASET_ARGS global
_build_dataset_args() {
  local dataset="$1" input_len="$2" output_len="$3"
  shift 3
  _DATASET_ARGS=()

  case "$dataset" in
    random)
      _DATASET_ARGS+=(--dataset-name random --random-input-len "$input_len" --random-output-len "$output_len")
      ;;
    generated-shared-prefix)
      local gsp_sys="${input_len}"
      local gsp_q=256
      local gsp_out="${output_len}"
      # Allow overrides via extra args
      for arg in "$@"; do
        case "$arg" in
          --gsp-system-prompt-len=*) gsp_sys="${arg#*=}" ;;
          --gsp-question-len=*) gsp_q="${arg#*=}" ;;
          --gsp-output-len=*) gsp_out="${arg#*=}" ;;
        esac
      done
      _DATASET_ARGS+=(--dataset-name generated-shared-prefix
                      --gsp-system-prompt-len "$gsp_sys"
                      --gsp-question-len "$gsp_q"
                      --gsp-output-len "$gsp_out")
      ;;
    prefix_repetition)
      local pr_prefix="${input_len}"
      local pr_suffix="${output_len}"
      for arg in "$@"; do
        case "$arg" in
          --prefix-repetition-prefix-len=*) pr_prefix="${arg#*=}" ;;
          --prefix-repetition-suffix-len=*) pr_suffix="${arg#*=}" ;;
        esac
      done
      _DATASET_ARGS+=(--dataset-name prefix_repetition
                      --prefix-repetition-prefix-len "$pr_prefix"
                      --prefix-repetition-suffix-len "$pr_suffix")
      ;;
    *)
      log "WARNING: Unknown dataset '$dataset', passing as-is"
      _DATASET_ARGS+=(--dataset-name "$dataset")
      ;;
  esac
}

# Run a single benchmark.
# Standard signature: run_bench label base_url dataset input_len output_len qps [extra_args...]
run_bench() {
  local label="$1" base_url="$2" dataset="$3" input_len="$4" output_len="$5" qps="$6"
  shift 6

  local out_dir="${RESULT_DIR}/${label}"
  mkdir -p "$out_dir"

  _build_dataset_args "$dataset" "$input_len" "$output_len" "$@"

  local cmd=(
    vllm bench serve
    --model "$MODEL_NAME"
    --tokenizer "$MODEL_PATH"
    --backend openai-chat
    --base-url "$base_url"
    --endpoint /v1/chat/completions
    --num-prompts 100
    --num-warmups "$WARMUP"
    --request-rate "$qps"
    --metric-percentiles 25,50,75,90,95,99
    --percentile-metrics ttft,tpot,itl,e2el
    --save-result
    --save-detailed
    --result-dir "$out_dir"
    "${_DATASET_ARGS[@]}"
  )

  # Pass through any extra args that aren't dataset overrides
  for arg in "$@"; do
    case "$arg" in
      --gsp-*|--prefix-repetition-*) ;; # already handled
      --num-prompts=*) cmd+=(--num-prompts "${arg#*=}") ;;
      *) cmd+=("$arg") ;;
    esac
  done

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

# Custbench-compatible run_bench variant.
# Args: label num_prompts input_len output_len qps [dataset] [extra_args...]
run_bench_custbench() {
  local label="$1" num_prompts="$2" input_len="$3" output_len="$4" qps="$5"
  local dataset="${6:-random}"
  shift 6 2>/dev/null || shift 5

  local out_dir="${RESULT_DIR}/${label}"
  mkdir -p "$out_dir"

  _build_dataset_args "$dataset" "$input_len" "$output_len" "$@"

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
    "${_DATASET_ARGS[@]}"
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

log "benchmark-helpers.sh loaded (MODEL_NAME=${MODEL_NAME:-<unset>})"
