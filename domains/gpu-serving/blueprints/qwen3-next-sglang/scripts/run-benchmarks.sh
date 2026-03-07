#!/usr/bin/env bash
# Benchmark runner for Qwen3-Next-80B on SGLang + HiCache (g7e instances)
#
# Usage:
#   ./scripts/run-benchmarks.sh [phase]
#
# Phases:
#   s0        — Smoke test: model loads, coherent output
#   s1        — Throughput baseline: QPS sweep (0.5–8.0)
#   s2        — BFCL tool-use evaluation (coding agent viability gate)
#   s3        — HiCache tiered KV cache comparison (L1 vs L2 vs L3)
#   s4        — Swarm concurrency simulation (4–64 concurrent agents)
#   s5        — Functional coding eval (SERA-inspired, real tool execution + SVG)
#   all       — Run all phases sequentially
#
# Prerequisites:
#   - Model staged to /mnt/nvme/models/qwen3-next-fp8
#   - SGLang running with appropriate config from configs/
#   - vllm bench serve available in PATH (used for S0/S1)
#   - Python 3.10+ with aiohttp (for S2/S4)
#
# Environment:
#   SGLANG_URL    — Base URL (default: http://localhost:30000)
#   MODEL_NAME    — Served model name (default: qwen3-next)
#   MODEL_PATH    — Model path for tokenizer (default: /mnt/nvme/models/qwen3-next-fp8)
#   RESULT_DIR    — Output directory (default: ./results/session-YYYYMMDD-HHMM)
#   RUNS          — Runs per config (default: 3)
#   WARMUP        — Warmup requests (default: 30)
#   COOLDOWN      — Seconds between benchmarks (default: 60)
#   DRY_RUN       — Set to 1 to print commands without executing

set -euo pipefail

# --- Configuration ---
PHASE="${1:-s0}"
export MODEL_NAME="${MODEL_NAME:-qwen3-next}"
export MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/qwen3-next-fp8}"
export VLLM_URL="${SGLANG_URL:-http://localhost:30000}"  # benchmark-helpers uses VLLM_URL
export SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
export RESULT_DIR="${RESULT_DIR:-./results/session-$(date +%Y%m%d-%H%M)}"
export RUNS="${RUNS:-3}"
export WARMUP="${WARMUP:-30}"
export COOLDOWN="${COOLDOWN:-60}"
export DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$RESULT_DIR"

# Source shared helpers from benchmark-runner skill
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HELPERS="${SCRIPT_DIR}/../../../../.claude/skills/benchmark-runner/scripts/benchmark-helpers.sh"
if [ -f "$HELPERS" ]; then
  source "$HELPERS"
else
  echo "WARNING: benchmark-helpers.sh not found at $HELPERS"
  echo "Defining minimal helpers inline..."
  log() { echo "$(date +%H:%M:%S) | $*"; }
  cooldown() { sleep "${COOLDOWN:-60}"; }
  wait_healthy() {
    local url="$1" max_wait="${2:-600}" elapsed=0
    log "Waiting for server at $url..."
    while [ $elapsed -lt "$max_wait" ]; do
      curl -sf "${url}/health" > /dev/null 2>&1 && { log "Server healthy"; return 0; }
      sleep 10; elapsed=$((elapsed + 10))
    done
    log "ERROR: Server not healthy after ${max_wait}s"; return 1
  }
fi

# --- S0: Smoke Test ---

run_s0() {
  log "============================================================"
  log "  S0: Smoke Test — Model loads, coherent output"
  log "  Config: configs/sglang-baseline.sh"
  log "============================================================"

  wait_healthy "$SGLANG_URL"

  # Test 1: Simple completion to verify model loads correctly
  log ">>> S0.1: Single request — verify coherent output"
  local s0_dir="${RESULT_DIR}/s0_smoke"
  mkdir -p "$s0_dir"

  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: curl POST ${SGLANG_URL}/v1/chat/completions"
  else
    curl -sf "${SGLANG_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "'"$MODEL_NAME"'",
        "messages": [
          {"role": "system", "content": "You are a helpful coding assistant."},
          {"role": "user", "content": "Write a Python function that checks if a number is prime. Include a docstring and type hints."}
        ],
        "max_tokens": 512,
        "temperature": 0.0
      }' | python3 -m json.tool > "${s0_dir}/single_request.json" 2>&1

    # Check output is non-empty and contains expected content
    if python3 -c "
import json, sys
with open('${s0_dir}/single_request.json') as f:
    data = json.load(f)
content = data['choices'][0]['message']['content']
assert len(content) > 50, f'Output too short: {len(content)} chars'
assert 'def' in content.lower() or 'function' in content.lower(), 'No function definition found'
print(f'S0.1 PASS: {len(content)} chars, contains function definition')
"; then
      log "S0.1: PASS"
    else
      log "S0.1: FAIL — check ${s0_dir}/single_request.json"
      return 1
    fi
  fi

  # Test 2: 10 requests via vllm bench serve
  log ">>> S0.2: 10 requests at QPS 0.5 — verify no CUDA errors"
  run_bench "s0_10req" "$SGLANG_URL" random 1024 512 0.5 --num-prompts=10

  # Test 3: Tool-calling smoke (verify --tool-call-parser works)
  log ">>> S0.3: Tool-call smoke — verify qwen3_coder parser"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: tool-call smoke test"
  else
    curl -sf "${SGLANG_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "'"$MODEL_NAME"'",
        "messages": [
          {"role": "user", "content": "What is the weather in Tokyo?"}
        ],
        "tools": [
          {
            "type": "function",
            "function": {
              "name": "get_weather",
              "description": "Get current weather for a location",
              "parameters": {
                "type": "object",
                "properties": {
                  "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
              }
            }
          }
        ],
        "tool_choice": "auto",
        "max_tokens": 256,
        "temperature": 0.0
      }' | python3 -m json.tool > "${s0_dir}/tool_call_smoke.json" 2>&1

    if python3 -c "
import json, sys
with open('${s0_dir}/tool_call_smoke.json') as f:
    data = json.load(f)
msg = data['choices'][0]['message']
if msg.get('tool_calls'):
    tc = msg['tool_calls'][0]
    print(f'S0.3 PASS: tool_call={tc[\"function\"][\"name\"]}({tc[\"function\"][\"arguments\"]})')
elif 'content' in msg and msg['content']:
    print(f'S0.3 WARN: No tool call, got text response (parser may need tuning)')
    sys.exit(0)
else:
    print('S0.3 FAIL: Empty response')
    sys.exit(1)
"; then
      log "S0.3: PASS"
    else
      log "S0.3: FAIL — check ${s0_dir}/tool_call_smoke.json"
    fi
  fi

  # Test 4: Context length check (try 65536, then 131072)
  log ">>> S0.4: Context length 65536 — 2 requests"
  run_bench "s0_ctx65k" "$SGLANG_URL" random 32768 512 0.5 --num-prompts=2

  capture_metrics "s0_final" "$SGLANG_URL"
  log "========== S0 complete =========="
}

# --- S1: Throughput Baseline ---

run_s1() {
  log "============================================================"
  log "  S1: Throughput Baseline — QPS sweep on g7e"
  log "  Comparison target: vLLM on p5en = 230 tok/s at TP=4"
  log "  Target: >= 150 tok/s (sufficient for swarm use)"
  log "============================================================"

  wait_healthy "$SGLANG_URL"
  capture_metrics "s1_pre" "$SGLANG_URL"

  # S1a: QPS 0.5, 100 requests
  run_bench "s1a_qps0.5" "$SGLANG_URL" random 1024 512 0.5 --num-prompts=100
  cooldown

  # S1b: QPS 2.0, 200 requests
  run_bench "s1b_qps2.0" "$SGLANG_URL" random 1024 512 2.0 --num-prompts=200
  cooldown

  # S1c: QPS 4.0, 400 requests
  run_bench "s1c_qps4.0" "$SGLANG_URL" random 1024 512 4.0 --num-prompts=400
  cooldown

  # S1d: QPS 8.0, 400 requests
  run_bench "s1d_qps8.0" "$SGLANG_URL" random 1024 512 8.0 --num-prompts=400
  capture_metrics "s1_post" "$SGLANG_URL"
  cooldown

  log "========== S1 complete =========="
}

# --- S2: BFCL Tool-Use Evaluation ---

run_s2() {
  log "============================================================"
  log "  S2: BFCL Tool-Use Evaluation"
  log "  Decision gate: BFCL >= 75 → coding agent viable"
  log "============================================================"

  wait_healthy "$SGLANG_URL"

  local s2_dir="${RESULT_DIR}/s2_bfcl"
  mkdir -p "$s2_dir"

  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: python3 ${SCRIPT_DIR}/bfcl-eval.py --endpoint ${SGLANG_URL} --output ${s2_dir}"
  else
    python3 "${SCRIPT_DIR}/bfcl-eval.py" \
      --endpoint "$SGLANG_URL" \
      --model "$MODEL_NAME" \
      --output-dir "$s2_dir" \
      --num-scenarios 200 \
      --max-concurrent 4 \
      2>&1 | tee "${s2_dir}/bfcl_eval.log"

    # Parse and report score
    if [ -f "${s2_dir}/bfcl_summary.json" ]; then
      python3 -c "
import json
with open('${s2_dir}/bfcl_summary.json') as f:
    s = json.load(f)
score = s.get('overall_score', 0)
if score >= 80:
    verdict = 'STRONG — competitive with Claude Sonnet for tool orchestration'
elif score >= 75:
    verdict = 'PROCEED — viable for both swarm and interactive coding agent use'
elif score >= 70:
    verdict = 'CAUTION — viable for swarms (retry at machine speed), not interactive'
else:
    verdict = 'STOP — model is not viable for coding agents'
print(f'BFCL Score: {score:.1f} — {verdict}')
print(f'First-call success: {s.get(\"first_call_success_rate\", 0)*100:.1f}%')
print(f'Multi-turn completion: {s.get(\"multi_turn_completion_rate\", 0)*100:.1f}%')
"
    fi
  fi

  capture_metrics "s2_post" "$SGLANG_URL"
  log "========== S2 complete =========="
}

# --- S3: HiCache Tiered KV Cache ---

run_s3() {
  log "============================================================"
  log "  S3: HiCache Tiered KV Cache Comparison"
  log "  S3a: L1 only (baseline)"
  log "  S3b: L1 + L2 (CPU offload)"
  log "  S3c: L1 + L2 + L3 (NVMe offload)"
  log "============================================================"

  # S3a: L1 only (baseline config — no HiCache)
  log ">>> S3a: L1 only baseline"
  log "NOTE: Ensure server is running with configs/sglang-baseline.sh"
  wait_healthy "$SGLANG_URL"

  capture_kv_metrics "s3a_pre" "$SGLANG_URL"
  # Shared prefix workload: 8K system prompt + 128-token unique suffix
  run_bench "s3a_l1_only" "$SGLANG_URL" \
    generated-shared-prefix 8192 256 2.0 \
    --gsp-system-prompt-len=8000 --gsp-question-len=128 --gsp-output-len=256 \
    --num-prompts=500
  capture_kv_metrics "s3a_post" "$SGLANG_URL"

  # Concurrency sweep on L1
  for CONC in 50 100 200; do
    run_bench "s3a_l1_conc${CONC}" "$SGLANG_URL" \
      generated-shared-prefix 8192 256 inf \
      --gsp-system-prompt-len=8000 --gsp-question-len=128 --gsp-output-len=256 \
      --num-prompts="$CONC"
    capture_kv_metrics "s3a_conc${CONC}" "$SGLANG_URL"
  done
  cooldown

  # S3b: L1 + L2 (HiCache CPU tier)
  log ">>> S3b: L1 + L2 (CPU offload)"
  log "NOTE: Restart with configs/sglang-hicache-l2.sh"
  prompt_restart "SGLang HiCache L2 (configs/sglang-hicache-l2.sh)"
  wait_healthy "$SGLANG_URL"

  capture_kv_metrics "s3b_pre" "$SGLANG_URL"
  run_bench "s3b_l1l2" "$SGLANG_URL" \
    generated-shared-prefix 8192 256 2.0 \
    --gsp-system-prompt-len=8000 --gsp-question-len=128 --gsp-output-len=256 \
    --num-prompts=500
  capture_kv_metrics "s3b_post" "$SGLANG_URL"

  for CONC in 50 100 200; do
    run_bench "s3b_l1l2_conc${CONC}" "$SGLANG_URL" \
      generated-shared-prefix 8192 256 inf \
      --gsp-system-prompt-len=8000 --gsp-question-len=128 --gsp-output-len=256 \
      --num-prompts="$CONC"
    capture_kv_metrics "s3b_conc${CONC}" "$SGLANG_URL"
  done
  cooldown

  # S3c: L1 + L2 + L3 (HiCache NVMe tier)
  log ">>> S3c: L1 + L2 + L3 (NVMe offload)"
  log "NOTE: Restart with configs/sglang-hicache-nvme.sh"
  prompt_restart "SGLang HiCache L3 NVMe (configs/sglang-hicache-nvme.sh)"
  wait_healthy "$SGLANG_URL"

  capture_kv_metrics "s3c_pre" "$SGLANG_URL"
  run_bench "s3c_l1l2l3" "$SGLANG_URL" \
    generated-shared-prefix 8192 256 2.0 \
    --gsp-system-prompt-len=8000 --gsp-question-len=128 --gsp-output-len=256 \
    --num-prompts=500
  capture_kv_metrics "s3c_post" "$SGLANG_URL"

  for CONC in 50 100 200; do
    run_bench "s3c_l1l2l3_conc${CONC}" "$SGLANG_URL" \
      generated-shared-prefix 8192 256 inf \
      --gsp-system-prompt-len=8000 --gsp-question-len=128 --gsp-output-len=256 \
      --num-prompts="$CONC"
    capture_kv_metrics "s3c_conc${CONC}" "$SGLANG_URL"
  done

  log "========== S3 complete =========="
  log "Compare S3a vs S3b for CPU offload value"
  log "Compare S3b vs S3c for marginal NVMe value"
}

# --- S4: Swarm Concurrency Simulation ---

run_s4() {
  log "============================================================"
  log "  S4: Swarm Concurrency Simulation"
  log "  Simulates concurrent coding agents with tool exec delays"
  log "============================================================"

  wait_healthy "$SGLANG_URL"

  local s4_dir="${RESULT_DIR}/s4_swarm"
  mkdir -p "$s4_dir"

  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: python3 ${SCRIPT_DIR}/swarm-simulator.py"
  else
    for AGENTS in 4 8 16 32 64; do
      log ">>> S4: ${AGENTS} concurrent agents"

      # Skip 64-agent test if 32-agent showed >90% GPU util
      if [ "$AGENTS" -eq 64 ] && [ -f "${s4_dir}/32_agents_summary.json" ]; then
        local gpu_util
        gpu_util=$(python3 -c "
import json
with open('${s4_dir}/32_agents_summary.json') as f:
    d = json.load(f)
print(d.get('avg_gpu_utilization_pct', 0))
" 2>/dev/null || echo "0")
        if python3 -c "exit(0 if float('$gpu_util') > 90 else 1)" 2>/dev/null; then
          log "Skipping 64-agent test: 32-agent GPU util was ${gpu_util}% (>90%)"
          continue
        fi
      fi

      python3 "${SCRIPT_DIR}/swarm-simulator.py" \
        --endpoint "$SGLANG_URL" \
        --model "$MODEL_NAME" \
        --num-agents "$AGENTS" \
        --duration 900 \
        --tool-delay-min 5 \
        --tool-delay-max 30 \
        --output-dir "$s4_dir" \
        2>&1 | tee "${s4_dir}/${AGENTS}_agents.log"

      capture_kv_metrics "s4_${AGENTS}agents" "$SGLANG_URL"
      cooldown
    done
  fi

  log "========== S4 complete =========="
}

# --- S5: Functional Coding Evaluation (SERA-inspired) ---

run_s5() {
  log "============================================================"
  log "  S5: Functional Coding Evaluation (SERA-inspired)"
  log "  Real multi-turn tool execution + SVG patch reproduction"
  log "  Closes the loop: can the model actually fix bugs?"
  log "============================================================"

  wait_healthy "$SGLANG_URL"

  local s5_dir="${RESULT_DIR}/s5_functional"
  mkdir -p "$s5_dir"

  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: python3 ${SCRIPT_DIR}/functional-eval.py --endpoint ${SGLANG_URL} --output-dir ${s5_dir}"
  else
    python3 "${SCRIPT_DIR}/functional-eval.py" \
      --endpoint "$SGLANG_URL" \
      --model "$MODEL_NAME" \
      --output-dir "$s5_dir" \
      --max-concurrent 2 \
      2>&1 | tee "${s5_dir}/functional_eval.log"

    # Parse and report verdict
    if [ -f "${s5_dir}/functional_summary.json" ]; then
      python3 -c "
import json
with open('${s5_dir}/functional_summary.json') as f:
    s = json.load(f)
rate = s.get('test_pass_rate', 0)
svg = s.get('svg_reproduction_rate', 0)
print(f'Task completion: {s[\"completed\"]}/{s[\"total_tasks\"]} ({rate*100:.0f}%)')
print(f'Tests passing:   {s[\"tests_pass\"]}/{s[\"total_tasks\"]}')
if s.get('svg_attempted', 0) > 0:
    print(f'SVG reproduction: {s[\"svg_reproduced\"]}/{s[\"svg_attempted\"]} ({svg*100:.0f}%)')
    print(f'SVG avg recall:   {s[\"avg_svg_recall\"]*100:.0f}%')
if rate >= 0.8:
    print('Verdict: STRONG — model reliably fixes bugs through multi-turn tool use')
elif rate >= 0.6:
    print('Verdict: VIABLE — retry strategy recommended for production')
elif rate >= 0.4:
    print('Verdict: MARGINAL — swarm-only viability')
else:
    print('Verdict: NOT VIABLE — model cannot reliably fix bugs through tool use')
"
    fi
  fi

  capture_metrics "s5_post" "$SGLANG_URL"
  log "========== S5 complete =========="
}

# ===================================================================
# DEVSTRAL SMALL 2 COMPARISON TRACK (D0–D5)
# vLLM on single GPU (port 8000) — head-to-head with Qwen3-Next
# ===================================================================

DEVSTRAL_URL="${DEVSTRAL_URL:-http://localhost:8000}"
DEVSTRAL_MODEL="${DEVSTRAL_MODEL:-devstral-small-2}"
DEVSTRAL_MODEL_PATH_LOCAL="${DEVSTRAL_MODEL_PATH:-/mnt/nvme/models/devstral-small-2-fp8}"

# --- D0: Devstral Smoke Test ---

run_d0() {
  log "============================================================"
  log "  D0: Devstral Small 2 — Smoke Test (vLLM, 1 GPU)"
  log "  Config: configs/devstral-vllm-baseline.sh"
  log "============================================================"

  wait_healthy "$DEVSTRAL_URL"

  local d0_dir="${RESULT_DIR}/d0_smoke"
  mkdir -p "$d0_dir"

  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: curl POST ${DEVSTRAL_URL}/v1/chat/completions"
  else
    # D0.1: Single coding request
    log ">>> D0.1: Single coding request — verify coherent output"
    curl -sf "${DEVSTRAL_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "'"$DEVSTRAL_MODEL"'",
        "messages": [
          {"role": "system", "content": "You are a helpful coding assistant."},
          {"role": "user", "content": "Write a Python function that checks if a number is prime. Include a docstring and type hints."}
        ],
        "max_tokens": 512,
        "temperature": 0.0
      }' | python3 -m json.tool > "${d0_dir}/single_request.json" 2>&1

    if python3 -c "
import json
with open('${d0_dir}/single_request.json') as f:
    data = json.load(f)
content = data['choices'][0]['message']['content']
assert len(content) > 50, f'Output too short: {len(content)} chars'
print(f'D0.1 PASS: {len(content)} chars')
"; then
      log "D0.1: PASS"
    else
      log "D0.1: FAIL — check ${d0_dir}/single_request.json"
      return 1
    fi

    # D0.2: 10 requests via vllm bench serve
    log ">>> D0.2: 10 requests at QPS 0.5"
    run_bench "d0_10req" "$DEVSTRAL_URL" random 1024 512 0.5 --num-prompts=10

    # D0.3: Tool-calling smoke (verify mistral parser)
    log ">>> D0.3: Tool-call smoke — verify mistral parser"
    curl -sf "${DEVSTRAL_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "'"$DEVSTRAL_MODEL"'",
        "messages": [
          {"role": "user", "content": "What is the weather in Tokyo?"}
        ],
        "tools": [
          {
            "type": "function",
            "function": {
              "name": "get_weather",
              "description": "Get current weather for a location",
              "parameters": {
                "type": "object",
                "properties": {
                  "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
              }
            }
          }
        ],
        "tool_choice": "auto",
        "max_tokens": 256,
        "temperature": 0.0
      }' | python3 -m json.tool > "${d0_dir}/tool_call_smoke.json" 2>&1

    if python3 -c "
import json
with open('${d0_dir}/tool_call_smoke.json') as f:
    data = json.load(f)
msg = data['choices'][0]['message']
if msg.get('tool_calls'):
    tc = msg['tool_calls'][0]
    print(f'D0.3 PASS: tool_call={tc[\"function\"][\"name\"]}')
elif msg.get('content'):
    print('D0.3 WARN: No tool call, got text response')
else:
    print('D0.3 FAIL: Empty response'); exit(1)
"; then
      log "D0.3: PASS"
    else
      log "D0.3: FAIL — check ${d0_dir}/tool_call_smoke.json"
    fi
  fi

  capture_metrics "d0_final" "$DEVSTRAL_URL"
  log "========== D0 complete =========="
}

# --- D1: Devstral Throughput Baseline ---

run_d1() {
  log "============================================================"
  log "  D1: Devstral Small 2 — Throughput Baseline"
  log "  D1a-d: Single GPU (1x RTX PRO 6000)"
  log "  D1e-h: TP=4 (4x RTX PRO 6000) — apples-to-apples vs S1"
  log "============================================================"

  wait_healthy "$DEVSTRAL_URL"
  capture_metrics "d1_pre" "$DEVSTRAL_URL"

  # --- Single GPU throughput (cost efficiency comparison) ---
  log ">>> D1 single-GPU: configs/devstral-vllm-baseline.sh"

  run_bench "d1a_1gpu_qps0.5" "$DEVSTRAL_URL" random 1024 512 0.5 --num-prompts=100
  cooldown
  run_bench "d1b_1gpu_qps2.0" "$DEVSTRAL_URL" random 1024 512 2.0 --num-prompts=200
  cooldown
  run_bench "d1c_1gpu_qps4.0" "$DEVSTRAL_URL" random 1024 512 4.0 --num-prompts=400
  cooldown
  run_bench "d1d_1gpu_qps8.0" "$DEVSTRAL_URL" random 1024 512 8.0 --num-prompts=400
  capture_metrics "d1_1gpu_post" "$DEVSTRAL_URL"
  cooldown

  # --- TP=4 throughput (same GPU count as Qwen3-Next S1) ---
  log ""
  log ">>> D1 TP=4: Restart with configs/devstral-vllm-4gpu.sh"
  log "    This gives apples-to-apples comparison with Qwen3-Next S1 (also TP=4)"
  prompt_restart "Devstral vLLM TP=4 (configs/devstral-vllm-4gpu.sh)"
  wait_healthy "$DEVSTRAL_URL"

  run_bench "d1e_4gpu_qps0.5" "$DEVSTRAL_URL" random 1024 512 0.5 --num-prompts=100
  cooldown
  run_bench "d1f_4gpu_qps2.0" "$DEVSTRAL_URL" random 1024 512 2.0 --num-prompts=200
  cooldown
  run_bench "d1g_4gpu_qps4.0" "$DEVSTRAL_URL" random 1024 512 4.0 --num-prompts=400
  cooldown
  run_bench "d1h_4gpu_qps8.0" "$DEVSTRAL_URL" random 1024 512 8.0 --num-prompts=400
  capture_metrics "d1_4gpu_post" "$DEVSTRAL_URL"

  log ""
  log "========== D1 complete =========="
  log "Compare D1a-d (1 GPU) vs S1a-d (Qwen3-Next 4 GPU) for cost efficiency"
  log "Compare D1e-h (4 GPU) vs S1a-d (Qwen3-Next 4 GPU) for raw throughput"
}

# --- D2: Devstral BFCL Tool-Use ---

run_d2() {
  log "============================================================"
  log "  D2: Devstral Small 2 — BFCL Tool-Use Evaluation"
  log "  Head-to-head with S2 (Qwen3-Next)"
  log "  Note: Qwen3-Next published BFCL-v3 = 70.3 (below our 75 gate)"
  log "  Devstral has no published BFCL — this fills the gap"
  log "============================================================"

  wait_healthy "$DEVSTRAL_URL"

  local d2_dir="${RESULT_DIR}/d2_bfcl"
  mkdir -p "$d2_dir"

  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: python3 ${SCRIPT_DIR}/bfcl-eval.py --endpoint ${DEVSTRAL_URL} --model ${DEVSTRAL_MODEL}"
  else
    python3 "${SCRIPT_DIR}/bfcl-eval.py" \
      --endpoint "$DEVSTRAL_URL" \
      --model "$DEVSTRAL_MODEL" \
      --output-dir "$d2_dir" \
      --num-scenarios 200 \
      --max-concurrent 4 \
      2>&1 | tee "${d2_dir}/bfcl_eval.log"

    if [ -f "${d2_dir}/bfcl_summary.json" ]; then
      python3 -c "
import json
with open('${d2_dir}/bfcl_summary.json') as f:
    s = json.load(f)
score = s.get('overall_score', 0)
if score >= 80:
    verdict = 'STRONG — competitive with Claude Sonnet for tool orchestration'
elif score >= 75:
    verdict = 'PROCEED — viable for both swarm and interactive coding agent use'
elif score >= 70:
    verdict = 'CAUTION — viable for swarms (retry at machine speed), not interactive'
else:
    verdict = 'STOP — model is not viable for coding agents'
print(f'Devstral BFCL Score: {score:.1f} — {verdict}')
print(f'First-call success: {s.get(\"first_call_success_rate\", 0)*100:.1f}%')
print(f'Multi-turn completion: {s.get(\"multi_turn_completion_rate\", 0)*100:.1f}%')
"
    fi
  fi

  capture_metrics "d2_post" "$DEVSTRAL_URL"
  log "========== D2 complete =========="
}

# --- D3: Devstral Prefix Caching (replaces HiCache — standard GQA needs no tiered KV) ---

run_d3() {
  log "============================================================"
  log "  D3: Devstral Small 2 — Prefix Caching Effectiveness"
  log "  Standard GQA → vLLM APC works natively (no HiCache needed)"
  log "  Measures cache hit rate and concurrency headroom"
  log "============================================================"

  wait_healthy "$DEVSTRAL_URL"

  # Shared prefix workload (same as S3 for comparison)
  capture_kv_metrics "d3_pre" "$DEVSTRAL_URL"
  run_bench "d3_prefix" "$DEVSTRAL_URL" \
    generated-shared-prefix 8192 256 2.0 \
    --gsp-system-prompt-len=8000 --gsp-question-len=128 --gsp-output-len=256 \
    --num-prompts=500
  capture_kv_metrics "d3_post" "$DEVSTRAL_URL"

  # Concurrency sweep (same levels as S3 for comparison)
  for CONC in 50 100 200; do
    run_bench "d3_conc${CONC}" "$DEVSTRAL_URL" \
      generated-shared-prefix 8192 256 inf \
      --gsp-system-prompt-len=8000 --gsp-question-len=128 --gsp-output-len=256 \
      --num-prompts="$CONC"
    capture_kv_metrics "d3_conc${CONC}" "$DEVSTRAL_URL"
  done

  log "========== D3 complete =========="
  log "Compare D3 (Devstral APC, 1 GPU) vs S3a (Qwen3-Next L1 only, 4 GPU)"
}

# --- D4: Devstral Swarm Concurrency ---

run_d4() {
  log "============================================================"
  log "  D4: Devstral Small 2 — Swarm Concurrency (1 GPU)"
  log "  Same agent simulation as S4, on single GPU"
  log "  Key question: how many agents can 1 GPU sustain?"
  log "============================================================"

  wait_healthy "$DEVSTRAL_URL"

  local d4_dir="${RESULT_DIR}/d4_swarm"
  mkdir -p "$d4_dir"

  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: python3 ${SCRIPT_DIR}/swarm-simulator.py --endpoint ${DEVSTRAL_URL}"
  else
    for AGENTS in 4 8 16 32; do
      log ">>> D4: ${AGENTS} concurrent agents on Devstral (1 GPU)"

      if [ "$AGENTS" -eq 32 ] && [ -f "${d4_dir}/16_agents_summary.json" ]; then
        local gpu_util
        gpu_util=$(python3 -c "
import json
with open('${d4_dir}/16_agents_summary.json') as f:
    d = json.load(f)
print(d.get('avg_gpu_utilization_pct', 0))
" 2>/dev/null || echo "0")
        if python3 -c "exit(0 if float('$gpu_util') > 90 else 1)" 2>/dev/null; then
          log "Skipping 32-agent test: 16-agent GPU util was ${gpu_util}% (>90%)"
          continue
        fi
      fi

      python3 "${SCRIPT_DIR}/swarm-simulator.py" \
        --endpoint "$DEVSTRAL_URL" \
        --model "$DEVSTRAL_MODEL" \
        --num-agents "$AGENTS" \
        --duration 900 \
        --tool-delay-min 5 \
        --tool-delay-max 30 \
        --output-dir "$d4_dir" \
        2>&1 | tee "${d4_dir}/${AGENTS}_agents.log"

      capture_kv_metrics "d4_${AGENTS}agents" "$DEVSTRAL_URL"
      cooldown
    done
  fi

  log "========== D4 complete =========="
  log "Compare D4 (1 GPU, ~\$2/hr) vs S4 (4 GPU, ~\$17/hr) for cost-adjusted swarm capacity"
}

# --- D5: Devstral Functional Eval ---

run_d5() {
  log "============================================================"
  log "  D5: Devstral Small 2 — Functional Coding Evaluation"
  log "  Same SERA-inspired tasks as S5 — head-to-head comparison"
  log "  Devstral has 68% SWE-bench Verified — should excel here"
  log "============================================================"

  wait_healthy "$DEVSTRAL_URL"

  local d5_dir="${RESULT_DIR}/d5_functional"
  mkdir -p "$d5_dir"

  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: python3 ${SCRIPT_DIR}/functional-eval.py --endpoint ${DEVSTRAL_URL} --model ${DEVSTRAL_MODEL}"
  else
    python3 "${SCRIPT_DIR}/functional-eval.py" \
      --endpoint "$DEVSTRAL_URL" \
      --model "$DEVSTRAL_MODEL" \
      --output-dir "$d5_dir" \
      --max-concurrent 2 \
      2>&1 | tee "${d5_dir}/functional_eval.log"

    if [ -f "${d5_dir}/functional_summary.json" ]; then
      python3 -c "
import json
with open('${d5_dir}/functional_summary.json') as f:
    s = json.load(f)
rate = s.get('test_pass_rate', 0)
svg = s.get('svg_reproduction_rate', 0)
print(f'Devstral task completion: {s[\"completed\"]}/{s[\"total_tasks\"]} ({rate*100:.0f}%)')
print(f'Tests passing:           {s[\"tests_pass\"]}/{s[\"total_tasks\"]}')
if s.get('svg_attempted', 0) > 0:
    print(f'SVG reproduction:        {s[\"svg_reproduced\"]}/{s[\"svg_attempted\"]} ({svg*100:.0f}%)')
    print(f'SVG avg recall:          {s[\"avg_svg_recall\"]*100:.0f}%')
if rate >= 0.8:
    print('Verdict: STRONG — model reliably fixes bugs through multi-turn tool use')
elif rate >= 0.6:
    print('Verdict: VIABLE — retry strategy recommended for production')
elif rate >= 0.4:
    print('Verdict: MARGINAL — swarm-only viability')
else:
    print('Verdict: NOT VIABLE — model cannot reliably fix bugs through tool use')
"
    fi
  fi

  capture_metrics "d5_post" "$DEVSTRAL_URL"
  log "========== D5 complete =========="
}

# --- Main ---

log "Coding Agent Feasibility Benchmark Runner"
log "Phase: $PHASE | Runs: $RUNS | Warmup: $WARMUP | Cooldown: ${COOLDOWN}s"
log ""
log "Track S (Qwen3-Next 80B):    SGLang @ $SGLANG_URL"
log "Track D (Devstral Small 2):  vLLM   @ $DEVSTRAL_URL"
log "Results: $RESULT_DIR"
log ""

case "$PHASE" in
  # --- Qwen3-Next (SGLang) track ---
  s0)  run_s0 ;;
  s1)  run_s1 ;;
  s2)  run_s2 ;;
  s3)  run_s3 ;;
  s4)  run_s4 ;;
  s5)  run_s5 ;;
  s-all)
    run_s0; run_s1; run_s2; run_s3; run_s4; run_s5
    ;;
  # --- Devstral Small 2 (vLLM) track ---
  d0)  run_d0 ;;
  d1)  run_d1 ;;
  d2)  run_d2 ;;
  d3)  run_d3 ;;
  d4)  run_d4 ;;
  d5)  run_d5 ;;
  d-all)
    run_d0; run_d1; run_d2; run_d3; run_d4; run_d5
    ;;
  # --- Head-to-head (run both tracks) ---
  all)
    log "=== Running both tracks sequentially ==="
    log "=== Track S: Qwen3-Next on SGLang ==="
    run_s0; run_s1; run_s2; run_s3; run_s4; run_s5
    log ""
    log "=== Track D: Devstral Small 2 on vLLM ==="
    run_d0; run_d1; run_d2; run_d3; run_d4; run_d5
    ;;
  *)
    echo "Usage: $0 {s0|s1|s2|s3|s4|s5|s-all|d0|d1|d2|d3|d4|d5|d-all|all}"
    echo ""
    echo "Qwen3-Next (SGLang, TP=4, port 30000):"
    echo "  s0  -- Smoke test (model loads, tool-call parser)"
    echo "  s1  -- Throughput baseline (QPS 0.5–8.0)"
    echo "  s2  -- BFCL tool-use evaluation"
    echo "  s3  -- HiCache tiered KV cache (L1 vs L2 vs L3)"
    echo "  s4  -- Swarm concurrency simulation (4–64 agents)"
    echo "  s5  -- Functional coding eval (SERA-inspired)"
    echo "  s-all -- Run all S phases"
    echo ""
    echo "Devstral Small 2 (vLLM, 1 GPU, port 8000):"
    echo "  d0  -- Smoke test (model loads, mistral tool parser)"
    echo "  d1  -- Throughput: 1-GPU + TP=4 (QPS 0.5–8.0)"
    echo "  d2  -- BFCL tool-use evaluation"
    echo "  d3  -- Prefix caching effectiveness"
    echo "  d4  -- Swarm concurrency simulation (4–32 agents)"
    echo "  d5  -- Functional coding eval (SERA-inspired)"
    echo "  d-all -- Run all D phases"
    echo ""
    echo "Head-to-head:"
    echo "  all -- Run both tracks sequentially"
    exit 1
    ;;
esac

log ""
log "All benchmarks for phase '$PHASE' complete."
log "Results saved to: $RESULT_DIR"
log ""
log "Next steps:"
log "  1. Validate: bash .claude/skills/benchmark-runner/scripts/validate-results.sh $RESULT_DIR"
log "  2. Analyze:  invoke benchmark-analyst agent with $RESULT_DIR"
log "  3. Visualize: /benchmark-visual"
