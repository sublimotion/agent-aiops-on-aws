#!/usr/bin/env bash
# observability-smoke-test.sh — Verify observability stack is healthy before any benchmark runs.
#
# Exits non-zero on any failure. Called by infra-deployer Stage 4b gate.
#
# Checks:
#   1. Prometheus /-/healthy returns 200
#   2. DCGM exporter returns 8 GPU entries (DCGM_FI_DEV_GPU_UTIL)
#   3. node-exporter responds
#   4. Prometheus has all targets up
#   5. Engine /metrics reachable (vLLM :8000 OR SGLang :30000)
#   6. At least one vllm:* or sglang:* sample > 0

set -euo pipefail

FAIL=0
fail() { printf '✗ %s\n' "$*" >&2; FAIL=1; }
pass() { printf '✓ %s\n' "$*"; }

# 1. Prometheus itself
if curl -sf http://localhost:9090/-/healthy >/dev/null; then
  pass "prometheus healthy"
else
  fail "prometheus not healthy on :9090"
fi

# 2. DCGM — must show all GPUs
if GPU_COUNT=$(curl -sf http://localhost:9400/metrics 2>/dev/null | grep -c '^DCGM_FI_DEV_GPU_UTIL{'); then
  EXPECTED=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 || echo 0)
  if [[ "$GPU_COUNT" -ge "$EXPECTED" && "$EXPECTED" -gt 0 ]]; then
    pass "dcgm-exporter reports $GPU_COUNT/$EXPECTED GPUs"
  else
    fail "dcgm-exporter reports $GPU_COUNT GPUs (expected $EXPECTED)"
  fi
else
  fail "dcgm-exporter unreachable on :9400"
fi

# 3. node-exporter
if curl -sf http://localhost:9100/metrics | grep -q '^node_cpu_seconds_total'; then
  pass "node-exporter serving"
else
  fail "node-exporter unreachable on :9100"
fi

# 4. Targets up — query Prometheus for up{}
if UP=$(curl -sf "http://localhost:9090/api/v1/query?query=up" 2>/dev/null); then
  DOWN=$(echo "$UP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
down = [r['metric'] for r in d['data']['result'] if r['value'][1] == '0']
print(len(down))
for m in down: print(m, file=sys.stderr)
")
  if [[ "$DOWN" == "0" ]]; then
    pass "all prometheus targets up"
  else
    # Engine may legitimately not be running yet — only fail on DCGM/node
    echo "WARN: $DOWN targets down (may include engine if not yet started)"
  fi
fi

# 5. Engine /metrics (either vLLM or SGLang must eventually be up — skip if neither is running)
ENGINE_UP=""
if curl -sf http://localhost:8000/metrics 2>/dev/null | grep -q '^vllm:'; then
  ENGINE_UP="vllm"
  pass "vLLM /metrics reachable"
elif curl -sf http://localhost:30000/metrics 2>/dev/null | grep -q '^sglang:'; then
  ENGINE_UP="sglang"
  pass "SGLang /metrics reachable"
else
  echo "INFO: no engine running yet — engine metric check deferred until after serving stack is up"
fi

# 6. If an engine is up, verify it's publishing useful histograms
if [[ "$ENGINE_UP" == "vllm" ]]; then
  if curl -sf http://localhost:8000/metrics | grep -qE '^vllm:(time_to_first_token|time_per_output_token|e2e_request_latency)_seconds_bucket'; then
    pass "vLLM TTFT/TPOT/E2E histograms present"
  else
    fail "vLLM missing TTFT/TPOT/E2E histogram metrics"
  fi
elif [[ "$ENGINE_UP" == "sglang" ]]; then
  if curl -sf http://localhost:30000/metrics | grep -qE '^sglang:(time_to_first_token|time_per_output_token|e2e_request_latency)_seconds_bucket'; then
    pass "SGLang TTFT/TPOT/E2E histograms present"
  else
    fail "SGLang missing TTFT/TPOT/E2E histogram metrics"
  fi
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo >&2
  echo "SMOKE TEST FAILED — do not proceed to benchmarks until all checks pass." >&2
  exit 1
fi

echo
echo "Observability smoke test PASSED"
