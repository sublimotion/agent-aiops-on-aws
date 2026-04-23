#!/bin/bash
# llm-d on HyperPod — L2/L3 Cache Integration Test
#
# Tests whether KV cache entries written by one vLLM replica are readable
# by another replica via the L2 daemon (same node) or L3 FSx (cross-node).
#
# Usage:
#   ./test_cache_integration.sh <gateway-url> [namespace]
#
# Prerequisites:
#   - vLLM replicas deployed via llm-d with ≥2 replicas
#   - For L2 test: LMCacheConnectorV1 + L2 daemon configured
#   - For L3 test: FSx mounted + LMCache FSx backend configured

set -euo pipefail

GATEWAY_URL="${1:?Usage: $0 <gateway-url> [namespace]}"
NAMESPACE="${2:-llmd-validation}"
MODEL="RedHatAI/Qwen3-32B-FP8-dynamic"

# A long system prompt (~2K tokens) to create meaningful cache entries.
# The key is that this prefix is identical across requests — if L2/L3 works,
# the second replica should find it cached instead of recomputing.
SYSTEM_PROMPT="You are an expert infrastructure engineer specializing in Kubernetes, GPU scheduling, and distributed systems. You have deep knowledge of vLLM, llm-d, NIXL, EFA, and KV cache architectures. When answering questions, provide detailed technical explanations with specific configuration examples. Always consider failure modes and recommend defensive configurations. Your responses should reference specific Kubernetes resources (Deployments, Services, ConfigMaps, CRDs) and include kubectl commands where appropriate. You understand the difference between HyperPod managed infrastructure and vanilla EKS, including the L2 tiered storage daemon on port 9200, the Inference Operator CRDs, KEDA autoscaling, and managed observability via AMP/AMG. You are familiar with the Gateway API Inference Extension, including InferencePool, EPP ext_proc filters, and scorer plugins like PrecisePrefixCacheScorer and LoadAwareScorer. When discussing KV cache, you distinguish between L0 (GPU HBM prefix cache), L1 (CPU DRAM via LMCache), L2 (per-node daemon via shared memory IPC), and L3 (cross-node via FSx Lustre or Redis). You understand that LMCache uses LMCacheConnectorV1 as the KV connector interface and that the L2 daemon communicates via /dev/shm/ai_toolkit_cache. For disaggregated inference, you know that NIXL uses libfabric over EFA for point-to-point KV transfers between prefill and decode pods, and that this requires the llm-d-aws container image. You always validate your suggestions against known blockers: NCCL bugs on Blackwell PCIe, LMCache NSA/MLA incompatibility, and Envoy ext_proc timeout defaults."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

get_vllm_pods() {
  kubectl get pods -n "$NAMESPACE" --no-headers -o custom-columns='NAME:.metadata.name,IP:.status.podIP,NODE:.spec.nodeName' 2>/dev/null | grep -i "vllm\|model-server" | head -10
}

get_pod_metric() {
  local pod_ip="$1"
  local metric="$2"
  kubectl exec -n "$NAMESPACE" deploy/bench-runner -- \
    curl -s "http://${pod_ip}:8000/metrics" 2>/dev/null | grep "^${metric}" | head -1 || true
}

# If no bench-runner pod, try direct port-forward or exec into a vllm pod
get_pod_metric_fallback() {
  local pod_name="$1"
  local metric="$2"
  kubectl exec -n "$NAMESPACE" "$pod_name" -- \
    curl -s "http://localhost:8000/metrics" 2>/dev/null | grep "^${metric}" | head -1 || true
}

send_request() {
  local url="$1"
  local user_msg="$2"
  local start_ms end_ms elapsed_ms response

  start_ms=$(python3 -c "import time; print(int(time.time()*1000))")
  response=$(curl -s --max-time 60 "${url}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
      --arg model "$MODEL" \
      --arg system "$SYSTEM_PROMPT" \
      --arg user "$user_msg" \
      '{
        model: $model,
        messages: [
          {role: "system", content: $system},
          {role: "user", content: $user}
        ],
        max_tokens: 32,
        temperature: 0
      }')" 2>/dev/null)
  end_ms=$(python3 -c "import time; print(int(time.time()*1000))")
  elapsed_ms=$((end_ms - start_ms))

  echo "$elapsed_ms"
}

send_request_to_pod() {
  local pod_name="$1"
  local pod_ip="$2"
  local user_msg="$3"
  local start_ms end_ms elapsed_ms

  start_ms=$(python3 -c "import time; print(int(time.time()*1000))")
  kubectl exec -n "$NAMESPACE" "$pod_name" -- \
    curl -s --max-time 60 "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
      --arg model "$MODEL" \
      --arg system "$SYSTEM_PROMPT" \
      --arg user "$user_msg" \
      '{
        model: $model,
        messages: [
          {role: "system", content: $system},
          {role: "user", content: $user}
        ],
        max_tokens: 32,
        temperature: 0
      }')" >/dev/null 2>&1
  end_ms=$(python3 -c "import time; print(int(time.time()*1000))")
  elapsed_ms=$((end_ms - start_ms))

  echo "$elapsed_ms"
}

# ---------------------------------------------------------------------------
# Test 1: Baseline — same pod, prefix cache (L0)
# ---------------------------------------------------------------------------

test_l0_prefix_cache() {
  echo "=== Test 1: L0 Prefix Cache (same pod, GPU memory) ==="
  echo ""

  local pods
  pods=$(get_vllm_pods)
  if [ -z "$pods" ]; then
    echo "  [FAIL] No vLLM pods found"
    return 1
  fi

  local pod_name pod_ip
  pod_name=$(echo "$pods" | head -1 | awk '{print $1}')
  pod_ip=$(echo "$pods" | head -1 | awk '{print $2}')
  echo "  Target pod: $pod_name ($pod_ip)"

  # Get cache metrics before
  local hits_before
  hits_before=$(get_pod_metric_fallback "$pod_name" "vllm:prefix_cache_hit_rate" | awk '{print $2}')
  echo "  Cache hit rate before: ${hits_before:-N/A}"

  # First request (cold — computes KV for system prompt)
  echo "  Sending request 1 (cold)..."
  local ttft1
  ttft1=$(send_request_to_pod "$pod_name" "$pod_ip" "What is an InferencePool?")
  echo "  TTFT (cold): ${ttft1}ms"

  # Second request (warm — same prefix, should hit L0 cache)
  echo "  Sending request 2 (warm, same prefix)..."
  local ttft2
  ttft2=$(send_request_to_pod "$pod_name" "$pod_ip" "How does EPP routing work?")
  echo "  TTFT (warm): ${ttft2}ms"

  # Get cache metrics after
  local hits_after
  hits_after=$(get_pod_metric_fallback "$pod_name" "vllm:prefix_cache_hit_rate" | awk '{print $2}')
  echo "  Cache hit rate after: ${hits_after:-N/A}"

  if [ -n "$ttft1" ] && [ -n "$ttft2" ] && [ "$ttft2" -lt "$ttft1" ]; then
    echo "  [PASS] L0 prefix cache working — warm TTFT ${ttft2}ms < cold TTFT ${ttft1}ms"
  else
    echo "  [INFO] TTFT comparison inconclusive (cold=${ttft1}ms, warm=${ttft2}ms)"
    echo "         Check cache metrics for confirmation"
  fi
  echo ""
}

# ---------------------------------------------------------------------------
# Test 2: L2 — cross-pod cache sharing via daemon
# ---------------------------------------------------------------------------

test_l2_cross_pod() {
  echo "=== Test 2: L2 Cross-Pod Cache Sharing (same node, L2 daemon) ==="
  echo ""

  local pods
  pods=$(get_vllm_pods)
  local pod_count
  pod_count=$(echo "$pods" | wc -l | tr -d ' ')

  if [ "$pod_count" -lt 2 ]; then
    echo "  [SKIP] Need ≥2 vLLM pods for cross-pod test (found $pod_count)"
    return 0
  fi

  local pod1_name pod1_ip pod1_node pod2_name pod2_ip pod2_node
  pod1_name=$(echo "$pods" | sed -n '1p' | awk '{print $1}')
  pod1_ip=$(echo "$pods" | sed -n '1p' | awk '{print $2}')
  pod1_node=$(echo "$pods" | sed -n '1p' | awk '{print $3}')
  pod2_name=$(echo "$pods" | sed -n '2p' | awk '{print $1}')
  pod2_ip=$(echo "$pods" | sed -n '2p' | awk '{print $2}')
  pod2_node=$(echo "$pods" | sed -n '2p' | awk '{print $3}')

  echo "  Pod A: $pod1_name on $pod1_node ($pod1_ip)"
  echo "  Pod B: $pod2_name on $pod2_node ($pod2_ip)"

  if [ "$pod1_node" = "$pod2_node" ]; then
    echo "  [INFO] Same node — L2 daemon sharing is possible"
  else
    echo "  [INFO] Different nodes — L2 test will show L3 (cross-node) behavior instead"
  fi

  # Check if LMCache connector is configured
  local kv_config
  kv_config=$(kubectl get pod -n "$NAMESPACE" "$pod1_name" -o jsonpath='{.spec.containers[0].args}' 2>/dev/null)
  if echo "$kv_config" | grep -q "LMCacheConnectorV1"; then
    echo "  [OK] LMCacheConnectorV1 configured"
  else
    echo "  [WARN] LMCacheConnectorV1 NOT configured — L2 sharing won't work"
    echo "         Apply Stage 4 config overlay first"
  fi

  # Check L2 daemon env var
  local l2_url
  l2_url=$(kubectl get pod -n "$NAMESPACE" "$pod1_name" -o jsonpath='{.spec.containers[0].env[?(@.name=="LMCACHE_REMOTE_URL")].value}' 2>/dev/null)
  if echo "$l2_url" | grep -q "sagemaker-hyperpod"; then
    echo "  [OK] LMCACHE_REMOTE_URL = $l2_url"
  else
    echo "  [WARN] LMCACHE_REMOTE_URL not set to L2 daemon"
  fi

  # Step 1: Warm Pod A's cache with the shared prefix
  echo ""
  echo "  Step 1: Warming Pod A cache..."
  local ttft_a
  ttft_a=$(send_request_to_pod "$pod1_name" "$pod1_ip" "Explain L2 tiered storage")
  echo "  Pod A TTFT (cold): ${ttft_a}ms"

  # Step 2: Send same prefix to Pod B — if L2 works, Pod B should find cached KV
  echo "  Step 2: Testing Pod B with same prefix..."
  local ttft_b
  ttft_b=$(send_request_to_pod "$pod2_name" "$pod2_ip" "Explain L2 tiered storage")
  echo "  Pod B TTFT: ${ttft_b}ms"

  # Step 3: Send same prefix to Pod B again (should definitely be warm now)
  echo "  Step 3: Pod B again (should be fully warm)..."
  local ttft_b2
  ttft_b2=$(send_request_to_pod "$pod2_name" "$pod2_ip" "How does the EPP scorer pipeline work?")
  echo "  Pod B TTFT (2nd): ${ttft_b2}ms"

  echo ""
  echo "  Results summary:"
  echo "    Pod A (cold):      ${ttft_a}ms"
  echo "    Pod B (L2 test):   ${ttft_b}ms"
  echo "    Pod B (2nd, warm): ${ttft_b2}ms"

  if [ -n "$ttft_a" ] && [ -n "$ttft_b" ]; then
    if [ "$ttft_b" -lt "$ttft_a" ]; then
      echo "  [PASS] L2 cross-pod sharing likely working — Pod B faster than Pod A cold"
    elif [ -n "$ttft_b2" ] && [ "$ttft_b2" -lt "$ttft_a" ]; then
      echo "  [PARTIAL] Pod B 2nd request faster — L2 may have async write delay"
    else
      echo "  [FAIL] No speedup on Pod B — L2 sharing may not be working"
      echo "         Check: L2 daemon logs, LMCache version compat, /dev/shm mount"
    fi
  fi
  echo ""
}

# ---------------------------------------------------------------------------
# Test 3: L3 — FSx Lustre file-based cache
# ---------------------------------------------------------------------------

test_l3_fsx() {
  echo "=== Test 3: L3 FSx Lustre Cache ==="
  echo ""

  local pods
  pods=$(get_vllm_pods)
  local pod1_name
  pod1_name=$(echo "$pods" | head -1 | awk '{print $1}')

  # Check FSx mount
  local fsx_mount
  fsx_mount=$(kubectl exec -n "$NAMESPACE" "$pod1_name" -- df -h /mnt/fsx 2>/dev/null || true)
  if [ -z "$fsx_mount" ]; then
    echo "  [SKIP] FSx not mounted at /mnt/fsx — apply Stage 5 config first"
    return 0
  fi
  echo "  [OK] FSx mounted"

  # Check for kvcache directory
  local kv_dir
  kv_dir=$(kubectl exec -n "$NAMESPACE" "$pod1_name" -- ls /mnt/fsx/kvcache/ 2>/dev/null | wc -l || echo "0")
  echo "  KV cache files before: $kv_dir"

  # Send a request to generate cache
  echo "  Sending request to generate KV cache..."
  send_request_to_pod "$pod1_name" "" "What is FSx Lustre?" >/dev/null 2>&1

  # Wait briefly for async write
  sleep 3

  # Check again
  local kv_dir_after
  kv_dir_after=$(kubectl exec -n "$NAMESPACE" "$pod1_name" -- ls /mnt/fsx/kvcache/ 2>/dev/null | wc -l || echo "0")
  echo "  KV cache files after: $kv_dir_after"

  if [ "$kv_dir_after" -gt "$kv_dir" ]; then
    echo "  [PASS] New KV cache files written to FSx"
  else
    echo "  [INFO] No new files — LMCache may not be configured for FSx backend"
    echo "         Check LMCACHE_REMOTE_URL includes file:///mnt/fsx/kvcache"
  fi

  # Cross-pod read test
  local pod_count
  pod_count=$(echo "$pods" | wc -l | tr -d ' ')
  if [ "$pod_count" -ge 2 ]; then
    local pod2_name
    pod2_name=$(echo "$pods" | sed -n '2p' | awk '{print $1}')
    local kv_visible
    kv_visible=$(kubectl exec -n "$NAMESPACE" "$pod2_name" -- ls /mnt/fsx/kvcache/ 2>/dev/null | wc -l || echo "0")
    if [ "$kv_visible" -gt 0 ]; then
      echo "  [PASS] Pod B can see $kv_visible KV cache files on FSx (cross-pod visible)"
    else
      echo "  [INFO] Pod B sees 0 files on FSx"
    fi
  fi
  echo ""
}

# ---------------------------------------------------------------------------
# Test 4: EPP routing with cache awareness
# ---------------------------------------------------------------------------

test_epp_routing() {
  echo "=== Test 4: EPP Prefix-Aware Routing ==="
  echo ""

  # Send same prefix twice via gateway — EPP should route to same pod
  echo "  Sending request 1 via gateway..."
  local ttft1
  ttft1=$(send_request "$GATEWAY_URL" "What is prefix caching?")
  echo "  TTFT: ${ttft1}ms"

  echo "  Sending request 2 (same prefix) via gateway..."
  local ttft2
  ttft2=$(send_request "$GATEWAY_URL" "How does prefix caching improve latency?")
  echo "  TTFT: ${ttft2}ms"

  echo "  Sending request 3 (same prefix) via gateway..."
  local ttft3
  ttft3=$(send_request "$GATEWAY_URL" "What are the cache tiers?")
  echo "  TTFT: ${ttft3}ms"

  echo ""
  echo "  Results: ${ttft1}ms → ${ttft2}ms → ${ttft3}ms"
  if [ -n "$ttft2" ] && [ -n "$ttft1" ] && [ "$ttft2" -lt "$ttft1" ]; then
    echo "  [PASS] TTFT decreasing — EPP likely routing to cached pod"
  else
    echo "  [INFO] TTFT not clearly decreasing — check EPP logs for routing decisions"
  fi

  # Check EPP logs for routing evidence
  local epp_pod
  epp_pod=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | grep "epp" | head -1 | awk '{print $1}')
  if [ -n "$epp_pod" ]; then
    echo ""
    echo "  EPP recent routing decisions:"
    kubectl logs -n "$NAMESPACE" "$epp_pod" --tail=10 2>/dev/null | grep -i "route\|score\|select\|prefix\|cache" | tail -5 || echo "    (no routing logs found)"
  fi
  echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "=============================================="
echo "llm-d on HyperPod — Cache Integration Test"
echo "=============================================="
echo "Gateway: $GATEWAY_URL"
echo "Namespace: $NAMESPACE"
echo "Model: $MODEL"
echo "Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

echo "Discovering vLLM pods..."
get_vllm_pods
echo ""

test_l0_prefix_cache
test_l2_cross_pod
test_l3_fsx
test_epp_routing

echo "=============================================="
echo "Integration test complete."
echo "=============================================="
