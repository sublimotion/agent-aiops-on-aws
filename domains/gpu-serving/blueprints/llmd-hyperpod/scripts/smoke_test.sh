#!/bin/bash
# llm-d on HyperPod — Integration Smoke Test
# Usage: ./smoke_test.sh [stage]
# Runs all stages if no argument, or a specific stage (1-7)

set -uo pipefail

NAMESPACE="${NAMESPACE:-llmd-validation}"
GATEWAY_NS="${GATEWAY_NS:-istio-system}"
HP_NS="hyperpod-inference-system"
PASS=0
FAIL=0
SKIP=0

pass() { echo "  [PASS] $1"; ((PASS++)); }
fail() { echo "  [FAIL] $1"; ((FAIL++)); }
skip() { echo "  [SKIP] $1"; ((SKIP++)); }

header() { echo -e "\n=== Stage $1: $2 ==="; }

# ---------- Stage 1: Infrastructure Discovery ----------
stage1() {
  header 1 "HyperPod Infrastructure Discovery"

  # Existing workloads (isolation check)
  EXISTING_GPU_PODS=$(kubectl get pods -A --no-headers -o custom-columns='NS:.metadata.namespace,NAME:.metadata.name' 2>/dev/null | grep -v "$NAMESPACE" | grep -v "kube-system\|istio-system\|$HP_NS" || true)
  if [ -n "$EXISTING_GPU_PODS" ]; then
    echo "  [INFO] Existing pods on cluster (verify no GPU conflict):"
    echo "$EXISTING_GPU_PODS" | head -5 | sed 's/^/    /'
    GPU_USED=$(kubectl get pods -A -o jsonpath='{range .items[*]}{.spec.containers[*].resources.limits.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | awk '{s+=$1} END {print s+0}')
    echo "  [INFO] Total GPUs in use across cluster: $GPU_USED"
  else
    pass "No existing workload pods found outside system namespaces"
  fi

  # L2 daemon (ai-toolkit runs as a DaemonSet in aws-hyperpod namespace)
  if kubectl get pods -n aws-hyperpod 2>/dev/null | grep -qi "ai-toolkit"; then
    pass "L2 ai-toolkit daemon running"
  elif kubectl get ds -A 2>/dev/null | grep -qi "l2\|tiered\|ai-toolkit"; then
    pass "L2 daemon DaemonSet found"
  else
    skip "L2 daemon not found"
  fi

  # Inference Operator
  if kubectl get pods -n "$HP_NS" 2>/dev/null | grep -q "controller-manager"; then
    pass "Inference Operator controller-manager running"
  else
    skip "Inference Operator not found (may not be installed yet)"
  fi

  # ADOT collector
  if kubectl get pods -A 2>/dev/null | grep -qi "adot\|otel"; then
    pass "ADOT/OpenTelemetry collector found"
  else
    skip "ADOT collector not found"
  fi

  # FSx CSI
  if kubectl get csidriver 2>/dev/null | grep -q "fsx.csi.aws.com"; then
    pass "FSx CSI driver registered"
  else
    skip "FSx CSI driver not found"
  fi

  # GPU nodes (HyperPod labels GPUs via nvidia.com/gpu resource, not a label)
  GPU_COUNT=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | awk '$1>0' | wc -l | tr -d ' ')
  if [ "$GPU_COUNT" -gt 0 ]; then
    pass "Found $GPU_COUNT GPU node(s)"
  else
    fail "No GPU nodes found"
  fi

  # System node taints (document for EPP tolerations)
  echo "  [INFO] System node taints:"
  kubectl get nodes -l node-role.kubernetes.io/system=true -o jsonpath='{range .items[*]}{.metadata.name}: {.spec.taints}{"\n"}{end}' 2>/dev/null || echo "    (no system nodes found by label)"
}

# ---------- Stage 2: Gateway Stack ----------
stage2() {
  header 2 "llm-d Gateway Stack"

  # Gateway API CRDs
  if kubectl get crd gateways.gateway.networking.k8s.io &>/dev/null; then
    pass "Gateway API CRDs installed"
  else
    fail "Gateway API CRDs not found — run install-gateway-provider-dependencies.sh"
  fi

  # Istio control plane
  if kubectl get pods -n "$GATEWAY_NS" 2>/dev/null | grep -q "istiod"; then
    pass "Istio control plane running"
  else
    skip "Istio not found (may use KGateway instead)"
  fi

  # Gateway resource
  GW_STATUS=$(kubectl get gateways -A -o jsonpath='{.items[0].status.conditions[?(@.type=="Programmed")].status}' 2>/dev/null || true)
  if [ "$GW_STATUS" = "True" ]; then
    pass "Gateway resource Programmed"
  elif [ -n "$GW_STATUS" ]; then
    fail "Gateway exists but not Programmed (status: $GW_STATUS)"
  else
    fail "No Gateway resource found"
  fi

  # InferencePool CRD
  if kubectl get crd inferencepools.inference.networking.k8s.io &>/dev/null; then
    pass "InferencePool CRD installed"
  else
    skip "InferencePool CRD not found (may be installed with EPP)"
  fi

  # Check for IEC CRD coexistence
  if kubectl get crd inferenceendpointconfigs.inference.sagemaker.aws.amazon.com &>/dev/null; then
    pass "IEC CRD coexists with Gateway API CRDs"
  else
    skip "IEC CRD not found (operator may not be installed)"
  fi
}

# ---------- Stage 3: vLLM Baseline ----------
stage3() {
  header 3 "vLLM Deployment via llm-d (Baseline)"

  # vLLM pods (llm-d labels with llm-d.ai/role=decode)
  VLLM_PODS=$(kubectl get pods -n "$NAMESPACE" -l llm-d.ai/role=decode --no-headers 2>/dev/null | grep -c "Running" || true)
  if [ "$VLLM_PODS" -gt 0 ]; then
    pass "Found $VLLM_PODS vLLM decode pod(s) Running"
  else
    VLLM_PODS=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | grep -c "decode.*Running" || true)
    if [ "$VLLM_PODS" -gt 0 ]; then
      pass "Found $VLLM_PODS vLLM pod(s) Running (matched by name)"
    else
      fail "No vLLM pods found Running"
      return
    fi
  fi

  # Health check
  VLLM_POD=$(kubectl get pods -n "$NAMESPACE" -l llm-d.ai/role=decode --no-headers 2>/dev/null | grep "Running" | head -1 | awk '{print $1}')
  if [ -n "$VLLM_POD" ]; then
    HEALTH=$(kubectl exec -n "$NAMESPACE" "$VLLM_POD" -- curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || true)
    if [ "$HEALTH" = "200" ]; then
      pass "vLLM health endpoint returns 200"
    else
      fail "vLLM health check failed (HTTP $HEALTH)"
    fi

    # Prefix cache metric
    PREFIX_METRIC=$(kubectl exec -n "$NAMESPACE" "$VLLM_POD" -- bash -c 'curl -s http://localhost:8000/metrics | grep "^vllm:prefix_cache_queries_total"' 2>/dev/null || true)
    if [ -n "$PREFIX_METRIC" ]; then
      pass "Prefix cache metrics exposed ($PREFIX_METRIC)"
    else
      skip "Prefix cache metrics not found"
    fi
  fi

  # EPP
  EPP_PODS=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | grep -c "epp.*Running" || true)
  if [ "$EPP_PODS" -gt 0 ]; then
    pass "EPP pod(s) Running"
  else
    skip "EPP pods not found"
  fi

  # Chat completion via gateway (port-forward test)
  # Get served model name directly from vLLM /v1/models endpoint
  MODEL_NAME=$(kubectl exec -n "$NAMESPACE" "$VLLM_POD" -- curl -s http://localhost:8000/v1/models 2>/dev/null | jq -r '.data[0].id' 2>/dev/null || true)
  GW_SVC=$(kubectl get svc -n "$NAMESPACE" -l app.kubernetes.io/component=inference-gateway --no-headers 2>/dev/null | head -1 | awk '{print $1}')
  if [ -n "$GW_SVC" ] && [ -n "$MODEL_NAME" ]; then
    # Port-forward in background, test, then clean up
    kubectl port-forward -n "$NAMESPACE" "svc/$GW_SVC" 18888:80 &>/dev/null &
    PF_PID=$!
    sleep 5
    RESP=$(curl -s --max-time 30 "http://localhost:18888/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}],\"max_tokens\":16}" 2>/dev/null || true)
    kill $PF_PID 2>/dev/null || true
    if echo "$RESP" | jq -e '.choices[0].message.content' &>/dev/null; then
      pass "Chat completion works via Gateway (model=$MODEL_NAME)"
    else
      fail "Chat completion via Gateway failed: $RESP"
    fi
  else
    skip "No Gateway service or model found"
  fi
}

# ---------- Stage 4: L2 Integration ----------
stage4() {
  header 4 "L2 Managed Tiered Storage Integration (CRITICAL)"

  VLLM_POD=$(kubectl get pods -n "$NAMESPACE" -l llm-d.ai/role=decode --no-headers 2>/dev/null | grep "Running" | head -1 | awk '{print $1}')
  if [ -z "$VLLM_POD" ]; then
    fail "No vLLM pod found — run Stage 3 first"
    return
  fi

  # Check if LMCache connector is configured
  KV_CONFIG=$(kubectl get pod -n "$NAMESPACE" "$VLLM_POD" -o jsonpath='{.spec.containers[0].args}' 2>/dev/null || true)
  if echo "$KV_CONFIG" | grep -q "LMCacheConnectorV1"; then
    pass "LMCacheConnectorV1 configured in vLLM args"
  else
    fail "LMCacheConnectorV1 not configured — apply L2 overlay"
  fi

  # Check LMCACHE env vars
  LMCACHE_URL=$(kubectl get pod -n "$NAMESPACE" "$VLLM_POD" -o jsonpath='{.spec.containers[0].env[?(@.name=="LMCACHE_REMOTE_URL")].value}' 2>/dev/null || true)
  if echo "$LMCACHE_URL" | grep -q "sagemaker-hyperpod"; then
    pass "LMCACHE_REMOTE_URL points to L2 daemon ($LMCACHE_URL)"
  else
    fail "LMCACHE_REMOTE_URL not set to sagemaker-hyperpod URL"
  fi

  # Check /dev/shm hostPath mount (shares POSIX shm with ai-toolkit daemon)
  SHM_MOUNT=$(kubectl get pod -n "$NAMESPACE" "$VLLM_POD" -o jsonpath='{.spec.volumes[*].hostPath.path}' 2>/dev/null || true)
  if echo "$SHM_MOUNT" | grep -q "/dev/shm"; then
    pass "/dev/shm hostPath mounted (shared with ai-toolkit)"
  else
    fail "/dev/shm hostPath not mounted — L2 shared memory IPC won't work"
  fi

  # Check LMCache version
  LMCACHE_VER=$(kubectl exec -n "$NAMESPACE" "$VLLM_POD" -- pip show lmcache 2>/dev/null | grep "Version:" || true)
  if [ -n "$LMCACHE_VER" ]; then
    pass "LMCache installed: $LMCACHE_VER"
  else
    skip "Could not determine LMCache version"
  fi

  # Check vLLM logs for L2 connection
  if kubectl logs -n "$NAMESPACE" "$VLLM_POD" 2>/dev/null | grep -q "SageMaker HyperPod connector created successfully"; then
    pass "LMCache SageMaker HyperPod connector connected"
  elif kubectl logs -n "$NAMESPACE" "$VLLM_POD" 2>/dev/null | grep -q "Shared memory opened"; then
    pass "LMCache shared memory opened"
  elif kubectl logs -n "$NAMESPACE" "$VLLM_POD" 2>/dev/null | grep -q "LMCacheConnectorV1"; then
    pass "vLLM logs show LMCache connector initialization"
  else
    skip "No LMCache connection in logs"
  fi

  # Check L2 health — external prefix cache queries should be > 0 if connected
  EXT_QUERIES=$(kubectl exec -n "$NAMESPACE" "$VLLM_POD" -- bash -c 'curl -s localhost:8000/metrics | grep "^vllm:external_prefix_cache_queries_total"' 2>/dev/null | awk '{print $2}' || true)
  if [ -n "$EXT_QUERIES" ] && [ "$(echo "$EXT_QUERIES > 0" | bc 2>/dev/null || echo 0)" = "1" ]; then
    pass "L2 external cache queries active ($EXT_QUERIES queries)"
  elif [ -n "$EXT_QUERIES" ]; then
    skip "External prefix queries = $EXT_QUERIES (send requests to populate)"
  else
    skip "Could not read external prefix cache metrics"
  fi

  echo ""
  echo "  [ACTION] Manual smoke test for cross-pod KV sharing:"
  echo "    1. Send: curl <gw>/v1/chat/completions -d '{\"messages\":[{\"role\":\"system\",\"content\":\"You are a helpful assistant for code review.\"},{\"role\":\"user\",\"content\":\"Hello\"}]}'"
  echo "    2. Send same request again"
  echo "    3. Check EPP logs: second request should route to same pod (prefix hit)"
  echo "    4. Force route to different pod and compare TTFT (should be faster if L2 sharing works)"
}

# ---------- Stage 5: L3 Integration ----------
stage5() {
  header 5 "L3 FSx Lustre Cross-Node Cache"

  VLLM_POD=$(kubectl get pods -n "$NAMESPACE" -l llm-d.ai/role=decode --no-headers 2>/dev/null | grep "Running" | head -1 | awk '{print $1}')
  if [ -z "$VLLM_POD" ]; then
    fail "No vLLM pod found"
    return
  fi

  # FSx PVC
  if kubectl get pvc -n "$NAMESPACE" 2>/dev/null | grep -qi "fsx"; then
    pass "FSx PVC found"
  else
    fail "No FSx PVC — provision FSx Lustre PV first"
  fi

  # FSx mount
  FSX_MOUNT=$(kubectl exec -n "$NAMESPACE" "$VLLM_POD" -- df -h /mnt/fsx 2>/dev/null || true)
  if [ -n "$FSX_MOUNT" ]; then
    pass "FSx mounted at /mnt/fsx"
  else
    fail "FSx not mounted in vLLM pod"
  fi

  # Check for KV cache files
  KV_FILES=$(kubectl exec -n "$NAMESPACE" "$VLLM_POD" -- ls /mnt/fsx/kvcache/ 2>/dev/null | wc -l || true)
  if [ "$KV_FILES" -gt 0 ]; then
    pass "Found $KV_FILES KV cache files on FSx"
  else
    skip "No KV cache files on FSx yet (run inference first)"
  fi
}

# ---------- Stage 6: Observability ----------
stage6() {
  header 6 "Observability Integration"

  echo "  [INFO] Observability checks require AMP/AMG console access."
  echo "  [ACTION] Verify in Managed Grafana:"
  echo "    1. vLLM metrics (vllm_*) visible in AMP"
  echo "    2. EPP metrics visible in AMP"
  echo "    3. DCGM GPU metrics visible"
  echo "    4. Pre-built dashboards render with data"

  # Check ServiceMonitor
  if kubectl get servicemonitor -n "$NAMESPACE" 2>/dev/null | grep -qi "vllm\|llm"; then
    pass "ServiceMonitor for vLLM found"
  else
    skip "No ServiceMonitor found — ADOT may use pod annotation scraping"
  fi
}

# ---------- Stage 7: CRD Coexistence ----------
stage7() {
  header 7 "CRD Coexistence"

  # Check both CRD types exist
  GATEWAY_CRDS=$(kubectl get crd 2>/dev/null | grep -c "inference.networking.k8s.io" || true)
  IEC_CRDS=$(kubectl get crd 2>/dev/null | grep -c "inference.sagemaker.aws.amazon.com" || true)

  if [ "$GATEWAY_CRDS" -gt 0 ] && [ "$IEC_CRDS" -gt 0 ]; then
    pass "Both Gateway API Inference CRDs ($GATEWAY_CRDS) and IEC CRDs ($IEC_CRDS) coexist"
  elif [ "$GATEWAY_CRDS" -gt 0 ]; then
    skip "Only Gateway API CRDs found (IEC not installed)"
  elif [ "$IEC_CRDS" -gt 0 ]; then
    skip "Only IEC CRDs found (Gateway API not installed)"
  else
    fail "Neither CRD set found"
  fi

  # Check operator logs for interference (filter out known EnableFailed errors)
  OPERATOR_ERRORS=$(kubectl logs -n "$HP_NS" deployment/hyperpod-inference-controller-manager --tail=50 2>/dev/null | grep -i "error\|conflict" | grep -v "EnableFailed\|EnableClusterInference\|watcher" || true)
  if [ -n "$OPERATOR_ERRORS" ]; then
    fail "Operator logs show unexpected errors"
    echo "$OPERATOR_ERRORS" | head -3 | sed 's/^/    /'
  else
    pass "No unexpected errors in operator logs (EnableFailed is known/non-blocking)"
  fi
}

# ---------- Main ----------
echo "============================================"
echo "llm-d on HyperPod — Integration Smoke Test"
echo "============================================"
echo "Namespace: $NAMESPACE"
echo "Gateway NS: $GATEWAY_NS"
echo "Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

STAGE="${1:-all}"

case "$STAGE" in
  1) stage1 ;;
  2) stage2 ;;
  3) stage3 ;;
  4) stage4 ;;
  5) stage5 ;;
  6) stage6 ;;
  7) stage7 ;;
  all)
    stage1; stage2; stage3; stage4; stage5; stage6; stage7
    ;;
  *) echo "Usage: $0 [1-7|all]"; exit 1 ;;
esac

echo ""
echo "============================================"
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
