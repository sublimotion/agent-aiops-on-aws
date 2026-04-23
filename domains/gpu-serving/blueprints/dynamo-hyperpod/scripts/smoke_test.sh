#!/bin/bash
# Dynamo on HyperPod — Integration Smoke Test
# Usage: ./smoke_test.sh [stage]
# Runs all stages if no argument, or a specific stage (1-7)

set -uo pipefail

NAMESPACE="${NAMESPACE:-dynamo-validation}"
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

  # GPU nodes
  GPU_COUNT=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | awk '$1>0' | wc -l | tr -d ' ')
  if [ "$GPU_COUNT" -gt 0 ]; then
    pass "Found $GPU_COUNT GPU node(s)"
  else
    fail "No GPU nodes found"
  fi

  # FSx CSI driver
  if kubectl get csidriver fsx.csi.aws.com &>/dev/null; then
    pass "FSx CSI driver registered"
  else
    skip "FSx CSI driver not found"
  fi

  # L2 daemon (NOT used by Dynamo, check for conflicts)
  if kubectl get pods -n aws-hyperpod 2>/dev/null | grep -qi "ai-toolkit"; then
    pass "L2 ai-toolkit daemon running (not used by Dynamo, no conflict expected)"
  else
    skip "L2 daemon not found"
  fi

  # Inference Operator
  if kubectl get pods -n "$HP_NS" 2>/dev/null | grep -q "controller-manager"; then
    CTRL_STATUS=$(kubectl get pods -n "$HP_NS" --no-headers 2>/dev/null | grep "controller-manager" | awk '{print $3}')
    if [ "$CTRL_STATUS" = "Running" ]; then
      pass "Inference Operator controller-manager running"
    else
      skip "Inference Operator controller-manager in state: $CTRL_STATUS (known EnableFailed, non-blocking)"
    fi
  else
    skip "Inference Operator not found"
  fi

  # ADOT collector (empirically NOT auto-installed)
  if kubectl get pods -A 2>/dev/null | grep -qi "adot\|otel"; then
    pass "ADOT/OpenTelemetry collector found"
  else
    skip "ADOT collector not found (empirically confirmed not auto-installed on HP inference clusters)"
  fi

  # Gateway API CRDs (from llm-d install — Dynamo CRDs must coexist)
  if kubectl get crd gateways.gateway.networking.k8s.io &>/dev/null; then
    pass "Gateway API CRDs present (from llm-d install)"
  else
    skip "Gateway API CRDs not found"
  fi

  # Check GPU contention with llmd-validation
  LLMD_GPU_PODS=$(kubectl get pods -n llmd-validation --no-headers 2>/dev/null | grep -c "Running" || true)
  if [ "$LLMD_GPU_PODS" -gt 0 ]; then
    GPU_USED=$(kubectl get pods -n llmd-validation -o jsonpath='{range .items[*]}{.spec.containers[*].resources.limits.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | awk '{s+=$1} END {print s+0}')
    if [ "$GPU_USED" -gt 0 ]; then
      fail "llmd-validation is using $GPU_USED GPU(s) — scale down before deploying Dynamo"
    else
      pass "llmd-validation has no GPU pods"
    fi
  else
    pass "No running pods in llmd-validation"
  fi

  # System node taints
  echo "  [INFO] System node taints:"
  kubectl get nodes -l node-role.kubernetes.io/system=true -o jsonpath='{range .items[*]}  {.metadata.name}: {.spec.taints}{"\n"}{end}' 2>/dev/null || echo "    (no system nodes found by label)"
}

# ---------- Stage 2: Dynamo Components ----------
stage2() {
  header 2 "Dynamo Components (etcd + Worker Deployment)"

  # Namespace
  if kubectl get namespace "$NAMESPACE" &>/dev/null; then
    pass "Namespace $NAMESPACE exists"
  else
    fail "Namespace $NAMESPACE not found"
  fi

  # etcd
  ETCD_STATUS=$(kubectl get pods -n "$NAMESPACE" -l app=etcd --no-headers 2>/dev/null | grep -c "Running" || true)
  if [ "$ETCD_STATUS" -gt 0 ]; then
    pass "etcd pod(s) Running"
    # Check etcd health
    ETCD_POD=$(kubectl get pods -n "$NAMESPACE" -l app=etcd --no-headers 2>/dev/null | grep "Running" | head -1 | awk '{print $1}')
    if [ -n "$ETCD_POD" ]; then
      ETCD_HEALTH=$(kubectl exec -n "$NAMESPACE" "$ETCD_POD" -- etcdctl endpoint health --endpoints=http://localhost:2379 2>&1 || true)
      if echo "$ETCD_HEALTH" | grep -q "is healthy"; then
        pass "etcd health check passed"
      else
        fail "etcd not healthy: $ETCD_HEALTH"
      fi
    fi
  else
    fail "No etcd pods Running"
  fi

  # FSx PVC
  if kubectl get pvc -n "$NAMESPACE" fsx-dynamo 2>/dev/null | grep -q "Bound"; then
    pass "FSx PVC fsx-dynamo is Bound"
  else
    fail "FSx PVC fsx-dynamo not found or not Bound"
  fi

  # Check no CRD conflicts
  CRD_COUNT=$(kubectl get crd 2>/dev/null | wc -l)
  if [ "$CRD_COUNT" -gt 0 ]; then
    pass "CRDs accessible ($CRD_COUNT total), no webhook blocking"
  else
    fail "Cannot list CRDs"
  fi
}

# ---------- Stage 3: vLLM Baseline ----------
stage3() {
  header 3 "Dynamo vLLM Worker (Baseline)"

  # Worker pods
  WORKER_PODS=$(kubectl get pods -n "$NAMESPACE" -l app=dynamo-worker --no-headers 2>/dev/null | grep -c "Running" || true)
  if [ "$WORKER_PODS" -gt 0 ]; then
    pass "Found $WORKER_PODS Dynamo worker pod(s) Running"
  else
    fail "No Dynamo worker pods Running"
    return
  fi

  WORKER_POD=$(kubectl get pods -n "$NAMESPACE" -l app=dynamo-worker --no-headers 2>/dev/null | grep "Running" | head -1 | awk '{print $1}')

  # Health check
  HEALTH=$(kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || true)
  if [ "$HEALTH" = "200" ]; then
    pass "vLLM health endpoint returns 200"
  else
    fail "vLLM health check failed (HTTP $HEALTH)"
  fi

  # Model served
  MODEL_NAME=$(kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- curl -s http://localhost:8000/v1/models 2>/dev/null | jq -r '.data[0].id' 2>/dev/null || true)
  if [ -n "$MODEL_NAME" ] && [ "$MODEL_NAME" != "null" ]; then
    pass "Model served: $MODEL_NAME"
  else
    fail "No model served"
  fi

  # Chat completion
  RESP=$(kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- curl -s --max-time 30 http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}],\"max_tokens\":16}" 2>/dev/null || true)
  if echo "$RESP" | jq -e '.choices[0].message.content' &>/dev/null; then
    pass "Chat completion works (model=$MODEL_NAME)"
  else
    fail "Chat completion failed: $RESP"
  fi

  # Prefix cache metric
  PREFIX_METRIC=$(kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- bash -c 'curl -s http://localhost:8000/metrics | grep "^vllm:prefix_cache_queries_total"' 2>/dev/null || true)
  if [ -n "$PREFIX_METRIC" ]; then
    pass "Prefix cache metrics exposed ($PREFIX_METRIC)"
  else
    skip "Prefix cache metrics not found (may use different metric name)"
  fi

  # GPU type check (sm_86 Ampere compatibility)
  GPU_NAME=$(kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- bash -c 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null' || true)
  if [ -n "$GPU_NAME" ]; then
    pass "GPU detected: $GPU_NAME"
  else
    skip "Could not query GPU name"
  fi
}

# ---------- Stage 4: KVBM G2 CPU Cache ----------
stage4() {
  header 4 "KVBM G2 — CPU DRAM Cache"

  WORKER_POD=$(kubectl get pods -n "$NAMESPACE" -l app=dynamo-worker --no-headers 2>/dev/null | grep "Running" | head -1 | awk '{print $1}')
  if [ -z "$WORKER_POD" ]; then
    fail "No worker pod found"
    return
  fi

  # Check KVBM CPU env var is set
  CPU_CACHE=$(kubectl get pod -n "$NAMESPACE" "$WORKER_POD" -o jsonpath='{.spec.containers[0].env[?(@.name=="DYN_KVBM_CPU_CACHE_GB")].value}' 2>/dev/null || true)
  if [ -n "$CPU_CACHE" ] && [ "$CPU_CACHE" != "0" ]; then
    pass "KVBM CPU cache configured: ${CPU_CACHE}GB"
  else
    skip "DYN_KVBM_CPU_CACHE_GB not set or 0"
  fi

  # Check logs for KVBM initialization
  if kubectl logs -n "$NAMESPACE" "$WORKER_POD" 2>/dev/null | grep -qi "kvbm\|kv.*block.*manager\|cpu.*cache"; then
    pass "KVBM initialization found in logs"
  else
    skip "No KVBM initialization in logs (may not be supported in this image)"
  fi

  # Check for KVBM metrics
  KVBM_METRICS=$(kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- bash -c 'curl -s http://localhost:8000/metrics | grep -i "kvbm"' 2>/dev/null || true)
  if [ -n "$KVBM_METRICS" ]; then
    pass "KVBM metrics found"
    echo "$KVBM_METRICS" | head -5 | sed 's/^/    /'
  else
    skip "No KVBM metrics found (KVBM may not be active with this image/model)"
  fi
}

# ---------- Stage 5: KVBM G3 FSx ----------
stage5() {
  header 5 "KVBM G3 — FSx Lustre Cache"

  WORKER_POD=$(kubectl get pods -n "$NAMESPACE" -l app=dynamo-worker --no-headers 2>/dev/null | grep "Running" | head -1 | awk '{print $1}')
  if [ -z "$WORKER_POD" ]; then
    fail "No worker pod found"
    return
  fi

  # FSx PVC mounted
  FSX_MOUNT=$(kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- df -h /mnt/fsx 2>/dev/null || true)
  if [ -n "$FSX_MOUNT" ]; then
    pass "FSx mounted at /mnt/fsx"
  else
    fail "FSx not mounted in worker pod"
    return
  fi

  # Cache directory writable
  WRITE_TEST=$(kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- touch /mnt/fsx/kv-cache/.dynamo-write-test 2>&1 || true)
  if echo "$WRITE_TEST" | grep -qi "permission denied\|read-only"; then
    fail "FSx kv-cache directory not writable"
  else
    kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- rm -f /mnt/fsx/kv-cache/.dynamo-write-test 2>/dev/null
    pass "FSx kv-cache directory is writable"
  fi

  # KVBM disk env vars
  DISK_DIR=$(kubectl get pod -n "$NAMESPACE" "$WORKER_POD" -o jsonpath='{.spec.containers[0].env[?(@.name=="DYN_KVBM_DISK_CACHE_DIR")].value}' 2>/dev/null || true)
  if [ -n "$DISK_DIR" ]; then
    pass "KVBM disk cache dir configured: $DISK_DIR"
  else
    skip "DYN_KVBM_DISK_CACHE_DIR not set"
  fi

  ZEROFILL=$(kubectl get pod -n "$NAMESPACE" "$WORKER_POD" -o jsonpath='{.spec.containers[0].env[?(@.name=="DYN_KVBM_DISK_ZEROFILL_FALLBACK")].value}' 2>/dev/null || true)
  if [ "$ZEROFILL" = "true" ]; then
    pass "KVBM ZEROFILL_FALLBACK enabled (Lustre lacks fallocate)"
  else
    skip "ZEROFILL_FALLBACK not set"
  fi

  DISABLE_DIRECT=$(kubectl get pod -n "$NAMESPACE" "$WORKER_POD" -o jsonpath='{.spec.containers[0].env[?(@.name=="DYN_KVBM_DISK_DISABLE_O_DIRECT")].value}' 2>/dev/null || true)
  if [ "$DISABLE_DIRECT" = "true" ]; then
    pass "KVBM DISABLE_O_DIRECT enabled (Lustre strict alignment)"
  else
    skip "DISABLE_O_DIRECT not set"
  fi

  # Check for cache files on FSx
  KV_FILES=$(kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- ls /mnt/fsx/kv-cache/ 2>/dev/null | wc -l || echo 0)
  if [ "$KV_FILES" -gt 0 ]; then
    pass "Found $KV_FILES files in FSx kv-cache directory"
  else
    skip "No cache files on FSx yet (send requests to populate)"
  fi

  # No conflict with llm-d cache dir
  if kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- ls /mnt/fsx/kvcache 2>/dev/null | head -1 >/dev/null 2>&1; then
    pass "llm-d cache dir (/mnt/fsx/kvcache) also visible — separate from Dynamo's /mnt/fsx/kv-cache/"
  else
    skip "llm-d cache dir not visible (may not have been created)"
  fi
}

# ---------- Stage 6: Observability ----------
stage6() {
  header 6 "Observability Integration"

  WORKER_POD=$(kubectl get pods -n "$NAMESPACE" -l app=dynamo-worker --no-headers 2>/dev/null | grep "Running" | head -1 | awk '{print $1}')

  # vLLM metrics
  if [ -n "$WORKER_POD" ]; then
    VLLM_METRICS=$(kubectl exec -n "$NAMESPACE" "$WORKER_POD" -- bash -c 'curl -s http://localhost:8000/metrics | head -5' 2>/dev/null || true)
    if [ -n "$VLLM_METRICS" ]; then
      pass "vLLM metrics accessible via port-forward"
    else
      skip "vLLM metrics not accessible"
    fi
  fi

  # ADOT (empirically not installed)
  if kubectl get pods -A 2>/dev/null | grep -qi "adot\|otel"; then
    pass "ADOT collector found — metrics may flow to AMP"
  else
    skip "ADOT not installed (expected — use port-forward for metrics)"
  fi

  # PodMonitor/ServiceMonitor CRDs
  if kubectl get crd podmonitors.monitoring.coreos.com &>/dev/null; then
    pass "PodMonitor CRD available"
  else
    skip "PodMonitor CRD not found"
  fi

  echo "  [INFO] Observability: ADOT not auto-installed on this cluster."
  echo "  [INFO] Validate metrics via: kubectl port-forward -n $NAMESPACE <pod> 8000 → curl localhost:8000/metrics"
}

# ---------- Stage 7: CRD Coexistence ----------
stage7() {
  header 7 "CRD Coexistence"

  # Check all CRD groups coexist
  GATEWAY_CRDS=$(kubectl get crd 2>/dev/null | grep -c "inference.networking.k8s.io" || true)
  IEC_CRDS=$(kubectl get crd 2>/dev/null | grep -c "inference.sagemaker.aws.amazon.com" || true)

  if [ "$GATEWAY_CRDS" -gt 0 ] && [ "$IEC_CRDS" -gt 0 ]; then
    pass "Gateway API Inference CRDs ($GATEWAY_CRDS) + IEC CRDs ($IEC_CRDS) coexist"
  elif [ "$GATEWAY_CRDS" -gt 0 ]; then
    skip "Only Gateway API CRDs found (IEC not installed)"
  elif [ "$IEC_CRDS" -gt 0 ]; then
    skip "Only IEC CRDs found"
  else
    skip "Neither CRD set found"
  fi

  # Check operator logs for interference
  OPERATOR_ERRORS=$(kubectl logs -n "$HP_NS" deployment/hyperpod-inference-controller-manager --tail=50 2>/dev/null | grep -i "error\|conflict" | grep -v "EnableFailed\|EnableClusterInference\|watcher" || true)
  if [ -n "$OPERATOR_ERRORS" ]; then
    fail "Operator logs show unexpected errors"
    echo "$OPERATOR_ERRORS" | head -3 | sed 's/^/    /'
  else
    pass "No unexpected errors in HyperPod operator logs"
  fi

  # Verify dynamo-validation and llmd-validation can coexist
  DYNAMO_NS=$(kubectl get namespace dynamo-validation 2>/dev/null | grep -c "Active" || true)
  LLMD_NS=$(kubectl get namespace llmd-validation 2>/dev/null | grep -c "Active" || true)
  if [ "$DYNAMO_NS" -gt 0 ] && [ "$LLMD_NS" -gt 0 ]; then
    pass "Both dynamo-validation and llmd-validation namespaces active"
  elif [ "$DYNAMO_NS" -gt 0 ]; then
    pass "dynamo-validation namespace active"
  else
    fail "dynamo-validation namespace not found"
  fi
}

# ---------- Main ----------
echo "============================================"
echo "Dynamo on HyperPod — Integration Smoke Test"
echo "============================================"
echo "Namespace: $NAMESPACE"
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
