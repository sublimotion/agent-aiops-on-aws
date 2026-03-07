#!/usr/bin/env bash
# Pre-flight validation for GLM-5 llm-d deployment
set -euo pipefail

echo "=== GLM-5 llm-d Pre-Flight Check ==="
PASS=0
FAIL=0

check() {
  local desc="$1" cmd="$2"
  echo -n "  $desc: "
  if eval "$cmd" >/dev/null 2>&1; then
    echo "PASS"
    ((PASS++))
  else
    echo "FAIL"
    ((FAIL++))
  fi
}

# GPU checks
echo ""
echo "--- GPU ---"
check "nvidia-smi" "nvidia-smi"
check "8 GPUs detected" "[ $(nvidia-smi -L | wc -l) -eq 8 ]"
check "H200 GPU type" "nvidia-smi -L | head -1 | grep -qi H200"

# NVLink
echo ""
echo "--- NVLink ---"
check "NVLink active" "nvidia-smi nvlink --status | grep -q Active"

# EFA
echo ""
echo "--- EFA ---"
check "EFA devices" "ls /dev/infiniband/uverbs* 2>/dev/null"

# Storage
echo ""
echo "--- Storage ---"
check "NVMe RAID0 mounted" "mountpoint -q /mnt/nvme"
check "FSx mounted" "mountpoint -q /mnt/fsx"

# Kubernetes
echo ""
echo "--- Kubernetes ---"
check "kubectl connectivity" "kubectl get nodes"
check "Gateway API CRDs" "kubectl get crd inferencepool.inference.networking.x-k8s.io 2>/dev/null"
check "Redis running" "kubectl get pods -n ml-inference -l app=redis-glm5 --field-selector=status.phase=Running 2>/dev/null | grep -q Running"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && echo "All checks passed!" || echo "WARNING: $FAIL checks failed"
exit "$FAIL"
