#!/usr/bin/env bash
# Pre-flight validation for GLM-5 LMCache deployment
set -euo pipefail

echo "=== GLM-5 LMCache Pre-Flight Check ==="
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

# GDS
echo ""
echo "--- GDS ---"
check "nvidia-fs module" "lsmod | grep -q nvidia_fs"
check "libcufile.so exists" "[ -f /usr/lib/x86_64-linux-gnu/libcufile.so ]"
check "cufile.json exists" "[ -f /etc/cufile.json ]"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && echo "All checks passed!" || echo "WARNING: $FAIL checks failed"
exit "$FAIL"
