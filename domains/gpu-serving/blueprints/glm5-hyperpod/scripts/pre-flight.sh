#!/usr/bin/env bash
# Pre-flight GPU/NVLink/EFA validation for HyperPod GPU nodes.
# Run via SSM session on the GPU node after cluster creation.
#
# Usage: ./scripts/pre-flight.sh
# Or run individual checks: ./scripts/pre-flight.sh gpu|nvlink|efa|xid

set -euo pipefail
CHECK="${1:-all}"

pass=0
fail=0

check_pass() { echo "  PASS: $1"; ((pass++)); }
check_fail() { echo "  FAIL: $1"; ((fail++)); }

# GPU Inventory
run_gpu_check() {
    echo "=== GPU Inventory ==="
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,ecc.mode.current \
        --format=csv,noheader 2>/dev/null || { check_fail "nvidia-smi not available"; return; }

    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -eq 8 ]; then
        check_pass "GPU count: ${GPU_COUNT}"
    else
        check_fail "GPU count: ${GPU_COUNT} (expected 8)"
    fi

    # ECC check
    ECC_OFF=$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader 2>/dev/null | grep -c "Disabled" || true)
    if [ "$ECC_OFF" -eq 0 ]; then
        check_pass "ECC enabled on all GPUs"
    else
        check_fail "ECC disabled on ${ECC_OFF} GPUs"
    fi

    # Temperature check
    MAX_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | sort -n | tail -1)
    if [ "${MAX_TEMP:-99}" -lt 80 ]; then
        check_pass "Max GPU temperature: ${MAX_TEMP}C"
    else
        check_fail "Max GPU temperature: ${MAX_TEMP}C (>= 80C)"
    fi
    echo ""
}

# NVLink Topology
run_nvlink_check() {
    echo "=== NVLink Topology ==="
    nvidia-smi topo -m 2>/dev/null || { check_fail "topo query failed"; return; }

    # Check for NVSwitch connectivity (NV18 = NVLink via NVSwitch)
    NVLINK_PAIRS=$(nvidia-smi topo -m 2>/dev/null | grep -c "NV" || true)
    if [ "$NVLINK_PAIRS" -gt 0 ]; then
        check_pass "NVLink pairs detected: ${NVLINK_PAIRS}"
    else
        check_fail "No NVLink connectivity detected"
    fi
    echo ""
}

# EFA Adapters
run_efa_check() {
    echo "=== EFA Adapters ==="
    if command -v fi_info &>/dev/null; then
        EFA_COUNT=$(fi_info -p efa 2>/dev/null | grep -c "provider: efa" || true)
        echo "  EFA providers: ${EFA_COUNT}"
        if [ "$EFA_COUNT" -gt 0 ]; then
            check_pass "EFA adapters detected"
        else
            check_fail "No EFA adapters found"
        fi
    else
        echo "  fi_info not available — checking sysfs"
        IB_COUNT=$(ls /sys/class/infiniband/ 2>/dev/null | wc -l)
        if [ "$IB_COUNT" -gt 0 ]; then
            check_pass "InfiniBand devices: ${IB_COUNT}"
        else
            check_fail "No InfiniBand/EFA devices in sysfs"
        fi
    fi
    echo ""
}

# Xid Errors
run_xid_check() {
    echo "=== Xid Error Check ==="
    XID_COUNT=$(dmesg 2>/dev/null | grep -c "NVRM: Xid" || true)
    if [ "$XID_COUNT" -eq 0 ]; then
        check_pass "No Xid errors in dmesg"
    else
        check_fail "Found ${XID_COUNT} Xid errors in dmesg"
        dmesg 2>/dev/null | grep "NVRM: Xid" | tail -5
    fi
    echo ""
}

# PCIe Link
run_pcie_check() {
    echo "=== PCIe Link ==="
    nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current \
        --format=csv,noheader 2>/dev/null || { check_fail "PCIe query failed"; return; }
    check_pass "PCIe link info retrieved"
    echo ""
}

case "$CHECK" in
    gpu)    run_gpu_check ;;
    nvlink) run_nvlink_check ;;
    efa)    run_efa_check ;;
    xid)    run_xid_check ;;
    pcie)   run_pcie_check ;;
    all)
        run_gpu_check
        run_nvlink_check
        run_efa_check
        run_xid_check
        run_pcie_check
        ;;
    *)
        echo "Usage: $0 [all|gpu|nvlink|efa|xid|pcie]"
        exit 1
        ;;
esac

echo "=== Summary: ${pass} passed, ${fail} failed ==="
[ "$fail" -eq 0 ] || exit 1
