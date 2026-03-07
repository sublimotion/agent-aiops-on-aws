#!/usr/bin/env bash
# Validates GDS (GPU Direct Storage) readiness
set -euo pipefail

echo "=== GDS Validation ==="

# Check nvidia-fs module
echo -n "nvidia-fs module: "
if lsmod | grep -q nvidia_fs; then
  echo "LOADED"
else
  echo "NOT LOADED — attempting modprobe..."
  modprobe nvidia-fs 2>/dev/null && echo "  Loaded successfully" || echo "  FAILED (GDS unavailable, POSIX fallback)"
fi

# Check libcufile symlink
echo -n "libcufile.so symlink: "
if [ -L /usr/lib/x86_64-linux-gnu/libcufile.so ]; then
  echo "OK ($(readlink -f /usr/lib/x86_64-linux-gnu/libcufile.so))"
elif [ -f /usr/lib/x86_64-linux-gnu/libcufile.so.0 ]; then
  echo "MISSING — creating..."
  ln -sf /usr/lib/x86_64-linux-gnu/libcufile.so.0 /usr/lib/x86_64-linux-gnu/libcufile.so
  ldconfig
  echo "  Created successfully"
else
  echo "NOT FOUND (libcufile.so.0 missing)"
fi

# Check cufile.json
echo -n "cufile.json: "
[ -f /etc/cufile.json ] && echo "OK" || echo "MISSING"

# Test cuFile transfer if gdsio is available
echo ""
echo "--- cuFile Transfer Test ---"
if command -v gdsio >/dev/null 2>&1; then
  TEST_DIR="/mnt/fsx/gds-test-$$"
  mkdir -p "$TEST_DIR"
  echo "Running 4KB GPU↔disk transfer test..."
  gdsio -f "$TEST_DIR/test" -d 0 -w 1 -s 4K -x 0 -I 1 2>&1 || echo "  gdsio test failed (GDS may not be functional)"
  rm -rf "$TEST_DIR"
else
  echo "gdsio not found — skipping transfer test"
  echo "Install from CUDA toolkit: apt install cuda-gds-tools-*"
fi

echo ""
echo "=== GDS Validation Complete ==="
