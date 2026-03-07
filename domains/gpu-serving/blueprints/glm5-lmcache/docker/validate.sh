#!/usr/bin/env bash
# Validates SGLang + LMCache + GDS + GPU environment
set -euo pipefail

echo "=== SGLang + LMCache Validation ==="

# Check SGLang
echo -n "SGLang: "
python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "MISSING"

# Check LMCache
echo -n "LMCache: "
python3 -c "import lmcache; print(lmcache.__version__)" 2>/dev/null || echo "MISSING"

# Check GPU count
echo -n "GPUs: "
python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "MISSING"

# Check GPU type
echo -n "GPU Type: "
python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "UNKNOWN"

# Check cuFile/GDS
echo -n "cuFile (GDS): "
if [ -f /usr/lib/x86_64-linux-gnu/libcufile.so ]; then
  echo "AVAILABLE"
else
  echo "NOT FOUND (POSIX fallback)"
fi

# Check nvidia-fs module
echo -n "nvidia-fs module: "
lsmod 2>/dev/null | grep -q nvidia_fs && echo "LOADED" || echo "NOT LOADED"

echo ""
echo "=== Validation Complete ==="
