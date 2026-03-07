#!/usr/bin/env bash
# Validates the SGLang GLM-5 container environment
set -e

echo "=== SGLang GLM-5 Container Validation ==="
echo ""

echo "1. Checking SGLang..."
python3 -c "import sglang; print(f'  SGLang version: {sglang.__version__}')" 2>/dev/null || \
    echo "  FAIL: sglang not importable"

echo ""
echo "2. Checking transformers..."
python3 -c "
import transformers
print(f'  transformers version: {transformers.__version__}')
" 2>/dev/null || echo "  FAIL: transformers not importable"

echo ""
echo "3. Checking GPU availability..."
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'  GPUs: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'    [{i}] {name} ({mem:.0f} GB)')
" 2>/dev/null || echo "  FAIL: CUDA not available"

echo ""
echo "4. Checking serve entrypoint..."
if [ -x /usr/local/bin/serve ]; then
    echo "  serve script: OK"
else
    echo "  FAIL: /usr/local/bin/serve not found or not executable"
fi

echo ""
echo "5. Checking model mount point..."
if [ -d /opt/ml/model ]; then
    count=$(find /opt/ml/model -maxdepth 1 -type f 2>/dev/null | wc -l)
    echo "  /opt/ml/model exists ($count files)"
else
    echo "  /opt/ml/model not mounted (expected — operator will mount at runtime)"
fi

echo ""
echo "=== Validation Complete ==="
