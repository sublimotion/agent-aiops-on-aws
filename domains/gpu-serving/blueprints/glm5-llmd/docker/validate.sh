#!/usr/bin/env bash
# Validates vLLM GLM-5 + LMCache + GDS + GPU environment
set -euo pipefail

echo "=== vLLM GLM-5 Validation ==="

# Check vLLM
echo -n "vLLM: "
python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "MISSING"

# Check LMCache
echo -n "LMCache: "
python3 -c "import lmcache; print(lmcache.__version__)" 2>/dev/null || echo "MISSING"

# Check KVConnector
echo -n "LMCacheConnectorV1: "
python3 -c "from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import LMCacheConnectorV1; print('AVAILABLE')" 2>/dev/null || echo "NOT FOUND"

# Check GPU count
echo -n "GPUs: "
python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "MISSING"

# Check GPU type
echo -n "GPU Type: "
python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "UNKNOWN"

# Check DeepGEMM
echo -n "DeepGEMM: "
python3 -c "import deep_gemm; print('AVAILABLE')" 2>/dev/null || echo "NOT FOUND"

# Check tool-call parser
echo -n "GLM47 Parser: "
python3 -c "from vllm.entrypoints.openai.tool_parsers.glm47_tool_parser import GLM47ToolParser; print('AVAILABLE')" 2>/dev/null || echo "NOT FOUND (try: from vllm.entrypoints.openai.tool_parsers import ToolParserManager)"

# Check cuFile/GDS
echo -n "cuFile (GDS): "
if python3 -c "import nvidia.cufile" 2>/dev/null; then
  echo "AVAILABLE"
else
  echo "NOT FOUND (POSIX fallback)"
fi

echo ""
echo "=== Validation Complete ==="
