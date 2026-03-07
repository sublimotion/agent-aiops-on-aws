#!/usr/bin/env bash
# Stage Devstral Small 2 FP8 to NVMe for benchmarking
# The official HuggingFace checkpoint ships with native FP8 weights.
#
# Usage:
#   ./scripts/stage-devstral.sh                    # Download from HF to NVMe
#   HF_TOKEN=xxx ./scripts/stage-devstral.sh       # If gated access needed
#   DEVSTRAL_MODEL_PATH=/mnt/nvme/models/devstral-small-2-fp8 ./scripts/stage-devstral.sh
set -euo pipefail

MODEL_ID="${DEVSTRAL_MODEL_ID:-mistralai/Devstral-Small-2-24B-Instruct-2512}"
MODEL_PATH="${DEVSTRAL_MODEL_PATH:-/mnt/nvme/models/devstral-small-2-fp8}"

echo "=== Staging Devstral Small 2 FP8 ==="
echo "Source: $MODEL_ID"
echo "Target: $MODEL_PATH"

mkdir -p "$MODEL_PATH"

# Option 1: huggingface-cli (preferred — handles LFS)
if command -v huggingface-cli &>/dev/null; then
  echo "Using huggingface-cli..."
  huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_PATH" --local-dir-use-symlinks False
# Option 2: git lfs
elif command -v git &>/dev/null && git lfs version &>/dev/null; then
  echo "Using git lfs clone..."
  GIT_LFS_SKIP_SMUDGE=0 git clone "https://huggingface.co/${MODEL_ID}" "$MODEL_PATH"
else
  echo "ERROR: Need either huggingface-cli or git-lfs to download model"
  echo "Install: pip install huggingface_hub[cli]"
  exit 1
fi

echo ""
echo "=== Model staged ==="
du -sh "$MODEL_PATH"
ls -la "$MODEL_PATH"/*.safetensors 2>/dev/null | head -5
echo ""
echo "Model size (FP8): ~26 GB"
echo "Single RTX PRO 6000 (96 GB): fits with ~70 GB for KV cache"
