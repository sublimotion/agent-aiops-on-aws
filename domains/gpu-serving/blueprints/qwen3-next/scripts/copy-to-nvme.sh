#!/usr/bin/env bash
# Copies model from FSx to NVMe RAID0 for fastest serving.
# Run on the p5en node via SSM or kubectl exec.
#
# Usage: ./copy-to-nvme.sh [model-name]
#   Default: qwen3-next-fp8
#   For BF16: ./copy-to-nvme.sh qwen3-next
set -euo pipefail
MODEL="${1:-qwen3-next-fp8}"
SRC="/mnt/fsx/models/${MODEL}"
DST="/mnt/nvme/models/${MODEL}"

if [ -f "$DST/config.json" ]; then
  echo "Model already staged at $DST"
  du -sh "$DST"
  exit 0
fi

if [ ! -d "$SRC" ]; then
  echo "ERROR: Source not found: $SRC"
  exit 1
fi

mkdir -p "$DST"
echo "Copying model from $SRC to $DST..."
# Use cp without trailing slash to avoid double-nesting
cp -r "$SRC"/* "$DST"/
echo "Done. Model at $DST ($(du -sh "$DST" | cut -f1))"
ls "$DST"/*.safetensors 2>/dev/null | wc -l | xargs -I{} echo "{} safetensor shards"
