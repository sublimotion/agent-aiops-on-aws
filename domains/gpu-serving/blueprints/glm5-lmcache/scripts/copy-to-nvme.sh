#!/usr/bin/env bash
# Copies GLM-5 model from FSx to NVMe RAID0.
# Usage: ./copy-to-nvme.sh [model-name]
set -euo pipefail
MODEL="${1:-GLM-5-FP8}"
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
cp -r "$SRC"/* "$DST"/
echo "Done. Model at $DST ($(du -sh "$DST" | cut -f1))"
ls "$DST"/*.safetensors 2>/dev/null | wc -l | xargs -I{} echo "{} safetensor shards"
