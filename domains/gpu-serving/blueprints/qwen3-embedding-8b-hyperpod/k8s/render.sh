#!/usr/bin/env bash
# Render per-instance-type vLLM manifests from the template.
# Usage: ./render.sh <instance_type> <slug>
#   e.g. ./render.sh ml.g6e.xlarge   g6e-xlarge
#   e.g. ./render.sh ml.g6e.2xlarge  g6e-2xlarge
#   e.g. ./render.sh ml.g5.2xlarge   g5-2xlarge
set -euo pipefail
INSTANCE="$1"
SLUG="$2"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/vllm-embedding-${SLUG}.yaml"

# Namespace per variant so deployments don't collide; Service name carries the slug.
sed -e "s|cto-embedding|cto-embedding-${SLUG}|g" \
    -e "s|ml.g5.2xlarge|${INSTANCE}|g" \
    -e "s|qwen3-embedding-vllm|qwen3-embedding-${SLUG}|g" \
    -e "s|qwen3-embedding-svc|qwen3-embedding-${SLUG}-svc|g" \
    -e "s|app: qwen3-embedding|app: qwen3-embedding-${SLUG}|g" \
    "$DIR/_template.yaml" > "$OUT"

echo "Rendered $OUT"
