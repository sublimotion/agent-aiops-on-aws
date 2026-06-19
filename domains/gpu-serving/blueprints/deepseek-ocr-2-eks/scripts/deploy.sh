#!/usr/bin/env bash
# Deploy DeepSeek-OCR-2 vLLM endpoint on the g6e-ocr nodegroup.
# Usage: ./deploy.sh [g6e-2xlarge]  (only 2xlarge scaffolded for now)
set -euo pipefail

SIZE="${1:-g6e-2xlarge}"
MANIFEST="$(dirname "$0")/../k8s/vllm-deepseek-ocr-${SIZE}.yaml"
NS="cto-ocr-${SIZE}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 1
fi

echo ">> kubectl context"
kubectl config current-context

echo ">> apply manifest"
kubectl apply -f "$MANIFEST"

echo ">> waiting for pod Ready (up to 20 min for first-boot download)"
kubectl -n "$NS" wait --for=condition=ready pod -l workload=deepseek-ocr --timeout=1200s

echo ">> endpoint ready; test:"
echo "   kubectl -n $NS port-forward svc/deepseek-ocr-${SIZE}-svc 8000:8000"
echo "   curl http://localhost:8000/health"
