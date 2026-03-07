#!/usr/bin/env bash
# Applies an InferenceEndpointConfig CRD manifest with ECR URI substitution.
#
# Usage: ./scripts/apply-crd.sh <manifest-path> <ecr-image-uri> [model-bucket] [region]
# Example: ./scripts/apply-crd.sh manifests/glm5-inference-endpoint-h200.yaml \
#            123456789012.dkr.ecr.us-west-2.amazonaws.com/glm5/sglang-glm5:v0.5.2 \
#            glm5-models-abc123 us-west-2

set -euo pipefail
MANIFEST="${1:?Usage: $0 <manifest-path> <ecr-image-uri> [model-bucket] [region]}"
ECR_IMAGE="${2:?Usage: $0 <manifest-path> <ecr-image-uri> [model-bucket] [region]}"
MODEL_BUCKET="${3:-}"
AWS_REGION="${4:-us-west-2}"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: ${MANIFEST}"
    exit 1
fi

echo "=== Applying InferenceEndpointConfig ==="
echo "Manifest:     ${MANIFEST}"
echo "ECR Image:    ${ECR_IMAGE}"
echo "Model Bucket: ${MODEL_BUCKET:-<not substituted>}"
echo "Region:       ${AWS_REGION}"
echo ""

# Substitute placeholders
TEMP=$(mktemp)
sed -e "s|\${ECR_URI}|${ECR_IMAGE}|g" \
    -e "s|\${MODEL_BUCKET}|${MODEL_BUCKET}|g" \
    -e "s|\${AWS_REGION}|${AWS_REGION}|g" \
    "$MANIFEST" > "$TEMP"

echo "--- Rendered manifest ---"
cat "$TEMP"
echo ""
echo "--- Applying ---"

kubectl apply -f "$TEMP"
rm -f "$TEMP"

echo ""
echo "Monitor deployment:"
echo "  kubectl get inferenceendpointconfig -w"
echo "  kubectl get pods -w"
