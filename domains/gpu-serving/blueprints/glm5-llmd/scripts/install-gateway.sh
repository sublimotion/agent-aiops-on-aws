#!/usr/bin/env bash
# Install Gateway API CRDs + Envoy Gateway + llm-d EPP
set -euo pipefail

echo "=== Installing Gateway API + llm-d EPP ==="

# 1. Gateway API CRDs (inference extension)
echo "Installing Gateway API CRDs..."
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.0/standard-install.yaml
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/latest/download/manifests.yaml

# 2. Envoy Gateway
echo "Installing Envoy Gateway..."
helm install eg oci://docker.io/envoyproxy/gateway-helm \
  --version v1.2.0 \
  --namespace envoy-gateway-system \
  --create-namespace \
  --wait

# 3. Create Gateway resource
echo "Creating Gateway resource..."
kubectl apply -f - <<'EOF'
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: llm-gateway
  namespace: envoy-gateway-system
spec:
  gatewayClassName: eg
  listeners:
    - name: http
      protocol: HTTP
      port: 80
EOF

# 4. Wait for gateway to be ready
echo "Waiting for gateway to be ready..."
kubectl wait --for=condition=Programmed gateway/llm-gateway -n envoy-gateway-system --timeout=120s

echo ""
echo "=== Gateway API installed ==="
echo "Gateway IP: $(kubectl get gateway llm-gateway -n envoy-gateway-system -o jsonpath='{.status.addresses[0].value}' 2>/dev/null || echo 'pending')"
