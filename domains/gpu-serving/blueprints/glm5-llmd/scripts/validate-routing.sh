#!/usr/bin/env bash
# Validates llm-d prefix-cache-aware routing
set -euo pipefail

echo "=== llm-d Routing Validation ==="

# Get gateway IP
GATEWAY_IP=$(kubectl get gateway llm-gateway -n envoy-gateway-system -o jsonpath='{.status.addresses[0].value}' 2>/dev/null)
if [ -z "$GATEWAY_IP" ]; then
  echo "ERROR: Gateway IP not found. Run install-gateway.sh first."
  exit 1
fi

ENDPOINT="http://${GATEWAY_IP}/v1"
echo "Gateway endpoint: ${ENDPOINT}"

# Check model listing
echo ""
echo "--- Model Listing ---"
curl -s "${ENDPOINT}/models" | python3 -m json.tool 2>/dev/null || echo "FAIL: Cannot reach /v1/models"

# Send identical requests — should route to same pod (prefix cache hit)
echo ""
echo "--- Prefix Cache Routing Test ---"
SYSTEM_PROMPT="You are a helpful assistant specialized in analyzing financial reports."

echo "Request 1 (cold)..."
RESP1=$(curl -s -w "\n%{http_code}" "${ENDPOINT}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm5",
    "messages": [
      {"role": "system", "content": "'"${SYSTEM_PROMPT}"'"},
      {"role": "user", "content": "Summarize Q1 revenue trends."}
    ],
    "max_tokens": 50
  }')
echo "Response: $(echo "$RESP1" | head -1 | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("choices",[{}])[0].get("message",{}).get("content","")[:80])' 2>/dev/null || echo 'parse error')"

echo "Request 2 (same prefix, should hit cache)..."
RESP2=$(curl -s -w "\n%{http_code}" "${ENDPOINT}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm5",
    "messages": [
      {"role": "system", "content": "'"${SYSTEM_PROMPT}"'"},
      {"role": "user", "content": "What were Q2 operating expenses?"}
    ],
    "max_tokens": 50
  }')
echo "Response: $(echo "$RESP2" | head -1 | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("choices",[{}])[0].get("message",{}).get("content","")[:80])' 2>/dev/null || echo 'parse error')"

# Check EPP metrics
echo ""
echo "--- EPP Metrics ---"
EPP_POD=$(kubectl get pods -n ml-inference -l app=glm5-epp -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -n "$EPP_POD" ]; then
  kubectl exec -n ml-inference "$EPP_POD" -- curl -s localhost:9090/metrics 2>/dev/null | grep -E "prefix_cache|routing" | head -10 || echo "No EPP metrics found"
else
  echo "EPP pod not found"
fi

echo ""
echo "=== Routing Validation Complete ==="
