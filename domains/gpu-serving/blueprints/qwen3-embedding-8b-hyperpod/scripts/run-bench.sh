#!/usr/bin/env bash
# Run the embedding benchmark against the port-forwarded vLLM endpoint.
# Usage: ./run-bench.sh [workload_catalog_id]
# Defaults to `concurrency-sweep`.
set -euo pipefail
WORKLOAD="${1:-concurrency-sweep}"
BLUEPRINT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$BLUEPRINT_DIR/../../../../.." && pwd)"
RESULTS_DIR="$BLUEPRINT_DIR/results"
mkdir -p "$RESULTS_DIR"

# Port-forward in the background
echo "[setup] port-forward svc/qwen3-embedding-g5-4xlarge-svc 8000:8000"
kubectl --context finetune-eks -n cto-embedding-g5-4xlarge port-forward svc/qwen3-embedding-g5-4xlarge-svc 8000:8000 &
PF_PID=$!
trap "kill $PF_PID 2>/dev/null || true" EXIT
sleep 5

# Health check
for i in $(seq 1 10); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "[setup] endpoint healthy"
    break
  fi
  echo "[setup] waiting for /health... ($i/10)"
  sleep 3
done

# Smoke-test the embedding endpoint
echo
echo "[smoke] POST /v1/embeddings"
curl -sf -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-8B","input":"hello world"}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); e=d['data'][0]['embedding']; print(f'OK — vector dim={len(e)}, first={e[:3]}')"

# Run benchmark via the standard runner
echo
echo "[bench] $WORKLOAD against local endpoint"
"$REPO_ROOT/standards/benchmark-commons/runner/run-benchmark.sh" \
  --endpoint http://localhost:8000 \
  --workload "$WORKLOAD" \
  --sidecar "$BLUEPRINT_DIR/benchmark-g5-4xlarge.yaml" \
  --tool vllm \
  --output "$RESULTS_DIR" \
  --tag "hyperpod-g5-4xlarge-bf16"
