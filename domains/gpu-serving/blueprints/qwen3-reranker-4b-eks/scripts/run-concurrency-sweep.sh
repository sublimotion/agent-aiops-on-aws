#!/usr/bin/env bash
# Qwen3-Reranker-4B — Stage 6 concurrency sweep (iteration 1).
# Levels [1,4,16,64]; 10 warmup + 50 steady per level. Pair length 1024, k=50.
# Emits ONE Common Benchmark Artifact.
set -euo pipefail

NS="${NS:-cto-reranker-g6e-2xlarge}"
SVC="${SVC:-qwen3-reranker-g6e-2xlarge-svc}"
ENDPOINT="${ENDPOINT:-http://localhost:8000}"

HERE="$(cd "$(dirname "$0")" && pwd)"
BP_DIR="$(cd "$HERE/.." && pwd)"
ARTIFACT_DIR="$BP_DIR/results/artifacts"
mkdir -p "$ARTIFACT_DIR"

for bin in python3 kubectl lsof curl; do
  command -v "$bin" >/dev/null 2>&1 || { echo "ERROR: '$bin' not found" >&2; exit 127; }
done
python3 -c "import aiohttp" 2>/dev/null || { echo "ERROR: aiohttp missing" >&2; exit 127; }

PF_OWNED=0
if ! lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[pf] starting kubectl port-forward -> $NS/$SVC:8000"
  kubectl -n "$NS" port-forward "svc/$SVC" 8000:8000 >/tmp/pf-reranker-sweep.log 2>&1 &
  PF_PID=$!
  PF_OWNED=1
  for _ in $(seq 1 30); do
    sleep 0.5
    curl -sS -o /dev/null -w '%{http_code}' "$ENDPOINT/health" 2>/dev/null | grep -q 200 && break
  done
fi
trap '[[ "$PF_OWNED" == "1" ]] && kill "$PF_PID" 2>/dev/null || true' EXIT

curl -sS -o /dev/null -w '%{http_code}' "$ENDPOINT/health" | grep -q 200 \
  || { echo "ERROR: endpoint not healthy" >&2; exit 3; }

TS_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ARTIFACT_DIR/qwen3-reranker-4b_eks_g6e-2xl_vllm_concurrency-sweep_${TS_UTC}.json"

exec python3 "$HERE/_concurrency_sweep.py" --out "$OUT"
