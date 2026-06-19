#!/usr/bin/env bash
# DeepSeek-OCR-2 — Stage 6 batch-throughput (open-loop @ c=32, 60s steady),
# stratified corpus round-robin across scripts/test-assets/.
set -euo pipefail

NS="${NS:-cto-ocr-g6e-2xlarge}"
SVC="${SVC:-deepseek-ocr-g6e-2xlarge-svc}"
ENDPOINT="${ENDPOINT:-http://localhost:8000}"

HERE="$(cd "$(dirname "$0")" && pwd)"
BP_DIR="$(cd "$HERE/.." && pwd)"
ASSETS_DIR="${ASSETS_DIR:-$HERE/test-assets}"
ARTIFACT_DIR="$BP_DIR/results/artifacts"
mkdir -p "$ARTIFACT_DIR"

for bin in python3 kubectl lsof curl; do
  command -v "$bin" >/dev/null 2>&1 || { echo "ERROR: '$bin' not found" >&2; exit 127; }
done
python3 -c "import aiohttp" 2>/dev/null || { echo "ERROR: aiohttp missing" >&2; exit 127; }
for f in receipt.png article.png table.png formula.png dense.png handwritten.png; do
  [[ -f "$ASSETS_DIR/$f" ]] || { echo "ERROR: corpus asset missing: $ASSETS_DIR/$f (run scripts/test-assets/generate_corpus.py)" >&2; exit 2; }
done

PF_OWNED=0
if ! lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[pf] starting kubectl port-forward -> $NS/$SVC:8000"
  kubectl -n "$NS" port-forward "svc/$SVC" 8000:8000 >/tmp/pf-ocr-batch.log 2>&1 &
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
OUT="$ARTIFACT_DIR/deepseek-ocr-2_eks_g6e-2xl_vllm_batch-throughput_${TS_UTC}.json"

exec python3 "$HERE/_batch_throughput.py" --assets-dir "$ASSETS_DIR" --out "$OUT"
