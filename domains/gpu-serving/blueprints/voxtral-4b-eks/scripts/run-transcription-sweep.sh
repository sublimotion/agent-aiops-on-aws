#!/usr/bin/env bash
# Voxtral-Mini-3B Stage 6 transcription concurrency sweep.
# Levels [1, 4, 16]; 5 warmup + 30 steady per level; round-robin 3s/10s/30s corpus.
# Emits ONE Common Benchmark Artifact under results/artifacts/.
set -euo pipefail

NS="${NS:-cto-voxtral-g6e-2xlarge}"
SVC="${SVC:-voxtral-g6e-2xlarge-svc}"
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
for f in short-3s.wav medium-10s.wav long-30s.wav; do
  [[ -f "$ASSETS_DIR/$f" ]] || { echo "ERROR: corpus asset missing: $ASSETS_DIR/$f (run scripts/test-assets/generate_audio.py)" >&2; exit 2; }
done

PF_OWNED=0
if ! lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[pf] starting kubectl port-forward -> $NS/$SVC:8000"
  kubectl -n "$NS" port-forward "svc/$SVC" 8000:8000 >/tmp/pf-voxtral-sweep.log 2>&1 &
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
OUT="$ARTIFACT_DIR/voxtral-mini-3b_eks_g6e-2xl_vllm_transcription-sweep_${TS_UTC}.json"

exec python3 "$HERE/_transcription_sweep.py" --assets-dir "$ASSETS_DIR" --out "$OUT"
