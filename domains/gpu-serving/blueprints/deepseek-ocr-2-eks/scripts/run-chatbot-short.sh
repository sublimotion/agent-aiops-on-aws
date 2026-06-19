#!/usr/bin/env bash
# DeepSeek-OCR-2 — Stage 6 "chatbot-short" latency cell @ c=1 (BF16).
#
# Goal: produce ONE valid Common Benchmark Artifact end-to-end to prove Stage 6
# wiring. This is NOT a concurrency sweep; c=1 only. VLM = non-streaming,
# e2e-only latency (no TTFT). Usage tokens come from response.usage.
#
# Runs:
#   - ensure port-forward to svc/deepseek-ocr-g6e-2xlarge-svc:8000 (spawn if absent)
#   - 10 warmup requests (grounding prompt + bundled test image)
#   - 100 steady-state sequential requests (c=1), record per-request e2e ms
#   - compute mean / p50 / p90 / p95 / p99
#   - emit results/artifacts/deepseek-ocr-2_eks_g6e-2xl_vllm_chatbot-short_<ts>.json
#
# Output path is printed on the final line.

set -euo pipefail

NS="${NS:-cto-ocr-g6e-2xlarge}"
SVC="${SVC:-deepseek-ocr-g6e-2xlarge-svc}"
ENDPOINT="${ENDPOINT:-http://localhost:8000}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-OCR-2}"
MAX_TOKENS="${MAX_TOKENS:-256}"
WARMUP="${WARMUP:-10}"
TOTAL="${TOTAL:-100}"
CONCURRENCY=1

HERE="$(cd "$(dirname "$0")" && pwd)"
BP_DIR="$(cd "$HERE/.." && pwd)"
ASSET_PATH="${ASSET_PATH:-$HERE/test-assets/sample-doc.png}"
ARTIFACT_DIR="$BP_DIR/results/artifacts"
mkdir -p "$ARTIFACT_DIR"

for bin in jq curl base64 python3 kubectl lsof; do
  command -v "$bin" >/dev/null 2>&1 || { echo "ERROR: '$bin' not found" >&2; exit 127; }
done
[[ -f "$ASSET_PATH" ]] || { echo "ERROR: asset missing: $ASSET_PATH" >&2; exit 2; }

# --- port-forward ensure -----------------------------------------------------
PF_OWNED=0
if ! lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[pf] starting kubectl port-forward -> $NS/$SVC:8000"
  kubectl -n "$NS" port-forward "svc/$SVC" 8000:8000 >/tmp/pf-ocr-chatbot-short.log 2>&1 &
  PF_PID=$!
  PF_OWNED=1
  # wait up to 15s for health
  for _ in $(seq 1 30); do
    sleep 0.5
    if curl -sS -o /dev/null -w '%{http_code}' "$ENDPOINT/health" 2>/dev/null | grep -q 200; then
      break
    fi
  done
fi
trap '[[ "$PF_OWNED" == "1" ]] && kill "$PF_PID" 2>/dev/null || true' EXIT

if ! curl -sS -o /dev/null -w '%{http_code}' "$ENDPOINT/health" | grep -q 200; then
  echo "ERROR: endpoint $ENDPOINT/health not healthy" >&2
  exit 3
fi

# --- build request body once (image is constant) -----------------------------
B64="$(base64 < "$ASSET_PATH" | tr -d '\n')"
DATA_URL="data:image/png;base64,${B64}"
PROMPT=$'<image>\n<|grounding|>Convert the document to markdown. '

BODY="$(jq -nc \
  --arg model "$MODEL" \
  --arg prompt "$PROMPT" \
  --arg url   "$DATA_URL" \
  --argjson max_tokens "$MAX_TOKENS" \
  '{model:$model, messages:[{role:"user", content:[
     {type:"text", text:$prompt},
     {type:"image_url", image_url:{url:$url}}
   ]}], max_tokens:$max_tokens, temperature:0.0, stream:false}')"

BODY_FILE="$(mktemp)"; printf '%s' "$BODY" > "$BODY_FILE"
SAMPLE_FILE="$(mktemp)"; : > "$SAMPLE_FILE"
trap '[[ "$PF_OWNED" == "1" ]] && kill "$PF_PID" 2>/dev/null || true; rm -f "$BODY_FILE" "$SAMPLE_FILE"' EXIT

fire_one() {
  # prints: <e2e_ms> <completion_tokens> <prompt_tokens>
  local resp_file; resp_file="$(mktemp)"
  local t0 t1 e2e_ms completion prompt
  t0=$(python3 -c 'import time;print(time.perf_counter())')
  local code
  code=$(curl -sS -o "$resp_file" -w '%{http_code}' \
    -X POST "$ENDPOINT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "@$BODY_FILE")
  t1=$(python3 -c 'import time;print(time.perf_counter())')
  if [[ "$code" != "200" ]]; then
    rm -f "$resp_file"
    echo "ERROR $code" >&2
    return 1
  fi
  e2e_ms=$(python3 -c "print(($t1-$t0)*1000.0)")
  completion=$(jq -r '.usage.completion_tokens // 0' < "$resp_file")
  prompt=$(jq -r '.usage.prompt_tokens // 0' < "$resp_file")
  rm -f "$resp_file"
  echo "$e2e_ms $completion $prompt"
}

# --- warmup ------------------------------------------------------------------
echo "[warmup] $WARMUP requests"
for i in $(seq 1 "$WARMUP"); do
  fire_one >/dev/null || { echo "warmup failed at i=$i" >&2; exit 4; }
done

# --- steady state ------------------------------------------------------------
echo "[steady] $TOTAL sequential requests (c=$CONCURRENCY)"
RUN_T0=$(python3 -c 'import time;print(time.perf_counter())')
failed=0
for i in $(seq 1 "$TOTAL"); do
  if ! out=$(fire_one); then
    failed=$((failed+1))
    continue
  fi
  echo "$out" >> "$SAMPLE_FILE"
  if (( i % 20 == 0 )); then echo "  progress: $i/$TOTAL"; fi
done
RUN_T1=$(python3 -c 'import time;print(time.perf_counter())')
DURATION_S=$(python3 -c "print($RUN_T1-$RUN_T0)")
completed=$(wc -l < "$SAMPLE_FILE" | tr -d ' ')
echo "[done] completed=$completed failed=$failed duration_s=$DURATION_S"

# --- emit artifact via python (stats + schema) -------------------------------
TS_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_PATH="$ARTIFACT_DIR/deepseek-ocr-2_eks_g6e-2xl_vllm_chatbot-short_${TS_UTC}.json"

SAMPLE_FILE="$SAMPLE_FILE" \
ARTIFACT_PATH="$ARTIFACT_PATH" \
DURATION_S="$DURATION_S" \
TOTAL="$TOTAL" \
COMPLETED="$completed" \
FAILED="$failed" \
MODEL_ID="$MODEL" \
python3 - <<'PY'
import json, os, statistics, uuid, datetime, pathlib

sample_file = os.environ["SAMPLE_FILE"]
artifact_path = os.environ["ARTIFACT_PATH"]
duration_s = float(os.environ["DURATION_S"])
total = int(os.environ["TOTAL"])
completed = int(os.environ["COMPLETED"])
failed = int(os.environ["FAILED"])
model_id = os.environ["MODEL_ID"]

e2e_ms, completion_tokens, prompt_tokens = [], [], []
with open(sample_file) as f:
    for line in f:
        parts = line.split()
        if len(parts) != 3: continue
        e2e_ms.append(float(parts[0]))
        completion_tokens.append(int(parts[1]))
        prompt_tokens.append(int(parts[2]))

def pct(xs, p):
    if not xs: return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs)-1, int(round((p/100.0)*(len(xs)-1)))))
    return xs[k]

mean = statistics.fmean(e2e_ms) if e2e_ms else 0.0
total_out = sum(completion_tokens)
total_in = sum(prompt_tokens)
error_rate = failed / total if total else 0.0
req_tp = completed / duration_s if duration_s > 0 else 0.0
out_tps = total_out / duration_s if duration_s > 0 else 0.0

artifact = {
    "schema_version": "1.0.0",
    "artifact_id": str(uuid.uuid4()),
    "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source_tool": {
        "name": "custom",
        "version": "0.1.0-deepseek-ocr-chatbot-short",
        "enrichment_version": "1.0.0"
    },
    "model": {
        "name": "DeepSeek-OCR-2",
        "id": model_id,
        "architecture": "vlm",
        "parameters_total": "8B",
        "quantization": "bf16",
        "max_model_len": 8192
    },
    "engine": {
        "name": "vllm",
        "version": "0.19.1",
        "container_image": "vllm/vllm-openai:v0.19.1",
        "base_image": None,
        "dockerfile": None,
        "tensor_parallel": 1,
        "pipeline_parallel": 1,
        "data_parallel": None,
        "expert_parallel": None,
        "replicas": 1,
        "reasoning": False,
        "kv_cache_dtype": "auto",
        "attention_backend": "flash-attn",
        "speculative_decode": None,
        "extra_args": {
            "trust-remote-code": True,
            "max-num-seqs": 32,
            "gpu-memory-utilization": 0.90
        }
    },
    "infrastructure": {
        "substrate": "eks",
        "instance_type": "g6e.2xlarge",
        "region": "us-east-2",
        "gpu": {
            "name": "L40S",
            "arch": "sm_89",
            "count": 1,
            "vram_gb": 48,
            "interconnect": "none"
        }
    },
    "workload": {
        "use_case": "ocr",
        "catalog_id": "chatbot-short",
        "modality": "multimodal",
        "dataset": {
            "type": "synthetic-single-image",
            "source": "scripts/test-assets/sample-doc.png",
            "input_tokens": {"mean": statistics.fmean(prompt_tokens) if prompt_tokens else 0},
            "output_tokens": {"mean": statistics.fmean(completion_tokens) if completion_tokens else 0}
        },
        "load": {
            "type": "sequential",
            "concurrency": 1,
            "num_prompts": total,
            "warmup_requests": int(os.environ.get("WARMUP", "10"))
        },
        "api": {
            "type": "chat",
            "streaming": False,
            "endpoint": "/v1/chat/completions",
            "prompt_template": "<image>\n<|grounding|>Convert the document to markdown. "
        }
    },
    "metrics": {
        "duration_s": duration_s,
        "completed": completed,
        "failed": failed,
        "error_rate": error_rate,
        "e2e_ms": {
            "mean": mean,
            "p50": pct(e2e_ms, 50),
            "p90": pct(e2e_ms, 90),
            "p95": pct(e2e_ms, 95),
            "p99": pct(e2e_ms, 99)
        },
        "output_toks_per_s": out_tps,
        "request_throughput": req_tp,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "max_concurrent_requests": 1
    },
    "extensions": {
        "modality": "vision-language",
        "notes": "Stage 6 smoke cell: c=1 sequential, non-streaming, e2e-only (no TTFT for VLM non-stream). Proves Stage 6 artifact wiring."
    }
}

pathlib.Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
with open(artifact_path, "w") as f:
    json.dump(artifact, f, indent=2)
print(artifact_path)
PY

echo "ARTIFACT: $ARTIFACT_PATH"
