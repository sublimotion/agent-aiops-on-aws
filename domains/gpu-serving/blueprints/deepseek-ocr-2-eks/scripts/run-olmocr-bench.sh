#!/usr/bin/env bash
# Run olmOCR-bench against the DeepSeek-OCR-2 vLLM endpoint.
#
# Reference (from allenai/olmocr README):
#   pip install "olmocr[bench]"
#   # Download the 1,400-doc standard set
#   olmocr-bench-download --dataset standard --out /data/olmocr-bench
#   # Run the model runner against an OpenAI-compatible endpoint
#   OPENAI_API_BASE="$API_BASE" OPENAI_API_KEY=dummy \
#     olmocr-bench-run \
#       --model deepseek-ai/DeepSeek-OCR-2 \
#       --prompt-template "<image>\n<|grounding|>Convert the document to markdown. " \
#       --dataset /data/olmocr-bench \
#       --out /data/olmocr-bench/results-${PRECISION}.jsonl
#   # Score
#   olmocr-bench-score /data/olmocr-bench/results-${PRECISION}.jsonl \
#       > /data/olmocr-bench/score-${PRECISION}.json
#
# Subscores reported: arxiv_math, old_scans, tables, headers_footers,
#                     multi_column, long_tiny_text, base
# DeepSeek self-reports overall 76.3 on this benchmark.

set -euo pipefail

API_BASE="${API_BASE:-http://deepseek-ocr-g6e-2xlarge-svc.cto-ocr-g6e-2xlarge.svc.cluster.local:8000/v1}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-OCR-2}"
PRECISION="${PRECISION:-bf16}"
DATASET_S3="${DATASET_S3:-s3://agent-aiops-bench-us-east-2/datasets/olmocr-bench/}"
DATASET_LOCAL="${DATASET_LOCAL:-/data/olmocr-bench}"
OUT_DIR="${OUT_DIR:-$(dirname "$0")/../results/olmocr-bench}"
PROMPT='<image>\n<|grounding|>Convert the document to markdown. '

echo "[olmocr-bench] API_BASE=${API_BASE}  MODEL=${MODEL}  PRECISION=${PRECISION}"

# Gate: dataset must be mirrored before running
if ! aws s3 ls "$DATASET_S3" >/dev/null 2>&1; then
  echo "[olmocr-bench] ERROR: dataset not yet mirrored at ${DATASET_S3}" >&2
  echo "[olmocr-bench] Mirror the standard set with:" >&2
  echo "  olmocr-bench-download --dataset standard --out ./olmocr-bench" >&2
  echo "  aws s3 sync ./olmocr-bench ${DATASET_S3}" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$DATASET_LOCAL"

# Invocation skeleton (uncomment when container + deps land):
#
# pip install "olmocr[bench]" >/dev/null
# aws s3 sync "$DATASET_S3" "$DATASET_LOCAL"
# OPENAI_API_BASE="$API_BASE" OPENAI_API_KEY=dummy \
#   olmocr-bench-run \
#     --model "$MODEL" \
#     --prompt-template "$PROMPT" \
#     --dataset "$DATASET_LOCAL" \
#     --out "$OUT_DIR/results-${PRECISION}.jsonl"
# olmocr-bench-score "$OUT_DIR/results-${PRECISION}.jsonl" \
#     > "$OUT_DIR/score-${PRECISION}.json"
# cat "$OUT_DIR/score-${PRECISION}.json"

echo "[olmocr-bench] STUB: runner not yet wired. Uncomment invocation block once"
echo "               olmocr[bench] is installed in the harness container and the"
echo "               dataset is mirrored to ${DATASET_S3}."
exit 0
