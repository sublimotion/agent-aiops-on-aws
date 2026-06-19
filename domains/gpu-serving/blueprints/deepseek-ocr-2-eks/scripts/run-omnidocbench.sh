#!/usr/bin/env bash
# Run OmniDocBench (opendatalab/OmniDocBench) against DeepSeek-OCR-2 outputs.
#
# Reference runner image: ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204
#
# Workflow:
#   1. Generate predictions by hitting the vLLM endpoint with grounding prompt
#      per document in the OmniDocBench corpus. Write per-doc markdown files
#      to $PRED_DIR (one .md per gt.json entry).
#   2. Invoke the containerized scorer which compares predictions against gt.json
#      and emits: text_edit, formula_cdm, table_teds, layout, reading_order.
#
#   docker run --rm \
#     -v "$GT_DIR:/data/gt" \
#     -v "$PRED_DIR:/data/pred" \
#     -v "$OUT_DIR:/data/out" \
#     ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204 \
#     --gt /data/gt/gt.json \
#     --pred /data/pred \
#     --out /data/out/omnidocbench-${PRECISION}.json
#
# DeepSeek-OCR-2 community leaderboard score: 90.25.

set -euo pipefail

API_BASE="${API_BASE:-http://deepseek-ocr-g6e-2xlarge-svc.cto-ocr-g6e-2xlarge.svc.cluster.local:8000/v1}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-OCR-2}"
PRECISION="${PRECISION:-bf16}"
DATASET_S3="${DATASET_S3:-s3://agent-aiops-bench-us-east-2/datasets/omnidocbench/}"
GT_DIR="${GT_DIR:-/data/omnidocbench/gt}"
PRED_DIR="${PRED_DIR:-/data/omnidocbench/pred-${PRECISION}}"
OUT_DIR="${OUT_DIR:-$(dirname "$0")/../results/omnidocbench}"
RUNNER_IMAGE="${RUNNER_IMAGE:-ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204}"
PROMPT='<image>\n<|grounding|>Convert the document to markdown. '

echo "[omnidocbench] API_BASE=${API_BASE}  MODEL=${MODEL}  PRECISION=${PRECISION}"

# Gate: dataset must be mirrored
if ! aws s3 ls "${DATASET_S3}gt.json" >/dev/null 2>&1; then
  echo "[omnidocbench] ERROR: dataset not yet mirrored at ${DATASET_S3}" >&2
  echo "[omnidocbench] Expected: ${DATASET_S3}gt.json + source documents/" >&2
  echo "[omnidocbench] Mirror OmniDocBench release to ${DATASET_S3} then re-run." >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$GT_DIR" "$PRED_DIR"

# Invocation skeleton (uncomment when container + deps land):
#
# aws s3 sync "$DATASET_S3" "$GT_DIR"
#
# # Step 1: generate predictions (one .md per doc)
# python3 "$(dirname "$0")/_generate_omnidocbench_preds.py" \
#   --api-base "$API_BASE" --model "$MODEL" \
#   --prompt "$PROMPT" \
#   --gt "$GT_DIR/gt.json" --out "$PRED_DIR"
#
# # Step 2: score
# docker run --rm \
#   -v "$GT_DIR:/data/gt" \
#   -v "$PRED_DIR:/data/pred" \
#   -v "$OUT_DIR:/data/out" \
#   "$RUNNER_IMAGE" \
#   --gt /data/gt/gt.json \
#   --pred /data/pred \
#   --out /data/out/omnidocbench-${PRECISION}.json
#
# cat "$OUT_DIR/omnidocbench-${PRECISION}.json"

echo "[omnidocbench] STUB: runner not yet wired. Uncomment invocation block once"
echo "               prediction generator lands and dataset is mirrored to"
echo "               ${DATASET_S3}."
exit 0
