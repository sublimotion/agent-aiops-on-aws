#!/usr/bin/env bash
# SERA Evaluation Suite
#
# Runs the full eval suite against a fine-tuned Devstral checkpoint:
#   D2-SERA: BFCL tool-use evaluation
#   D4-SERA: Swarm concurrency (4 replicas)
#   D5-SERA: Functional coding evaluation
#
# Prerequisites:
#   - Fine-tuned model served via vLLM on the target endpoint
#   - Eval scripts from qwen3-next-sglang blueprint
#
# Usage:
#   bash sera-eval.sh --endpoint http://localhost:8000 --model devstral-sera \
#       --output-dir /path/to/results

set -euo pipefail

ENDPOINT=""
MODEL="devstral-sera"
OUTPUT_DIR=""
SGLANG_SCRIPTS="../../qwen3-next-sglang/scripts"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --endpoint) ENDPOINT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$ENDPOINT" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: sera-eval.sh --endpoint URL --output-dir DIR [--model NAME]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  SERA Evaluation Suite"
echo "=========================================="
echo "  Endpoint: $ENDPOINT"
echo "  Model:    $MODEL"
echo "  Output:   $OUTPUT_DIR"
echo ""

# --- D2-SERA: BFCL ---
echo "--- D2-SERA: BFCL Tool-Use Evaluation ---"
python3 "$SGLANG_SCRIPTS/bfcl-eval.py" \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR/d2-bfcl" \
    2>&1 | tee "$OUTPUT_DIR/d2-bfcl.log"
echo ""

# --- D5-SERA: Functional Eval ---
echo "--- D5-SERA: Functional Coding Evaluation ---"
python3 "$SGLANG_SCRIPTS/functional-eval.py" \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR/d5-functional" \
    2>&1 | tee "$OUTPUT_DIR/d5-functional.log"
echo ""

# --- Summary ---
echo "=========================================="
echo "  SERA Evaluation Complete"
echo "=========================================="
echo "  D2 BFCL:      $OUTPUT_DIR/d2-bfcl/"
echo "  D5 Functional: $OUTPUT_DIR/d5-functional/"
echo ""
echo "  Compare with baseline:"
echo "    D2 baseline: 91.7% BFCL"
echo "    D5 baseline: 100% task completion, 97% SVG recall"
echo "=========================================="
