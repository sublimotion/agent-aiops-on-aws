#!/bin/bash
# Re-evaluate the 99 instances that failed Docker builds in the original run.
# Uses --namespace swebench to PULL pre-built images from Docker Hub instead of building locally.
#
# Prerequisites:
#   - Docker running with enough disk (~100GB for images)
#   - swebench installed: pip install swebench
#   - predictions_errors_only.jsonl in results/
#
# Usage: bash scripts/reeval_errors.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BLUEPRINT_DIR="$(dirname "$SCRIPT_DIR")"
PREDICTIONS="$BLUEPRINT_DIR/results/predictions_errors_only.jsonl"
RUN_ID="claude-code-verify-lite-errors-v2"

echo "=== Re-evaluating 99 error instances with pre-built Docker images ==="
echo "Predictions: $PREDICTIONS"
echo "Run ID: $RUN_ID"
echo ""

# Count predictions
N=$(wc -l < "$PREDICTIONS" | tr -d ' ')
echo "Instances to evaluate: $N"
echo ""

# Activate venv if exists
if [ -f "$HOME/swebench-env/bin/activate" ]; then
    source "$HOME/swebench-env/bin/activate"
fi

# Run evaluation pulling pre-built images from Docker Hub
python3 -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path "$PREDICTIONS" \
    --max_workers 8 \
    --run_id "$RUN_ID" \
    --namespace swebench \
    --cache_level instance

echo ""
echo "=== Done ==="
echo "Check logs/ for per-instance results"
echo "Merge with original eval_report.json to get combined totals"
