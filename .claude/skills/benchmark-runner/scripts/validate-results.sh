#!/usr/bin/env bash
# Post-benchmark result validation.
# Checks that benchmark output files exist, are non-empty, and contain expected fields.
#
# Usage:
#   ./validate-results.sh <result_dir>
#
# Exit codes:
#   0 — All checks passed
#   1 — One or more checks failed

set -euo pipefail

RESULT_DIR="${1:?Usage: validate-results.sh <result_dir>}"
PASS=0
FAIL=0
WARN=0

check() {
  local desc="$1" result="$2"
  if [ "$result" = "0" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: $desc"
  else
    FAIL=$((FAIL + 1))
    echo "  FAIL: $desc"
  fi
}

warn() {
  WARN=$((WARN + 1))
  echo "  WARN: $1"
}

echo "=== Benchmark Result Validation ==="
echo "Directory: $RESULT_DIR"
echo ""

# 1. Directory exists and is non-empty
test -d "$RESULT_DIR"
check "Result directory exists" "$?"

file_count=$(find "$RESULT_DIR" -type f | wc -l | tr -d ' ')
[ "$file_count" -gt 0 ]
check "Result directory is non-empty ($file_count files)" "$?"

# 2. Check each benchmark label subdirectory
echo ""
echo "--- Benchmark Subdirectories ---"
label_count=0
for label_dir in "$RESULT_DIR"/*/; do
  [ -d "$label_dir" ] || continue
  label=$(basename "$label_dir")

  # Skip metric files at top level
  [[ "$label" == *"_metrics"* ]] && continue

  label_count=$((label_count + 1))

  # Check for run logs
  run_logs=$(find "$label_dir" -name "run_*.log" -type f | wc -l | tr -d ' ')
  [ "$run_logs" -gt 0 ]
  check "[$label] Has run logs ($run_logs)" "$?"

  # Check for JSON results
  json_files=$(find "$label_dir" -name "*.json" -type f | wc -l | tr -d ' ')
  if [ "$json_files" -gt 0 ]; then
    check "[$label] Has JSON results ($json_files)" "0"

    # Validate JSON is parseable and has expected fields
    for json in "$label_dir"/*.json; do
      [ -f "$json" ] || continue
      if python3 -c "import json; json.load(open('$json'))" 2>/dev/null; then
        check "[$label] $(basename "$json") is valid JSON" "0"

        # Check for key metrics fields
        has_ttft=$(python3 -c "
import json
d = json.load(open('$json'))
print('1' if any('ttft' in str(k).lower() for k in d.keys()) else '0')
" 2>/dev/null || echo "0")
        [ "$has_ttft" = "1" ]
        check "[$label] $(basename "$json") contains TTFT metrics" "$?"
      else
        check "[$label] $(basename "$json") is valid JSON" "1"
      fi
    done
  else
    warn "[$label] No JSON results (--save-result may not have been set)"
  fi
done

[ "$label_count" -gt 0 ]
check "At least one benchmark label found" "$?"

# 3. Check for metrics captures
echo ""
echo "--- Metrics Captures ---"
metrics_files=$(find "$RESULT_DIR" -maxdepth 1 -name "*_metrics.txt" -type f | wc -l | tr -d ' ')
if [ "$metrics_files" -gt 0 ]; then
  check "Prometheus metrics captured ($metrics_files files)" "0"

  # Check that metrics files are non-empty
  for mf in "$RESULT_DIR"/*_metrics.txt; do
    [ -f "$mf" ] || continue
    size=$(wc -c < "$mf" | tr -d ' ')
    if [ "$size" -gt 100 ]; then
      check "[$(basename "$mf")] Non-trivial size (${size}B)" "0"
    else
      warn "[$(basename "$mf")] Suspiciously small (${size}B) — server may not have been running"
    fi
  done
else
  warn "No Prometheus metrics captured (capture_metrics not called?)"
fi

# 4. Check for KV metrics (optional)
kv_files=$(find "$RESULT_DIR" -maxdepth 1 -name "*_kv_metrics.txt" -type f | wc -l | tr -d ' ')
if [ "$kv_files" -gt 0 ]; then
  echo ""
  echo "--- KV Cache Metrics ---"
  check "KV cache metrics captured ($kv_files files)" "0"
fi

# 5. Summary
echo ""
echo "=== Summary ==="
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Warnings: $WARN"
echo ""

if [ "$FAIL" -gt 0 ]; then
  echo "RESULT: FAIL ($FAIL checks failed)"
  exit 1
else
  echo "RESULT: PASS (all checks passed, $WARN warnings)"
  exit 0
fi
