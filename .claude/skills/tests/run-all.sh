#!/bin/bash
# Run all skill tests
#
# Usage: ./run-all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  Skill Test Suite"
echo "========================================"
echo ""

# Functional tests (automated)
echo "Running functional tests..."
"$SCRIPT_DIR/test-functional.sh"
FUNC_EXIT=$?
echo ""

# Trigger tests (manual — prints test cases)
echo "Printing trigger test cases (manual verification)..."
"$SCRIPT_DIR/test-triggers.sh"
echo ""

echo "========================================"
if [ "$FUNC_EXIT" -eq 0 ]; then
  echo "  All automated tests PASSED"
else
  echo "  $FUNC_EXIT automated tests FAILED"
fi
echo "  Trigger tests require manual verification"
echo "========================================"

exit $FUNC_EXIT
