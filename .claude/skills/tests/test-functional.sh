#!/bin/bash
# Skill Functional Tests
# Validates that skills produce correct outputs given sample inputs.
#
# Usage: ./test-functional.sh
#
# Tests verify structural correctness of outputs, not content quality.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PASS=0
FAIL=0

check() {
  local desc="$1"
  local result="$2"
  if [ "$result" = "0" ]; then
    PASS=$((PASS + 1))
    echo "  PASS: $desc"
  else
    FAIL=$((FAIL + 1))
    echo "  FAIL: $desc"
  fi
}

echo "=== Skill Functional Tests ==="
echo ""

# --- terraform-automation ---
echo "--- terraform-automation ---"

# Check SKILL.md has required YAML frontmatter
grep -q '^---$' "$REPO_ROOT/.claude/skills/terraform-automation/SKILL.md"
check "SKILL.md has YAML frontmatter delimiters" "$?"

grep -q '^name: terraform-automation$' "$REPO_ROOT/.claude/skills/terraform-automation/SKILL.md"
check "SKILL.md has name field" "$?"

grep -q '^description:' "$REPO_ROOT/.claude/skills/terraform-automation/SKILL.md"
check "SKILL.md has description field" "$?"

# Check description includes trigger phrases
grep -q 'Do NOT use for' "$REPO_ROOT/.claude/skills/terraform-automation/SKILL.md"
check "Description includes negative triggers" "$?"

# Check references exist
test -f "$REPO_ROOT/.claude/skills/terraform-automation/references/terraform-patterns.md"
check "references/terraform-patterns.md exists" "$?"

# Check scripts exist
test -f "$REPO_ROOT/.claude/skills/terraform-automation/scripts/validate.sh"
check "scripts/validate.sh exists" "$?"

# Check validate.sh is executable-parseable
bash -n "$REPO_ROOT/.claude/skills/terraform-automation/scripts/validate.sh"
check "scripts/validate.sh passes syntax check" "$?"

# Check no duplicate steering files
test ! -d "$REPO_ROOT/.claude/skills/terraform-automation/steering"
check "No duplicate steering/ directory" "$?"

# Check troubleshooting section exists
grep -q '## Troubleshooting' "$REPO_ROOT/.claude/skills/terraform-automation/SKILL.md"
check "Troubleshooting section exists" "$?"

# Check success criteria section exists
grep -q '## Success Criteria' "$REPO_ROOT/.claude/skills/terraform-automation/SKILL.md"
check "Success Criteria section exists" "$?"

echo ""

# --- visual-explainer ---
echo "--- visual-explainer ---"

grep -q '^---$' "$REPO_ROOT/.claude/skills/visual-explainer/SKILL.md"
check "SKILL.md has YAML frontmatter delimiters" "$?"

grep -q '^name: visual-explainer$' "$REPO_ROOT/.claude/skills/visual-explainer/SKILL.md"
check "SKILL.md has name field" "$?"

grep -q 'Do NOT use for' "$REPO_ROOT/.claude/skills/visual-explainer/SKILL.md"
check "Description includes negative triggers" "$?"

# Check templates exist
for template in benchmark-comparison.html architecture.html audit-report.html; do
  test -f "$REPO_ROOT/.claude/skills/visual-explainer/templates/$template"
  check "Template $template exists" "$?"
done

# Check references exist
test -f "$REPO_ROOT/.claude/skills/visual-explainer/references/css-patterns.md"
check "references/css-patterns.md exists" "$?"

test -f "$REPO_ROOT/.claude/skills/visual-explainer/references/troubleshooting.md"
check "references/troubleshooting.md exists" "$?"

grep -q '## Troubleshooting' "$REPO_ROOT/.claude/skills/visual-explainer/SKILL.md"
check "Troubleshooting section exists" "$?"

grep -q '## Success Criteria' "$REPO_ROOT/.claude/skills/visual-explainer/SKILL.md"
check "Success Criteria section exists" "$?"

echo ""

# --- deployment-orchestrator ---
echo "--- deployment-orchestrator ---"

test -f "$REPO_ROOT/.claude/skills/deployment-orchestrator/SKILL.md"
check "SKILL.md exists" "$?"

grep -q '^---$' "$REPO_ROOT/.claude/skills/deployment-orchestrator/SKILL.md"
check "SKILL.md has YAML frontmatter delimiters" "$?"

grep -q '^name: deployment-orchestrator$' "$REPO_ROOT/.claude/skills/deployment-orchestrator/SKILL.md"
check "SKILL.md has name field" "$?"

grep -q 'Do NOT use for' "$REPO_ROOT/.claude/skills/deployment-orchestrator/SKILL.md"
check "Description includes negative triggers" "$?"

grep -q '## Pre-Flight Checklists' "$REPO_ROOT/.claude/skills/deployment-orchestrator/SKILL.md"
check "Pre-Flight Checklists section exists" "$?"

grep -q '## Failure Recovery Playbooks' "$REPO_ROOT/.claude/skills/deployment-orchestrator/SKILL.md"
check "Failure Recovery Playbooks section exists" "$?"

grep -q '## Success Criteria' "$REPO_ROOT/.claude/skills/deployment-orchestrator/SKILL.md"
check "Success Criteria section exists" "$?"

echo ""

# --- benchmark-runner ---
echo "--- benchmark-runner ---"

test -f "$REPO_ROOT/.claude/skills/benchmark-runner/SKILL.md"
check "SKILL.md exists" "$?"

grep -q '^---$' "$REPO_ROOT/.claude/skills/benchmark-runner/SKILL.md"
check "SKILL.md has YAML frontmatter delimiters" "$?"

grep -q '^name: benchmark-runner$' "$REPO_ROOT/.claude/skills/benchmark-runner/SKILL.md"
check "SKILL.md has name field" "$?"

grep -q 'Do NOT use for' "$REPO_ROOT/.claude/skills/benchmark-runner/SKILL.md"
check "Description includes negative triggers" "$?"

grep -q '## Phase Structure' "$REPO_ROOT/.claude/skills/benchmark-runner/SKILL.md"
check "Phase Structure section exists" "$?"

grep -q '## Benchmark Execution Rules' "$REPO_ROOT/.claude/skills/benchmark-runner/SKILL.md"
check "Benchmark Execution Rules section exists" "$?"

grep -q '## Metrics Checklist' "$REPO_ROOT/.claude/skills/benchmark-runner/SKILL.md"
check "Metrics Checklist section exists" "$?"

grep -q '## Troubleshooting' "$REPO_ROOT/.claude/skills/benchmark-runner/SKILL.md"
check "Troubleshooting section exists" "$?"

grep -q '## Success Criteria' "$REPO_ROOT/.claude/skills/benchmark-runner/SKILL.md"
check "Success Criteria section exists" "$?"

# Check references
for ref in benchmark-patterns.md metrics-reference.md troubleshooting.md; do
  test -f "$REPO_ROOT/.claude/skills/benchmark-runner/references/$ref"
  check "references/$ref exists" "$?"
done

# Check scripts
test -f "$REPO_ROOT/.claude/skills/benchmark-runner/scripts/benchmark-helpers.sh"
check "scripts/benchmark-helpers.sh exists" "$?"

bash -n "$REPO_ROOT/.claude/skills/benchmark-runner/scripts/benchmark-helpers.sh"
check "scripts/benchmark-helpers.sh passes syntax check" "$?"

test -f "$REPO_ROOT/.claude/skills/benchmark-runner/scripts/validate-results.sh"
check "scripts/validate-results.sh exists" "$?"

bash -n "$REPO_ROOT/.claude/skills/benchmark-runner/scripts/validate-results.sh"
check "scripts/validate-results.sh passes syntax check" "$?"

# Check templates
test -f "$REPO_ROOT/.claude/skills/benchmark-runner/templates/run-benchmarks.sh.tmpl"
check "templates/run-benchmarks.sh.tmpl exists" "$?"

test -f "$REPO_ROOT/.claude/skills/benchmark-runner/templates/custbench-runner.sh.tmpl"
check "templates/custbench-runner.sh.tmpl exists" "$?"

echo ""

# --- Blueprint artifact templates ---
echo "--- Blueprint Artifact Templates ---"

for bp_dir in "$REPO_ROOT"/domains/*/blueprints/*/; do
  bp_name=$(basename "$bp_dir")
  test -f "${bp_dir}lessons.md"
  check "[$bp_name] lessons.md exists" "$?"

  test -d "${bp_dir}results"
  check "[$bp_name] results/ directory exists" "$?"
done

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
exit $FAIL
