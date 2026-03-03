#!/bin/bash
# Skill Trigger Tests
# Validates that skills activate for the right prompts and don't activate for wrong ones.
#
# Usage: ./test-triggers.sh
#
# This script defines test cases. Each case is a prompt and the expected skill(s) that should trigger.
# Run manually by pasting prompts into Claude and checking which skill loads.
# Future: automate via Skills API.

set -e

echo "=== Skill Trigger Tests ==="
echo ""

PASS=0
FAIL=0
TOTAL=0

# Helper: record test case (manual verification)
test_case() {
  local skill="$1"
  local should_trigger="$2"  # "yes" or "no"
  local prompt="$3"
  TOTAL=$((TOTAL + 1))
  echo "  [$TOTAL] Skill: $skill | Should trigger: $should_trigger"
  echo "       Prompt: \"$prompt\""
}

echo "--- terraform-automation ---"
echo ""
echo "Should trigger:"
test_case "terraform-automation" "yes" "Create a production VPC with Terraform"
test_case "terraform-automation" "yes" "Run Checkov scan on my infrastructure code"
test_case "terraform-automation" "yes" "Deploy this Terraform configuration to AWS"
test_case "terraform-automation" "yes" "Validate my Terraform files"
test_case "terraform-automation" "yes" "Debug this terraform apply failure"
test_case "terraform-automation" "yes" "Create an S3 bucket with encryption using Terraform"

echo ""
echo "Should NOT trigger:"
test_case "terraform-automation" "no" "Explain what this main.tf file does"
test_case "terraform-automation" "no" "What is the difference between Terraform and CloudFormation?"
test_case "terraform-automation" "no" "Write a Python script to parse JSON"
test_case "terraform-automation" "no" "Create a CDK stack for my application"

echo ""
echo "--- visual-explainer ---"
echo ""
echo "Should trigger:"
test_case "visual-explainer" "yes" "Visualize the benchmark results"
test_case "visual-explainer" "yes" "Generate a diagram of this architecture"
test_case "visual-explainer" "yes" "Create an audit visual from the readiness report"
test_case "visual-explainer" "yes" "Show me a compound recap of the deployment"
test_case "visual-explainer" "yes" "Turn this benchmark report into an interactive HTML page"

echo ""
echo "Should NOT trigger:"
test_case "visual-explainer" "no" "Write a benchmark report in markdown"
test_case "visual-explainer" "no" "Edit the existing HTML template"
test_case "visual-explainer" "no" "Build a React dashboard"
test_case "visual-explainer" "no" "Create a readiness audit markdown file"

echo ""
echo "--- deployment-orchestrator ---"
echo ""
echo "Should trigger:"
test_case "deployment-orchestrator" "yes" "Run the pre-flight checklist before the capacity block"
test_case "deployment-orchestrator" "yes" "Do a readiness audit of the deployment"
test_case "deployment-orchestrator" "yes" "The deployment failed, how do I recover?"
test_case "deployment-orchestrator" "yes" "Validate the deployment is working correctly"
test_case "deployment-orchestrator" "yes" "What's the deployment checklist for GPU serving?"

echo ""
echo "Should NOT trigger:"
test_case "deployment-orchestrator" "no" "Create a new VPC with Terraform"
test_case "deployment-orchestrator" "no" "Analyze the benchmark results"
test_case "deployment-orchestrator" "no" "Write a new spec for a model"
test_case "deployment-orchestrator" "no" "What is a capacity block?"

echo ""
echo "=== Total test cases: $TOTAL ==="
echo ""
echo "To run: paste each prompt into Claude and verify the correct skill loads."
echo "Track results in .claude/skills/tests/results/trigger-results-<date>.md"
echo ""
echo "Target: 90%+ correct triggers, 0% false triggers on negative cases."
