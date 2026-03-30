#!/usr/bin/env bash
# Collect HTML reports into docs/reports/ and push to upstream (public repo) gh-pages branch.
# Usage: ./scripts/publish-reports.sh [--dry-run]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCS="$REPO_ROOT/docs"
REPORTS="$DOCS/reports"
DRY_RUN=false

[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

echo "==> Collecting reports into $REPORTS"
rm -rf "$REPORTS"
mkdir -p "$REPORTS"

copied=0
skipped=0

copy_report() {
  local src="$1" dst="$2"
  if [[ -f "$REPO_ROOT/$src" ]]; then
    cp "$REPO_ROOT/$src" "$REPORTS/$dst"
    echo "  + $dst"
    copied=$((copied + 1))
  else
    echo "  - SKIP (not found): $src"
    skipped=$((skipped + 1))
  fi
}

# GPU Serving
copy_report "domains/gpu-serving/blueprints/ray-serve-ft/results/ray-ft-visual-20260323.html" "ray-ft-visual-20260323.html"
copy_report "domains/gpu-serving/blueprints/glm5-lmcache/results/benchmark-visual-20260307.html" "benchmark-visual-glm5-lmcache-20260307.html"
copy_report "domains/gpu-serving/blueprints/glm5-llmd/results/benchmark-visual-20260307.html" "benchmark-visual-glm5-llmd-20260307.html"
copy_report "domains/gpu-serving/blueprints/qwen3-next-sglang/results/benchmark-visual-20260303.html" "benchmark-visual-qwen3-next-sglang-20260303.html"
copy_report "domains/gpu-serving/blueprints/qwen3-next-custbench/results/session-20260226/benchmark-visual-report.html" "benchmark-visual-report-custbench-20260226.html"
copy_report "domains/gpu-serving/blueprints/qwen3-next-g7e/results/benchmark-visual-20260225.html" "benchmark-visual-qwen3-next-g7e-20260225.html"
copy_report "domains/gpu-serving/blueprints/qwen3-next/results/benchmark-visual-20260224.html" "benchmark-visual-qwen3-next-20260224.html"
copy_report "domains/gpu-serving/blueprints/kimi-k2.5/results/benchmark-visual-20260221.html" "benchmark-visual-kimi-k2.5-20260221.html"
copy_report "domains/gpu-serving/blueprints/nemotron-super/results/benchmark-report.html" "benchmark-report-nemotron-super.html"
copy_report "domains/gpu-serving/blueprints/ray-serve-video/results/benchmark-visual-report.html" "benchmark-visual-ray-serve-video-20260327.html"

# Autoresearch
copy_report "domains/autoresearch/blueprints/verifier-reward/results/verifier-reward-visual.html" "verifier-reward-visual.html"
copy_report "domains/autoresearch/blueprints/agent-harness/results/sera32b-coalignment-20260322.html" "sera32b-coalignment-20260322.html"
copy_report "domains/autoresearch/blueprints/agent-harness/results/thunderagent-phase2b-20260320.html" "thunderagent-phase2b-20260320.html"
copy_report "domains/autoresearch/blueprints/agent-harness/results/agent-swarm-20260319.html" "agent-swarm-20260319.html"
copy_report "domains/autoresearch/blueprints/agent-harness/results/visual-explainer-20260318.html" "visual-explainer-harness-20260318.html"
copy_report "domains/autoresearch/blueprints/agent-harness/results/visual-explainer-20260315.html" "visual-explainer-harness-20260315.html"
copy_report "domains/autoresearch/blueprints/training-recipes/results/benchmark-report.html" "benchmark-report-training-recipes.html"
copy_report "domains/autoresearch/blueprints/finetuning-recipes/results/benchmark-report.html" "benchmark-report-finetuning-recipes.html"
copy_report "domains/autoresearch/blueprints/verification-primitives/results/verification-primitives-visual.html" "verification-primitives-visual.html"

# Agent Runtime
copy_report "domains/agent-runtime/blueprints/research-agent/results/audit-visual-20260223.html" "audit-visual-research-agent-20260223.html"
copy_report "domains/agent-runtime/blueprints/research-agent/results/architecture-research-agent-20260223.html" "architecture-research-agent-20260223.html"

# Site (cross-domain)
copy_report "site/compatibility-matrix.html" "compatibility-matrix.html"
copy_report "site/model-comparison.html" "model-comparison.html"
copy_report "site/provider-comparison.html" "provider-comparison.html"
copy_report "site/benchmark-report.html" "site-benchmark-report.html"

echo "==> Collected $copied reports ($skipped skipped)"

if $DRY_RUN; then
  echo "==> Dry run — not pushing. Files are in $REPORTS"
  exit 0
fi

# Push docs/ to upstream gh-pages using a temporary directory
echo "==> Pushing to upstream gh-pages branch"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

cd "$TMPDIR"
git init -q
git checkout -q -b gh-pages
cp -r "$DOCS/"* .
git add -A
git commit -q -m "Update report card site $(date +%Y-%m-%d)"

UPSTREAM_URL=$(cd "$REPO_ROOT" && git remote get-url upstream)
git remote add upstream "$UPSTREAM_URL"
git push -f upstream gh-pages

echo "==> Done! Enable GitHub Pages on gh-pages branch at:"
echo "    https://github.com/sublimotion/agent-aiops-on-aws/settings/pages"
