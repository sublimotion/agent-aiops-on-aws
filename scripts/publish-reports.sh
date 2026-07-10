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
# Preserve cross-domain files that have no source under domains/
BACKUP="$DOCS/reports.bak"
rm -rf "$BACKUP"
mkdir -p "$BACKUP"
for f in pareto-frontier.html benchmark-visual-deepseek-v4-flash-20260519.html benchmark-visual-kimi-k2.6-nvfp4-20260617.html inference-bottleneck-migration-20260612.html coding-agents-modelcards-20260629.html; do
  [[ -f "$REPORTS/$f" ]] && cp "$REPORTS/$f" "$BACKUP/$f"
done
rm -rf "$REPORTS"
mkdir -p "$REPORTS"
trap 'rm -rf "$BACKUP"' EXIT

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
copy_report "domains/gpu-serving/blueprints/glm5.2/results/benchmark-visual-report.html" "benchmark-visual-glm5.2-20260623.html"
copy_report "domains/gpu-serving/blueprints/glm5.2/results/benchmark-visual-report-b300.html" "benchmark-visual-glm5.2-b300-20260627.html"
copy_report "domains/gpu-serving/blueprints/minimax-m2/results/report.html" "benchmark-visual-minimax-m2-20260628.html"
copy_report "domains/gpu-serving/blueprints/qwen3-next-sglang/results/benchmark-visual-20260303.html" "benchmark-visual-qwen3-next-sglang-20260303.html"
copy_report "domains/gpu-serving/blueprints/qwen3-next-custbench/results/session-20260226/benchmark-visual-report.html" "benchmark-visual-report-custbench-20260226.html"
copy_report "domains/gpu-serving/blueprints/qwen3-next-g7e/results/benchmark-visual-20260225.html" "benchmark-visual-qwen3-next-g7e-20260225.html"
copy_report "domains/gpu-serving/blueprints/qwen3-next/results/benchmark-visual-20260224.html" "benchmark-visual-qwen3-next-20260224.html"
copy_report "domains/gpu-serving/blueprints/kimi-k2.5/results/benchmark-visual-20260221.html" "benchmark-visual-kimi-k2.5-20260221.html"
copy_report "domains/gpu-serving/blueprints/nemotron-super/results/benchmark-report.html" "benchmark-report-nemotron-super.html"
copy_report "domains/gpu-serving/blueprints/nemotron-ultra/results/benchmark-visual-report.html" "benchmark-visual-nemotron-ultra-20260606.html"
copy_report "domains/gpu-serving/blueprints/ray-serve-video/results/benchmark-visual-report.html" "benchmark-visual-ray-serve-video-20260327.html"
copy_report "domains/gpu-serving/blueprints/qwen3-235b-b300/results/benchmark-visual-report.html" "benchmark-visual-qwen3-235b-b300-20260422.html"
copy_report "domains/gpu-serving/blueprints/kimi-k2.6-speculative/docs/roofline-explainer.html" "roofline-explainer-kimi-k2.6.html"
copy_report "domains/gpu-serving/blueprints/qwen3-235b-speculative/docs/benchmark-report.html" "benchmark-visual-qwen3-235b-speculative.html"

# GPU Serving — HyperPod & EKS
copy_report "domains/gpu-serving/blueprints/gemma4-hyperpod/results/hyperpod-3model-visual-20260407.html" "hyperpod-3model-visual-20260407.html"
copy_report "domains/gpu-serving/blueprints/qwen3-embedding-8b-hyperpod/results/visual-report.html" "qwen3-embedding-8b-hyperpod-visual.html"
copy_report "domains/gpu-serving/blueprints/qwen3-reranker-4b-eks/results/visual-report.html" "qwen3-reranker-4b-eks-visual.html"
copy_report "domains/gpu-serving/blueprints/voxtral-4b-eks/results/visual-report.html" "voxtral-4b-eks-visual.html"
copy_report "domains/gpu-serving/blueprints/deepseek-ocr-2-eks/results/visual-report.html" "deepseek-ocr-2-eks-visual.html"

# Autoresearch
copy_report "domains/autoresearch/blueprints/verifier-reward/results/verifier-reward-visual.html" "verifier-reward-visual.html"
copy_report "domains/autoresearch/blueprints/agent-harness/results/sera32b-coalignment-20260322.html" "sera32b-coalignment-20260322.html"
copy_report "domains/autoresearch/blueprints/agent-harness/results/thunderagent-phase2b-20260320.html" "thunderagent-phase2b-20260320.html"
copy_report "domains/autoresearch/blueprints/agent-harness/results/agent-swarm-20260319.html" "agent-swarm-20260319.html"
copy_report "domains/autoresearch/blueprints/agent-harness/results/visual-explainer-20260318.html" "visual-explainer-harness-20260318.html"
copy_report "domains/autoresearch/blueprints/training-recipes/results/benchmark-report.html" "benchmark-report-training-recipes.html"
copy_report "domains/autoresearch/blueprints/finetuning-recipes/results/benchmark-report.html" "benchmark-report-finetuning-recipes.html"
copy_report "domains/autoresearch/blueprints/verification-primitives/results/verification-primitives-consolidated.html" "verification-primitives-consolidated.html"
copy_report "domains/autoresearch/blueprints/verification-flywheel/results/flywheel-visual-20260426.html" "flywheel-visual-20260426.html"
copy_report "domains/autoresearch/blueprints/learned-verifier/results/verification-visual-explainer.html" "learned-verifier-visual.html"
copy_report "domains/autoresearch/blueprints/tiny-judge/results/judge-visual.html" "tiny-judge-visual.html"
copy_report "domains/autoresearch/blueprints/pivot-analysis/results/pivot-visual.html" "pivot-analysis-visual.html"
copy_report "domains/autoresearch/blueprints/self-coding-agent-loop/spec-explainer.html" "self-coding-agent-loop-visual.html"
copy_report "domains/autoresearch/blueprints/trinity-coordinator/docs/explainer.html" "trinity-coordinator-visual.html"

# AI Infra — cold-start lab. Filenames kept bare because the reports cross-link
# each other by bare filename; the recap + tiers pages ship so those links resolve
# even though only the progress report and explainer have index cards.
copy_report "domains/ai-infra/reports/cold-start-progress-report.html" "cold-start-progress-report.html"
copy_report "domains/ai-infra/reports/cold-start-explainer.html" "cold-start-explainer.html"
copy_report "domains/ai-infra/reports/dynamo-snapshot-recap.html" "dynamo-snapshot-recap.html"
copy_report "domains/ai-infra/reports/optimization-tiers.html" "optimization-tiers.html"

# Cross-domain (existing files in docs/reports/ — preserved across rebuilds)
for f in pareto-frontier.html benchmark-visual-deepseek-v4-flash-20260519.html benchmark-visual-kimi-k2.6-nvfp4-20260617.html inference-bottleneck-migration-20260612.html coding-agents-modelcards-20260629.html; do
  if [[ -f "$DOCS/reports.bak/$f" ]]; then cp "$DOCS/reports.bak/$f" "$REPORTS/$f"; copied=$((copied+1)); echo "  + $f (preserved)"; fi
done

echo "==> Collected $copied reports ($skipped skipped)"

# Auto-sync config.json visibility allow-list from the index's cards.
# index.html's JS hides any card whose data-id is NOT in config.json "visible".
# Regenerating it from the index here makes the index the single source of truth,
# so a new card can never be silently hidden by a stale config (the 2026-06-26 bug).
echo "==> Syncing config.json visibility from index.html cards"
python3 - "$DOCS/index.html" "$DOCS/config.json" <<'PYSYNC'
import json, re, sys
index_path, config_path = sys.argv[1], sys.argv[2]
ids = re.findall(r'data-id="([^"]+)"', open(index_path).read())
try:
    cfg = json.load(open(config_path))
except Exception:
    cfg = {}
before = set(cfg.get("visible", []))
cfg["visible"] = ids  # exact index card set, in document order
json.dump(cfg, open(config_path, "w"), indent=2)
after = set(ids)
added, removed = sorted(after - before), sorted(before - after)
print(f"  config.visible: {len(ids)} ids"
      + (f" | +{len(added)} added" if added else "")
      + (f" | -{len(removed)} removed" if removed else "")
      + (" | in sync" if not added and not removed else ""))
PYSYNC

if $DRY_RUN; then
  echo "==> Dry run — not pushing. Files are in $REPORTS"
  exit 0
fi

# Push docs/ to upstream gh-pages using a temporary directory
echo "==> Pushing to upstream gh-pages branch"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR" "$BACKUP"' EXIT

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
