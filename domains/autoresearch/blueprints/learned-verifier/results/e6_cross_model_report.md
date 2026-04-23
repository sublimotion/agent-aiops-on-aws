# E6: Cross-Model Transfer Validation Results

**Date**: 2026-04-08
**Claude data**: 300 instances (175 pass)
**Qwen3.5 data**: 93 instances (11 pass)
**Features**: 3-feature ablation (no svg_accepted)

## Key Finding: Features Transfer, Thresholds Don't

The 3 behavioral features (`total_cost_usd`, `tokens_per_edit`, `loop_count`) carry real predictive signal within any model family, but the decision boundaries are model-specific. A verifier trained on Claude traces is **worse than random** on Qwen3.5 traces (AUC=0.363). This isn't noise — transfer fails in both directions.

**This means the verifier needs continuous learning, not one-shot training.**

## Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Zero-shot transfer (Claude RF → Qwen3.5) | **0.363** | > 0.65 | FAIL |
| Reverse transfer (Qwen3.5 RF → Claude) | **0.410** | > 0.65 | FAIL |
| Per-family ensemble AUC | **0.801** | > 0.72 | PASS |
| Model-agnostic 3-feat AUC (Claude CV) | **0.738** | > 0.70 | PASS |
| Bidirectional mean AUC | **0.387** | — | Below chance |

### svg_accepted Ablation

- 4-feature RF (with svg): AUC = 0.756
- 3-feature RF (no svg):   AUC = 0.738
- Drop: -0.019 (minimal — SVG is not load-bearing for the RF)

### Transfer by Harness

| Direction | AUC | n | Passes |
|-----------|-----|---|--------|
| Claude RF → Qwen3.5 (all) | 0.363 | 93 | 11 |
| Claude RF → Qwen3.5 SERA | 0.425 | 50 | 7 |
| Claude RF → Qwen3.5 OpenCode | 0.247 | 43 | 4 |
| Qwen3.5 RF → Claude | 0.410 | 300 | 175 |

OpenCode transfer is worst (AUC=0.247) — harness differences compound model differences.

### Ensemble vs Single RF

- Single RF (combined data, 5-fold CV): AUC = 0.751
- Per-model ensemble: AUC = **0.801**
- Delta: **+0.049** (ensemble wins)

## Why Transfer Fails: Feature Distribution Shift

| Feature | Claude mean | Claude std | Qwen3.5 mean | Qwen3.5 std | Shift ratio |
|---------|------------|-----------|-------------|------------|-------------|
| total_cost_usd | 0.37 | 0.22 | 0.40 | 0.60 | 1.08x |
| tokens_per_edit | 589K | 360K | 493K | 780K | 0.84x |
| loop_count | 14.9 | 5.7 | 22.4 | 23.0 | **1.50x** |

The means aren't wildly different, but the **variances** are. Qwen3.5 has 4x higher variance on loop_count and 2x on tokens_per_edit. Claude's RF learns tight decision boundaries that don't generalize to Qwen's wider spread.

The shift is qualitative, not just quantitative:
- Claude Code has fixed turn budgets → loop_count clusters tightly around 15
- OpenCode/SERA have variable turn limits → loop_count ranges 3-82
- A threshold like "loop_count > 20 = struggling" is true for Claude but meaningless for Qwen

## Architecture Implication: Continuous Learning

The verifier is not a static classifier. It's a continuously-learning system:

```
New model onboarded
  → v009 rubric handles cold start (model-agnostic, precision=0.92)
  → traces accumulate with rubric labels
  → at ~50-100 traces, train model-specific RF
  → RF handles easy cases, rubric handles edge cases
  → RF retrains periodically as traces accumulate
```

**Per-model routing** (`model_family` → model-specific RF) is required, not optional. The ensemble AUC=0.801 proves model-specific RFs outperform any universal model.

## Feature Approximation Caveats

Qwen3.5 features are approximations from swarm-level aggregates:
- `total_cost_usd`: tokens x Haiku rate proxy (self-hosted, no real pricing)
- `tokens_per_edit`: total tokens if fix_generated (no per-tool breakdown; Claude uses tokens/n_edits)
- `loop_count`: turns_used (Claude uses repeated-action loop detection)
- `svg_accepted`: excluded (Claude-only SVG pipeline)

These approximations may attenuate the transfer signal. However, the **per-model ensemble still succeeds** under these same approximations, confirming the core finding: signal exists per-model, not cross-model.
