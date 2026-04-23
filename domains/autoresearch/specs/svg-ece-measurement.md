# Autoresearch Spec: SVG ECE Measurement

## Status: DRAFT

## Overview

Measure the Expected Calibration Error (ECE) of SVG consensus scores on our n=300 SWE-bench Lite dataset with gold labels from Docker evaluation. This is a pure analysis experiment — zero API cost, zero compute — that answers a critical gating question: are SVG scores calibrated enough to serve as RL reward signals?

**Core hypothesis**: SVG consensus scores have ECE < 0.1, making them viable as reward signals for RL post-training (Phase 4 on the verification spectrum).

**Motivation**: SWE-RM's headline finding (arXiv:2512.21919) — two verifiers with nearly identical ranking performance (+4.7% vs +4.5% TTS) had 7x ECE difference (0.078 vs 0.541). The poorly calibrated one caused RL training collapse. Ranking accuracy (AUC) is insufficient; calibration (ECE) determines whether a verifier is RL-ready. Our SVG has AUC 0.981 and precision 1.000, but we have never measured ECE.

**Depends on**: verification-primitives-swebench (Docker eval gold labels), verification-primitives (SVG consensus scores from Phase 0)

## Components

### 1. Compute
- **Platform**: Local laptop — pure Python analysis
- **Instance Type**: N/A
- **GPUs**: None required

### 2. Codebase
- **Source**: New analysis script in blueprint directory
- **Fixed files**:
  - SVG consensus scores: `blueprints/verification-primitives/results/production-run1/svg_results.jsonl` (n=300, continuous scores)
  - Gold labels: `blueprints/verification-primitives-swebench/results/eval_report.json` + `eval_report_errors_v2.json` (combined 175/300 resolved)
- **Agent-editable files**:
  - `blueprints/svg-ece-measurement/scripts/ece_analysis.py`
- **Agent instructions**: N/A — manual analysis

### 3. Experiment Protocol
- **Metric**: ECE (Expected Calibration Error) — lower is better. Target: < 0.1
- **Secondary metrics**: MCE (Maximum Calibration Error), reliability diagram, Brier score
- **Time budget**: 1 hour (data loading + analysis + visualization)
- **Loop structure**: Single-pass analysis, no iteration needed
- **Termination**: Analysis complete when ECE computed and reliability diagram generated
- **Logging**: Results in `blueprints/svg-ece-measurement/results/ece_report.md`

### 4. Networking
- **Access**: Local — no API calls

### 5. Storage
- **Data**: Existing results from verification-primitives and verification-primitives-swebench blueprints
- **Results**: `blueprints/svg-ece-measurement/results/`

## Analysis Design

### Step 1: Join SVG Scores with Gold Labels

For each of the 300 SWE-bench Lite instances:
- SVG consensus score: continuous value from `svg_results.jsonl` (line_recall at threshold)
- Gold label: resolved (1) or unresolved (0) from Docker eval reports (v1 + v2 combined)

### Step 2: Compute ECE

```
ECE = Σ (|B_m| / n) * |accuracy(B_m) - confidence(B_m)|
```

Where B_m are 10 equal-width bins of SVG scores, accuracy is the fraction of gold-passing patches in each bin, and confidence is the mean SVG score in each bin.

### Step 3: Additional Calibration Metrics

- **MCE** (Maximum Calibration Error): max over bins of |accuracy - confidence|
- **Brier Score**: mean squared error between SVG score and binary gold outcome
- **Reliability Diagram**: plot accuracy vs confidence per bin with sample counts

### Step 4: RL-Readiness Assessment

| ECE Range | Assessment | Action |
|-----------|-----------|--------|
| < 0.05 | Excellent calibration | SVG ready for RL reward signal |
| 0.05 - 0.10 | Acceptable | SVG usable for RL with temperature scaling |
| 0.10 - 0.20 | Marginal | Use for rejection sampling SFT only, not RL |
| > 0.20 | Poor | Recalibrate via Platt scaling before any use |

### Step 5: Conditional Analysis

If ECE > 0.1, test whether simple recalibration fixes it:
- **Platt scaling**: logistic regression on SVG scores → calibrated probabilities
- **Temperature scaling**: single parameter T on logit(SVG_score)
- Re-measure ECE after recalibration

## Success Criteria

1. ECE computed and reported with confidence interval (bootstrap, 1000 samples)
2. Reliability diagram generated showing calibration per bin
3. RL-readiness assessment determined (go/no-go for SVG as RL reward)
4. If ECE > 0.1: recalibration tested and post-calibration ECE reported
5. Results documented in format compatible with the Learned Verifier Experiment project notes

## Non-Requirements
- Any API calls or model inference
- New data collection
- RL training infrastructure
- Anything beyond analysis of existing data

## Known Limitations
- n=300 is small for 10-bin ECE — some bins may have < 10 samples. Report per-bin sample counts and use adaptive binning if needed.
- SVG scores from Phase 0 were computed on the verification-primitives harness (custom SERA), not Claude Code. The Docker eval gold labels are from the Claude Code + primitives run. If the SVG scores and gold labels come from different runs, the join may not be valid — verify instance IDs match.
- ECE is sensitive to binning strategy. Report both equal-width and equal-count (quantile) binning.

## Relationship to Other Specs

- **verification-primitives**: Source of SVG consensus scores (Phase 0, n=300)
- **verification-primitives-swebench**: Source of Docker eval gold labels (175/300 resolved)
- **verifier-reward**: v009 adversarial rubric provides comparison point — compute v009 ECE alongside SVG ECE

## Key References

- SWE-RM (arXiv:2512.21919) — calibration > ranking for RL integration
- SERA (arXiv:2601.20789) — SVG consensus methodology
- Naeini et al. (2015) — ECE definition and reliability diagrams

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
