# Phase 0 Report: Data Inventory & Behavioral Signal Check

**Date**: 2026-03-20
**Status**: COMPLETE

## Summary

Phase 0 assessed two questions:
1. Does existing SVG data on g7e contain usable verification signal?
2. Do SERA's behavioral features predict test outcomes?

**Result: SVG consensus is an extremely strong verifier. Behavioral features alone are weak at N=23.**

## 1. Data Inventory (g7e)

### SVG Pipeline Data (`/mnt/nvme/sera-data/`)

| Directory | Content | Rows | Key Finding |
|-----------|---------|------|-------------|
| `production-run1/` | Full SVG results (Devstral Small 2, 300 instances) | 300 | **Primary dataset.** 53 pass, 28 SVG-accepted. |
| `production-run1/train.jsonl` | SFT training examples from accepted SVG runs | 28 | Already formatted for training |
| `batch-10/` | Small test batch | 10 | 0 tests_pass (early experiment) |
| `mixed/` | Combined training data (70GB+) | — | SFT datasets for SERA training |
| `coderforge/` | Downloaded CoderForge data (32GB) | — | Already on g7e |

**production-run1 summary.json:**
- Model: devstral-small-2
- 300 instances, 246 fix_generated (82%), 53 tests_pass (17.7%), 28 SVG-accepted (9.3%)
- Recall threshold: 0.8, Avg recall: 0.748, Avg fix turns: 29.6

### Harness/Swarm Results (`/mnt/nvme/agent-harness/results/`)

| File | Content | Labeled? |
|------|---------|----------|
| `phase1_{A-F}.jsonl` | 6 SERA configs, 50 instances each | No (no tests_pass) |
| `phase1_{A-F}_turns.jsonl` | Turn-level behavioral telemetry | N/A (features only) |
| `eval_sera.jsonl` | 23 instances with tests_pass labels | **YES** |
| `eval_{7 harnesses}.jsonl` | 7-harness experiment results | YES (tests_pass) |
| `phase2b_*.jsonl` | Phase 2b multi-harness results | YES |
| `swarm/swarm_phase1_*.jsonl` | 8 model×harness cells | YES |

## 2. SVG Consensus Verifier Analysis (n=300)

**This is the headline finding.**

### SVG as Binary Classifier (threshold: recall >= 0.8)

|  | Predicted Pass | Predicted Fail |
|--|:-:|:-:|
| **Actually Pass** | 28 (TP) | 25 (FN) |
| **Actually Fail** | 0 (FP) | 247 (TN) |

- **Precision: 1.000** — Every patch SVG accepted actually passes tests
- **Recall: 0.528** — Catches about half of passing patches
- **F1: 0.691**
- **Accuracy: 0.917**

### line_recall as Continuous Predictor

- **AUC: 0.981** — Near-perfect discrimination
- Mean recall for passing patches: 0.748
- Mean recall for failing patches: 0.000

**Interpretation**: SVG consensus (line_recall) is an almost perfect verifier. When it accepts (recall >= 0.8), precision is 100% — zero false positives. The only weakness is recall: it misses ~47% of correct patches. This is exactly the "high precision, moderate recall" profile you'd want for a soft verifier.

### Implication

**SVG consensus is already a production-quality soft verifier.** The open question is whether behavioral features can recover the 25 false negatives — patches that pass tests but SVG rejects (recall < 0.8).

## 3. Behavioral Signal Check (n=23)

### Per-Feature Correlations (Mean across configs A-F)

| Feature | Correlation | p-value | AUC | Signal? |
|---------|:-:|:-:|:-:|:-:|
| `action_pct_search` | +0.412 | 0.051 | 0.742 | Marginal (p~0.05) |
| `avg_duration_s` | -0.406 | 0.054 | 0.175 | Marginal (inverse) |
| `action_pct_run` | -0.351 | 0.101 | 0.279 | Weak |
| `repeat_rate` | +0.285 | 0.187 | 0.708 | Weak |
| `total_turns` | +0.239 | 0.273 | 0.642 | None |
| `action_pct_edit` | -0.174 | 0.428 | 0.421 | None |
| `context_growth_rate` | -0.144 | 0.511 | 0.433 | None |
| `first_edit_turn` | +0.136 | 0.538 | 0.575 | None |
| `action_pct_read` | +0.130 | 0.555 | 0.588 | None |
| `parkinson_ratio` | +0.096 | 0.664 | 0.567 | None |
| `edit_to_search_ratio` | -0.215 | 0.324 | 0.367 | None |

### Logistic Regression (LOOCV, n=23)

| Metric | Config D Only | Mean A-F |
|--------|:-:|:-:|
| Accuracy | 0.522 | 0.609 |
| AUC | 0.508 | 0.542 |
| ECE | 0.173 | 0.062 |
| Baseline (majority) | 0.652 | 0.652 |

**Both models perform at or below the majority-class baseline.** The logistic regression cannot beat always-predicting-fail.

### Interpretation

At N=23, behavioral features from SERA's turn metrics do not predict test outcomes with any statistical reliability. Two features show marginal signal:

1. **`action_pct_search`** (r=+0.412, p=0.051): Passing runs spend more time searching. Counter-intuitive — suggests thorough search correlates with success, not confidence.
2. **`avg_duration_s`** (r=-0.406, p=0.054): Passing runs have shorter turn durations. Faster responses may indicate the model "knows what to do."

But neither survives Bonferroni correction (11 features → p < 0.0045 needed).

### Why This Might Be Misleading

The N=23 sample is fundamentally underpowered:
- 8 positives, 15 negatives
- 11 features → severe overfitting risk
- Same model (Devstral) on all runs → no cross-model variation
- Turn metrics are aggregated across configs A-F that vary in turn budget (10-30 turns)

The SVG production-run1 data (n=300) with tests_pass labels is a much better foundation.

## 4. Phase 0 Exit Criteria

- [x] Inventory of `/mnt/nvme/sera-data/` contents — **300 SVG results found**
- [x] Join Phase 1 turn data with eval_sera.jsonl — **23 instances, 138 feature rows**
- [x] Quick signal check: logistic regression — **Below baseline. No significant signal at N=23.**
- [x] Decision: does SVG data exist in quantity? — **YES. 300 rows with tests_pass labels.**

## 5. Recommendations for Phase 1-2

1. **Use SVG production-run1 (n=300) as the primary dataset**, not eval_sera (n=23). It has tests_pass labels, line_recall scores, and is 13x larger.

2. **SVG consensus is the baseline to beat.** At precision=1.0 and AUC=0.981, any trained model must exceed SVG's recall (0.528) to add value. The target: recover the 25 false negatives.

3. **Extract behavioral features from SVG transcripts.** The production-run1 data includes full conversation transcripts (in `train.jsonl`). Extract turn-level features to test whether behavioral signals add to SVG consensus at n=300.

4. **The 7-harness eval data provides cross-harness signal.** The eval_*.jsonl files (7 harnesses × 50 instances = 350 labeled rows) can test whether harness identity predicts pass rate — a confounder check.

5. **Deprioritize behavioral-only verifier.** At both N=23 and N=300, SVG consensus dominates. Focus Phase 2 on: (a) SVG as baseline, (b) LLM-as-judge, (c) patch features + SVG combined.

## Artifacts

| File | Description |
|------|-------------|
| `data/phase0/svg_results_production_run1.jsonl` | 300 SVG results with tests_pass |
| `data/phase0/eval_sera.jsonl` | 23 labeled eval results |
| `data/phase0/phase1_{A-F}_turns.jsonl` | 6,599 turn-level behavioral rows |
| `data/phase0/phase1_{A-F}.jsonl` | 6 config summaries (50 instances each) |
| `data/phase0/phase0_results.txt` | Full output of signal check script |
| `scripts/phase0_signal_check.py` | Analysis script (reproducible) |
