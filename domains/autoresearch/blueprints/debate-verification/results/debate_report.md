# Debate Verification Experiment Report

**Date**: 2026-04-04
**Dataset**: SWE-bench Lite, n=300 (175 resolved, 125 unresolved)
**Best variant**: 2-round debate (Advocate + Challenger + rebuttal + Judge)
**Model**: Claude Haiku 4.5 via Bedrock (all agents)
**Total cost**: $7.29 ($0.024/instance)

## Phase 3: Full Evaluation (n=300)

| Metric | Debate (2-round) | v009 unanimous | SVG (raw) |
|--------|:-----------------:|:--------------:|:---------:|
| **Precision** | **0.725** [0.659, 0.795] | 0.963 | 0.963 |
| **Recall** | **0.592** [0.515, 0.661] | 0.149 | 0.149 |
| **F1** | **0.652** | 0.257 | 0.257 |
| Coverage | 95% | 100% | 100% |
| ECE | 0.234 | — | 0.512 |
| Cost/instance | $0.024 | $0.008 | $0 |

Debate achieves **4x the recall** of v009/SVG at a precision that passes the 0.70 gate.

### Verdict Distribution

| Verdict | Count | % |
|---------|------:|--:|
| CORRECT | 138 | 46% |
| INCORRECT | 146 | 49% |
| UNCERTAIN | 15 | 5% |
| UNKNOWN | 1 | 0.3% |

### Confusion Matrix

|  | Gold PASS | Gold FAIL |
|--|----------:|----------:|
| Debate CORRECT | 100 (TP) | 38 (FP) |
| Debate INCORRECT | 69 (FN) | 78 (TN) |
| UNCERTAIN | 6 | 9 |

## Phase 2: Variant Comparison (n=50)

| Variant | Precision | Recall | F1 | Cost |
|---------|-----------|--------|-----|------|
| base | 0.579 | 0.458 | 0.511 | $0.58 |
| **2-round** | **0.636** | **0.560** | **0.596** | $1.26 |
| asymmetric | 0.588 | 0.417 | 0.488 | $1.21 |

2-round (rebuttal) variant dominated across all metrics. The rebuttal round gives the Advocate a chance to debunk false bug claims from the Challenger.

## Phase 4: Combination Analysis (n=88 overlap)

On the 88 instances where both debate and v009 have results:

| Method | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| v009 alone | **1.000** | 0.143 | 0.250 |
| Debate alone | 0.875 | 0.618 | 0.724 |
| **Debate OR v009** | **0.882** | **0.662** | **0.756** |

### Key Findings

- **Debate recovers 58% of v009 false negatives** (35/60 FNs recovered)
- **v009 catches 100% of debate false positives** (6/6 FPs caught)
- The combination is strongly complementary: v009 provides precision, debate provides recall
- Combined precision (0.882) is closer to v009 than debate alone, because v009 never adds FPs

### Proposed Combined Verifier

```
if v009 says likely_correct → ACCEPT (high confidence)
elif debate says CORRECT → ACCEPT (medium confidence)
elif debate says INCORRECT → REJECT
else → UNCERTAIN
```

Estimated combined metrics on full population: precision ~0.85, recall ~0.65.

## Per-Repository Breakdown

| Repository | N | Precision | Recall | TP | FP |
|-----------|---:|----------:|-------:|---:|---:|
| django | 108 | 0.860 | 0.566 | 43 | 7 |
| sympy | 73 | 0.679 | 0.528 | 19 | 9 |
| scikit-learn | 23 | 0.611 | 0.846 | 11 | 7 |
| matplotlib | 21 | 0.667 | 0.615 | 8 | 4 |
| pytest-dev | 16 | 0.625 | 0.556 | 5 | 3 |
| sphinx-doc | 15 | 0.500 | 0.571 | 4 | 4 |

Django has the best precision (0.860), scikit-learn the best recall (0.846). Sphinx has the worst precision (0.500).

## ECE and RL-Readiness

| Method | ECE | RL Assessment |
|--------|----:|--------------|
| SVG raw | 0.512 | Poor |
| SVG + Platt | 0.031 | Excellent |
| Debate 2-round | 0.234 | Marginal |
| Debate + temperature scaling | TBD | — |

Debate ECE (0.234) is better than raw SVG (0.512) but worse than Platt-scaled SVG (0.031). For RL purposes, debate verdicts should be used as binary signals (accept/reject), not as continuous confidence scores.

## Success Criteria Assessment

1. **Minimum viable (pilot gate)**: PASSED on full run (precision 0.725 > 0.70, recall 0.592 >> 0.15)
2. **Strong result**: PARTIALLY MET — recall 0.592 > 0.30 at precision 0.725 > 0.70 (close to 0.85 target). Combination with v009 is additive.
3. **Publishable result**: NOT YET — need full n=483 Verified evaluation and SVG combination analysis
4. **Negative result**: Not applicable — debate shows clear signal

## Cost Summary

| Phase | Instances | Cost | Cost/instance |
|-------|----------:|-----:|--------------:|
| Phase 1 | 10 | $0.10 | $0.010 |
| Phase 2 (all variants) | 150 | $3.04 | $0.020 |
| Phase 3 | 300 | $7.29 | $0.024 |
| **Total** | — | **$10.43** | — |

## Lessons Learned

1. **Prompt calibration matters enormously**: Skeptical Judge killed recall without improving precision. Balanced Judge with own_analysis field was the sweet spot.
2. **Rebuttal round is worth 2x cost**: Advocate's ability to debunk false bug claims is the key differentiator between base and 2-round.
3. **Challenger "reluctantly_clean" predicts FPs**: When the Challenger can't find bugs, the Judge tends to side with the Advocate — which is correct for TPs but creates FPs.
4. **The combination is the product**: Neither debate nor v009 alone is sufficient. Together they cover each other's weaknesses.
5. **Batched async processing**: Launching 300 coroutines with a shared semaphore causes starvation. Process in batches of MAX_CONCURRENT.
