# Phase 3: Combined Learned Verifier Report

**Dataset**: VP SWE-bench production eval
**Instances**: 300 (pass=175, fail=125)
**Evaluation**: 5-fold stratified cross-validation
**Signal sources**: Behavioral (tiny-judge) + v009 rubric + Debate verdicts + SVG consensus

## Best Model

**selected_4__RandomForest**: AUC=0.756, F1=0.724, P@R30=0.966, ECE=0.083

## Improvement Over Baselines

| Baseline | AUC | P@R≥30% | ECE | F1 |
|----------|-----|---------|-----|----|
| v009-only (RF) | 0.682 | 0.758 | 0.092 | 0.706 |
| Behavioral-only (RF) | 0.730 | 0.826 | 0.029 | 0.754 |
| **selected_4__RandomForest** | **0.756** | **0.966** | **0.083** | **0.724** |

## Full Model Comparison

| Feature Set | Model | AUC | AUC±std | F1 | P@R≥30% | P@R≥50% | ECE | Brier |
|-------------|-------|-----|---------|----|---------|---------|----|-------|
| all_signals | LogisticRegression | 0.690 | 0.696±0.049 | 0.699 | 0.756 | 0.713 | 0.130 | 0.233 |
| all_signals | RandomForest | 0.725 | 0.737±0.067 | 0.745 | 0.837 | 0.800 | 0.051 | 0.205 |
| all_signals | XGBoost | 0.672 | 0.670±0.077 | 0.713 | 0.740 | 0.723 | 0.239 | 0.278 |
| beh_debate | LogisticRegression | 0.706 | 0.705±0.041 | 0.715 | 0.768 | 0.767 | 0.115 | 0.220 |
| beh_debate | RandomForest | 0.721 | 0.729±0.070 | 0.765 | 0.788 | 0.756 | 0.067 | 0.205 |
| beh_debate | XGBoost | 0.686 | 0.691±0.068 | 0.743 | 0.753 | 0.753 | 0.235 | 0.270 |
| beh_v009 | LogisticRegression | 0.664 | 0.675±0.039 | 0.682 | 0.737 | 0.706 | 0.122 | 0.237 |
| beh_v009 | RandomForest | 0.731 | 0.738±0.056 | 0.752 | 0.841 | 0.778 | 0.036 | 0.202 |
| beh_v009 | XGBoost | 0.673 | 0.668±0.068 | 0.732 | 0.730 | 0.722 | 0.246 | 0.279 |
| beh_v009_debate | LogisticRegression | 0.685 | 0.686±0.034 | 0.695 | 0.767 | 0.712 | 0.130 | 0.231 |
| beh_v009_debate | RandomForest | 0.724 | 0.734±0.065 | 0.761 | 0.819 | 0.768 | 0.054 | 0.204 |
| beh_v009_debate | XGBoost | 0.679 | 0.679±0.065 | 0.743 | 0.738 | 0.737 | 0.213 | 0.267 |
| behavioral_only | LogisticRegression | 0.690 | 0.692±0.043 | 0.709 | 0.773 | 0.742 | 0.082 | 0.225 |
| behavioral_only | RandomForest | 0.730 | 0.739±0.052 | 0.754 | 0.826 | 0.780 | 0.029 | 0.204 |
| behavioral_only | XGBoost | 0.679 | 0.683±0.051 | 0.728 | 0.768 | 0.727 | 0.244 | 0.279 |
| debate_only | LogisticRegression | 0.682 | 0.689±0.038 | 0.706 | 0.750 | 0.724 | 0.076 | 0.219 |
| debate_only | RandomForest | 0.646 | 0.653±0.032 | 0.705 | 0.726 | 0.688 | 0.089 | 0.235 |
| debate_only | XGBoost | 0.598 | 0.603±0.020 | 0.668 | 0.675 | 0.675 | 0.257 | 0.311 |
| selected_4 | LogisticRegression | 0.635 | 0.649±0.045 | 0.560 | 0.710 | 0.710 | 0.102 | 0.235 |
| selected_4 | RandomForest | 0.756 | 0.765±0.066 | 0.724 | 0.966 | 0.859 | 0.083 | 0.196 |
| selected_4 | XGBoost | 0.722 | 0.727±0.085 | 0.717 | 0.897 | 0.802 | 0.243 | 0.267 |
| svg_only | LogisticRegression | 0.537 | 0.531±0.058 | 0.256 | 0.617 | 0.599 | 0.086 | 0.239 |
| svg_only | RandomForest | 0.552 | 0.547±0.066 | 0.307 | 0.674 | 0.610 | 0.108 | 0.248 |
| svg_only | XGBoost | 0.556 | 0.547±0.065 | 0.696 | 0.674 | 0.610 | 0.059 | 0.249 |
| v009_only | LogisticRegression | 0.660 | 0.669±0.082 | 0.631 | 0.789 | 0.718 | 0.111 | 0.225 |
| v009_only | RandomForest | 0.682 | 0.688±0.090 | 0.706 | 0.758 | 0.727 | 0.092 | 0.219 |
| v009_only | XGBoost | 0.632 | 0.639±0.099 | 0.654 | 0.774 | 0.698 | 0.312 | 0.328 |

## Feature Importance (Best Model)

| Rank | Feature | Importance | Signal Source |
|------|---------|------------|--------------|
| 1 | beh_total_cost_usd | 0.4201 | Behavioral |
| 2 | beh_tokens_per_edit | 0.3272 | Behavioral |
| 3 | beh_loop_count | 0.2007 | Behavioral |
| 4 | svg_accepted | 0.0519 | SVG |

## Forward Feature Selection

| Step | Added Feature | P@R≥30% | AUC | ECE |
|------|---------------|---------|-----|-----|
| 1 | beh_total_cost_usd | 0.841 | 0.708 | 0.122 |
| 2 | beh_tokens_per_edit | 0.871 | 0.722 | 0.084 |
| 3 | svg_accepted | 0.917 | 0.741 | 0.069 |
| 4 | beh_loop_count | 0.966 | 0.756 | 0.083 |

## Calibration Analysis

| Method | P@R≥30% | AUC | ECE | Brier |
|--------|---------|-----|-----|-------|
| uncalibrated | 0.966 | 0.756 | 0.083 | 0.196 |
| sigmoid | 0.932 | 0.754 | 0.064 | 0.197 |
| isotonic | 0.922 | 0.751 | 0.059 | 0.193 |

## Signal Contribution Analysis

Does adding each signal improve over behavioral-only?

- **beh_v009**: AUC=0.731 (Δ=+0.002 vs behavioral-only), P@R30=0.841, ECE=0.036
- **beh_debate**: AUC=0.721 (Δ=-0.009 vs behavioral-only), P@R30=0.788, ECE=0.067
- **beh_v009_debate**: AUC=0.724 (Δ=-0.005 vs behavioral-only), P@R30=0.819, ECE=0.054
- **all_signals**: AUC=0.725 (Δ=-0.005 vs behavioral-only), P@R30=0.837, ECE=0.051

## Success Criteria

1. **Combined beats behavioral-only**: AUC 0.756 vs 0.730 → PASS (Δ=+0.027)
2. **P@R≥30% > 0.85**: 0.966 → PASS
3. **ECE < 0.1 (RL-ready)**: 0.083 → PASS
4. **Post-calibration ECE < 0.1**: 0.059 → PASS
