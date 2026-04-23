# Pivot Point Analysis Report

**Dataset**: VP SWE-bench production eval (n=300)
**Evaluated**: 282 issues with gold labels
**Pass rate**: 175/282 = 62.1%

## Pivot Ranking by Outcome Variance

| Rank | Pivot | MI (bits) | Risk Diff | 95% CI | p-value | n |
|------|-------|-----------|-----------|--------|---------|---|
| 1 | Tool Usage (used vs not) | 0.0668 | +46.3% | [+31.1%, +61.4%] | 4.86e-07*** | 282 |
| 2 | Many vs Few Actions | 0.0439 | -23.8% | [-34.9%, -12.8%] | 4.71e-05*** | 282 |
| 3 | High vs Low Explore Ratio | 0.0219 | -16.9% | [-28.0%, -5.7%] | 4.67e-03** | 282 |
| 4 | Large vs Small Patch | 0.0087 | -10.6% | [-21.9%, +0.6%] | 8.56e-02 | 282 |
| 5 | Revised After Failure vs Submitted Despite | 0.0021 | +5.6% | [-8.3%, +19.6%] | 5.00e-01 | 207 |
| 6 | Adversarial Review vs Tests Only | 0.0007 | +4.2% | [-12.7%, +21.1%] | 7.01e-01 | 249 |
| 7 | Early vs Late First Edit | 0.0003 | +1.8% | [-9.5%, +13.2%] | 8.06e-01 | 282 |

### Tool Usage (used vs not)

- **n_used**: 249
- **n_unused**: 33
- **rate_used**: 67.5%
- **rate_unused**: 21.2%
- **risk_difference**: 46.3%
- **odds_ratio**: 7.7037
- **p_value**: 0.0000
- **mutual_information**: 0.0668

### Many vs Few Actions

- **n_many**: 135
- **n_few**: 147
- **rate_many**: 49.6%
- **rate_few**: 73.5%
- **risk_difference**: -23.8%
- **odds_ratio**: 0.3558
- **p_value**: 0.0000
- **mutual_information**: 0.0439
- **median_action_count**: 23.0000

### High vs Low Explore Ratio

- **n_high_explore**: 140
- **n_low_explore**: 142
- **rate_high_explore**: 53.6%
- **rate_low_explore**: 70.4%
- **risk_difference**: -16.9%
- **odds_ratio**: 0.4846
- **p_value**: 0.0047
- **mutual_information**: 0.0219
- **median_explore_ratio**: 0.2381

### Large vs Small Patch

- **n_large**: 141
- **n_small**: 141
- **rate_large**: 56.7%
- **rate_small**: 67.4%
- **risk_difference**: -10.6%
- **odds_ratio**: 0.6350
- **p_value**: 0.0856
- **mutual_information**: 0.0087
- **median_patch_len**: 767.0000

### Revised After Failure vs Submitted Despite

- **n_revised**: 55
- **n_submitted**: 152
- **rate_revised**: 72.7%
- **rate_submitted**: 67.1%
- **risk_difference**: 5.6%
- **odds_ratio**: 1.3072
- **p_value**: 0.5001
- **mutual_information**: 0.0021

### Adversarial Review vs Tests Only

- **n_reviewed**: 213
- **n_tests_only**: 36
- **rate_reviewed**: 68.1%
- **rate_tests_only**: 63.9%
- **risk_difference**: 4.2%
- **odds_ratio**: 1.2052
- **p_value**: 0.7009
- **mutual_information**: 0.0007

### Early vs Late First Edit

- **n_early**: 151
- **n_late**: 131
- **rate_early**: 62.9%
- **rate_late**: 61.1%
- **risk_difference**: 1.8%
- **odds_ratio**: 1.0815
- **p_value**: 0.8059
- **mutual_information**: 0.0003
- **median_pct**: 0.3333
- **timing**: median=33.3%, IQR=[23.6%, 42.9%]

## Composition Pattern Analysis

| Pattern | n | Pass | Fail | Pass Rate |
|---------|---|------|------|-----------|
| other | 3 | 3 | 0 | 100.0% |
| generate_only | 4 | 3 | 1 | 75.0% |
| generate_run | 27 | 19 | 8 | 70.4% |
| full_pipeline | 208 | 140 | 68 | 67.3% |
| gen_run_iterate | 7 | 3 | 4 | 42.9% |
| ignore | 33 | 7 | 26 | 21.2% |

### Pairwise Significance Tests

| Pattern A | Pattern B | Rate A | Rate B | OR | p-value |
|-----------|-----------|--------|--------|-----|---------|
| full_pipeline | ignore | 67.3% | 21.2% | 7.65 | 1.14e-06*** |
| generate_run | ignore | 70.4% | 21.2% | 8.82 | 2.04e-04*** |
| gen_run_iterate | generate_run | 42.9% | 70.4% | 0.32 | 2.11e-01 |
| full_pipeline | gen_run_iterate | 67.3% | 42.9% | 2.75 | 2.27e-01 |
| gen_run_iterate | ignore | 42.9% | 21.2% | 2.79 | 3.38e-01 |
| full_pipeline | generate_run | 67.3% | 70.4% | 0.87 | 8.30e-01 |

## Early-Stopping Rule Analysis

| Rule | Abort | Continue | Abort Pass% | Continue Pass% | False Aborts | Precision | Recall |
|------|-------|----------|-------------|----------------|--------------|-----------|--------|
| No edit by 30% of actions | 165 | 117 | 61.8% | 62.4% | 102 | 38.2% | 58.9% |
| No edit by 40% of actions | 85 | 197 | 58.8% | 63.5% | 50 | 41.2% | 32.7% |
| No edit by 50% of actions | 35 | 247 | 42.9% | 64.8% | 15 | 57.1% | 18.7% |
| No edit by 60% of actions | 20 | 262 | 35.0% | 64.1% | 7 | 65.0% | 12.1% |
| No VP verification tool used (end of session) | 33 | 249 | 21.2% | 67.5% | 7 | 78.8% | 24.3% |
| No VP tools AND late edit (>40% of actions) | 15 | 267 | 13.3% | 64.8% | 2 | 86.7% | 12.1% |

## Timing Analysis

### First Edit (Explore→Implement transition)
- n=285, median=33.3%, IQR=[23.8%, 42.9%]
- Pass group: median=32.4% (n=175)
- Fail group: median=33.3% (n=107)
- **Edit checkpoint (40%) vs empirical median (33.3%)**: aligned (within 10pp)

### Action Budget Usage
- Pass: median=21 actions, mean=22
- Fail: median=30 actions, mean=26

## Key Findings

1. **Highest-variance pivot**: Tool Usage (used vs not) (MI=0.0668 bits, risk diff=+46.3%, p=4.86e-07)
2. **Second pivot**: Many vs Few Actions (MI=0.0439 bits, p=4.71e-05)
3. **Third pivot**: High vs Low Explore Ratio (MI=0.0219 bits, p=4.67e-03)

**Checkpoint validation**: The 40% edit nudge is well-placed — empirical first-edit median is 33.3% of action budget.

**Best early-stopping rule**: "No edit by 30% of actions" — precision=38.2%, recall=58.9%, would abort 165 issues with 102 false aborts.
