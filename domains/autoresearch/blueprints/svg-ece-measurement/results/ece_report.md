# SVG ECE Measurement Report

**Date**: 2026-04-04

## Data Summary

- Joined instances: **282**
- Gold resolved (label=1): **175** (62.1%)
- SVG score > 0: **50** (17.7%)
- SVG score = 0: **232**
- SVG-only (no gold label): 18
- Gold-only (no SVG score): 0
- Base rate (gold pass): **0.621**

## SVG Score Distribution

| Score Range | Count | Gold Pass | Gold Pass Rate |
|-------------|------:|----------:|---------------:|
| = 0.0 | 232 | 136 | 0.586 |
| (0, 0.5] | 8 | 6 | 0.750 |
| (0.5, 1.0) | 26 | 18 | 0.692 |
| = 1.0 | 16 | 15 | 0.938 |

## Calibration Metrics (Equal-Width, 10 bins)

| Metric | Value |
|--------|------:|
| **ECE** | **0.5115** |
| ECE 95% CI (bootstrap) | [0.4602, 0.5802] |
| MCE | 0.8571 |
| Brier Score | 0.5120 |

### Equal-Width Bins

| Bin | Count | Confidence | Accuracy | Gap |
|-----|------:|-----------:|---------:|----:|
| [0.00, 0.10] | 232 | 0.000 | 0.586 | 0.586 |
| [0.10, 0.20] | 1 | 0.143 | 1.000 | 0.857 |
| [0.20, 0.30) | 0 | — | — | — |
| [0.30, 0.40] | 4 | 0.353 | 0.750 | 0.397 |
| [0.40, 0.50] | 2 | 0.473 | 0.500 | 0.027 |
| [0.50, 0.60] | 3 | 0.530 | 0.667 | 0.137 |
| [0.60, 0.70] | 6 | 0.636 | 0.333 | 0.303 |
| [0.70, 0.80] | 7 | 0.751 | 0.571 | 0.179 |
| [0.80, 0.90] | 10 | 0.867 | 1.000 | 0.133 |
| [0.90, 1.00] | 17 | 0.996 | 0.941 | 0.055 |

### Reliability Diagram (ASCII)

```
Accuracy
1.0 |                                        
1.0 |    *                    *  . 
0.9 |                         .  * 
0.8 |                      .       
0.7 |                *             
0.6 | *                 .  *       
0.5 |             *  .             
0.4 |          .                   
0.3 |                   *          
0.2 |                              
0.1 |    .                         
0.0 | .                            
    +------------------------------
     0.0 0.1 0.4 0.5 0.5 0.6 0.8 0.9 1.0 
              Confidence
  * = accuracy, . = perfect calibration line
```

## Calibration Metrics (Quantile Bins)

| Metric | Value |
|--------|------:|
| **ECE** | **0.4843** |
| ECE 95% CI (bootstrap) | [0.4326, 0.5583] |
| MCE | 0.5391 |

### Quantile Bins

| Bin | Count | Confidence | Accuracy | Gap |
|-----|------:|-----------:|---------:|----:|
| [0.00, 0.78] | 253 | 0.046 | 0.585 | 0.539 |
| [0.78, 1.00] | 29 | 0.937 | 0.931 | 0.006 |

## RL-Readiness Assessment

- ECE = **0.5115**
- Assessment: **POOR — Recalibrate via Platt scaling before any use**

## Recalibration Results

| Method | ECE | ECE 95% CI | MCE | Brier | Parameters |
|--------|----:|-----------|----:|------:|-----------|
| Raw | 0.5115 | [0.4602, 0.5802] | 0.8571 | 0.5120 | — |
| Platt scaling | 0.0312 | [0.0240, 0.0984] | 0.2331 | 0.2285 | a=1.232, b=0.347 |
| Temperature scaling | 0.2664 | [0.2093, 0.3268] | 0.2783 | 0.2994 | T=19.9 |

### Platt Scaling Bins

| Bin | Count | Confidence | Accuracy | Gap |
|-----|------:|-----------:|---------:|----:|
| [0.00, 0.10) | 0 | — | — | — |
| [0.10, 0.20) | 0 | — | — | — |
| [0.20, 0.30) | 0 | — | — | — |
| [0.30, 0.40) | 0 | — | — | — |
| [0.40, 0.50) | 0 | — | — | — |
| [0.50, 0.60] | 232 | 0.586 | 0.586 | 0.000 |
| [0.60, 0.70] | 5 | 0.674 | 0.800 | 0.126 |
| [0.70, 0.80] | 19 | 0.759 | 0.526 | 0.233 |
| [0.80, 0.90] | 26 | 0.820 | 0.962 | 0.141 |
| [0.90, 1.00) | 0 | — | — | — |

### Temperature Scaling Bins

| Bin | Count | Confidence | Accuracy | Gap |
|-----|------:|-----------:|---------:|----:|
| [0.00, 0.10) | 0 | — | — | — |
| [0.10, 0.20) | 0 | — | — | — |
| [0.20, 0.30) | 0 | — | — | — |
| [0.30, 0.40] | 232 | 0.308 | 0.586 | 0.278 |
| [0.40, 0.50] | 7 | 0.492 | 0.714 | 0.222 |
| [0.50, 0.60] | 27 | 0.515 | 0.704 | 0.188 |
| [0.60, 0.70] | 16 | 0.692 | 0.938 | 0.245 |
| [0.70, 0.80) | 0 | — | — | — |
| [0.80, 0.90) | 0 | — | — | — |
| [0.90, 1.00) | 0 | — | — | — |

## Cross-Tabulation: SVG Accepted vs Gold Resolved

| | Gold Resolved | Gold Unresolved | Total |
|---|---:|---:|---:|
| SVG Accepted | 26 | 1 | 27 |
| SVG Rejected | 149 | 106 | 255 |
| Total | 175 | 107 | 282 |

- SVG Precision: 0.963
- SVG Recall: 0.149

## Interpretation

The SVG line_recall score distribution is extremely sparse: 232/282 instances have score=0.0. This creates a degenerate calibration scenario where most of the ECE weight falls on the [0.0, 0.1) bin. The score is not a continuous confidence estimate in the traditional sense — it is a code overlap metric (line recall against gold patch).

**Key caveat**: The SVG scores are from the SERA harness (Phase 0, verification-primitives), while gold labels are from the Claude Code + primitives production run (verification-primitives-swebench). These are different patches for the same instances. The SVG line_recall measures how close the SERA patch is to the gold patch, NOT the Claude Code patch.

## SWE-RM Comparison Context

SWE-RM (arXiv:2512.21919) found that two verifiers with similar ranking performance had 7x ECE difference (0.078 vs 0.541). The poorly calibrated one caused RL training collapse. Our SVG ECE result should be compared against these benchmarks:

| Verifier | ECE | RL Outcome |
|----------|----:|-----------|
| SWE-RM-LLM (well-calibrated) | 0.078 | Stable RL training |
| SWE-RM-Verifier (poorly calibrated) | 0.541 | RL collapse |
| **SVG line_recall (ours, raw)** | **0.512** | — |
| **SVG line_recall (ours, Platt)** | **0.031** | — |
| **SVG line_recall (ours, Temp)** | **0.266** | — |
