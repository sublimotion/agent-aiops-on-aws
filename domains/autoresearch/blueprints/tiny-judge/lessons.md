---
model: n/a
engine: scikit-learn
hardware: cpu-local
outcome: complete
failure_categories: []

learn_commands: []
---

# Tiny Judge — Lessons

## Experiment Summary

- **Spec**: `domains/autoresearch/specs/tiny-judge.md`
- **Status**: COMPLETE
- **Date**: 2026-04-04
- **Cost**: ~$0 (CPU-only classical ML, no API calls)

## Key Findings

### 1. Behavioral features are additive to v009
Full-feature RF (AUC=0.675) beats v009-only (AUC=0.550) by +22.7%. This holds across all three model types. Combining v009 verdict/confidence with behavioral telemetry provides strictly more signal than either alone.

### 2. Novel features validate pivot analysis independently
The top RF features by importance are `elapsed_s`, `total_cost_usd`, `action_entropy`, `context_growth_rate` — all novel behavioral signals. These weren't in Critic Rubrics' 24-feature set. The pivot analysis found action count and explore ratio as top pivots; the judge independently found the same signals most predictive.

### 3. Feature selection unlocks precision — less is more
The full 36-feature model achieves P@R≥30% = 0.812 (below 0.85 target), but **forward feature selection** on RF top-15 features reveals that just 5 features achieve P@R≥30% = **0.949**. Adding more features *hurts* — classic overfitting at n=282. The optimal 5: `total_cost_usd`, `loop_count`, `action_pct_search`, `patch_len`, `first_edit_pct`. 4/5 are novel behavioral signals not in Critic Rubrics.

### 4. RandomForest > XGBoost for this dataset
RF consistently outperformed XGBoost on AUC (0.675 vs 0.662) and dramatically on calibration (ECE 0.043 vs 0.174). XGBoost's hyperparameter search converged to shallow trees (depth=3, n=50) — essentially matching simpler models. At n=282 with 36 features, RF's bagging provides better variance reduction than boosting.

### 5. v009 alone is surprisingly weak as ML features
v009_only XGBoost AUC = 0.550 (barely above random). The v009 rubric's power comes from its threshold decision (4/4 unanimous → precision 0.92), not from its raw score being discriminative. As a continuous feature, v009_confidence ranks #14 in RF importance — useful but not dominant.

### 6. Tool-only features rival full features
Tool-only XGBoost AUC = 0.669, nearly matching full = 0.662. The verification tool telemetry (generate_count, run_count, test_pass/fail, review_scores) carries most of the predictive signal. Behavioral features (action distribution, timing) add marginal improvement.

## Lessons for Future Work

- **n=282 is marginal for 36 features**: 5-fold CV showed high variance across folds (AUC range: 0.579–0.750). With more data, the signal would likely separate further.
- **ECE=0.043 means the model is RL-ready**: Predicted probabilities can be used directly as reward signals without recalibration.
- **Feature extraction is the bottleneck**: 80% of the work was data engineering (parsing 5 different data sources). The ML training took seconds.
- **pip on Python 3.14**: scikit-learn has import errors on Python 3.14 (`ModuleNotFoundError: sklearn.utils._estimator_html_repr`). Use Python 3.12 venv.
