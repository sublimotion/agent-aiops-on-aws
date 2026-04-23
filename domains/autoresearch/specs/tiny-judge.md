# Autoresearch Spec: Tiny Judge (Feature-Based Verifier)

## Status: COMPLETE

## Overview

Train a classical ML verifier (XGBoost/logistic regression) on behavioral features + v009 signals to predict gold test outcomes. This attacks the recall ceiling — v009 adversarial rubric has 0.92 precision but only 7.7% coverage (37/483 patches selected). By combining v009 verdicts with behavioral telemetry (Parkinson's ratio, tool composition pattern, context growth rate), we aim to improve recall while maintaining precision > 0.85.

**Core hypothesis**: An XGBoost model on [v009_verdict, v009_confidence, diff_size, first_edit_turn, Parkinson_ratio, loop_count, tool_composition_pattern, action_pct_search, context_growth_rate] predicts gold outcomes better than v009 alone, achieving recall > 0.30 while maintaining precision > 0.85.

**Motivation**: Phase 0 showed behavioral features are underpowered at n=23 (AUC 0.542, below majority-class baseline). But the verification primitives experiment provides n=300 with richer features (tool use patterns, composition type, checkpoint response) and gold labels from Docker eval. Critic Rubrics (arXiv:2603.03800) achieved +15.9 Best@8 with 24 behavioral features and no model training. Our novel signals (action_distribution, context_growth_rate) are not in their feature set.

**Depends on**: verification-primitives (telemetry data, n=300), verification-primitives-swebench (gold labels from Docker eval), verifier-reward (v009 labels on n=483)

## Components

### 1. Compute
- **Platform**: Local laptop or EC2 — scikit-learn / XGBoost
- **Instance Type**: Any CPU
- **GPUs**: None required for classical ML. If extending to small LLM fine-tuning, g7e instance.

### 2. Codebase
- **Source**: New scripts in blueprint directory
- **Fixed files**:
  - Telemetry: `blueprints/verification-primitives/results/telemetry/` (251 JSONL files)
  - Gold labels: `blueprints/verification-primitives-swebench/results/eval_report.json` + `eval_report_errors_v2.json`
  - v009 labels: `blueprints/verifier-reward/results/` (n=483 SWE-bench Verified evaluations)
  - Predictions: `blueprints/verification-primitives-swebench/results/predictions_lite.jsonl`
- **Agent-editable files**:
  - `blueprints/tiny-judge/scripts/extract_features.py` — parse telemetry into feature matrix
  - `blueprints/tiny-judge/scripts/train_judge.py` — XGBoost / logistic regression training + evaluation
  - `blueprints/tiny-judge/scripts/calibration.py` — ECE measurement on trained model
- **Agent instructions**: N/A

### 3. Experiment Protocol
- **Metric**: AUC-ROC (higher is better), precision at recall > 0.30 (higher is better)
- **Secondary**: ECE (lower is better), F1, feature importance ranking
- **Time budget**: 4-6 hours (feature extraction + model training + analysis)
- **Loop structure**: Feature extraction → baseline models → feature ablation → calibration analysis
- **Termination**: Best model identified with cross-validated metrics
- **Logging**: `blueprints/tiny-judge/results/judge_report.md` + `blueprints/tiny-judge/results/models/`

### 4. Networking
- **Access**: Local or SSH to EC2 for data access
- **API calls**: ~$5-10 for v009 labels on primitives experiment data (if not already computed)

### 5. Storage
- **Data**: Existing telemetry + eval results from prior experiments
- **Models**: Serialized XGBoost/sklearn models in results directory
- **Results**: `blueprints/tiny-judge/results/`

## Feature Engineering

### Feature Vector (per issue)

| Feature | Source | Type | Novel? |
|---------|--------|------|--------|
| `v009_verdict` | verifier-reward v009 rubric | binary | No |
| `v009_confidence` | v009 rubric score (0-1) | continuous | No |
| `diff_size_chars` | patch diff character count | continuous | No |
| `diff_size_lines` | patch diff line count | continuous | No |
| `files_modified` | count of modified files | integer | No |
| `first_edit_turn` | turn number of first file edit | integer | No |
| `parkinson_ratio` | first_edit_turn / total_turns | continuous | Partial |
| `total_turns` | total turns consumed | integer | No |
| `loop_count` | repeated action sequences | integer | No |
| `action_pct_search` | fraction of turns spent searching | continuous | **Yes** |
| `action_pct_read` | fraction of turns spent reading | continuous | **Yes** |
| `action_pct_edit` | fraction of turns spent editing | continuous | **Yes** |
| `context_growth_rate` | context tokens per turn growth | continuous | **Yes** |
| `tool_composition` | one-hot: ignore / generate_run / gen_run_iterate / full_pipeline / full_pipeline_iterate | categorical | **Yes** |
| `tool_count` | total verification tool invocations | integer | No |
| `first_tool_turn` | turn number of first tool use | integer | No |
| `first_tool_ratio` | first_tool_turn / total_turns | continuous | No |
| `adversarial_review_used` | whether adversarial_review was invoked | binary | No |
| `test_pass_count` | number of generated tests that passed | integer | No |
| `test_fail_count` | number of generated tests that failed | integer | No |

### Baselines to Beat

| Method | Precision | Recall | AUC | Coverage |
|--------|-----------|--------|-----|----------|
| v009 4/4 unanimous | 0.92 | 0.14 | — | 7.7% |
| SVG consensus (threshold=0.8) | 1.00 | 0.528 | 0.981 | 52.8% |
| Majority class (always predict fail) | 0.417 | 0.0 | 0.5 | 0% |
| Random | 0.583 | 0.583 | 0.5 | 100% |

## Experiment Phases

### Phase 1: Feature Extraction (~1 hour)

1. Parse 300 Claude Code trajectory logs into per-issue feature vectors
2. Parse 251 tool telemetry JSONL files for tool use patterns
3. Join with gold labels from Docker eval (175 resolved / 125 unresolved)
4. Validate feature distributions — check for missing data, outliers, class balance

### Phase 2: Classical ML Training (~2 hours)

1. **XGBoost** — 5-fold stratified cross-validation
   - Hyperparameter search: max_depth [3,5,7], n_estimators [50,100,200], learning_rate [0.01,0.1,0.3]
   - Class weights to handle imbalance (175:125 resolved:unresolved)
2. **Logistic Regression** — L1/L2 regularization sweep
3. **Random Forest** — as ensemble baseline
4. Report: AUC, precision@recall>0.30, F1, feature importance

### Phase 3: Feature Ablation (~1 hour)

1. **v009-only model**: just v009_verdict + v009_confidence
2. **Behavioral-only model**: all features except v009
3. **Novel features only**: action_pct_*, context_growth_rate, tool_composition
4. **Full model**: all features
5. Measure: does combining v009 + behavioral beat either alone? Are novel features additive?

### Phase 4: Calibration Analysis (~30 min)

1. Compute ECE on best model's predicted probabilities (10-bin, both equal-width and quantile)
2. Generate reliability diagram
3. Compare: model ECE vs raw v009 ECE vs SVG ECE (from svg-ece-measurement spec)
4. If ECE > 0.1: apply Platt scaling, re-measure

### Phase 5 (Optional): Small LLM Judge

If classical ML shows strong signal, test whether a small LLM can replicate:
1. Format features + patch as a prompt → Haiku → binary verdict
2. Compare: prompted Haiku vs trained XGBoost on same features
3. This tests whether the signal is in the features (XGBoost sufficient) or requires reasoning (LLM needed)

Cost: ~$5 for 300 Haiku calls

## Success Criteria

1. **Minimum viable**: Any model beats v009-only baseline (AUC > v009 AUC) with recall > 0.20 at precision > 0.85
2. **Strong result**: Recall > 0.30 at precision > 0.85 (4x v009's coverage). Feature ablation shows behavioral features are additive to v009.
3. **Publishable result**: Novel features (action_distribution, context_growth_rate, tool_composition) are top-5 in feature importance AND not present in Critic Rubrics' 24 features. ECE < 0.1 (RL-ready).
4. **Negative result (still valuable)**: At n=300, behavioral features don't add to v009. Analysis of why — sample size? feature noise? v009 already captures the signal? Documents the boundary condition for feature-based verification.

## Non-Requirements
- LLM fine-tuning (Phase 5 is optional prompting, not fine-tuning)
- RL training or reward model training
- Real-time inference — this is offline batch analysis
- Deployment of the trained model
- Cross-benchmark generalization (SWE-bench Lite only for now)

## Known Limitations
- **n=300 is marginal for 20+ features**: Risk of overfitting. Mitigation: 5-fold CV, L1 regularization, monitor train/val gap.
- **Class imbalance**: 175:125 (58:42) is mild but present. Use stratified splits and class weights.
- **Feature extraction depends on log format**: Claude Code logs may not have perfectly structured per-turn data. May need heuristic parsing.
- **Confounded features**: tool_composition correlates with gold outcome by design (the experiment showed tools help). The model may learn "tools used → pass" rather than anything deeper. Feature importance analysis must account for this.
- **v009 labels on primitives data**: If v009 wasn't run on the 300 primitives instances, we need ~$3 to generate those labels (300 × $0.01/call).

## Relationship to Other Specs

- **svg-ece-measurement**: Provides SVG ECE baseline for calibration comparison
- **verification-primitives**: Source of behavioral telemetry and tool composition data
- **verification-primitives-swebench**: Source of gold labels
- **verifier-reward**: Source of v009 rubric methodology and SWE-bench Verified labels
- Future: If successful, trained model feeds into E6 (rejection sampling SFT) and E7 (PivotRL as reward signal)

## Key References

- Critic Rubrics (arXiv:2603.03800) — 24 behavioral features, +15.9 Best@8, no model training
- SWE-RM (arXiv:2512.21919) — execution-free reward model, calibration > ranking
- "Smaller Models, Smarter Rewards" (arXiv:2510.23083) — Phi-4 (14B) as code reward model
- Haize Labs J1-micro — 1.7B judge via RL-trained instance rubrics (RewardBench 80.7%)
- FLIP (arXiv:2602.13551) — backward inference for small reward models

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
