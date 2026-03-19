# Learned Verifier: Closing the Fix-to-Pass Gap

**Status**: DRAFT — pending data collection
**Created**: 2026-03-19
**Depends on**: agent-harness (Phase 1/2 complete), agent-swarm (Phase 1 complete)

## Executive Summary

Train a model to predict whether a coding agent's patch will pass tests, then use it to select the best patch from N candidates. This is the coding-agent instantiation of Best-of-N with a learned reward model.

**The core bet**: the binding constraint on coding agent performance is verification quality, not model scale, harness design, or finetuning.

## Reality Check: What We Actually Have

A thorough audit (2026-03-19) of all existing experiment data revealed a significant gap between what we assumed and what exists.

### Data Inventory (Audited)

| Source | Files | Rows | patch_diff | transcript | tests_pass | turn_metrics |
|--------|------:|-----:|:----------:|:----------:|:----------:|:------------:|
| Harness Phase 1 (A-F) | 6 | 300 | NONE | NONE | NONE | YES (6,599 turn rows) |
| Harness Phase 2 (sera, langgraph, aider) | 3 | 150 | NONE | NONE | NONE | partial |
| Harness Phase 2b (6 harnesses) | 6 | 270 | NONE | NONE | always false* | partial |
| Harness Eval (7 harnesses) | 7 | 248 | NONE | NONE | YES (60 pass) | NONE |
| Swarm Phase 1 (9 configs) | 9 | 491 | NONE | NONE | partial (12 pass) | NONE |
| Recovered diffs | 1 | 1 | 1 file | NONE | N/A | N/A |
| **TOTAL** | **32** | **~1,459** | **1** | **0** | **~72 pass** | **6,599 turns** |

*Phase 2b `pass` field is a pre-eval placeholder (always false). Actual test results are only in eval_* files.

### What This Means

1. **Zero patch diffs exist.** The instrumentation was deployed to g7e but has never been run. We cannot train any patch-level model without collecting new data.
2. **Zero transcripts exist.** No conversation logs from any run. Trajectory-level verifier is not possible with current data.
3. **50 unique issues is the real sample size**, not 1,459. Each issue appears across multiple configs, but they are repeated measurements of the same 50 underlying problems.
4. **Only 72 positives (tests_pass=true)** across all eval files. Class imbalance is ~1:8.
5. **Devstral swarm files lack tests_pass entirely** — different schema from qwen/swesmith runs.
6. **OpenCode + Claude Code adapters cannot capture transcripts** — they are bash scripts that return JSON on stdout. Only the SERA path (which calls `run_instrumented_loop`) can save conversation history.

### Instrumentation Status

| Component | Code Ready | Deployed to g7e | Data Produced |
|-----------|:----------:|:---------------:|:-------------:|
| `harness_eval.py` — patch_diff capture | YES | YES | NO |
| `harness_eval.py` — transcript capture | YES | YES | NO |
| `swarm_eval.py` — patch_diff capture | YES | YES | NO |
| `swarm_eval.py` — transcript (SERA only) | YES | YES | NO |
| OpenCode adapter — transcript capture | NO (not possible) | N/A | N/A |
| Claude Code adapter — transcript capture | NO (not possible) | N/A | N/A |

### Known Code Risks

1. **`_get_git_diff` only captures unstaged changes.** If an agent runs `git add` or `git commit`, the diff is empty — producing a false negative for `fix_generated`.
2. **`_sanitize_transcript` truncates tool outputs at 8K chars.** `read_file` can return 50K chars. A verifier training on truncated transcripts may miss the code the agent actually examined.
3. **50-issue subset is not pinned to a dataset revision.** `load_subset()` uses seed=42 with round-robin repo stratification, but depends on HuggingFace dataset row ordering. If Princeton updates SWE-bench_Lite, the subset changes silently.
4. **Restart strategy wipes diffs.** If `restart_at` triggers a `git checkout -f HEAD`, changes from the first half of the run are destroyed. The final `_get_git_diff` only captures second-half changes.

---

## Experiment Design

### Statistical Constraints

These constraints are non-negotiable and shape every design decision:

| Constraint | Value | Implication |
|-----------|-------|-------------|
| Unique issues (N) | 50 | This is the effective sample size for generalization |
| Max positive examples | ~72 (current) + new runs | Class imbalance ~1:8 |
| Evaluation method | Leave-one-issue-out CV (LOIO-CV) | Train on 49 issues, test on 1, repeat 50x |
| Model complexity ceiling | Low (XGBoost, logistic regression, small MLP) | LLM fine-tuning is not viable at N=50 |
| Confidence intervals | Wide (~+/-15pp for proportions over 50 trials) | This is a pilot study, not a definitive experiment |

**Why not fine-tune an LLM?** With 50 unique issues, each appearing ~12x with different patches, an LLM will memorize `problem_statement → pass_probability` rather than learning general patch quality assessment. The model sees the same problem statement ~12 times during training — it learns issue difficulty, not patch quality. Fine-tuning requires 10K-100K+ unique inputs for the (problem, patch) → quality mapping.

**The right framing**: this experiment answers "Is there learnable signal in patch features that predicts test outcomes, separable from issue difficulty and harness artifacts?" If yes, collect more data (SWE-bench full: 2,294 issues) and scale up. If no, LLM fine-tuning won't help.

### Confounders to Control

The label `tests_pass` encodes: `f(issue_difficulty, model_capability, harness_quality, patch_quality)`. The verifier only sees (problem_statement, patch_diff). Three confounders must be addressed:

1. **Issue difficulty** — easy issues pass more often regardless of patch quality. Control: pairwise training within each issue (compare passing vs failing patches for the same issue).
2. **Harness bias** — a harness bug causing spurious failures looks like "bad patch" to the verifier. Control: train within-harness, test cross-harness.
3. **Model identity** — model-specific code style correlates with pass rate. Control: include model_name as feature, measure feature importance, verify patch features dominate.

### Scope and Non-Goals

**In scope**: Proof-of-concept on SWE-bench Lite 50 showing that patch features predict test outcomes better than random selection.

**Not in scope** (requires more data):
- Generalizing to unseen repos or languages
- Replacing test suites in production CI/CD
- Fine-tuning an LLM verifier
- Process reward modeling (per-turn verification)

---

## Phase 1: Data Collection (PREREQUISITE)

**Goal**: Collect patch diffs and transcripts for all SERA-path runs. Collect patch diffs only for adapter-path runs (OpenCode, Claude Code).

### 1.1 Fix Known Code Issues Before Running

| Issue | Fix | Priority |
|-------|-----|----------|
| `_get_git_diff` misses staged/committed changes | Change to `git diff HEAD` (captures both staged and unstaged vs HEAD) | CRITICAL |
| 50-issue subset not pinned | Add `datasets_version` or dump the 50 instance_ids to a manifest file | HIGH |
| `_sanitize_transcript` 8K truncation | Increase to 16K or make configurable; add a `truncated: true` flag per message | MEDIUM |
| Restart strategy wipes first-half diffs | Capture pre-restart diff and store as `partial_diff_before_restart` | LOW |

### 1.2 Instrumented SERA Runs (Full Signal)

These produce: patch_diff + full transcript + turn_metrics + tests_pass.

| Run | Model | Issues | Expected Trajectories | Priority | Est. GPU-hours |
|-----|-------|--------|----------------------|----------|---------------|
| SERA-1 | Devstral 24B | 50 | 50 (new — no tests_pass in existing data) | HIGH | ~4h on 1x B200 |
| SERA-2 | Qwen3.5-397B | 50 | 50 (re-run with instrumentation) | HIGH | ~8h on 4x B200 |
| SERA-3 | Qwen 2.5 Coder 32B | 50 | 50 (re-run with instrumentation) | MEDIUM | ~4h on 1x B200 |
| SERA-4 | SWE-agent-LM 32B | 50 | 50 (re-run with instrumentation) | LOW | ~4h on 1x B200 |

**Total: 200 trajectories with full signal. ~20 GPU-hours.**

### 1.3 Adapter Runs — Patch Diff Only

OpenCode and Claude Code adapters cannot return transcripts (bash subprocess, JSON on stdout). We can only capture `git diff HEAD` after the adapter completes.

| Run | Model | Harness | Issues | Signal |
|-----|-------|---------|--------|--------|
| ADAPT-1 | Devstral 24B | OpenCode | 50 | patch_diff + tests_pass only |
| ADAPT-2 | Devstral 24B | Claude Code | 50 | patch_diff + tests_pass only |
| ADAPT-3 | Qwen3.5-397B | OpenCode | 50 | patch_diff + tests_pass only |

**Total: 150 additional patch diffs (no transcripts). ~12 GPU-hours.**

### 1.4 Generate N=16 Candidate Patches (for Best-of-N Evaluation)

For the verifier to be useful, we need multiple candidate patches per issue. Run the cheapest high-fix-rate config 16 times per issue with different random seeds.

| Generator | Config | Issues | Candidates/Issue | Total Patches |
|-----------|--------|--------|-----------------|---------------|
| Devstral 24B × SERA | Phase 1 Config D (30 turns) | 50 | 16 | 800 |

**Each patch needs**: patch_diff (captured), tests_pass (run gold tests), problem_statement (from SWE-bench).

**This is the most expensive step: ~64 GPU-hours** (800 runs × ~5 min each on 1x B200). Can be parallelized across 2 GPUs.

**Alternative (cheaper)**: Generate N=4 per issue (200 runs, ~16 GPU-hours). Lower statistical power for best-of-N but sufficient to establish whether ranking signal exists.

### 1.5 Data Manifest

After collection, produce a single manifest file:

```json
{
  "dataset_version": "princeton-nlp/SWE-bench_Lite@<commit_hash>",
  "subset_seed": 42,
  "subset_size": 50,
  "instance_ids": ["astropy__astropy-7746", "..."],
  "collection_date": "2026-03-XX",
  "runs": [
    {
      "run_id": "sera_devstral_001",
      "model": "devstral-small-2",
      "harness": "sera",
      "instance_id": "astropy__astropy-7746",
      "has_patch_diff": true,
      "has_transcript": true,
      "has_turn_metrics": true,
      "tests_pass": false,
      "fix_generated": true
    }
  ]
}
```

### 1.6 Exit Criteria for Phase 1

Phase 1 is complete when:
- [ ] At least 200 trajectories with patch_diff (from SERA runs)
- [ ] At least 100 trajectories with full transcript (SERA only)
- [ ] N=16 (or N=4) candidate patches per issue for at least 40 issues, each with tests_pass label
- [ ] Manifest file validates: no missing fields, no null patch_diffs where fix_generated=true
- [ ] `_get_git_diff` confirmed to use `git diff HEAD` (not just `git diff`)
- [ ] 50-issue subset pinned in manifest with dataset version

---

## Phase 2: Baselines (No Training Required)

**Goal**: Establish performance baselines before training anything. If baselines are strong, training may not be needed.

### 2.1 Random Baseline

For each issue with N candidates:
- `P(at least 1 pass in N)` = `1 - (1 - p_i)^N` where `p_i` is issue-level pass rate
- Random top-1 selection: expected pass rate = per-issue average pass rate
- Report: mean and 95% CI across 50 issues

### 2.2 Simple Heuristics

| Heuristic | Input | Rationale |
|-----------|-------|-----------|
| Shortest patch (fewest diff lines) | patch_diff | Minimal changes less likely to introduce bugs. Strong baseline in program repair literature. |
| Smallest diff bytes | patch_diff | Variant of above |
| Fewest files touched | patch_diff | Single-file fixes more likely correct |
| Fewest turns used | turn_metrics | Agent found fix quickly = higher confidence |
| Lowest token count | turn_metrics | Less exploration = more decisive |
| Earliest first_edit_turn | turn_metrics | Didn't waste turns exploring — knew what to do |
| No test file modifications | patch_diff | Patches that modify test files are suspicious |

Evaluate each heuristic: given N candidates, rank by heuristic, measure top-1 and top-3 pass rate via LOIO-CV.

### 2.3 LLM-as-Judge (Zero-Shot)

**This is the most important baseline.** If a frontier LLM can predict pass/fail without training, fine-tuning adds no value.

```
Prompt: Given this bug report and proposed patch, predict whether
the patch will pass the project's test suite.

Bug report: {problem_statement}
Patch: {patch_diff}

Answer YES or NO, then explain your reasoning.
```

Test with:
- Claude Sonnet 4.6 (fast, cheap)
- Claude Opus 4.6 (strongest reasoning)

Evaluate: rank N candidates by P(YES), measure top-1 pass rate. Compare to heuristics and random.

**Cost estimate**: 50 issues × 16 candidates × ~$0.01/call = ~$8 for Sonnet, ~$40 for Opus.

### 2.4 Exit Criteria for Phase 2

- [ ] All baselines computed with LOIO-CV
- [ ] Results table with 95% confidence intervals
- [ ] Decision: does any baseline already meet or exceed the 7-harness ensemble ceiling (32%)?
- [ ] If LLM-as-judge > 40% top-1, reconsider whether training is needed at all

---

## Phase 3: Trained Models

**Only proceed if Phase 2 shows headroom** — i.e., oracle ceiling (best of N) substantially exceeds best baseline, indicating learnable signal that baselines miss.

### 3.1 Feature Engineering

Extract from each (issue, patch) pair:

**Patch features** (from patch_diff):
- Lines added / removed / changed
- Files touched (count)
- Whether test files were modified
- Whether imports were added/removed
- Diff entropy (information-theoretic complexity)
- AST-level: number of function/class changes, scope depth
- Similarity to other candidates for same issue (diversity signal)

**Behavioral features** (from turn_metrics, SERA only):
- Turns used
- First edit turn / Parkinson's ratio
- Tokens consumed
- Action distribution (% search vs read vs edit)
- Repeat rate (repeated_action count)
- Context growth rate

**Issue features** (from problem_statement):
- Repo name (categorical)
- Problem statement length
- Number of files mentioned
- Whether error traceback is included

### 3.2 Model A: XGBoost Classifier

- **Input**: Feature vector from 3.1
- **Output**: P(tests_pass)
- **Training**: LOIO-CV (train on 49 issues, predict on 1, repeat 50x)
- **Hyperparameter tuning**: 5-fold inner CV within each LOIO fold (nested CV)
- **Class imbalance**: Use `scale_pos_weight` or SMOTE
- **Feature importance**: SHAP values to identify which features carry signal
- **Key question**: Do patch features dominate, or do issue/harness features dominate? If the latter, the verifier is learning issue difficulty, not patch quality.

### 3.3 Model B: Pairwise Ranking (Within-Issue)

Instead of binary classification, train to rank patches:
- For each issue, form all (passing_patch, failing_patch) pairs
- Train a pairwise ranking model (LambdaMART or RankNet)
- **Input**: Feature difference vector (patch_A features - patch_B features)
- **Output**: P(A is better than B)
- **Advantage**: Controls for issue difficulty (both patches are for the same issue)
- **Evaluation**: NDCG@1 and NDCG@3 across 50 issues via LOIO-CV

### 3.4 Model C: Embedding-Based Ranking (If Signal Exists)

Only attempt if Models A/B show clear signal above baselines:
- Use a frozen code LLM (CodeBERT, StarCoder-base, or Devstral) to embed (problem_statement, patch_diff)
- Train a small MLP ranking head on top of frozen embeddings
- **Do NOT fine-tune the LLM backbone** — insufficient data
- **Evaluation**: Same LOIO-CV framework as above

### 3.5 Evaluation Metrics

| Metric | Definition | Why |
|--------|-----------|-----|
| **Top-1 pass rate** | Fraction of issues where the top-ranked candidate passes | Primary metric — this is what matters in practice |
| **Top-3 pass rate** | Fraction where at least one of top-3 passes | Shows ranking quality beyond top pick |
| **NDCG@N** | Normalized discounted cumulative gain | Standard ranking metric |
| **Precision-at-threshold** | If P(pass) > 0.5, predict pass | Binary classification quality |
| **Feature importance** | SHAP values for top features | Interpretability — what did the model learn? |
| **Confounder check** | Accuracy when model/harness features are ablated | Does the model rely on patch features or metadata? |

### 3.6 Exit Criteria for Phase 3

- [ ] At least one trained model beats best Phase 2 baseline by >5pp on top-1 pass rate (outside CI overlap)
- [ ] Feature importance shows patch features (not issue/harness metadata) in top-5
- [ ] Confounder check: performance doesn't collapse when model_name and harness_name are removed
- [ ] Results reproducible: LOIO-CV variance reported, no cherry-picked folds

---

## Phase 4: Scale Decision

**Only if Phase 3 succeeds.**

### If Signal Exists (Phase 3 positive):
1. **Collect data on full SWE-bench Lite** (300 issues) or SWE-bench Verified (500 issues)
2. Re-run Phase 3 models on larger dataset — does the signal hold?
3. If yes at 300+ issues: NOW consider LLM fine-tuning (embedding-based or full)
4. If yes at 500+ issues: Implement verifier-in-loop agent (Phase 5)

### If No Signal (Phase 3 negative):
1. The fix-to-pass gap is not patchable by learned verification at this data scale
2. Pivot to: stronger test generation (automated test synthesis), or domain-specific formal verification (Leanstral pattern)
3. Write up negative result — this is still valuable for the bitter lesson thesis

### Phase 5: Verifier-in-Loop Agent (Contingent)

Integrate the trained verifier as a tool in the agent loop (Leanstral pattern for general code):
- Agent generates patch → verifier scores → agent iterates or submits
- Measure: does early rejection signal break the Parkinson's pattern?
- Measure: wall-clock time and cost vs. unverified agent
- Compare to: LLM-as-judge in-loop (no training, just API calls)

---

## Infrastructure Requirements

### Compute (g7e.24xlarge — 4x RTX PRO 6000 Blackwell)

| Phase | GPU-hours | Wall-clock (est.) | Cost @ $10.20/hr |
|-------|----------:|-------------------:|------------------:|
| Phase 1: SERA instrumented runs | 20h | 10h (2 GPU parallel) | $102 |
| Phase 1: Adapter patch-diff runs | 12h | 6h (2 GPU parallel) | $61 |
| Phase 1: N=16 candidate generation | 64h | 32h (2 GPU parallel) | $326 |
| Phase 1: N=4 alternative | 16h | 8h (2 GPU parallel) | $82 |
| Phase 2: LLM-as-judge | 0 (API) | <1h | ~$48 (API costs) |
| Phase 3: XGBoost/ranking | 0 (CPU) | <1h | $0 |
| Phase 3: Embedding extraction | 2h | 1h | $10 |
| **Total (N=16 path)** | **~98h** | **~50h** | **~$547** |
| **Total (N=4 path)** | **~50h** | **~25h** | **~$303** |

### Software

- Python 3.11+, vLLM v0.16.0
- XGBoost, scikit-learn, SHAP
- SWE-bench dataset (pin version in manifest)
- Anthropic API access (for LLM-as-judge baseline)

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| No learnable signal in patches at N=50 | HIGH | Experiment is negative | Frame as pilot; define scale-up criteria |
| git diff returns empty for committed changes | HIGH | Missing patch diffs | Fix `_get_git_diff` to use `git diff HEAD` before any data collection |
| LLM-as-judge beats all trained models | MEDIUM | Training is wasted effort | Run Phase 2 fully before Phase 3; this outcome is still useful |
| Verifier learns issue difficulty, not patch quality | MEDIUM | False positive — model appears good but doesn't generalize | Pairwise within-issue training; feature ablation tests |
| SWE-bench Lite subset changes silently | LOW | Results not reproducible | Pin instance_ids in manifest file |
| g7e instance terminated mid-collection | LOW | Lost partial data | Incremental writes (already implemented); sync regularly |
| Class imbalance defeats classifier | MEDIUM | All predictions = "fail" | Pairwise ranking (avoids classification); weighted loss |

---

## Success Criteria

### Minimum Viable Result (pilot success)
- Top-1 pass rate of any trained model > best heuristic baseline + 5pp
- Feature importance shows patch_diff features in top-3
- Written analysis of what patch features predict test passage

### Strong Result (justifies scale-up)
- Top-1 pass rate > 7-harness ensemble ceiling (32%)
- Cross-harness generalization: model trained on SERA data predicts OpenCode/Claude Code outcomes
- Clear evidence that patch complexity features (not issue identity) drive predictions

### Negative Result (still publishable)
- Documented evidence that at N=50, patch features do not predict test outcomes above baseline
- Analysis of why (issue difficulty dominates? harness artifacts? insufficient variation?)
- Recommendation: minimum data scale needed for signal detection

---

## Relationship to Other Blueprints

- **agent-harness**: Provides Phase 1/2 results, harness_eval.py infrastructure, eval framework
- **agent-swarm**: Provides Phase 1 model×harness matrix, swarm_eval.py, concurrent runner
- **bitter-lesson-time-horizon** (blog): Verifier strength as the third axis in the time horizon equation
- **Leanstral clipping**: Template pattern (sparse specialist + perfect verifier + MCP)
- **RALPH Loop**: Production example of verifier-in-loop (terraform toolchain = strong verifier)
