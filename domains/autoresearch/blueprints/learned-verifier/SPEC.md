# Learned Verifier: Closing the Fix-to-Pass Gap

**Status**: DRAFT — pending data collection
**Created**: 2026-03-19
**Updated**: 2026-03-19 (v2 — soft verifier framing, SERA SVG integration)
**Depends on**: agent-harness (Phase 1/2 complete), agent-swarm (Phase 1 complete)

## Executive Summary

Train a model to predict whether a coding agent's patch will pass tests, then use it to select the best patch from N candidates. This is the coding-agent instantiation of Best-of-N with a learned reward model.

**The core bet**: the binding constraint on coding agent performance is verification quality, not model scale, harness design, or finetuning.

## The Verification Spectrum

Verification is not binary (have/don't have). It's a spectrum of signal strength:

```
Hard verifier     Lean compiler, terraform validate       100% precision, deterministic
                  ↑
Strong verifier   Test suite (SWE-bench gold tests)       50-80% precision, expensive (Docker)
                  ↑
Soft verifier     SERA SVG consensus, behavioral          Probabilistic, cheap
                  telemetry, Phoenix trace analysis
                  ↑
Learned verifier  Model trained on soft verifier signals   Approximates strong verifier
                  ↑
No verifier       Blind submission                         19% precision (Codex)
```

**Key insight**: SERA's instrumented agent loop is already a soft verifier. It collects behavioral signals every turn — action types, context growth, repetition patterns, edit timing. These signals correlate with patch quality but aren't currently used for selection. The Phoenix AI engineering loop (Arize) does the same thing at the observability layer — traces as verification signal.

The learned verifier doesn't need to be built from scratch. It needs to make SERA's existing soft signals **predictive**.

### SERA as Soft Verifier

SERA's harness captures per-turn telemetry that encodes agent confidence and competence:

| Signal | What It Measures | Verification Intuition |
|--------|-----------------|----------------------|
| `action_type` distribution | % search vs read vs edit vs run | Confident agent edits early; confused agent keeps searching |
| `repeated_action` | Same tool call made twice | Agent is stuck / looping — low-quality trajectory |
| `first_edit_turn` / Parkinson's ratio | When first edit happens relative to budget | Late edits = low confidence, correlates with failure |
| `context_tokens` growth rate | Per-turn context size | Runaway context = lost focus, information overload |
| `diff_size` | Size of generated patch | Oversized patches = shotgun approach |
| `edit_count` | Number of edit operations | Too many edits = thrashing, not converging |
| `turns_used` / `total_turns` | Budget consumption | 30/30 turns used = likely didn't converge |

**These signals are already collected in 6,599 turn-level rows** from Phase 1 (configs A-F). They have never been used as predictive features.

### SERA's SVG Pipeline as Consensus Verifier

The full SERA methodology (Tim Dettmers, Ai2 — "Soft-Verified Efficient Repository Agents") includes a 7-stage Soft-Verified Generation pipeline in `sera-datagen.py`:

1. **GENERATE**: Model fixes a bug via multi-turn tool use → `patch_1`
2. **VERIFY**: Run repo test suite (hard verification)
3. **DESCRIBE**: Convert `patch_1` to a PR description (natural language)
4. **REPRODUCE**: Model generates a second fix from PR description alone → `patch_2`
5. **SCORE**: Compute line-level recall between `patch_1` and `patch_2`
6. **FILTER**: Accept if tests_pass AND recall ≥ 0.8
7. **FORMAT**: Save as training pairs

Stage 5 (`stage_score`) is a **consensus verifier**: if two independent generations converge on the same code changes, confidence is higher. The scoring function computes set intersection of non-header diff lines:

```python
def stage_score(original_patch, repro_patch):
    original_lines = {line.strip() for line in original_patch.split("\n")
                      if line.strip() and not line.startswith(("---","+++","@@","diff","index"))}
    repro_lines = {line.strip() for line in repro_patch.split("\n")
                   if line.strip() and not line.startswith(("---","+++","@@","diff","index"))}
    return len(repro_lines & original_lines) / len(original_lines)
```

**The `harness_eval.py` fork only uses stages 1-2, discarding the SVG consensus signal.** The full pipeline stores both patches AND both conversation transcripts (`fix_messages`, `repro_messages`). This is existing infrastructure we can leverage.

### Connection to Phoenix / Observability-as-Verification

The Phoenix AI engineering loop (Arize) demonstrates the same pattern at a different layer: give the agent programmatic access to its own traces → agent analyzes patterns → agent hypothesizes root causes → agent experiments. Their finding: "agent behavior is documented by telemetry, not code" — traces are the real verification signal for agent systems.

This validates our approach: SERA's turn-level telemetry IS the agent's trace data. Training a verifier on it is automating what the Phoenix demo does manually.

---

## Reality Check: What We Actually Have

A thorough audit (2026-03-19) of all existing experiment data revealed a significant gap between what we assumed and what exists.

### Data Inventory (Audited)

| Source | Files | Rows | patch_diff | transcript | tests_pass | turn_metrics |
|--------|------:|-----:|:----------:|:----------:|:----------:|:------------:|
| Harness Phase 1 (A-F) | 6+6 | 300+6,599 | NONE | NONE | NONE | YES (6,599 turn rows) |
| Harness Phase 2 (sera, langgraph, aider) | 3 | 150 | NONE | NONE | NONE | partial |
| Harness Phase 2b (6 harnesses) | 6 | 270 | NONE | NONE | always false* | partial |
| Harness Eval (7 harnesses) | 7 | 248 | NONE | NONE | YES (60 pass) | NONE |
| Swarm Phase 1 (9 configs) | 9 | 491 | NONE | NONE | partial (12 pass) | NONE |
| Recovered diffs | 1 | 1 | 1 file | NONE | N/A | N/A |
| SERA datagen (`sera-datagen.py`) | — | — | YES (both patches) | YES (both transcripts) | YES | — |
| **TOTAL** | **32+** | **~1,459** | **1** | **0** | **~72 pass** | **6,599 turns** |

*Phase 2b `pass` field is a pre-eval placeholder (always false). Actual test results are only in eval_* files.

**Key discovery**: The original `sera-datagen.py` in `gpu-serving/blueprints/devstral-sera/scripts/` stores BOTH patches, BOTH full conversation transcripts, PR descriptions, line-level recall scores, AND test results. If any SVG pipeline runs were completed previously, that data contains exactly the training signal we need. Check `/mnt/nvme/sera-data/` on g7e for completed SVG runs.

### What This Means

1. **Zero patch diffs in the harness/swarm results.** The instrumentation was deployed to g7e but has never been run.
2. **Zero transcripts in harness/swarm results.** Transcript capture only works through the SERA path.
3. **50 unique issues is the real sample size**, not 1,459. Repeated measurements of 50 underlying problems.
4. **Only 72 positives (tests_pass=true)** across all eval files. Class imbalance is ~1:8.
5. **BUT: 6,599 turn-level rows exist** from Phase 1 with behavioral features. These are soft verifier signals that have never been used as predictive features.
6. **AND: `sera-datagen.py` already captures exactly what we need** — both patches, both transcripts, line-recall scores. We may have existing SVG data on g7e.

### Instrumentation Status

| Component | Code Ready | Deployed to g7e | Data Produced |
|-----------|:----------:|:---------------:|:-------------:|
| `harness_eval.py` — patch_diff capture | YES | YES | NO |
| `harness_eval.py` — transcript capture | YES | YES | NO |
| `swarm_eval.py` — patch_diff capture | YES | YES | NO |
| `swarm_eval.py` — transcript (SERA only) | YES | YES | NO |
| `sera-datagen.py` — full SVG pipeline | YES | ON g7e | CHECK `/mnt/nvme/sera-data/` |
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

**Why not fine-tune an LLM?** With 50 unique issues, each appearing ~12x with different patches, an LLM will memorize `problem_statement -> pass_probability` rather than learning general patch quality assessment. The model sees the same problem statement ~12 times during training — it learns issue difficulty, not patch quality. Fine-tuning requires 10K-100K+ unique inputs for the (problem, patch) -> quality mapping.

**The right framing**: this experiment answers "Is there learnable signal in patch features and behavioral telemetry that predicts test outcomes, separable from issue difficulty and harness artifacts?" If yes, collect more data (SWE-bench full: 2,294 issues) and scale up. If no, LLM fine-tuning won't help.

### Confounders to Control

The label `tests_pass` encodes: `f(issue_difficulty, model_capability, harness_quality, patch_quality)`. The verifier only sees (problem_statement, patch_diff, behavioral_features). Three confounders must be addressed:

1. **Issue difficulty** — easy issues pass more often regardless of patch quality. Control: pairwise training within each issue (compare passing vs failing patches for the same issue).
2. **Harness bias** — a harness bug causing spurious failures looks like "bad patch" to the verifier. Control: train within-harness, test cross-harness.
3. **Model identity** — model-specific code style correlates with pass rate. Control: include model_name as feature, measure feature importance, verify patch features dominate.

### Scope and Non-Goals

**In scope**: Proof-of-concept on SWE-bench Lite 50 showing that soft verifier signals and/or patch features predict test outcomes better than random selection.

**Not in scope** (requires more data):
- Generalizing to unseen repos or languages
- Replacing test suites in production CI/CD
- Fine-tuning an LLM verifier
- Process reward modeling (per-turn verification)

---

## Phase 0: Recover Existing Data

**Goal**: Before collecting anything new, determine what usable data already exists.

### 0.1 Check for SVG Pipeline Output on g7e

The `sera-datagen.py` pipeline writes results to `/mnt/nvme/sera-data/`. Each completed run produces:
- `fix_patch` and `repro_patch` (both diffs)
- `fix_messages` and `repro_messages` (both full transcripts)
- `line_recall` (consensus score)
- `tests_pass` (hard verification)
- `accepted` (soft verification: tests_pass AND recall >= 0.8)

```bash
ssh g7e "ls -la /mnt/nvme/sera-data/ && wc -l /mnt/nvme/sera-data/*.jsonl 2>/dev/null"
```

If SVG data exists, it contains EXACTLY the training signal for a patch-level verifier + the consensus baseline. This could shortcut Phase 1 entirely.

### 0.2 Behavioral Baseline with Existing Turn Data

We already have 6,599 turn-level rows (Phase 1 configs A-F, all SERA × Devstral 24B). These contain soft verifier signals but lack `tests_pass` labels.

**The join problem**: Phase 1 summary files have `fix_generated` but not `tests_pass`. The `eval_sera.jsonl` file has `tests_pass` for 23 instances. If we join Phase 1 turn data with eval_sera labels, we get 23 labeled examples with full behavioral features.

23 examples is thin, but sufficient for a quick signal check: do behavioral features (turns_used, first_edit_turn, repeat_count, context_growth_rate) separate passes from failures? A simple logistic regression on 23 examples with 5 features can answer this in minutes on a laptop.

### 0.3 Exit Criteria for Phase 0

- [ ] Inventory of `/mnt/nvme/sera-data/` contents (SVG pipeline output)
- [ ] Join Phase 1 turn data with eval_sera.jsonl to get labeled behavioral features
- [ ] Quick signal check: logistic regression on behavioral features vs tests_pass (23 examples)
- [ ] Decision: does SVG data exist in quantity? If yes, skip to Phase 2 baselines

---

## Phase 1: Data Collection

**Goal**: Collect patch diffs, transcripts, and behavioral telemetry with tests_pass labels.

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

### 1.3 SVG Consensus Runs (Soft Verification Signal)

Run the full `sera-datagen.py` SVG pipeline to get consensus scores. Each issue gets TWO patches and a line-recall score.

| Run | Model | Issues | Output | Priority |
|-----|-------|--------|--------|----------|
| SVG-1 | Devstral 24B | 50 | 50 × (patch_1, patch_2, recall, both transcripts) | HIGH |

**This doubles our patch diversity** (100 patches for 50 issues) and provides a consensus baseline for free.

### 1.4 Adapter Runs — Patch Diff Only

OpenCode and Claude Code adapters cannot return transcripts (bash subprocess, JSON on stdout). We can only capture `git diff HEAD` after the adapter completes.

| Run | Model | Harness | Issues | Signal |
|-----|-------|---------|--------|--------|
| ADAPT-1 | Devstral 24B | OpenCode | 50 | patch_diff + tests_pass only |
| ADAPT-2 | Devstral 24B | Claude Code | 50 | patch_diff + tests_pass only |
| ADAPT-3 | Qwen3.5-397B | OpenCode | 50 | patch_diff + tests_pass only |

**Total: 150 additional patch diffs (no transcripts). ~12 GPU-hours.**

### 1.5 Generate N=16 Candidate Patches (for Best-of-N Evaluation)

For the verifier to be useful, we need multiple candidate patches per issue. Run the cheapest high-fix-rate config 16 times per issue with different random seeds (temperature > 0).

| Generator | Config | Issues | Candidates/Issue | Total Patches |
|-----------|--------|--------|-----------------|---------------|
| Devstral 24B x SERA | Phase 1 Config D (30 turns) | 50 | 16 | 800 |

**Each patch needs**: patch_diff (captured), tests_pass (run gold tests), behavioral features (turn_metrics), problem_statement (from SWE-bench).

**This is the most expensive step: ~64 GPU-hours** (800 runs x ~5 min each on 1x B200). Can be parallelized across 2 GPUs.

**Alternative (cheaper)**: Generate N=4 per issue (200 runs, ~16 GPU-hours). Lower statistical power for best-of-N but sufficient to establish whether ranking signal exists.

### 1.6 Data Manifest

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
      "has_svg_recall": false,
      "tests_pass": false,
      "fix_generated": true
    }
  ]
}
```

### 1.7 Exit Criteria for Phase 1

Phase 1 is complete when:
- [ ] At least 200 trajectories with patch_diff (from SERA runs)
- [ ] At least 100 trajectories with full transcript (SERA only)
- [ ] At least 50 SVG consensus scores (from sera-datagen.py pipeline)
- [ ] N=16 (or N=4) candidate patches per issue for at least 40 issues, each with tests_pass label
- [ ] Manifest file validates: no missing fields, no null patch_diffs where fix_generated=true
- [ ] `_get_git_diff` confirmed to use `git diff HEAD` (not just `git diff`)
- [ ] 50-issue subset pinned in manifest with dataset version

---

## Phase 2: Baselines (No Training Required)

**Goal**: Establish performance baselines before training anything. If baselines are strong, training may not be needed.

### 2.0 Behavioral Signal Check (Existing Data)

**Can run immediately with existing data** — no new collection needed.

Join Phase 1 turn data (6,599 rows, configs A-F) with eval_sera.jsonl (23 labeled examples):
- Aggregate per-issue behavioral features: avg turns, first_edit_turn, repeat_count, context growth rate, action distribution
- Fit logistic regression: behavioral features -> tests_pass
- Report: AUC, accuracy, feature coefficients
- If AUC > 0.65 on 23 examples: strong signal exists in behavioral data

**This takes 30 minutes on a laptop and costs nothing.** It answers the most basic question: do SERA's soft verifier signals predict test outcomes at all?

### 2.1 Random Baseline

For each issue with N candidates:
- `P(at least 1 pass in N)` = `1 - (1 - p_i)^N` where `p_i` is issue-level pass rate
- Random top-1 selection: expected pass rate = per-issue average pass rate
- Report: mean and 95% CI across 50 issues

### 2.2 Simple Heuristics

**Behavioral heuristics (from SERA soft verifier signals):**

| Heuristic | Input | Verification Intuition |
|-----------|-------|----------------------|
| Fewest turns used | turn_metrics | Agent converged quickly = higher confidence |
| Earliest first_edit_turn | turn_metrics | Didn't waste turns exploring — knew what to do |
| Lowest repeat_count | turn_metrics | No looping = clean trajectory |
| Lowest context growth rate | turn_metrics | Stayed focused, didn't bloat context |
| Highest edit-to-search ratio | turn_metrics | More editing than searching = decisive |

**Patch heuristics:**

| Heuristic | Input | Rationale |
|-----------|-------|-----------|
| Shortest patch (fewest diff lines) | patch_diff | Minimal changes less likely to introduce bugs |
| Fewest files touched | patch_diff | Single-file fixes more likely correct |
| No test file modifications | patch_diff | Patches that modify test files are suspicious |

**Consensus heuristic (from SERA SVG pipeline):**

| Heuristic | Input | Rationale |
|-----------|-------|-----------|
| Highest SVG line-recall | sera-datagen output | Two independent generations agree = higher quality (SERA's core insight) |
| Majority vote among N candidates | N patch_diffs | Cluster patches by similarity, pick largest cluster |

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

**Cost estimate**: 50 issues x 16 candidates x ~$0.01/call = ~$8 for Sonnet, ~$40 for Opus.

### 2.4 SVG Consensus Baseline

Use SERA's describe-reproduce-score pipeline as a standalone verifier:
- For each candidate patch, run stages 3-5 (describe -> reproduce -> score)
- Rank candidates by line-recall score
- Measure top-1 pass rate

This uses SERA's own soft verification as a baseline. If it works well, we don't need to train anything — just integrate the SVG scoring step into the selection pipeline.

**Cost**: ~2 extra inference calls per candidate (describe + reproduce). For N=16 x 50 issues = 1,600 extra calls. Cheap on self-hosted Devstral.

### 2.5 Exit Criteria for Phase 2

- [ ] Behavioral signal check completed (logistic regression on 23 examples)
- [ ] All baselines computed with LOIO-CV on N-candidate data
- [ ] Results table with 95% confidence intervals
- [ ] Decision: does any baseline already meet or exceed the 7-harness ensemble ceiling (32%)?
- [ ] If LLM-as-judge > 40% top-1, reconsider whether training is needed at all
- [ ] If SVG consensus > 40% top-1, integrate it directly without training

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

**Behavioral features** (from SERA soft verifier — turn_metrics):
- Turns used
- First edit turn / Parkinson's ratio
- Tokens consumed
- Action distribution (% search vs read vs edit vs run_command)
- Repeat rate (repeated_action count)
- Context growth rate (tokens at turn N vs turn 1)
- Edit success rate (edit_applied / total edit attempts)

**Consensus features** (from SVG pipeline):
- Line-recall between original and reproduced patch
- Whether reproduction also passes tests
- PR description length / quality

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
- **Key question**: Do patch features dominate, behavioral features dominate, or do issue/harness features dominate? This tells us which verification tier carries the most signal.

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
| **Verification tier analysis** | Compare: behavioral-only vs patch-only vs combined | Which soft verifier tier carries the most signal? |

### 3.6 Exit Criteria for Phase 3

- [ ] At least one trained model beats best Phase 2 baseline by >5pp on top-1 pass rate (outside CI overlap)
- [ ] Feature importance shows patch or behavioral features (not issue/harness metadata) in top-5
- [ ] Confounder check: performance doesn't collapse when model_name and harness_name are removed
- [ ] Verification tier analysis: clear winner between behavioral, patch, and consensus features
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
- Agent generates patch -> verifier scores -> agent iterates or submits
- Measure: does early rejection signal break the Parkinson's pattern?
- Measure: wall-clock time and cost vs. unverified agent
- Compare to: LLM-as-judge in-loop (no training, just API calls)
- Compare to: SVG consensus in-loop (describe-reproduce-score as verification step)

---

## Infrastructure Requirements

### Compute (g7e.24xlarge — 4x RTX PRO 6000 Blackwell)

| Phase | GPU-hours | Wall-clock (est.) | Cost @ $10.20/hr |
|-------|----------:|-------------------:|------------------:|
| Phase 0: Check existing data | 0 | 1h (laptop) | $0 |
| Phase 1: SERA instrumented runs | 20h | 10h (2 GPU parallel) | $102 |
| Phase 1: SVG consensus runs | 8h | 4h (2 GPU parallel) | $41 |
| Phase 1: Adapter patch-diff runs | 12h | 6h (2 GPU parallel) | $61 |
| Phase 1: N=16 candidate generation | 64h | 32h (2 GPU parallel) | $326 |
| Phase 1: N=4 alternative | 16h | 8h (2 GPU parallel) | $82 |
| Phase 2: Baselines + LLM-as-judge | 0 (API) | <1h | ~$48 (API costs) |
| Phase 2: SVG consensus baseline | 4h | 2h | $20 |
| Phase 3: XGBoost/ranking | 0 (CPU) | <1h | $0 |
| Phase 3: Embedding extraction | 2h | 1h | $10 |
| **Total (N=16 path)** | **~110h** | **~57h** | **~$608** |
| **Total (N=4 path)** | **~62h** | **~33h** | **~$364** |

### Software

- Python 3.11+, vLLM v0.16.0
- XGBoost, scikit-learn, SHAP
- SWE-bench dataset (pin version in manifest)
- Anthropic API access (for LLM-as-judge baseline)
- `sera-datagen.py` (for SVG consensus runs)

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| No learnable signal in patches at N=50 | HIGH | Experiment is negative | Frame as pilot; define scale-up criteria |
| git diff returns empty for committed changes | HIGH | Missing patch diffs | Fix `_get_git_diff` to use `git diff HEAD` before any data collection |
| LLM-as-judge beats all trained models | MEDIUM | Training is wasted effort | Run Phase 2 fully before Phase 3; this outcome is still useful |
| SVG consensus baseline is already sufficient | MEDIUM | No training needed | This is a GOOD outcome — integrate SVG scoring directly |
| Verifier learns issue difficulty, not patch quality | MEDIUM | False positive — model appears good but doesn't generalize | Pairwise within-issue training; feature ablation tests |
| Behavioral features alone predict as well as patch features | MEDIUM | Patch diffs unnecessary for verifier | Good news — cheaper verification; still needs validation at scale |
| SWE-bench Lite subset changes silently | LOW | Results not reproducible | Pin instance_ids in manifest file |
| g7e instance terminated mid-collection | LOW | Lost partial data | Incremental writes (already implemented); sync regularly |
| Class imbalance defeats classifier | MEDIUM | All predictions = "fail" | Pairwise ranking (avoids classification); weighted loss |

---

## Success Criteria

### Minimum Viable Result (pilot success)
- Top-1 pass rate of any method (trained or baseline) > random selection + 5pp
- Clear identification of which verification tier (behavioral, patch, consensus) carries signal
- Written analysis connecting results to the verification spectrum framework

### Strong Result (justifies scale-up)
- Top-1 pass rate > 7-harness ensemble ceiling (32%)
- Cross-harness generalization: verifier trained on SERA data predicts OpenCode/Claude Code outcomes
- Verification tier analysis shows patch or behavioral features dominate over issue identity
- SVG consensus as a practical zero-training verifier for Best-of-N selection

### Negative Result (still publishable)
- Documented evidence that at N=50, neither patch features nor behavioral telemetry predict test outcomes above baseline
- Analysis of why (issue difficulty dominates? harness artifacts? insufficient variation?)
- Recommendation: minimum data scale needed for signal detection
- Contribution: the verification spectrum framework and soft verifier taxonomy

---

## Relationship to Other Blueprints and Research

- **agent-harness**: Provides Phase 1/2 results, harness_eval.py infrastructure, eval framework
- **agent-swarm**: Provides Phase 1 model x harness matrix, swarm_eval.py, concurrent runner
- **devstral-sera** (gpu-serving): Contains the full SVG pipeline (`sera-datagen.py`) — verification infrastructure we haven't been using
- **bitter-lesson-time-horizon** (blog): Verifier strength as the third axis in the time horizon equation; the soft verifier taxonomy maps to different autonomy ceilings
- **Leanstral clipping**: Template pattern (sparse specialist + perfect verifier + MCP) — hard verification end of the spectrum
- **RALPH Loop**: Production example of verifier-in-loop (terraform toolchain = strong verifier)
- **Phoenix AI Engineering Loop clipping**: Observability-as-verification pattern; "agent behavior documented by telemetry, not code" supports SERA-as-soft-verifier framing
- **SERA (Ai2/Tim Dettmers)**: Published methodology — soft verification on partially correct data; SVG pipeline as consensus verifier; 54.2% SWE-Bench Verified with SFT only
