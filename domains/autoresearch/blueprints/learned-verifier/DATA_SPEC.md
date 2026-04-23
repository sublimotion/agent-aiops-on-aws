# Learned Verifier — Data Specification

**Date**: 2026-03-22
**Status**: Active — inventory complete, collection gaps identified, fixes deployed

## 1. Data Inventory (Audited 2026-03-22)

### 1.1 Salvageable Data (Exists on g7e)

| Source | Location | Rows | Has Diff | Has Transcript | Has tests_pass | Has Behavioral |
|--------|----------|-----:|:--------:|:--------------:|:--------------:|:--------------:|
| **SVG production-run1** | `sera-data/production-run1/svg_results.jsonl` | 300 | No (not extracted) | 28 (in `train.jsonl`) | **Yes** (53 pass) | Partial (turns, latency) |
| **OpenCode × SERA-32B diffs** | `agent-harness-scripts/results/diffs/opencode/` | 27 | **Yes** (27 .diff files, 13K lines) | No | No (summary only) | No |
| **Droid × Devstral diffs** | `agent-harness/results/diffs/droid/` | 24 | **Yes** (24 .diff files) | No | **Yes** (`eval_droid.jsonl`) | No |
| **Droid temp diffs** | `/tmp/droid_diff_*.diff` | 50 | **Yes** (24+ non-empty) | No | Join with `eval_droid.jsonl` | No |
| **Droid trajectories** | `agent-harness/results/trajectories/droid/` | 27 | Embedded | **Yes** (raw events) | Join with eval | Partial |
| **Droid raw trajectories** | `agent-harness/results/trajectories/raw/droid_*.jsonl` | 50 | Embedded | **Yes** (raw events) | Join with eval | Partial |
| **Phase 1 turn metrics** | `agent-harness-scripts/results/phase1_{A-F}_turns.jsonl` | 6,599 | No | No | No | **Yes** (full behavioral) |
| **Phase 1 summaries** | `agent-harness-scripts/results/phase1_{A-F}.jsonl` | 300 | No | No | No | Partial |
| **Phase 2 summaries** | `agent-harness-scripts/results/phase2_{sera,langgraph,aider}.jsonl` | 150 | No | No | No | Partial |
| **Phase 2b summaries** | `agent-harness/results/phase2b_{6 harnesses}.jsonl` | ~270 | No | No | No | Partial |
| **Eval results (7 harnesses)** | `agent-harness/results/eval_{7}.jsonl` | ~248 | No | No | **Yes** (~60 pass) | No |
| **Swarm results (9 configs)** | `agent-harness/results/swarm/swarm_phase1_*.jsonl` | ~450 | No | No | **Yes** (~30 pass) | Partial (first_edit_turn) |
| **SERA-32B × SERA shards** | `agent-harness-scripts/results/phase2_sera_s{0-3}.jsonl` | 50 | No | No | No (fix_generated only) | Partial |
| **SERA-32B × OpenCode shards** | `agent-harness-scripts/results/phase2_opencode_s{0-3}.jsonl` | 50 | No | No | No (fix_generated only) | Partial |
| **CoderForge** | `sera-data/coderforge/train_trl.jsonl` | ~258K | Unknown | Yes | Yes | Unknown |

### 1.2 Data That Is Gone (Cannot Be Recovered)

| Source | What's Lost | Why |
|--------|------------|-----|
| **All SERA-path diffs** (Phase 1 A-F, Phase 2, swarm × SERA, SERA-32B × SERA) | Patch diffs for ~600+ runs | `run_sera` / `run_instrumented_loop` never called `_get_git_diff`; workspaces cleaned in `finally` |
| **All SERA-path transcripts** | Full conversation histories | `run_instrumented_loop` doesn't save messages; only turn-level metrics survive |
| **Phase 2b Devstral diffs** (ohmypi, piagent, claude_code, codex, opencode) | 270 patch diffs | Runs predated diff capture in `multi_harness_eval.py`; workspaces cleaned |
| **Swarm diffs** (all 9 model×harness configs) | 450 patch diffs | `swarm_eval.py` tried `eval_result.patch_diff` but `EvalResult` has no such field |

### 1.3 Joinable Label Sources

To create labeled (diff, outcome) pairs, join these:

| Diff Source | Label Source | Join Key | Expected Pairs |
|-------------|-------------|----------|----------------|
| OpenCode × SERA-32B diffs (27) | Need gold_eval run | `instance_id` | ~27 |
| Droid diffs (24) | `eval_droid.jsonl` | `instance_id` | 24 (2 pass, 22 fail) |
| Droid temp diffs (50) | `eval_droid.jsonl` | `instance_id` | ~24 matched |
| SVG production-run1 | Self-contained | `instance_id` | 300 (53 pass, 247 fail) |
| SVG train.jsonl transcripts | SVG results | `instance_id` | 28 (all pass+accepted) |

**Total immediately usable labeled diffs**: ~24 (Droid) + need to run gold_eval on OpenCode × SERA-32B (27)

**Total labeled summary rows (no diffs)**: ~900+ across all eval files

## 2. Root Cause: Three Bugs in Diff/Transcript Capture

### Bug 1: `EvalResult` missing `patch_diff` field
- **File**: `harness_eval.py` (line ~505)
- **Impact**: `run_instrumented_loop` returns `EvalResult` without diff. All code that tries `eval_result.patch_diff` gets `AttributeError` or `None`.
- **Fix**: Add `patch_diff: Optional[str] = None` to `EvalResult` dataclass. Set it at end of `run_instrumented_loop` via `_get_git_diff(workspace)`.

### Bug 2: `run_sera` in `multi_harness_eval.py` never captures diff
- **File**: `multi_harness_eval.py` (line ~199-232)
- **Impact**: SERA harness results have `diff=None` even when workspace has changes. Only `run_cli_harness` captures diffs.
- **Fix**: Add `result.diff = _get_git_diff(workspace)` before `run_sera` returns. Must happen BEFORE workspace cleanup in `finally`.

### Bug 3: `_get_git_diff` uses `git diff` (unstaged only)
- **File**: `harness_eval.py` (line ~431)
- **Impact**: If agent runs `git add` or `git commit`, diff is empty. False negatives for `fix_generated`.
- **Fix**: Change to `git diff HEAD` to capture both staged and unstaged changes vs HEAD.

### Fix Status

| Bug | Fixed | Deployed |
|-----|:-----:|:--------:|
| Bug 1: EvalResult.patch_diff | **Yes** (2026-03-22) | Both copies on g7e (`agent-harness-scripts/` + `agent-harness/`) |
| Bug 2: run_sera diff capture | **Yes** (2026-03-22) | `multi_harness_eval.py` + `swarm_eval.py` (fallback to `_get_git_diff`) |
| Bug 3: git diff HEAD | **Yes** (2026-03-22) | Both copies of `harness_eval.py` |

## 3. Collection Plan (Re-runs Required)

### Priority 1: High-Value Re-runs (Immediately Actionable)

These re-runs collect diffs + labels for the verifier with the fixed code.

| Run | Model | Harness | Issues | GPU-hours | Output |
|-----|-------|---------|--------|-----------|--------|
| **R1: SERA-32B × SERA** | allenai/SERA-32B | SERA | 50 | ~4h (4 GPU shards) | 50 diffs + tests_pass + behavioral |
| **R2: Qwen3.5 × SERA** | Qwen3.5-397B-A17B | SERA | 50 | ~4h (TP4) | 50 diffs + tests_pass + behavioral |
| **R3: Devstral × SERA** | Devstral Small 2 24B | SERA | 50 | ~4h (1 GPU) | 50 diffs + tests_pass + behavioral |
| **R4: gold_eval on OpenCode × SERA-32B** | n/a | n/a | 27 | ~1h | 27 tests_pass labels for existing diffs |

**Total: ~13 GPU-hours, yields ~177 labeled diffs**

### Priority 2: Multi-Harness Diversity (Extends Verifier Training Data)

| Run | Model | Harness | Issues | GPU-hours | Output |
|-----|-------|---------|--------|-----------|--------|
| R5: Devstral × OpenCode | Devstral Small 2 24B | OpenCode | 50 | ~4h | 50 diffs + tests_pass |
| R6: Devstral × Claude Code | Devstral Small 2 24B | Claude Code | 50 | ~8h (TP2) | 50 diffs + tests_pass |
| R7: Qwen3.5 × OpenCode | Qwen3.5-397B-A17B | OpenCode | 50 | ~4h (TP4) | 50 diffs + tests_pass |

**Total: ~16 GPU-hours, yields ~150 additional labeled diffs**

### Priority 3: Best-of-N Candidate Generation (Phase 3 of Verifier Spec)

| Run | Model | Harness | Issues | N per Issue | Total Patches | GPU-hours |
|-----|-------|---------|--------|-------------|---------------|-----------|
| R8: Devstral × SERA (N=4) | Devstral Small 2 24B | SERA | 50 | 4 | 200 | ~16h |
| R9: Devstral × SERA (N=16) | Devstral Small 2 24B | SERA | 50 | 16 | 800 | ~64h |

**Use N=4 first. Only scale to N=16 if Phase 2 baselines show headroom.**

## 4. Target Dataset for Verifier Training

After Priority 1 + existing data:

| Data Type | Rows | Positive (pass) | Negative (fail) | Features Available |
|-----------|-----:|:---------------:|:----------------:|-------------------|
| SVG results (production-run1) | 300 | 53 | 247 | tests_pass, line_recall, accepted, turns, latency |
| SVG transcripts | 28 | 28 | 0 | Full message history (pass-only) |
| SERA re-runs (R1-R3) | 150 | ~30-50 | ~100-120 | **patch_diff**, tests_pass, full behavioral |
| Droid diffs + eval | 24 | 2 | 22 | patch_diff, tests_pass |
| OpenCode × SERA-32B + gold_eval (R4) | 27 | ~5-10 | ~17-22 | patch_diff, tests_pass |
| Eval labels (7 harnesses) | ~248 | ~60 | ~188 | tests_pass (no diffs) |
| Phase 1 behavioral (A-F) | 6,599 turns | n/a | n/a | Full turn-level features |
| **Total labeled diffs** | **~501** | **~90-140** | **~360-410** | |
| **Total labeled rows (any)** | **~750+** | **~145-210** | **~540-605** | |

### Class Balance

Expected positive rate: ~20-25%. Manageable with:
- `scale_pos_weight` in XGBoost
- Pairwise ranking (within-issue, avoids class imbalance)
- SMOTE for logistic regression

### Feature Matrix

| Feature Group | Source | Availability | N Features |
|---------------|--------|:------------:|:----------:|
| **Patch features** | patch_diff | After re-runs | ~10 (lines added/removed, files touched, test mods, diff entropy) |
| **Behavioral features** | turn_metrics | Phase 1 data + re-runs | ~11 (turns, first_edit, repeat_rate, action_pct_*, context_growth) |
| **SVG consensus** | SVG pipeline | production-run1 only | 3 (line_recall, accepted, repro_turns) |
| **Issue features** | SWE-bench metadata | Always available | ~5 (repo, statement_length, traceback_present) |
| **Model/harness identity** | Run metadata | Always available | 2 (categorical, for confounder check) |

## 5. Schema: Per-Run Output File

After fixes, each JSONL result row should contain:

```json
{
  "instance_id": "django__django-11039",
  "model": "sera-32b",
  "harness": "sera",
  "tests_pass": true,
  "fix_generated": true,
  "turns_used": 18,
  "tokens_consumed": 72000,
  "input_tokens": 65000,
  "output_tokens": 7000,
  "total_latency_ms": 95000,
  "first_edit_turn": 8,
  "error": null,
  "patch_diff": "diff --git a/django/... (full diff content)"
}
```

Diffs are also saved to `results/diffs/{harness}/{instance_id}.diff` as separate files.

Trajectories (when available) saved to `results/trajectories/{harness}/{instance_id}.jsonl`.

## 6. Validation Checklist (Run Before Each Collection)

- [ ] `_get_git_diff` uses `git diff HEAD` (not `git diff`)
- [ ] `EvalResult` has `patch_diff` field
- [ ] `run_sera` sets `result.diff = _get_git_diff(workspace)` before returning
- [ ] Diff is captured BEFORE workspace cleanup (`finally` block)
- [ ] Run 1 issue end-to-end, verify diff file exists and is non-empty
- [ ] Verify `tests_pass` label is correct (run gold_eval on 3 known-pass issues)
- [ ] 50-issue subset matches manifest (instance_ids pinned)
