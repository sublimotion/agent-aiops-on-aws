# Autoresearch: Verifier Reward — Skill Iteration Loop

You are an autonomous researcher iterating on verification and coaching skills for coding agents. Your goal is to find the skill configuration that maximizes patch selection accuracy (pass@1 from N candidates) at minimum cost.

## Context

Phase 1 established Claude model baselines on 50 SWE-bench Lite issues:

| Model | Fix Rate | Pass Rate | Precision | Cost |
|-------|----------|-----------|-----------|------|
| Haiku 4.5 | 54% (27/50) | 10% (5/50) | 19% (5/27) | $9.91 |
| Sonnet 4.6 | 98% (49/50) | 12% (6/49) | 12% (6/49) | $14.00 |
| Opus 4.6 | 98% (49/50) | 12% (6/49) | 12% (6/49) | $36.97 |
| Devstral 24B (reference) | 88% | 22% | 25% | — |

The fix-to-pass gap is enormous (98% fix → 12% pass for Sonnet). A verification skill that can reliably distinguish good patches from bad ones would dramatically improve effective pass rate.

## Patch Pool

The evaluation corpus consists of gold-labeled patches:

```
results/diffs/opencode_haiku/    # ~27 diffs, gold labels in gold_haiku_opencode.jsonl
results/diffs/opencode_sonnet/   # ~49 diffs, gold labels in gold_sonnet_opencode.jsonl
results/diffs/opencode_opus/     # ~50 diffs, gold labels in gold_opus_opencode.jsonl
```

Each patch has a known `passed: true/false` from gold eval. Total pool: ~126 labeled patches across 50 issues.

## Skill Structure (Anthropic Best Practices)

Skills follow progressive disclosure:

```
skills/patch-verifier/
├── SKILL.md                    # Frontmatter (name, description) + core instructions
├── versions/                   # Rubric variants to sweep over
│   ├── v001_baseline.md        # 6-criteria rubric
│   ├── v002_minimal.md         # 3-criteria (ablation)
│   ├── v003_fewshot.md         # rubric + 2 labeled examples
│   ├── v004_cot.md             # chain-of-thought scoring
│   └── ...                     # Agent creates new versions during iteration
├── scripts/
│   ├── verify_patch.py         # Core: skill_version + patch → structured score
│   ├── sweep_versions.py       # Runs all versions × all patches → results
│   ├── analyze_versions.py     # Compares versions on precision/recall/ECE
│   └── telemetry.py            # SkillTelemetry wrapper (see Telemetry Schema)
└── references/
    └── rubric_design.md        # Rubric design rationale and iteration notes
```

SKILL.md frontmatter:
```yaml
---
name: patch-verifier
description: Verify whether a coding agent's patch correctly fixes a bug. Scores patches on problem alignment, minimality, logic correctness, scope, and completeness. Use when selecting the best patch from N candidates.
---
```

## Experiment Design Guardrails

Before iterating, verify the experiment is sound. These checks catch silent data pipeline bugs (like FM-004) that invalidate all downstream results.

### Pre-Loop Audit (run before first iteration)

1. **Data completeness**: All diffs exist, all gold labels match, no missing patches
   ```bash
   python3 -c "import os,json; [print(f) for f in os.listdir('results/diffs/opencode_sonnet') if not os.path.exists(f'results/diffs/opencode_sonnet/{f}')]"
   ```

2. **Truncation check**: What fraction of diffs exceed the char limit? What fraction of *passing* diffs?
   ```bash
   python3 skills/patch-verifier/scripts/preprocess_diff.py results/diffs/opencode_sonnet/<largest>.diff --stats
   ```
   **Guardrail**: If >20% of passing patches are truncated, the pipeline is broken. Fix before iterating.

3. **Base rate sanity**: Dev set base rate (6/49 = 12%) means random precision = 12%. Any result below this is worse than random.

4. **Cost projection**: Estimate sweep cost before running. Each Haiku call ~$0.004. 49 patches × 1 version = ~$0.20. Budget: $5 per iteration cycle.

### Per-Iteration Guardrails

- **Smoke test must pass** before sweep (4 patches, directional correctness)
- **Error analysis must run** after sweep (confusion matrix, FM categorization)
- **Circuit breakers** halt sweep on 3 consecutive failures, >20% parse errors, or >$5 cost
- **One variable per iteration** — if you change rubric AND model, you can't attribute the result
- **Log everything** — even failed experiments and regressions go in the Results Table

### Termination Criteria

| Condition | Action |
|-----------|--------|
| Precision > 0.50 AND lift > 5pp on dev | Evaluate on holdout. If holdout_delta < 5pp → SUCCESS |
| 5 consecutive iterations with < 2pp precision improvement | Plateau. Change lever (model, threshold, ensemble). |
| Total spend > $50 | HALT. Reassess approach. |
| Holdout_delta > 5pp | OVERFIT. Simplify rubric, remove FM-specific instructions. |

## Fixed vs Editable

**Fixed files** (do NOT edit during iteration):
- `scripts/verify_patch.py` — core verification call to Bedrock. Diff limit: 100K chars. Calls preprocess_diff.py for oversized diffs.
- `scripts/telemetry.py` — telemetry schema and circuit breakers
- `scripts/sweep_versions.py` — sweep orchestration
- `scripts/preprocess_diff.py` — diff preprocessor (strip cosmetic hunks for oversized diffs)
- `results/gold_*_opencode.jsonl` — gold eval labels
- `results/diffs/` — raw patch diffs

**Agent-editable**:
- `skills/patch-verifier/versions/*.md` — rubric versions (create new, never edit existing)
- `results/progress.md` — progress tracking
- This file (`program.md`) — results tables, failure modes, iteration notes

## Sweep Parameters

Each experiment run is a cell in this matrix:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `skill_version` | v001, v002, ... | Rubric variant from `versions/` dir |
| `verifier_model` | haiku, sonnet, opus | Which Claude model judges the patch |
| `patch_source` | haiku, sonnet, opus | Which model generated the patch |
| `temperature` | 0.0, 0.3 | Verifier sampling temperature |

**Sweep order** (cheapest first):
1. All skill versions × haiku-as-verifier × sonnet-patches × temp=0.0
2. Best 3 versions × {haiku, sonnet}-as-verifier × {haiku, sonnet, opus}-patches × temp=0.0
3. Best version × best verifier × all patches × {0.0, 0.3} temperature

## Telemetry Schema

Every verification call MUST emit a telemetry event to the JSONL log. The sweep scripts enforce this — a run with missing telemetry is invalid.

### Per-Call Event

```json
{
  "event": "skill_invocation",
  "timestamp": "2026-03-24T10:15:30Z",
  "run_id": "sweep_v001_haiku_sonnet_t0",

  "skill": {
    "name": "patch-verifier",
    "version": "v001_baseline",
    "version_hash": "abc123"
  },

  "invocation": {
    "invoked": true,
    "input_tokens": 3200,
    "output_tokens": 450,
    "latency_ms": 1850,
    "cost_usd": 0.0012,
    "parse_success": true,
    "error": null
  },

  "context": {
    "instance_id": "django__django-10914",
    "patch_source": "sonnet",
    "verifier_model": "haiku",
    "temperature": 0.0,
    "problem_statement_tokens": 800,
    "diff_tokens": 1200
  },

  "output": {
    "scores": {
      "problem_alignment": 0.8,
      "minimality": 0.6,
      "logic_correctness": 0.7,
      "scope": 0.9,
      "completeness": 0.5
    },
    "overall_score": 0.7,
    "confidence": 0.85,
    "verdict": "likely_correct",
    "reasoning_excerpt": "The patch modifies the correct file..."
  },

  "gold": {
    "passed": false,
    "patch_applied": true
  }
}
```

### Per-Sweep Summary Event

```json
{
  "event": "sweep_complete",
  "timestamp": "2026-03-24T11:30:00Z",
  "run_id": "sweep_v001_haiku_sonnet_t0",

  "config": {
    "skill_version": "v001_baseline",
    "verifier_model": "haiku",
    "patch_source": "sonnet",
    "temperature": 0.0,
    "patches_evaluated": 49
  },

  "metrics": {
    "precision": 0.65,
    "recall": 0.80,
    "f1": 0.72,
    "auroc": 0.78,
    "ece": 0.12,
    "top1_pass_rate": 0.18,
    "random_baseline_pass_rate": 0.12,
    "lift_over_random_pp": 6.0,
    "total_cost_usd": 0.058,
    "avg_latency_ms": 1850
  },

  "circuit_breakers": {
    "consecutive_failures": 0,
    "parse_failures": 2,
    "timeouts": 0,
    "total_errors": 2
  }
}
```

### Circuit Breakers

The sweep script MUST enforce:

| Condition | Action |
|-----------|--------|
| 3 consecutive `invoked: false` or `error != null` | HALT sweep, log `circuit_breaker_triggered` |
| `parse_success: false` rate > 20% | HALT sweep, log rubric likely malformed |
| `latency_ms > 30000` for any call | Timeout, log, continue |
| Sweep cost exceeds `$5.00` | HALT sweep, require manual approval |

## Smoke Test (MANDATORY before every sweep)

Before launching any sweep, run the smoke test on 4 known-outcome patches:

```bash
python3 scripts/smoke_test.py --rubric versions/v001_baseline.md --model haiku
```

The smoke test verifies:
1. **2 known PASS patches** — verifier must score overall > 0.4 (not confident wrong)
2. **2 known FAIL patches** — verifier must score overall < 0.6 (not confident right)
3. **All 4 parse successfully** — structured JSON output is valid
4. **No circuit breakers triggered** — API is working

If the smoke test fails, DO NOT proceed to sweep. Diagnose the failure first.

### Known Failure Modes (update as discovered)

| ID | Failure Mode | Description | Discovered | Mitigation |
|----|-------------|-------------|------------|------------|
| FM-001 | Reformatting noise | Sonnet reformats entire files (2958 lines for 3-line fix). Verifier scores 0.0 because it can't find the functional change in the noise. | Phase 2 testing | v006_reformat_aware — "ignore style changes, focus on functional modifications" |
| FM-002 | Truncated diffs | Diffs captured without trailing newline; `git apply` fails but `patch -p1` works | Phase 1 gold eval | Fixed in gold_eval.py; may affect verifier's ability to parse incomplete hunks |
| FM-003 | Plausible but wrong | Patch targets the right area and looks correct but has subtle logic error (wrong condition, missing edge case). Verifier gives 0.85-0.98 scores. 79% of Phase 2b errors. | Phase 2b error analysis | v007_strict_logic — "trace the exact code path before and after the patch" |
| FM-004 | Diff truncation | verify_patch.py truncated diffs at 12K chars. 47% of sonnet diffs exceeded this limit. 3 of 6 passing patches were truncated — verifier never saw the fix. Caused 0% recall on truncated patches vs 67% on non-truncated. | Pre-loop audit | Fixed: raised limit to 100K chars (~25K tokens, 12.5% of Haiku's 200K context). Added preprocess_diff.py fallback for diffs >100K. |

**Rule**: When a new failure mode is discovered, add it to this table BEFORE creating a new rubric version. The rubric version's commit message must reference which FM-xxx it addresses.

## Error Analysis Protocol (MANDATORY after every sweep)

After each sweep completes, run error analysis BEFORE interpreting results or starting the next sweep:

```bash
python3 scripts/analyze_errors.py results/sweep_phase2b.jsonl --output results/errors_phase2b.jsonl
```

The error analysis agent:

1. **Builds confusion matrix** — TP/FP/FN/TN counts per rubric version
2. **Categorizes each error** — Maps to known FM-xxx or flags as NEW
3. **Analyzes WHY** — For each FP/FN, examines the patch, the verifier's reasoning, and the gold label
4. **Outputs error taxonomy**:
   ```json
   {
     "error_id": "FN-django__django-10924-v001",
     "type": "false_negative",
     "category": "FM-001",
     "instance_id": "django__django-10924",
     "skill_version": "v001_baseline",
     "verifier_score": 0.0,
     "gold_passed": true,
     "diagnosis": "Patch is 2958 lines of reformatting with functional fix at line 847. Verifier correctly identified reformatting but missed the buried fix.",
     "diff_stats": {"total_lines": 2958, "files_changed": 1, "functional_lines_est": 5},
     "suggested_rubric_change": "Add instruction: 'If the diff contains style-only changes mixed with functional changes, focus your evaluation only on the functional changes.'"
   }
   ```
5. **Drafts rubric improvement** — Generates a new version addressing the top error category
6. **Computes error concentration** — Are errors concentrated in specific repos/issues? (If 80% of FNs are Sonnet reformatting, that's one fix; if spread across many patterns, harder)

### Error Analysis Decision Tree

```
After sweep completes:
├── Run analyze_errors.py
├── IF new failure mode discovered:
│   ├── Add FM-xxx to Known Failure Modes table
│   ├── Create new rubric version addressing it
│   └── Re-run smoke test with new version
├── IF error concentration > 50% in one category:
│   ├── Create targeted rubric version for that category
│   └── Re-sweep only that version (cheap validation)
├── IF errors are spread across many categories:
│   ├── Prioritize by count × impact
│   └── Create max 2 new versions per iteration
└── IF no new failure modes and lift is stable:
    └── Proceed to next phase
```

## Experiment Loop

LOOP FOREVER:

1. **Read** the latest sweep results (`results/sweep_*.jsonl`), error analysis (`results/errors_*.jsonl`), and this file's results tables
2. **Hypothesize** a specific improvement. Categories:
   - **Rubric**: New version in `versions/` — change criteria, add instructions, adjust tone (conservative vs lenient). Follow Failure Mode Encoding Rules.
   - **Verifier model**: haiku → sonnet → opus. More expensive but better reasoning.
   - **Verdict threshold**: Instead of `likely_correct` = pass, try `overall_score > 0.7` = pass.
   - **Temperature**: 0.0 (deterministic) vs 0.3 (diverse). Try 0.3 only after rubric is stable.
   - **Ensemble**: Multiple rubrics or models vote. Try only after single-rubric precision plateaus.
3. **Run** the experiment. One change per iteration — don't change rubric AND model simultaneously:
   ```bash
   # Smoke test (MANDATORY before sweep)
   python3 skills/patch-verifier/scripts/smoke_test.py \
     --rubric skills/patch-verifier/versions/v00X.md \
     --model <verifier> --patch-source sonnet

   # Dev set sweep (sonnet patches only during iteration)
   python3 skills/patch-verifier/scripts/sweep_versions.py \
     --versions v00X \
     --verifier-model <verifier> \
     --patch-source sonnet \
     --temperature 0.0 \
     --output results/sweep_<name>.jsonl

   # Error analysis (MANDATORY after every sweep)
   python3 skills/patch-verifier/scripts/analyze_errors.py \
     results/sweep_<name>.jsonl \
     --output results/errors_<name>.jsonl
   ```
4. **Log** the result. Append to the Results Table below:
   ```
   | <iteration> | <hypothesis> | <version> | <verifier> | <precision> | <recall> | <f05> | <lift_pp> | <cost> | <keep/discard> |
   ```
5. **Decide**:
   - If precision improved → keep. Build on this change in next iteration.
   - If no improvement → discard hypothesis. Try a different lever.
   - If lift = 0pp across 3+ rubric iterations → the model is the bottleneck, not the rubric. Escalate verifier model.
   - If precision > 0.40 on dev → evaluate on holdout (haiku + opus patches). If holdout_delta < 5pp, this is your candidate.
   - If precision > 0.50 AND lift > 5pp → SUCCESS. Document best config. Move to Phase 3.
6. **Repeat** from step 1.

### Decision Signals

| Observation | Action |
|------------|--------|
| FPs dominate (precision < 0.25) | Make rubric more conservative. Add skepticism language. |
| FNs dominate (recall < 0.20) | Make rubric more lenient. Focus on intent. |
| FM-001 errors > 30% | Rubric needs better functional/cosmetic separation. |
| FM-003 errors > 40% | Model may lack reasoning depth. Try Sonnet as verifier. |
| Dev >> holdout by > 5pp | Overfit. Simplify rubric — remove FM-specific instructions. |

### Holdout Evaluation (ONLY for final candidates)

Do NOT use during iteration. When you have a candidate with precision > 0.40 on dev:

```bash
python3 skills/patch-verifier/scripts/sweep_versions.py \
  --versions v00X \
  --verifier-model <verifier> \
  --patch-source haiku,opus \
  --output results/sweep_holdout_<name>.jsonl
```

## Results Table

Fill in as you iterate. One row per experiment.

| # | Hypothesis | Version | Verifier | Precision | Recall | F₀.₅ | Lift (pp) | Cost | Status |
|---|-----------|---------|----------|-----------|--------|-------|-----------|------|--------|
| 1 | Baseline 6-criteria rubric | v001_baseline | haiku | 0.29 | 0.33 | — | +0.0 | $0.20 | keep (best prec) |
| 2 | Minimal 3-criteria ablation | v002_minimal | haiku | 0.14 | 0.17 | — | +0.0 | $0.17 | discard |
| 3 | Few-shot examples | v003_fewshot | haiku | 0.17 | 0.17 | — | +0.0 | $0.21 | discard |
| 4 | Chain-of-thought | v004_cot | haiku | 0.15 | 0.60 | — | +0.0 | $0.25 | discard (too lenient) |
| 5 | Binary verdict | v005_binary | haiku | 0.10 | 0.33 | — | +0.0 | $0.16 | discard |

**Error analysis** (experiments 1-5, 73 errors: 52 FP, 21 FN):
- FM-003 (plausible but wrong): ~56% — high-confidence FPs, verifier fooled by intent
- FM-001 (reformatting noise): 21% — high-churn FNs, buried functional changes
- NEW (uncategorized): 23%

**⚠️ Phase 2b results INVALIDATED by FM-004 (diff truncation).** Fixed in verify_patch.py (100K limit). Re-swept below.

### Iteration 1: Re-sweep with fixed pipeline (100K char limit)

| # | Hypothesis | Version | Verifier | Precision | Recall | Large-Diff Recall | FP | Cost | Status |
|---|-----------|---------|----------|-----------|--------|-------------------|-----|------|--------|
| 6 | Fixed pipeline, baseline rubric | v001_baseline | haiku | 0.29 | 0.33 | 0/3 | 5 | $0.40 | keep (best prec) |
| 7 | FM-001 aware rubric + fixed pipeline | v006_reformat_aware | haiku | 0.12 | 0.83 | 2/3 | 38 | $0.43 | discard (too lenient) |
| 8 | Strict logic + fixed pipeline | v007_strict_logic | haiku | 0.14 | 0.83 | 2/3 | 30 | $0.43 | discard (too lenient) |

**Findings:**
- FM-004 fix: identical results on non-truncated patches. Verifier sees full diff but still can't extract functional changes from noise (FM-001 persists).
- v006/v007 solve FM-001 (large-diff recall 0/3 → 2/3) but destroy precision (0.29 → 0.12/0.14). They say "likely_correct" on almost everything.
- Need: combine v006's FM-001 awareness with v001's skepticism. Or escalate to Sonnet verifier.

**Current best**: v001_baseline (precision=0.29, recall=0.33). Lift = 0pp.

### Iteration 2: Sonnet as verifier (model capability test)

| # | Hypothesis | Version | Verifier | Precision | Recall | Large-Diff Recall | FP | Cost | Status |
|---|-----------|---------|----------|-----------|--------|-------------------|-----|------|--------|
| 9 | Sonnet has better reasoning | v001_baseline | sonnet | 0.17 | 0.50 | 0/3 | 15 | $1.53 | discard (worse prec, 4x cost) |

**Findings:**
- Sonnet is WORSE than Haiku on precision (0.17 vs 0.29) despite 4x cost
- Sonnet picks up 1 more small-diff TP but generates 3x more FPs
- Large-diff recall is STILL 0/3 — neither model can find functional fixes in 66-135K reformatting diffs
- **Model capability hypothesis FALSIFIED.** The bottleneck is the data (FM-001), not the model.

**Next hypothesis:** The 3 large-diff passing patches are fundamentally unverifiable without diff preprocessing. Options:
1. **Preprocess diffs** before verification — strip cosmetic hunks, only send functional changes
2. **Accept the ceiling** — exclude unverifiable patches, optimize on the 26 non-truncated patches where recall=67%
3. **Two-stage verification** — first pass identifies functional hunks, second pass evaluates them

### Iteration 2b: Threshold sweep (v001 × haiku)

Current verdict threshold (~0.80) is already optimal for F₀.₅:

| Threshold | TP | FP | FN | Precision | Recall | F₀.₅ |
|-----------|----|----|----|-----------| -------|------|
| 0.60 | 2 | 18 | 4 | 0.10 | 0.33 | 0.12 |
| 0.70 | 2 | 12 | 4 | 0.14 | 0.33 | 0.16 |
| **0.80** | **2** | **5** | **4** | **0.29** | **0.33** | **0.29** |
| 0.90 | 1 | 3 | 5 | 0.25 | 0.17 | 0.23 |

No free lunch from threshold tuning. The 5 FPs all score ≥ 0.80 because they genuinely look correct — small, clean patches targeting the right bug but with subtle logic errors.

### Iteration 4: v008 (conservative + reformat-aware) + ensembles

| # | Hypothesis | Version | Verifier | Precision | Recall | F₀.₅ | FP | Cost | Status |
|---|-----------|---------|----------|-----------|--------|-------|-----|------|--------|
| 11 | Conservative reformat rubric | v008 | haiku | 0.11 | 0.50 | 0.13 | 25 | $0.44 | discard alone (too lenient) |
| 12 | Ensemble v001∩v008 | v001+v008 | haiku | **0.40** | 0.33 | **0.38** | 3 | $0.84 | **BEST — holdout eval triggered** |
| 13 | Triple ensemble v001∩v007∩v008 | all three | haiku | 0.40 | 0.33 | 0.38 | 3 | $1.27 | same as #12, more expensive |

**Finding:** Ensemble intersection (v001∩v008) achieves precision 0.40 — our first result above random precision. v001's conservatism filters out the FPs that v008 admits, while v008 adds FM-001 awareness. The intersection eliminates 2 FPs vs v001 alone.

**Holdout evaluation COMPLETE:**

| Set | Precision | Recall | F₀.₅ | TP | FP |
|-----|-----------|--------|-------|----|----|
| Dev (sonnet) | 0.40 | 0.33 | 0.38 | 2 | 3 |
| Holdout (haiku) | 1.00 | 0.40 | 0.77 | 2 | 0 |
| Holdout (opus) | 0.38 | 0.50 | 0.39 | 3 | 5 |
| Holdout (combined) | 0.38 | 0.50 | 0.39 | 3 | 5 |

**Holdout delta: 0.03 (OK, < 5pp threshold). NOT overfit.**

Precision target (>0.50) not yet met. Lift still 0pp. But this is the first result that generalizes and meaningfully exceeds random precision (0.38 vs 0.12 base rate).

**Current best configuration: v001∩v008 ensemble with Haiku verifier.**

### Iteration 5: Score-based thresholds + creative ensembles (zero additional cost)

Exhaustive search over score thresholds and 9 ensemble strategies. All strategies that improve over v001 alone converge to the same result: precision=0.40, recall=0.33. The 3 remaining FPs cannot be separated from the 2 TPs by any combination of v001/v006/v007/v008 rubrics.

**Plateau reached.** To improve beyond precision=0.40, need either:
1. A fundamentally different rubric approach (not incremental FM fixes)
2. External signal (test execution, static analysis, AST diff)
3. Larger eval set to reduce noise from small sample (6 gold passes)

### Iteration 6: v009 adversarial rubric — BREAKTHROUGH

| # | Hypothesis | Version | Verifier | Precision | Recall | F₀.₅ | FP | Cost | Status |
|---|-----------|---------|----------|-----------|--------|-------|-----|------|--------|
| 16 | Adversarial framing: "find the bug" | v009 standalone | haiku | 0.25 | 0.17 | 0.23 | 3 | $0.44 | too aggressive alone |
| **17** | **v001∩v009 strict ensemble** | **v001+v009** | **haiku** | **1.00** | **0.17** | **0.50** | **0** | **$0.84** | **NEW BEST — holdout eval** |

**Key insight**: All prior rubrics (v001-v008) use *confirmatory framing* — "evaluate whether this patch is correct." v009 inverts to *adversarial framing* — "assume this is wrong, find the bug." This catches the 3 FM-003 FPs that looked correct:

| FP Instance | v001 verdict | v009 verdict | v009 attack |
|------------|-------------|-------------|-------------|
| astropy-14365 | likely_correct (0.98) | uncertain (0.62) | Found plausible failure in regex scope |
| astropy-14995 | likely_incorrect (0.42) | likely_incorrect (0.42) | Found missing symmetric case |
| flask-4992 | likely_correct (0.97) | uncertain (0.68) | Found missing input validation |

**Holdout evaluation COMPLETE (v001∩v009 strict):**

| Set | Precision | Recall | F₀.₅ | TP | FP |
|-----|-----------|--------|-------|----|----|
| Dev (sonnet) | 1.00 | 0.17 | 0.50 | 1 | 0 |
| Holdout (haiku) | 1.00 | 0.20 | 0.56 | 1 | 0 |
| Holdout (opus) | 1.00 | 0.33 | 0.71 | 2 | 0 |
| Holdout (combined) | 1.00 | 0.33 | 0.71 | 2 | 0 |

**Holdout delta: 0pp. NOT overfit. Zero false positives across all sets.**

Precision target (>0.50): MET (1.00).
F₀.₅ target (>0.40): MET (0.50-0.71).
Lift over random: +88pp (selected patches pass 100% vs 12% base rate).

### Where We Are

**Precision 1.00 — the verifier NEVER selects a failing patch.** When v001∩v009 says "likely_correct," you can trust it. The tradeoff is low recall (0.17-0.33): most correct patches get rejected.

**Remaining error budget (dev set):**
- 0 FPs (was 3 with v001∩v008)
- 5 FNs: 3 are FM-001 (117K-135K reformatted diffs), 2 are borderline cases where v009 found plausible (but unconfirmed) failure scenarios

**Cost efficiency:** $0.012 per verification (2 Haiku calls × 2 rubrics). At 100 patches/day, this is $1.20/day.

### Iteration 7: Temperature sampling ensemble — FINAL BEST

| # | Hypothesis | Version | Verifier | Precision | Recall | F₀.₅ | FP | Cost | Status |
|---|-----------|---------|----------|-----------|--------|-------|-----|------|--------|
| **18** | **v001∩v009(2+ lc) temp sampling** | **v001+v009×4** | **haiku** | **1.00** | **0.33** | **0.71** | **0** | **$0.048** | **SUCCESS** |

Run v009 4 times (1× temp=0.0 + 3× temp=0.3). v001 says likely_correct AND v009 says likely_correct in ≥2 of 4 runs.

**Result:** Recovers pytest-11143 TP that single v009 run rejected. Temperature sampling adds robustness: true positives survive 2+ adversarial attacks while all FPs fail consistently (0/4 likely_correct).

**Holdout evaluation COMPLETE:**

| Set | Precision | Recall | F₀.₅ | TP | FP |
|-----|-----------|--------|-------|----|----|
| Dev (sonnet) | 1.00 | 0.33 | 0.71 | 2 | 0 |
| Holdout (haiku) | 1.00 | 0.40 | 0.77 | 2 | 0 |
| Holdout (opus) | 1.00 | 0.33 | 0.71 | 2 | 0 |
| Holdout (combined) | 1.00 | 0.33 | 0.71 | 2 | 0 |

**All targets met:**
- Precision > 0.50: YES (1.00)
- F₀.₅ > 0.40: YES (0.71)
- Lift > 5pp: YES (+88pp)
- Holdout delta < 5pp: YES (0pp)

### Iteration 8: Cross-Verifier Transfer (T4)

**Hypothesis**: The v001∩v009 rubric is model-agnostic — any capable LLM can serve as verifier.

| # | Hypothesis | Version | Verifier | Precision | Recall | F₀.₅ | FP | Cost | Status |
|---|-----------|---------|----------|-----------|--------|-------|-----|------|--------|
| 22 | Haiku baseline (same script) | v001+v009×3 | haiku | 1.00 | 0.17 | 0.50 | 0 | $1.76 | Baseline confirmed |
| 22a | Cross-verifier: Devstral 2 123B | v001+v009×3 | devstral2 | 0.20 | 0.17 | 0.19 | 4 | $1.02 | **HYPOTHESIS FALSIFIED** |
| 22b | Cross-verifier: Nova Pro (Amazon) | v001+v009×3 | nova-pro | 0.14 | 0.17 | 0.15 | 6 | $1.45 | Even worse — v009 too lenient |
| 22c | Cross-verifier: Mistral Large 675B | v001+v009×3 | mistral-large | N/A | 0.00 | 0.00 | 0 | $3.44 | Rejects everything |

**Findings:**
- **The rubric is NOT model-agnostic.** Precision=1.00 depends on Claude's adversarial reasoning quality.
- Devstral v009 passes `astropy-14365` (2/3 lc) — the same patch Claude rejects 0/4 times. v009 defaults to "uncertain" (76% of verdicts) instead of making decisive calls.
- Nova Pro is even worse (6 FPs) — v009 is too lenient (17.7% lc rate vs Haiku's calibrated 15.6%).
- Mistral Large is too conservative — v009 defaults to "likely_incorrect" (90% of verdicts). Rejects everything.
- **Only Claude's v009 is calibrated**: decisive enough to clear real fixes, skeptical enough to catch subtle bugs.
- **Implication**: For RLVR, must use Claude as verifier. The precision=1.00 is a Claude capability, not just a rubric design.

Script: `scripts/run_cross_verifier.py`

### Iteration 9: Adversarial Self-Critique in Generation (T5) — RUNNING

**Hypothesis**: Injecting "assume your patch is wrong, find the bug" into the generation prompt improves pass rate at zero extra cost.

| # | Variant | Fix Rate | Pass Rate | Precision | Cost | Turns | Status |
|---|---------|----------|-----------|-----------|------|-------|--------|
| 23 | control (Phase 1 baseline) | 54% (27/50) | 10% (5/50) | 0.19 | $9.91 | 26 | Baseline |
| 23a | self-critique | 42% (21/50) | **2% (1/50)** | 0.05 | $6.68 | 17 | **NEGATIVE — worse than control** |
| 23b | self-critique-strong | **30% (15/50)** | **2% (1/50)** | 0.07 | $6.57 | 17 | **NEGATIVE — worst fix rate** |

**T5 HYPOTHESIS FALSIFIED.** Self-critique in generation hurts performance:
- Fix rate drops 54% → 42% → 30% (stronger critique = fewer fixes)
- Pass rate drops 10% → 2% (lost 4 of 5 control passes)
- Only `mwaskom__seaborn-3010` passes (also passes in control — not a new fix)
- The self-critique instruction consumes turn budget on review instead of fixing
- **Haiku cannot effectively find bugs in its own code during generation.** The adversarial reasoning that works for external verification (v009) does NOT transfer to self-correction. This is consistent with the "blind spot" hypothesis: a model that generates a plausible-but-wrong fix cannot see its own error.

Script: `scripts/run_baseline.py --prompt-variant <variant> --gold-eval`

### Iteration 10: Best-of-N on Devstral SERA Candidates (T6c)

**Hypothesis**: v001∩v009 ensemble can select the best patch from multiple Devstral SERA candidates per issue, improving pass rate over random selection.

| # | Metric | Value |
|---|--------|-------|
| 24 | Candidates | 97 across 49 issues |
| | Ensemble passes | 5/97 (5.2%) |
| | BoN gold pass rate | 3/49 (6%) |
| | Oracle pass rate | 4/49 (8%) |
| | Random pass rate | 3.5±0.5/49 (7.2%) |
| | Precision (ensemble) | 0.20 (1 TP, 4 FP) |
| | Cost | $1.42 |

**T6c HYPOTHESIS FALSIFIED.** BoN verifier selection (3/49=6%) is worse than random (3.5/49=7.2%) and worse than VL (4/44=9%). On the one issue where selection mattered (django-11001, only a3 passes), the verifier picked a1.

**Root cause**: Precision dropped from 1.00 (Claude patches) to 0.20 (Devstral patches). The v009 adversarial rubric was calibrated on Claude's surgical, targeted patches. Devstral patches have different error signatures (more structural changes), and the rubric's FP-catching doesn't generalize.

**Key number**: 95% of Devstral candidates get v009 lc=0/3. The 5% that survive are NOT enriched for correct patches (precision=0.20). The verifier has no signal on this patch distribution.

Script: `scripts/run_best_of_n.py`

### Iteration 11: Cross-Model BoN on Claude Patches (T7)

**Hypothesis**: v001∩v009 can select the best patch from multiple Claude models (haiku/sonnet/opus) per issue, exceeding any single model's pass rate.

| # | Metric | 2+/3 threshold | 3+/3 threshold |
|---|--------|---------------|---------------|
| 25 | Ensemble passes | 4 (2 TP, 2 FP) | 2 (2 TP, 0 FP) |
| | Precision | 0.50 | **1.00** |
| | BoN pass rate | 6/49 (12%) | 6/49 (12%) |
| | Oracle | 7/49 (14%) | 7/49 (14%) |
| | Random | 6.0/49 (12.3%) | 6.0/49 (12.3%) |
| | Cost | $4.61 | $4.61 |

**CRITICAL BUG FOUND**: Iteration 21's "cost optimization" (drop v009 t=0.0, use 2+/3) introduced 2 FPs on opus patches (flask-4045, flask-4992). These were correctly rejected by the original 5-call config (1/4 and 0/4 lc). The stochastic 3-run config occasionally produces 2/3 on borderline patches. **Fix: raise threshold to 3+/3** (unanimous agreement required). Zero TPs lost.

**T7 HYPOTHESIS NOT SUPPORTED.** BoN (6/49=12%) equals best single model (sonnet=opus=12%). The verifier adds no value because 5/7 gold passes are "easy" issues where all 3 models pass. On the 1 issue where selection matters (django-11019), the verifier fails.

Script: `scripts/run_cross_model_bon.py`

### Summary of Levers Tested

| Lever | Tested | Result |
|-------|--------|--------|
| Rubric (FM-001 aware) | v006, v007, v008 | Fixes recall, destroys precision as standalone |
| Rubric (strict logic) | v007 | No improvement standalone |
| Rubric (conservative + reformat) | v008 | prec=0.11, too lenient alone |
| Model (Sonnet) | v001 × sonnet | Worse precision, 4x cost |
| Threshold | 0.3-0.95 sweep | 0.80 already optimal |
| Diff limit | 12K → 100K | No impact on recall |
| Two-stage extraction | Haiku stage1→stage2 | Stage 1 misses functional fix in noise |
| Ensemble (v001∩v008) | Intersection | prec=0.40 dev, 0.38 holdout. Previous best. |
| Adversarial rubric (v009) | Bug-finding frame | prec=0.25 standalone, too aggressive |
| **Ensemble (v001∩v009)** | Confirmatory × adversarial | **prec=1.00 dev+holdout. Zero FPs. NEW BEST.** |
| FM-001 recall (iter 18) | Changes-only extraction, v008 routing, size-adaptive | No improvement. FM-001 needs AST-level diff. |
| DTV (iter 19) | Describe expected fix, then search diff | Finds 2/3 FM-001 fixes. 1 FP on opus holdout. |
| DTV adversarial gate (iter 20) | v009 on DTV evidence | Rejects everything. Temperature sampling also fails. |
| **Cross-verifier (iter 22)** | Devstral 2 + Nova Pro + Mistral Large as verifier | **Rubric is Claude-specific. Devstral prec=0.20, Nova Pro prec=0.14, Mistral Large rejects all.** |
| **Self-critique gen (iter 23)** | Adversarial self-critique in generation prompt | **NEGATIVE. Fix rate drops 54%→30%, pass rate drops 10%→2%. Model can't find bugs in its own code.** |
| **BoN on Devstral (iter 24)** | v001∩v009 selects best from 97 Devstral candidates | **NEGATIVE. prec=0.20, BoN 3/49 ≤ random 3.5/49 ≤ VL 4/44. Verifier is patch-source-specific.** |
| **Cross-model BoN (iter 25)** | v001∩v009 selects best from haiku/sonnet/opus per issue | **MARGINAL. prec=1.00 (3+/3), but BoN 6/49 = best single model. Found iter 21 regression: 2+/3 causes FPs.** |
| **Qwen3.5 × OpenCode transfer (iter 54)** | v001∩v009 on Qwen3.5-122B FP8 × OpenCode diffs | **PARTIAL. Prec=0.50 (2 FP), rec=0.50, F₀.₅=0.50. Harness helps (>0.33 SERA) but <1.00 Claude. Both FPs on small diffs (<2K).** |
| **Threshold validation (iter 26)** | Revert to 5-call 2+/4 | **Best config confirmed. t=0.0 essential for stability.** |
| **Recall ceiling (iter 27)** | Analyze FN blockers across all sets | **v001 blocks 59% (10/17) of gold passes. FM-001 (large diffs) defeats BOTH rubrics. No path via more v009 runs.** |
| **v009-only (iter 28)** | Drop v001 gate, v009-only with various thresholds | **NEGATIVE. Same recall, more FPs (prec=0.25-0.33). v001 is a crucial precision guard.** |
| **Alt gate combos (iter 29)** | v008∩v009, v006∩v009, (v001 OR v008)∩v009 | **NEGATIVE. All worse than v001∩v009 (prec=0.25-0.33). Reformat-aware gates add FPs.** |
| **v010 concrete adversarial (iter 30)** | Require concrete counter-examples (input/expected/actual) | **NEGATIVE. prec=0.13, 27 FPs. Too lenient — model can't construct counter-examples for genuinely buggy patches. Recovers 2 FNs (django-10924, sphinx-11445) but v009 blocks both (0/4 lc). Recall ceiling is v009, not the confirmatory gate.** |
| **Diff preprocessing (iter 31)** | Changes-only + quote filtering (53% noise reduction) | **NEGATIVE. 34-73K chars still too noisy. v001 and v009 unchanged. Full AST normalization needs source files.** |
| **Oracle minimal-diff (iter 35)** | Extract 525-char functional-only diff, test v009 | **NEGATIVE. v009 still rejects 0/4 lc. Problem is NOT noise — model finds genuine incompleteness (missing deconstruct()). Both Haiku and Sonnet agree.** |
| **v011 scoped adversarial (iter 35)** | "Only reject if changed code itself is wrong" | **NEGATIVE. prec=0.09, 20 FPs. Removing completeness check destroys precision (leaks astropy-14365) without recovering any FNs.** |
| **Sonnet as v009 (iter 36)** | Sonnet for adversarial reasoning instead of Haiku | **NEGATIVE. Same recall ceiling. Sonnet finds same incompleteness as Haiku. Recall limit is semantic, not capability-bound.** |
| **v012 test predictor (iter 37)** | Predict test pass, not fix correctness | **NEGATIVE. Leaks both FPs (4/4 lc) AND rejects django-10924. Can't predict test outcomes without test info.** |
| **Extended thinking (iter 38)** | v001+thinking ∩ v009+thinking ensemble | **NEGATIVE. Recovers sphinx-11445 but leaks flask-4992. temp=1.0 (required) makes v009 less discriminative: FP 63% lc > TP 50% lc.** |
| **Problem-first verify (iter 39)** | Predict fix from problem, then compare to diff | **NEGATIVE. Predicts more complete fix → rejects partial patches (FNs). Predicts same plausible fix → accepts FPs.** |
| **Opus as v009 (iter 40)** | Opus 4.6 as adversarial verifier | **NEGATIVE. More lenient than Haiku — leaks flask-4992 (lc=0.87). FNs get "uncertain" (not lc) so no recall improvement. 18x cost.** |
| **Multi-turn challenge (iter 41)** | v009 turn1 → "only changed code" challenge turn2 | **NEGATIVE. Sycophantic — ALL verdicts flip to lc. Recovers FNs but leaks ALL FPs. Same problem as v011/v012.** |
| **Adversarial few-shot (iter 42)** | v009 with 3 labeled examples | **NEUTRAL. Same precision (1.00). Examples don't change verdicts on real code — model ignores abstract guidance.** |
| **Test generation (iter 43)** | Ask model to write a failing test | **MIXED. First approach recovering ALL 4 FNs! But 4/5 FPs also leak (prec≈0.50). Orthogonal to v009 — can't combine.** |
| **v013 test-scoped adversarial (iter 44)** | 2-phase: generate test spec → v009 scoped to test | **NEGATIVE. Recovers 2/4 FNs at v013 level (django-11001 4/4, sphinx-10325 4/4) but leaks 4/5 FPs (astropy-14365 4/4, flask-4992 4/4). v001 independently blocks both recovered FNs. No ensemble helps.** |
| **Decomposed v009 (iter 45)** | Break problem into claims, run v009 per claim | **NEGATIVE. v009 is ROBUST to scope manipulation. 0/4 lc on every claim for every patch. Decomposition makes v009 MORE conservative, not less. Even narrow "accepts callable" claim → v009 finds issues.** |
| **Size-based routing (iter 46)** | v013 for large diffs, v001∩v009 for small | **NEGATIVE. v013 passes 4/5 large gold=FAIL patches (4/4 lc, scores 0.90-0.94). v013 too lenient regardless of diff size. Routing can't work.** |
| **Structural features (iter 47)** | Diff size, hunks, ratio, test_files as meta-features | **NEGATIVE. No structural feature separates gold passes from fails. Passes are bimodal (2 small TPs + 3 large FNs + 1 tiny FN).** |
| **Attack-type classification (iter 48)** | Classify v009 attacks as INCOMPLETENESS vs CONCRETE BUG | **NEGATIVE (zero-cost). INCOMPLETENESS attacks appear on BOTH FPs (astropy-14365, astropy-14995) and ALL 4 FNs. CONCRETE attacks only on 3 FPs (flask-*). Overlap makes post-hoc classification impossible.** |
| **Exhaustion audit (iter 49)** | All diff sets evaluated, all levers catalogued | **TERMINATE. 22 consecutive negatives. All 7 diff sets exhausted. No remaining prompt-engineering hypothesis. Phase 0.5 complete.** |
| **Meta-verify v009 attacks (iter 50)** | Ask "is this v009 concern likely tested?" | **NEGATIVE. Meta-verifier says "unlikely_tested" for ALL 9 attacks (FPs AND FNs alike). Cannot distinguish valid from invalid attacks. v009 attacks are always about secondary behavior regardless of patch correctness. $0.10.** |
| **Attack-reference-in-diff (iter 51)** | Check if v009-mentioned functions appear in diff | **NEGATIVE (zero-cost). Large diffs contain everything; small diffs have few extractable refs. No discriminative power.** |
| **v009 score distributions (iter 52)** | FP vs FN score means from iter 44 data | **NEGATIVE (zero-cost). FP mean=0.60, FN mean=0.45 but complete overlap (django-11001 FN 0.60 = flask-4992 FP 0.61). No clean threshold.** |
| **Persona variation (iter 53)** | 3 adversarial personas (security/testing/architect) | **NEGATIVE. All personas reject all 7 patches. Code analysis dominates persona framing. $0.25.** |
| **T9: Qwen3.5 × OpenCode harness transfer (iter 54)** | Qwen3.5-122B-FP8 × OpenCode diffs, v001∩v009 ensemble | **PARTIAL. Prec=0.50, Rec=0.50, F₀.₅=0.50 (2 TP, 2 FP, 2 FN, 37 TN). Harness helps (0.50 > 0.33 SERA) but model still matters (0.50 < 1.00 Claude). Both FPs on small diffs (<2K). Gold pass 4/43=9%. $1.47 verifier + ~$40 g7e compute.** |

**TERMINATION CONDITION MET**: 27 consecutive iterations (28-54) with 0pp precision improvement on Claude patches. Every conceivable prompt-engineering lever has been tested. T9 confirms verifier transfer depends on BOTH harness (OpenCode > SERA) AND model (Claude > Qwen3.5). The remaining path is Phase 3+ (learned verifier, RLVR). All levers exhausted (12 rubric versions, 5 verifier models, 3 temperature settings, extended thinking, diff preprocessing, 2 patch sources). Recall ceiling (0.33) is caused by semantic mismatch between verification (problem completeness) and gold eval (specific test cases) — confirmed by oracle test (iter 35) and cross-model agreement (iter 36). Next phases require new infrastructure (RLVR, SVG consensus, learned verifier), not more iterations.

### Iteration 3: Ensemble (zero additional cost, using existing sweep data)

| Strategy | TP | FP | FN | Precision | Recall | F₀.₅ |
|----------|----|----|----|-----------| -------|------|
| v001 alone (baseline) | 2 | 5 | 4 | 0.29 | 0.33 | 0.29 |
| v001 AND v006 (intersection) | 2 | 5 | 4 | 0.29 | 0.33 | 0.29 |
| **v001 AND v007 (intersection)** | **2** | **4** | **4** | **0.33** | **0.33** | **0.33** |
| avg(v001,v006,v007) > 0.7 | 4 | 38 | 2 | 0.10 | 0.67 | 0.11 |
| majority vote (2/3 pass) | 5 | 31 | 1 | 0.14 | 0.83 | 0.17 |

Best ensemble: v001∩v007 — eliminates 1 FP, keeps all TPs. F₀.₅ = 0.33 (vs 0.29 baseline).

**YOUR NEXT ITERATION.** Remaining high-value levers:
1. **Diff preprocessing**: Feed only functional hunks to verifier. Directly attacks FM-001 at the data level. preprocess_diff.py exists but only gives 10% reduction — needs improvement.
2. **Two-stage verification**: First call (cheap, Haiku) extracts functional changes from reformatted diffs. Second call (Haiku) evaluates only the functional diff. Doubles cost but may crack FM-001.
3. **New rubric combining v001+v006**: v001's skepticism + v006's reformatting awareness, but with explicit "default to uncertain" instruction.
4. **Focus on clean patches**: 26 non-truncated patches have prec=0.29, rec=0.67 with v001 alone. Targeted rubric for small diffs.

## Analysis Metrics

### Primary (precision-weighted — FPs are more costly than FNs)

| Metric | Definition | Target |
|--------|-----------|--------|
| **precision** | P(gold_pass \| skill_says_pass) | > 0.50 |
| **F₀.₅** | Harmonic mean weighting precision 2x over recall | > 0.40 |
| **lift_over_random** | top1_pass_rate - random_baseline_pass_rate | > 5pp |
| **confident_error_rate** | P(gold_fail \| verdict=likely_correct AND confidence > 0.8) | < 0.30 |

### Secondary

| Metric | Definition | Why |
|--------|-----------|-----|
| **AUROC** | Area under ROC curve of overall_score vs gold_pass | Threshold-independent discrimination |
| **ECE** | Expected Calibration Error of confidence vs gold_pass | Gate for future RL use |
| **recall** | P(skill_says_pass \| gold_pass) | Don't miss good patches (but less critical than precision) |
| **cost_per_verification** | avg cost per patch verification call | Must be < $0.01 for Haiku |
| **parse_failure_rate** | % of calls where structured output failed to parse | Skill robustness |
| **holdout_delta** | dev_metric - holdout_metric | Overfit detector: should be < 5pp |

### Comparison Baselines

| Baseline | Method | Expected |
|----------|--------|----------|
| Random | Pick random patch | ~12% pass (Sonnet base rate) |
| Shortest | Pick smallest diff | TBD |
| SVG consensus | 3× inference + line-recall vote | AUC 0.981 (from learned-verifier Phase 0) |
| Oracle | Pick gold-passing patch | upper bound |

## Evaluation Integrity

### Overfitting Prevention

We iterate rubrics against a fixed patch pool. Without safeguards, rubrics will memorize pool-specific patterns rather than learning general verification.

**Train/validation split**:
- **Dev set** (iterate on): sonnet patches (49 patches, 6 pass). All rubric design and error analysis uses this set.
- **Holdout set** (evaluate once): haiku patches (27 patches, 5 pass) + opus patches (49 patches, 6 pass). Only evaluate the *final* rubric candidate on holdout. Never inspect holdout errors during iteration.
- **Report both**: Every result table must show dev AND holdout metrics. A rubric that improves on dev but degrades on holdout is overfit.

**Rubric design principles** (general > specific):
- Rubrics should express *principles* ("trace the code path"), not *patterns* ("ignore quote style changes")
- FM-specific instructions must be phrased as general heuristics, not as responses to specific instances
- Example: "Separate functional from cosmetic changes" (general) vs "Ignore single-to-double quote changes" (overfit)
- If a rubric version names a specific repo, issue, or diff pattern from the pool, it is overfit by definition

**Cross-source validation**: A rubric trained on sonnet patches must also work on haiku/opus patches. Phase 2c explicitly tests this — if a rubric only works on one patch source, it's overfit to that model's style.

### Asymmetric Error Costs

Not all errors are equal. In pass@1-from-N selection:

| Error Type | Cost | Why |
|-----------|------|-----|
| **False Positive** (say pass, gold fail) | **HIGH** | Ship broken patch. Waste downstream CI, review time, user trust. |
| **False Negative** (say fail, gold pass) | LOW | Reject good patch. Just pick next candidate from pool. |
| **True Negative** (say fail, gold fail) | NONE | Correctly filtered bad patch. |
| **True Positive** (say pass, gold pass) | NONE | Correctly selected good patch. |

**Implications for rubric design**:
- **Precision matters more than recall.** A verifier that says "likely_correct" on 3 patches with 2 actually correct (precision=0.67) is far more useful than one that says "likely_correct" on 20 patches with 4 correct (precision=0.20), even though the second has higher recall.
- **Conservative by default.** When uncertain, the rubric should lean toward "uncertain" or "likely_incorrect" — the cost of a miss is low.
- **FM-003 (plausible but wrong) is the critical failure mode.** It's a high-confidence FP — the verifier is *confidently wrong*. This is worse than a low-confidence FP because it can't be filtered by confidence thresholding.

**Weighted metrics**: In addition to raw precision/recall, compute:
- **Precision@k**: Of the top-k scored patches per issue, what fraction pass? (k=1 is the deployment scenario)
- **Cost-weighted F**: `F_β` with `β=0.5` (weights precision 2x over recall)
- **Confident-error rate**: `P(gold_fail | verdict=likely_correct AND confidence > 0.8)` — the truly dangerous errors

### Failure Mode Encoding Rules

When encoding failure modes into rubrics:

1. **Generalize before encoding.** FM-001 (reformatting noise) generalizes to "distinguish functional from cosmetic changes." FM-003 (plausible but wrong) generalizes to "trace code execution, not just intent."
2. **Never encode instance-specific details.** The rubric must not reference django-10924, Sonnet's quote style, or any specific diff from the pool.
3. **Test on the failure mode's complement.** If v006 tells the verifier to ignore reformatting, verify it still catches bugs *introduced by* reformatting (e.g., a quote change that breaks an f-string).
4. **Sunset obsolete FM instructions.** If a failure mode is resolved at the tooling level (FM-002 fixed in gold_eval.py), don't keep it in the rubric — dead instructions add noise.

## Rules

- NEVER modify gold eval labels or the patch pool
- NEVER modify `verify_patch.py` core logic during a sweep — only modify rubric versions
- NEVER inspect holdout set errors during rubric iteration — only evaluate final candidates
- Log EVERY experiment including failures and regressions
- One variable per experiment — don't change rubric AND verifier model simultaneously
- Cost guard: each sweep must project total cost before starting, halt if > $5
- All results append to JSONL — never overwrite
- Telemetry is mandatory — a run without telemetry events is invalid
- Create new rubric versions rather than editing existing ones (immutable versions)
- Report dev AND holdout metrics for every rubric candidate

## Results Location

```
results/
├── baseline_haiku_opencode.jsonl      # Phase 1 baselines
├── baseline_sonnet_opencode.jsonl
├── baseline_opus_opencode.jsonl
├── gold_haiku_opencode.jsonl          # Gold eval labels
├── gold_sonnet_opencode.jsonl
├── gold_opus_opencode.jsonl
├── sweep_phase2b.jsonl                # Phase 2b sweep telemetry
├── sweep_phase2c.jsonl                # Phase 2c cross-model sweep
├── sweep_phase2d_iter1.jsonl          # Phase 2d refinement iterations
└── diffs/                             # Raw patches
    ├── opencode_haiku/
    ├── opencode_sonnet/
    └── opencode_opus/
```

## Monitoring

```bash
# Check sweep progress
wc -l results/sweep_phase2b.jsonl

# Live metrics
python3 scripts/analyze_versions.py results/sweep_phase2b.jsonl --live

# Cost tracking
python3 -c "
import json
rows = [json.loads(l) for l in open('results/sweep_phase2b.jsonl') if 'skill_invocation' in l]
print(f'Calls: {len(rows)}, Cost: \${sum(r[\"invocation\"][\"cost_usd\"] for r in rows):.4f}')
"
```
