---
name: patch-verifier
description: Verify whether a coding agent's patch correctly fixes a bug. Scores patches on structured criteria (problem alignment, minimality, logic correctness, scope, completeness). Use when selecting the best patch from N candidates or filtering low-quality fixes.
---

# Patch Verifier

## Overview

Scores agent-generated patches against a structured rubric to predict whether the patch will pass gold tests. Designed for post-generation verification — the agent has already produced a diff, and this skill evaluates it.

## Usage

Given a problem statement and a patch diff, load the active rubric version from `versions/` and score the patch.

### Input

1. **Problem statement**: The bug report or issue description
2. **Patch diff**: The `git diff` output from the agent
3. **Rubric version**: Which `versions/v*.md` to use (default: latest)

### Output

Structured JSON with per-criterion scores (0.0-1.0), overall score, confidence, verdict, and reasoning.

### Scripts

- `scripts/verify_patch.py` — Core verification: calls Claude API with rubric + patch → structured score
- `scripts/sweep_versions.py` — Batch evaluation: all versions × all patches → telemetry JSONL
- `scripts/analyze_versions.py` — Compare versions on precision/recall/ECE/AUROC
- `scripts/analyze_errors.py` — Post-sweep error analysis: FP/FN categorization, failure mode matching, recommendations
- `scripts/smoke_test.py` — Pre-sweep validation: 4 known-outcome patches, directional correctness checks
- `scripts/telemetry.py` — Telemetry event emitter and circuit breaker logic
- `scripts/preprocess_diff.py` — Diff preprocessor: strips cosmetic hunks for oversized diffs
- `scripts/run_iteration.py` — Utility: smoke test → sweep → error analysis → holdout eval
- `scripts/verify_patch_ensemble.py` — Production ensemble: v001∩v009(2+ lc), precision=1.00
- `scripts/describe_then_verify.py` — Two-stage for FM-001: describe expected fix → search large diff (experimental)

### Rubric Versions

| Version | Description | Addresses |
|---------|-------------|-----------|
| v001_baseline | 6-criteria rubric | — |
| v002_minimal | 3-criteria ablation | — |
| v003_fewshot | Baseline + 2 labeled examples | — |
| v004_cot | Chain-of-thought scoring | — |
| v005_binary | Binary yes/no, no rubric | — |
| v006_reformat_aware | Reformatting-aware rubric | FM-001 |
| v007_strict_logic | Strict logic tracing, conservative | FM-003 |
| v008_conservative_reformat | Conservative + reformatting-aware | FM-001 + FM-003 |
| v009_adversarial | Adversarial bug-finding rubric | FM-003 (breakthrough) |

### Known Failure Modes

| ID | Name | Description |
|----|------|-------------|
| FM-001 | Reformatting noise | Large diffs dominated by style changes hiding small functional fixes. Addressed by v006. |
| FM-002 | Truncated diffs | Diffs missing trailing newline. Fixed in gold_eval.py. |
| FM-003 | Plausible but wrong | Patch targets right area, looks correct, but has subtle logic error. Verifier gives high scores (0.85-0.98). 79% of Phase 2b errors. |
| FM-004 | Diff truncation | 47% of diffs exceeded 12K char limit. 3/6 passing patches invisible to verifier. Fixed: 100K limit + preprocess_diff.py. |

### References

- `references/rubric_design.md` — Design rationale and iteration history
- `versions/` — Immutable rubric versions (never edit, only create new)
