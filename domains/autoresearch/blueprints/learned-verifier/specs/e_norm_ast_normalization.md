# E_norm: AST Patch Normalization for Cross-Model Verifier Transfer

**Date**: 2026-04-26
**Status**: Ready to run
**Depends on**: verifier-reward blueprint (diffs, v009 rubric, gold labels)
**Compute**: g7e.24xlarge (Docker repo cache already warm from gold_eval runs)
**Estimated cost**: ~$5 API (v009 re-evaluation) + ~$5 g7e compute (1 hour)

---

## Motivation

v009 adversarial rubric precision drops sharply across scaffolds:

| Patch Source | Harness | v009 Precision | Median Diff Size |
|-------------|---------|---------------|-----------------|
| Claude Sonnet | OpenCode | **1.00** | 11.7K |
| Qwen3.5 122B | OpenCode | **0.50** | 8.6K |
| Devstral 24B | SERA | **0.20** | 1.5K |

E6 (cross-model behavioral RF transfer) showed zero-shot AUC=0.363 — features carry signal but decision boundaries are model-specific. This experiment tests whether the *rubric-based* verifier (v009) has the same problem for the same reason (scaffold-specific formatting) or a different one.

### Prior art (iter 31 — INCONCLUSIVE)

Iter 31 tested heuristic cosmetic filtering on 3 FM-001 diffs:
- Stripped context lines + quote-only change pairs
- 46-53% size reduction
- **Verdicts unchanged** — remaining noise includes line wrapping, import reordering, string format changes

Conclusion from iter 31: *"Full AST normalization needs original source files (not just diffs)."*

This experiment solves that blocker by cloning repos at base_commit.

### Inspiration

- **Shopify Engineering** (2026-04-22): Switching from JSON DSL to Python DSL improved syntactic correctness by +22pp and semantic correctness by +13pp — representation change alone, no new data, no new model.
- **Agentless** (arXiv:2407.01489): AST-normalized majority voting via `ast.parse()` → `ast.unparse()` + comment stripping + canonical diff.
- **R4P** (arXiv:2510.22775): Verification accuracy positively correlates with diff size — SERA's minimal diffs (1.5K) may simply lack surface area for adversarial verification.

---

## Hypothesis

AST-level patch normalization closes a significant portion of the cross-scaffold v009 precision gap by:
1. Eliminating scaffold-specific formatting artifacts (whitespace, comment style, import ordering)
2. Collapsing semantically identical patches that differ textually
3. Potentially increasing effective diff size for minimal patches (via structural context augmentation)

---

## Design

### Step 0: Scaffold Discriminability (BEFORE normalization)

Measure whether surface features distinguish scaffolds — if a trivial classifier can tell OpenCode from SERA patches, those distinguishing features are scaffold artifacts, not correctness signals.

```bash
python3 scripts/normalize_patches_v2.py --discriminability \
  ../verifier-reward/results/diffs/opencode_sonnet \
  ../verifier-reward/results/diffs/qwen35_opencode \
  ../verifier-reward/results/diffs/devstral_sera_verifier_loop
```

### Step 1: Full AST Normalization

For each patch:
1. Clone repo at `base_commit` (reuse `swebench-repo-cache` Docker volume)
2. Read original Python source files touched by the diff
3. Apply patch, read modified source files
4. AST-normalize both: `strip_comments_and_docstrings()` → `ast.parse()` → `ast.unparse()`
5. Generate canonical diff between normalized before/after

```bash
# On g7e (Docker volume warm):
for model in sonnet haiku opus; do
  python3 scripts/normalize_patches_v2.py --model $model
done
python3 scripts/normalize_patches_v2.py \
  --diff-dir ../verifier-reward/results/diffs/qwen35_opencode \
  --output-dir results/diffs_normalized/qwen35_opencode
python3 scripts/normalize_patches_v2.py \
  --diff-dir ../verifier-reward/results/diffs/devstral_sera_verifier_loop \
  --output-dir results/diffs_normalized/devstral_sera_verifier_loop
```

**Alternative (no Docker):** Use `--local-repos /tmp/swebench-repos` to clone repos locally (~5GB disk).

### Step 2: Scaffold Discriminability (AFTER normalization)

Re-run discriminability test on normalized diffs. If max_ratio drops, normalization is removing scaffold signal.

### Step 3: Re-run v009 on Normalized Patches

Uses verifier-reward sweep infrastructure:

```bash
cd ../verifier-reward

# Control: Claude/OpenCode (MUST stay at precision ≥ 0.90)
python3 scripts/sweep_versions.py --versions v009 --verifier-model haiku \
  --patch-source sonnet \
  --diff-dir ../learned-verifier/results/diffs_normalized/opencode_sonnet \
  --temperature 0.0 --output results/sweep_enorm_control.jsonl

# Test A: Qwen3.5/OpenCode (currently 0.50)
python3 scripts/sweep_versions.py --versions v009 --verifier-model haiku \
  --patch-source qwen35 \
  --diff-dir ../learned-verifier/results/diffs_normalized/qwen35_opencode \
  --temperature 0.0 --output results/sweep_enorm_qwen35.jsonl

# Test B: Devstral/SERA (currently 0.20)
python3 scripts/sweep_versions.py --versions v009 --verifier-model haiku \
  --patch-source devstral \
  --diff-dir ../learned-verifier/results/diffs_normalized/devstral_sera_verifier_loop \
  --temperature 0.0 --output results/sweep_enorm_devstral.jsonl
```

### Step 4: Full v009 4/4 Ensemble on Normalized Patches

If Step 3 (single v009 run) shows signal, re-run full ensemble (1×t=0.0 + 3×t=0.3, unanimous) for precise comparison against baseline.

---

## Controls

| Control | Purpose | Failure mode it catches |
|---------|---------|------------------------|
| **A: Claude/OpenCode normalized** | Does normalization hurt in-distribution precision? | Normalization strips cues v009 needs for adversarial reasoning |
| **B: Original diffs (existing results)** | Baseline comparison | — |
| **C: Normalization stats** | What % of files actually normalize? What's the compression ratio? | Pipeline silently failing (AST parse errors, empty output) |

---

## Success Criteria

| Outcome | Interpretation | Next step |
|---------|---------------|-----------|
| Control precision drops < 0.90 | Normalization strips adversarial cues | **STOP** — approach is counterproductive |
| Qwen precision improves ≥ 0.70 AND Devstral ≥ 0.40 | Gap is primarily formatting | Integrate normalization into verification pipeline |
| < 5pp improvement on any scaffold | Gap is primarily semantic | Confirms E6: continuous learning (per-model RF) required, not normalization |
| Normalization collapses cosmetic-only diffs to empty | FM-001 diffs were entirely cosmetic | Separate finding: some patches have zero functional change |
| Discriminability ratio drops > 50% after normalization | Scaffold signal is mostly formatting | Normalization is removing the right thing |

---

## Data Flow

```
verifier-reward/results/diffs/{scaffold}/          ← raw diffs (existing)
    ↓ normalize_patches_v2.py (this experiment)
learned-verifier/results/diffs_normalized/{scaffold}/ ← canonical diffs (new)
    ↓ sweep_versions.py (verifier-reward infrastructure)
verifier-reward/results/sweep_enorm_*.jsonl        ← v009 results on normalized diffs
    ↓ analysis
learned-verifier/results/enorm_report.md           ← findings
```

---

## Script

`scripts/normalize_patches_v2.py` — supports Docker (repo cache volume) and local repo clones.

Key functions:
- `normalize_via_docker()` — clone at base_commit in Docker, read before/after, AST-normalize
- `normalize_via_local()` — same using local git repos
- `compute_discriminability()` — surface feature comparison across scaffolds
- `normalize_file()` — `strip_comments_and_docstrings()` → `ast.parse()` → `ast.unparse()`

---

## Risks

1. **AST parse failures on old Python**: Some SWE-bench repos use Python 2 syntax or have files that don't parse. Fallback: pass through unnormalized. Track failure rate.
2. **ast.unparse loses semantically meaningful formatting**: f-strings, multiline expressions, decorator ordering may change. These are usually not verification-relevant but worth monitoring.
3. **Normalization makes all scaffolds look the same but v009 still fails**: Would mean the precision gap is about *solution strategy* (different scaffolds solve problems differently), not formatting. Valuable negative result.

---

## Connection to Other Experiments

- **E6 (cross-model RF transfer)**: E_norm tests the rubric side; E6 tested the behavioral RF side. Together they determine if cross-scaffold transfer needs normalization (E_norm), continuous learning (E6), or both.
- **Iter 31 (cosmetic filtering)**: E_norm is the full version of what iter 31 attempted.
- **E5 (constraint verification)**: If E_norm fails (gap is semantic), E5's behavioral constraint extraction becomes the next approach — verify against extracted requirements rather than patch code.

---

## References

- [[Patch-Normalization-Cross-Model-Verifier-Transfer-Research]] — 22-source deep research
- [[Shopify-Flow-Verification-Flywheel-Parallels]] — Shopify DSL transpiler inspiration
- [[Verifier-Reward-Experiment-Results]] — baseline v009 cross-model results
- Agentless `postprocess_data.py`: https://github.com/OpenAutoCoder/Agentless
- R4P: arXiv:2510.22775
