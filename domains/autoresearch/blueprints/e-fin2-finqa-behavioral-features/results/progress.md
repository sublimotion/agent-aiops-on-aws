# Progress — E_fin2 FinQA Behavioral-Feature Existence Off-Coding

**STATUS: COMPLETE — PARTIAL-FAIL (predicted, publishable).** Behavioral RF
AUC **0.569** (CI [0.430, 0.709]) vs coding baseline 0.756. Phase 2 is
coding/long-horizon-specific. See `analysis.md`.

| Stage | Status | Notes |
|-------|--------|-------|
| 0 Program-field gate | ✅ | 100/100 E_fin1 examples join to czyssrs/FinQA dev.json with non-empty `qa.program`. |
| 0 Baseline re-quote | ✅ | Phase-3 selected_4 RF AUC **0.756**; behavioral-only 0.730 > v009 0.682 = debate 0.682; difficulty-conditioning regressed 0.756→0.743 (carried, not re-run). |
| 1 Feature extraction | ✅ | 9 agent-trajectory `beh_*` + 8 gold-program `prog_*` + 5 E_fin1 skill signals → `features.csv` (100 × 24). |
| 2 Phase-3 RF fit | ✅ | Verbatim recipe (n_estimators=200, max_depth=7, balanced, seed=42; 5-fold pooled OOF). behavioral_all 0.569; behavioral_4 0.462; program 0.408. |
| 3 Forward selection | ✅ | [out_tokens, op_count, op_diversity] → 0.790 OOF, flagged as small-n in-sample-selected overfit (2/3 are difficulty feats). |
| 4 Head-to-head | ✅ | behavioral 0.569 ≈ skill 0.557 ≈ combined 0.545. Phase-3 ordering collapses; fusion adds nothing. |
| 5 Difficulty paradox | ✅ | DOCUMENTED (no conditioned RF). No sign reversal. Harder tasks pass more (0.80 vs 0.64) — op-count is a poor difficulty axis. |
| 6 Report + lessons | ✅ | analysis.md, lessons.md, README.md. |

## Headline (n=100, base rate 0.71)

| Model | AUC | CI95 |
|---|---:|---|
| Coding baseline (Phase-3 selected_4) | 0.756 | — |
| behavioral_all (agent-trajectory) | **0.569** | [0.430, 0.709] |
| program_structural (gold-program) | 0.408 | [0.282, 0.527] |
| skill_verifier (E_fin1) | 0.557 | — |
| behavioral + skill | 0.547 | — |

## Verdict
- **RQ1 (universal?): NO.** Behavioral collapses 0.756→0.569 on short clean
  derivations. Loop/revision/cost signatures are structurally absent.
- **RQ2 (ordering?): collapses.** All signals 0.55–0.57; behavioral does not
  beat skill; combining adds nothing. Reproduces E_fin1's null.
- **RQ3 (paradox?): not present** (documented per prior, not re-run).
- **Scoping**: Phase 2 behavioral verification is long-horizon/multi-step
  specific. Skill/outcome verification is more general but also fails on FinQA's
  same-tier numeric reasoning (E_fin1).
