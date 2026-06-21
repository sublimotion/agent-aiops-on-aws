# E_fin2 — Behavioral-Feature Existence Off-Coding (FinQA reasoning programs)

Tests whether the playbook's most distinctive claim — the free $0 **behavioral
RandomForest** ("process predicts quality", **AUC 0.756** in coding, Phase-3
`selected_4`) — exists and discriminates **off-coding** on FinQA's executable
reasoning programs. Near-worst-case for behavioral signal: financial
derivations are 1–2 ops, short and clean.

Spec: `domains/autoresearch/specs/e-fin2-finqa-behavioral-features.md`
Depends on: **E_fin1** (reuses its 100 generated answers + exact-match labels;
no new generation) and the `verifier-reward` / `verification-primitives`
Phase-3 RF recipe.

## Result

**PARTIAL-FAIL as predicted (publishable scoping result).** Behavioral RF
**AUC 0.569** (95% CI [0.430, 0.709], crosses 0.5) — inside the spec's 0.55–0.65
band, ~0.19 below the 0.756 coding baseline. The Phase-3 ordering collapses
(behavioral ≈ skill ≈ combined, all 0.55–0.57). **Scoping verdict: behavioral
verification (Phase 2) is coding/long-horizon-specific; on short numeric tasks
neither behavioral nor skill verification discriminates.** See
`results/analysis.md`.

## What this runs (data-only, local — no GPU, no API)

1. **Join** each E_fin1 answer (by `id`) back to czyssrs/FinQA `dev.json` to
   recover `qa.program` / `qa.steps`.
2. **Extract** two feature families: agent-trajectory `beh_*` (the direct
   Phase-3 process analog) and gold-program `prog_*` (task-side difficulty).
3. **Fit** the verbatim Phase-3 RandomForest recipe; report AUC vs 0.756 +
   forward-selected subset.
4. **Head-to-head**: behavioral vs E_fin1 skill-verifier vs combined.
5. **Document** (not re-run) the difficulty/strategy paradox.

## Layout

```
scripts/
  extract_features.py   # join + feature extraction -> results/features.csv
  analyze.py            # Phase-3 RF recipe, head-to-head, difficulty check (python3.13!)
results/
  features.csv          # 100 x 24 (label + beh_* + prog_* + skill_*)
  rf_results.json       # all AUCs, CIs, univariate table, forward selection, paradox
  analysis.md           # full writeup + scoping verdict
  progress.md           # stage table + headline
```

## Reproduce

```bash
# FinQA original (MIT) must be cloned for qa.program (HF mirror drops it):
git clone --depth 1 https://github.com/czyssrs/FinQA /tmp/FinQA
# IMPORTANT: python3.13 — macOS python3.14 sklearn is broken (E_harness1 lesson)
python3.13 scripts/extract_features.py
python3.13 scripts/analyze.py
```

Findings: `results/analysis.md` and `lessons.md`.
