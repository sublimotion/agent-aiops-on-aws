---
experiment: E_harness1
model: n/a (data-only re-analysis)
engine: n/a
hardware: local-cpu (python3.13 + scikit-learn 1.8.0)
outcome: complete
failure_categories: []

learn_commands: []
---

# Lessons — e-harness1-harness-behavioral-interaction

## Verdict: complements (with a substitution caveat)

Behavioral verification and harness-engineering are **complements, not substitutes** — but not cleanly. Full write-up in `results/report.md`.

## Key findings (2026-06-21)

- **Behavioral RF does NOT collapse on a better harness.** Within the improved (full-pipeline) verification arm, AUC=0.700 [0.618, 0.772] — clears chance. The H-substitute prediction ("AUC → chance as harness improves") is falsified.
- **The weak harness is where the RF is near-random within-condition** (ignore arm AUC 0.42–0.45, CI straddles 0.5), NOT where it shines. A degenerate harness (everyone exhausts budget, loop_count cv=0.24) gives the RF no within-arm variance to read. Behavioral signal needs trajectory *variance* to discriminate — it peaks on *middle* harnesses, not the weakest.
- **Much of the pooled 0.758 is BETWEEN-condition (harness-quality) signal.** The pooled RF partly reads "did this run use verification at all?" (the pivot-analysis +46.3pp tool-adoption lever) rather than "did this attempt fail within a fixed harness." Conditioning on harness removes that crutch (within-improved 0.70).
- **Optimal feature set changes per condition.** The canonical `selected_4` (cost/tokens_per_edit/loop/svg) is NOT re-selected in either single-harness condition. Improved arm keeps `loop_count + svg_accepted + total_tokens`; weak arm abandons behavioral entirely for rubric/debate. Confirms the cross-scaffold prior (`enew_report.md`): re-select features per harness generation.
- **Failures relocate only weakly.** Clean-trajectory failures (18/81 in good harness) are near-invisible to any signal (residual AUC 0.556, CI crosses chance). The residual that survives lives in `enew2_total_errors` and `v009_lc_count`, NOT cost/loop. Confident-wrong-with-clean-trace is the durable blind spot.
- **Improved harness compresses cost variance (0.31×) and edit-fraction (0.19×) but NOT loop_count variance (1.06×).** Loop shifts *level* down (19.7→14.7) without tightening spread — variance is the unstable quantity (matches e6 cross-model 1.50× loop var shift).

## Data-shape lessons (reusable)

- The Phase-3 RF features + gold labels live ONLY in `learned-verifier/results/combined_features.csv` (the Claude-Code VP production eval, n=300). It carries the verification-scaffold conditions inline as `beh_comp_*` one-hots — that IS the pre/post harness-improvement axis. No need to re-join the VP `eval_full300_*.jsonl` (those have pass/fail only, no behavioral features).
- The agent-harness phase2 SERA/LangGraph/Aider files have trajectory metadata only — **no RF features, no gold labels** — so partition (b) is distribution-shift-only, not AUC-capable. Only 3 harnesses ran (4 blocked); the obsidian "7-harness" assumption is wrong.
- Pooled selected_4 reproduction = AUC 0.758 (vs published 0.756) — pipeline faithful.

## Environment gotcha

- System `python3` is 3.14 with a **broken sklearn install** (`No module named 'sklearn.utils._estimator_html_repr'`). Use **`python3.13`** (sklearn 1.8.0) for any scikit-learn work in this repo.
