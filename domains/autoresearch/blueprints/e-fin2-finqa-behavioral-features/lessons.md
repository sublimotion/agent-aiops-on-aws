# Lessons — e-fin2-finqa-behavioral-features

## Result summary
Behavioral/process verification (the Phase-3 free-$0 RandomForest, **AUC 0.756**
in coding) **does not transfer to short numeric reasoning**. On FinQA's 1–2-op
derivations the agent-trajectory behavioral RF scores **AUC 0.569**
(95% CI [0.430, 0.709], crosses 0.5), a partial-fail inside the predicted
0.55–0.65 band. The Phase-3 ordering (behavioral > LLM-signals) collapses:
behavioral 0.569 ≈ skill-verifier 0.557 ≈ combined 0.545.

## Lessons captured during the run

- **Behavioral verification is a long-horizon phenomenon, not a universal one.**
  The Phase-3 RF's top features (cost 0.42, tokens_per_edit 0.33, loop_count
  0.20) read *process pathologies* (loops, cost blowups, repeated edits). On a
  one-shot numeric answer these are **structurally absent**: `beh_revision_count`
  and `beh_abstain` are ~0 across 98–99/100 examples with **zero RF importance**;
  cost is flat (one short Haiku call each). No substrate → no signal. This is the
  honest scope bound for the playbook's "process predicts quality" claim.

- **Two feature families must be kept separate off-coding.** FinQA's gold
  `qa.program` is a property of the *task*, not the agent's behavior. Treating
  gold-program structure (op count/diversity) as a "behavioral" feature
  conflates difficulty with process. Reported them separately: agent-trajectory
  `beh_*` (the true Phase-3 analog, 0.569) vs gold-program `prog_*` (difficulty
  proxy, 0.408). A naive "just throw the program features in" would have inflated
  the apparent behavioral signal.

- **Forward selection overfits at n=100 — report it, don't believe it.** Greedy
  forward selection hit OOF AUC 0.790, but on the same n=100 it selects on, and
  2/3 picks were difficulty (gold-program) features. Pre-committed fixed feature
  sets are the trustworthy number; none cleared 0.6. Selecting features on the
  metric you report is optimistically biased — always pair it with a fixed-set
  baseline.

- **Reproduces E_fin1's null from a second angle.** E_fin1 found the *skill*
  verifier sits at base rate off-coding (no verification asymmetry — verifier
  redoes the same numeric reasoning). E_fin2 finds the *behavioral* verifier also
  sits at base rate (no process trace to read). Two independent verification
  primitives both fail on the same short-numeric structure → the failure is
  domain-structural, not primitive-specific.

- **Op-count is a poor difficulty axis on FinQA.** Harder (≥2-op) tasks passed
  *more* (0.80 vs 0.64 for 1-op). Single-op questions are dominated by
  table-lookup/units ambiguity (E_fin1's label-noise finding), not arithmetic
  difficulty. No Simpson's-paradox sign reversal appeared — consistent with the
  coding prior (`enew_report.md`) that the reversal is a long-horizon-strategy
  artifact, which one-shot FinQA answers lack.

## Carryover applied (priors that held)
- **python3.13 for sklearn** (E_harness1): macOS python3.14 sklearn is broken
  (`No module named sklearn.utils._estimator_html_repr`). Used python3.13 +
  sklearn 1.8.0 throughout. Saved a debugging cycle.
- **Verbatim Phase-3 RF recipe** (n_estimators=200, max_depth=7,
  class_weight=balanced, seed=42; nan→-999; 5-fold pooled OOF) reused exactly
  from E_harness1's reproduction of `train_combined.py` — no recipe drift.
- **Baseline numbers carried, not re-derived** (the `learned-verifier` repo is
  not in this tree): 0.756 / behavioral 0.730 / v009 0.682 / debate 0.682;
  difficulty-conditioning regressed 0.756→0.743 → RQ3 documented, not re-run.
- **czyssrs/FinQA hard gate** (E_fin1): the HF mirror drops `qa.program`; cloned
  the GitHub original. Re-clone needed (the /tmp clone from E_fin1 was gone).

## Limitations
- n=100, base rate 0.71 → wide CIs (behavioral AUC CI includes 0.5). The
  *direction* (collapse to chance) is robust (mirrored by E_fin1); point
  estimates are imprecise.
- Single model (Haiku), single corpus. FinQA's short-derivation structure is the
  point, not a sampling artifact — it is the only finance corpus with an
  executable process trace.
