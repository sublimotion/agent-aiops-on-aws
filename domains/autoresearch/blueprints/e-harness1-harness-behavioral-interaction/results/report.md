# E_harness1 — Harness × Behavioral-Verifier Interaction

**Date**: 2026-06-21 · **Mode**: local, data-only (no GPU, no API generation, no new trajectories)
**Verdict**: **Complements — with a substitution caveat.** Behavioral discrimination *survives* harness improvement but does not transfer as-is: the optimal feature set changes per harness, and a substantial share of the pooled RF's power reads *harness quality* rather than *failure within a fixed harness*.

---

## Stage 0 — data on disk & priors (all repo-verified)

| Prior | Quoted value | Source (repo) |
|-------|--------------|---------------|
| Phase-3 RF (selected_4) AUC | **0.756** | `learned-verifier/results/phase3_report.md` |
| Importances | cost **0.420**, tokens/edit **0.327**, loop **0.201**, svg **0.052** | same |
| Harnesses that actually ran | **3** (SERA, LangGraph, Aider); 4 blocked (SWE-agent, OpenHands, Claude Code, OpenCode) | `agent-harness/lessons.md` |
| Cross-scaffold transfer fails both ways | Claude→Qwen3.5 **0.363**, reverse **0.410**; loop_count var shift **1.50×** | `learned-verifier/results/e6_cross_model_report.md` / `enew_report.md` |
| Dominant lever = adopting tooling at all | Tool-use risk-diff **+46.3%** [+31.1, +61.4], p=4.86e-07 | `pivot-analysis/results/pivot_report.md` |

**Reproduction check**: re-fitting selected_4 with the verbatim Phase-3 recipe (RF 200×depth-7, class-balanced, 5-fold stratified, pooled OOF) on `combined_features.csv` gives **AUC=0.7579** — matches the published 0.756. The pipeline is faithful.

### Critical data-shape finding (resolves the spec's Stage-0 caveat)
The 4 RF features (`beh_total_cost_usd`, `beh_tokens_per_edit`, `beh_loop_count`, `svg_accepted`) **and** gold labels exist *only* in `combined_features.csv` — the Claude-Code VP SWE-bench production eval (n=300, 175 pass / 125 fail). That corpus encodes the **harness-quality axis inline** via the verification-scaffold composition one-hots:

| Condition (verification scaffold) | n | pass | pass-rate | Role |
|-----------------------------------|---|------|-----------|------|
| `beh_comp_ignore` (no verification) | 33 | 7 | 0.212 | **weak / pre-improvement** |
| `beh_comp_full_pipeline` (gen→run→review) | 208 | 140 | 0.673 | **improved / post-improvement** |
| (`generate_run` 27, `gen_run_iterate` 7, `generate_only` 4, `other` 3 — too thin to fit) | | | | excluded |

The same split coincides with the pivot-analysis lever: tool-not-used (n=33, pr=0.21) vs tool-used (n=249, pr=0.67).

**The agent-harness SERA/LangGraph/Aider eval has NO RF features and NO gold labels** — only trajectory metadata (turns, edits, repeat_count). So partition (b) supports a **distribution-shift report only**, not an RF re-fit. This is an honest limitation, reported below; the interaction verdict rests on partition (a).

---

## 1. Behavioral feature distributions per condition (compression test)

Variance ratio = improved ÷ weak (verification-scaffold axis). <1 ⇒ the improved harness **compresses** that signal.

| Feature | Weak mean (var) | Improved mean (var) | **Var ratio** | Compressed? |
|---------|-----------------|---------------------|---------------|-------------|
| `beh_total_cost_usd` | 0.644 (0.083) | 0.334 (0.026) | **0.31×** | strong ✓ |
| `beh_action_pct_edit` | 0.110 (0.0128) | 0.086 (0.0025) | **0.19×** | strong ✓ |
| `beh_tokens_per_edit` | 722k (1.75e11) | 576k (1.01e11) | **0.58×** | moderate ✓ |
| `beh_loop_count` | 19.7 (22.2) | 14.7 (23.5) | **1.06×** | **no** ✗ |
| `beh_adversarial_review_used` | 0.0 (0) | 1.0 (0) | n/a | (by construction) |

**Reading**: the improved harness halves the *mean* cost and loop_count and **compresses the variance of cost (0.31×) and edit-fraction (0.19×)** — exactly the budget-exhaustion / thrashing signatures the RF's top two features read. But **loop_count variance does not compress** (1.06×): the improved harness shifts the loop-count *level* down (19.7→14.7) without tightening its spread. The thrashing signal is relocated, not erased. (This mirrors the e6 cross-model finding that loop_count is the least transferable feature — variance, not mean, is the unstable quantity.)

---

## 2. Phase-3 RF re-fit *within* each condition (the core test)

5-fold stratified, pooled OOF, bootstrap 95% CI (2000 resamples).

| Condition | feature set | n | **AUC** | 95% CI | P@R≥30% |
|-----------|-------------|---|---------|--------|---------|
| POOLED (baseline repro) | selected_4 | 300 | **0.758** | [0.701, 0.810] | 0.917 |
| POOLED | beh_3 | 300 | 0.738 | [0.678, 0.792] | 0.915 |
| **weak / ignore** | selected_4 | 33 | **0.423** | [0.215, 0.636] | 0.25 |
| **weak / ignore** | beh_3 | 33 | 0.451 | [0.237, 0.676] | 0.25 |
| **improved / full_pipeline** | selected_4 | 208 | **0.700** | [0.618, 0.772] | 0.879 |
| **improved / full_pipeline** | beh_3 | 208 | 0.668 | [0.591, 0.740] | 0.913 |

**Three findings, none matching the naive H-substitute story:**

1. **Discrimination does NOT collapse on the better harness.** Within the improved harness, AUC=0.700 (CI clears 0.5 comfortably). The H-substitute prediction — "AUC drops toward chance as the harness improves" — is **falsified for the improved arm.**

2. **The weak harness is where the RF is *near-random* within-condition** (AUC 0.42–0.45, CIs straddle 0.5). This is the opposite of the spec's framing that behavioral verification has highest ROI on weak harnesses. Mechanistically: the weak (no-verification) arm is a near-degenerate cell — everyone exhausts budget (loop_count tight at ~19.7, cv=0.24) and only 7/33 pass, so there is almost no *within-arm* behavioral spread for the RF to read. The behavioral signal needs a harness that produces **variance** in trajectories to discriminate.

3. **Most of the pooled 0.758 is BETWEEN-condition signal.** The pooled RF separates classes largely because no-verification trajectories (high cost, high loop, pr=0.21) look different from full-pipeline ones (low cost, pr=0.67). Conditioning on a fixed harness removes that crutch and the within-arm AUC drops to 0.70 (improved) / ~chance (weak). **A large fraction of the RF's apparent power is reading harness quality, not failure-within-harness** — the direct answer to RQ1.

---

## 3. Failure-relocation probe (H-complement test)

"Clean trajectory" := within the improved harness, `loop_count ≤ median (14.5)` **and** `cost ≤ median (0.294)` — the failure signatures suppressed.

| Group (improved harness) | n | pass | fail | pass-rate |
|--------------------------|---|------|------|-----------|
| clean trajectory | 81 | 63 | 18 | **0.778** |
| dirty trajectory | 127 | 77 | 50 | 0.606 |

Clean trajectories pass more often (0.78 vs 0.61) — the behavioral signal is real even within the good harness. **But 18 clean-trajectory failures remain.** Can anything flag them?

| Signal on clean cases | RF AUC (n=81) | 95% CI |
|-----------------------|---------------|--------|
| behavioral (beh_3) | 0.638 | [0.482, 0.783] |
| residual (v009 + debate + svg + errors + read:edit) | **0.556** | [0.410, 0.703] |

Both CIs **cross 0.5**. The strongest individual separators of clean failures vs clean passes:

| Feature | pass mean | fail mean | Δ |
|---------|-----------|-----------|---|
| `enew2_total_errors` | 1.49 | 2.28 | **−0.79** |
| `v009_lc_count` | 1.63 | 1.11 | +0.52 |
| `v009_mean_score` | 0.687 | 0.588 | +0.10 |

**Reading**: failures **partially relocate** — clean failures carry slightly elevated runtime-error counts (`enew2_total_errors`) and slightly lower rubric agreement (`v009_lc_count`). But the relocated signal is **weak** (residual AUC 0.556, not significant). Clean-trajectory failures are **largely invisible to behavioral signal** — the confident-wrong-with-clean-trace mode the spec anticipated is real and is the residual blind spot. This is *partial* complementarity, not the strong "AUC holds because failures cleanly relocate" version of H-complement.

---

## 4. Per-condition forward feature selection (does the optimal set change?)

Greedy forward selection (pooled-OOF AUC) over all ~80 numeric features, **independently per condition**:

| Condition | Forward-selected set | Final AUC |
|-----------|----------------------|-----------|
| weak / ignore | `beh_token_efficiency_ratio` → `debate_advocate_confidence` → `beh_v009_verdict` | 0.879* |
| improved / full_pipeline | `beh_loop_count` → `svg_accepted` → `beh_total_tokens` | 0.758 |
| POOLED | `beh_total_cache_read_tokens` → `debate_verdict_correct` → `svg_accepted` → `enew1_n_writes` | 0.821 |

\* weak-condition n=33 → forward AUC is heavily optimistic / unstable; treat as directional only.

**The optimal feature set changes across conditions** — and crucially, **the canonical selected_4 (cost / tokens_per_edit / loop / svg) is not re-selected in either single-harness condition.** The improved-harness search keeps `loop_count` + `svg_accepted` but swaps cost→`total_tokens`; the weak search abandons the behavioral trio entirely for rubric/debate signals. This confirms RQ2 (**yes, the feature set must be re-selected per harness generation**) and is consistent with the repo's cross-scaffold prior (`enew_report.md`: feature *boundaries* don't transfer; `e6`: thresholds don't transfer even when features do).

---

## 5. Partition (b) — 3-harness distribution shift (supplement, distribution-only)

Devstral-24B across the harnesses that actually ran (50-issue subset, seed 42). **No RF features / no gold labels in this corpus** → no AUC. Proxies only.

| Harness | fix-rate | edit-rate | turns (mean) | first-edit turn | loop proxy (repeat_count) | latency s |
|---------|----------|-----------|--------------|-----------------|---------------------------|-----------|
| SERA | 0.46 | 0.48 | 29.1 | 13.5 | 11.3 (var 72.9) | 87.6 |
| LangGraph | 0.62 | 0.64 | 28.4 | 19.7 | — (not logged) | 50.2 |
| Aider | 0.00 | 0.00 | 0.0 | — | — | 5.9 |

The "better" harness (LangGraph, higher fix-rate) pushes first-edit *later* (19.7 vs 13.5) and runs faster — a different behavioral fingerprint per harness, echoing the partition-(a) finding that trajectory shape is harness-specific. Aider degenerate (can't drive Devstral). This corpus cannot test AUC durability; it only shows the *inputs* to the behavioral features shift markedly across harnesses, reinforcing "re-select per harness."

---

## Verdict — Complements, with a substitution caveat

**Not substitutes.** Behavioral discrimination does **not** collapse as the harness improves: within the improved (full-pipeline) harness AUC=0.700 [0.618, 0.772], well above chance. The H-substitute prediction is falsified.

**Not clean complements either.** Failures relocate only *weakly* — clean-trajectory failures (18/81 in the good harness) are near-invisible (residual AUC 0.556, CI crosses chance). The verifier survives, but it loses the failures that matter most under a good harness (confident-wrong, clean trace).

**The load-bearing finding (RQ1 + RQ2):**
- A large share of the pooled RF's 0.758 is **between-condition (harness-quality) signal**: it partly reads "did this trajectory use verification at all?" (the +46.3pp pivot lever) rather than "did *this* attempt fail within a fixed harness." Conditioning on harness removes that crutch (within-improved 0.70, within-weak ≈ chance).
- The optimal feature set **changes per condition** — selected_4 is a *pooled compromise*, not the within-harness optimum. Re-selection per harness generation is mandatory (confirms the cross-scaffold prior).

### Playbook caveat (where behavioral verification has ROI on the harness-quality axis)
- **Behavioral verification's marginal value is highest on harnesses that produce trajectory *variance*** — not the weakest harnesses (degenerate, everyone-exhausts-budget, no within-arm spread → within-weak AUC ≈ chance), and not the cleanest (failures go invisible). It peaks in the **middle**: harnesses good enough to vary but not so good that failures are silent.
- **Re-select features per harness generation.** Do not ship `selected_4` as a fixed verifier across harness upgrades; its top features compress (cost 0.31×, edit% 0.19×) as the harness improves and the within-harness optimum drifts to `loop_count + svg + total_tokens`.
- **As harnesses improve, pair behavioral signal with a *content* verifier** (rubric/debate) — the residual clean-failure signal that does survive lives in `enew2_total_errors` and `v009_lc_count`, not in cost/loop. This is the complement that keeps the verifier alive across harness evolution.

---

## Files
- `scripts/analyze_harness_interaction.py` — partition (a): per-condition RF, distribution shift, relocation, forward selection.
- `scripts/agent_harness_proxy_shift.py` — partition (b): 3-harness trajectory distribution (distribution-only).
- `results/harness_interaction_results.json` — full numeric results (partition a).
- `results/agent_harness_proxy_shift.json` — partition (b) distributions.

## Known limitations
- Weak/ignore condition n=33 (7 pass) — within-condition AUC and forward-selection are unreliable; CIs reported, thin cells not over-read.
- "Harness quality" operationalized via the VP verification-scaffold split — a coarse proxy for the full harness-quality axis (the spec's stated limitation).
- Partition (b) (SERA/LangGraph/Aider) lacks RF features and gold labels → no AUC, distribution-shift only.
- selected_4 reproduction (0.758) confirms the pipeline; minor digit differences from 0.756 are RNG/sklearn-version noise.
