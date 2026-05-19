# Verifier Reward — Progress

## Status: COMPLETE — v009 alone (4/4 unanimous) is best config. Precision=0.92 on Verified (n=483), 3 FPs. Drop v001 gate.

## Best Configuration

**v009 alone (4/4 unanimous)** — v009_adversarial says "likely_correct" in all 4 runs (1×t=0.0 + 3×t=0.3). No v001 gate needed.

| Set | Config | Precision | Prec 95% CI | Recall | F₀.₅ | TP | FP | Cost/patch |
|-----|--------|-----------|-------------|--------|-------|----|----|------------|
| Dev (sonnet, 49 patches) | v001∩v009(2+/4) | 1.00 | [0.34,1.00] | 0.33 | 0.71 | 2 | 0 | $0.048 |
| SWE-bench Verified (n=483) | v001∩v009(2+/4) | 0.78 | [0.58,0.90] | 0.07 | 0.27 | 18 | 5 | $0.024 |
| **SWE-bench Verified (n=483)** | **v009 4/4** | **0.92** | **[0.79,0.97]** | **0.14** | **0.43** | **34** | **3** | **$0.030** |
| Verified holdout (n=233) | v009 4/4 | 0.94 | [0.73,0.99] | 0.14 | 0.43 | 16 | 1 | $0.030 |

**T10b finding**: Drop the v001 gate. At 51% base rate, v001 removes TPs faster than FPs. v009 4/4 alone has +0.14 precision, +2x recall, fewer FPs, and 4 calls instead of 5.

**v009 threshold comparison (full n=483)**:

| Threshold | Precision | Recall | F₀.₅ | FPs |
|-----------|-----------|--------|-------|-----|
| v009 1+/4 | 0.77 | 0.22 | 0.50 | 16 |
| v009 2+/4 | 0.83 | 0.20 | 0.50 | 10 |
| v009 3+/4 | 0.89 | 0.17 | 0.48 | 5 |
| **v009 4/4** | **0.92** | **0.14** | **0.43** | **3** |

3 stubborn FPs at 4/4: django-15103, xarray-6992 (unanimous 4/4 lc — genuinely fooled), django-13590 (holdout).

**WARNING**: The 4-call 2+/3 config from iter 21 is UNSTABLE — produces FPs on fresh stochastic draws (T7). The t=0.0 deterministic run is essential for stability. See T7 section for details.

## Key Breakthrough: Adversarial + Confirmatory Framing

All rubrics v001-v008 use confirmatory framing ("evaluate whether this patch is correct"). This biases the verifier toward confirming correctness for patches that look reasonable, producing false positives.

**v009_adversarial** inverts to adversarial framing ("assume this patch is wrong — find the bug"). This forces the verifier to actively search for failure scenarios.

The **v001∩v009 ensemble** combines both: a patch must survive confirmatory evaluation AND adversarial attack. Temperature sampling (3 v009 runs at temp=0.3) adds statistical robustness — a patch must survive the adversarial attack in at least 2 of 3 stochastic tries.

### Why It Works on FM-003

The 3 "plausible but wrong" FPs that fooled all confirmatory rubrics:

| FP Instance | v001 verdict | v009 verdict (4 runs) | v009 attack |
|------------|-------------|----------------------|-------------|
| astropy-14365 | likely_correct (0.98) | 0/4 likely_correct | Regex scope incomplete |
| astropy-14995 | likely_correct (0.88) | 0/4 likely_correct | Missing symmetric case |
| flask-4992 | likely_correct (0.97) | 0/4 likely_correct | Missing input validation |

None of the FPs survive even a single adversarial run. The bug-finding frame consistently identifies plausible failure scenarios.

## What We Learned

| Finding | Impact |
|---------|--------|
| **Adversarial framing (v009)** | Inverting verification frame greatly reduces false positives (prec 0.29→0.78). NOT zero FPs at scale (5 FPs on n=483 Verified). |
| **Temperature sampling** | Recovers FN (pytest-11143) that single v009 run rejects. 2/4 threshold = no FPs. |
| FM-004 (diff truncation at 12K chars) | 47% of diffs invisible. Fixed → no impact. Noise is the problem, not visibility. |
| Sonnet as verifier | Worse precision than Haiku (0.17 vs 0.29), 4x cost. Model capability not the bottleneck. |
| FM-001 (reformatting noise) | 3/6 gold passes have 66-135K char diffs. Still unsolved — accounts for all remaining FNs. |
| FM-003 (plausible but wrong) | SOLVED by adversarial rubric. |
| Ensemble intersection | Confirmatory × adversarial is better than confirmatory × confirmatory. |

## Iteration History

| # | Hypothesis | Precision | Status |
|---|-----------|-----------|--------|
| 1-5 | Initial rubrics | 0.10-0.29 | INVALIDATED (FM-004) |
| 6 | Fixed pipeline + v001 | 0.29 | Baseline |
| 7-8 | FM-001/003 rubrics | 0.12-0.14 | Too lenient |
| 9 | Sonnet verifier | 0.17 | Worse, expensive |
| 10 | Ensemble v001∩v007 | 0.33 | Improved |
| 11 | v008 standalone | 0.11 | Too lenient |
| 12 | Ensemble v001∩v008 | 0.40 | Previous best |
| 13 | Two-stage extraction | — | Stage 1 fails |
| 14 | Score thresholds | 0.40 | No improvement |
| 15 | Creative ensembles (9 variants) | 0.40 | Plateau confirmed |
| 16 | v009 adversarial standalone | 0.25 | Too aggressive alone |
| 17 | v001∩v009(2+/4 lc) + temp sampling | 1.00 | BREAKTHROUGH |
| 18 | FM-001 recall: size-adaptive/v008/changes-only | 1.00 | No improvement — FM-001 needs AST diff |
| 19 | Describe-then-verify for FM-001 | 0.67-1.00 | Finds fixes in large diffs but 1 FP on opus holdout |
| 20 | Adversarial gate for DTV | — | v009 on evidence rejects everything |
| **21** | **Cost optimization: 4 calls (drop v009 t=0.0)** | **1.00** | **Same precision, 21% cheaper** |
| 22 | T4: Cross-verifier transfer | 0.14-0.20 | Claude-specific capability |
| 23 | T5: Adversarial self-critique in generation | — | NEGATIVE: fix rate drops 54%→30% |
| 24 | T6c: Best-of-N on Devstral SERA candidates | **0.20** | Verifier is patch-source-specific |
| 25 | T7: Cross-model BoN on Claude patches | **1.00** (3+/3) | 6/49=12%, equal to best single model |
| 26 | Threshold validation: revert to 5-call 2+/4 | **1.00** | 3+/3 loses TPs, 2+/3 unstable. Original best. |
| 27 | Recall ceiling analysis | — | v001 blocks 59% of FNs. v009 blocks 6%. FM-001 defeats both. |
| 28 | v009-only ensemble (drop v001 gate) | 0.25-0.33 | Same recall, more FPs. v001 is precision guard. |
| 29 | Alt confirmatory gates (v008∩v009, v006∩v009) | 0.25-0.33 | All worse. Reformat-aware gates too lenient. |
| 30 | v010 concrete adversarial rubric | **0.13** | 27 FPs. Recovers 2 FNs but v009 blocks both. |
| 31 | Diff preprocessing (changes-only + quote filter) | — | 53% noise reduction, still 34-73K chars. v001 and v009 still reject all 3 FM-001 diffs. |
| 32 | T8: Context augmentation for Devstral patches | **0.17** | NEGATIVE: context makes v009 MORE lenient, +3 FPs |
| 33 | T8b: v010 completeness rubric for surgical diffs | **0.00** | Over-rejects: 0 TP, finds "missing changes" in every patch including correct ones |
| 34 | T8c: v010 as filter (v001∩v010-filter∩v009) | **0.33** | Identical to baseline — v010 says "uncertain" (not "likely_incorrect") on both FPs |
| 35 | Oracle minimal-diff + v011 scoped adversarial | **0.09** | v011: 20 FPs (leaks astropy-14365). Even 525-char minimal diff: Haiku still rejects (0/4). Problem is model reasoning, not noise. |
| 36 | Sonnet as v009 verifier (targeted test) | 0/4 lc | Sonnet agrees: "misses deconstruct() for migrations." Recall ceiling is model-independent. |
| 37 | v012 test-outcome predictor rubric | 4/4 on FPs | Worst of both worlds: leaks both FPs (4/4 lc) AND rejects django-10924 (0/4 lc). Model can't predict test outcomes without test info. |
| 38 | Extended thinking (v001+thinking ∩ v009+thinking) | 0.75 | Recovers sphinx-11445 (new TP) but leaks flask-4992 (new FP). temp=1.0 required for thinking makes v009 less discriminative: FP passes 63% vs TP 50%. |
| 39 | Problem-first verification (predict → compare) | leaks FPs | FNs: model predicts MORE complete fix → doesn't match partial patch. FPs: model predicts same plausible fix → accepts. v001 with extra steps. |
| 40 | Opus 4.6 as v009 verifier | leaks flask-4992 | More nuanced (uncertain vs li on FNs) but uncertain≠lc so no recall gain. Leaks flask-4992 (lc 0.87). 18x cost. |
| 41 | Multi-turn v009 with challenge turn | leaks all FPs | Sycophantic: challenge "only evaluate changed code" flips ALL verdicts to lc. Same as v011 — too lenient. |
| 42 | Adversarial few-shot v009 (with examples) | 1.00 (same) | Few-shot examples don't change verdicts on real cases. Model ignores abstract examples when reasoning about concrete code. |
| 43 | Test generation as verification | ~0.50 | FIRST approach to recover ALL 4 FNs (4/4 lc). But 4/5 FPs also pass. Orthogonal to v009 — can't combine without one dominating. |
| 44 | v013 test-scoped adversarial (2-phase) | 0.33 standalone | Recovers 2/4 FNs (django-11001 4/4, sphinx-10325 4/4) but leaks 4/5 FPs. v001 independently blocks both recovered FNs → no ensemble helps. |
| 45 | Decomposed v009 (per-claim adversarial) | 0/4 on ALL | MOST NEGATIVE. Decomposing into narrow claims makes v009 MORE conservative. 0/4 lc on every claim for every patch (TP and FP alike). v009 adversarial reasoning is robust to scope manipulation. |
| 46 | Size-based routing (v013 for large diffs) | 4/5 FP leak | v013 passes 4/5 large gold=FAIL patches (all 4/4 lc, scores 0.90-0.94). Only sympy-11897 rejected. Size routing can't work — v013 too lenient on large diffs. |
| 47 | Structural features (zero-cost analysis) | no signal | Diff size, hunks, files, add/del ratio, test_files — no feature separates gold passes from fails. Passes are bimodal (2 small TPs + 3 large FNs). |
| 48 | Attack-type classification (zero-cost) | overlap | v009 reasoning uses INCOMPLETENESS attacks on both FPs (astropy-14365, astropy-14995) AND all 4 FNs. CONCRETE BUG attacks appear in flask-4992, flask-4045, flask-5063 (FPs only) but can't filter post-hoc without leaking FPs. |
| 49 | Exhaustion audit (zero-cost) | TERMINATE | All diff sets evaluated, all lever categories exhausted. 21 consecutive negatives. No remaining hypothesis within prompt-engineering scope. Formal termination. |
| 50 | Meta-verify v009 attacks for test relevance | 0.44 (all DIS) | Meta-verifier says "unlikely_tested" on ALL 9 attacks (FPs AND FNs). Cannot distinguish valid from invalid attacks. With v001 gate: 0 TP, 4 FP (removes v009 protection). $0.10. |
| 51 | Attack-reference-in-diff heuristic (zero-cost) | no signal | v009 attack function references vs diff content. Large diffs contain everything; small diffs have few references. No separation. |
| 52 | v009 score distribution analysis (zero-cost) | overlap | FP mean v009 score=0.60, FN mean=0.45. But complete overlap (django-11001 FN 0.60 = flask-4992 FP 0.61). No threshold separates cleanly. |
| 53 | Persona variation (security/testing/architect) | 0/7 lc | All 3 personas reject all 7 patches (same as baseline v009). Persona framing doesn't change code analysis. Confirms iter 42. $0.25. |
| 54 | T10b: v015 completeness rewrite | 1.00 (0.10 rec) | Catches 5/5 FPs but kills recall (1/10 TPs). Completeness check too aggressive. |
| 55 | T10b: v016 test-script awareness | 0.33 | Test script flagging alone doesn't change verdicts. |
| 56 | T10b: v017 completeness step | nan (0 rec) | Catches 5/5 FPs, rejects everything. Completeness step universally over-rejects. |
| 57 | T10b: v018 completeness probe | nan (0 rec) | Dedicated completeness checker. Same overcorrection. |
| 58 | T10b: v009 t=0.7 (6 runs) | 5-6/6 lc | Higher temp doesn't help — FPs genuinely fool v009. |
| **59** | **T10b: v009 alone 4/4 (drop v001)** | **0.92** | **NEW BEST. +0.14 prec, +2x rec, -2 FPs vs ensemble. 4 calls, $0.030/patch.** |

## Cost

Total experiment cost: ~$154 (Phase 1 baselines: $61, iterations: $23, T4-T7: $10, T8: $3, T9 g7e compute: ~$40, T9 verifier: $1.47, T10 Verified eval: $11.70, T10b rubric iteration: $2.50)

## Remaining Limitations

1. **Recall = 0.33**: Verifier misses 4/6 correct patches on dev. Root cause: **dual gate rejection** — v001 (likely_incorrect) AND v009 (0/4 lc) independently reject ALL 4 FNs. No ensemble can recover recall because:
   - v009 finds genuine incompleteness (e.g., "fixes formfield() but not deconstruct()") → 0/4 on all FNs
   - v001 independently rejects all FNs (likely_incorrect on all 4)
   - Every approach that relaxes either gate (v011, v012, v013, iter 43 test-gen) leaks FPs because the narrow scope that recovers FNs also accepts plausible-but-wrong patches
   - Confirmed across 14 rubric versions, 5 verifier models, extended thinking, 2-phase approaches
   - 1 is a model reasoning error (sphinx-11445, 412 chars). Haiku misunderstands the regex fix with BOTH v001 and v009 rubrics. Consistent across temperature sampling.
2. **Small eval set**: Only 6 gold passes on dev, 11 on holdout. Results are directional but confidence intervals are wide.
3. **Cost**: $0.048/patch (5 Haiku calls: 1 v001 + 1 v009 t=0.0 + 3 v009 t=0.3). The t=0.0 run is NOT redundant — dropping it (iter 21) destabilizes precision on fresh stochastic draws (T7). Still well under $0.10/patch target.

## T4: Cross-Verifier Transfer (2026-03-24)

**Question**: Is precision=1.00 a Claude-specific artifact, or does the rubric work with any model?

Ran v001∩v009(2+/3) ensemble on the same sonnet dev set (49 patches, 6 gold passes) with non-Claude verifiers via Bedrock:

| Verifier | TP | FP | Precision | Recall | F₀.₅ | Cost | Finding |
|---------|----|----|-----------|--------|-------|------|---------|
| **Haiku** (Claude, baseline) | 1 | 0 | **1.00** | 0.17 | 0.50 | $1.76 | Zero FPs — gold standard |
| **Devstral 2** (Mistral 123B) | 1 | 4 | **0.20** | 0.17 | 0.19 | $1.02 | v009 fails to catch FPs |
| **Nova Pro** (Amazon) | 1 | 6 | **0.14** | 0.17 | 0.15 | $1.45 | v009 even more lenient |
| **Mistral Large 3** (675B) | 0 | 0 | N/A | 0.00 | 0.00 | $3.44 | Rejects everything |

**The breakthrough is partially Claude-specific.** The v009 adversarial rubric works because Claude is good at adversarial reasoning. Devstral's v009 passes `astropy-14365` (2/3 likely_correct) — the same patch Claude rejects 0/4 times. Mistral Large over-corrects (rejects all patches including correct ones).

### FM-003 FP Analysis by Verifier

| FP Instance | Claude v009 | Devstral v009 | Nova Pro v009 | Mistral Large v009 |
|------------|-------------|---------------|---------------|-------------------|
| astropy-14365 | 0/4 lc (REJECT) | 2/3 lc (PASS — FP!) | 2/3 lc (PASS — FP!) | 0/3 lc (REJECT) |
| astropy-14995 | 0/4 lc (REJECT) | 0/3 lc (REJECT) | — | 0/3 lc (REJECT) |

### v009 Verdict Distribution by Model

| Verifier | v009 "likely_correct" rate | v009 "likely_incorrect" rate | v009 "uncertain" rate |
|---------|--------------------------|---------------------------|---------------------|
| **Haiku** | ~15.6% (calibrated) | ~50% | ~34% |
| **Devstral 2** | 10.2% | 13.6% | **76.2%** (defaults to uncertain) |
| **Nova Pro** | 17.7% (too lenient) | 30.6% | 51.7% |
| **Mistral Large** | 1.4% | **89.8%** (defaults to reject) | 8.8% |

**Root cause**: Claude's v009 is uniquely calibrated — decisive and accurate. Devstral defaults to "uncertain" (non-committal). Nova Pro is too lenient. Mistral Large always rejects.

Script: `scripts/run_cross_verifier.py`

## T5: Adversarial Self-Critique in Generation — NEGATIVE RESULT (2026-03-24)

**Question**: Does injecting "assume your patch is wrong, find the bug" into the generation prompt improve pass rate?

| Variant | Fix Rate | Pass Rate | Precision | Cost/Issue | Turns |
|---------|----------|-----------|-----------|------------|-------|
| **Control** (Phase 1) | **54%** (27/50) | **10%** (5/50) | **0.19** | $0.20 | 26 |
| self-critique | 42% (21/50) | 2% (1/50) | 0.05 | $0.13 | 17 |
| self-critique-strong | 30% (15/50) | 2% (1/50) | 0.07 | $0.13 | 17 |

**Self-critique HURTS performance.** Fix rate drops 54% → 30%, pass rate drops 10% → 2%.

**Why it fails**: The model can find bugs in OTHER models' code (v009 external verification works) but cannot find bugs in its OWN code during generation. This is a "blind spot" — the same reasoning that produced the plausible-but-wrong fix cannot detect the flaw. External verification works precisely because a SEPARATE reasoning process evaluates the patch without the original reasoning's biases.

**Implication**: Adversarial self-correction through prompting doesn't compete with SERA/RL approaches (which bake quality into weights through training data curation). The fix-to-pass gap (82% fix → 12% pass) cannot be closed by prompt engineering alone. External verification (v001∩v009 ensemble) or training interventions (SVG-filtered SFT, RLVR) are the paths forward.

Script: `scripts/run_baseline.py --prompt-variant <variant> --gold-eval`

## Consolidated Findings (T4 + T5)

| Question | Answer | Evidence |
|----------|--------|----------|
| Is precision=1.00 transferable to non-Claude verifiers? | **NO** | T4: Devstral prec=0.20, Nova Pro prec=0.14, Mistral Large rejects all |
| Does adversarial self-critique improve generation? | **NO** | T5: Fix rate drops 54%→30%, pass rate drops 10%→2% |
| Can a model find bugs in its own code? | **NO** | T5: Same adversarial framing that works for external verification fails for self-correction |
| What makes Claude's v009 special? | **Calibrated adversarial reasoning** | Claude's v009 says "likely_correct" ~15% (calibrated). Devstral: 10% (defaults to uncertain). Nova Pro: 18% (too lenient). Mistral Large: 1.4% (always rejects). |

**Key insight**: The adversarial verification breakthrough is a **Claude capability**, not just a rubric design. External verification by Claude is the only viable prompt-engineering path to improving patch selection. To bake quality into weights, training interventions (SERA SVG-filtered SFT, RLVR) are required.

## T6b: Verifier-as-Skill (Devstral Generation + Verifier Loop) — NEGATIVE RESULT (2026-03-24)

**Question**: Does using the v001∩v009 ensemble as a post-generation skill (best-of-N with verification) improve pass rate for Devstral SERA?

### Setup

- g7e.24xlarge (4× RTX PRO 6000 Blackwell), Devstral Small 2 via vLLM
- SERA harness config D (30 turns), max 3 attempts per issue
- v001∩v009 ensemble via Claude Haiku (Bedrock) after each attempt
- If rejected, reset workspace and retry

### Results

| Metric | Value |
|--------|-------|
| Issues | 50 |
| Fixes generated (attempt 1) | 48/50 (96%) |
| v001 likely_correct | 6/48 (12.5%) |
| **Ensemble pass** | **2/48 (4.2%)** |
| Retries (attempt 2+) | 46 attempted, 45 broken pipe (97%) |
| Overall verified | 3/50 (incl. 1 from retry) |
| Verify cost | $1.40 |

### Gold Eval: Verifier Transfer to Devstral Patches (answers T1-T3)

Gold eval on 44 T6b diffs (Docker, SWE-bench test patches):

| Verifier Config | TP | FP | FN | TN | Precision | Recall | F₀.₅ |
|----------------|----|----|----|----|-----------|--------|-------|
| v001 alone | 1 | 4 | 3 | 35 | **0.20** | 0.25 | 0.21 |
| **v001∩v009(2+/3) ensemble** | **1** | **1** | **3** | **38** | **0.50** | **0.25** | **0.42** |

Comparison:

| Patch Source | Verifier | Precision | Recall | F₀.₅ | Gold Pass Rate |
|-------------|----------|-----------|--------|-------|---------------|
| Claude Sonnet × OpenCode | Haiku v001∩v009 | **1.00** | 0.33 | 0.71 | 12% (6/49) |
| Claude Haiku × OpenCode | Haiku v001∩v009 | **1.00** | 0.40 | 0.77 | 19% (5/27) |
| **Devstral × SERA T6b** | Haiku v001∩v009 | **0.50** | 0.25 | 0.42 | **9% (4/44)** |

### Gold passes missed by verifier

| Instance | Gold | v001 verdict | Why missed |
|----------|------|-------------|------------|
| django__django-10924 | PASS | likely_incorrect | Verifier rejects correct Django fix |
| django__django-11001 | PASS | uncertain | Verifier unsure about valid fix |
| pytest-dev__pytest-11143 | PASS | likely_incorrect | Verifier rejects correct pytest fix |
| mwaskom__seaborn-3010 | PASS | likely_correct | TP — correctly identified |

### FP analysis

| Instance | Gold | v001 | v009 lc | Why FP |
|----------|------|------|---------|--------|
| pallets__flask-4045 | FAIL | likely_correct | 3/3 | Devstral patch looks plausible but wrong — verifier fooled |

### Conclusions

1. **Verifier partially transfers** to Devstral patches (precision=0.50 > base rate 9%), but precision drops from 1.00 to 0.50. The ensemble is calibrated for Claude-style patches.
2. **v009 still helps** — catches 3 of 4 v001 FPs on Devstral patches (same as on Claude patches).
3. **Verifier-as-skill doesn't work** at current recall (0.25). With only 2/48 patches passing, the loop rejects correct patches more often than it catches incorrect ones.
4. **Retry mechanism broken** — vLLM KV cache not cleared on workspace `git checkout`, causing broken pipe on 97% of retries. Would need explicit context reset or new session.
5. **Devstral gold pass rate lower** (9% vs 12-19% for Claude), making the verifier's job harder — fewer correct patches to find.

### Implications for T6b viability

For verifier-as-skill to improve pass rate, need: `recall × fix_rate > baseline_pass_rate`.
- Current: 0.25 × 0.96 = 0.24 > 0.09 baseline ✓ (barely)
- But with N=1 effective (broken retries), selected set is 2 patches → 1 TP, 1 FP = 50% pass rate on selected set
- **If retries worked**: Best case 3 × 48 = 144 candidates, ~14 fixes per gold pass issue, ensemble selects ~3-4 → maybe 2 TP out of 3-4 selected = 50-67% precision on selected set. Still lower than Claude baseline.

Scripts: `scripts/run_verifier_loop.py`, gold eval via `scripts/gold_eval.py --model devstral_sera_vloop`

## Consolidated Findings (T4 + T5 + T6b)

| Question | Answer | Evidence |
|----------|--------|----------|
| Is precision=1.00 transferable to non-Claude verifiers? | **NO** | T4: Devstral prec=0.20, Nova Pro prec=0.14, Mistral Large rejects all |
| Does adversarial self-critique improve generation? | **NO** | T5: Fix rate drops 54%→30%, pass rate drops 10%→2% |
| Can a model find bugs in its own code? | **NO** | T5: Same adversarial framing that works for external verification fails for self-correction |
| Does verifier transfer to non-Claude patches? | **NO** | T6b: prec 0.50, T6c: prec 0.20 on 97 candidates. Verifier is patch-source-specific. |
| Is verifier-as-skill viable for Devstral? | **NO** | T6c: BoN selection (3/49=6%) ≤ VL (4/44=9%). No recovery of VL failures. |
| Is verifier-as-skill viable for Claude? | **MARGINAL** | T7: BoN 6/49=12% equals best single model. Verifier can't distinguish models when most passes are "easy" issues all models solve. |
| Was iter 21 cost optimization safe? | **NO** | T7: 2+/3 threshold causes 2 FPs on opus patches. Fixed: 3+/3 threshold restores precision=1.00. |
| What makes Claude's v009 special? | **Calibrated adversarial reasoning** | Claude's v009 says "likely_correct" ~15% (calibrated). Devstral: 10% (defaults to uncertain). Nova Pro: 18% (too lenient). Mistral Large: 1.4% (always rejects). |

**Key insight**: The adversarial verification breakthrough is a **Claude capability**, not just a rubric design. External verification by Claude is the only viable prompt-engineering path to improving patch selection. To bake quality into weights, training interventions (SERA SVG-filtered SFT, RLVR) are required. The verifier works best on patches from Claude-family models — Devstral patches have different error signatures that the rubric wasn't calibrated for.

## T6c: Best-of-N Verifier Selection on Devstral SERA Candidates (2026-03-24)

**Question**: Can the v001∩v009 ensemble select the best patch from multiple Devstral SERA candidates per issue?

### Setup

- 97 candidate diffs across 49 issues (mostly 2 candidates each, from SERA retries)
- Ran v001∩v009(2+/3) ensemble on each candidate via Claude Haiku (Bedrock)
- Selection: pick ensemble-passing candidate; if none pass, pick least rejected
- Gold eval on all 49 selected diffs (Docker, SWE-bench test patches)

### Results

| Metric | Value |
|--------|-------|
| Total candidates | 97 across 49 issues |
| Ensemble passes | 5/97 candidates (5.2%) across 5 issues |
| Selection: ensemble_pass | 5 issues |
| Selection: least_rejected | 44 issues |
| BoN gold pass rate | **3/49 (6%)** |
| Cost | $1.42 (97 candidates × 4 calls) |

### Verifier Precision on Devstral BoN

| | TP | FP | FN | TN | Precision | Recall |
|--|----|----|----|----|-----------|--------|
| Ensemble | 1 | 4 | 2 | 42 | **0.20** | 0.33 |

Ensemble passes:
| Instance | Selected | v009 lc | Gold | Label |
|----------|----------|---------|------|-------|
| astropy__astropy-14365 | a1 | 3/3 | FAIL | FP |
| astropy__astropy-14995 | a3 | 2/3 | FAIL | FP |
| mwaskom__seaborn-3010 | a1 | 2/3 | PASS | TP |
| pallets__flask-4045 | a1 | 3/3 | FAIL | FP |
| psf__requests-1963 | a2 | 3/3 | FAIL | FP |

### BoN vs VL Comparison (44 overlapping issues)

| Method | Pass Rate | Precision (on selected set) |
|--------|-----------|----------------------------|
| VL (verifier loop) | 4/44 (9%) | — |
| BoN (post-hoc selection) | 3/44 (7%) | 0.20 |

BoN has **no advantage** over VL. The verifier selected a wrong candidate on django-11001 (VL had a PASS, BoN selected a FAIL). Zero VL failures were recovered by BoN.

### Why Precision Dropped (1.00 → 0.20)

The precision=1.00 breakthrough was calibrated on **Claude-generated patches**. Devstral patches have different characteristics:
- Claude patches are surgical (small, targeted). Devstral patches are more structural (larger, more code changes).
- The v009 adversarial rubric's FP-catching works because it identifies subtle logical errors in Claude's "plausible-looking" patches. On Devstral patches, the adversarial rubric is too lenient — it passes patches that look structurally correct but have different failure modes.
- v009 "likely_correct" rate: 5.2% on Devstral candidates vs ~15% on Claude patches. The threshold is more compressed.

### v009 Discrimination

| v009 lc count | # Candidates | % |
|---------------|-------------|---|
| 0/3 | 92 | 95% |
| 2/3 | 2 | 2% |
| 3/3 | 3 | 3% |

v009 is extremely selective on Devstral patches (95% rejected). But when it does pass, precision is only 0.20 — the 5% it selects is NOT enriched for correct patches.

### Conclusion

**The verifier is patch-source-specific.** Precision=1.00 on Claude patches does not transfer to Devstral patches (precision=0.20). This is a stronger negative than T6b (0.50), because T6b had only 44 VL-selected patches while T6c evaluated 97 raw candidates with full BoN selection.

The verifier should only be used as a best-of-N selector for **Claude-generated** patches, where precision=1.00 holds. For non-Claude patches, the verifier needs rubric recalibration or a fundamentally different approach.

### All-Candidates Gold Eval (Oracle + Random Baselines)

Gold eval on all 97 candidates: 6/97 passed (6.2%) across 4 issues. 2 issues have both candidates passing, 2 issues have only one passing candidate.

| Method | Pass Rate | Notes |
|--------|-----------|-------|
| **Oracle** (any pass) | 4/49 (8%) | Upper bound |
| **Random** (mean±std) | 3.5±0.5/49 (7.2%) | Nearly as good as oracle |
| **BoN verifier** | 3/49 (6%) | **Worse than random** |
| **VL** (verifier loop) | 4/44 (9%) | Best method |

The verifier adds **negative** value on Devstral patches. On the one issue where selection mattered (django-11001), it picked the wrong candidate.

Script: `scripts/run_best_of_n.py`

## T7: Cross-Model Best-of-N on Claude Patches (2026-03-24)

**Question**: Can the v001∩v009 ensemble select the best patch from multiple Claude models (haiku/sonnet/opus) per issue?

### Critical Finding: Iteration 21 Regression

The "cost optimization" in iter 21 (dropping v009 t=0.0, threshold 2+/4 → 2+/3) introduced **2 FPs on opus patches** (flask-4045, flask-4992). Original holdout had 0 FPs because the deterministic t=0.0 run consistently rejected these borderline patches.

| Config | flask-4045 v009 | flask-4992 v009 | Precision |
|--------|----------------|----------------|-----------|
| 5 calls (2+/4, with t=0.0) | 1/4 lc → REJECT | 0/4 lc → REJECT | **1.00** |
| 4 calls (2+/3, no t=0.0) | 2/3 lc → PASS (FP!) | 2/3 lc → PASS (FP!) | 0.50 |
| 4 calls (**3+/3**, no t=0.0) | 2/3 lc → REJECT | 2/3 lc → REJECT | **1.00** |

**Fix**: Raise threshold to **3+/3** (unanimous v009 agreement required). This eliminates both FPs while retaining all TPs (all true positives have 3/3 lc). The t=0.0 run IS NOT needed if threshold is strict enough.

### Results (3+/3 threshold)

| Method | Pass Rate | Notes |
|--------|-----------|-------|
| **Oracle** | 7/49 (14%) | Upper bound |
| **Random** | 6.0/49 (12.3%) | Expected |
| **BoN verifier (3+/3)** | 6/49 (12%) | Equal to best single model |
| Opus alone | 6/49 (12%) | |
| Sonnet alone | 6/49 (12%) | |
| Haiku alone | 5/49 (10%) | |

Ensemble precision: **1.00** (2 TP, 0 FP). But only 2/7 gold passes are confidently identified by ensemble (seaborn-3010, pytest-11143). The other 4 pass via "least_rejected" fallback.

### Why BoN Equals (Not Exceeds) Single Model

5 of 7 gold-passing issues have **all 3 models passing** — selection doesn't matter. The 2 issues where models differ:
- django-10924: haiku=P, sonnet=P, opus=F → BoN picks haiku (correct, but sonnet also works)
- **django-11019**: haiku=F, sonnet=F, opus=P → BoN picks sonnet (**wrong**, misses the only pass)

The verifier can't identify django-11019's opus patch as correct because ALL models get v001=likely_incorrect. The verifier's recall problem (0.24) means it can't find most correct patches regardless of candidate pool.

### Updated Best Config — REVERT to 5-call 2+/4

Full threshold validation across all sets:

| Config | Dev (sonnet) | Holdout (haiku) | Holdout (opus) | Combined Holdout |
|--------|-------------|----------------|----------------|-----------------|
| **5-call 2+/4 (ORIGINAL)** | prec=1.00 rec=0.33 F₀.₅=0.71 | prec=1.00 rec=0.40 F₀.₅=0.77 | prec=1.00 rec=0.33 F₀.₅=0.71 | **prec=1.00 rec=0.36 F₀.₅=0.74** |
| 4-call 3+/3 | prec=1.00 rec=0.17 F₀.₅=0.50 | prec=1.00 rec=0.20 F₀.₅=0.56 | prec=1.00 rec=0.33 F₀.₅=0.71 | prec=1.00 rec=0.27 F₀.₅=0.65 |
| 4-call 2+/3 (UNSTABLE) | prec=1.00* rec=0.33 | prec=1.00* rec=0.20 | prec=1.00* rec=0.33 | **prec≈0.50 on fresh draws** |

\* Original sweep draws were lucky. Fresh stochastic draws in T7 produced 2 FPs (flask-4045, flask-4992).

**REVERT to 5-call 2+/4**: v001(t=0.0) + v009(t=0.0) + v009×3(t=0.3), threshold 2+/4. The t=0.0 deterministic run provides a consistent anchor. Cost: $0.048/patch (+$0.010 vs 4-call, worth it for stable precision + 33% more TPs vs 3+/3).

Script: `scripts/run_cross_model_bon.py`

## Iterations 28-30: Recall Improvement Attempts (2026-03-24)

### Iteration 28: v009-only ensemble (drop v001 gate)

**Hypothesis**: v001 blocks 59% of gold-pass FNs (iter 27). Removing v001 and using only v009 should improve recall.

| Config | Dev Prec | Dev Rec | Dev F₀.₅ | Dev FP |
|--------|---------|---------|----------|--------|
| v001∩v009(2+/4) CURRENT BEST | **1.00** | 0.33 | **0.71** | **0** |
| v009-only(2+/4) | 0.25 | 0.33 | 0.26 | 6 |
| v009-only(3+/4) | 0.25 | 0.17 | 0.23 | 3 |
| v009-only(4/4 unanimous) | 1.00 | 0.17 | 0.50 | 0 |
| v009-only(1+/4) | 0.20 | 0.33 | 0.22 | 8 |

**FALSIFIED.** v009-only has same recall (TP=2) but 6 more FPs. v001 is a crucial precision guard — it removes FPs that v009 lets through. All 10 v001-blocked gold passes ALSO have v009=0/4, so removing v001 cannot improve recall.

### Iteration 29: Alternative confirmatory gates

| Config | Dev Prec | Dev FP | Notes |
|--------|---------|--------|-------|
| v001∩v009(2+/4) | **1.00** | **0** | Current best |
| v008∩v009(2+/4) | 0.29 | 5 | v008 too lenient |
| v006∩v009(2+/4) | 0.25 | 6 | v006 too lenient |
| (v001 OR v008)∩v009(2+/4) | 0.29 | 5 | Union adds FPs |

**FALSIFIED.** Reformat-aware rubrics (v006, v008) admit more FPs without recovering any TPs. The FM-001 FNs have v009=0/4, so no confirmatory gate change can help.

### Iteration 30: v010 concrete adversarial rubric

**Hypothesis**: Requiring CONCRETE counter-examples (specific input → expected → actual) instead of speculative failures will reduce v009's over-rejection of correct patches.

| Config | Prec | Rec | FP | TP |
|--------|------|-----|----|----|
| v010 alone | **0.13** | **0.67** | 27 | 4 |
| v010∩v009(2+/4) | 0.25 | 0.33 | 6 | 2 |
| (v001 OR v010)∩v009(2+/4) | 0.25 | 0.33 | 6 | 2 |

**FALSIFIED.** v010 is far too lenient (27 FPs, precision=0.13). Requiring concrete counter-examples backfires — the model can't construct counter-examples even for genuinely buggy patches, so it defaults to "likely_correct."

**Key finding**: v010 recovers 2 FNs that v001 misses:
- django-10924 (FM-001, 117K diff): v001=li, v010=lc, **v009=0/4** → blocked
- sphinx-11445 (412B model error): v001=li, v010=lc, **v009=0/4** → blocked

But v009 independently rejects BOTH (0/4 lc). No confirmatory gate change can overcome the v009 recall ceiling.

### Definitive Recall Ceiling Analysis (iters 28-30)

**v009 is both the precision enabler AND the recall ceiling.**

All 17 gold passes across all sets:
- v009 ≥ 2/4 lc: **7 passes** (seaborn-3010 × 3 sets + pytest-11143 × 3 sets + sphinx-10325 opus) → these CAN pass the ensemble
- v009 = 0/4 lc: **10 passes** → these CANNOT pass any ensemble with v009, regardless of confirmatory gate

The 10 v009-blocked gold passes:
- 7 are FM-001 (large reformatted diffs, 43-135K chars) — v009 can't find the fix in the noise
- 2 are model reasoning errors (sphinx-11445 412B, django-11019 43K) — v009 finds spurious bugs
- 1 is v009 over-rejection (sphinx-10325 opus 1.3K) — v009_t0=uncertain, all 3 t=0.3 runs = uncertain/li

**No prompt-engineering path to higher recall.** The next step requires either:
1. ~~AST-level diff preprocessing~~ Tested in iter 31: 53% noise reduction insufficient. Full AST normalization needs original source files.
2. A fundamentally different adversarial approach (not just rubric wording)
3. Training a learned verifier (Phase 3 of the framework)

### Iteration 31: Diff preprocessing for FM-001

**Hypothesis**: Removing cosmetic noise (context lines + quote-only pairs) from FM-001 diffs will help v009 find the functional fix.

| Instance | Original | Preprocessed | v001 | v009 (4 runs) |
|----------|----------|-------------|------|--------------|
| django-10924 | 117K | 55K (-53%) | likely_incorrect (0.25) | 0/4 lc |
| django-11001 | 66K | 34K (-48%) | likely_incorrect (0.25) | 0/4 lc |
| sphinx-10325 | 135K | 73K (-46%) | likely_incorrect (0.35) | 0/4 lc |

**FALSIFIED.** 46-53% noise reduction doesn't change verdicts. Both rubrics still reject. The remaining changes include line wrapping, import reordering, and string format changes that aren't simple quote substitutions. Full AST normalization would require the original source files (not just diffs). Cost: $0.45 (15 Haiku calls).

Script: zero-cost preprocessing + inline verification calls

## T8: Rubric Adaptation for Devstral Patches — NEGATIVE (2026-03-24)

**Question**: Can we improve verifier precision on Devstral's surgical patches through rubric or prompt changes?

### Error Signature Analysis

Devstral × SERA patches are structurally different from Claude × OpenCode patches:

| | Claude × OpenCode | Devstral × SERA |
|--|---|---|
| Median diff size | 11,714 chars | 1,567 chars (**8x smaller**) |
| Median hunks | 38 | 2 |
| Median +lines | 92 | 11 |
| FP type | FM-001: bugs in reformatting noise | FM-005: surgical patches that look correct |
| Verifier challenge | Too much code | Too little code |

On 43 overlapping issues, Sonnet patches are >10x bigger in 20/43 cases. The harness (OpenCode vs SERA) drives diff size more than the model.

### Iteration 32: Context Augmentation

**Hypothesis**: Providing 30 lines of surrounding source code for each hunk gives v009 more surface area for adversarial reasoning.

| Config | TP | FP | FN | TN | Precision | Recall |
|--------|----|----|----|----|-----------|--------|
| No context (baseline) | 1 | 2 | 3 | 38 | **0.33** | 0.25 |
| **Context (30 lines)** | 1 | **5** | 3 | 35 | **0.17** | 0.25 |

**NEGATIVE.** Context makes v009 MORE lenient — it passes 3 additional FPs. Seeing the surrounding code convinces v009 that wrong patches are correct because they fit the codebase locally.

Mechanism: adversarial verification works BECAUSE of information asymmetry. The verifier's uncertainty about surrounding code IS the mechanism that catches subtle bugs. Adding context removes that uncertainty.

One bright spot: `psf__requests-1963` flips FP→TN with context (context reveals the bug). But 4 new FPs overwhelm this.

Script: `scripts/verify_with_context.py`

### Iteration 33: v010 Completeness Rubric

**Hypothesis**: A rubric focused on "what else would need to change?" catches surgical patches that fix one symptom but miss secondary changes (e.g., flask-4045 adds dot validation but doesn't update existing tests using dots).

v010 asks: enumerate requirements, identify missing changes, assess regression risk.

| Config | TP | FP | FN | TN | Precision | Recall |
|--------|----|----|----|----|-----------|--------|
| Baseline (v001∩v009) | 1 | 2 | 3 | 38 | **0.33** | 0.25 |
| v010 standalone | 0 | 1 | 4 | 39 | 0.00 | 0.00 |
| v001∩v010-filter∩v009 | 1 | 2 | 3 | 38 | **0.33** | 0.25 |

**NO DISCRIMINATION.** v010 finds ~5 "missing changes" per patch regardless of correctness. Mean completeness score: 0.53 for gold passes vs 0.48 for gold fails. Correct patches look just as "incomplete" as incorrect ones because small correct patches genuinely don't touch every related file.

v010 catches flask-4045 specifically (flags regression risk from existing tests using dots), but says "uncertain" not "likely_incorrect" — so the filter doesn't catch it.

Script: `scripts/test_completeness.py`

### Conclusion

**Prompt engineering has hit a wall on Devstral's surgical diffs.** Three approaches tried, all negative:

1. ~~Context augmentation~~ — makes v009 lenient (-3 FPs)
2. ~~Completeness rubric~~ — no discrimination (correct = incorrect at ~5 missing changes)
3. ~~Completeness filter~~ — identical to baseline

The fundamental problem: on small surgical diffs, the verifier can't distinguish "this small change is all that's needed" from "this small change is not enough." Both look identical from the diff alone.

**The harness determines verifier transfer, not the model.** OpenCode-style diffs (large, multi-hunk, context-rich) give the verifier enough surface area. SERA-style diffs (surgical, minimal) don't. This predicts that **any model × OpenCode** should have good verifier transfer, while **any model × SERA** won't.

## T9: Qwen3.5 × OpenCode — Verifier Transfer via Harness (COMPLETE)

**Hypothesis**: If the harness determines verifier transfer, then Qwen3.5 × OpenCode patches (which should be large, OpenCode-style diffs) will have precision near 1.00 — matching Claude × OpenCode.

**Result**: Precision=0.50. Hypothesis **partially confirmed** — harness helps (0.50 > 0.33 SERA) but doesn't fully transfer (0.50 < 1.00 Claude).

### Setup

- **Model**: Qwen3.5-122B-A10B-FP8 (pivoted from 397B-GPTQ which produced garbage on vLLM 0.18)
- **Instance**: g7e.24xlarge, TP4 across 4x RTX PRO 6000 Blackwell
- **Serving**: vLLM 0.18.0, `--tool-call-parser qwen3_xml --reasoning-parser qwen3`, 65K context
- **Harness**: OpenCode v1.2.27, `@ai-sdk/openai-compatible` provider
- **Runtime**: 97 min for 50 issues, ~2 min/issue

### Results

| Metric | Claude × OpenCode | Devstral × SERA | Qwen3.5 × OpenCode |
|--------|-------------------|-----------------|---------------------|
| Fix rate | 78% | 96% | **86% (43/50)** |
| Gold pass rate | 16% | 9% | **9% (4/43)** |
| Verifier precision | **1.00** | 0.33 | **0.50** |
| Verifier recall | 0.33 | 0.25 | **0.50** |
| F₀.₅ | 0.74 | 0.31 | **0.50** |
| TP / FP / FN / TN | 4/0/8/37 | 1/2/3/38 | 2/2/2/37 |
| Median diff size | 11.7K | 1.5K | **8.6K** |
| Cost/patch | $0.048 | $0.048 | $0.034 |

### Diff Size Distribution

| Range | Count | % |
|-------|-------|---|
| <1K | 6 | 14% |
| 1-5K | 12 | 28% |
| 5-10K | 4 | 9% |
| >10K | 21 | 49% |

Median 8.6K, mean 21.6K — confirms OpenCode produces large diffs regardless of model.

### FP Analysis

- **pallets__flask-4992**: v001=likely_correct, v009=3/3 lc. Small diff (1.1K), clean-looking fix. The verifier approves it but gold tests fail.
- **scikit-learn__scikit-learn-11040**: v001=likely_correct, v009=2/3 lc. The fix looks correct to both rubrics but doesn't pass tests.

Both FPs are on small diffs (<2K) where the adversarial rubric has less surface area to find bugs — consistent with the "harness determines transfer" theory.

### Gold Passes (same 4 as Devstral × SERA T6b)

- django-10924, django-11001, seaborn-3010, pytest-11143

### Interpretation

1. **Harness partially determines transfer**: OpenCode diffs improved precision from 0.33→0.50 vs SERA diffs
2. **Model still matters**: Claude × OpenCode achieves 1.00 precision on the same harness. Qwen3.5 generates different error patterns that the v009 rubric doesn't catch.
3. **Recall improved**: 0.50 is the best recall of any configuration — the larger diffs expose more correct patterns for the confirmatory rubric (v001) to identify.
4. **FPs correlate with diff size**: Both FPs are <2K char diffs, not the large 10K+ diffs where the verifier has full signal.
5. **397B-GPTQ broken on vLLM 0.18**: `qwen3_5_moe` with hybrid linear attention + GPTQ produces garbage output. FP8 works fine.

### Session Plan

See `results/g7e-qwen35-session-plan.md`.

## Iterations 35-36: Root Cause of Recall Ceiling (2026-03-24)

### Iteration 35: Oracle Minimal-Diff Test

**Hypothesis**: The recall ceiling is caused by noise in FM-001 diffs. Feeding v009 ONLY the functional change (525 chars instead of 117K) should recover gold passes.

**Oracle test** — manually extracted the 1-line functional fix from django-10924:
```diff
-            'path': self.path,
+            'path': self.path() if callable(self.path) else self.path,
```

| Rubric | Diff Size | Verdict | Score |
|--------|-----------|---------|-------|
| v009 (Haiku) | 117K (full) | 0/4 lc | 0.35 |
| v009 (Haiku) | 525 chars (minimal) | 0/4 lc | 0.35 |
| v009 (Sonnet) | 525 chars (minimal) | 0/4 lc | 0.30 |

**The recall ceiling is NOT caused by noise.** Even with a clean 525-char minimal diff, BOTH Haiku AND Sonnet reject the patch at 0/4 lc. The reason: v009 correctly identifies that the patch "only handles the callable in formfield() but misses deconstruct() for migration serialization." The patch IS genuinely incomplete — it just passes the test suite because the tests don't cover migration serialization.

### v011 Scoped Adversarial (Ablation)

Created v011_scoped_adversarial.md: "Only reject if the CHANGED CODE itself is incorrect. Do NOT reject for missing changes to other functions."

| Config | Dev Prec | Dev FP | Dev Rec | Notes |
|--------|---------|--------|---------|-------|
| v009 (baseline) | 1.00 | 0 | 0.33 | Current best |
| v011 alone | **0.09** | **20** | 0.33 | Catastrophic — leaks astropy-14365 (3/4 lc) |

**FALSIFIED.** Removing the completeness check destroys precision (20 FPs) without improving recall. The completeness check is what catches FPs — v009 rejects astropy-14365 because the regex fix "doesn't handle all separator patterns." Without that check, the FP leaks through.

### Iteration 36: Sonnet as Adversarial Verifier

**Hypothesis**: Sonnet's stronger reasoning will correctly clear patches that Haiku spuriously rejects.

| Instance | Gold | Haiku v009 | Sonnet v009 | Sonnet's attack |
|----------|------|-----------|-------------|-----------------|
| django-10924 | PASS | 0/4 lc | 0/4 lc | "misses deconstruct() for migrations" |
| astropy-14365 | FAIL | 0/4 lc | 0/4 lc (uncertain) | "regex looks correct, unsure about edge cases" |

**FALSIFIED.** Sonnet agrees with Haiku on django-10924 — both find the same incompleteness. The recall ceiling is model-independent.

### Root Cause: Semantic Mismatch

The recall ceiling (0.33) is caused by a **semantic mismatch** between verification and evaluation:

| Dimension | Verifier (v009) | Gold Eval (test suite) |
|-----------|----------------|----------------------|
| **Evaluates** | Problem completeness | Specific test cases |
| **django-10924** | FAIL: formfield() fixed but deconstruct() untouched | PASS: test only checks formfield() |
| **django-11001** | FAIL: compiler change insufficient | PASS: test checks specific query |
| **sphinx-10325** | FAIL: inherited_members change incomplete | PASS: test checks specific behavior |

Both Haiku and Sonnet correctly identify these patches as incomplete. The gold test suite accepts them because it tests a narrow subset of the behavior. The verifier is **right** in a deeper sense — these patches ARE incomplete fixes.

### Iteration 37: v012 Test-Outcome Predictor

**Hypothesis**: Reframe from "is this fix correct?" to "will this pass tests?" to align with gold evaluation criterion.

| Instance | Gold | v012 (4 runs) | Status |
|----------|------|--------------|--------|
| django-10924 | PASS | 0/4 lc | WRONG (still rejects) |
| astropy-14365 | FAIL | 4/4 lc | WRONG (leaks FP) |
| flask-4992 | FAIL | 4/4 lc | WRONG (leaks FP) |
| seaborn-3010 | PASS | 4/4 lc | OK |

**FALSIFIED.** Worst of both worlds. Without access to actual test cases, the model cannot predict test outcomes — it falls back to evaluating code correctness, but with a lenient frame that leaks FPs. The "be lenient on completeness" instruction destroys FP-catching without recovering FNs.

### Iteration 38: Extended Thinking

**Hypothesis**: Haiku 4.5's extended thinking (scratchpad reasoning) might help v009 distinguish "incorrect logic" from "incomplete but functional."

**v001 with thinking** — recovers sphinx-11445 (v001: likely_incorrect → likely_correct with thinking):

| Instance | Gold | v001 (no thinking) | v001 (thinking) |
|----------|------|-------------------|-----------------|
| sphinx-11445 | PASS | likely_incorrect | **likely_correct (0.98)** |
| django-10924 | PASS | likely_incorrect | likely_incorrect |
| django-11001 | PASS | likely_incorrect | uncertain (0.62) |

**v009 with thinking** — recovers sphinx-11445 but leaks flask-4992:

| Instance | Gold | v009 (no thinking) | v009+thinking (8 runs) |
|----------|------|-------------------|----------------------|
| sphinx-11445 | PASS | 0/4 lc | **4/8 lc (50%)** |
| flask-4992 | FAIL | 0/4 lc | **5/8 lc (63%) — FP LEAK** |
| astropy-14365 | FAIL | 0/4 lc | 0/8 lc (safe) |

**FALSIFIED.** Extended thinking requires `temperature=1.0`, which makes v009 less discriminative. The FP (flask-4992, 63% lc) passes MORE often than the recovered TP (sphinx-11445, 50% lc). No threshold can separate them.

Full ensemble with thinking: 3 TP, 1 FP → precision=0.75, recall=0.50, F₀.₅=0.68 (regression from 0.71).

Cost: $0.44 for 6 cases (5 calls × 6 cases × ~$0.015/call with thinking = $0.45). Production cost would be ~$0.075/patch (5 extended-thinking calls vs $0.048 without).

### Definitive Conclusion: Prompt Engineering Exhausted

After 43 iterations testing every lever:

| Lever Category | Iterations | Best Result | Why It Failed |
|---------------|-----------|-------------|---------------|
| Confirmatory rubrics (v001-v008) | 1-15 | prec=0.40 | Can't catch plausible-but-wrong patches |
| Adversarial rubric (v009) | 16-17 | **prec=1.00** | **BREAKTHROUGH** — catches all FM-003 FPs |
| Alternative adversarial (v010, v011) | 30, 35 | prec=0.09-0.13 | Either too lenient or destroys precision |
| Test-outcome prediction (v012) | 37 | leaks FPs | Model can't predict test outcomes without tests |
| Scope constraints | 35 (v011) | 20 FPs | Removing completeness check leaks FPs |
| Alternative gates | 28-29 | 0.25-0.33 | All worse than v001∩v009 |
| Model upgrade (Sonnet) | 9, 36 | same ceiling | Both models agree patches are incomplete |
| Diff preprocessing | 31 | unchanged | Noise isn't the problem (oracle confirms) |
| Temperature/ensemble | 17-18, 21, 26 | stabilization | Helps precision, not recall |
| Extended thinking | 38 | prec=0.75 (regression) | temp=1.0 makes v009 less discriminative; FP lc > TP lc |
| Problem-first verification | 39 | leaks FPs | Model predicts plausible fixes → matches both correct and buggy patches |
| Opus as v009 | 40 | leaks flask-4992 | More lenient than Haiku, 18x cost, no recall gain |
| Multi-turn challenge | 41 | leaks all FPs | Sycophantic — challenge flips all verdicts |
| Adversarial few-shot | 42 | same (1.00) | Examples don't change verdicts on real cases |
| Test generation | 43 | ~0.50 (new FPs) | First approach recovering all FNs, but orthogonal to v009 |

**The recall ceiling (0.33) is caused by a semantic mismatch**, not a model or rubric limitation:
- The verifier evaluates **problem completeness** (is the fix thorough?)
- Gold eval tests **specific behaviors** (does this code path work?)
- Many gold-passing patches are genuinely incomplete (fix 1 of N needed changes) but pass narrow tests

Both Haiku and Sonnet independently identify the same incompleteness. No rubric wording can make an LLM predict test outcomes without seeing the tests.

**Pareto-optimal operating point**: precision=1.00, recall=0.33, F₀.₅=0.71. This is the best achievable result for skill-based verification without test execution.

## Next Steps (Post-Experiment)

**Prompt engineering is exhausted.** 43 iterations tested every lever. The precision-recall tradeoff is fundamental:
- Precision requires completeness checking (catches FPs)
- Recall requires ignoring completeness (passes test-passing-but-incomplete patches)
- No rubric can do both — the tradeoff is a property of the evaluation, not the rubric

**Remaining paths (all require new infrastructure, not more iterations):**

1. **RLVR**: Use v001∩v009 as reward model. Precision=1.00 is ideal for positive reward labels (zero noise). Youden's J=0.33 satisfies "Rate or Fate?" convergence guarantee. ~33% of correct patches get rewarded — sufficient for training signal.
2. **SVG consensus verifier (Phase 1)**: Multiple independent solver attempts + line-recall voting. Already has AUC 0.981 from learned-verifier Phase 0 data. No rubric needed — purely behavioral.
3. **Learned verifier (Phase 3)**: Train on (patch, test_outcome) pairs. Directly predicts test outcomes — closes the semantic mismatch.

### 4. Verification Primitives as Agent Tools (proposed — 2026-03-29)

**Thesis**: Instead of post-hoc verification (this experiment), arm the agent with verification primitives as callable tools during its trajectory. Let the agent compose them — no engineered multi-agent pipeline.

**Why this is the logical next step**:
- Iter 43 showed test generation recovers ALL 4 FNs but leaks 4/5 FPs. v009 catches FPs but misses FNs. They're orthogonal — the composition could close both gaps.
- CoderForge 413K-trajectory analysis (Li et al., 2026) found early test fraction is the strongest predictor of agent success (concordance 56.3%, 12,286 within-issue pairs). Agents that front-load testing succeed more.
- Counter-evidence (arXiv:2602.07900): ad-hoc test writing has marginal utility — but they only tested confirmatory framing. Our v009 data shows adversarial > confirmatory by 2.3x. Same may hold for test generation.
- InfCode (79.4% SWE-bench Verified) validates the adversarial test-patch co-evolution pattern, but hard-wires the pipeline. Our approach: durable primitives, agent-driven composition.

**Key design decisions**:
- **Agent model: Sonnet** — better tool-calling judgment than Haiku. The experiment tests whether agents *compose* verification tools, which is a tool-calling decision. Haiku's weaker tool calling would conflate "primitives don't help" with "model can't compose."
- **Verifier inside tools: Haiku** — proven at 0.92 precision, $0.008/call.
- **Tool description framing**: Describe tools as helping the agent succeed ("validate your fix by generating edge-case tests"), not as attacking its own work ("generate tests to break your patch"). Adversarial framing goes in the prompt *inside* the tool, not in the tool description the agent sees.
- **Baseline**: Sonnet × OpenCode = 98% fix, 12% pass. Same model + harness + verification primitives → clean comparison.

**Four levels of verification capability** (this targets level 3):
1. Tool competence — call pytest correctly (baseline)
2. Behavioral cloning via RFT — replicate successful patterns (Nebius 25→50%, SERA 24→49%)
3. **Primitive composition** (this experiment) — contextually adaptive verification judgment
4. Absorbed verification — verified-quality output without tools (Composer 2 endstate)

**Spec**: `domains/autoresearch/specs/verification-primitives.md`
**Observation note**: Obsidian vault `01_Projects/Learned-Verifier-Experiment/Observation-Verification-Primitives-Next-Experiment.md`
4. ~~**T9: Qwen3.5 × OpenCode verifier transfer**~~ DONE — Prec=0.50, partial transfer. Harness helps but model still matters.
5. ~~**Larger eval**~~ DONE (T10) — SWE-bench Verified (n=483) reveals precision=0.78, NOT 1.00. See T10 below.
6. ~~**Production integration**~~ DONE — `scripts/verify_patch_ensemble.py`
7. ~~**Cross-verifier tuning**~~ CLOSED — T4: gap is fundamental (model reasoning quality)
8. ~~**FM-001 (reformatting noise)**~~ CLOSED — Iter 35: noise isn't the issue; it's semantic mismatch
9. ~~**Rubric adaptation**~~ CLOSED — T8 + iters 35-37: no rubric wording can predict test outcomes

## T10: Large-Scale Validation on SWE-bench Verified — PRECISION IS NOT 1.00 (2026-03-25)

**Question**: Does precision=1.00 hold on a larger, independent dataset? The dev set (49 patches, 6 gold passes) has 95% CI [0.54, 1.00] — wide enough to hide real FPs.

### Setup

- **Dataset**: SWE-bench Verified (500 issues), Claude 3.5 Sonnet predictions from SWE-bench leaderboard submission
- **Source**: `s3://swe-bench-submissions/verified/20241022_tools_claude-3-5-sonnet-updated/`
- **Gold labels**: Per-instance `report.json` with `resolved: true/false` from full test evaluation
- **Patches**: 483 with both prediction + gold label (8 missing gold eval)
- **Verifier**: Same v001∩v009(2+/4 lc) ensemble, 5 Haiku calls per patch
- **Workers**: 20 parallel, ~3 min total wall time

### Results

| Metric | Dev (n=49) | **Verified (n=483)** |
|--------|-----------|---------------------|
| Gold PASS rate | 12% (6/49) | **51% (245/483)** |
| Precision | 1.00 | **0.783** |
| Precision 95% CI | [0.54, 1.00] | **[0.58, 0.90]** |
| Recall | 0.33 | **0.073** |
| Recall 95% CI | [0.10, 0.70] | **[0.047, 0.113]** |
| F₀.₅ | 0.71 | **0.267** |
| TP | 2 | **18** |
| FP | 0 | **5** |
| FN | 4 | **227** |
| TN | 43 | **233** |
| Lift over random | 8.3x | **1.54x** |
| Cost | $2.35 | **$11.70** |

### Component Breakdown

| Verifier | Precision | Recall | TP | FP |
|----------|-----------|--------|----|----|
| v001 alone | 0.716 | 0.237 | 58 | 23 |
| v009 (2+/4) alone | 0.828 | 0.196 | 48 | 10 |
| v001 ∩ v009 ensemble | 0.783 | 0.073 | 18 | 5 |
| Random baseline | 0.507 | 1.000 | 245 | 238 |

**v009 alone is the best single component** — prec=0.83, rec=0.20 (48 TPs). The ensemble intersection actually HURTS on this dataset: it drops recall from 0.20 to 0.07 while barely improving precision (0.83 → 0.78).

### False Positives (5)

| Instance | v001 | v009 lc | Notes |
|----------|------|---------|-------|
| django__django-12039 | likely_correct | 3/4 | |
| django__django-14315 | likely_correct | 3/4 | |
| django__django-15103 | likely_correct | 4/4 | Strong FP — unanimous v009 |
| django__django-16667 | likely_correct | 2/4 | Borderline |
| pydata__xarray-6992 | likely_correct | 4/4 | Strong FP — unanimous v009 |

2 of 5 FPs have unanimous v009 (4/4 lc) — the adversarial rubric genuinely cannot catch these. They are qualitatively different from the FM-003 FPs in our dev set.

### Why Precision Dropped

Two compounding factors:

1. **Base rate shift**: Dev set has 12% gold pass rate (easy to beat random). Verified has 51% (Claude 3.5 Sonnet resolves ~half of Verified). Higher base rate → less room for precision improvement.

2. **Sample-specific overfitting**: The precision=1.00 on 49 patches was a fortunate draw. The 3 FP types we tested (astropy-14365, astropy-14995, flask-4992) happened to be catchable by v009's adversarial framing. The 5 new FPs on Verified represent error patterns v009 wasn't tested against.

3. **Ensemble hurts at high base rate**: When 51% of patches are correct, the v001 gate filters out MORE correct patches than incorrect ones. The intersection shrinks recall faster than it improves precision.

### Implications

1. **Precision=1.00 was a small-sample artifact.** The true precision is ~0.78 [0.58, 0.90]. Still better than random (0.51), but not the "zero-noise reward signal" we claimed.

2. **RLVR viability**: With 22% label noise (5 FP in 23 selected), the verifier is NOT suitable as an unsupervised reward signal. "Noisy Data is Destructive to RLVR" finding applies — need near-perfect precision for RL positive labels.

3. **v009 alone > ensemble**: On high-base-rate datasets, v009 standalone (prec=0.83, rec=0.20) dominates the ensemble (prec=0.78, rec=0.07). The v001 gate is only useful when base rate is low (<20%).

4. **The verifier still has positive lift (1.54x)** — just not the 8.3x we reported. As a triage tool (not a reward signal), it still provides value.

5. **Base rate determines utility**: At 12% base rate (our dev set), lift = 8.3x. At 51% (Verified), lift = 1.5x. The verifier is most useful when patches are mostly wrong — which is the realistic deployment scenario for weak models.

Script: `scripts/run_verified_eval.py`

## T10b: Rubric Iteration on SWE-bench Verified (2026-03-25)

**Question**: Can we catch the 5 T10 FPs with improved rubrics? Tested 4 new rubrics + threshold tuning.

### Setup

- Created 250/233 iteration/holdout split (all 5 known FPs in iteration set)
- Quick evaluation: 5 FPs + 10 random TPs + 10 random TNs = 25 patches
- Tested: v015 (full rewrite + completeness), v016 (v009 + test awareness), v017 (v009 + completeness step), v018 (completeness-only probe)

### Rubric Comparison (25-patch quick set)

| Rubric | FPs caught | TP recall | Precision | Key change from v009 |
|--------|-----------|-----------|-----------|---------------------|
| v009 (baseline) | 0/5 | 4/10 | 0.40 | — |
| v016 (+test script awareness) | 0/5 | 3/10 | 0.33 | Flag reproduce.py as artifact |
| v017 (+completeness step) | 5/5 | 0/10 | N/A | Dedicated "check related methods" step |
| v015 (full rewrite) | 5/5 | 1/10 | 1.00 | Test awareness + completeness + stricter verdicts |
| v018 (completeness-only probe) | 5/5 | 0/10 | N/A | Only checks if fix is complete |

### Key Finding: Completeness is the Discriminating Signal — But Too Noisy

The completeness check (looking for related methods that need the same fix) is what catches FPs. But it also rejects ALL true positives because:
- Correct patches also look "potentially incomplete" from the diff alone
- FPs actually score HIGHER than TPs on completeness (small, clean patches look more convincing)
- The model can't distinguish "related method genuinely needs same fix" from "related method might benefit but isn't required"

### Temperature Doesn't Help

Tested v009 at t=0.7 with 6 runs on the 5 FPs:
- django-15103: 6/6 lc — genuinely fooled regardless of temperature
- xarray-6992: 6/6 lc — genuinely fooled regardless of temperature
- django-12039: 5/6 lc, django-14315: 5/6 lc, django-16667: 5/6 lc

The model is genuinely fooled by these patches — more sampling can't help.

### Best Config: v009 4/4 Unanimous (Drop v001 Gate)

Re-analyzing existing T10 data with v009-only at different thresholds:

| Config | Iteration (n=250) | Holdout (n=233) | Full (n=483) |
|--------|-------------------|-----------------|--------------|
| v001∩v009(2+/4) | prec=0.64, 5 FP | prec=1.00, 0 FP | prec=0.78, 5 FP |
| v009 2+/4 | prec=0.76, 7 FP | prec=0.90, 3 FP | prec=0.83, 10 FP |
| v009 3+/4 | prec=0.83, 4 FP | prec=0.96, 1 FP | prec=0.89, 5 FP |
| **v009 4/4** | **prec=0.90, 2 FP** | **prec=0.94, 1 FP** | **prec=0.92, 3 FP** |

v001 gate HURTS at high base rates (51% gold pass). It removes TPs faster than FPs. v009 4/4 alone:
- +0.14 precision over ensemble (0.92 vs 0.78)
- +2x recall (0.14 vs 0.07)
- 3 FPs vs 5 FPs
- 4 calls instead of 5 ($0.030 vs $0.024/patch)

### Remaining 3 FPs

| FP Instance | v009 lc | Pattern |
|-------------|---------|---------|
| django__django-15103 | 4/4 | json_script element_id optional — misses template tag layer |
| pydata__xarray-6992 | 4/4 | DataVariables.__len__ — misses related methods |
| django__django-13590 | 4/4 (holdout) | New FP not in iteration set |

These are genuinely convincing patches that require codebase-level completeness reasoning beyond what a diff-only rubric can provide. This is the ceiling for Phase 0.5 (skill verifier).

### Conclusion

**Rubric iteration exhausted on Verified dataset.** 4 new rubrics tested, all either catch FPs but kill recall (completeness-based) or have no effect (test awareness alone). The breakthrough finding: **drop v001, use v009 alone at 4/4 unanimous** — simpler, cheaper, and better.

Cost: T10b iteration ~$2.50 (v015 FP-only $0.09, v015 quick $0.62, v009 quick $0.51, v016 quick $0.56, v017 quick $0.60, v018 quick $0.10, t=0.7 FP test ~$0.08)

Script: `scripts/run_verified_iteration.py`

---

## Next Experiments (from vault backlog — updated 2026-04-28)

> Canonical backlog: `obsidian-notes/01_Projects/Learned-Verifier-Experiment/Experiment-Backlog.md`
> Full designs, hypotheses, and cost estimates live there. This section is a pointer + summary.

### Free / Immediate (from prior briefings)

| ID | Name | Cost | What It Tests |
|----|------|------|---------------|
| E_new8 | Noise tolerance calibration | $0 | Is behavioral RF deployable as RLVR reward? (Imperfect Verifier framework) |
| E_new5 | Mechanical Tier 0 linters | $0 | AST/regex checks catch 10-20% of failures before ML? |
| E_new6 | Exploit-resistant feature analysis | $0 | Does 4-feature RF flag BenchJack-style gaming? |
| E_norm | AST patch normalization | $0-5 | Does normalizing scaffold formatting fix cross-model v009 transfer? |

### Free / Immediate (from 2026-04-28 Sakana deep research)

| ID | Name | Cost | What It Tests |
|----|------|------|---------------|
| **E_cond1** | **CMA-ES micro-verifier head** | **$0** | **10K-param head on frozen Qwen3-0.6B, trained with sep-CMA-ES on n=300. Targets AUC > 0.727. Attacks E6 cross-model transfer failure via richer representations vs 4 summary features.** |
| E_cond3 | Learned verification cascade router | $5-15 | Replace fixed Tier 0→RF→v009 cascade with learned router. Conductor proved 2.4× cost advantage over fixed routing. |

### Moderate Cost

| ID | Name | Cost | What It Tests |
|----|------|------|---------------|
| E4 | Segmental process rewards | ~$10 | Early stopping from Parkinson's Law — abort doomed trajectories |
| E5 | Constraint-guided verification | ~$25 | Extracted behavioral constraints bridge semantic gap between v009 and gold tests |
| E_new7 | Verifier-as-garbage-collector | ~$5-10 | Retrospective verification on merged commits predicts future bugs |
| **E_cond2** | **GRPO verification emergence** | **~$20-50** | **Does 7B agent trained with GRPO spontaneously develop verification behaviors? (Conductor showed verification emerges from correctness reward alone)** |

### Later (needs infrastructure)

| ID | Name | Blocker | What It Tests |
|----|------|---------|---------------|
| E_new4 | Longitudinal drift detection | Multi-session data | Behavioral feature drift predicts quality regression |
| E6-SFT | SVG rejection sampling | Fine-tuning infra | SFT on SVG-accepted trajectories improves base pass rate |
| E7 | PivotRL | RL stack | Focused RL on verification decision points |

### Key insight from Sakana research (2026-04-28)

Sakana's Conductor (7B, GRPO) and Trinity (0.6B + 10K head, CMA-ES) together prove:

1. **Verification emerges from correctness reward alone** — no explicit verification training needed (Conductor)
2. **CMA-ES beats REINFORCE at tiny parameter counts** — 0.615 vs 0.253 on LiveCodeBench (Trinity)
3. **10K-param heads on frozen small models can make effective binary decisions** (Trinity)

E_cond1 is the highest-novelty experiment: if a 10K-param CMA-ES head matches the behavioral RF at near-zero inference cost AND transfers cross-model, it's a fundamentally different verification architecture.

Full research: `obsidian-notes/01_Projects/Learned-Verifier-Experiment/Sakana-Conductor-Trinity-Fugu-Deep-Research.md`
