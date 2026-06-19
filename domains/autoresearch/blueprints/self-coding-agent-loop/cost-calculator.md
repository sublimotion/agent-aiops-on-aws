# Cost Calculator — Continuous-Calibration RLVR

**Purpose**: quantify cost-per-percentage-point-improvement so you can answer "is this economic?" before committing to more rounds.
**Status**: DRAFT — numbers will be backfilled from Round 1.

---

## Per-round cost breakdown

| Component | Where it runs | Unit cost | Typical per-round |
|---|---|---|---|
| SFT training | p4de.24xlarge spot us-east-1c | $13.50/hr | ~4hr → **$54** |
| Patch generation (600 tasks) | p4de (same node, TP=8 inference) | $13.50/hr | ~2hr → **$27** |
| Docker gold eval (600 tasks) | m7i.16xlarge spot (or reuse swebench-eval m7i.4xlarge) | $2.00/hr × 4x slow on smaller box | ~15hr on 16xl or ~60hr on 4xl → **$30-120** |
| Verifier recalibration | laptop or p4de | effectively $0 | **$0** |
| Round-end plateau check + summary | laptop | $0 | **$0** |
| S3 storage (checkpoints + predictions) | S3 us-east-1 | $0.023/GB-month | **~$2 / month** |
| **Round total (m7i.16xl)** | | | **~$110** |
| **Round total (m7i.4xl reused)** | | | **~$200** |

### Amortization across rounds

- Setup cost (split generation, Gen0 re-baseline, V1b_bootstrap): ~$80 one-time
- Each subsequent round: ~$110

---

## Cost-per-percentage-point-improvement

This is the metric that matters. Formula:

```
$/pp = total_round_cost / (Gen_N_gold_pass_rate - Gen_{N-1}_gold_pass_rate) * 100
```

### Expected scenarios (placeholders until Round 1 completes)

| Scenario | Δ (pp) | $/pp |
|---|---|---|
| Round 1 home run (STaR usually works on fresh data) | +8 to +15pp | ~$7-14 / pp |
| Round 1 modest (typical for iterative STaR) | +3 to +7pp | ~$16-37 / pp |
| Round 2 continuation | +2 to +5pp | ~$22-55 / pp |
| Round 2 diminishing | 0 to +2pp | ~$55-∞ / pp (stop triggered) |

### Break-even points

| Alternative | Cost equivalence | When to prefer |
|---|---|---|
| Human code reviewer | ~$50/hr; ~30min/patch at scale | If Δ is < 3pp, human review is cheaper per correct patch |
| Just run Docker gold eval in production | ~$0.05/patch Docker + instance time | If your volume is < 5K patches/day |
| Scale up base model (30B → 235B) | ~10-20× inference cost | If you need ceiling-lifting, not drift-control |
| Switch to Claude Sonnet as agent | ~$3/1M tokens | If your agent calls are cheap and you value quality |

---

## When to STOP spending on rounds

Hard stop conditions (don't spend another round):

- **Δ < 1pp for 2 consecutive rounds** → plateau. Spend on harness (Loop 3) or base model (Loop 4) instead.
- **$/pp > $100** → diminishing returns exceeded. Human-review or alternative strategies are cheaper.
- **Verifier-gold agreement < 0.80** and recalibration didn't fix it → verifier is the bottleneck. Don't train more, fix the verifier.
- **Gen-N regressed** (Δ < 0) → stop, investigate. Don't throw money at a regression.

---

## Total experiment budget (2-round minimum-viable, current run)

| Item | Planned | Actual (as of 2026-05-11) |
|---|---|---|
| p4de spot (~12hr expected for 2 rounds) | ~$165 | $47 spent (first 3.5hr) |
| m7i eval (swebench-eval reused) | ~$0 (stopped/running as needed) | running, ~$2 |
| Debug time (API fits, VLM surprises) | $0 (budgeted) | **~$50 — unexpected, captured in failure-modes.md** |
| Storage + data transfer | ~$5 | $2 |
| **Total expected** | **~$170** | **~$100 so far** |
| **2-round total** | **$300-500** | **projecting $400-600** |

The debug overshoot is real; honesty compels acknowledging this is the cost of a first-run pipeline. Teams who copy this recipe should expect their first run to be ~20% more expensive than the steady-state per-round number.

---

## Cost comparison vs alternatives

**Full Docker gold eval in production** (no verifier): at 5K patches/day × $0.05/patch Docker = **$250/day = $7,500/month** ongoing.

**Our continuous-calibration pipeline at steady state**: $110/round × 1 round/month = **$110/month** + $15/month drift monitoring = **$125/month**. **60× cheaper at steady state**, but only pays off if:
- Volume > 500 patches/day (below this, Docker is fine)
- You can tolerate ~85% verifier precision instead of 100% gold
- The drift monitor catches silent failures (else the cheaper signal is worse quality)

---

## What drives cost down across rounds

1. **Parquet + Docker image pre-caching** (one-time): save 1-2hr on Round 1; free for subsequent rounds.
2. **Pre-tokenized dataset cached on NVMe**: save 9min per round.
3. **Round N+1 SFT overlapping Round N eval** (orchestrator does this): halves wall-clock, ~40% cost saving across 5 rounds.
4. **Reuse stopped m7i.4xlarge** (already has Docker + swebench harness): $0 setup vs ~$30 to spin up fresh m7i.16xlarge.
5. **Early stopping on plateau**: if Round 2 Δ < 1pp, we stop at $200 instead of pushing to $500 on diminishing rounds.

---

## What drives cost up

1. **First-time pipeline standup**: expect ~$50 in debugging even with this catalog (FM-3.1 through FM-4.4).
2. **Recalibration when verifier drifts**: extra ~$30 for a bootstrap re-train if drift alarm fires.
3. **Generating our own trajectories (Arms B/D)**: NOT in the current scope; adds ~$50/iter for 4K generations × 8 completions on p4de.
4. **Evaluating on full 5K control set instead of 600**: ~8x Docker eval cost. Current plan uses 600 — 300 control + 300 drift_audit — which is adequate for SLO decisions.

---

## Cost SLO (the product number)

**Target**: maintain $/pp ≤ $30 at steady state after the first 2 rounds.
**If exceeded**: the recipe isn't economic for your workload; use a different approach (see "Break-even points" above).

This SLO is more important than absolute accuracy numbers. A 50% gold-pass model that costs $15/pp to train is a better product than a 55% model that costs $80/pp.
