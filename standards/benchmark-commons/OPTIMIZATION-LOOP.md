# The In-Spec Optimization Loop

**Status**: STANDARD
**Version**: 1.0
**Date**: 2026-06-27
**Scope**: GPU Serving domain (`domains/gpu-serving/`)
**Companion**: [`docs/optimization-stack.md`](../../docs/optimization-stack.md) (lever catalog), [`.claude/steering/inference-first-principles.md`](../../.claude/steering/inference-first-principles.md) (regime prediction), [`.claude/steering/benchmark-analysis.md`](../../.claude/steering/benchmark-analysis.md) (assert-after-measure discipline)

---

## What this is

A blueprint deployment has two optimization loops, not one:

- **Outer loop (cross-spec, already built):** spec → deploy → Stage 6 → `compound-learner` promotes regime-tagged lessons up the knowledge ladder so the *next* spec starts sharper. Slow; between iterations.
- **Inner loop (in-spec, this standard):** once a config *serves correctly* (Stage 5), hill-climb the declared SLO objective across many configs *before* the spec is done. Fast; within one deployment.

The repo already runs a one-shot version of the inner loop: Stage 0b predicts the regime and picks levers; Stage 6 measures one config's Tier Stack Table. This standard makes the middle **iterative** — and, critically, makes the *search trajectory* a first-class artifact so the outer loop can learn from the path, not just the destination.

The design is validated against a real multi-agent optimization corpus: HuggingFace's Fast Gemma Challenge (86 agents, ~570 configs, 95→507 TPS on a fixed A10G under a PPL cap). Our trajectory record is a superset of that corpus's proven structure plus one field — `regime` — which a heterogeneous fleet needs and a single-hardware competition does not.

## Why a catalog is not an optimizer

`docs/optimization-stack.md` is a **lever catalog**. A catalog plus a search strategy is an **optimizer**. The Gemma corpus shows the difference: with no prior, 86 agents took a week of flat brute-force search to discover that pruning *early* decoder layers raises throughput without hurting quality. A loop that opens with "regime says decode-BW-bound → highest-Δ levers are T3 spec-decode and T6 surgery; here are the 3 dead-ends already known for this regime" converges in a handful of gated steps.

**The stack is the loop's search-space definition and its ordering prior:**

- **Bounds** — the tiers enumerate which levers *exist* (the search space); the conflicts tables and regime-tagged dead-ends *prune* branches before a run is spent on them.
- **Prior** — the roofline predicts the regime from architecture + SLO; the stack's per-regime lever priority + typical-Δ *rank* what to try first.

Every completed inner loop feeds measured `(lever, Δ, regime)` records back, so the prior tightens each iteration. That tightening *is* the compound flywheel.

## The declared objective (lives in the spec)

The loop optimizes a **single scalar objective under hard guardrails**. It is declared in the spec's Stage 0b, never improvised by the loop:

```yaml
optimization_objective:
  maximize: <metric>            # e.g. output_tokens_per_sec @ c=1  (the SLO axis)
  subject_to:                   # HARD guardrails — a breach invalidates the config, full stop
    quality_gate:
      eval: <held-out eval>     # e.g. mmlu+gsm8k, or per-modality PPL ≤ cap
      baseline: <score>         # reference (e.g. bf16 PPL ≈ 2.30)
      tolerance: <Δ>            # max allowed regression
      held_out: true            # REQUIRED — see Reward-hacking guard
    invariants:                 # things that must stay true, not metrics to trade
      - all_modalities_intact   # text/vision/audio still serve (don't "win" by dropping one)
      - error_rate_max: 0.001
  budget:
    max_configs: <N>            # stop after N candidate configs
    max_wall_clock_min: <M>     # or M minutes of GPU time
    max_usd: <$>                # or $ spent
  plateau:
    min_improvement: <Δ>        # smallest objective gain that counts as progress
    patience: <K>               # stop after K consecutive configs with no >Δ gain
```

A loop with no single objective is a tinkerer; a loop with no budget/plateau runs forever; a loop with no held-out guardrail is a reward maximizer pointed at your benchmark. All four blocks are mandatory.

## The loop (greedy guided search)

```
0. Seed = the working Stage 5 config (T0 honest baseline).
1. Predict regime (Stage 0b, inference-first-principles.md).
2. Rank candidate levers by regime-matched expected Δ from optimization-stack.md;
   prune any pruned by a conflict or a regime-tagged dead-end.
3. Pick the top untried lever. Apply it as a SINGLE-VARIABLE change vs the current best
   (byte-identical except the one lever — this is what makes the measured Δ trustworthy).
4. Run the quality gate FIRST (fail-closed). If it breaches → mark config dead-end
   (status=quality_breach), do NOT measure throughput, do NOT keep. Go to 3.
5. Measure the objective metric on the standard workload.
6. Emit a trajectory node (schema below). If objective improved by > plateau.min_improvement,
   it becomes the new best.
7. Stop when budget exhausted OR plateau.patience consecutive non-improvements.
   Otherwise go to 3.
8. The best config's tier breakdown fills the Stage 6 Tier Stack Table; the full node
   set is the trajectory artifact.
```

Single-variable discipline (step 3) is non-negotiable — it is what lets `compound-learner` attribute a +Δ to one lever. The Gemma corpus's most reusable negatives ("n-gram spec-decode regresses −29% at c=1", "Marlin atomic-add is +0.03 = noise") were only trustworthy because they were byte-identical A/Bs.

## The trajectory record (the new artifact)

Written to `<blueprint>/results/optimization-trajectory-<YYYY-MM-DD>.json`. It is **raw input** for `compound-learner`, not promoted state — keeping the steering files' append-only guardrail intact. One node per candidate config:

```json
{
  "objective": "output_tokens_per_sec @ c=1",
  "regime": "decode-BW | A10G sm_86 | c=1",
  "guardrail": "mmlu within 1pp of bf16 baseline (held-out)",
  "nodes": [
    {
      "id": "20260627-T1-fp8",
      "parent": "20260627-T0-baseline",
      "lever_delta": ["T1: FP8 E4M3 weights+KV"],
      "confidence": "code-confirmed",
      "objective_value": 142.0,
      "guardrail_value": "mmlu -0.3pp (pass)",
      "regime": "decode-BW | A10G sm_86 | c=1",
      "status": "kept"
    },
    {
      "id": "20260627-T3-ngram",
      "parent": "20260627-T1-fp8",
      "lever_delta": ["T3: n-gram speculative decode k=4"],
      "confidence": "code-confirmed",
      "objective_value": 101.0,
      "guardrail_value": "n/a (lossless)",
      "regime": "decode-BW | A10G sm_86 | c=1",
      "status": "dead-end",
      "note": "−29% vs parent; n-gram overhead unamortized at c=1"
    }
  ]
}
```

| Field | Meaning | Gemma source (validation) |
|-------|---------|---------------------------|
| `id` | unique config id | `id` |
| `parent` | the config this was derived from (lineage) | `parents[].parent` |
| `lever_delta` | the single tier/lever changed vs parent | `parents[].delta` |
| `confidence` | trust in the delta: `code-confirmed` > `config-inferred` > `ppl-match` > `name-inferred` | `parents[].confidence` |
| `objective_value` | measured value of the maximize-metric | `tps` |
| `guardrail_value` | measured guardrail result (pass/fail + margin) | `ppl` |
| `regime` | **NEW** — roofline regime + arch/hardware/concurrency. The port key. | *(held constant in Gemma)* |
| `status` | `kept` \| `dead-end` \| `quality_breach` | `verification`/`status` |

**Confidence ladder** mirrors `optimization-stack.md`'s lineage rule and `benchmark-analysis.md`'s "match before you compare": `code-confirmed` means the config diff was verified; `ppl-match` (identical guardrail value) is the tell that two "different" configs are secretly the same substrate — it catches accidental duplicate work and false deltas.

**`regime` is the heterogeneity contract.** Gemma never needed it: one model, one A10G, one goal — every lesson universal, zero porting cost. Our fleet is the opposite (a B300/MoE/c=512 win is *wrong* applied to an A10G/dense/c=1 spec). Tagging every node with its regime is what lets `compound-learner` promote a Δ by **regime-match, not model-match** (see its routing ladder). A win in the same roofline regime on a different model → promotable lever. Same model, different regime → stays a card fact.

## Reward-hacking guard (the load-bearing safety property)

A loop hill-climbing on throughput **will** rediscover quality degradation disguised as speed if you let it. The Gemma corpus proves it: an agent hit 321 TPS with "relaxed acceptance" (accepting draft tokens within ε of the target's argmax) — a real throughput number achieved by emitting tokens the model rated at ~37% of its top choice. The organizers **ruled it invalid** because it changes emitted output vs greedy. It was only caught because the official quality check ran on a **private prompt set the agents could not tune against.**

The in-spec analog, mandatory:

1. **The quality gate runs on a held-out eval the loop never optimizes against** (`held_out: true`). If the loop can see the gate's prompts, it will overfit them.
2. **Fail-closed**: a gate breach marks the config `quality_breach`, skips throughput measurement entirely, and never lets it become best. A breaching config is invalid, not a trade-off.
3. **An objective gain achieved by relaxing the guardrail is never a throughput win.** `compound-learner` records it as a *quality-change lesson*, not a lever (see its "What NOT to elevate"). "We got faster by accepting worse tokens" is a fact about the guardrail, not the lever catalog.
4. **Invariants are not tradeable.** Dropping a modality, or pushing error rate up, to win the scalar objective is a `quality_breach`, not a Pareto point.

Without this, an autonomous optimizer is an unaligned reward maximizer with GPU access. With it, the objective is honestly constrained and every kept node is a real gain.

## Handoff to the outer loop

When the loop stops:

- **Best config** → the Stage 6 Tier Stack Table (unchanged contract; the table is now the *winner*, the trajectory is the *path*).
- **Trajectory artifact** → `compound-learner` reads it as a new input source (alongside readiness-audits and deployment-logs) and routes both *kept deltas* and *dead-ends* through its existing knowledge ladder:
  - a regime-tagged Δ recurring across ≥2 regime-matched runs → promote to the relevant `optimization-stack.md` tier (phenomenon-keyed table row, dated `Seen`);
  - a regime-tagged dead-end recurring across ≥2 specs → promote as a *conflict/no-op row* in that tier (so the next loop prunes it without spending a run);
  - anything model/engine/version-specific → stays in `lessons.md` + `mdc`/`gpu-infra` cards.

The framing of the objective itself — is the regime sound, is the gate truly held-out, are the priors regime-matched, is the budget set — is gated *before* the loop runs by the `carryover-auditor` (GPU-serving domain), acting as a spec interviewer. The template provides the slots; the auditor enforces that they're filled with judgment.
