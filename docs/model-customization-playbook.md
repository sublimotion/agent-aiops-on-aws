# Customizing an Open-Weight Model from Agent Traces — Playbook

**Date:** 2026-06-21 · **Scope:** what we have a validated recipe for, and where the frontier (closed loop / RLVR) is blocked. Companion to `verification-program-retrospective.md`.

> **One-line answer to "do we have a good recipe?":** Yes — for **one-shot rejection-sampling SFT** on traces filtered by a *verifiable* reward. No — for a **closed self-improvement loop or RLVR**, which is blocked by the same reward-transfer wall the verifier retrospective documents.

---

## The recipe that works (validated, end-to-end)

**Rejection-sampling SFT**: collect agent trajectories → keep only those that passed a verifiable check (reward=1) → LoRA-SFT the base model on them → evaluate on a gold eval.

### Result (the proof)

`Qwen3.5-27B + SFT-D` → **46.7% SWE-bench Lite (140/300, full Docker gold eval)**, **71.1% precision** (patch→pass), 0 eval errors. Source: `domains/autoresearch/blueprints/rejection-sampling-sft/results/progress.md`.

- Competitive with published 32B results (SERA-32B 49.5%, CoderForge 59.4% — different eval sets).
- Only **12pp behind** our own VP+Sonnet-4.6 run (58.3%) — a frontier closed model at ~100× the inference cost.
- **Total program cost ~$175** (training ~$100, eval ~$75); inference ~$0.25/issue vs ~$5/issue frontier.

### The exact config (reproduce this)

| Knob | Value |
|------|-------|
| Base | Qwen3.5-27B (dense) — also worked on Qwen2.5-Coder-32B; **failed to help** on Qwen3-235B-A22B MoE (see pitfalls) |
| Method | LoRA, `r=16, alpha=32`, 1 epoch, ~375 steps |
| Hardware | 2×H200 DDP (27B), ~12.8 hrs |
| Data | 11,983 gold-labeled (reward=1) CoderForge trajectories |
| Eval | SERA harness, 30 turns, 65K context, g7e.12xlarge TP2, **Docker gold** (not self-judged) |

### Why it works — the conditions that make it valid

1. **The filter is a *verifiable* reward.** Gold pass came from running the actual SWE-bench Docker test suite. This is the load-bearing requirement: the recipe works because the trace filter is ground-truth, not an LLM judge. (Cf. retrospective: LLM-judge rewards don't transfer; execution rewards do.)
2. **High base precision helps.** The 27B already had 71% patch→pass precision — "the ideal profile for rejection sampling: high precision means the verifier's job is easy."
3. **One-shot, no distribution shift.** SFT trains once on a fixed filtered set. The reward only has to be valid *at collection time* — it never has to survive the model drifting off-distribution. That's why this works where RLVR doesn't.

---

## Honest scope boundaries (read before generalizing)

**1. The data was CoderForge (Claude-generated), not strictly "our own experiment traces."** The recipe is "SFT on gold-filtered *agent* traces"; the validated instance used a curated public trajectory set, not traces harvested from this repo's own agent-harness/swarm runs. Using *our* traces is plausible but **untested** — the self-coding-agent-loop tried exactly this (Nebius OpenHands traces, Qwen3-Coder-30B) and trained an adapter but **never ran the measurement leg**, so there's no number for "our-own-traces SFT."

**2. It's validated on benchmarks with gold tests only.** The whole recipe hinges on a verifiable filter. On problems *without* gold tests (the real goal of self-supervision), you'd need a verifier — and we've shown three times those don't transfer (retrospective). So the recipe customizes a model on problems you can *already* verify; it does **not** manufacture novel supervision.

**3. One generation, not a loop.** This is one SFT pass. It says nothing about whether a second generation (train on Gen1's filtered output) beats the first — the "self-improvement" claim is unproven and structurally hits the no-off-benchmark-verifier wall.

---

## Validated pitfalls (paid for)

- **MoE base models break LoRA-SFT.** Qwen3-235B-A22B + SFT got 74% fix rate but only **17% gold** — high fix, wrong patches. And the 122B MoE LoRA attempt (coderforge-eval) hit training instability: v1 loss-collapsed to memorization (8e-5), v2 spot-reclaimed before a checkpoint, blew the budget. **Use dense bases (27B/32B) for this recipe; MoE LoRA is unsolved here.**
- **Fix rate is deceptive — track gold pass, not fix rate.** The 235B's 74% fix / 17% gold gap is the canonical trap. A patch that *applies* is not a patch that *passes*.
- **Parkinson's Law is inherited and model-specific.** SFT on Claude-generated traces teaches the 27B to over-explore (first edit at 76% of turn budget) — wasteful for a smaller model. Trace-source habits transfer whether you want them or not.
- **50-task subsets are high-variance** — the 50-issue read (32%) was pessimistic vs the full-300 (46.7%). Eval on the full set before believing a number.
- **Use `python3.13` for any local sklearn/analysis** (macOS 3.14 sklearn broken).

---

## The frontier (what's NOT a recipe yet, and why)

| Ambition | Status | Blocker |
|----------|--------|---------|
| One-shot rejection-sampling SFT | ✅ **recipe** (above) | — |
| SFT on *our own* harvested traces | ⚠️ untested | self-coding-loop adapter trained, measurement leg never ran |
| Closed self-improvement loop (Gen2 > Gen1) | ❌ unproven | needs a verifier for Gen-N's self-generated, off-benchmark traces |
| RLVR (RL from verifiable rewards) | ❌ blocked | reward verifier collapses OOD (flywheel RF AUC 0.625 in-dist → **0.283** OOD); RL *moves* the model off-distribution by construction → reward density → 0 |

The wall is singular and well-characterized: **every loop that requires the reward to survive the model's own distribution shift is blocked, because the verifier doesn't transfer.** One-shot SFT is the recipe precisely because it sidesteps that — it never needs the reward to hold off-distribution.

### The two scoped experiments that would extend the playbook (both on the critical path, neither blocked)

1. **Does SFT on *our own* traces work?** Run the self-coding-loop's existing adapter through generate→Docker-eval→metrics. Cheap (adapter's on S3); turns "untested" into a number. *Caveat: it's still one-shot SFT, a replication — not a loop.*
2. **Mandatory per-loop verifier recalibration.** The flywheel's OOD failure implies RLVR is only viable if you re-fit the verifier on the *target* distribution every loop. Testing that is the one design change that could unblock the loop — but it's research, not a recipe.

---

## Bottom line

**You have a good, cheap, validated recipe for the 80% case:** rejection-sampling SFT on verifiable-reward-filtered traces gets a 27B dense model to ~80% of frontier coding performance for ~$175. It is real and reproducible. **You do not have a playbook for the self-improving loop or RLVR** — and that's not for lack of trying; it's the documented consequence of verifiers not transferring across the distribution shift those loops induce.

## Source files

- `domains/autoresearch/blueprints/rejection-sampling-sft/results/progress.md` (the 46.7% recipe + configs + pitfalls)
- `domains/autoresearch/blueprints/self-coding-agent-loop/` (the untested own-traces adapter)
- `domains/autoresearch/blueprints/coderforge-eval/` (MoE LoRA instability)
- `domains/autoresearch/blueprints/verification-flywheel/results/SUMMARY.md` (reward OOD failure)
- `docs/verification-program-retrospective.md` (why the loop is blocked)
