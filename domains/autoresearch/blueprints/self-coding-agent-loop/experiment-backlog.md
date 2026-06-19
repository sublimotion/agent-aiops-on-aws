# Experiment Backlog — Deferred

**Parent**: [self-coding-agent-loop](../../../domains/autoresearch/specs/self-coding-agent-loop.md)
**Current scope**: 2-round product-first (Round 1 → Round 2 if Δ≥3pp). Everything below is deferred.
**Status key**: 🔵 NOT_STARTED · 🟡 PARTIAL · ✅ DONE · ❌ REJECTED (with reason)

Decision rule: **do not start any of these until the current 2-round experiment completes** and the runbook + failure-modes are solid. These are "earn the right to run them" items.

---

## Rounds 3-5: Drift trajectory extension

**Status**: 🔵 NOT_STARTED
**Cost**: ~$330 additional (3 × ~$110)
**Duration**: 3-4 weeks wall-clock
**Trigger**: Round 2 clears +3pp threshold AND verifier-gold agreement on drift_audit ≥ 0.85 AND time/money budget approved

**What it adds**: full 5-point drift trajectory (Gen1..Gen5), enables Phase 2 entry gate evaluation (requires 3 consecutive rounds stable).
**What it doesn't add**: the core "does continuous calibration work?" question is already answered by Round 2. Extension is for convergence studies and publishable drift curves.
**Decision artifact**: a plot of verifier ECE vs Round N across 5 generations.

---

## Arm B: GRPO with gold reward

**Status**: 🔵 NOT_STARTED
**Hypothesis**: GRPO beats iterative STaR by ≥2pp on the same data, justifying the RL machinery.
**Cost**: ~$930 (3 iterations)
**Duration**: 2 weeks
**Trigger**: Arm A (current pipeline) improves ≥3pp per round AND we want to answer the RL-vs-SFT question for publication/internal clarity.
**Dependencies**:
- Arm A works (current pipeline)
- Generation pipeline proven (so we can generate 4K tasks × group_size 8 = 32K rollouts)
- Budget doubled from current ~$500 to ~$1,430

**Why deferred**: product-first framing says RL vs STaR is a research question. If STaR works, ship STaR. Save GRPO for when the base-case STaR pipeline is a known-good product.

---

## Arm C: Verifier-STaR (verifier replaces gold reward)

**Status**: 🔵 NOT_STARTED
**Hypothesis**: Arm C ≈ Arm A within 3pp, proving the verifier is a viable gold-replacement for SFT curation at scale.
**Cost**: ~$870 (3 iterations, Haiku verifier + p4de train)
**Duration**: 2 weeks
**Trigger**: Arm A works AND V1b_validate passes (verifier meets V1b-unlock on target-dist traces)
**This is the headline research question.** If A and C track within 3pp, continuous calibration is validated.

**Why deferred in current 2-round scope**: 2-round Arm A alone establishes whether the pipeline works. Arm C only tells us whether verifier-as-reward works — orthogonal, and we can add it as Round 3+ if the base pipeline looks solid.

---

## Arm D: GRPO + verifier (full continuous RL)

**Status**: 🔵 NOT_STARTED
**Gate**: Arm C passed AND Arm B > Arm A by ≥2pp
**Cost**: ~$860 (2 iterations)
**Duration**: 2 weeks
**Only run if both RL and verifier independently validated.**

---

## Arm E: Gold, subsampled to verifier recall rate

**Status**: 🔵 NOT_STARTED
**Hypothesis**: If E ≈ C, the verifier tax is pure sparsity (low recall), not noise. If E > C, verifier adds noise on top of sparsity.
**Cost**: ~$460
**Duration**: 1 week
**Trigger**: Arm C ran and we want to explain WHY (noise vs sparsity).

---

## V1b_rubric: rebuild v009 rubric for target distribution

**Status**: 🔵 NOT_STARTED
**Cost**: $20-30 per rubric variant
**Duration**: 2 days
**Trigger**: V1b_validate fails AND V1b_bootstrap ECE flat (RF can't do it alone; rubric is the bottleneck)

From prior experiments we know v009 is Claude-specific. No pre-tested replacement exists. Running this would involve:
1. Sample 100 labeled patches from target distribution (Qwen3-Coder × OpenHands)
2. Identify v009 systematic error modes on that distribution
3. Author 2-3 rubric variants (temp=0.0, temp=0.3 per T10b protocol)
4. Pick variant with highest precision at recall ≥ 0.10

**Why deferred**: only relevant if V1b fails. First check whether the V1b_bootstrap-only recalibration is sufficient.

---

## Qwen3-Coder-480B-A35B FP8 inference comparison

**Status**: 🔵 NOT_STARTED
**Hypothesis**: The 480B teacher that generated our training data has a known ceiling (Nebius reports 66.2% SWE-bench). Serving it on p4de (FP8 TP=8) gives us an upper bound for "what the data could teach us."
**Cost**: ~$50 (1 day inference on p4de, no training)
**Duration**: 2 days
**Trigger**: Useful as an oracle reference if our Gen5 plateaus below 50% — "are we close to the teacher ceiling?"

**Why deferred**: pure inference comparison, not part of training loop.

---

## Cross-harness verifier transfer

**Status**: 🔵 NOT_STARTED
**Hypothesis**: A verifier trained on OpenHands traces generalizes to SERA/swe-agent traces with ≤0.05 precision drop.
**Cost**: ~$100 (reuse existing SERA + Claude traces from agent-swarm blueprint)
**Duration**: 3 days
**Trigger**: current pipeline works AND we want to claim harness-agnostic verifier

**Relevance to product**: if true, we can amortize verifier development across harness families. If false, each harness needs its own verifier.

---

## Reward-hacking adversarial probing

**Status**: 🔵 NOT_STARTED
**Hypothesis**: Deliberately-constructed patches that score high on verifier but fail gold can be produced systematically. Useful for stress-testing the SLO.
**Cost**: ~$50 (Haiku + Docker eval on 100 adversarial candidates)
**Duration**: 1 week
**Trigger**: any round shows "verifier score ↑ gold ↓" signature — we need to characterize the failure mode.

**This is the safety counterpart to the training loop.** Currently relying on drift_audit to detect this passively; an active probe would be stronger.

---

## Qwen3.5-27B VLM pipeline completion

**Status**: ❌ REJECTED (for now)
**Reason**: Spent ~5hr and $50 trying to make the VLM path work. Qwen3-Coder-30B-A3B achieves the same experimental goals with a straightforward text-only path. The only thing the VLM pipeline would add is "we can also train VLMs as coding agents" — which is a different experiment entirely.

**When to reconsider**: if a future experiment explicitly needs multimodal input (screenshot-based debugging, for example). Until then, staying on text-only models.

---

## Larger training horizons (5 epochs, 131K context)

**Status**: 🔵 NOT_STARTED
**Hypothesis**: Matching Nebius's 5-epoch / 131K-context recipe gets our model to ~50% without any RL. Their recipe took 16× B200 × 65hr.
**Cost**: ~$600 (p4de × 130hr, 2× Nebius's B200 wall-clock but 4× cheaper)
**Duration**: 6 days wall-clock
**Trigger**: Round 1+2 plateau early AND we want to test "more epochs beats more rounds."

**Why deferred**: it's a Loop 2 parameter sweep, not a Loop 1 question. Our hypothesis is that the verifier signal is the bottleneck, not training horizon.

---

## SERA harness variant (fallback if OpenHands wobbles)

**Status**: 🟡 PARTIAL (Gen0 was trained on SERA)
**Hypothesis**: SERA harness is cheaper at inference (~$30/round vs ~$50 for OpenHands gen cost) with ≤3pp accuracy loss.
**Cost**: $0 additional if we can reuse existing SERA adapter artifacts
**Duration**: 1 day
**Trigger**: OpenHands v0.54 fails to install or runs slower than expected; need fallback.

---

## Custom lightweight harness (distilled from OpenHands)

**Status**: 🔵 NOT_STARTED
**Hypothesis**: A small distilled harness (minimal tool calls, shortest prompt) gives most of OpenHands's accuracy at 1/5 the token cost.
**Cost**: ~$300 (development + eval)
**Duration**: 3 weeks
**Trigger**: steady-state deployment AND token cost is a bottleneck. Premature for a 2-round pipeline validation.

---

## Per-family verifier routing (from E_env)

**Status**: ❌ REJECTED (pre-flight said no)
**Reason**: E_env measured pooled-vs-routed on the deployable feature set. Pooled RF had higher AUC (0.784) than sample-weighted per-cell (0.639). Per-pipeline routing doesn't help on the features we'd have at deployment.

**When to reconsider**: if we get richer per-trace features (beyond the 6-intersection set) that expose model-family signals.

---

## Publish public dataset of drift_audit + verifier scores

**Status**: 🔵 NOT_STARTED
**Rationale**: A labeled trajectory dataset with (model_version, verifier_score, gold_label) triples across Gen0..Gen5 would be a useful research artifact. Low marginal cost to release.
**Cost**: ~$0 (reformat + sanitize existing outputs)
**Duration**: 1 week
**Trigger**: all 5 rounds done AND results are scientifically interesting enough to be worth releasing.

**Why deferred**: not a product priority. Release only if the drift trajectory itself is noteworthy.

---

## Docker image for the pipeline (turnkey deployment)

**Status**: 🔵 NOT_STARTED
**Purpose**: Package orchestrator + scripts + pinned deps into a Docker image that teams can `docker run` on any GPU box.
**Cost**: ~$0 (engineering only)
**Duration**: 1 week
**Trigger**: we've run this 2+ times and the recipe is frozen.

**Product-first framing**: this is the real deliverable. Get the recipe stable first, THEN package it.

---

## Runbook dry-run on a second (agent, model) combination

**Status**: 🔵 NOT_STARTED
**Purpose**: Confirm the runbook generalizes beyond Qwen3-Coder × OpenHands × Nebius. E.g., try SERA × Devstral × SWE-agent-trajectories.
**Cost**: ~$400 (one full 2-round run on a different combo)
**Duration**: 1 week
**Trigger**: current 2-round run produces the artifacts we want AND we want to claim harness-agnostic recipes.

**This is the single most important deferred experiment** from a product-first perspective. One successful run is a prototype; two are a recipe.

---

## Dependencies graph

```
Current 2-round Arm A run
    │
    ├── Round 1 succeeds (Δ ≥ +3pp)? 
    │     ├── YES → Round 2 runs
    │     └── NO → stop; debug; relaunch from Round 1
    │
    ├── Round 2 continues? (Δ ≥ +3pp)
    │     ├── YES → backlog candidates unlock
    │     └── NO → plateau; runbook + failure-modes only
    │
    ├── Backlog tier 1 (unlocks if Round 2 succeeds)
    │     ├── Rounds 3-5 extension
    │     └── Runbook dry-run on 2nd (model, harness)
    │
    ├── Backlog tier 2 (unlocks if Round 5 succeeds)
    │     ├── Arm B (GRPO gold) — RL research question
    │     ├── Arm C (Verifier STaR) — headline research question
    │     ├── Cross-harness transfer
    │     └── Adversarial probing
    │
    └── Backlog tier 3 (unlocks if all above succeed)
          ├── Arms D, E
          ├── Dataset release
          └── Docker packaging
```

---

## 🔴 URGENT: SWE-ReBench Docker eval infrastructure

**Status**: 🔵 NOT_STARTED — **blocks Round 1 gold eval**
**Context**: FM-6.3 — SWE-ReBench eval images are not in the public `swebench/` Docker Hub namespace. We cannot gold-evaluate our generated patches without solving this.
**Cost**: $0 (research) → ~$50 (build custom images for eval set)
**Duration**: 1-3 days
**Trigger**: IMMEDIATE — Round 1 generation will produce patches we can't evaluate.

**Resolution paths to investigate** (in priority order):

1. **Check Nebius's own registry**: `nebius/...` Docker Hub, or their private GCR/ECR. Schema check of `nebius/SWE-rebench` dataset might expose `docker_image` field.
2. **Build-on-demand from setup_shell**: if the SWE-ReBench instance metadata includes setup commands, build images locally on m7i as needed. Expensive first-run but cacheable.
3. **Instance overlap with SWE-bench Verified**: find the subset of our round_1_control + drift_audit_300 that overlaps with `princeton-nlp/SWE-bench_Verified` (which DOES have public images). Use only those for eval. Trade-off: smaller effective eval set.
4. **Switch eval substrate entirely**: re-sample round_1_control and drift_audit_300 from SWE-bench Verified instances not used in training. This preserves the rolling-rounds structure but re-partitions around a different benchmark.

**Decision needed before Round 1's SFT finishes (~3 hours from now)**: path 3 or 4 is simplest to unblock. Path 1 would be ideal if Nebius has a registry.

---

## Meta: what's NOT in this backlog

Things I considered and explicitly chose NOT to add because they're solved problems, dead ends, or out of scope:

- **Bigger model (>30B-A3B)**: budget-prohibitive on single p4de.
- **Different base tokenizer**: would invalidate existing training data.
- **Custom reward models beyond RF+v009**: E_constraint_agent said no.
- **Novel RL algorithms beyond GRPO**: product-first; GRPO works.
- **Transfer to other languages beyond Python**: SWE-ReBench V2 has them but not a priority.
- **Online learning (train on individual customer requests)**: explicitly out of scope per spec; RLVR batch training only.
