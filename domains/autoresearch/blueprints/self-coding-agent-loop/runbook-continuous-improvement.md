# Runbook: Continuous-Calibration RLVR for Coding Agents

**Status**: DRAFT — numbers backfilled from Round 1 (pending)
**Audience**: ML platform engineer who needs to deploy continuous-improvement for a coding agent in the next quarter
**Scope**: applied recipe, not a research contribution
**Related**: [self-coding-agent-loop spec](../../../domains/autoresearch/specs/self-coding-agent-loop.md), [failure-modes.md](failure-modes.md), [cost-calculator.md](cost-calculator.md), [experiment-backlog.md](experiment-backlog.md)

---

## When to use this recipe

Use continuous-calibration RLVR when **all** of these are true:

- [ ] You have a coding agent at 30-60% SWE-bench Lite pass rate
- [ ] You have ≥10K labeled agent trajectories (gold pass/fail) you can afford to spend on training
- [ ] You have a stable harness + inference path (OpenHands v0.54+ or equivalent)
- [ ] You can run Docker gold eval on 100-500 tasks per round (~$20-40 / round)
- [ ] The agent will deploy to tasks where gold tests DO NOT exist (else just use gold directly)
- [ ] You accept that the verifier is imperfect — this recipe buys you 10-100× cheaper reward signal, not better quality

**Do NOT use this recipe when**:
- Base rate is already > 70% (noise dominates signal)
- You can afford full Docker gold eval in production (just use that)
- Your agent harness changes frequently (each harness shift invalidates calibration)
- You have < 2K labeled trajectories (RF can't converge)

---

## Architecture

```
Round N:
  1. SFT on round_N_train (~500-800 instances, ~4K trajectories, gold-filtered)
  2. Generate patches on round_N_control (never-seen) + drift_audit_300 (shared)
  3. Docker gold eval on both
  4. Recalibrate verifier RF on cumulative gold labels
  5. Append drift trajectory point N
  6. Plateau check → stop early if Δ < 1pp

Round N+1 starts from Round N's adapter. Drift_audit_300 is write-locked; re-scored every round with identical inputs → pure drift signal across model versions.
```

---

## Decision tree — at each round end

```
┌─────────────────────────────────────────┐
│ Round N complete. What happens next?    │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴──────────┐
        │ Gen-N gold pass on  │
        │ round_N_control?    │
        └─────────┬──────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
      ▼           ▼           ▼
  Δ ≥ +3pp   Δ < +3pp    Δ < 0 (regression)
  (improved)  (plateau)   (collapse)
      │           │           │
      │      ┌────┴─────┐     │
      │      │ Verifier │     │ Red alarm:
      │      │ ECE on   │     │ Pause training.
      │      │ drift?   │     │ Diagnose before
      │      └────┬─────┘     │ next round.
      │           │           │
      │    ┌──────┼──────┐    │
      │    │             │    │
      │    ▼             ▼    │
      │ ECE stable    ECE >0.1│
      │ (≤prior +0.02)        │
      │    │             │    │
      │    ▼             ▼    │
      │  Stop:      Loop 1    │
      │  "STaR is   recal +   │
      │  saturated" rerun     │
      │                       │
      ▼                       │
 Proceed to Round N+1 ────────┘
```

---

## SLO thresholds (to instrument)

| Signal | Green | Yellow (investigate) | Red (halt) |
|---|---|---|---|
| Gen-N gold pass on round_N_control (Δ vs Gen_{N-1}) | ≥ +3pp | [0, +3pp) | < 0pp |
| Verifier-gold agreement on drift_audit_300 | ≥ 0.85 | [0.80, 0.85) | < 0.80 |
| Verifier ECE on drift_audit_300 | ≤ 0.1 | (0.1, 0.2] | > 0.2 |
| rate(v009_fail ∧ rf_pass) drift | Δ ≤ 5pp from baseline 47% | (5pp, 10pp] | > 10pp |
| Verifier score ↑ while gold ↓ | never | any instance | any pattern across 3+ instances |

The last row is the reward-hacking alarm — if it fires, halt immediately and investigate even if every other SLO is green.

---

## Operational cadence

Weekly (first 4 weeks of deployment):
- Review drift trajectory dashboard
- Check cost-per-pp-improvement against budget
- Spot-check 10 verifier "pass" decisions against Docker gold

Monthly (steady state):
- Full drift report
- Budget review
- Decide whether to run another training round or hold steady

---

## Budget planning

*Numbers TBD after Round 1 completes. Placeholder structure:*

| Scenario | Per-round cost | Cost per pp improvement | Break-even vs engineer review |
|---|---|---|---|
| Round 1 (first training on fresh base) | [pending] | [pending] | [pending] |
| Round N > 1 (incremental) | [pending] | [pending] | [pending] |
| Drift recalibration only (no new SFT) | [pending] | [pending] | [pending] |

See [cost-calculator.md](cost-calculator.md) for the full model.

---

## Infrastructure minimums

**Training node**: 1× p4de.24xlarge spot (us-east-1c, ~$13.50/hr) OR 2× H200 DDP equivalent. MoE 30B-A3B fits comfortably. Dense 32B also fits. >480B does NOT.

**Eval node**: 1× m7i.16xlarge spot (~$2/hr) for Docker gold eval. Pre-pull SWE-bench images (~1-5GB each × N instances) before evaluating — saves 1-2hr wall-clock per round.

**Storage**:
- S3 bucket for durable checkpoint backup (spot reclaim insurance)
- Local NVMe for training working set (p4de instance-store has 6.5TB LVM pre-formatted as `/opt/dlami/nvme`)
- Laptop archive for post-run artifacts

**Software pins** (known-good as of 2026-05-11):
- transformers: from git main (5.8.0.dev0) — pre-built wheels don't yet support Qwen3.5/Qwen3-Coder configs
- trl: 1.4.0
- peft: 0.19.1
- vllm: 0.20.2 (supports transformers 5.x)
- torch: 2.11.0 + cu121

---

## Launch procedure

1. **Pre-flight**: splits manifest present, Gen0 adapter on S3 OR accept fresh-start, Nebius dataset mirrored, SSH between nodes verified.
2. **Round 1**: one SFT + one eval cycle. **Stop and review before committing to Round 2** — this is the go/no-go on whether the pipeline works.
3. **Round 2**: only if Round 1 Δ ≥ 3pp. Extends to 3+ rounds only if Round 2 also clears threshold.
4. **Phase 2 transition**: only after 3 consecutive rounds of drift-trajectory stability (verifier-gold agreement ≥ 0.85 on drift_audit_300 across 3 model generations).

The 2-round minimum-viable gate is the critical product discipline — most teams over-scope this experiment. Round 1 tells you if the pipeline works; Round 2 tells you if improvement compounds; everything after is about ceiling-finding.

---

## Common failures (catalog in [failure-modes.md](failure-modes.md))

- VLM vs text model confusion (Qwen3.5-27B is VLM, Qwen3-Coder-30B-A3B is text)
- trl API churn between 1.3 and 1.4 (SFTConfig, SFTTrainer, callback base class)
- Chat template expects dict for tool_calls.function.arguments, Nebius stores as JSON string
- Ephemeral NVMe on DL AMI is pre-formatted as LVM (don't try to mkfs)
- PEFT adapter key mismatches when base model family changes

---

## Open questions (being answered by Round 1/2)

1. Does Qwen3-Coder-30B-A3B + Nebius OpenHands trajectories + gold-filtered SFT beat the raw base model by ≥ 3pp?
2. Does Round 2 Gen2 add another ≥ 3pp on top of Round 1's Gen1?
3. Does verifier-gold agreement on drift_audit_300 stay above 0.85 across Gen1 and Gen2?
4. Is the `v009_fail ∧ rf_pass` override signal present in Gen1/Gen2 outputs (as it was in Claude x OpenCode)?

Answers feed directly into the SLO table above.
