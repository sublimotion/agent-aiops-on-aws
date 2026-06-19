# Autoresearch Spec: Self-Coding Agent Loop

## Status: PAUSED (2026-05-11) — Round 1 SFT succeeded; generate+eval incomplete

**Session outcome** (full writeup: [SESSION_SUMMARY.md](../blueprints/self-coding-agent-loop/SESSION_SUMMARY.md)):

- ✅ **Round 1 SFT validated**: Gen1 LoRA adapter trained on Qwen3-Coder-30B-A3B-Instruct (13M params, loss 0.98→0.62 over 267 steps, clean convergence). Saved to `s3://agent-aiops-artifacts/self-coding-agent-loop/runs/round_1/adapter/`.
- ❌ **Round 1 generate+eval NOT completed**: vLLM GPU-memory race after SFT (patched but untested); OpenHands eval harness integration was under-scoped.
- ✅ **Product deliverables complete**: [runbook](../blueprints/self-coding-agent-loop/runbook-continuous-improvement.md), [22-failure catalog](../blueprints/self-coding-agent-loop/failure-modes.md), [cost model](../blueprints/self-coding-agent-loop/cost-calculator.md), [13-item backlog](../blueprints/self-coding-agent-loop/experiment-backlog.md), [visual explainer](../blueprints/self-coding-agent-loop/spec-explainer.html).
- 💰 **Spend**: ~$360 (72% debug/idle on remote p4de — captured in failure-modes for next team).

**Current launch reality**:
- **Base model**: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (NOT Qwen3.5-27B VLM; see [failure-modes.md](../blueprints/self-coding-agent-loop/failure-modes.md) FM-3.3).
- **Scope**: product-first 2-round (only Round 1 SFT actually executed; gen+eval deferred).
- **Only Arm A** (iterative STaR). Arms B/C/D/E in [experiment-backlog.md](../blueprints/self-coding-agent-loop/experiment-backlog.md).
- **Gen0 adapter**: none — fresh LoRA on raw base (the pre-existing Qwen3.5-27B adapter was incompatible).
- **Harness**: OpenHands v0.54 (planned; integration incomplete).
- **Training infra**: p4de.24xlarge spot us-east-1c (now TERMINATED, instance i-03e9a2bf15709bbdb).
- **Eval infra**: swebench-eval m7i.4xlarge (now STOPPED, instance i-02b3e99702834e4a9).
- **Eval substrate**: SWE-rebench v1 (not SWE-bench; not SWE-rebench-V2). Images at `docker.io/swerebenchv2/*`. Eval harness: SWE-rebench-V2's `scripts/eval.py`.

**Why this pivot from the spec body below**: 5-round / 5-arm scope was research-novelty-motivated. After novelty assessment (continuous-calibration RLVR has prior art in PRMs, DeepSeek-R1, Math-Shepherd), reframed to applied/product outcomes. Deliverable is the runbook + recipe + documented failure surface, not a drift-trajectory paper.

**To resume this experiment**: see SESSION_SUMMARY.md "What to do next". Short version: install OpenHands locally, smoke-test generate on TinyLlama, THEN pay for 30B scale. Gen1 adapter is already on S3 waiting to be evaluated.

---

## Status (original): DRAFT

## Overview

A **two-phase strategy** for building a self-improving coding agent:

- **Phase 1 (Extract & Train)**: Generate trajectories on SWE-ReBench V2's 32K tasks, evaluate against gold tests, train with RL directly. Gold tests *are* the reward — no verifier needed.
- **Phase 2 (Continuous Loop)**: Deploy on real-world tasks where no gold tests exist. The cascade verifier becomes the reward signal. Continuously improve.

The verifier is calibrated *during* Phase 1 (we have ground truth to compare against) so it's ready for Phase 2.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Gold-Labeled RL                                            │
│  Reward = gold test pass/fail (free, perfect signal)                 │
│                                                                      │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │ Nebius   │──▶│ resolved     │──▶│ RL (GRPO)    │                │
│  │ OH 67K   │   │ column       │   │ reward=pass  │                │
│  │ (gold)   │   │ (pre-eval'd) │   │              │                │
│  └──────────┘   └──────┬───────┘   └──────────────┘                │
│                         │                                            │
│         ┌───────────────┤ (side product: verifier calibration)       │
│         ▼               ▼                                            │
│  ┌─────────────┐  ┌──────────────┐                                  │
│  │ Calibrate   │  │ Held-out     │                                  │
│  │ verifier    │  │ eval (200)   │                                  │
│  │ (ECE→0)     │  │              │                                  │
│  └─────────────┘  └──────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: Verifier-in-the-Loop (production, no gold tests)           │
│  Reward = cascade verifier (calibrated in Phase 1)                   │
│                                                                      │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │ Generate │──▶│ Verifier     │──▶│ RL/SFT       │──┐             │
│  │ on new   │   │ scores       │   │ (continuous)  │  │             │
│  │ tasks    │   │ (ECE<0.1)    │   │              │  │             │
│  └──────────┘   └──────────────┘   └──────────────┘  │             │
│       ▲                                               │             │
│       └───────────────────────────────────────────────┘             │
│                                                                      │
│  Monitor: verifier drift detection → recalibrate if needed           │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Two Phases?

**Phase 1 is an opportunity we're leaving on the table.** SWE-ReBench V2 gives us 32K tasks with *perfect reward signal* (gold tests). At our current 46.7% pass rate, generating on all 32K yields ~15K correct trajectories — enough for RL directly, no SFT→DPO progression needed.

**Phase 2 is the production reality.** Real-world tasks (customer PRs, new repos, internal code) don't have gold tests. The verifier must be the reward. But by then, we've already calibrated it against thousands of gold labels from Phase 1.

### Sub-Loops (active in both phases)

The meta-loop still orchestrates multiple optimization axes:

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐
│  Loop 1:    │  │  Loop 2:    │  │  Loop 3:    │  │  Loop 4:   │
│  Reward     │  │  Data/RL    │  │  Agent      │  │  Model     │
│  Model      │  │  Training   │  │  Behavior   │  │  Selection │
│  Calibration│  │             │  │  & Harness  │  │            │
└─────────────┘  └─────────────┘  └─────────────┘  └────────────┘
```

- **Phase 1**: Loop 2 dominates (data is abundant, reward is free). Loop 1 runs passively (calibrating verifier on the side). Loop 3 optimizes generation efficiency.
- **Phase 2**: Loop 1 becomes critical (verifier IS the reward). Loop 2 continues. Loop 3/4 activate when plateaus hit.

## Outer Scheduler (meta-program.md)

```yaml
objective: maximize gold_pass_rate on swe_rebench_held_out (200 tasks)
global_budget: $3000, 8 weeks

# Priority ordering (check in this order each cycle)
bottleneck_detection:
  1_reward_model:
    trigger: drift-alarm level reached (see Verifier Readiness Levels table)
    action: activate Loop 1
    rationale: "Can't trust training signal — everything downstream is poisoned"

  2_data_quality:
    trigger: reward_model_ok AND (yield < 3% OR gold_pass stagnant 2+ iters)
    action: activate Loop 2
    rationale: "Signal is clean but not producing learning"

  3_agent_behavior:
    trigger: data_ok AND (parkinsons > 0.7 OR edit_rate < 0.3)
    action: activate Loop 3
    rationale: "Model has capacity, data is clean, but agent wastes turns"

  4_model_selection:
    trigger: all_loops_ok AND gold_pass plateau at < 40%
    action: activate Loop 4
    rationale: "Current model family has hit its ceiling"

# Global termination
terminate_when:
  - gold_pass_rate >= 50% on held_out (mission accomplished)
  - delta < 1pp across 3 consecutive outer cycles
  - budget exhausted

# Shared resources (see §Data & Evaluation Infrastructure for the authoritative split)
train_set: 8K SWE-ReBench V2 Python tasks
control_set: 5K SWE-ReBench V2 Python tasks (never trained on; Docker gold eval every iteration)
calibration_set: 2K SWE-ReBench V2 Python tasks (gold labels + verifier scores; for ECE/precision)
```

## Data & Evaluation Infrastructure

### SWE-ReBench V2

The generation pool and held-out eval use SWE-ReBench V2 — a decontaminated, multilingual benchmark that's 60x larger than SWE-bench Lite.

| Property | Value |
|----------|-------|
| Total tasks | 32,000+ executable + 100K additional |
| Languages | 20 (Python, JS, Go, Rust, Java, Lua, Scala, ...) |
| Source | Automated pipeline extracting from real PRs |
| Docker images | Pre-built per-task environments (public SWE-bench registry; pulled locally as needed) |
| HuggingFace | `nebius/SWE-rebench-V2` (collection) |
| Paper | arxiv 2602.23866 |
| Eval speed | ~200-400 tasks/hr on m7i.16xlarge spot (self-hosted Docker, parallel) |

**Why SWE-ReBench over SWE-bench:**
- **Decontaminated**: Tasks post-date training cutoffs — no leakage
- **Scale**: 32K tasks vs 300 (Lite) or 500 (Verified) — enough for train/eval split
- **Multilingual**: Tests generalization beyond Python (future extension)
- **Reproducible**: Pre-built Docker images, automated pipeline

**Our split strategy (authoritative — experimental arms in Loop 2 reference these sizes):**

| Split | n | Purpose |
|---|---|---|
| `train_set` | 8K | Each iteration samples from this pool to generate trajectories for training. |
| `control_set` | 5K | Never trained on. Full Docker gold eval every iteration (SE ≈ 0.7pp at 50% baseline). |
| `calibration_set` | 2K | Gold-labeled subset used to measure verifier ECE / precision / readiness level. |

Total: 15K Python tasks. If SWE-ReBench V2's Python subset is smaller than 15K at sample time, shrink `train_set` (e.g. 6K) before touching `control_set` or `calibration_set` — statistical power and verifier calibration should not be compromised.

The former "Held-out eval (200 tasks)" from the outer scheduler is a quick-look subset of `control_set`, not a separate split.

### Evaluation Options

| Method | Speed | Cost | When to use |
|--------|-------|------|-------------|
| **Nebius pre-computed labels** | instant | $0 | Arms A/C/E — trajectories ARE the dataset, `resolved` column is the gold signal |
| **m7i.16xlarge spot (CPU Docker parallel)** | ~200-400 tasks/hr | ~$2/hr spot, ~$30 per 4K-task eval | Arms B/D — grading OUR newly-generated trajectories |
| **Self-hosted Docker on g7e or p4de** | ~50 tasks/hr | free between GPU cycles | Dev iterations, <200 tasks, or opportunistic use of training node |
| **SWE-bench Lite 300** (our infra) | ~50 tasks/hr | ~$15/run | Gen0 OpenHands re-baseline; cross-comparison with baseline |

The biggest cost reduction vs the original spec: Arms A/C/E use the `resolved` column from `nebius/SWE-rebench-openhands-trajectories` directly — no Docker runs needed. Arms B/D still gold-eval ~4K our-own-generated trajectories per iteration, which m7i.16xlarge spot handles in ~10-20 hours for ~$20-40/iter.

### Pre-Existing Trajectory Datasets

**Critical resource**: Nebius has already published trajectories on SWE-ReBench tasks.

| Dataset | Trajectories | Valid patches | Model | Harness | License |
|---------|-------------|---------------|-------|---------|---------|
| `nebius/SWE-rebench-openhands-trajectories` | 67,074 | 32,161 | Qwen3-Coder-480B-A35B | OpenHands v0.54 | CC-BY-4.0 |

**Published results from training on this data:**

| Model | Active params | Before RFT | After RFT | Δ | Training |
|-------|-------------|-----------|----------|---|----------|
| Qwen3-30B-A3B | 3B | 25.2% | **50.3%** | +25.1pp | 5 epochs, 8.9B tokens, 16×B200 65hr |
| Qwen3-235B-A22B | 22B | 46.2% | **61.7%** | +15.5pp | 3 epochs, 5.3B tokens, 32×H200 60hr |
| Our Qwen3.5-27B + SFT-D | 27B | ~25%* | **46.7%** | +22pp | 1 epoch, ~1.5B tokens, 2×H200 13hr |

*Estimated base Qwen3.5-27B without SFT on SWE-bench Lite.

**Critical comparison — why they beat us despite smaller active params:**

| Factor | Nebius (50.3%, 3B active) | Us (46.7%, 27B active) |
|--------|--------------------------|------------------------|
| Training data | 32K trajectories | 12K trajectories |
| Filtering | Patch applies (weak) | Gold test pass (strong) |
| Epochs | **5** | 1 |
| Sequence length | 131K | 65K |
| Total tokens seen | **8.9B** | ~1.5B |
| Harness | OpenHands | SERA (CoderForge) |
| Teacher model | Qwen3-Coder-480B | Claude (CoderForge) |

**Key insight: We are severely undertrained.** Their 3B-active model sees 6x more tokens than our 27B-active model. The low-hanging fruit is:
1. More epochs (1 → 3-5)
2. More data (12K → 32K, using their trajectories)
3. Longer sequences (65K → 131K)

**Their filtering is weaker than ours.** They keep any trajectory where the patch applies (not necessarily correct). Gold-eval filtering would give them even better results. This is exactly what our experiment tests — gold filtering (Arm A) vs verifier filtering (Arm C) vs their approach (patch-applies only).

**Implications for our experiment:**

1. **Arm A (STaR) has a quick win**: Take their 67K OpenHands trajectories, filter by the `resolved` column (32,161 passes), train for 3-5 epochs at 131K context. Expected: 55%+ (matching their 235B from our better filtering + more active params). *No Docker eval needed* — Nebius already ran gold eval and shipped the labels.
2. **Our differentiator is filtering quality**: Nebius filters on "patch applies" (noisy signal, 70K trajectories) but also ships the `resolved` gold-eval column (47.9% passes). We can filter on gold pass (clean) or verifier (cheap). Better filtering = better data = better model.
3. **Upper bound is known**: 61.7% (235B, 22B active, 3 epochs on 32K). Our 27B dense should reach 55-60% with same data and proper training.
4. **The verifier's value**: Replace Docker gold eval on new trajectories (~$30/4K-task run on m7i.16xlarge) with $0.03/patch Haiku scoring. If precision holds, same data quality at ~10-30x less eval cost.

**Strategy shift**: Rather than generating our own trajectories from scratch:
- Use Nebius OpenHands 67K trajectories as training data (already generated, gold-labeled via `resolved` column, CC-BY-4.0)
- Filter on the `resolved` column (32,161 gold-passing trajectories) — no TractoAI, no Docker, no eval cost
- Train for 3-5 epochs (not 1!) at 131K context
- Compare gold-filtered (`resolved=1`) vs verifier-filtered vs patch-applies filter (`model_patch != ''`)
- Then run GRPO on top (our value-add: RL vs their RFT) — for GRPO arms we generate our own trajectories and gold-eval them on m7i.16xlarge spot

### SWE-bench Lite 300 (Cross-Comparison)

Our measured baseline (Gen0) uses SWE-bench Lite 300 for direct comparison with VP+Sonnet 4.6 (58.3%) and published results. SWE-ReBench held-out becomes the primary eval once V1 (verifier transfer) passes.

## Loop 1: Reward Model Calibration

**Owner**: `learned-verifier` repo (`/Users/phi/Documents/workbench/learned-verifier/`)

**Purpose**: Calibrate the cascade verifier so it can replace gold tests in Phase 2. During Phase 1, this loop runs *passively* — every gold-evaluated trajectory is also scored by the verifier, giving us free calibration data. The output is a verifier at the **RL-ready** readiness level (see table below).

**Phase 1 role (passive)**: Absorb thousands of (verifier_prediction, gold_truth) pairs from Loop 2's gold evaluations. Train RF, iterate rubric, measure ECE — all using free labels.

**Phase 2 role (critical)**: The verifier IS the reward. If it drifts, training degrades. Loop 1 monitors drift and triggers recalibration.

### Verifier Readiness Levels (single source of truth)

All downstream gates reference these levels — do not inline numeric thresholds elsewhere.

| Level | ECE | Precision on target-distribution traces | Unlocks |
|---|---|---|---|
| **SFT-ready** | < 0.3 | >= 0.85 | Arm C (verifier-STaR), Arm E |
| **RL-ready** | < 0.1 | >= 0.90 | Arm D (verifier-GRPO), Phase 2 entry |
| **Drift alarm** | — | < 0.80 (rolling) | Pause, return to Loop 1 |
| **V1b unlock** | — | >= 0.70 | Arms C/D/E may *start* (below SFT-ready; gate is looser because Arm C measures empirical convergence, not deployment readiness) |

Target distribution = the trace distribution the verifier will score at deployment time, NOT the Claude×OpenCode distribution the RF was originally trained on. See M2 / V1b_bootstrap for why this matters.

```yaml
# loop1-program.md
objective: reach SFT-ready, then RL-ready (see Verifier Readiness Levels table)
metrics:
  primary: ECE on calibration_set
  secondary: precision@recall_0.10
  diagnostic: [FP_count, recall, rubric_agreement_rate]

starting_point: cascade_verifier_v009_4of4
  # Current: precision=0.92, recall=0.14, ECE unknown on SWE-ReBench

iterate:
  collect_labels:
    source: gold eval results from Loop 2 (patch + pass/fail)
    format: TraceInput via SERA adapter (from_sera_jsonl / from_sera_trajectory_dir)
    minimum: 200 new labels per iteration

  retrain_rf:
    # RF gate is the $0/1ms fast path — most volume hits this
    features: [total_cost_usd, tokens_per_edit, loop_count, first_edit_pct,
               edit_count, search_count, bash_count, elapsed_s, ...]
    method: calibrated_rf.py (isotonic regression + Platt scaling)
    target: ECE < 0.15 on held-out fold

  iterate_rubric:
    # v009 adversarial rubric — the $0.03/patch LLM judge
    method: A/B test rubric variants on 100 labeled patches
    keep: variant with highest precision at recall >= 0.10

  rebuild_rubric_for_distribution:
    # When the target trace distribution is NOT Claude (e.g. Qwen3.5, Devstral),
    # v009 itself may fail to transfer — T4 cross-verifier showed v009 is
    # Claude-specific (Devstral prec=0.20, Nova Pro prec=0.14, Mistral Large
    # refuses everything). RF bootstrap alone is insufficient when the rubric
    # is the bottleneck (see V1b_rubric prerequisite).
    trigger: V1b_validate fails AND V1b_bootstrap ECE is stable with added labels
             (i.e. RF is well-calibrated but overall cascade still misses)
    method:
      1. sample 100 target-distribution labeled patches
      2. run current v009 → identify systematic error modes (e.g. rejects terse
         diffs, over-accepts refactors, etc.)
      3. author 2-3 rubric variants addressing those modes
      4. A/B test at temp=0.0 and temp=0.3 (following T10b protocol)
      5. keep variant with highest precision at recall >= 0.10 on target distribution
    cost: ~$5-10 per variant × 2-3 variants = ~$20-30
    note: this is the ONLY pre-Phase-2 lever if the RF is good but the rubric
          isn't. No pre-tested constraint-verifier backup exists (E_constraint_agent
          was negative).

  bootstrap_new_distribution:
    # When traces shift (new model, new harness), RF needs recalibration.
    # Per M2 (2026-05-09), this is the DEFAULT path for new deployment
    # targets, not an edge case. Pair with rebuild_rubric_for_distribution
    # if V1b_validate still fails after bootstrap.
    trigger: any new (agent_model, scaffold) tuple, OR ECE > 0.3 on current distribution
    method: FlywheelBootstrap (200 labels minimum, 400-500 budgeted if ECE
            doesn't converge at 200)

terminate_when:
  - reached SFT-ready level (see Verifier Readiness Levels table)
  - reached RL-ready level (stretch goal)
  - 3 consecutive iterations with < 0.02 ECE improvement

output:
  - updated cascade model (committed to learned-verifier repo)
  - ECE certificate (logged to results/)
  - readiness-level flag (SFT-ready / RL-ready / neither)
```

### Key Insight: Claude-First Traces

The starting distribution is **Claude Sonnet 4.6 traces** (what customers actually use). This means:
- The verifier was calibrated on Claude-style edits (large, well-structured diffs)
- Transferring to open-weight models (smaller diffs, different patterns) requires Loop 1 recalibration
- Customer adoption path: Claude traces → verifier validates → confidence to deploy open-weight

**M2 measured this directly (2026-05-09)**: even same-model/different-scaffold (claude-sonnet on OpenHands vs OpenCode) drops the RF from AUC 0.727 in-sample to 0.486 OOD. On Qwen3.5×OpenCode the RF essentially flatlines (prob std=0.018). Treat `bootstrap_new_distribution` as the *default* path when starting a new deployment target, not an edge case triggered by ECE > 0.3.

## Loop 2: Training (Phase 1 = Gold RL, Phase 2 = Verifier RL)

**Owner**: This blueprint directory

**Purpose**: Train the coding agent. In Phase 1, gold tests provide perfect reward signal — go straight to RL. In Phase 2, the calibrated verifier replaces gold tests.

### Phase 1: Gold-Labeled RL on SWE-ReBench

```yaml
# loop2-phase1-program.md
objective: maximize gold_pass_rate using gold test reward
reward: binary (1 = gold tests pass, 0 = fail)
data_source: SWE-ReBench V2 (32K tasks) + SWE-ReBench V1
eval_on: Nebius pre-computed labels (Arms A/C/E); m7i.16xlarge spot for our-own-generated trajectories (Arms B/D)

why_rl_directly:
  # We have enough data for RL without SFT→DPO staging:
  # 32K tasks × 66% fix rate = ~21K trajectories with patches
  # 21K × 71% precision = ~15K positive reward trajectories
  # 15K is more than enough for GRPO (needs ~1K per iteration)
  # Gold test = perfect reward signal (ECE = 0 by definition)

training:
  method: grpo (group relative policy optimization)
  reward: gold_test_pass (binary, from Docker eval)
  base: Qwen3.5-27B (current best, 46.7% baseline)
  iterations: generate → self-host Docker gold eval on m7i → train → repeat
  per_iteration:
    generate: 4K tasks sampled from train_set (half the 8K pool; leaves room for re-sampling)
    eval: Docker gold test on m7i.16xlarge spot (~10-20 hr for 4K tasks in parallel, ~$20-40)
    expected_positives: ~1870 (at 46.7% pass rate, improving each iter)
    train: GRPO on (task, trajectory, pass/fail) triples
  kl_coeff: 0.05
  hardware: p4de.24xlarge spot in us-east-1 az6 (8× A100 80GB NVSwitch); FSDP or DDP, ~13-17 hrs/iteration
  checkpointing: local NVMe (/mnt/nvme) for speed, AND async rsync every 1 hr to the laptop — p4de NVMe is ephemeral instance-store and is lost on spot reclaim, so local alone is not a backup
  reclaim_handler: trap the 2-minute AWS spot termination signal → write final checkpoint → blocking rsync to laptop before the instance dies. See §Backup discipline below.

# Simultaneously calibrate verifier (side product, feeds Loop 1)
verifier_calibration:
  # Every gold-evaluated trajectory is also scored by cascade verifier
  # This gives us (verifier_prediction, gold_truth) pairs for free
  # Feed to Loop 1 to minimize ECE
  expected_labels_per_iter: 4000 (all generated, not just passes)

decisions:
  - if gold_pass improves >= 3pp: continue iterating
  - if gold_pass plateaus 2 iters: activate Loop 3 (harness) or Loop 4 (model)
  - if train_set exhausted: switch to SWE-ReBench V1 tasks (do NOT dip into control_set or calibration_set)

terminate_when:
  - gold_pass >= 60% on SWE-bench Lite 300 (matches CoderForge)
  - generation pool exhausted (32K tasks used)
  - Loop 1 reports verifier RL-ready (see Verifier Readiness Levels table)
```

**Why skip SFT/DPO?** With gold tests as reward:
- SFT is just "RL with reward=1 only" — discards negative signal
- DPO needs paired preferences — gold tests give us that for free (same task, pass vs fail)
- GRPO uses the full reward distribution directly — most sample-efficient

**Data sources for Phase 1:**

| Source | Tasks | Expected yield | Notes |
|--------|-------|----------------|-------|
| SWE-ReBench V2 (Python) | ~15K | ~7K passes | Primary pool |
| SWE-ReBench V2 (other langs) | ~17K | Future extension | Multilingual |
| SWE-ReBench V1 | Additional | Supplement | If V2 exhausted |
| SWE-bench Lite 300 | 300 | 140 (measured) | Cross-comparison only |

### Phase 2: Verifier-in-the-Loop (Continuous)

```yaml
# loop2-phase2-program.md
objective: continuous improvement on tasks WITHOUT gold tests
reward: cascade verifier score (calibrated in Phase 1 to RL-ready level)
data_source: new tasks (customer repos, fresh PRs, internal code)
gate: Loop 1 certifies RL-ready level (see Verifier Readiness Levels table)

training:
  method: grpo (same as Phase 1, swap reward source)
  reward: cascade_verifier_continuous_score
  per_iteration:
    generate: on new tasks (no gold tests available)
    verify: cascade verifier scores each trajectory
    train: GRPO on (task, trajectory, verifier_score) triples
  kl_coeff: 0.05

# Drift detection — verifier may degrade on new distributions
drift_monitor:
  every_N_iterations: 3
  method: hold back 100 tasks, run Docker gold eval, compare verifier vs gold
  threshold: drift-alarm level (see Verifier Readiness Levels table) — pause and recalibrate
  # Secondary signal (added from E_attr, 2026-05-09): baseline rate(v009_fail AND rf_pass) = 47% on Phase 3 Claude×OpenCode.
  # That cell is the "RF overrides v009's over-rejection" region, not an adversarial-patch region.
  # Alarm when its rate drifts >10pp from baseline — indicates trajectory-distribution shift,
  # not adversarial exploits. Use as an early-warning signal BEFORE gold-eval precision drops.
  secondary_signal:
    name: v009_fail_rf_pass_rate
    baseline: 0.47  # Phase 3 Claude×OpenCode, 141/300
    alarm_threshold: abs(current - baseline) > 0.10

decisions:
  - if verifier drift detected: pause, recalibrate (Loop 1)
  - if gold_pass on held-out eval plateaus: activate Loop 3/4
  - if new language/domain: bootstrap verifier on new distribution first

terminate_when:
  - never (continuous improvement)
  - budget exhausted
```

### Experimental Design: Gold vs Verifier Learning Rate

The core experiment validates whether continuous learning (Phase 2) is viable. Designed to avoid known pitfalls identified in review.

**Critical prerequisites** (the V1 family is staged — each gate assumes the previous passed):

- **(V1b_bootstrap)** Run `FlywheelBootstrap` with labels from the **target distribution = Qwen3.5-27B × OpenHands v0.54** (locked 2026-05-09) *before* any V1b_validate attempt. This step is **mandatory**, not conditional. *Why*: M2 measured the Claude-trained RF hitting AUC 0.486 on OpenHands claude-sonnet (same model, different scaffold) and AUC 0.538 on Qwen3.5×OpenCode, with probability std collapsing to 0.018 — the RF falls off its training distribution into flat leaf regions. Assume V1b_validate on the pre-bootstrap RF will fail; don't waste the budget on the test. *Bootstrap size*: the spec currently says ~200 labels, but Phase 3's original RF used 300. Treat 200 as a *lower* bound and budget for 400-500 if the bootstrap converges poorly (ECE still > 0.3 after 200). See [lessons](../blueprints/self-coding-agent-loop/pre-loop-micros-results.md) for why the minimum is unsettled.
- **(V1b_validate)** On the recalibrated RF from V1b_bootstrap, check precision on the Qwen3.5-27B × OpenHands calibration set. Pass = V1b unlock level (see Verifier Readiness Levels table) → Arms C/D/E may start. Fail = return to V1b_bootstrap with more labels. *Earlier evidence*: raw v009 precision on Qwen3.5-397B×OpenCode was 0.33-0.50 without bootstrap. That's a different (model, harness) cell — the recalibrated RF on this experiment's target distribution is the thing under test, not v009 alone and not the earlier number.
- **(V1b_rubric)** The cascade verifier combines the RF (retrained by V1b_bootstrap) with the v009 rubric (historically trained on Claude traces only). Prior T4 cross-verifier experiments showed v009 is Claude-specific (Devstral prec=0.20, Nova Pro prec=0.14, Mistral Large refuses everything). E_constraint_agent confirmed no generic rubric backup exists. If V1b_validate fails and the RF is already well-bootstrapped (ECE flat with added labels), the bottleneck is the rubric, not the RF. **In that case**: run Loop 1's `rebuild_rubric_for_distribution` step to author target-distribution rubric variants before Arms C/D/E can start. Expected cost: ~$20-30 per rubric-rebuild pass.
- **(V1c_openhands)** Validate verifier on OpenHands-format trajectories (different trace format than SERA). The learned-verifier has an OpenHands adapter (`from_openhands_trajectory`) — use it. *M2 caveat*: the adapter is a format converter, not a domain adapter. V1c passing does not imply V1b_validate passes on OpenHands traces.

**Shortcut: Use Nebius trajectories for Arm A.** The `nebius/SWE-rebench-openhands-trajectories` dataset has 67K trajectories with pre-computed gold labels in the `resolved` column (32,161 pass = 47.9%). Skip generation AND skip gold eval entirely for Arm A. This saves weeks of compute. Arm A becomes: "filter Nebius trajectories by `resolved == 1`, SFT on passes."

```yaml
# Dataset split: see §Data & Evaluation Infrastructure (authoritative).
# Sizes referenced below (8K train_set, 5K control_set, 2K calibration_set) come from that table.

# Five experimental arms
arms:
  A_iterative_sft:
    description: "Iterative STaR — simplest baseline, proven at this scale"
    method: generate → gold eval → filter passes → SFT on accepted (repeat)
    reward: gold test pass/fail (used for filtering only)
    train_data: 8K train tasks
    eval: control set (5K tasks, Docker gold)
    rationale: "CoderForge is basically this. Cheapest, most stable. Must beat this to justify RL."
    training:
      method: lora_sft
      rank: 16
      alpha: 32
      epochs: 1
      per_iteration: generate on 4K tasks, keep passes (~1900), SFT
      data: cumulative across iterations

  B_grpo_gold:
    description: "GRPO with gold reward — tests if RL adds value over STaR"
    method: GRPO with binary gold test reward
    reward: gold test pass/fail
    group_size: 8 (generate 8 completions per task, compute relative advantage)
    train_data: 8K train tasks (sample 1K per iteration × 8 completions = 8K generations)
    eval: control set (5K tasks, Docker gold)
    rationale: "If B > A, RL machinery is worth the complexity."
    training:
      method: grpo
      kl_coeff: 0.05
      per_iteration: 1K tasks × 8 completions = 8K generations, gold eval all

  C_verifier_sft:
    description: "Iterative STaR with verifier instead of gold — continuous learning proxy"
    method: generate → verifier score → filter reward=1 → SFT on accepted (repeat)
    reward: cascade verifier (no gold tests used)
    train_data: 8K train tasks
    eval: control set (5K tasks, Docker gold)
    gate: V1b_validate must pass at V1b-unlock level (see Verifier Readiness Levels)
    rationale: "If C ≈ A, verifier can replace gold tests for SFT curation."

  D_verifier_grpo:
    description: "GRPO with verifier reward — full continuous RL"
    method: GRPO with verifier score as reward
    group_size: 8
    reward: cascade verifier continuous score
    train_data: 8K train tasks
    eval: control set (5K tasks, Docker gold)
    gate: V1b_validate must pass AND Arm B > Arm A (proves GRPO adds value)
    rationale: "Only run if both RL and verifier independently validated."

  E_gold_subsampled:
    description: "Disentangle noise from sparsity — gold labels at verifier recall rate"
    method: same as Arm A (iterative SFT), but randomly discard 86% of positives
    reward: gold test (but only keep 14% of passes, matching verifier recall)
    train_data: 8K train tasks
    eval: control set (5K tasks, Docker gold)
    rationale: "If E ≈ C, then the verifier tax is all sparsity (low recall). If E > C, verifier adds noise on top of sparsity."

# What we measure
metrics_per_iteration:
  - gold_pass_rate on control set (primary — 5K tasks, SE ~0.7pp at 50%)
  - gold_pass_rate on train set (overfitting detector — should not exceed control by >10pp)
  - verifier_vs_gold agreement on calibration set (ECE, precision, recall)
  - tokens_consumed per gold_pass (efficiency)
  - task diversity in accepted set (unique repos, unique failure modes)

# Key questions answered (in priority order)
questions:
  Q1_does_iteration_help: "A_iter2 > A_iter1? (iterative STaR works at all)"
  Q2_rl_adds_value: "B > A? (GRPO beats STaR — justifies RL complexity)"
  Q3_verifier_viable: "C ≈ A? (verifier can replace gold for SFT curation)"
  Q4_verifier_tax_source: "E vs C: is the gap noise or sparsity?"
  Q5_full_continuous: "D > C? (verifier RL beats verifier SFT — justifies Phase 2 RL)"
```

**Decision tree:**

```
Start: Run A (iterative STaR) — 2 iterations
  │
  ├─ A improves >= 3pp? ──── YES ──→ STaR works. Run B (GRPO gold) to compare.
  │                                    │
  │                                    ├─ B > A by >= 2pp? ── YES ──→ RL adds value.
  │                                    │                               Run D (verifier GRPO).
  │                                    │
  │                                    └─ B ≈ A? ── STaR is sufficient.
  │                                                  Run C (verifier STaR) instead.
  │
  └─ A stalls? ──── Harness issue (Loop 3) or model ceiling (Loop 4).
                    Don't proceed to RL.

Meanwhile (parallel): Run V1b_bootstrap → V1b_validate → (V1b_rubric if needed).
  │
  ├─ V1b_bootstrap: FlywheelBootstrap on target-distribution traces (200 labels minimum, 400-500 budgeted)
  │   (mandatory — M2 showed claude-trained RF → AUC 0.486 on same-model-different-scaffold)
  │
  ├─ V1b_validate: recalibrated RF reaches V1b-unlock level on target traces? ── YES ──→ Run C and E.
  │
  ├─ NO? ──── Did ECE drop meaningfully with added labels during V1b_bootstrap?
  │            ├─ YES → extend V1b_bootstrap with 200 more labels and re-validate.
  │            └─ NO  → rubric bottleneck. Run V1b_rubric (rebuild_rubric_for_distribution in Loop 1).
  │                     If that still fails, no pre-tested rubric replacement exists (E_constraint_agent neg).
  │                     Block C/D/E; escalate to external review of rubric strategy.
```

**Statistical design notes:**
- Control set = 5K (not 200). SE = 0.7pp at baseline 50%. Detects 3pp difference at 95% power.
- Eval control set fully every iteration (not a 200-task subset).
- Overfitting gate: if train_pass_rate - control_pass_rate > 15pp, stop and investigate.
- GRPO group_size=8: sufficient for relative advantage estimation; 1K tasks × 8 = 8K generations per iteration.

**Cost estimates (conservative, using Nebius trajectories for Arm A):**

| Arm | Compute per iteration | Iterations | Total |
|-----|----------------------|------------|-------|
| A (STaR, Nebius data) | filter `resolved=1` (free) + p4de spot train (~$230) | 3 | ~$690 |
| B (GRPO, gold) | p4de inference gen 4K (~$50) + m7i spot gold eval (~$30) + p4de spot train (~$230) | 3 | ~$930 |
| C (STaR, verifier) | Haiku verify Nebius data (~$60) + p4de spot train (~$230) | 3 | ~$870 |
| D (GRPO, verifier) | p4de inference gen 4K (~$50) + Haiku verify (~$120) + m7i control eval (~$30) + p4de spot train (~$230) | 2 | ~$860 |
| E (STaR, subsampled) | Same as A but fewer accepted | 2 | ~$460 |
| Control eval (shared) | subsample 5K from Nebius `resolved` column (free) OR m7i 5K Docker run | per-iter | $0 (Nebius) / ~$40 (m7i) |
| Gen0 OpenHands re-baseline (added 2026-05-09) | SWE-bench Lite 300 eval of existing Gen0 checkpoint under OpenHands v0.54 | once | ~$15 |
| V1b_bootstrap (target-dist RF retrain, added 2026-05-09) | 200 Qwen3.5-27B × OpenHands gold labels + RF retrain | once (re-run if V1b_validate fails with ECE trending down) | ~$30 |
| V1b_validate / V1c_openhands | Haiku + 200 traces | once | ~$20 |
| **Total (if all arms run)** | | | **~$3,855** |
| **Minimum viable (Gen0 rebase + V1b_bootstrap + V1b_validate + A + C)** | | | **~$1,615** |

**Why costs rose vs. the pre-p4de estimate**: switched training infra to **p4de.24xlarge spot in us-east-1 az6** (~$230/iter vs ~$50 for 2×H200 third-party spot). Trade-off: higher per-iteration training cost, but keeps training + generation + bursty inference on one AWS node, avoids third-party cloud sprawl, and fits the 131K-context Nebius recipe natively if we chase that upgrade.

**Why costs dropped vs. the post-p4de estimate**: TractoAI removed — Nebius ships pre-computed gold labels via the `resolved` column on `SWE-rebench-openhands-trajectories`. Arms A/C/E have zero eval cost; only Arms B/D need Docker eval on newly-generated trajectories and those run on m7i.16xlarge spot (~$30/iter).

Full five-arm scope (~$3,855) fits under the $3,000/8-week outer-scheduler budget only if we trim iteration counts (3→2 for A/B/C, 2→1 for D/E). Minimum viable is well under budget.

Note: Arm A and C skip generation entirely by using Nebius pre-generated trajectories.
Only Arms B and D require our own generation (for GRPO's multiple-completion-per-task requirement).
Generation runs on the same p4de node between training steps (TP=8 serves Qwen3.5-27B fast enough to finish 4K-task batches in ~10-12 hr, replacing the separate g7e generation cluster).

**Backup discipline (p4de spot reclaim mitigation):**

p4de uses ephemeral NVMe instance-store — it is **lost when the instance is reclaimed**. Writing checkpoints to local NVMe only protects against process crashes, not against spot reclaim. The durable destination is this laptop (and/or S3 as a secondary).

- During training: write checkpoint + optimizer state to `/mnt/nvme` every 1 hr (fast local write), but **also** kick off an async `rsync --partial --append-verify` to the laptop (`~/workbench/self-coding-agent-loop/runs/<iter>/`) as soon as the checkpoint file is fsynced. `--partial` ensures a mid-sync reclaim leaves a resumable file instead of deleting everything.
- At hourly checkpoints, also sync: generated trajectories (`trajectories/iter_N/*.jsonl`), gold eval results (`gold_eval/iter_N.json`), verifier scores (`verifier/iter_N.jsonl`), training logs (`logs/`). These are not reproducible cheaply.
- Spot-reclaim signal: AWS gives a 2-minute warning. Wire a `term-handler.sh` that catches the signal, writes a final checkpoint, and runs `rsync` blocking for up to 100 sec before the instance dies. Any checkpoint that reaches the laptop survives.
- After each iteration (successful or reclaimed): confirm the laptop has the iteration's final checkpoint + all trajectories + all eval results before relaunching. Do not rely on being able to re-attach to the same NVMe — you can't.
- S3 as secondary (optional): for really large runs, mirror to S3 in us-east-1 same region to avoid egress. But the laptop is the authoritative backup, not S3.

### Execution Order (staged, not all-at-once)

```
Week 1:    Gen0 re-baseline on OpenHands v0.54 (SWE-bench Lite 300, ~$15) — establishes the real "must beat" number
           V1b_bootstrap — FlywheelBootstrap on 200 Qwen3.5-27B × OpenHands traces, retrain RF (~$30)
Week 1-2:  V1b_validate (check recalibrated RF reaches V1b-unlock level on Qwen3.5-27B × OpenHands traces, ~$20)
           If V1b_validate fails with ECE trending down, extend to 400-500 bootstrap labels before proceeding.
           A iteration 1 (OpenHands, generate 4K, gold eval, SFT) — runs in parallel with V1b_bootstrap/validate
Week 3-4:  A iteration 2 + evaluate
           If A works: start B iteration 1
           If V1b_validate passes: start C iteration 1
Week 5-6:  Compare A vs B vs C
           If B > A: start D
           Run E (subsampled control)
Week 7-8:  Final iterations, full control eval, write up
```

### Bootstrap (Gen0) — Already Done

| Generation | Source | Gold Pass | Method |
|------------|--------|-----------|--------|
| **Gen0 (current)** | CoderForge 12K gold SFT | **46.7%** (SWE-bench Lite 300) | LoRA SFT |
| Gen1-A | Gen0 + iterative STaR (gold) | Target: 52%+ | Arm A |
| Gen1-B | Gen0 + GRPO (gold) | Target: 53%+ (must beat A) | Arm B |
| Gen1-C | Gen0 + iterative STaR (verifier) | Target: 50%+ (within 3pp of A) | Arm C |

All arms start from the same Gen0 checkpoint. Divergence between arms isolates each variable.

## Loop 3: Agent Behavior & Harness

**Owner**: This blueprint directory (harness configs)

**Purpose**: Optimize the agent scaffolding. This is now largely **answered** by the Nebius OpenHands results — harness choice dominates model choice at this scale.

**Key evidence:**
- OpenHands + Qwen3-Coder-480B → 50.3% (30B student), 61.7% (235B student)
- SERA + Qwen3.5-27B → 46.7% (our baseline)
- OpenCode + Devstral → 22% on our 50-task subset
- Our 8-harness experiment: harness accounts for up to 50pp spread on same model

**Decision: Use OpenHands for training data, validated by Nebius at scale.**

The `nebius/SWE-rebench-openhands-trajectories` dataset was generated with OpenHands v0.54. By training on these trajectories, our model learns OpenHands-style behavior implicitly. For inference, we can use either OpenHands or a lighter harness.

```yaml
# loop3-program.md
objective: minimize wasted_turns while maintaining fix_rate >= 60%
metrics:
  primary: efficiency = gold_pass / tokens_consumed
  secondary: parkinsons_ratio (lower is better, target < 0.4)
  diagnostic: [first_edit_turn, edit_rate, loop_count, context_overflow_rate]

harness_decision:
  # Locked 2026-05-09: all-OpenHands. Train AND eval on OpenHands v0.54.
  # Rationale: Loop 3's primary recommendation + avoids training/inference mismatch.
  # Qwen3.5 is harness-insensitive (16pp spread vs Devstral's 50pp, per agent-swarm
  # memory), so the Gen0 re-baseline from SERA→OpenHands is expected to land in the
  # 45-50% range.
  training_data: OpenHands trajectories (Nebius 32K dataset)
  inference_harness: OpenHands v0.54 (matches training distribution)
  gen0_rebaseline: REQUIRED before Arm A iteration 1 (~$15, one-time; see Execution Order)
  rejected_alternatives:
    - SERA at inference: creates training/inference mismatch; rejected for this experiment
    - Hybrid SERA+OpenHands eval: doubles eval cost (~$80/iter instead of $40)
    - Custom lightweight harness distilled from OpenHands: deferred to post-experiment

interventions:
  turn_pressure:
    variants: [no_pressure, pressure_at_40pct, two_stage]
    best_known: two_stage (edit@40% + verify@55%) — from VP experiment
    measure: fix_rate AND parkinsons_ratio

  context_management:
    variants: [no_trim, trim_at_20K, sliding_window]
    best_known: trim_at_20K chars (from agent-swarm experiment)
    measure: context_overflow_rate, fix_rate

  verification_injection:
    variants: [no_feedback, feedback_on_reject]
    best_known: self-critique HURTS (T5 result), external feedback TBD
    measure: gold_pass_rate, avg_retries

iterate:
  - pick highest-impact intervention (by gap between current and best_known)
  - A/B test on 100 tasks from generation pool
  - if improvement >= 3pp: adopt, update harness config
  - if neutral: discard, try next intervention

terminate_when:
  - parkinsons < 0.4 AND fix_rate >= 70% AND no intervention improves >= 2pp
  - 3 consecutive interventions show no improvement

output:
  - updated harness config (sera_config.yaml)
  - updated system prompt / tool injection rules
```

## Loop 4: Model Architecture & Selection

**Owner**: This blueprint directory (model configs)

**Purpose**: Select the best base model for the current task distribution. Only activated when other loops plateau.

```yaml
# loop4-program.md
objective: find model with highest gold_pass ceiling for given compute budget
metrics:
  primary: gold_pass_rate at fixed compute ($X per eval)
  secondary: tokens_per_gold_pass (efficiency)
  diagnostic: [context_window_utilization, tool_call_accuracy, code_understanding]

candidates:
  current_best: Qwen3.5-27B (46.7% gold pass on SWE-bench Lite 300, 71% precision)

  evaluate_when_triggered:
    - Qwen3.5-27B (baseline, dense 27B)
    - Qwen3-Coder variants (if released)
    - DeepSeek-V3/R2 (MoE, if accessible)
    - Codestral/Devstral next-gen
    - Qwen4 family (when available)

  eval_protocol:
    tasks: 200 from held-out set (same as global eval)
    harness: best from Loop 3
    budget: 30 turns, 65K context
    metric: gold_pass_rate (Docker verified)

decisions:
  - if new model > current by >= 5pp: adopt as new base
  - if new model > current by 2-5pp: run extended eval (full 200 tasks)
  - if MoE vs dense tradeoff: prefer dense at same active params (our finding)

terminate_when:
  - current model > all candidates on held-out eval
  - compute budget for model eval exhausted

output:
  - selected base model for next Loop 2 iteration
  - model comparison report
```

## Interaction Between Loops

```
PHASE 1 (gold tests available):

Loop 2 (RL Training) ──── generates trajectories ────▶ m7i.16xlarge Docker eval
    │                                                  (Arms B/D only; A/C/E use
    │                                                   Nebius pre-computed labels)
    │ trained model                                           │ labels
    ▼                                                         ▼
Loop 3 (Harness) ◀── efficiency metrics            Loop 1 (Verifier Cal.)
Loop 4 (Model)   ◀── plateau detection                    │
                                                           │ ECE certificate
                                                           ▼
                                              Phase 2 gate: RL-ready?

PHASE 2 (no gold tests):

Loop 1 (Reward Model) ←──── drift monitoring samples
    │
    │ verifier scores (the reward signal)
    ▼
Loop 2 (RL Training) ←── harness config from Loop 3
    │                 ←── base model from Loop 4
    ▼
Loop 3/4 ←── plateau detection
```

### Phase Transition Gate

```
Phase 1 → Phase 2 requires ALL of:
  ✓ Arm C ≈ Arm A (within 3pp) — verifier-driven learning works empirically
  ✓ Verifier at RL-ready level on target-distribution traces (ECE < 0.1, precision >= 0.90 — see Verifier Readiness Levels table)
  ✓ Generation pool < 2K tasks remaining (Phase 1 data exhausted)

Note: recall >= 0.20 is NOT required. Low recall (14%) is compensated by
high generation volume. The real gate is empirical: does Arm C converge?
If it does at 14% recall, that's sufficient.

Until gate passes: stay in Phase 1 (free, perfect reward signal)
After gate passes: can operate on ANY task (production deployment)
```

This means: **exhaust the free gold-labeled data first** (Phase 1), then graduate to verifier-based continuous improvement (Phase 2) only once empirically proven (Arm C result, not just theoretical ECE threshold).

## Measured Baseline (Gen0): SWE-bench Lite 300

Full evaluation completed 2026-05-06. This is the starting point the loop must beat.

> **Harness re-baseline required (2026-05-09)**: Gen0 was measured on SERA, but this experiment uses OpenHands v0.54 end-to-end (see Loop 3 §harness_decision). Before Arm A iteration 1, re-evaluate the same Gen0 checkpoint on SWE-bench Lite 300 under OpenHands. Expected: 45-50% (Qwen3.5 is harness-insensitive per agent-swarm findings). All Gen1+ "must beat baseline" comparisons reference the OpenHands re-baseline, not the 46.7% SERA number below. Cost: ~$15, one-time, Week 1.

```
Model:     Qwen3.5-27B + SFT-D (LoRA r=16, CoderForge 12K gold trajectories)
Harness:   SERA, 30 turns, 65K context
Infra:     g7e.12xlarge (2×RTX PRO 6000), vLLM TP2
Cost:      $100 training + $15 eval = $115 total to reach this baseline
```

| Metric | Value | Implication for Loop |
|--------|-------|---------------------|
| **Gold pass rate** | **140/300 = 46.7%** | Gen1 must beat 46.7% |
| Fix rate | 197/300 = 65.7% | High generation rate — rejection sampling viable |
| Precision | 140/197 = 71.1% | Verifier's job is easy (most patches are correct) |
| Parkinson's ratio | 0.76 | Loop 3 target: reduce to < 0.4 |
| Tokens/issue | ~200K avg | Room for efficiency gains |

**Per-repo baseline** (for detecting improvements on hard repos):

| Repo | Pass Rate | n | Notes |
|------|-----------|---|-------|
| django | 84% | 88 | Ceiling; little room to improve |
| pytest | 80% | 10 | Strong |
| scikit-learn | 75% | 8 | Strong |
| sympy | 62% | 53 | Room to improve (+15pp target) |
| matplotlib | 54% | 13 | Mid-range |
| sphinx | 43% | 7 | Harder |
| requests/pylint/astropy | 33% | 12 | Hardest; model lacks domain knowledge |

**Key implications:**
- 71% precision means the verifier only needs to reject 29% of patches → low recall is acceptable
- Sympy alone (53 instances, 62%) offers the most room for improvement on a single repo
- The 34% non-patch rate (103/300 produced no patch) is the other lever — Loop 3 should reduce this
- VP+Sonnet 4.6 got 58.3% on same eval — only 12pp gap to close with a 27B model

## Starting Point: Validation Phase

Before any loop runs autonomously, validate assumptions:

| # | Validation | Pass Criteria | Blocks | Status |
|---|-----------|---------------|--------|--------|
| **M** | **Pre-loop micro-experiments** (E_env, E_attr, E_constraint_agent) | Ran before V1/V1b_* to de-risk Loop 1 design. See [`pre-loop-micros.md`](pre-loop-micros.md). | V1 gate design | **DONE (2026-05-09)** |
| **M2** | **Cross-pipeline transfer test** (E_transfer) | RF trained on claude_opencode_300, evaluated OOD on OpenHands 7 cells + Qwen3.5×OpenCode. | V1/V1b_* gate design | **DONE (2026-05-09)** |
| V1 | Verifier transfers to SWE-ReBench | Precision >= 0.75 on 100 tasks | All loops | NOT STARTED |
| V2 | Claude traces produce learnable signal | Gen0 from Claude traces > random by 10pp | Loop 2 | NOT STARTED |
| V3 | SERA adapter correctly featurizes | RF predictions match manual inspection on 20 traces | Loop 1 | DONE (adapter built) |
| V4 | Gold eval infrastructure works | Docker eval completes 200 tasks < 4 hours | Loop 2 | **DONE** (197 in ~4hrs) |

**M is the pre-flight check.** Three micros totaling ~$50 and 1.5 weeks informed by Eigen AI "Reliable Post-Training" (arXiv 2601.22607). Spec: [`pre-loop-micros.md`](pre-loop-micros.md). Results (2026-05-09, actual spend ~$2):

- **E_env**: `proceed_single_rf` (Δ AUC = -0.144). No per-pipeline ensemble needed on the deployable feature set. The E6 routed-ensemble gain (AUC 0.801) came from model-specific features unavailable at Loop 1 deployment time, not from the architecture itself.
- **E_attr**: The `v009_fail ∧ rf_pass` cell (n=141, gold=72.3% vs agree_fail gold=37.4%) is an "override v009" region, not the adversarial-patch region the spec originally hypothesized. Use its baseline rate (47% on Phase 3) as a drift signal, not as a failure-mode classifier.
- **E_constraint_agent**: `negative_result` (P@R≥0.30 = 0.286, far below 0.80 threshold). No constraint-verifier backup if V1b_rubric fails. *Caveat*: n=43 with only 4 gold-positive patches is underpowered for a definitive answer; two of the four "passes" were style-only accidents that Haiku correctly flagged as non-fixes.

**M2 is the cross-pipeline transfer test.** Spurred by E_env's observation that same-model/different-scaffold cells had wildly different AUCs. Ran the Phase 3 4-feature RF (trained on `claude_opencode_300`, in-sample 5-fold AUC=0.727 on the intersection feature set) out-of-sample on 7 OpenHands cells and the Qwen3.5×OpenCode set. Results:

| Transfer cell | n | pos_rate | AUC |
|---|---|---|---|
| openhands::claude-sonnet (same model, diff scaffold) | 299 | 0.435 | **0.486** |
| openhands::claude-haiku | 299 | 0.288 | 0.411 |
| openhands::gpt-4o | 300 | 0.213 | 0.554 |
| openhands::gpt-4o-mini | 300 | 0.077 | 0.357 |
| openhands::deepseek | 300 | 0.077 | 0.335 |
| openhands::qwen-72b | 300 | 0.077 | 0.444 |
| openhands::llama-70b | 300 | 0.107 | 0.728 |
| qwen35_opencode (diff-only proxies, 5/6 features NaN) | 43 | 0.093 | 0.538 |

Every cell except llama-70b degrades below useful. Critically, same-model/different-scaffold (claude-sonnet on OpenHands) drops from 0.727 in-sample to 0.486 out-of-sample — the RF's probability output becomes near-constant (std 0.018), meaning it falls off its training distribution into a flat leaf region. **V1's "verifier transfers to SWE-ReBench" precondition is now at high risk on features alone.** See `learned-verifier/docs/e_transfer_report.md`.

**V1 is the hard gate.** If the verifier doesn't transfer, the entire meta-loop is blocked. Our current evidence: precision=0.92 on SWE-bench Verified, but SWE-ReBench is a different distribution.

**V4 is validated.** Full 300-instance Docker gold eval completes in ~4 hours on g7e.12xlarge (reusing GPU instance for both generation and eval). No separate m7i needed.

## Compute & Cost

### Phase 1: Gold-Labeled RL

| Step | Compute | Cost/iteration | Iterations |
|------|---------|----------------|------------|
| Generate 4K trajectories (Arms B/D only) | p4de inference (TP=8, between training) | $50 | 2-3 |
| Gold eval (Arms B/D) | m7i.16xlarge spot, ~4K tasks in ~15 hr | $30 | 2-3 |
| Gold eval (Arms A/C/E) | Nebius `resolved` column (pre-computed) | $0 | — |
| GRPO training | p4de.24xlarge spot (~13-17 hrs) | $230 | 2-3 |
| Held-out eval (5K control, from Nebius) | Nebius `resolved` column | $0 | per-iter |
| Loop 1 (verifier cal.) | Haiku API + local RF | $15 | 2-3 (passive) |
| Loop 3 (harness A/B) | p4de inference between GPU cycles | $0 | 1-2 |
| **Phase 1 subtotal** | | | **~$1,615 minimum, ~$3,855 full 5-arm** |

### Phase 2: Verifier-in-the-Loop (Continuous)

| Step | Compute | Cost/iteration |
|------|---------|----------------|
| Generate on new tasks | p4de TP=8 between training | $50 |
| Verifier scoring | Haiku API (5K × $0.03) | $150 |
| GRPO training | p4de.24xlarge spot | $230 |
| Drift monitoring | Haiku + m7i Docker eval (100 tasks) | $15 |
| **Phase 2 per-iteration** | | **~$445** |

### Total Budget

| Phase | Cost | Duration |
|-------|------|----------|
| Already spent (Gen0 baseline) | $175 | Done |
| Phase 1 (minimum viable: A + V1b_bootstrap + V1b_validate + C, 2-3 iters) | ~$1,615 | 4 weeks |
| Phase 1 (full 5 arms, 2-3 iters each) | ~$3,855 | 6-8 weeks |
| Phase 2 (continuous, 3 iterations) | ~$1,335 | Ongoing |
| **Total (minimum viable + Phase 2)** | **~$3,125** | 7-8 weeks |
| **Total (full + Phase 2)** | **~$5,365** | 10-12 weeks |

## Success Criteria

1. **Iterative STaR works (Arm A)**: Gen1-A > Gen0 (46.7%) by >= 3pp on 5K control set
2. **RL adds value (Arm B vs A)**: B > A by >= 2pp (justifies GRPO complexity over STaR)
3. **Verifier viable (Arm C vs A)**: C within 3pp of A (continuous learning works)
4. **Verifier tax isolated (Arm E vs C)**: Identifies whether gap is noise or sparsity
5. **Match published results**: Best arm reaches >= 55% on control set (CoderForge territory)
6. **Generalization**: Control pass rate improves while train-control gap < 15pp (not memorization)
7. **Verifier reaches V1b-unlock on target traces (V1b_bootstrap → V1b_validate)**: prerequisite for Arms C/D/E
8. **Efficiency**: Minimum viable experiment (A + V1b_bootstrap + V1b_validate + C) < $900

## Non-Requirements

- Multi-language support (Python-first)
- Novel verifier architecture (iterate v009, don't reinvent)
- Human-in-the-loop after validation phase
- Full SWE-ReBench coverage (sample, don't exhaust)
- Real-time serving (batch optimization loop)

## Known Limitations

- **Gold eval bottleneck**: Nebius removed this at scale by pre-computing the `resolved` column on the OpenHands 67K dataset. Arms A/C/E need no Docker runs. Arms B/D still run ~4K Docker evals per iteration on m7i.16xlarge spot (~10-20 hr, ~$30). If we later need to gold-eval at 32K scale on trajectories Nebius didn't cover, self-hosted m7i scales linearly ($0.008/task) but wall-clock becomes the constraint (32K tasks = ~80-160 hr on a single m7i, or parallelize across multiple spot boxes).
- **GRPO on coding tasks is unproven at this scale**: Most published RL results (DeepSeek-R1, Qwen3-Coder) use proprietary setups. Our contribution is showing it works with open tooling.
- **Mode collapse risk**: RL on binary reward may narrow the model's strategy space. Monitor fix rate diversity and Parkinson's ratio across iterations.
- **Phase 2 depends on verifier quality**: If ECE never reaches < 0.1, Phase 2 is blocked. But Phase 1 alone may get us to 55-60% — enough to match published results.
- **Recall ceiling (14%)**: In Phase 2, verifier recall limits training signal. Compensate with generation volume (generate 10x, verifier keeps top 14%).
- **Outer scheduler is manual initially**: First few cycles, human decides bottleneck. Automate after patterns emerge.

## References

- SWE-ReBench V2: arxiv 2602.23866, `nebius/SWE-rebench-V2` (HuggingFace collection)
  - Blog: https://nebius.com/blog/posts/meet-swe-rebench-v2
  - Infrastructure: https://nebius.com/blog/posts/infrastructure-behind-swe-rebench
  - Eval: Nebius ships pre-computed `resolved` labels on `SWE-rebench-openhands-trajectories`; self-host Docker on m7i.16xlarge spot for new trajectories
- Learned verifier repo: `/Users/phi/Documents/workbench/learned-verifier/`
- Cascade verifier experiments: `domains/autoresearch/blueprints/verifier-reward/`
- Rejection sampling SFT: `domains/autoresearch/blueprints/rejection-sampling-sft/`
  - Full 300 eval results: `results/qwen35-sft-d/gold_eval_consolidated.json`
- Verification primitives: `domains/autoresearch/blueprints/verification-primitives-swebench/`
- CoderForge: Together AI, 155K trajectories, 59.4% SWE-bench Verified
- SERA-32B: Allen AI, arxiv 2601.20789
- EntroPO: 60.4% Verified with DPO on Qwen2.5-Coder-32B

---

> **Note**: Operational artifacts (lessons, results, sub-loop program.md files)
> belong in the blueprint directory at `domains/autoresearch/blueprints/self-coding-agent-loop/`.
