# Autoresearch Spec: E_harness2 — Layered Ablation of Harness Authoring (JIT vs offline, self vs external)

## Status: DRAFT

## Overview

A **layered ablation** isolating two axes of the *intervention* design space that Life-Harness and our own verifier work jointly opened up: **when** a harness is authored (offline-evolve-then-freeze vs runtime/JIT) and **who** authors it (the worker self-correcting vs an external verifier agent watching it). Each layer adds exactly one variable so the *delta* is the measurement.

Run on **DBBench** (AgentBench's SQL env — deterministic, rule-governed, needs only a SQLite/MySQL harness + an LLM endpoint; the lightest of Life-Harness's environments), driven by **Bedrock Claude via the OpenAI Agents SDK** (`agents[litellm]` → `litellm/bedrock/...`). No GPU serving; API-driven; runs detached on the agent runtime.

**Why this matters (the thesis it tests):** Life-Harness builds the harness *offline* — a Codex agent reads a frozen model's *training* failures, authors layer interventions, freezes, and reuses across 18 models. We test whether that offline pass is necessary, and whether **authoring locus** (self vs external) is the dominant axis — predicted by our verifier-reward T5 result (self-critique in generation *hurts*: fix rate 54%→30%; "the model finds bugs in others' code but not its own"). If external > self holds for *harness authoring* as it did for *verification*, both obey one law: **the engine that builds/checks an agent must be external to the agent it builds/checks.**

**Verification is the meta-primitive.** The other intervention types (contract, skill, action-guard, trajectory-regulation) are failure-mode-*specific* patches; verification is failure-mode-*general* — it *detects* which interventions to author. The external-authoring arm (L3) is therefore a verifier driving harness construction in-loop. This spec measures whether that authoring-verifier's signal transfers across the 2 worker models but needs per-model recalibration — the [[e-fin1-finqa-skill-verifier]] / [[e-fin2-finqa-behavioral-features]] transfer law applied to the authoring engine.

**Depends on**: Life-Harness repo (github.com/Tianshi-Xu/Life-Harness, CC-BY; the DBBench harness + AgentBench fork) for L1; verifier-reward T5/T4 priors; the agent-runtime (`managed-agent-runner`) for detached execution.

**Source**: derived from [[Life-Harness-Runtime-Interface-Adaptation]] + the conversation's intervention-dimension synthesis (2026-06-21).

## The layered ablation (each step = one isolated delta)

| Layer | Adds | Isolates | Reference |
|-------|------|----------|-----------|
| **L0 — bare** | agent on DBBench, no harness layers | floor | Life-Harness `w/o` ≈ 48.4% Pass@1 |
| **L1 — offline-frozen** | Life-Harness harness, evolved from train failures, frozen for test | **does the harness help at all** (replication GATE) | Life-Harness `w/` ≈ 64.6% |
| **L2 — JIT self** | the worker authors its own interventions in-loop from its own failures (no train/freeze phase) | **authoring time**: offline → runtime (self-locus) | — |
| **L3 — JIT external** | a *separate* verifier agent watches the worker and authors interventions in-loop | **authoring locus**: self → external | — |

**The three measured deltas:**
- **L0→L1**: replication gate. If this does not reproduce roughly +12–16pp, the DBBench wiring is wrong — STOP and fix setup before trusting L2/L3.
- **L1→L2**: is the offline evolve-then-freeze phase necessary, or does runtime JIT recover its gain?
- **L2→L3** (the headline): does *external* authoring beat *self* authoring? **Prediction: L3 > L2**, per the T5 self-critique-hurts prior. If so, harness authoring obeys the same external>self law as verification.

## Research Questions

1. **L0→L1**: Does the Life-Harness harness reproduce its DBBench gain on Bedrock Claude (a model not in their 18-backbone set)? (Also a cross-model-transfer datapoint for *their* claim.)
2. **L1→L2**: Can runtime JIT authoring recover the frozen-offline harness's gain without a separate training phase, or does offline evolution capture something self-correction cannot?
3. **L2→L3**: Is authoring locus (external vs self) the dominant axis? Does an external verifier-authored harness beat a self-authored one — the T5 law, applied to harness construction rather than patch generation?
4. **Transfer of the authoring engine**: across 2 worker models, does the external verifier's failure-detection signal port, but its threshold/calibration need re-fit? (Transfer law on the meta-primitive.)

## Components

### 1. Compute
- **Platform**: API-driven (Bedrock via OpenAI Agents SDK + LiteLLM). **No GPU.** Runs as an `agent-runner` batch job (detached) or locally.
- **GPUs**: none.

### 2. Data / Code
- **Env**: DBBench from the Life-Harness repo (`AgentBench/`, tasks in `data/dbbench/*.jsonl` — `standard.jsonl`/`dev.jsonl`). SQLite/MySQL execution is the deterministic ground-truth oracle (analog of the Docker/exact-match oracle in prior experiments).
- **Harness (L1)**: Life-Harness's 4 layers as integrated in their AgentBench fork — used frozen.
- **Models**: 2 Bedrock Claude tiers as the *worker* (e.g. Haiku 4.5 + Sonnet/Opus) to test authoring-engine transfer (RQ4). The *external authoring agent* (L3) is a separate Bedrock Claude instance.

### 3. Experiment Protocol
- **Metric (primary)**: DBBench Pass@1 per layer; the three deltas (L0→L1, L1→L2, L2→L3) with bootstrap CIs. Secondary: per-failure-type breakdown (which Life-Harness layer/failure class each arm recovers), cost/eval, authored-intervention count.
- **Ground truth**: SQL execution correctness (DBBench's built-in oracle). Free, objective, no human review.
- **Loop structure**:
  - L0: run worker bare over the DBBench eval split.
  - L1: apply the frozen Life-Harness harness (their offline-evolved layers); re-run.
  - L2: worker runs with an in-loop self-authoring step — after a failed task, it writes interventions (contract notes / action-guards / skills) into a session-scoped harness state applied to subsequent tasks. SDK in-loop primitives (apply-patch / shell / tool-result hooks), not a post-run pass.
  - L3: a separate verifier agent observes the worker's trajectory, classifies the primary failure type (the meta-primitive), and authors the intervention the worker then uses. Same in-session application as L2; only the *author* differs.
- **Holding fixed across L2/L3**: same DBBench split, same task order, same worker model, same intervention *types* available — only authoring time (L1 vs L2) and locus (L2 vs L3) vary.
- **Termination**: all 4 layers scored on the same eval split for both worker models.
- **Logging**: per-task {layer, model, pass/fail, failure_type, interventions_authored, tokens, cost}; per-layer aggregate Pass@1 + deltas + CIs.

### 4. Networking
- **Access**: outbound to Bedrock (IRSA if on the runtime). DBBench MySQL/SQLite runs in-container/local.

### 5. Storage
- **Data**: Life-Harness repo cloned (small); DBBench task splits.
- **Results**: blueprint `results/` (per-layer Pass@1, delta table, failure-type breakdown, transfer check).

## Success Criteria

- [ ] **Stage-0 SDK smoke test**: OpenAI Agents SDK (`agents[litellm]`) drives Bedrock Claude through one real DBBench task end-to-end (tool calls execute, SQL runs) BEFORE the full run. The SDK is GPT-native; this gate de-risks the off-OpenAI-model path.
- [ ] **L0→L1 replication gate passes**: the frozen Life-Harness harness reproduces a material DBBench gain (~+12–16pp) on Bedrock Claude. If not, setup is broken — do not report L2/L3.
- [ ] L0/L1/L2/L3 Pass@1 measured on the same eval split, both worker models, with bootstrap CIs.
- [ ] **Three deltas reported** (L0→L1, L1→L2, L2→L3) with the headline verdict on L2→L3 (external vs self).
- [ ] Per-failure-type breakdown: which arm recovers which Life-Harness failure class.
- [ ] Transfer check (RQ4): does the L3 authoring-verifier's signal port across the 2 models / need recalibration?
- [ ] Carryover audit complete (below).

## Non-Requirements

- **No GPU serving / no open-weight self-hosting** — Bedrock API only (DBBench is light; this is deliberately the cheapest Life-Harness env).
- **No reproduction of Life-Harness's full 18-backbone × 8-env sweep** — one env (DBBench), the layers used as-is for L1.
- **No weight updates / fine-tuning** — frozen models; the whole point is interface-side adaptation.
- **No GPT/OpenAI-model arm** — on-thesis stack is Bedrock; SDK-portability to Bedrock is itself part of the test.

## Known Limitations

- **The arXiv ID (2605.22166) is future-dated**; Life-Harness numbers are taken at face value. Stage 0 must confirm the DBBench harness code actually runs and reproduces *directionally* before L1 is trusted as a baseline.
- **The repo ships NO trajectory corpus** — all traces are generated here by running the agent. L0/L1 generation is the cost floor; cheaper than GPU rollouts but not free.
- DBBench is one deterministic env; a result here bounds the JIT/locus claim to rule-governed domains (Life-Harness's own stated limit — open-ended tasks excluded).
- The OpenAI Agents SDK is GPT-native; any L2/L3 underperformance could be an SDK-on-Bedrock artifact rather than a real authoring-axis effect — the Stage-0 smoke test and the L0/L1 replication gate are the guards against misreading that.
- Self-authoring (L2) may be confounded by the worker's context window filling with its own intervention notes — cap/measure intervention state size.

## Carryover Audit (spec-design gate)

- [ ] Ran `carryover-auditor` — scan `verifier-reward/lessons.md` (T5 self-critique-hurts; T4 calibration is model-specific), `verification-primitives*/lessons.md` (checkpoint/adoption), `agent-harness/lessons.md`, and `e-harness1`/`e-fin1`/`e-fin2` results (the transfer law).
- [ ] Carry the priors explicitly: **T5 — self-critique in generation HURTS (54%→30%)** is the basis for the L2<L3 prediction; **T4 — calibration is model-specific** is the basis for RQ4; **E_fin1/E_fin2 — primitives port, calibration doesn't** is the law this extends to harness authoring. The macOS **python3.14 sklearn broken → use python3.13** lesson applies if any local RF/analysis is done.
- [ ] Confirm Stage-0 SDK-on-Bedrock smoke test is a hard gate (it is) — do not anchor L1 on Life-Harness numbers without a directional local reproduction first.

---

> **Note**: Operational artifacts belong in `domains/autoresearch/blueprints/e-harness2-jit-vs-offline-authoring/`, not in this spec.
