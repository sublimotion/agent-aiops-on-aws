# Autoresearch Spec: Kernel Optimization Mini-Loop (Generalized, Headroom-Gated)

## Status: DRAFT

## Overview

A reusable, target-agnostic wrapper around the five-primitive kernel-optimization
harness (`generate → cascaded L0–L4 verify → select → telemetry → constraint DB`)
proven in [`kernel-optimization-agent`](kernel-optimization-agent.md). That blueprint
hard-wired the harness to one target (Kimi K2.6, 384-expert MoE + MLA, H200→B300).
This spec generalizes it so the same loop can be pointed at **any (model, kernel, GPU,
quant, phase) tuple** — with one new mandatory gate that the K2.6 run taught us the
hard way.

**The one thing this spec adds over the parent: a fail-closed profile-for-headroom
gate (Stage A) that must pass before a single candidate is generated.** The K2.6
run burned Phase-1 budget discovering *after the fact* that MLA decode was already at
100–122% BW utilization (no headroom) and MoE dispatch was memory-bound at 4.6% BW by
architecture — concluding stock vLLM was near-optimal *for that regime*, with custom
kernels yielding <1% e2e. See `kernel-optimization-agent/results/report.md:466,482,507`.
The lesson: **a kernel mini-loop is only worth running where a roofline shows real
headroom, in the exact regime you will serve.** This spec makes that a gate, not a
discovery.

Grounding docs (read before running):
- Regime-scoping law and the K2.6 findings: `docs/inference-optimization-guide.md` §12
  (caveat box) — every "near-optimal" claim is scoped to `hardware × kernel ×
  concurrency × quant × phase`; never inherit across regimes.
- Attention family → serving behavior: `docs/inference-optimization-guide.md` §15.
- Loop contract (objective, held-out gate, plateau/budget exit, reward-hacking guard):
  `standards/benchmark-commons/OPTIMIZATION-LOOP.md`.
- Five primitives in detail: `kernel-optimization-agent.md` §Verification Framework
  Mapping (reused verbatim — do NOT redefine here).
- Hardware/runtime quirks: `.claude/steering/tech-stack.md` (sm_120 NCCL version rule,
  nerdctl/`--network host` convention, driver/CUDA pairing).

## Verification Framework Mapping

**Reused unchanged from [`kernel-optimization-agent.md`](kernel-optimization-agent.md)**
(§Primitives 1–5): Generator (LLM emits Triton/TileLang/CUDA, self-advancing State
Vector 0–5), Verifier (cascaded L0 parse → L1 compile → L2 correctness → L3 ncu profile
→ L4 e2e, short-circuit on failure, speedup CI must exclude 1.0), Selector (top-K
leaderboard, champion-beating promotion, cherry-pick upstream PRs individually),
Telemetry (append-only JSONL feature vectors), Constraint DB (append-only hard/soft/
positive rules + Freeze Manager plateau detection).

This spec changes only the **envelope** around those primitives: (1) the Stage A
headroom gate, (2) a target-descriptor so the loop is parameterized rather than
K2.6-hardcoded, and (3) g7e/sm_120 as the default first hardware.

## Components

### 1. Compute

- **Platform**: bare-metal GPU (operator workstation, NOT agent-runner — kernel work
  needs nerdctl/ncu/root, and `agent-runner` has no infra perms).
- **Default first target**: **g7e.24xlarge** (4× RTX PRO 6000 Blackwell, sm_120, 96GB
  GDDR7, PCIe). Rationale: it is the hardware we own, and it is where kernels are
  genuinely *missing* — per `SERVING_COMPAT_MATRIX.md`, sm_120 has **no working
  FlashMLA / TRTLLM-MLA** (BF16-only KV), so MLA-family models fall back to Triton.
  That is real, documented roofline headroom, unlike the K2.6/H200 MoE result.
- **Other valid targets** (per headroom, not default): B200 (sm_100), B300 (sm_103),
  p5en H200 (sm_90) for architecture-portable algorithmic work.
- **Runtime** (g7e bare-metal traps, from devstral-sera / qwen3-next-g7e lessons):
  - `nerdctl` (not docker). **ALL** container commands need `--network host` (no CNI
    plugin on bare metal → silent DNS failure otherwise).
  - GPU flag uses count syntax `--gpus <N>`, not device IDs; do NOT combine `-d` with
    `--rm` (nerdctl rejects it).
  - PYTHONPATH-NVMe trick for persistent installs without image rebuild.
- **NCCL gate (P0, sm_120)**: NCCL 2.25.1 has a shared-memory bug on Blackwell sm_120 —
  ALL collective ops fail (`NCCL_P2P_DISABLE=1` does not help); fixed in 2.26.2 (NGC
  25.03+). Single-GPU kernel work is unaffected, but **any TP>1 L4 serving benchmark
  must either use NCCL ≥2.26.2 OR run TP=1 OR confirm vLLM custom-allreduce is active**
  (vLLM inference uses custom allreduce, so TP>1 serving can work — verify no NCCL
  collective in the path). Multi-GPU *profiling* (distributed roofline) needs ≥2.26.2.
  (devstral-sera/lessons.md NCCL entry.)
- **Profiling**: NSight Compute (ncu), NSight Systems (nsys). Full ncu roofline on
  sm_120 may be partial — fall back to achieved-BW vs peak-BW from nsys + a
  hand-computed roofline; record `method_used` (see Stage A output schema).

### 2. Codebase

- **Source**: the harness/skills from `kernel-optimization-agent` blueprint
  (`skills/profile-kernel`, `generate-candidate`, `verify-kernel`, `manage-constraints`,
  `cherry-pick-eval`, `diagnose-bottleneck`), generalized to read a target descriptor.
- **Fixed files (agent must NOT edit — they define the metric)**: the correctness
  reference (PyTorch eager implementation of the target kernel), the L2 input generator,
  the L4 serving benchmark harness + workload card, the held-out quality eval.
- **Agent-editable files**: candidate kernel sources under `candidates/`, the constraint
  DB, telemetry JSONL.
- **Agent instructions**: `program.md` in the blueprint (per target).

### 3. Experiment Protocol

**Stage 0c — Config validation (fail-closed) + freshness check. Before Stage A.**

- If the target uses a serving engine for L4, run `python3
  standards/serving-commons/resolver/validate-serving-config.py --sidecar benchmark.yaml
  --corpus-root .` — exit 2 blocks the loop; fix the config first.
- **Freshness re-check of the headroom premise (MANDATORY, fail-closed)**: the "no
  working FlashMLA/TRTLLM-MLA on sm_120" claim in `SERVING_COMPAT_MATRIX.md` is
  field-dated and may go stale. Run `mdc prs flashinfer` + `mdc prs sglang` and record
  the result as `premise_stale: true|false` in the Stage A output. If a sm_120 MLA kernel
  landed since the matrix was dated, the premise is void — **STOP and re-scope the
  target**; do not proceed to Stage A profiling on the stale premise.

**Stage A — Headroom gate (NEW, MANDATORY, FAIL-CLOSED). Runs before any candidate.**

*Pre-A (model download, if not cached):* export `HF_TOKEN="$(cat
~/.cache/huggingface/token)"` (the CLI does NOT auto-read the token file — unauth mode
drops throughput ~10×); set `HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0` (Xet
silently stalls on large multi-shard models); download in a retry loop as PID 1; clear
stale `*.lock` before retry. (kimi-k2.6-nvfp4 #L5, kimi-k2.6-speculative #L15.)

1. Profile the target kernel in the **exact serving regime** (hardware, quant, phase,
   concurrency) with ncu/nsys. Produce a roofline classification: compute-bound,
   memory-bound, or latency/overhead-bound; and the achieved vs peak utilization. Record
   `method_used` (ncu_full / nsys_achieved_bw / gauge_inferred) as you profile — per the
   output schema below.
1b. **Verify the active backend, and measure the fallback baseline — do NOT assume a
   fallback path is slow.** Confirm via engine logs/flags which backend is actually
   running (e.g. Triton MLA on sm_120). Then measure *that baseline's* achieved BW vs
   peak. **This is the spec's biggest risk**: MLA decode is memory-bound at arithmetic
   intensity ≈1.0 at *all* AIs (kernel-optimization-agent report.md:507 — H200
   FlashInfer MLA was already >100% BW-util, no kernel rewrite helped). A Triton sm_120
   fallback that already achieves high BW is **near-optimal, not headroom.** PASS on an
   MLA/BW-bound target ONLY if the measured fallback shows **<70% BW utilization** in
   the target regime. "The optimized path doesn't exist on sm_120" is *not* evidence of
   headroom — only a measured sub-ceiling baseline is.
2. Compute the **e2e headroom ceiling**: `max_e2e_gain ≈ (1 − achieved_util) ×
   (kernel_share_of_e2e_time_at_target_concurrency)`. The kernel share comes from an
   nsys wall-clock breakdown at the target concurrency (this is the number K2.6 got
   wrong by intuition — MoE was 4.6% BW but only 3% of e2e, so ceiling was <1%).
3. **Gate decision** (fail-closed):
   - `PASS` if the kernel is not at its roofline ceiling (per 1b) AND `max_e2e_gain ≥
     threshold` (**default 10% e2e** — kernel-dev is expensive at ~15 min/candidate;
     match `optimization-stack.md`'s structural-lane criterion, not the 5% tier-refine
     one, so a launched loop targets something worth ~one node).
   - `FAIL` → **do not generate candidates.** Record the roofline verdict + the regime
     tuple as a `dead-<n>` constraint and a §12-style scoped finding, and STOP. A
     FAIL here is a *valid, cheap result* (it is the K2.6 outcome, reached in hours
     not days).
4. If PASS, seed the constraint DB with: (a) hardware facts (sm_120 smem/BW/instruction
   availability, missing-backend facts from `SERVING_COMPAT_MATRIX.md`); (b) the roofline
   target (achieved vs peak, bottleneck class); (c) **inherited dead-ends from the
   parent** — notably `dead-003`: DeepGEMM/FlashMoE are EP kernels requiring NVSHMEM +
   `torchrun --nproc N`, not callable in a single-GPU loop; skip in cherry-pick. Then
   proceed to Stage B.

**Stage A output schema** (`results/stageA-roofline-<target>.json`) MUST include
`method_used: [ncu_full | nsys_achieved_bw | gauge_inferred]` — ncu roofline may be
partial on sm_120 / driver-580 (kimi-k2.6-nvfp4 #L8 saw DCGM PROF unavailable). On
fallback, record the hand-computed AI + peak FLOPS/BW; on gauge-inferred, record which
engine gauges and why ncu/nsys failed.

**Stage B — Optimization loop (parent harness, unchanged).**

```
while budget remaining AND not all regions frozen:
  GENERATE candidate at current State Vector position (0→5)
  VERIFY cascaded L0→L1→L2→L3→L4 (short-circuit on first fail)
  RECORD telemetry (pass or fail) to JSONL
  UPDATE constraint DB
  SELECTOR: promote iff L4 speedup CI excludes 1.0 AND beats champion
            AND held-out quality gate PASSES (fail-closed)
  if 3 consecutive non-improvements at current State: ADVANCE State Vector
  if 3 consecutive non-improvements across ALL States: FREEZE region, redirect
```

- **Objective** (declare one, per `OPTIMIZATION-LOOP.md`): e.g. `output_tokens_per_sec
  @ c=<target>` (throughput regime) or `TPOT_ms @ c=1` (latency regime). Stated with
  its full regime tuple.
- **Held-out quality gate** (reward-hacking guard): correctness eval the loop never
  optimizes against; breach → `quality_breach`, candidate cannot be promoted, throughput
  not even measured. Invariants (output correctness, no modality dropped) are not
  tradeable.
- **Search space**: tile sizes (M∈[16,64], K∈[64,256], N∈[64,256]), num_warps∈{4,8,16},
  num_stages∈{2,4,6}, fused/separate, persistent/single-shot — bounded per target by
  Stage A constraints.
- **Correctness tolerance**: rtol=1e-3 (FP16/BF16), 1e-2 (FP8/FP4 + 100-input +
  distribution check); reject if any input exceeds 5× tolerance.
- **Time budget**: ~15 min/candidate (L0 1s + L1 10s + L2 30s + L3 2min + L4 5min);
  ~32/session.
- **Termination**: all regions frozen OR budget exhausted OR plateau (min_improvement
  not met for `patience` consecutive promotions).

**Collaboration mechanics** (≥8 candidates): shared append-only candidate/telemetry
JSONL; candidate schema = {hypothesis, parent, lever_delta, expected_value, verifier_level,
stop_condition, state, status}; novelty gate rejects duplicate code/config hashes and
repeated dead-ends; per-wave portfolio split across State-Vector exploration vs champion
exploitation vs cherry-pick vs diagnose; critic merge after each wave dedupes lineage,
re-checks the held-out gate, updates the frontier; trace capture stores *why* each
candidate was tried.

**Trajectory record** (handoff to outer loop): per `OPTIMIZATION-LOOP.md` — objective,
regime tuple, guardrail, and a lineage of nodes {id, parent, lever_delta, confidence,
objective_value, guardrail_value, status}.

### 4. Networking

- SSH to operator workstation (e.g. g7e: `ssh -i ~/.ssh/g7e-bench.pem ec2-user@<ip>`).
- No cluster scheduling from within the loop (no agent-runner infra perms).

### 5. Storage

- **Data**: model weights + profiling artifacts on `/mnt/nvme`.
- **NVMe pre-flight (P1)**: on fresh g7e spot nodes the instance-store NVMe is raw and
  unmounted — RAID-0 + `mkfs.xfs` + mount at `/mnt/nvme` first (via privileged
  `nsenter -t 1 -m` pod if on EKS). Spot reclaim wipes it → re-run on every fresh node.
  K8s pods mounting `/mnt/nvme` need `mountPropagation: HostToContainer` when the mount
  happens post-kubelet, else pods see "No such file or directory" (kimi-k2.6-nvfp4
  #L1/#L12).
- **Results**: `results/telemetry.jsonl` (append-only), `results/stageA-roofline-<target>.json`
  (the gate verdict incl. `method_used` — kept even on FAIL), constraint DB, leaderboard.

## Success Criteria

Concrete, testable — note that a Stage-A FAIL is a success (cheap, correct null):

1. **Stage A gate ran and produced a scoped roofline verdict** for the target regime,
   stored as `results/stageA-roofline-<target>.json` (with `method_used`), before any
   candidate generation — including a *measured* baseline BW-utilization for the active
   backend (step 1b), not an assumption that a missing optimized path implies headroom.
2. If Stage A `FAIL`: loop did NOT proceed; a §12-style scoped finding + `dead-<n>`
   constraint recorded. (This is the ideal-cost outcome for a no-headroom target.)
3. If Stage A `PASS`: ≥1 promoted candidate with L4 speedup CI excluding 1.0 AND
   held-out quality gate intact, OR a documented plateau/budget exit with lineage.
4. Every performance claim in the blueprint states its full regime tuple (per §12 law).
5. Constraint DB seeded with ≥ the target's hardware facts + roofline; telemetry has
   full feature vectors for every candidate (pass and fail).
6. (If ≥2 targets run) convergence measured: candidates-per-promotion, first-decile vs
   last-decile, to quantify constraint-DB learning.

## Non-Requirements

- Not a replacement for `kernel-optimization-agent` — that stays the K2.6/MoE+MLA
  reference instance. This spec is the reusable envelope + headroom gate.
- Not multi-node / EP / disagg kernel work (frontier-only — see
  [[project_pd_disagg_frontier_only]]).
- Not a learned selector (that is the parent spec's Phase 4 follow-on; this spec only
  produces the telemetry it would train on).
- Does NOT relaunch the blocked CuTe-DSL NVFP4 MoE-GEMM A/B — that is gated on upstream
  merges (FlashInfer #3645, SGLang #28354), tracked in
  `domains/gpu-serving/blueprints/kimi-k2.6-cutedsl-moe/`. If chosen as a target, Stage 0c
  resolver + the parent's blocked-status check apply first.

## Known Limitations

- sm_120 NCCL is broken (2.25.1) — irrelevant for single-GPU kernel work, but any TP>1
  L4 serving benchmark must use vLLM custom-allreduce path or TP=1.
- ncu roofline may be partial on sm_120; fall back to nsys achieved-BW + hand-computed
  roofline and record the method.
- Blackwell-native paths (TMA, tcgen05, FP4/NVFP4) do not exist on sm_120 the way they
  do on sm_100/103 — a g7e win may not transfer up, and a B200/B300 win may not transfer
  down. Regime-scope every result.

## First Target (this launch)

**First Target hypothesis (pending Stage A measurement): g7e sm_120, MLA-family model on
the Triton fallback path** (e.g. a Kimi/DeepSeek MLA model, or validating MSA's sm_120
sparse/FP4-indexer claims — see §15 watch-items). `SERVING_COMPAT_MATRIX.md` documents no
working FlashMLA/TRTLLM-MLA on sm_120, so MLA models fall back to Triton — **but a missing
optimized path is a hypothesis of headroom, not evidence of it.** The loop is justified
*only if* Stage A step 1b *measures* the Triton fallback at <70% BW utilization; MLA
decode is BW-bound at all AIs, so the fallback may already be near-ceiling — the honest
K2.6-style null. Stage 0c must first re-check via `mdc prs` that no sm_120 MLA kernel
landed since the compat matrix was dated. Stage A decides whether the gap is real before
any code is generated.

## Carryover Audit (spec-design gate)

Before running, confirm no prior lesson was left behind:
- [ ] Ran `carryover-auditor` (or equivalent self-check) over every `domains/**/lessons.md`
      whose stack overlaps this experiment — especially `kernel-optimization-agent`
      (the near-optimal-is-regime-scoped lesson), `kimi-k2.6-cutedsl-moe` (upstream-blocked,
      do-not-relaunch), `devstral-sera` (sm_120 NCCL breakage), and any g7e blueprint.
- [ ] Every applicable prior lesson — especially `outcome: failure`/`partial` — is a
      protocol step, environment check, or success criterion here, OR noted as N/A with
      its source (`<blueprint>/lessons.md` #N).

---

> **Note**: Operational artifacts (lessons, results, roofline verdicts, analysis) belong
> in the blueprint directory, not in this spec.
