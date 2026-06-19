# Self-Coding Agent Loop — Session Summary

**Date**: 2026-05-10 through 2026-05-11
**Status**: Round 1 SFT succeeded; generate + eval pipeline incomplete; experiment terminated for product-first wrap-up
**Total spend**: ~$355 (p4de spot 26hr @ $13.59/hr + m7i + Bedrock Haiku)

---

## What we actually produced

### Working artifacts
- **Gen1 LoRA adapter** on `Qwen/Qwen3-Coder-30B-A3B-Instruct` (13M trainable params), saved to `s3://agent-aiops-artifacts/self-coding-agent-loop/runs/round_1/adapter/`
  - Trained on 4,258 gold-passing Nebius OpenHands trajectories (round_1_train split, 496 unique instances)
  - 267 steps, loss 0.98 → 0.62, token accuracy 76% → 84%, 1 epoch, LoRA r=16 α=32 lr=2e-5 max_seq 8192
  - Training: 3.6hr on 8× A100 80GB (p4de.24xlarge spot)
- **Instance-level rolling-rounds data splits** (6,306 Nebius instances partitioned into 5 round_N_train × 5 round_N_control × final_stress_500 × drift_audit_300 × v1b_bootstrap_200, SHA-pinned manifest)
- **Nebius OpenHands trajectories** (2.1GB parquet, 67K trajectories, 32K gold-passing) — locally cached + S3 mirrored

### Product deliverables (the actual output of this session)
- [runbook-continuous-improvement.md](runbook-continuous-improvement.md) — operational playbook: when to use, SLO thresholds, decision tree, infrastructure minimums, launch procedure
- [failure-modes.md](failure-modes.md) — **22 real failures** cataloged across 6 layers (AWS, AMI, Python/ML, Data, Ops, Model-family). ~5 hours of debug time spent translated into ~5 minutes of reading for the next team.
- [cost-calculator.md](cost-calculator.md) — $/pp-improvement model, break-even points vs alternatives (human review, Docker production, scale-up)
- [experiment-backlog.md](experiment-backlog.md) — 13 deferred experiments with explicit triggers, dependencies graph, unlock tiers
- [spec-explainer.html](spec-explainer.html) — interactive visual explainer of the two-phase architecture, rolling rounds, drift trajectory concept
- [pre-loop-micros-results.md](pre-loop-micros-results.md) — E_env, E_attr, E_constraint_agent, E_transfer results from the 4 micro-experiments that ran before this launch

### Infrastructure code
- `scripts/orchestrator.sh` — 5-round (configurable) orchestrator with overlap of round N+1 SFT and round N Docker eval, plateau detection, Phase 2 gate check
- `scripts/round_runner.py` — per-round SFT + generation (VLM-aware, fresh-LoRA fallback, SIGUSR1 checkpoint handler, parquet join for trajectory)
- `scripts/verifier_recalibrate.py` — Loop 1 recalibration, drift trajectory appender
- `scripts/docker_gold_eval.py` — **rewritten** to use SWE-rebench-V2 eval harness (not SWE-bench — they are different)
- `scripts/build_splits.py` — instance-level partitioner with invariant assertions
- `scripts/drift_trajectory_report.py` — aggregates the 24-datapoint drift curve and computes Phase 2 gate
- `scripts/launch_p4de.sh` + `user_data.sh` + `term_handler.sh` — spot launch with reclaim sync + S3 daemon
- `scripts/v1b_bootstrap.py` + `v1b_validate.py` — verifier recalibration pipeline

---

## What did NOT work (honest record)

### Round 1 generate step failed
After SFT completed, `round_runner.generate_patches()` tried to launch vLLM with the Gen1 LoRA adapter. Failed at vLLM startup: GPU memory race (prior SFT allocations not fully released; vLLM wanted 92% of free VRAM).

Even had the memory race been fixed, the generate step calls `openhands.core.main` with a CLI signature I wrote speculatively. OpenHands is not installed in the venv. The real OpenHands eval harness is at `evaluation/swe_bench/` in their repo with its own orchestration — a separate integration project.

### Downstream never ran
- No Gen1 patches generated on round_1_control or drift_audit_300
- No Docker gold eval
- No verifier recalibration for Round 1
- No drift trajectory point
- No plateau check, no Phase 2 gate check

### So: is the pipeline validated?
**Half of it.** SFT pipeline: validated end-to-end on real 30B-A3B MoE + real Nebius data + real LoRA config. Generate + eval pipeline: code written but never exercised.

The honest product statement: this recipe produces a trained Gen1 adapter that SHOULD improve over the base model, but we did not measure by how much. The $355 spent got us:
- A working SFT recipe on 30B-A3B
- 22 documented failure modes
- 13 backlog items
- 4 product docs that capture the recipe
- An adapter sitting on S3 waiting for someone to measure it

That's not nothing, but it's also not the "Round 1 gold pass rate > Gen0 baseline by X pp" number we wanted.

---

## What to do next (separate engineering cycle, not this session)

### Critical path to completing Round 1
1. **Install OpenHands v0.54 on a p4de instance**. Budget 2-4 hours of engineering time for integration issues (Docker-in-Docker for OpenHands runtime + vLLM LoRA serving with qwen3_moe).
2. **Fix the vLLM memory race** — the patch in this session added a 30s sleep + empty_cache, but better architecture is a subprocess fork boundary between SFT and generate.
3. **Run Round 1 generate** on the Gen1 adapter already on S3 — no need to redo SFT.
4. **Run Docker gold eval via SWE-rebench-V2's `eval.py`** on m7i.
5. **Get the first real number**: Gen1's gold pass rate on drift_audit_300.

Estimated additional spend: $50-150 depending on integration smoothness.

### Before re-launching
- **Local smoke-test the generate pipeline on a 1B model first**. FM-5.3: debugging on $13.59/hr compute is expensive. Get the full SFT → generate → eval loop working on TinyLlama locally, THEN pay for 30B scale.
- **Pick a specific OpenHands eval integration approach**: either upstream's `evaluation/swe_bench/run_infer.py` (well-trodden) or write a minimal batch runner ourselves (more control, more work). Document the choice.

### Open questions waiting for Round 1 numbers
1. Does Qwen3-Coder-30B-A3B + Nebius gold-filtered SFT beat the base model by ≥3pp?
2. Does Gen1's verifier-gold agreement on drift_audit stay ≥ 0.85?
3. How does the `v009_fail ∧ rf_pass` override signal behave on Qwen3-Coder outputs vs Claude×OpenCode?

None of these can be answered without completing the generate step above.

---

## Cost accounting

| Component | Spent |
|---|---|
| p4de.24xlarge spot (26 hr × $13.59) | $355 |
| m7i.4xlarge start/run time (~3 hr × $0.77) | $2 |
| Bedrock Haiku (pre-loop micros + V1b_bootstrap) | ~$2 |
| S3 storage + transfer | ~$1 |
| **Total** | **~$360** |

| Category | Breakdown |
|---|---|
| Productive compute (Round 1 SFT) | ~4 hr × $13.59 = $54 |
| Data prep + orchestration scaffolding | ~3 hr × $13.59 = $41 |
| Debug / idle (the expensive bit) | ~19 hr × $13.59 = $258 |
| m7i + misc | ~$7 |

**Debug/idle dominates.** 72% of spend went to keeping p4de alive while I fought through 22 failure modes remotely. The failure-modes catalog captures that learning for free re-use; if a second team runs this recipe with the catalog + local smoke-tested generate, their debug/idle should drop by 80%+.

---

## Decisions made this session (for the record)

1. **Product-first framing over research-novelty**: ship runbook + failure catalog + backlog, not a drift-trajectory paper.
2. **Base model**: Qwen3-Coder-30B-A3B-Instruct (not Qwen3.5-27B VLM). Resolved FM-3.3, FM-6.1.
3. **Harness**: OpenHands v0.54 throughout. Train distribution = inference distribution.
4. **Scope**: 2 rounds minimum-viable. Arms B/D/E deferred to backlog with explicit triggers.
5. **Eval benchmark**: SWE-rebench v1 (from `nebius/SWE-rebench`). Images at `docker.io/swerebenchv2/*`. Eval via SWE-rebench-V2's `scripts/eval.py`. Resolved FM-6.3.
6. **Instance-level splits, not trajectory-level**: 6,306 unique instances partitioned into rolling-rounds with `drift_audit_300` write-locked across all generations. Avoids instance-id leakage.
7. **Spot reclaim handling**: authoritative backup to S3 same-region (us-west-2 bucket), best-effort secondary to laptop. Resolved FM-3.7, updated memory file.
8. **Terminated before completing generate/eval**: honest acknowledgment that generate-pipeline integration was underscoped. Better product value to ship docs + adapter than burn more $ on remote debugging.

---

## Artifacts index

### Specs
- [spec: self-coding-agent-loop](../../specs/self-coding-agent-loop.md) — main spec, updated with IN_PROGRESS preamble reflecting product-first reality
- [spec: pre-loop-micros](../../specs/pre-loop-micros.md) — the 4 micros that ran before the main launch

### Operational docs (this directory)
- `runbook-continuous-improvement.md`
- `failure-modes.md`
- `cost-calculator.md`
- `experiment-backlog.md`
- `spec-explainer.html` (interactive visual)
- `SESSION_SUMMARY.md` (this file)

### Data
- `data/splits/*.jsonl` (13 split files, instance-level disjoint)
- `data/splits/splits_manifest.json` (SHA-pinned, invariants documented)
- `data/nebius/trajectories.parquet` (2.1GB, local + S3 mirror)

### S3 persistent artifacts
- `s3://agent-aiops-artifacts/self-coding-agent-loop/gen0/` — original Gen0 (Qwen3.5-27B VLM, rejected)
- `s3://agent-aiops-artifacts/self-coding-agent-loop/runs/round_1/adapter/` — **Gen1 LoRA** (Qwen3-Coder-30B-A3B-Instruct, the new baseline to measure)
- `s3://agent-aiops-artifacts/self-coding-agent-loop/runs/round_1/round_runner.log` — full SFT training log
- `s3://agent-aiops-artifacts/self-coding-agent-loop/data/splits/` — all split files
- `s3://agent-aiops-artifacts/self-coding-agent-loop/data/nebius/trajectories.parquet`
- `s3://agent-aiops-artifacts/self-coding-agent-loop/repo-snapshot.tar.gz` (frozen at 2026-05-11 10:05 UTC)

### Scripts
- `scripts/*` (13 Python + Bash files, all syntax-validated)

---

## Next session prerequisites

Before re-launching to complete Round 1 generate+eval:
- [ ] Install OpenHands v0.54 locally; smoke-test batch-eval loop on TinyLlama-1.1B
- [ ] Verify OpenHands `evaluation/swe_bench/run_infer.py` works with a vLLM-served LoRA adapter
- [ ] Pick eval output path that `docker_gold_eval.py` can parse (currently writes SWE-bench-format; `eval.py` reads slightly different schema — reconcile)
- [ ] Pre-download the 600 SWE-rebench v1 task records to m7i (cacheable)
- [ ] Restart swebench-eval m7i (stopped, not terminated)
- [ ] Consider whether to resume this experiment at all, or pivot to the "runbook dry-run on a 2nd (model, harness) combo" backlog item first — that might be higher-leverage than completing this one
