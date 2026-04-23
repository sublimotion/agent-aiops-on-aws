# Autoresearch Spec: CoderForge Eval

## Status: DRAFT

## Overview

Evaluate CoderForge-Preview-trained models against our existing agent harness matrix, then test whether CoderForge's open dataset can produce a competitive Qwen3.5-32B fine-tune. CoderForge-Preview (Together AI) is the largest open test-verified coding agent dataset — 155K verified trajectories across 1,655 repos, trained with Qwen3-Coder-480B as teacher. Their 32B model (Qwen3-32B base) achieves 59.4% on SWE-bench Verified, beating SERA-32B (49.5%) by 10pp.

**Critical finding**: Together AI released only the dataset (`togethercomputer/CoderForge-Preview`), NOT the trained 32B model weights. Community fine-tunes exist only at 4B (jackyk02) and 8B (LAION) scale. This experiment has two paths: (1) evaluate what exists, (2) train on the dataset ourselves.

**Core hypothesis**: CoderForge's scale advantage (155K trajectories, 6x SERA's 25K) transfers to our eval setup, and training on the CoderForge dataset with a Qwen3.5 base produces a model that exceeds both SERA-32B and the original CoderForge-32B results — at a fraction of the cost of frontier API models.

**Builds on**: Agent Harness (Phase 1+2), Agent Swarm (multi-model matrix), Verifier Reward (v009 verifier).

## Cost-Performance Motivation

The compelling case for this experiment is not matching frontier models on absolute performance, but reaching competitive quality at radically lower inference cost. SWE-bench Verified scores in context:

```
79.2%  ████████████████████████████████████████  Claude 4.5 Opus (frontier)     ~$3.00/issue
75.6%  ██████████████████████████████████████    Claude Opus 4.6                ~$2.50/issue
74.8%  █████████████████████████████████████     Claude 4.5 Sonnet              ~$0.50/issue
69.6%  ████████████████████████████████████      Qwen3-Coder-480B + OpenHands   ~$0.10/issue (TP8)
60.4%  ████████████████████████████████          Qwen3-Coder-30B + EntroPO      ~$0.02/issue
59.4%  ████████████████████████████████          CoderForge-32B (reference)     ~$0.01/issue (TP2)
49.5%  ██████████████████████████                SERA-32B                       ~$0.01/issue (TP1)
```

**Cost model** (self-hosted on g7e, amortized over 1,000 issues):

| Model | TP | GPU-hours/issue | g7e $/hr | $/issue | SWE-bench % | $/resolved issue |
|-------|----|----|------|---------|------------|-----------------|
| **CoderForge-Qwen3.5-122B-A10B MoE (target)** | 4 | 0.017 | $8.40 | **$0.005** | ~60-65% | **$0.008** |
| CoderForge-Qwen3.5-32B dense (fallback) | 2 | 0.033 | $8.40 | $0.014 | ~60% | $0.023 |
| Qwen3-Coder-480B | 8 | 0.033 | $33.60* | $0.056 | ~55% (bash) | $0.10 |
| Claude 4.5 Sonnet (API) | — | — | — | $0.50 | ~75% | $0.67 |
| Claude Opus 4.6 (API) | — | — | — | $2.50 | ~76% | $3.29 |

\* Would require p5e or similar for TP8; g7e has 4 GPUs.

The MoE model is **84x cheaper per resolved issue** than Claude Sonnet — the most compelling cost-performance point in the open-source landscape.

**The key ratio**: Claude Sonnet resolves ~75% of issues at $0.67/resolved. If CoderForge-Qwen3.5-32B resolves ~60% at $0.023/resolved, that's **29x cheaper per resolved issue** — with only a 15pp quality gap. For bulk triage, CI/CD integration, or high-volume workloads where cost dominates, this is the compelling tradeoff.

**Break-even analysis**: At what volume does self-hosted training pay off?

| Item | Cost |
|------|------|
| Training (B200 spot us-west-2b, 3-4 days for MoE) | ~$2,200-3,500 |
| g7e inference (MoE, amortized) | $0.005/issue |
| Claude Sonnet API | $0.50/issue |

Training pays for itself after **~4,450-7,070 issues** ($2,200-3,500 ÷ ($0.50 - $0.005)). For a team running agents on hundreds of issues/month, ROI is < 3 months.

**What stacking our tools adds** (projected, MoE model):

| Configuration | Est. SWE-bench % | $/issue | $/resolved |
|---------------|-------------------|---------|------------|
| CoderForge-Qwen3.5-122B-A10B pass@1 | ~60-65% | $0.005 | $0.008 |
| + v009 best-of-4 selection | ~65% | $0.020 + $0.12 verifier | $0.22 |
| + ThunderAgent pass@8 | ~68% | $0.040 | $0.059 |
| + v009 on pass@8 candidates | ~70% | $0.040 + $0.24 verifier | $0.40 |

At ~70% with full stack, the gap to Sonnet (75%) narrows to 5pp while remaining **1.7x cheaper per resolved issue** — and with no API dependency, rate limits, or data leaving the network.

**Why MoE over dense**: The 122B-A10B MoE has only 10B active params per token (vs 32B dense), giving ~3x faster inference at TP4 on g7e. This directly translates to more attempts per dollar — making pass@8 with ThunderAgent practical at $0.04/issue instead of $0.11/issue. The MoE also has higher model capacity (122B total weights) to absorb the 155K CoderForge trajectories, potentially yielding better quality despite fewer active params per token. We already validated this architecture in our harness: Qwen3.5-122B-A10B FP8 achieved 86% fix rate on our 50-issue subset.

## Research Questions

1. **Does CoderForge's 6x data scale advantage hold on full SWE-bench Verified?** SERA-32B: 49.5%. Does a CoderForge-trained model on Qwen3.5 base exceed this?

2. **Does CoderForge + Qwen3.5 base exceed Qwen3 base?** CoderForge was trained on Qwen3-32B. Qwen3.5 has hybrid attention and better tool calling. Does the base upgrade matter when the SFT data is this strong?

3. **How does our v009 verifier perform on CoderForge patches?** CoderForge uses OpenHands (str_replace_editor + bash) — larger diffs than SERA, closer to OpenCode style. Predict: better verifier transfer than SERA (T8 showed harness determines transfer).

4. **Is CoderForge + verifier best-of-N viable?** pass@16 = 78.6% (their number). With our v009 verifier as selector, can we close the gap between pass@1 (59.4%) and pass@16 without running 16 attempts?

5. **What is the cost-adjusted Pareto frontier?** At each price point ($0.01, $0.05, $0.10, $0.50/issue), what is the best achievable pass rate with our stack?

## Phases

### Phase 0: Quick Signal — LAION Qwen3-8B (Day 1)

**Goal**: Get a directional signal using LAION's existing community fine-tune before investing in a full 32B training run.

**Model**: `laion/coderforge-31600__Qwen3-8B` (205 downloads, trained on 31.6K CoderForge trajectories, Qwen3-8B base)

**Why this works as a signal**: SERA-8B achieves 31.7% on SWE-bench Verified. If CoderForge-8B exceeds this on our 50-issue subset, the data quality/scale advantage is real and worth investing in a 32B run. If it's worse, the dataset may not be as clean as claimed, or the Qwen Coder chat template may cause issues with our harnesses.

**Steps**:
1. Download `laion/coderforge-31600__Qwen3-8B` to `/mnt/nvme/models/coderforge-8b/`
2. Serve via vLLM TP1 on g7e (8B fits in ~5 GB FP8, ~10 GB BF16)
3. Configure: Qwen Coder chat template (XML-formatted tool calling)
4. Run through SERA + OpenCode harnesses on 50-issue subset
5. Compare to SERA-8B (31.7% Verified) and base Qwen3-8B

**vLLM flags**:
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /mnt/nvme/models/coderforge-8b/ \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --tool-call-parser qwen3_xml \
    --port 8000
```

**Time**: ~2 hours (50 issues × 2 harnesses)
**Exit criteria**: Fix rate and pass rate compared to SERA-8B baseline. Go/no-go for Phase 1.

### Phase 1: Train CoderForge on Qwen3.5 (Days 2-6)

**Goal**: Train a Qwen3.5 model on the CoderForge dataset. Primary target is the 122B-A10B MoE (faster inference, better cost-performance); 32B dense is the fallback.

**Prerequisites**: Phase 0 shows positive signal (CoderForge-8B >= SERA-8B).

**Why MoE first**: The 122B-A10B MoE gives ~3x faster inference than 32B dense (10B active params vs 32B), making pass@8 with ThunderAgent practical at $0.005/issue. The MoE also has higher total capacity (122B weights) to absorb 155K trajectories. We already validated this architecture: base Qwen3.5-122B-A10B achieved 86% fix rate on our 50-issue subset.

**Training targets** (priority order):

| Priority | Base Model | Total / Active | TP (inference) | Training Memory | Why |
|----------|-----------|----------------|----------------|-----------------|-----|
| **P0** | Qwen3.5-122B-A10B | 122B / 10B MoE | TP4 on g7e | ~1.22TB (tight on B200) | Best cost-performance: $0.005/issue, 3x faster |
| **P1 (fallback)** | Qwen3.5-32B | 32B / 32B dense | TP2 on g7e | ~400GB (comfortable on B200) | Proven architecture, directly comparable to CoderForge reference |

**Training setup (P0 — MoE)**:

| Parameter | Together AI (reference) | Our run (MoE) |
|-----------|------------------------|---------------|
| Base model | Qwen3-32B (dense) | **Qwen3.5-122B-A10B** (MoE upgrade) |
| Dataset | CoderForge-Preview (155K) | Same (untruncated) |
| Hardware | 64 H100s (8 nodes) | p6-b200.48xlarge capacity block (8x B200, 1.46TB HBM3e) |
| Framework | FSDP2 + Ulysses | FSDP2 + Ulysses + MoE-aware sharding |
| Context | 128K | **128K** (match reference — no truncation) |
| Epochs | Not specified | 1-3 (start with 1) |
| LR | Not specified | 1e-5 (SERA's config) |

**MoE training memory analysis** (B200, 8 GPUs, ZeRO-3):

| Component | Total | Per GPU |
|-----------|-------|---------|
| Model weights (BF16) | 244 GB | 30 GB |
| Optimizer states (AdamW) | 976 GB | 122 GB |
| Subtotal (weights + opt) | 1.22 TB | 152 GB |
| Available for activations | 240 GB | **31 GB/GPU** |
| Activation memory (128K, 10B active) | | ~20-25 GB* |

\* MoE activations scale with active params (10B), not total params (122B). With gradient checkpointing, 31 GB/GPU should be sufficient. If tight, reduce to 64K context for first epoch as a sanity check, then attempt 128K.

**Fallback to 32B dense**: If MoE training fails (expert collapse, memory overflow, or router instability), fall back to Qwen3.5-32B dense which fits comfortably on B200 (~400GB total, ~50GB/GPU).

**Training data preparation**:
1. Download `togethercomputer/CoderForge-Preview` (use `trajectories-tokenized_qwencoder` subset if compatible with Qwen3.5 tokenizer, else raw trajectories)
2. Filter to successful trajectories only (155K of 258K)
3. Format for SFT: mask instructions, compute loss only on assistant responses
4. Full 128K context — no truncation. CoderForge's median trajectory is 38K tokens and the long-context trajectories (complex multi-step agent workflows) are the most valuable training signal. Truncating defeats the purpose.

**MoE-specific training considerations**:
- **Expert load balancing**: Monitor per-expert utilization. If experts collapse (< 3 of 128 experts active), add auxiliary load-balancing loss.
- **Router stability**: MoE routers can destabilize during SFT if the data distribution differs significantly from pretraining. Monitor router entropy; if it drops below 0.5, reduce LR.
- **All experts trained**: Unlike LoRA-on-MoE approaches, we do full fine-tune — all expert weights updated. This is expensive but maximizes capacity utilization.

**Compute options**:

| Option | Model | GPUs | Memory | Time (est.) | Cost |
|--------|-------|------|--------|-------------|------|
| **B200 spot us-west-2** (P0 MoE) | 122B-A10B | 8x B200 | 1.46TB HBM3e | ~3-4 days | **~$2,200-3,500** ($25-36/hr spot) |
| **B200 spot us-west-2** (P1 dense) | 32B | 8x B200 | 1.46TB HBM3e | ~1-2 days | **~$600-1,700** ($25-36/hr spot) |
| **B200 capacity block** (either) | Either | 8x B200 | 1.46TB HBM3e | As above | ~$5,400-14,400 ($75/hr reserved) |
| **Together API (either)** | Either | Managed | N/A | ~1-2 days | ~$2K-5K (est.) |

**B200 spot pricing** (us-west-2, as of 2026-03-28):
- us-west-2b: **$25.60/hr** (cheapest, stable $25-27 over 24h)
- us-west-2d: $29.83/hr
- us-west-2a: $35.56/hr

**Spot resilience**: S3 checkpointing every 500 steps to `s3://agent-aiops-checkpoints/coderforge-eval/`. On spot interruption (2-min warning), save current step and resume from latest checkpoint. FSDP2 `StateDictType.FULL_STATE_DICT` → sharded S3 upload. Estimated overhead: <5% wall time for checkpoint saves.

**Why us-west-2**: Existing experiment data (models, datasets, harness scripts) already staged on g7e in us-west-2. Same-region S3 access avoids cross-region transfer costs.

**Why B200 and not g7e**: Even the 32B dense model at 128K context requires ~400GB+ for optimizer states. g7e has 4x 96GB GDDR7 = 384GB total — insufficient. The 122B MoE needs ~1.22TB. B200 (1.46TB HBM3e) is the only self-hosted option. Together's API is the fallback.

**Why not truncate**: CoderForge's advantage over SERA is scale AND context (155K trajectories at 128K vs 25K at 32K). Truncating would eliminate the context advantage entirely, reducing the experiment to "SERA with more data" — a different hypothesis. Experiment integrity requires matching the reference training spec.

**Exit criteria**: Training converges (loss decreasing). Model generates coherent tool-calling responses. Expert utilization is balanced (no collapse).

### Phase 2: Evaluate CoderForge-Qwen3.5 (Days 5-9)

**Goal**: Run the trained model (MoE or dense fallback) on full SWE-bench Verified with Docker evaluation. Our T10 results proved that 50-issue Lite is not representative — precision=1.00 on Lite vs 0.78 on Verified, base rate 12% vs 51%. CoderForge reports 59.4% on full Verified; we must evaluate on the same benchmark to make a valid comparison.

**Evaluation tiers**:

| Tier | Benchmark | Issues | Docker | Purpose |
|------|-----------|--------|--------|---------|
| **Quick check** | SWE-bench Lite 50-issue subset | 50 | No | Sanity check before committing to full eval (~2 hours) |
| **Primary eval** | SWE-bench Verified (full) | 500 | **Yes** | Apples-to-apples with CoderForge reference (59.4%) and SERA (49.5%) |

The quick check on 50 issues catches catastrophic failures (broken tool calling, chat template issues). The primary eval on full Verified with Docker is the definitive result.

**Docker evaluation setup**:
- SWE-bench provides per-instance Docker images with pinned dependencies
- `swebench.harness.run_evaluation` handles: checkout base commit → apply patch → run test suite → report pass/fail
- Requires Docker on g7e (install via `sudo amazon-linux-extras install docker` or use existing `nerdctl`)
- Disk: ~50-100 GB for Docker images (NVMe has capacity)

**Eval matrix**:

| Priority | Model | Harness | Issues | Rationale |
|----------|-------|---------|--------|-----------|
| P0 | CoderForge-Qwen3.5-122B-A10B | OpenCode | 500 (Verified) | Best harness + fastest model |
| P1 | CoderForge-Qwen3.5-122B-A10B | SERA | 500 (Verified) | Continuity with SERA-32B data |
| P2 | CoderForge-Qwen3.5-32B (if trained) | OpenCode | 500 (Verified) | MoE vs dense comparison |

**vLLM serving (MoE)**:
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /mnt/nvme/models/coderforge-qwen35-122b-a10b/ \
    --tensor-parallel-size 4 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.95 \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --port 8000
```

Qwen3.5-122B-A10B FP8 fits TP4 on g7e (~29 GB/GPU, validated in our swarm experiments). ~3x faster inference than 32B dense due to 10B active params.

**vLLM serving (dense fallback)**:
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /mnt/nvme/models/coderforge-qwen35-32b/ \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.95 \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --port 8000
```

**Comparison baselines** (same benchmark, same eval method):

| Model | SWE-bench Verified | Eval Method | Source |
|-------|-------------------|-------------|--------|
| CoderForge-32B (Qwen3 base) | 59.4% | Docker, OpenHands | Together AI |
| SERA-32B (Qwen3 base) | 49.5% ± 1.9% | Docker, SERA scaffold | Allen AI |
| Devstral Small 2 | 46.8-50% | Docker, various | Mistral |
| SERA-8B | 31.7% ± 0.9% | Docker, SERA scaffold | Allen AI |

**Our prior 50-issue Lite results** (for quick-check comparison only, NOT definitive):

| Model | Harness | Fix Rate | Pass Rate |
|-------|---------|----------|-----------|
| Devstral 24B | OpenCode | 88% | 22% |
| SERA-32B | SERA | 64% | — |
| Qwen3.5-122B-A10B FP8 | OpenCode | 86% | 9% |

**Target**: Pass@1 > 50% on SWE-bench Verified (exceeding SERA-32B's 49.5%). Stretch: > 59.4% (exceeding CoderForge-32B reference, proving Qwen3.5 base upgrade adds value).

### Acceleration: ThunderAgent Scheduling (Phases 0, 2, 3)

All eval phases benefit from ThunderAgent (`thunder_proxy.py` from agent-swarm Phase 2b). During agent tool execution (40-60% of wall time), GPUs sit idle holding KV cache. ThunderAgent backfills those slots with other agents' inference requests.

**Measured performance** (from agent-swarm Phase 2b): 63% throughput improvement at N=8 agents, tc=2.0.

The MoE model amplifies ThunderAgent's benefit: faster inference = shorter GPU occupancy per request = more backfill opportunities = higher effective concurrency.

| Phase | Model | Sequential | With ThunderAgent |
|-------|-------|-----------|-------------------|
| Phase 0 (100 runs) | 8B | ~2 hours | ~1.2 hours |
| Phase 2 quick check (50 Lite) | MoE 122B | ~40 min | ~25 min |
| Phase 2 primary (500 Verified × 2 harnesses) | MoE 122B | ~24 hours | ~14 hours |
| Phase 3 (BoN ×4, 500 Verified) | MoE 122B | ~48 hours | ~28 hours |
| **Total eval** | | **~75 hours** | **~44 hours** |

MoE's ~3x inference speedup over dense reduces per-issue time from ~2 min to ~40 sec, cutting Phase 2+3 wall time substantially. ThunderAgent stacks on top.

**Configuration**:
- Phase 2 (quality-sensitive): `tc=1.5` — conservative, avoids quality degradation from preemption. Our Phase 2b data showed fix rate drops at tc=2.0 (98%→80%).
- Phase 3 (throughput-sensitive): `tc=2.0` — aggressive, acceptable since we're generating multiple candidates for best-of-N selection anyway.
- Proxy: `thunder_proxy.py` on port 9000, forwarding to vLLM on port 8000.

### Phase 3: Verifier Integration (Day 5-6)

**Goal**: Apply v009 verifier (4/4 unanimous) to CoderForge patches and measure best-of-N selection.

**Steps**:
1. Generate N=4 patches per issue on full SWE-bench Verified (4 runs of best harness from Phase 2) — use ThunderAgent at tc=2.0 for throughput
2. Run v009 verifier on all candidates
3. Select top-1 by verifier score
4. Docker eval on selected patches (same as Phase 2)

**Prediction** (based on T9/T10 findings): CoderForge + OpenCode should produce large diffs (>5K median) → verifier precision ~0.50-0.80. At ~50% base pass rate (similar to T10's 51%), v009 4/4 should achieve precision ~0.92 and recall ~0.14 (matching T10 results). The real test is whether best-of-4 selection closes the gap toward pass@16 (78.6%).

**Exit criteria**: Best-of-4 pass rate vs pass@1 on full SWE-bench Verified. Verifier precision/recall on CoderForge patches established.

## Components

### 1. Compute

- **Phase 0+2+3 (inference)**: g7e.24xlarge (4x RTX PRO 6000 Blackwell, 96GB GDDR7 each). Already provisioned.
- **Phase 1 (training)**: p6-b200.48xlarge capacity block (8x B200, 183GB HBM3e each) or Together API fine-tuning. g7e is NOT suitable for 128K context training.

| Phase | Infrastructure | Sequential | With ThunderAgent |
|-------|---------------|-----------|-------------------|
| Phase 0 (8B, 50 Lite) | g7e, 1 GPU | ~2 hours | ~1.2 hours |
| Phase 1 (MoE training) | B200 capacity block (8 GPUs) | 3-4 days | N/A (training) |
| Phase 1 (dense fallback) | B200 capacity block (8 GPUs) | 1-2 days | N/A (training) |
| Phase 2 quick check (MoE, 50 Lite) | g7e, TP4 | ~40 min | ~25 min |
| Phase 2 primary (MoE, 500 Verified) | g7e, TP4 + ThunderAgent (tc=1.5) | ~24 hours | ~14 hours |
| Phase 3 (BoN ×4, 500 Verified) | g7e, TP4 + ThunderAgent (tc=2.0) + Haiku API | ~48 hours | ~28 hours |

**MoE inference advantage**: 500 issues at ~40 sec/issue (MoE) vs ~2 min/issue (dense) = ~6 hours vs ~17 hours per harness run. ThunderAgent stacks on top for ~60% of that.

### 2. Codebase

- **Source**: Existing harness infrastructure from agent-harness and agent-swarm blueprints
- **Dataset**: `togethercomputer/CoderForge-Preview` on HuggingFace (258K trajectories, ~6.7B tokens)
- **Community model**: `laion/coderforge-31600__Qwen3-8B` (Phase 0)
- **Fixed files**:
  - SWE-bench Verified full dataset (500 issues) — primary eval benchmark
  - SWE-bench Lite 50-issue subset (seed 42) — quick checks only
  - Gold test patches + Docker evaluation images
  - `harness_eval.py`, `multi_harness_eval.py`
- **Scripts to create**:
  - `scripts/train_coderforge.py` — Training script with MoE-aware FSDP2 config for Qwen3.5-122B-A10B (+ dense fallback config)
  - `scripts/prepare_coderforge_data.py` — Dataset preparation (filter, format for Qwen3.5 tokenizer, full 128K context)
  - `scripts/monitor_experts.py` — MoE expert utilization monitoring (detect collapse during training)

### 3. Experiment Protocol

- **Primary metric**: Pass@1 on SWE-bench Verified (500 issues, Docker evaluation)
- **Secondary metrics**: Fix rate, precision, harness spread, turns used, tokens consumed
- **Primary eval**: Full SWE-bench Verified (500 issues) with Docker — matches CoderForge and SERA reference evaluations
- **Quick checks**: SWE-bench Lite 50-issue subset (seed 42) — for sanity checks and rapid iteration only, NOT for reporting results
- **Gold evaluation**: `swebench.harness.run_evaluation` — Docker container per instance, apply patch, run FAIL_TO_PASS + PASS_TO_PASS test suites
- **Logging**: JSONL per run in blueprint results directory
- **Why full Verified**: Our T10 experiment proved 50-issue Lite is not representative (precision 1.00→0.78, base rate 12%→51%). Investing $500-800 in training demands evaluation integrity.

### 4. Networking

- **Phase 0-2**: SSH to g7e, localhost vLLM serving
- **Phase 3**: Same + Anthropic API (Haiku) for v009 verifier calls
- **HuggingFace**: Dataset download (one-time, ~50-100 GB)

### 5. Storage

- **NVMe**: `/mnt/nvme/` — model weights, CoderForge dataset, training checkpoints
- **Results**: `domains/autoresearch/blueprints/coderforge-eval/results/`

## Success Criteria

### Phase 0: Quick Signal (50-issue Lite, directional only)
- CoderForge-8B fix rate >= SERA-8B (31.7% Verified equivalent) → proceed to Phase 1
- If CoderForge-8B < SERA-8B: investigate (chat template? data quality?) before proceeding

### Phase 1: Training Completes
- Training loss converges (MoE primary, dense fallback)
- Expert utilization balanced — no expert collapse (MoE-specific)
- Model generates coherent tool-calling responses in Qwen Coder format
- Checkpoint saved to NVMe

### Phase 2: Beats SERA-32B on Full SWE-bench Verified
- Pass@1 > 49.5% on SWE-bench Verified (500 issues, Docker) — exceeds SERA-32B
- Results directly comparable to published numbers (same benchmark, same eval method)
- Stretch: > 59.4% (exceeds CoderForge-32B reference, proving Qwen3.5 base upgrade matters)

### Phase 3: Verifier Adds Value on Full Verified
- Best-of-4 pass rate > pass@1 by >= 3pp on SWE-bench Verified
- Verifier precision > 0.50 on CoderForge patches (measured on full Verified with Docker gold labels)

## Non-Requirements

- Replicating Together AI's exact infrastructure (64 H100s) — B200 capacity block provides equivalent memory
- OpenHands scaffold (use our existing harnesses — but Docker eval matches their eval method)
- Context truncation — train at full 128K to match reference spec
- Full SWE-bench Lite (300 issues) — Lite is only for quick checks; Verified (500) is the primary benchmark

## Known Limitations

- **No official 32B weights**: Together AI didn't release them. Phase 1 training is required.
- **MoE training complexity**: Fine-tuning a 122B MoE is less proven than dense 32B SFT. Risks include expert collapse, router instability, and higher memory pressure. Dense 32B fallback mitigates this.
- **MoE memory is tight**: 122B weights + optimizer at 128K context uses ~1.22TB of B200's 1.46TB. Only ~31 GB/GPU left for activations. Gradient checkpointing is mandatory. If insufficient, first attempt at 64K context, then scale to 128K.
- **B200 capacity block availability**: Training requires high-memory GPUs. If capacity blocks are unavailable, Together API fine-tuning is the fallback.
- **Chat template compatibility**: CoderForge uses Qwen Coder's XML tool format. Our harnesses (OpenCode, SERA) use OpenAI-compatible tool calling. vLLM's `--tool-call-parser qwen3_xml` should bridge this, but needs validation.
- **LAION 8B fine-tune quality**: Community fine-tune, not official. Training recipe may differ from Together AI's. Phase 0 is directional, not definitive.
- **Docker disk usage**: SWE-bench Docker images may consume 50-100 GB. NVMe has capacity but needs monitoring.
- **Phase 2/3 wall time**: Full Verified (500 issues) is 10x the 50-issue subset. MoE's 3x inference speedup + ThunderAgent make this manageable (~14 hours for Phase 2, ~28 hours for Phase 3). Dense fallback would be ~3x slower.
- **CoderForge dataset is gated**: May require HuggingFace access request.
- **Qwen3.5 vs Qwen3 tokenizer**: CoderForge provides pre-tokenized trajectories for Qwen Coder. Need to verify Qwen3.5 tokenizer compatibility or re-tokenize from raw trajectories.

## Relationship to Other Specs

- **agent-harness**: Provides eval framework, 50-issue subset, harness adapters, comparison baselines
- **agent-swarm**: Provides multi-model comparison data (SERA-32B, Qwen3.5, Devstral)
- **verifier-reward**: Provides v009 verifier for Phase 3 best-of-N selection
- **training-recipes**: May share training infrastructure patterns if g7e training works
- **finetuning-recipes**: CoderForge training could become a recipe if successful

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
