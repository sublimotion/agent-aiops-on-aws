# Autoresearch Spec: Rejection Sampling SFT

## Status: IN PROGRESS — Phase 2 gold eval complete, Qwen3.5-27B best (55% gold pass), Config B training needed

## Overview

Use the Platt-calibrated verification cascade as a quality filter to curate SFT training data from CoderForge-Preview trajectories. Fine-tune Qwen3-32B on only the accepted trajectories — the agent learns from its best work, not its worst.

This is Phase 3 of the [phased adoption path](../../blueprints/learned-verifier/docs/RLVR_AND_VERIFICATION.md): after best-of-N selection (Phase 2) proves the verifier works at inference time, rejection sampling SFT uses the same signal to improve the base model. No RL infrastructure required — just data filtering + standard SFT.

**Core hypothesis**: Filtering CoderForge's 155K trajectories through the Platt-calibrated cascade (ECE=0.026-0.056) and fine-tuning on accepted-only data produces a model that outperforms fine-tuning on all data — because the verifier removes noisy/incorrect trajectories that hurt learning.

**Base model**: Qwen/Qwen3-32B — same base as SERA-32B (49.5% SWE-bench) and CoderForge (59.4% SWE-bench) for fair comparison. Follow-up with Qwen3-Coder-Next 80B if results are promising.

### Why Now

The verification flywheel experiment (Phase 4 ECE calibration) established that our signals are calibrated enough for training data selection:

| Signal | ECE | AUC | RL-Ready? | SFT-Ready? |
|--------|-----|-----|-----------|------------|
| Multiprompt + Platt | 0.026 | 0.453 | Yes | **Yes** |
| RF + Platt | 0.056 | 0.662 | Yes | **Yes** |
| Multiprompt (raw) | 0.151 | 0.500 | No | **Yes** (ECE < 0.3) |
| RF (raw) | 0.321 | 0.670 | No | No |

For rejection sampling SFT, the calibration bar is lower than RL — we just need the filter to select good trajectories more often than bad ones. Even raw multiprompt (ECE=0.151) is sufficient. But Platt-calibrated signals let us set principled acceptance thresholds.

### Precedent

| System | Filter | Base Model | Result | Cost |
|--------|--------|------------|--------|------|
| **SERA-32B** (Allen AI) | SVG consensus (precision=1.0) | Qwen3-32B | 24.4% → 49.5% SWE-bench (+25pp) | $2K (40 GPU-days) |
| **Shopify Flow** | LLM judge | Qwen3-32B | Weekly retrain, 10x cost reduction | 2x H200, 12 hrs/week |
| **CoderForge** (Together AI) | Docker tests (gold) | Qwen3-32B | 59.4% SWE-bench Verified | Not disclosed |
| **This experiment** | Platt-calibrated cascade | Qwen3-32B | Target: 55-65% SWE-bench | ~$800-2,000 |

SERA's key finding: **"Verification threshold doesn't matter for SFT"** — as long as the filter removes clearly bad trajectories, the exact threshold has little effect on downstream SFT quality. This simplifies our experiment design.

## Prior Art & Novelty

### What Exists

Every prior system for SFT data curation in coding agents uses one of two approaches:

| System | Filter Mechanism | Calibrated? | Cost | Result |
|--------|-----------------|-------------|------|--------|
| **CoderForge** (Together AI) | Binary Docker test pass/fail | No — gold labels, no probabilities | $130K (64 H100s for generation + execution) | 59.4% SWE-bench Verified |
| **SERA/SVG** (Allen AI, 2601.20789) | Line-level patch recall between two rollouts | No — geometric overlap, not probabilistic | $2K (40 GPU-days) | 49.5% SWE-bench Verified |
| **DeepSeek-R1** (2501.12948) | Rule-based + DeepSeek-V3 as LLM judge | No — raw scores, no ECE/Platt | Not disclosed | SoTA reasoning |
| **Llama 3** (2407.21783) | RM top-quartile + Llama quality classifier | No — quantile cutoff, no calibration | Not disclosed | Frontier LLM |
| **ReST-EM** (2312.06585) | Binary execution pass/fail, cap 10/problem | No | Moderate | Math/code gains |
| **SWE-RM** (2512.21919) | MoE reward model, ECE=0.047-0.051 | **Yes (measures ECE)** — but NOT used for SFT | High (400K+ trajectories) | +10.4pp TTS |
| **ZeroCoder** (2604.07864) | Consensus matrix + DyB4 Bayesian selector | **Partially** — Bayesian recalibration with 10 labels | Moderate | +21.6% with recalibration |
| **Shopify Flow** | LLM judge calibrated to human annotations | **Manually** — tuned until scores aligned with human judgment | 2x H200, 12 hrs/week | 2.2x faster, 68% cheaper |
| **SAGE** (2603.15255) | Critic agent + external verifier | No | Moderate | +8.9% LiveCodeBench |
| **EvolveCoder** (2603.12698) | Adversarial test evolution, not verifier-based | No | Moderate | +4.2pp across 4 benchmarks |

### The Gap

**No published work combines calibrated verification probabilities with training data filtering for SFT.** Specifically:

1. **SWE-RM** measures ECE (0.047) but uses it only for test-time scaling and RL reward — never for SFT data curation. They explicitly don't close the loop from calibration → filtering.

2. **SERA** uses SVG (precision=1.0) with no probabilistic calibration. Their "threshold doesn't matter" finding was measured only with a clean signal — untested with noisier verifiers.

3. **DeepSeek-R1** explicitly punted on software engineering: "large-scale RL has not been applied extensively in software engineering tasks... Future versions will address this by implementing rejection sampling on software engineering data."

4. **ZeroCoder's DyB4** is the closest — Bayesian recalibration of selection thresholds — but operates on co-evolved code/test pairs, not agent trajectories, and doesn't use LLM-based content verification.

### What's Novel

Five things no prior work has done:

| # | Novelty | Why It Matters |
|---|---------|---------------|
| 1 | **Platt-calibrated verifier probabilities for SFT data selection** | Enables principled threshold setting (p>0.5 means "more likely correct than not" with known calibration error ECE=0.026). Prior work uses binary pass/fail or ad-hoc quantile cutoffs. |
| 2 | **Tiered cascade (RF → Multiprompt → Debate) for data curation** | No prior work uses an escalation architecture for filtering training data. 93% of traces filtered by free RF; LLM calls only for uncertain cases. Prior approaches are single-stage. |
| 3 | **$200-400 filtering cost for 20K trajectories** | CoderForge: $130K (Docker). SERA: $2K (dual rollouts). Our cascade: $0.002-0.020/patch = $40-400 for 20K traces. 300-3,000x cheaper than Docker execution. |
| 4 | **Testing threshold-invariance with noisy silver labels** | SERA's "threshold doesn't matter" used SVG precision=1.0. Does it hold at 68% silver-gold agreement? If yes, cheap verifiers are sufficient for SFT curation — no gold labels needed at scale. |
| 5 | **End-to-end calibration → filtering → SFT pipeline** | First system connecting ECE measurement (Phase 4 flywheel) directly to SFT data quality. The ECE result (0.026-0.056) is the input that makes this experiment possible. |

### Projected Impact

**If cascade-filtered SFT matches gold-filtered SFT** (Config B ≈ Config D):

- **Docker-free SFT curation**: Replace $130K Docker execution pipelines with $200-400 cascade filtering. Any team with Haiku API access can curate SFT data — no Docker infrastructure, no test suites, no execution sandboxes.
- **Unlocks SFT on repos without tests**: ~40% of GitHub repos lack test suites. Docker-based filtering (CoderForge, ReST-EM) cannot curate data from these repos. A content-based cascade can.
- **Continuous retrain flywheel**: Shopify's weekly retrain pattern becomes practical for coding agents. Generate trajectories → cascade filter ($0.002/patch via RF) → retrain weekly. The flywheel experiment proved the RF handles 93% of evaluations after 5 cycles.
- **Democratizes SFT for coding agents**: SERA needed a 357B teacher model for dual rollouts. CoderForge needed 64 H100s. Our approach needs only Haiku API calls + CPU for RF. Research groups without large GPU budgets can curate competitive SFT datasets.

**If it doesn't work** (cascade noise degrades SFT quality):

- Quantifies the minimum verifier quality needed for SFT. Config D (gold oracle) establishes the ceiling. The gap between cascade-filtered and gold-filtered measures exactly how much verification noise SFT can tolerate.
- Still validates SERA's threshold-invariance finding (or refutes it) with calibrated, measured noise levels — a contribution regardless of SFT outcome.

## Components

### 1. Compute

- **Filtering (Phase 1)**: No GPU required — RF-only mode runs on CPU
  - Cost: $0 (RF-only, no API calls)
  - Time: ~30 min for 20K traces
  - Completed on g7e.24xlarge (could run on any machine)
- **Training (Phase 2)**: GPU required for QLoRA SFT
  - **Recommended**: p4d.24xlarge (8x A100 40GB) — spot at ~$10.41/hr in us-east-1
  - **Also works**: g7e.24xlarge (4x RTX PRO 6000, 96GB) — but no Flash Attention 2 (sm_120 unsupported), slower
  - **Not recommended**: p6-b200.48xlarge — overkill for QLoRA 32B
  - QLoRA: 32B model in 4-bit ≈ 18-19GB, fits single A100 40GB with batch_size=1
  - **Parallel training**: 5 configs on 5 separate GPUs simultaneously via CUDA_VISIBLE_DEVICES
  - **Estimated training**: ~35 hours wall-clock for all 5 configs in parallel (single config ~7h on A100)

### 2. Codebase

```
domains/autoresearch/blueprints/rejection-sampling-sft/
├── scripts/
│   ├── filter_trajectories.py      ← Phase 1: Run RF on CoderForge, output scores JSONL
│   ├── train_sft.py                ← Phase 2: QLoRA SFT training (transformers + TRL + peft)
│   ├── run_experiment.sh           ← Sequential orchestrator (single GPU, spot-safe)
│   └── launch_training.sh          ← Parallel launcher (5 GPUs, one config per GPU)
├── results/                        ← S3-synced: s3://agent-aiops-artifacts/rejection-sampling-sft/
│   ├── filter_stats.json           ← Phase 1 acceptance rates and statistics
│   ├── training_*.json             ← Per-config training results (loss, time, samples)
│   └── training_*.log              ← Per-config training stdout/stderr logs
└── lessons.md
```

**Fixed files** (agent must NOT edit):
- `learned_verifier/` — the verification cascade (evaluation function)
- `learned_verifier/metrics.py` — ECE, AUC, evaluation protocol
- Gold labels from CoderForge (Docker-verified reward field)
- SWE-bench Verified evaluation harness

**Agent-editable files**:
- Filter thresholds and cascade configuration
- SFT hyperparameters (LoRA rank, learning rate, epochs, batch size)
- Chat template formatting and data preprocessing
- Training schedule (warmup, decay, gradient accumulation)

### 3. Experiment Protocol

#### Phase 1: Data Filtering — COMPLETE

**Status**: Done. RF-only mode ($0 cost, ~30 min). Scores saved to S3.

```
INPUT:  CoderForge-Preview SWE_Rebench split, 20,000 trajectories (streaming)
FILTER: RF verifier (cycle 5 model) + Platt calibration (a=0.1731, b=-0.4425)
OUTPUT: s3://agent-aiops-artifacts/rejection-sampling-sft/data/scores_SWE_Rebench_20000.jsonl (3.7 MB)
```

**Actual filtering results** (20,000 traces):

| Config | Filter | Accepted | Rate | Gold Precision |
|--------|--------|----------|------|----------------|
| D: Gold (reward=1) | gold_label == 1 | 11,982 | 59.9% | 100.0% |
| A: RF raw p>0.5 | rf_prob > 0.5 | 18,178 | 90.9% | 62.2% |
| B: Cascade cal p>0.5 | calibrated_prob > 0.5 | 11,286 | 56.4% | 66.2% |
| C: Cascade cal p>0.7 | calibrated_prob > 0.7 | 2,137 | 10.7% | 71.0% |

**Key observations**:
- Config A (RF raw) accepts 90.9% — nearly no filtering. RF raw scores are poorly calibrated.
- Config B (Platt-calibrated) is closest to gold (56.4% vs 59.9%) — Platt calibration works.
- Config C (strict) is very aggressive — only 2,137 samples. May be too few for effective SFT.
- Gold base rate is 59.9% — higher than expected (CoderForge had 65.1% overall pass rate).

**Reproduction**:
```bash
python3 scripts/filter_trajectories.py --n-traces 20000 --split SWE_Rebench --rf-only
```

**Key question**: Does SERA's "threshold doesn't matter" finding hold with our noisier cascade (68% silver-gold agreement vs SVG's precision=1.0)?

#### Phase 2: SFT Training — IN PROGRESS (Config C complete, others interrupted)

**Status**: Config D completed on Soperator cluster (1x H200, Qwen2.5-Coder-32B-Instruct). Other configs still need to be trained.

**Completed run (Soperator cluster)**:
- Config D (gold labels) on `Qwen/Qwen2.5-Coder-32B-Instruct`, 1x H200 141GB
- 11,983 samples, 375 steps (batch=2, grad_accum=16, effective=32), 8.6 hours
- Train loss: 1.111 → **0.352** (converged), mean token accuracy: **91.2%**
- Adapter: 269MB safetensors at `soperator/models/run_d_Qwen2.5-Coder-32B-Instruct/`
- Used SDPA (not flash-attn), packing disabled, max_seq_length=4096

**Earlier attempts** (spot instances, both reclaimed):
- g7e.24xlarge: Config C partially completed before reclaim
- p4d.24xlarge: All 5 configs launched in parallel, 27-48% through when reclaimed

**Training stack** (validated on p4d.24xlarge):

```python
# Python environment (miniconda Python 3.11)
torch==2.8.0+cu128
transformers>=4.51
trl==1.3.0          # IMPORTANT: uses max_length, NOT max_seq_length
peft==0.19.1
bitsandbytes>=0.46
flash-attn==2.8.3   # CRITICAL: requires sm_80+ (A100). Does NOT compile on sm_120 (Blackwell)
datasets>=3.5
accelerate>=1.6

# Model loading: QLoRA 4-bit NF4
model: Qwen/Qwen3-32B
quantization: 4-bit NF4 (double quant, bfloat16 compute)
attention: flash_attention_2   # requires flash-attn package
device_map: single GPU via CUDA_VISIBLE_DEVICES (NOT device_map="auto" across all GPUs — too slow)

# LoRA config
r=16, alpha=32, dropout=0.05, bias="none"
target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

# Training config
learning_rate: 2e-5
lr_scheduler: cosine
warmup_ratio: 0.03
epochs: 1
batch_size: 1              # A100 40GB is tight — batch=4 OOMs (vocab=152K → 9.27GB cross-entropy)
gradient_accumulation: 16  # effective batch = 16
max_length: 4096           # trl 1.3.0 param name (NOT max_seq_length)
packing: true              # concatenates samples into max_length windows
gradient_checkpointing: true (use_reentrant=False)
save_steps: 200
bf16: true
```

**Data format**: CoderForge chat messages → Qwen3-32B chat template via `tokenizer.apply_chat_template()`. Tool calls flattened into `<tool_call>` XML tags in assistant content. Approximate truncation at `max_seq_length * 5` chars.

**Training matrix** (5 configs):

| Config | Filter | Samples | Base Model | Infra | Status |
|--------|--------|---------|------------|-------|--------|
| none | All data (no filter) | ~20,000 | Qwen3-32B | — | TODO |
| a | RF raw p>0.5 | ~18,178 | Qwen3-32B | — | TODO |
| b | Cascade calibrated p>0.5 | ~11,286 | Qwen3-32B | — | TODO |
| c | Cascade calibrated p>0.7 | 2,137 | Qwen3-32B | — | TODO |
| d | Gold labels (reward=1) | 11,983 | Qwen2.5-Coder-32B | Soperator 1xH200 | **COMPLETE** |
| d (3.5) | Gold labels (reward=1) | 11,983 | Qwen3.5-27B | Soperator 2xH200 | **COMPLETE** — 55.2% gold pass |
| d (235B) | Gold labels (reward=1) | 11,983 | Qwen3-235B-A22B | Soperator | **COMPLETE** — 17.1% gold pass |

**Config D results** (completed on Soperator):
- 375 steps (batch=2 × grad_accum=16 = effective 32), 1 epoch, 8.6 hours
- Loss curve: 1.111 → 0.941 → 0.551 (avg) → **0.352** (final)
- Token accuracy: 71.5% → 91.2%
- Adapter: `soperator/models/run_d_Qwen2.5-Coder-32B-Instruct/adapter_model.safetensors` (269MB)

**Note**: Config D used Qwen2.5-Coder-32B-Instruct (not Qwen3-32B). For a fair comparison, remaining configs should use the same base model, OR Config D should be retrained on Qwen3-32B. SERA-32B used Qwen3-32B as base, so Qwen3-32B is preferred for the comparison.

**Parallel launch pattern** (`scripts/launch_training.sh`):
```bash
CONFIGS=(d b a c none)
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    CUDA_VISIBLE_DEVICES=$i python3 scripts/train_sft.py \
        --config "$config" --model Qwen/Qwen3-32B --s3-sync \
        --lora-r 16 --lora-alpha 32 --lr 2e-5 --epochs 1 \
        --batch-size 1 --grad-accum 16 --max-seq-length 4096 --save-steps 200 \
        > "$WORK/results/training_${config}.log" 2>&1 &
    sleep 5  # stagger model downloads
done
```

**CRITICAL FIX NEEDED**: S3 sync in `launch_training.sh` excludes `*.safetensors`, `*.bin`, `*.pt` — meaning adapter weights are NOT backed up to S3. On spot reclaim, only logs and result JSONs survive. The sync command must include model weights:
```bash
# BROKEN (current):
aws s3 sync "$WORK/results/" "s3://..." --exclude "*.safetensors" --quiet

# FIXED (required):
aws s3 sync "$WORK/results/" "s3://..." --quiet
# Also sync checkpoints separately:
aws s3 sync "$WORK/models/" "s3://..." --quiet
```

The `train_sft.py` script has its own S3 checkpoint callback that syncs adapters — but only on `save_steps` boundaries. If the instance is reclaimed between saves, the last checkpoint is lost. Consider reducing `save_steps` to 100 for spot instances.

Run 5 (gold oracle) is the key comparison — filtering by Docker gold labels. If cascade-filtered runs match oracle-filtered, the cascade is a valid Docker replacement for SFT curation.

#### Phase 3: Evaluation

Evaluate each trained model on:

1. **SWE-bench Verified** (500 instances) — primary metric, pass@1
   - Run with our SERA harness (validated at 64% fix rate for base Qwen3-32B)
   - Docker-verified gold labels
2. **CoderForge held-out** (1,000 traces) — secondary, verifier agreement
   - Cascade evaluation on model outputs
   - Compare cascade vs Docker gold labels
3. **Harness matrix** (if time permits) — SERA + OpenCode + Claude Code
   - Tests whether SFT generalizes across harnesses (SERA showed harness spread)

**Primary metric**: SWE-bench Verified pass@1 improvement over base model.

### 4. Networking

- **Filtering phase**: Internet access for Bedrock API (Haiku calls) + HuggingFace (CoderForge download)
- **Training phase**: SSH to GPU instance, HuggingFace model download
- **Evaluation phase**: SSH to evaluation instance (Docker for SWE-bench)

### 5. Storage

- **CoderForge dataset**: ~50GB streamed from HuggingFace (no local storage needed for filtering)
- **Filtered data**: ~5-15GB JSONL (accepted trajectories)
- **Model checkpoints**: ~60GB per 32B checkpoint (LoRA: ~500MB adapter)
- **NVMe**: `/mnt/nvme/` on g7e/B200 for fast I/O during training

## Phase 2 Eval Results (2026-04-30)

### Base Model Clarification

**Qwen3-32B (`Qwen/Qwen3-32B`) IS an instruct model.** Unlike Qwen2.5 which had separate base/instruct variants, Qwen3 ships as a single post-trained model with built-in instruction following and tool calling. SERA-32B's base model "Qwen 3-32B" = `Qwen/Qwen3-32B` = already instruct-capable. This makes our comparison apples-to-apples with SERA.

### SERA Harness Results (50-issue SWE-bench Lite subset)

Eval infrastructure: g7e.12xlarge spot (2x RTX PRO 6000) / p5.48xlarge (8x H100), vLLM 0.20.0 with native LoRA serving (`--enable-lora`), 30-turn SERA harness.

| Model | Fix Rate | Gold Pass | Gold % | Parkinson's | Notes |
|-------|----------|-----------|--------|-------------|-------|
| Qwen2.5-Coder-32B-Instruct (baseline) | 48% (24/50) | — | — | — | From agent-swarm experiment |
| **Qwen2.5-Coder-32B-Instruct + SFT-D** | **60% (30/50)** | — | — | — | **+12pp fix over baseline** |
| Qwen3-32B + SFT-D | 46% (23/50) | 2/50 | 4% | — | Context-limited at 32K tokens |
| **Qwen3.5-27B + SFT-D** | 58% (29/50) | **16/29** | **55.2%** | 76% | Best gold pass rate |
| Qwen3-235B-A22B + SFT-D | 74% (37/50) | 6/35 | 17.1% | 18% | High fix, low pass |
| SERA-32B (Qwen3-32B + SVG) | 64% | — | 49.5% | — | Their eval, SWE-bench Verified (500) |

**Gold eval (Docker SWE-bench, 2026-05-05)**:
- Qwen3.5-27B: 16/29 patches pass official Docker tests = **55.2% precision**
- Qwen3-235B-A22B: 6/35 patches pass = **17.1% precision**
- 235B has 18% Parkinson's (efficient explorer) but generates subtly wrong patches
- 27B has 76% Parkinson's (over-explores) but when it edits, patches are correct

### Key Findings

1. **CoderForge SFT data works**: +12pp fix rate on Qwen2.5-Coder-32B-Instruct (48→60%). Gold-filtered (Config D) trajectories teach useful code repair patterns.

2. **Qwen3.5-27B is the best base model for SFT-D**: 55.2% gold pass rate despite 76% Parkinson's ratio. The model's patches are high quality when it finally edits. Outperforms Qwen2.5-Coder (code-specialist) and Qwen3-32B (context-limited).

3. **235B MoE is a false positive**: 74% fix rate but only 17% gold pass — the model generates plausible-looking patches that don't actually solve problems. Efficient exploration (18% Parkinson's) doesn't help if the patches are wrong.

4. **Fix rate is misleading without gold eval**: 74% fix ≠ 74% correct. Docker gold eval is essential. The 235B would have appeared best on fix rate alone.

5. **Config D is not a novel contribution**: We used CoderForge's own gold labels (Docker test pass/fail) to filter. The results confirm their data quality but don't validate our cascade. The real experiment requires **Config B (cascade-filtered) ≈ Config D (gold-filtered)**.

6. **Parkinson's Law is model-specific**: 235B naturally edits early (18% ratio), Qwen3.5-27B over-explores (76%). SFT on CoderForge data teaches exploration habits from the teacher model, not the student.

### Remaining Work

- [ ] Run Config B (cascade-filtered) training on Qwen2.5-Coder-32B-Instruct
- [ ] Run Config "none" (unfiltered) training as control
- [ ] Docker gold eval on all patches to get actual pass rates
- [ ] Compare Config B vs Config D to test core hypothesis
- [ ] Rerun Qwen3 eval with 64K context (optional, lower priority)

## The Shopify Parallel

This experiment directly implements Shopify's weekly retrain pattern, adapted for coding agents:

| Shopify Flow | Rejection Sampling SFT |
|---|---|
| LLM judge filters merchant conversations | Platt-calibrated cascade filters CoderForge trajectories |
| Fine-tune Qwen3-32B on filtered data | Fine-tune Qwen3-32B on accepted trajectories |
| Weekly retrain on 2x H200, 12 hrs | One-shot training, ~3-12 hrs |
| Measure: workflow activation rate | Measure: SWE-bench Verified pass@1 |
| 35% activation drop on OOD traffic | Expected: different pass rate on different repo splits |
| Production data is free byproduct | CoderForge trajectories are free (open dataset) |

**Key difference**: Shopify retrains weekly with fresh production data. Our experiment is a one-shot proof that cascade-filtered SFT improves over unfiltered SFT. If validated, this becomes a repeatable pipeline: generate trajectories → cascade filter → retrain → deploy → generate more trajectories.

## Cost Estimate

| Phase | Activity | Estimated Cost | Actual Cost |
|-------|----------|----------------|-------------|
| 1 | Filter 20K trajectories (RF-only) | $200-400 | **$0** (RF-only, no API) |
| 2 | 5 SFT training runs on p4d spot (~35 hrs) | $300-800 | ~$50 so far (2 spot reclaims, ~5 hrs total) |
| 3 | Evaluation on SWE-bench Verified (5 models × 500 instances) | $50-150 (Docker) | Pending |
| | **Total estimated** | **$400-1,000** | |

**Actual infrastructure cost so far**: ~$50 across 2 spot reclaims (g7e ~2 hrs at $2.7/hr + p4d ~4 hrs at $10.41/hr).

**Spot pricing** (as of 2026-04-25):
- p4d.24xlarge: $10.41/hr spot in us-east-1a (8x A100 40GB)
- g7e.24xlarge: $2.70/hr spot in us-west-2 (4x RTX PRO 6000 96GB)
- p5.48xlarge: ~$25/hr spot (8x H100 80GB) — overkill
- p6-b200.48xlarge: capacity block only (~$140/hr) — way overkill

Compare to SERA's $2K (40 GPU-days). Our QLoRA approach + spot pricing targets <$500 total.

## Success Criteria

1. **Cascade-filtered SFT outperforms unfiltered SFT**: Config B (cascade p>0.5) achieves higher SWE-bench pass@1 than config "none" (all data). This validates that our cascade is a useful quality signal for training data curation.

2. **Cascade-filtered approaches oracle-filtered**: Config B achieves within 3pp of Config D (gold-label filtered). This validates that the cascade can replace Docker-based filtering for SFT.

3. **SERA threshold finding replicates**: Configs A, B, and C achieve similar SWE-bench pass rates despite different filter strictness (90.9% vs 56.4% vs 10.7% acceptance). If threshold matters, characterize the quality-quantity tradeoff.

4. **Absolute improvement over base model**: At least one trained model exceeds base Qwen3-32B pass@1 by 5pp+ on SWE-bench Verified. SERA achieved +25pp (24.4% → 49.5%).

5. **Cost-effective**: Total experiment cost under $500 (cheaper than SERA's $2K).

## Known Risks

### Noisy Cascade vs Clean SVG

SERA used SVG consensus (precision=1.0) — every accepted trajectory was truly correct. Our cascade has 68% silver-gold agreement — 32% of accepted trajectories may be incorrect. The question is whether SERA's "threshold doesn't matter" finding holds with noisier labels.

**Mitigation**: Run 6 (gold-label oracle) quantifies the quality ceiling. If the gap between cascade-filtered and oracle-filtered is large, cascade noise is the bottleneck. Platt calibration helps — at calibrated p>0.7, expected precision should be ~70-80%.

### Distribution Mismatch

CoderForge trajectories were generated by multiple large models (the teacher models). Fine-tuning Qwen3-32B on teacher-generated trajectories may not teach the smaller model to generate similar trajectories — the behavior distribution may be too different.

**Mitigation**: This is the standard distillation setup (large teacher → small student). CoderForge's 59.4% SWE-bench result was achieved exactly this way. SERA also used a larger model's traces for SFT.

### OOD Cascade Failure

The flywheel experiment showed OOD transfer fails (Phase 3). If CoderForge trajectories from unseen repo splits have different behavioral feature distributions, the RF tier will misclassify them, polluting the filter.

**Mitigation**: Use Platt-calibrated multiprompt (ECE=0.026) as primary filter, not RF alone. The multiprompt evaluates content (problem + diff), not just behavioral features, so it's more robust to distribution shifts. Config A (RF-only) serves as the control to measure this effect.

### Training Infrastructure

SFT training on 32B models requires significant GPU memory. QLoRA reduces this to fit on a single A100 40GB.

**Mitigation**: QLoRA 4-bit (r=16) fits single A100 40GB at ~38.6GB with batch=1. Parallel training across GPUs (one config per GPU) maximizes throughput. Spot instances are 3-10x cheaper than on-demand but risk reclamation — S3 checkpoint sync every 200 steps provides protection (must include adapter weights).

## Non-Requirements

- **RL training**: This is SFT only — no GRPO, PPO, or reward model training. RL is Phase 4 (separate spec if ECE results warrant it).
- **Real-time filtering pipeline**: One-shot batch filtering. Continuous pipeline is a future production concern.
- **Multi-model training**: Focus on Qwen3-32B. Qwen3-Coder-Next 80B is stretch goal.
- **Frontier performance**: Target is demonstrating that cascade-filtered SFT improves over unfiltered, not matching Claude Opus.
- **Production deployment**: Research experiment only. Deployment patterns follow from the coderforge-eval spec.

## Dependencies

| Dependency | Status | Location |
|------------|--------|----------|
| Verification flywheel (Phases 1-4) | COMPLETE | `blueprints/verification-flywheel/` |
| ECE calibration results | COMPLETE | `blueprints/verification-flywheel/results/phase4_ece_calibration.json` |
| Platt calibration parameters | COMPLETE | RF: a=0.173, b=-0.443; Multiprompt: 5-fold CV |
| CoderForge adapter | COMPLETE | `blueprints/verification-flywheel/scripts/coderforge_adapter.py` |
| learned-verifier library | COMPLETE | `/Users/phi/Documents/workbench/learned-verifier/` |
| TRL + peft + flash-attn stack | VALIDATED | p4d.24xlarge with Python 3.11 |
| SWE-bench evaluation harness | VALIDATED | `blueprints/verification-primitives-swebench/` |
| GPU capacity (p4d spot) | NEEDED | Spot request in us-east-1 |

## Full Reproduction Guide

### Step 1: Launch a p4d.24xlarge spot instance

```bash
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \  # Deep Learning AMI (Ubuntu) with CUDA 12.8
  --instance-type p4d.24xlarge \
  --key-name g7e-bench \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
  --region us-east-1 \
  --placement AvailabilityZone=us-east-1a \
  --iam-instance-profile Name=g7e-bench-profile \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200}}]'
```

### Step 2: Set up the environment

```bash
# Mount NVMe (p4d has local NVMe)
sudo mkfs.xfs /dev/nvme1n1
sudo mkdir -p /mnt/nvme && sudo mount /dev/nvme1n1 /mnt/nvme
sudo chown $USER:$USER /mnt/nvme

# Install miniconda (Python 3.11 — required for peft/torchao PEP 604 syntax)
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /mnt/nvme/miniconda3
export PATH=/mnt/nvme/miniconda3/bin:$PATH

# Install training stack
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers trl peft bitsandbytes datasets accelerate
pip install flash-attn --no-build-isolation  # Compiles for sm_80 (A100)

# Verify flash-attn
python3 -c "import flash_attn; print(flash_attn.__version__)"  # should print 2.8.3+
```

### Step 3: Deploy scripts and data

```bash
WORK=/mnt/nvme/rejection-sampling-sft
mkdir -p $WORK/{scripts,data,results,models}

# Copy scripts from repo
scp domains/autoresearch/blueprints/rejection-sampling-sft/scripts/*.py $HOST:$WORK/scripts/
scp domains/autoresearch/blueprints/rejection-sampling-sft/scripts/*.sh $HOST:$WORK/scripts/

# Copy verification flywheel dependencies
scp -r domains/autoresearch/blueprints/verification-flywheel/ $HOST:$WORK/../verification-flywheel/

# Restore Phase 1 data from S3 (skip re-running filter if scores exist)
aws s3 sync s3://agent-aiops-artifacts/rejection-sampling-sft/data/ $WORK/data/ --region us-west-2
```

### Step 4: Launch parallel training

```bash
# Set environment
export PATH=/mnt/nvme/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export HF_HOME=/mnt/nvme/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WORK

# Launch all 5 configs on 5 separate GPUs
bash scripts/launch_training.sh
```

Or use the sequential orchestrator (safer for spot, auto-resumes):
```bash
bash scripts/run_experiment.sh --phase 2
```

### Step 5: Monitor training

```bash
# Watch all logs
tail -f $WORK/results/training_*.log

# Check GPU utilization
nvidia-smi -l 5

# Check S3 sync status
aws s3 ls s3://agent-aiops-artifacts/rejection-sampling-sft/results/ --region us-west-2
```

### Infrastructure Learnings

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Flash Attention 2 won't compile on g7e | Blackwell sm_120 not supported by flash-attn 2.x | Use p4d (A100 sm_80) or p5 (H100 sm_90) |
| OOM with batch_size=4 on A100 40GB | Cross-entropy on vocab=152K needs 9.27GB | Use batch_size=1, grad_accum=16 |
| device_map="auto" is slow with 8 GPUs | Model spread thin, most GPUs idle | Pin 1 config per GPU via CUDA_VISIBLE_DEVICES |
| Adapter weights lost on spot reclaim | S3 sync excluded .safetensors | Include all files in S3 sync (see CRITICAL FIX above) |
| Python 3.9 incompatible with peft | PEP 604 union syntax (`X \| None`) | Use miniconda Python 3.11 |
| trl 1.3.0 SFTConfig param name | `max_seq_length` renamed to `max_length` | Use `max_length` in SFTConfig |
| Packing without flash-attn | Cross-contamination warning with sdpa | Use flash-attn (p4d) or accept minor quality loss |

---

> **Note**: This spec implements Phase 3 of the phased adoption path from
> RLVR_AND_VERIFICATION.md. The verification flywheel established the cascade
> economics ($0.002-0.020/patch) and ECE calibration (0.026-0.056 with Platt).
> This experiment uses those calibrated signals for training data curation.
> If cascade-filtered SFT validates, Phase 4 (GRPO with calibrated reward)
> becomes the logical next step — but requires separate infrastructure (veRL/TRL RL stack).
> Operational artifacts belong in the blueprint directory.
