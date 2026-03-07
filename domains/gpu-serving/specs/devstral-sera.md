# Devstral Small 2 SERA Fine-Tuning Experiment

## Status: DRAFT (2026-03-03)

## Overview

Fine-tune Devstral Small 2 (24B) using the SERA (Soft-Verified Efficient Repository Agents) SVG pipeline to close the gap with Claude Opus 4.6 on coding agent tasks. Target: +5-12 pts SWE-bench improvement (68% → 73-80%) at ~$650-1,250 per iteration, running entirely on existing g7e.24xlarge infrastructure.

**Why Devstral Small 2:**
- Strongest $/quality ratio in our benchmarks: 91.7% BFCL, 100% functional eval, 97% SVG recall — on 1 GPU
- 24B dense, standard GQA, Apache 2.0 — no architecture complications for training
- Already proven in our serving stack: vLLM v0.15.0, 52.9 tok/s single-stream, zero failures at 32-agent swarm

**Why SERA:**
- Self-play data generation eliminates the most expensive part of fine-tuning: curating tool-use training examples
- SVG (Soft-Verified Generation) uses the model's own fix reproductions as quality signal — our D5 eval showed 97% SVG recall, confirming the signal is strong
- Allen AI demonstrated 2x SWE-bench improvement on Qwen 3-32B (24.4% → 49.5%) for ~$2,000 compute

## Parent Benchmarks

See `blueprints/qwen3-next-sglang/lessons.md` for the baseline eval results this experiment builds on:
- D2: BFCL 91.7% (Devstral, tool calling)
- D4: Swarm 0% failure at 32 agents (4 replicas)
- D5: Functional eval 100% completion, 97% SVG recall

---

## Components

### 1. Compute

| Phase | Instance | GPUs | Duration | Est. Cost |
|-------|----------|------|----------|-----------|
| Data generation | g7e.24xlarge (existing Feb 25) | 4x RTX PRO 6000 | 2-3 days | $800-1,200 |
| Fine-tuning | g7e.24xlarge (same instance) | 4x RTX PRO 6000 | 6-12 hrs | $100-200 |
| Evaluation | g7e.24xlarge (same instance) | 1 GPU (inference) | 2-4 hrs | $35-70 |
| **Total** | | | **3-5 days** | **$935-1,470** |

- **Region**: us-west-2 (existing instance)
- **Instance**: `i-03955d59a22d67ad1` (35.94.217.100), already running with all 4 GPUs loaded for vLLM
- **Key**: `~/.ssh/g7e-bench.pem`

### 1a. GPU & NCCL Pre-Flight (required for multi-GPU)

Before launching multi-GPU training (DeepSpeed ZeRO-2), run these diagnostics:

1. **GPU inventory**: `nvidia-smi` — verify 4x RTX PRO 6000 Blackwell, driver, CUDA, ECC
2. **Topology**: `nvidia-smi topo -m` — confirm PCIe-only (no NVLink on g7e)
3. **ECC/Xid**: `nvidia-smi --query-gpu=ecc.errors.*` + `dmesg | grep Xid` — zero uncorrected errors
4. **NCCL collective test**: Run `scripts/nccl_diag.py` (all_reduce, broadcast, barrier across 4 GPUs)
5. **NCCL version check**: Must be ≥ 2.26.2 for Blackwell sm_120 PCIe support

**BLOCKING**: NCCL ≤ 2.25.1 (NGC PyTorch 25.02) has shared memory bug on Blackwell PCIe. All collective ops fail with `Cuda failure 1 'invalid argument'`. Upgrade to NGC 25.03+ or use single-GPU training as workaround. See `blueprints/devstral-sera/lessons.md` for full details.

### 2. Model

- **Base model**: `mistralai/Devstral-Small-2-24B-Instruct-2512` (Apache 2.0)
- **Weights on instance**: `/mnt/nvme/models/devstral-small-2-fp8` (49 GB, FP8)
- **Training format**: QLoRA (4-bit NF4 base + LoRA adapters in BF16)
  - LoRA rank: 64
  - LoRA alpha: 128
  - Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Output**: Merged BF16 checkpoint + FP8 quantized checkpoint for inference
- **Serving**: vLLM v0.15.0 (same image as baseline)

### 3. Training Data — SVG Pipeline

The SVG (Soft-Verified Generation) pipeline generates training pairs where the model fixes a bug, then proves it can reproduce the fix from a PR description.

#### Data Sources

| Source | Repos | Issues | Purpose |
|--------|-------|--------|---------|
| SWE-bench Lite | 12 Python repos | 300 | Standard benchmark alignment |
| SWE-bench Verified | 12+ Python repos | 500 | Expanded coverage |
| Internal repos (opt.) | User's repos | Variable | Codebase specialization |

#### SVG Pipeline Steps

```
For each (repo, issue) pair:
  1. GENERATE: Model produces a fix using multi-turn tool use
  2. VERIFY: Run repo's test suite against the fix
  3. DESCRIBE: Convert fix to a PR description (natural language)
  4. REPRODUCE: Model generates fix from PR description alone
  5. SCORE: Compute line-level patch recall between (1) and (4)
  6. FILTER: Keep pairs where verify=PASS AND recall ≥ 0.8
  7. FORMAT: Convert to chat-format training examples
```

#### Data Generation Infrastructure

- 4x vLLM replicas (1 per GPU, ports 8000-8003) behind round-robin proxy
- Each replica handles SVG generation independently
- Estimated throughput: ~160 tok/s aggregate → ~200-400 training pairs/hour
- Target: 5,000-10,000 verified training pairs

#### Training Example Format

Each training example is a multi-turn conversation:

```json
{
  "messages": [
    {"role": "system", "content": "You are a coding assistant..."},
    {"role": "user", "content": "Fix this issue: <PR description>"},
    {"role": "assistant", "content": null, "tool_calls": [...]},
    {"role": "tool", "content": "<file contents>", "tool_call_id": "..."},
    {"role": "assistant", "content": null, "tool_calls": [...]},
    {"role": "tool", "content": "File written...", "tool_call_id": "..."},
    {"role": "assistant", "content": "I've fixed the issue by..."}
  ]
}
```

### 4. Storage

- **Model weights**: `/mnt/nvme/models/` (NVMe, already staged)
- **Training data**: `/mnt/nvme/sera-data/` (generated SVG pairs, ~5-20 GB)
- **Checkpoints**: `/mnt/nvme/sera-checkpoints/` (LoRA adapters + merged weights)
- **No FSx needed** — everything fits on instance NVMe

### 5. Training Configuration

```yaml
# QLoRA config
model_name_or_path: /mnt/nvme/models/devstral-small-2-fp8
output_dir: /mnt/nvme/sera-checkpoints/run-001
dataset: /mnt/nvme/sera-data/train.jsonl

# LoRA
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

# Training
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
num_train_epochs: 3
learning_rate: 2e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
max_seq_length: 8192
bf16: true

# Hardware
deepspeed: ds_config_zero2.json  # ZeRO-2 across 4 GPUs
gradient_checkpointing: true
```

**VRAM budget (QLoRA, 4x GPU with ZeRO-2):**

| Component | Per GPU |
|-----------|---------|
| Base model (NF4) | ~14 GB |
| LoRA adapters (BF16) | ~2 GB |
| Optimizer states (ZeRO-2) | ~8 GB |
| Activations + grad checkpointing | ~12 GB |
| **Total** | **~36 GB** |

Comfortable fit in 96 GB/GPU. Room to increase batch size or sequence length if needed.

---

## Experiment Phases

### Phase 0: Environment Setup (2 hrs)

1. Install training dependencies in a container on the Feb 25 instance
   - `axolotl` or `trl` (HuggingFace Transformers Reinforcement Learning)
   - `peft`, `bitsandbytes`, `deepspeed`
   - `datasets`, `wandb` (optional logging)
2. Download BF16 base weights (QLoRA needs unquantized base for NF4 quantization)
   - `mistralai/Devstral-Small-2-24B-Instruct-2512` (~48 GB BF16)
3. Verify single-GPU training smoke test (10 steps on dummy data)

### Phase 1: SVG Data Generation (2-3 days)

1. Write `scripts/sera-datagen.py` — SVG pipeline script
   - Input: list of (repo, issue, test_cmd) triples from SWE-bench
   - Uses 4x vLLM replicas for parallel generation
   - Implements the 7-step SVG pipeline (generate → verify → describe → reproduce → score → filter → format)
2. Run on SWE-bench Lite (300 issues) first as validation
   - Expected yield: ~150-200 verified pairs (50-65% pass rate based on D5 results)
3. Scale to SWE-bench Verified (500 issues)
   - Expected yield: ~300-400 additional pairs
4. Total target: 5,000-10,000 pairs (supplement with multiple fix attempts per issue)

**Data quality checkpoints:**
- Verify ≥50% of generated fixes pass repo tests
- SVG recall ≥0.8 for filtered pairs
- Manual spot-check 50 examples for format correctness

### Phase 2: Fine-Tuning (6-12 hrs)

1. Stop vLLM replicas (free GPU memory)
2. Launch QLoRA training with `trl` SFTTrainer or `axolotl`
3. Train for 3 epochs on filtered SVG data
4. Merge LoRA adapters into base model
5. Quantize merged checkpoint to FP8 for inference

**Checkpoints:**
- Save every 500 steps
- Evaluate on held-out 10% of SVG data at each checkpoint
- Track training loss curve for early stopping

### Phase 3: Evaluation (2-4 hrs)

Re-run the full benchmark suite on the fine-tuned checkpoint:

| Eval | Script | Baseline | Target |
|------|--------|----------|--------|
| D2-SERA | `scripts/bfcl-eval.py` | 91.7% | ≥92% (maintain) |
| D5-SERA | `scripts/functional-eval.py` | 100% / 97% SVG | ≥100% / ≥97% |
| D4-SERA | `scripts/swarm-simulator.py` (4 replicas) | 0% fail, 161 tok/s | ≥0% fail, ≥150 tok/s |
| SWE-bench-mini | SWE-bench Lite subset (50 issues) | ~68% (est.) | ≥73% |

**Success criteria:**
1. **Primary**: SWE-bench-mini ≥73% (+5 pts over base)
2. **Secondary**: No regression on BFCL (≥90%), functional eval (≥80%), swarm (≤5% fail)
3. **Stretch**: SWE-bench-mini ≥78% (+10 pts)

### Phase 4: Iteration (if Phase 3 succeeds)

1. Analyze failure patterns from SWE-bench evaluation
2. Generate targeted SVG data for weak categories
3. Retrain with augmented dataset
4. Re-evaluate — expect diminishing returns after 2-3 iterations

---

## Non-Requirements

- **No distributed training across instances** — QLoRA on 4 GPUs is sufficient for 24B
- **No RLHF/DPO** — SVG self-play is the training signal; no reward model needed
- **No production deployment** — this is an experiment; production serving is a separate spec
- **No custom CUDA kernels** — use off-the-shelf QLoRA + DeepSpeed
- **No SWE-bench full (2,294 issues)** — Lite + Verified (800 issues) is sufficient for the experiment

## Security Requirements

- Model weights stay on instance NVMe (no S3 upload of fine-tuned weights)
- Training data (SWE-bench repos) are public open-source
- No internal code used in Phase 1 (defer codebase-specific SERA to a future iteration)
- WandB logging is optional and can be disabled for air-gap

## Known Limitations

1. **FP8 base weights may not work with QLoRA** — bitsandbytes NF4 quantization expects BF16/FP16 base weights. Must download the BF16 checkpoint separately (~48 GB).
2. **SVG recall ≠ code quality** — high recall means the model is consistent, not necessarily correct. Filtering by test-pass ensures functional correctness.
3. **SWE-bench is Python-heavy** — improvements may not transfer to TypeScript, Go, Rust, etc. Codebase-specific SERA (future iteration) addresses this.
4. **Training data contamination** — Devstral may have seen SWE-bench issues in pre-training. Mitigate by comparing base vs fine-tuned on the same held-out set.
5. **vLLM tool-call parser quirks** — the Mistral parser's `call_0` ID format was already fixed in `bfcl-eval.py`. Fine-tuned model may generate different tool-call formats if training data uses a different template.
6. **Instance cost** — Feb 25 instance at $16.57/hr is running continuously. Total experiment cost includes idle time between phases. Coordinate phases to minimize gaps.

## Cost Considerations

| Phase | Duration | Instance Cost | Other | Total |
|-------|----------|--------------|-------|-------|
| P0: Setup | 2 hrs | $33 | — | $33 |
| P1: Data gen | 48-72 hrs | $800-1,200 | — | $800-1,200 |
| P2: Training | 6-12 hrs | $100-200 | — | $100-200 |
| P3: Eval | 2-4 hrs | $35-70 | — | $35-70 |
| **Total (1 iteration)** | **58-90 hrs** | | | **$968-1,503** |

**Break-even analysis**: At 10 engineers using Devstral for coding agents, SERA saves ~$10-20/eng/mo in retry costs (from higher accuracy). Break-even at ~50-150 engineer-months — worthwhile if the team is ≥10 engineers or if running multiple iterations.

## Scripts to Build

| Script | Purpose | Priority |
|--------|---------|----------|
| `scripts/sera-datagen.py` | SVG pipeline: generate, verify, describe, reproduce, score, filter | P0 |
| `scripts/sera-format.py` | Convert filtered SVG pairs to chat-format JSONL for training | P0 |
| `scripts/sera-train.sh` | Launch QLoRA training with trl/axolotl + DeepSpeed ZeRO-2 | P0 |
| `scripts/sera-merge.py` | Merge LoRA adapters + quantize to FP8 | P1 |
| `scripts/sera-eval.sh` | Run full eval suite (D2/D4/D5 + SWE-bench-mini) on fine-tuned model | P1 |
| `scripts/swebench-mini.py` | SWE-bench Lite subset evaluator (50 issues, repo checkout + test) | P1 |

---

> **Note**: Operational artifacts (lessons learned, benchmark results, training logs,
> checkpoint evaluations) belong in the blueprint directory, not in this spec.
> See `blueprints/devstral-sera/lessons.md`, `blueprints/devstral-sera/results/`, etc.
