# Autoresearch Spec: Finetuning Recipes 1.7B

## Status: COMPLETE (2026-04-02) — Best F1: 0.8954, target 0.932 not reached

## Overview
Second iteration of LoRA fine-tuning autoresearch, scaling from Qwen3-0.6B to **Qwen3-1.7B** (3x bigger) with smarter search strategy. Uses the **autoresearch-colab sentiment module** framework on dair-ai/emotion (6-class sentiment, 20K samples). An autonomous agent iterates on LoRA config and training hyperparameters, measuring macro F1 on validation set. Goal: beat 0.932 (Qwen-1.8B literature benchmark) or match it with fewer trained parameters.

Based on [autoresearch-colab/sentiment](https://github.com/aigorahub/autoresearch-colab) framework: `config.py` (search space + mutation), `data.py` (multi-format loading + chat templates), `evaluate.py` (F1/accuracy metrics), and `sentiment_autoresearch.ipynb` (Colab loop).

**Key improvements from Run 1** (Qwen3-0.6B, achieved 0.883):
- Smarter search: grid sweep of top parameters (lr, rank, epochs) first, then random mutation
- Faster iteration: batch_size 16-32, 1 epoch for early experiments, 3 epochs for promising configs
- Larger eval set: 500+ samples (full validation split), reduced noise
- Process restart every 20 experiments to mitigate memory leak
- Stretch goal: also explore Qwen3-4B for comparison

## Components

### 1. Compute
- **Platform**: EC2 GPU instance (SSH)
- **Instance Type**: g5.xlarge (1x NVIDIA A10G, 24GB GDDR6)
- **Why g5**: QLoRA uses ~8-10GB for 1.7B, ~16-20GB for 4B stretch. A10G (24GB) is sufficient and cost-efficient (~$1.01/hr vs g7e.24xlarge ~$8+/hr). Well-tested with Unsloth/PEFT.
- **Scaling**: Single GPU per experiment

### 2. Codebase
- **Source**: `https://github.com/aigorahub/autoresearch-colab`
- **Framework**: Unsloth + PEFT + TRL (via `pyproject.toml`)
- **Fixed files** (agent must NOT edit):
  - `sentiment/evaluate.py` — macro F1 metric, per-class metrics, inference harness
  - `sentiment/data.py` — data loading (CSV/JSON/JSONL/TSV/XLSX), train/val split, chat template formatting
  - Raw dataset files (downloaded dair-ai/emotion)
- **Agent-editable files**:
  - `sentiment/config.py` — search space boundaries, `ExperimentConfig` dataclass, `sample_config()` mutation function
  - Training script (notebook cells or standalone `train.py`)
- **Agent instructions**:
  - `program.md` — loop protocol: baseline → grid sweep (lr × rank × epochs) → random mutation on promising regions → keep/revert → repeat

### 3. Experiment Protocol
- **Metric**: `macro_f1` on held-out validation set (higher is better) — computed by `sentiment/evaluate.py`
- **Secondary**: `eval_loss` from SFTTrainer (lower is better)
- **Time budget**: 10-20 minutes per experiment (Qwen3-1.7B ~3x slower than 0.6B)
- **Loop structure**:
  1. **Baseline**: Default config (lr=2e-4, rank=16, epochs=3, batch=8)
  2. **Grid sweep**: 3×3×2 = 18 experiments on lr × rank × epochs
     - lr: [5e-5, 2e-4, 5e-4]
     - rank: [8, 16, 32]
     - epochs: [1, 3]
  3. **Random mutation**: 2-parameter mutations on top-3 configs from grid, 30+ experiments
  4. **Process restart**: Kill and restart training script every 20 experiments (memory leak mitigation)
  5. **Stretch goal**: Repeat grid sweep with Qwen3-4B (time permitting)
- **Termination**: Manual stop after 50+ experiments OR when target (macro F1 > 0.932) achieved
- **Logging**: `results.tsv` (commit, macro_f1, eval_loss, memory, status, description) + git commits per experiment

### 4. Networking
- **Access**: SSH to g5 instance (`ssh -i ~/.ssh/g7e-bench.pem ubuntu@34.217.78.214`)
- **Instance**: `i-09c93b84546b6e29e` (us-west-2a, g5.xlarge, 100GB gp3)
- **No VPC/EKS required**

### 5. Storage
- **Base models**: HuggingFace cache (Unsloth pre-quantized models)
  - `unsloth/Qwen3-1.7B` (~3-4GB)
  - `unsloth/Qwen3-4B` (~8-10GB, stretch goal)
- **Data**: dair-ai/emotion dataset (20K samples, 16K train / 2K validation / 2K test)
  - Download to `/mnt/nvme/datasets/emotion/`
  - Use full validation split (2K samples) for eval, not 300-sample subset
- **Adapters**: LoRA checkpoints in `output/`, best in `best_adapter/`
- **Export**: GGUF for CPU deployment via Ollama/llama.cpp (optional)

## Search Space (from `sentiment/config.py`)

### Phase 1: Grid Sweep (18 experiments)

| Parameter | Values |
|-----------|--------|
| `learning_rate` | 5e-5, 2e-4, 5e-4 |
| `lora_r` | 8, 16, 32 |
| `num_epochs` | 1, 3 |

**Fixed during grid**: `lora_alpha=2*lora_r`, `lora_dropout=0.05`, `per_device_batch=16`, `grad_accum_steps=1`, `warmup_ratio=0.03`, `weight_decay=0.01`, `lr_scheduler=cosine`, `optim=adamw_8bit`, `max_seq_length=512`

### Phase 2: Random Mutation (30+ experiments)

| Category | Parameter | Values |
|----------|-----------|--------|
| LoRA | `lora_r` | 4, 8, 16, 32, 64 |
| LoRA | `lora_alpha` | 8, 16, 32, 64, 128 |
| LoRA | `lora_dropout` | 0.0, 0.05, 0.1 |
| Training | `learning_rate` | 5e-5 to 5e-4 |
| Training | `num_epochs` | 1, 3, 5 |
| Training | `per_device_batch` | 8, 16, 32 |
| Training | `grad_accum_steps` | 1, 2, 4 |
| Training | `warmup_ratio` | 0.0, 0.03, 0.05, 0.1 |
| Training | `weight_decay` | 0.0 to 0.1 |
| Training | `lr_scheduler` | linear, cosine, constant_with_warmup |
| Training | `optim` | adamw_8bit, adamw_torch, paged_adamw_8bit |
| Data | `max_seq_length` | 256, 512, 768, 1024 |

**Mutation strategy**: Pick 2 parameters to mutate per experiment (same as Run 1), but only from promising regions identified by grid sweep.

### Phase 3: Stretch Goal (18+ experiments)

| Parameter | Values |
|-----------|--------|
| `base_model` | `unsloth/Qwen3-4B` |

Repeat grid sweep with 4B model (time/resource permitting).

## Use Case: Multi-Class Sentiment Classification

- **Dataset**: dair-ai/emotion (20K samples, 6 classes: sadness, joy, love, anger, fear, surprise)
- **Task**: Context-dependent sentiment classification
- **Baseline**: Qwen3-1.7B zero-shot (~0.60-0.70 macro F1 estimated)
- **Target**: 0.932 macro F1 (Qwen-1.8B literature benchmark)
- **Evaluation**: Full validation set (2K samples), not 300-sample subset
- **Known issue**: Qwen3 thinking mode outputs `<think>` tags — `evaluate.py` must strip these before classification

## Lessons from Run 1 (Qwen3-0.6B)

1. **Random mutation is too narrow**: Only 2 of 13 hyperparameters explored over 20+ experiments. Grid sweep forces coverage.
2. **300-sample eval is noisy**: Macro F1 varies ±0.02 between runs. Use full validation split (2K).
3. **Slow iteration hurts**: ~70 min/experiment on Qwen3-0.6B. Fix: batch_size 16-32, 1 epoch for early grid.
4. **Memory leak**: Training crashes after ~20 experiments. Fix: restart process every 20 experiments.
5. **Pareto frontier not explored**: Agent never varied batch_size or grad_accum_steps (speed/memory tradeoff). Grid sweep forces this.
6. **Qwen3 thinking mode**: `<think>` tags break classification. Fix already in `evaluate.py`.
7. **0.883 is respectable**: 55% of literature benchmark (0.932 for Qwen-1.8B + LoRA). Qwen3-1.7B should close the gap.

## Success Criteria

1. **Grid sweep completes**: 18 baseline experiments (lr × rank × epochs) in <5 hours
2. **Target metric achieved**: Macro F1 > 0.932 (literature benchmark) OR macro F1 > 0.900 with proof of Pareto optimality (fewer trained params/faster)
3. **Faster iteration**: Average <20 min/experiment (vs 70 min on 0.6B)
4. **Process stability**: No memory leak crashes with 20-experiment restarts
5. **Results logged**: `results.tsv` with commit, macro_f1, eval_loss, memory, status, description + git commits
6. **Lessons document**: Captures which hyperparameters matter most (lr, rank, epochs vs secondary params)
7. **Stretch goal achieved** (optional): Qwen3-4B grid sweep completes, showing scaling behavior

## Non-Requirements

- Production serving of the fine-tuned model (separate blueprint)
- DPO/RLHF alignment — supervised fine-tuning only
- Multi-node or multi-GPU training
- Full dair-ai/emotion test set evaluation (2K samples) — validation set (2K) is sufficient
- GGUF export — optional, not blocking
- SWE-bench or HumanEval evaluation (coding use case deferred)

## Known Limitations

- **Unsloth Blackwell (sm_120) compatibility**: Run 1 worked on g7e, but verify before starting
- **Qwen3-1.7B memory**: ~8-10GB for QLoRA, fits single GPU. Qwen3-4B may need 16-20GB.
- **dair-ai/emotion is imbalanced**: Some classes have <1000 samples. Macro F1 accounts for this but per-class metrics will vary.
- **Literature benchmark is Qwen-1.8B, not 1.7B**: May not be exactly comparable, but close enough for guidance.
- **Memory leak persists**: Unsloth/PEFT memory leak not fixed upstream. Mitigation: restart every 20 experiments.
- **Grid sweep is exhaustive**: 18 experiments × 15 min = 4.5 hours minimum. Budget accordingly.
- **Qwen3-4B stretch goal is aspirational**: May not fit time budget. Prioritize 1.7B completion.

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
