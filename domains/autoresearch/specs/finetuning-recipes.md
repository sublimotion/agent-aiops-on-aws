# Autoresearch Spec: Finetuning Recipes

## Status: DRAFT

## Overview
Run the autoresearch loop on LoRA/QLoRA fine-tuning using the **autoresearch-colab sentiment module** — the same framework as training-recipes but for fine-tuning instead of pre-training. An autonomous agent iterates on LoRA config, training hyperparameters, and data formatting, measuring macro F1 on a held-out validation set.

Based on [autoresearch-colab/sentiment](https://github.com/aigorahub/autoresearch-colab) which provides: `config.py` (search space + mutation), `data.py` (multi-format loading + chat templates), `evaluate.py` (F1/accuracy metrics), and `sentiment_autoresearch.ipynb` (Colab loop).

## Components

### 1. Compute
- **Platform**: Bare metal GPU instance (SSH)
- **Instance Type**: g7e.24xlarge (4x RTX PRO 6000 Blackwell, 96GB GDDR7 each)
- **Minimum**: Single GPU with 16GB+ VRAM (Qwen3.5-0.6B QLoRA fits in ~5GB)
- **Scaling**: Single GPU per experiment

### 2. Codebase
- **Source**: `https://github.com/aigorahub/autoresearch-colab`
- **Framework**: Unsloth + PEFT + TRL (via `pyproject.toml`)
- **Fixed files** (agent must NOT edit):
  - `sentiment/evaluate.py` — macro F1 metric, per-class metrics, inference harness
  - `sentiment/data.py` — data loading (CSV/JSON/JSONL/TSV/XLSX), train/val split, chat template formatting
  - Raw dataset files (CSV/JSON)
- **Agent-editable files**:
  - `sentiment/config.py` — search space boundaries, `ExperimentConfig` dataclass, `sample_config()` mutation function
  - Training script (notebook cells or standalone `train.py`)
- **Agent instructions**:
  - `program.md` — loop protocol using upstream conventions (git branching, results.tsv, auto-revert)

### 3. Experiment Protocol
- **Metric**: `macro_f1` on held-out validation set (higher is better) — computed by `sentiment/evaluate.py`
- **Secondary**: `eval_loss` from SFTTrainer (lower is better)
- **Time budget**: 10 minutes per experiment
- **Loop structure**: Baseline → mutate config (2 params via `sample_config()`) → train → evaluate → keep/revert → repeat
- **Termination**: Manual stop (upstream protocol: NEVER STOP until human interrupts)
- **Logging**: `results.tsv` (commit, metric, memory, status, description) + git commits per experiment

### 4. Networking
- **Access**: SSH to g7e instance
- **No VPC/EKS required**

### 5. Storage
- **Base models**: HuggingFace cache (Unsloth pre-quantized models, ~1-4GB each)
- **Data**: `sentiment/sample_data.csv` (included) or user-provided dataset on `/mnt/nvme/`
- **Adapters**: LoRA checkpoints in `output/`, best in `best_adapter/`
- **Export**: GGUF for CPU deployment via Ollama/llama.cpp

## Search Space (from `sentiment/config.py`)

| Category | Parameter | Values |
|----------|-----------|--------|
| LoRA | `lora_r` | 4, 8, 16, 32, 64 |
| LoRA | `lora_alpha` | 8, 16, 32, 64, 128 |
| LoRA | `lora_dropout` | 0.0, 0.05, 0.1 |
| Training | `learning_rate` | 5e-5 to 1e-3 |
| Training | `num_epochs` | 1-5 |
| Training | `per_device_batch` | 2, 4, 8, 16 |
| Training | `grad_accum_steps` | 1, 2, 4, 8 |
| Training | `warmup_ratio` | 0.0, 0.03, 0.05, 0.1 |
| Training | `weight_decay` | 0.0 to 0.1 |
| Training | `lr_scheduler` | linear, cosine, constant_with_warmup |
| Training | `optim` | adamw_8bit, adamw_torch, paged_adamw_8bit |
| Data | `max_seq_length` | 256, 512, 768, 1024 |
| Model | `base_model` | Qwen3.5-0.6B, Qwen3.5-1.7B |

## Use Cases

### Default: Sentiment Classification
- Context-dependent sentiment (e.g., "bitter" positive for coffee, negative for chocolate)
- Dataset: `sentiment/sample_data.csv` or user-provided CSV with `text`, `label`, `category` columns
- Models: Qwen3.5-0.6B (~1-2 min/experiment on T4), Qwen3.5-1.7B (~2-4 min on A100)

### Advanced: Coding Agent (future)
- Convert agent-harness Phase 2 trajectories (11 passing issues) to chat template format
- Swap base model to Qwen2.5-Coder-7B or Devstral Small 2
- Extend `max_seq_length` to 4096-8192
- Replace macro F1 eval with task-specific metric (mini SWE-bench or HumanEval)

## Success Criteria

1. Autoresearch loop runs autonomously for 20+ experiments using `sample_config()` mutation
2. Fine-tuned model improves macro F1 by >5% over base model zero-shot
3. Results logged to `results.tsv` with git commits per experiment
4. Best adapter exported as GGUF for CPU deployment
5. Lessons document captures transferable insights about LoRA fine-tuning on Blackwell

## Non-Requirements
- Production serving of the fine-tuned model (separate blueprint)
- DPO/RLHF alignment — supervised fine-tuning only
- Multi-node or multi-GPU training
- Full SWE-bench evaluation (requires Docker)

## Known Limitations
- Unsloth Blackwell (sm_120) compatibility: may need nightly builds — verify before first run
- `sample_data.csv` is small (~100 samples) — risk of overfitting, use more data for serious experiments
- Agent-harness trajectory data is small (11 passing issues) — coding agent use case needs data augmentation
- Qwen3.5-0.6B/1.7B are small models — may not generalize well on complex coding tasks

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
