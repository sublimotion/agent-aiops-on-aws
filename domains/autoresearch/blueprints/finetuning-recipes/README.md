# Autoresearch: Fine-Tuning Recipes

Autonomous LoRA/QLoRA fine-tuning optimization using the [autoresearch-colab sentiment module](https://github.com/aigorahub/autoresearch-colab). Same autoresearch pattern as training-recipes (hypothesize → edit → run → measure → keep/revert) but applied to fine-tuning instead of pre-training.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Claude Code (autoresearch loop)                │
│  sample_config() → train → evaluate_model() →   │
│  keep/revert → repeat                           │
├─────────────────────────────────────────────────┤
│  autoresearch-colab/sentiment/                  │
│  ├── config.py    ← search space + mutation     │
│  ├── data.py      ← loading + chat templates    │
│  └── evaluate.py  ← macro F1 metric (FIXED)     │
├─────────────────────────────────────────────────┤
│  Unsloth + PEFT + TRL (SFTTrainer)              │
│  FastLanguageModel, LoRA, QLoRA                  │
├─────────────────────────────────────────────────┤
│  Base Model: Qwen3.5-0.6B / 1.7B (Unsloth)     │
│  Dataset: CSV/JSON with text + label columns    │
├─────────────────────────────────────────────────┤
│  g7e.24xlarge — RTX PRO 6000 Blackwell (96 GB)  │
└─────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. SSH to g7e
ssh -i ~/.ssh/g7e-bench.pem ec2-user@<IP>

# 2. Clone and setup
cd /mnt/nvme
git clone https://github.com/aigorahub/autoresearch-colab.git finetuning
cd finetuning
uv sync

# 3. Option A: Run the Colab notebook loop
#    Open sentiment_autoresearch.ipynb — it handles everything

# 4. Option B: Run via Claude Code agent
export CLAUDE_CODE_USE_BEDROCK=1
export AWS_REGION=us-east-1
export ANTHROPIC_MODEL="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
claude --print --dangerously-skip-permissions \
  "Read program.md. Run the autoresearch fine-tuning loop on sentiment/sample_data.csv."
```

## What autoresearch-colab Provides

| File | Role | Editable? |
|------|------|-----------|
| `sentiment/config.py` | Search space, `ExperimentConfig`, `sample_config()` mutation | Yes |
| `sentiment/data.py` | Load CSV/JSON/JSONL/TSV/XLSX, train/val split, chat templates | No |
| `sentiment/evaluate.py` | Macro F1, accuracy, per-class metrics | No |
| `sentiment/sample_data.csv` | Example dataset (consumer product sentiment) | No |
| `sentiment_autoresearch.ipynb` | Built-in notebook loop (baseline → mutate → train → eval) | Yes |

## Search Space

The `sample_config(baseline, n_changes=2)` function mutates N params from the current best:

| Category | Parameters |
|----------|-----------|
| LoRA | rank (4-64), alpha (8-128), dropout (0-0.1) |
| Training | lr (5e-5 to 1e-3), epochs (1-5), batch (2-16), grad_accum (1-8) |
| Optimizer | adamw_8bit, adamw_torch, paged_adamw_8bit |
| Scheduler | linear, cosine, constant_with_warmup |
| Data | max_seq_length (256-1024) |
| Model | Qwen3.5-0.6B, Qwen3.5-1.7B |

## Output

- **Best LoRA adapter** in `best_adapter/` (HuggingFace format)
- **GGUF export** for CPU deployment via Ollama/llama.cpp
- **results.tsv** with full experiment history

## References

- [autoresearch-colab](https://github.com/aigorahub/autoresearch-colab) — upstream repo
- [Unsloth](https://github.com/unslothai/unsloth) — 2-5x faster LoRA fine-tuning
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Training Recipes Blueprint](../training-recipes/) — same pattern, pre-training
- [Agent Harness Blueprint](../agent-harness/) — source of coding agent trajectory data
