# Autoresearch: Fine-Tuning Recipe Optimization

You are an autonomous ML researcher. Your goal is to find the best LoRA fine-tuning recipe by iterating on training configuration using the autoresearch-colab sentiment module (Unsloth + LoRA).

## Setup (run once)

1. Clone the repo and install dependencies:
```bash
cd /mnt/nvme
git clone https://github.com/aigorahub/autoresearch-colab.git finetuning
cd finetuning
uv sync
```

2. Verify GPU access:
```bash
uv run python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')"
```

3. Prepare your dataset. Two options:

**Option A: Sentiment (default — included in repo)**
The `sentiment/sample_data.csv` is ready to use. Upload your own CSV with `text`, `label`, and optionally `category` columns.

**Option B: Coding agent (advanced)**
Convert agent-harness trajectories to the chat template format expected by `sentiment/data.py`:
```bash
python convert_trajectories.py --input /mnt/nvme/agent-harness/results/ --output data/coding_agent.jsonl
```

4. Open and run `sentiment_autoresearch.ipynb` OR use the CLI loop below.

## CLI Experiment Loop

This follows the upstream `program.md` protocol — git branching, results.tsv, auto-revert.

### Initialize

```bash
cd /mnt/nvme/finetuning
git checkout -b autoresearch/finetune-$(date +%b%d | tr '[:upper:]' '[:lower:]')
```

Create `results.tsv` with the header:
```
commit	val_metric	memory_gb	status	description
```

### The loop

The sentiment module provides the building blocks — you wire them together:

```python
from sentiment.config import ExperimentConfig, sample_config, DEFAULT_CONFIG
from sentiment.data import load_dataset, train_val_split, format_for_training
from sentiment.evaluate import evaluate_model, print_eval_summary
```

**Fixed files** (do NOT edit):
- `sentiment/evaluate.py` — defines macro F1 metric
- `sentiment/data.py` — data loading and formatting
- `sentiment/sample_data.csv` — raw dataset

**Agent-editable**:
- The training script (notebook cells or a `train.py` you create)
- `sentiment/config.py` — search space boundaries and default config

LOOP FOREVER:

1. **Read** the current config, results.tsv, and experiment history
2. **Hypothesize** a change. The `sample_config(baseline, n_changes=2)` function mutates N params from current best — use it or make targeted manual changes. Categories:
   - **LoRA**: rank (4-64), alpha (8-128), dropout (0-0.1)
   - **Training**: lr (5e-5 to 1e-3), epochs (1-5), batch size, grad accum, warmup
   - **Optimizer**: adamw_8bit, adamw_torch, paged_adamw_8bit
   - **Scheduler**: linear, cosine, constant_with_warmup
   - **Data**: max_seq_length (256-1024), system prompt variations
   - **Model**: Qwen3.5-0.6B vs 1.7B (via Unsloth pre-quantized)
3. **Git commit** the config change
4. **Run** training: `uv run python train.py > run.log 2>&1`
5. **Evaluate**: Read macro_f1 from run.log (`grep "Macro F1" run.log`)
6. **Log** to results.tsv:
   ```
   <commit>	<macro_f1>	<memory_gb>	<keep|discard|crash>	<description>
   ```
7. **Decide**: If macro_f1 improved → keep (advance branch). If equal/worse → `git reset --hard HEAD~1`
8. **Repeat**

### Metric

The primary metric is **macro F1** on the held-out validation set (computed by `sentiment/evaluate.py`). This handles class imbalance better than accuracy.

Secondary: eval_loss from the trainer (correlated but not always aligned with F1).

## Rules

- NEVER edit `sentiment/evaluate.py` — it defines the metric
- NEVER edit `sentiment/data.py` or the raw dataset — data pipeline is fixed
- NEVER stop the loop unless explicitly told to by the user
- Each experiment must complete within 10 minutes wall-clock
- Log EVERY experiment to results.tsv, including crashes
- Git commit before every run, revert on regression
- If you run out of ideas, think harder: read the Unsloth docs, try combining near-misses, try radical changes (different base model, aggressive rank, unusual schedulers)

## GGUF Export (after loop)

When the user stops the loop, export the best adapter:

```python
# Merge LoRA into base and save as GGUF
model.save_pretrained_gguf("best_gguf", tokenizer, quantization_method="q8_0")
```

This produces a file deployable via Ollama or llama.cpp on CPU — no GPU required for inference.

## Adapting for Other Domains

The sentiment module is a template. To fine-tune for a different task:

1. **Swap the dataset**: Any CSV/JSON with `text` + `label` columns works
2. **Edit system prompts** in `sentiment/data.py` (or override in training script)
3. **Adjust search space** in `sentiment/config.py` (e.g., longer sequences for code)
4. Keep `evaluate.py` as-is — macro F1 works for any classification task

For generation tasks (not classification), replace the eval with a task-specific metric (e.g., HumanEval pass@1, BLEU, exact match).
