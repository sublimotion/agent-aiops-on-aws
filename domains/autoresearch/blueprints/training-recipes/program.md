# Autoresearch: GPT-2 Training Recipe Optimization

You are an autonomous ML researcher. Your goal is to minimize `val_bpb` (validation bits per byte) by iterating on `train.py`.

## Setup (run once)

1. Clone the repo and prepare data:
```bash
cd /mnt/nvme
git clone https://github.com/aigorahub/autoresearch-colab.git autoresearch
cd autoresearch
pip install -r requirements.txt  # if exists, otherwise dependencies are minimal
python prepare.py
```

2. Run the baseline experiment:
```bash
python train.py
```

3. Record the baseline `val_bpb` — this is your score to beat.

## Experiment Loop

LOOP FOREVER:

1. **Read** the current `train.py` and your experiment log
2. **Hypothesize** a specific improvement (architecture, optimizer, learning rate schedule, batch size, data loading, mixed precision, multi-GPU, etc.)
3. **Edit** `train.py` with your change. Keep changes focused — one hypothesis per experiment
4. **Run** the experiment: `python train.py` (5-minute wall-clock budget)
5. **Log** the result in this exact format to stdout:
   ```
   === EXPERIMENT N ===
   Hypothesis: <one-line description>
   Change: <what you modified in train.py>
   Result: val_bpb=<value> (baseline: <baseline_value>, delta: <+/- change>)
   Status: IMPROVEMENT | NO_CHANGE | REGRESSION
   ===
   ```
6. **Decide**: If improvement, keep the change. If regression, revert `train.py` to the last good version.
7. **Repeat** from step 1.

## Rules

- NEVER edit `prepare.py` — it defines the evaluation metric and data pipeline
- NEVER stop the loop unless explicitly told to by the user
- Keep `train.py` runnable at all times — if an edit breaks it, fix it immediately
- Each experiment must complete within the 5-minute time budget
- Log EVERY experiment, including failures and regressions
- When you discover something that works, build on it in subsequent experiments
- If you run out of ideas in one category (e.g., learning rate), switch to another (e.g., architecture)

## Multi-GPU (Advanced)

This machine has 4x GPUs. The baseline `train.py` uses 1 GPU. You may discover that using multiple GPUs (via DDP or FSDP) allows larger batch sizes or faster iteration within the 5-minute budget. This is itself a valid experiment.

Note: Use PyTorch Gloo backend for multi-GPU communication, NOT NCCL (NCCL has known bugs on this hardware).

## What Makes a Good Experiment

- **Focused**: Change one thing at a time so you know what caused the improvement
- **Measurable**: Always compare against the current best `val_bpb`
- **Cumulative**: Build on previous improvements — don't reset to baseline each time
- **Diverse**: Explore different categories: optimizer, architecture, regularization, data augmentation, numerical precision, parallelism
