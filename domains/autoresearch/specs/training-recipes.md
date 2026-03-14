# Autoresearch Spec: Training Recipes

## Status: DRAFT

## Overview
Run the autoresearch loop on GPT-2 training recipe optimization using the autoresearch-colab framework. Claude Code iterates autonomously — hypothesizing improvements, editing `train.py`, running 5-minute experiments, and logging results — targeting lower `val_bpb` (validation bits per byte).

Based on [autoresearch-colab](https://github.com/aigorahub/autoresearch-colab) (Karpathy's autoresearch pattern packaged for single-GPU).

## Components

### 1. Compute
- **Platform**: Bare metal GPU instance (SSH)
- **Instance Type**: g7e.24xlarge (4x RTX PRO 6000 Blackwell, 96GB GDDR7 each)
- **Fallback**: g7e.12xlarge (1x RTX PRO 6000) or any CUDA GPU
- **Scaling**: Single instance, no autoscaling

### 2. Codebase
- **Source**: `https://github.com/aigorahub/autoresearch-colab`
- **Fixed files** (agent must NOT edit):
  - `prepare.py` — data download (FineWeb-Edu), tokenizer training, shard creation, evaluation metric
- **Agent-editable files**:
  - `train.py` — GPT model definition, optimizer, training loop (~630 lines). Agent modifies this to experiment with architecture, hyperparameters, training techniques
- **Agent instructions**:
  - `program.md` — loop protocol, logging format, experiment structure

### 3. Experiment Protocol
- **Metric**: `val_bpb` (validation bits per byte) — lower is better
- **Time budget**: 5 minutes per experiment (fixed wall-clock)
- **Loop structure**: Read baseline → hypothesize improvement → edit `train.py` → run experiment → log result → repeat
- **Termination**: Manual stop, or convergence detection (N experiments with no improvement)
- **Logging**: Each experiment logs to stdout in structured format: experiment number, hypothesis, val_bpb result, delta from baseline

### 4. Networking
- **Access**: SSH to g7e instance
- **No VPC/EKS required** — runs directly on the instance

### 5. Storage
- **Model weights**: Local disk (tiny GPT-2, <1GB)
- **Data**: FineWeb-Edu shards downloaded by `prepare.py` to local disk
- **Results**: `experiments.jsonl` in blueprint results directory

## Adaptations from Colab

The original repo targets Colab (T4/L4 single-GPU). Our g7e.24xlarge adaptations:

1. **Multi-GPU discovery** — Agent can discover and use DDP/FSDP across 4 GPUs (the original is single-GPU). This is itself an experiment the agent should discover.
2. **Faster iterations** — RTX PRO 6000 (Blackwell) is ~3-5x faster than T4. 5-minute experiments will cover more training steps, potentially enabling discoveries not possible on weaker hardware.
3. **Container runtime** — Use `sudo nerdctl` (not docker) on g7e instances.
4. **NVMe staging** — Data and checkpoints on `/mnt/nvme` for fast I/O.

## Success Criteria

1. Autoresearch loop runs autonomously for 20+ experiments without human intervention
2. At least one experiment improves `val_bpb` over the baseline `train.py`
3. Structured experiment log captures all hypotheses and results
4. Lessons document captures transferable insights about the autoresearch pattern itself

## Non-Requirements
- Production deployment — this is a research experiment
- Terraform infrastructure — bare metal SSH
- Multi-node distributed training
- Fine-tuning on custom datasets (that's the `sentiment/` variant, a future blueprint)
- Cost optimization — g7e spot is ~$5-8/hr, acceptable for research

## Known Limitations
- NCCL on Blackwell PCIe (g7e) is broken for NCCL <= 2.25.1. If agent discovers DDP, must use NCCL >= 2.26.2 or PyTorch Gloo backend.
- `prepare.py` downloads ~10GB of FineWeb-Edu data on first run
- The 5-minute time budget is wall-clock, not GPU-time — includes data loading overhead

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
> See `blueprints/training-recipes/lessons.md`, `blueprints/training-recipes/results/`, etc.
