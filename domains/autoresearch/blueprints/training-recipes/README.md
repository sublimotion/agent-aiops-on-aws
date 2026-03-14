# Training Recipes Autoresearch

Autonomous GPT-2 training recipe optimization using Claude Code in the autoresearch loop pattern.

## What This Is

Based on [autoresearch-colab](https://github.com/aigorahub/autoresearch-colab) — Claude Code autonomously iterates on `train.py` to minimize `val_bpb` (validation bits per byte). Each experiment runs for 5 minutes. The agent hypothesizes improvements, edits the training code, runs the experiment, logs results, and repeats.

## Quick Start

```bash
# 1. Setup remote (clone repo + prepare data)
./scripts/run-loop.sh

# 2. SSH to instance and start the loop
ssh -i ~/.ssh/g7e-bench.pem ec2-user@35.94.217.100
cd /mnt/nvme/autoresearch
# Launch Claude Code with program.md as context
```

## Architecture

```
autoresearch-colab/
├── prepare.py    # FIXED — data download, tokenizer, eval metric
├── train.py      # AGENT-EDITABLE — GPT model, optimizer, training loop
└── program.md    # Agent instructions (copied from this blueprint)
```

The agent edits ONLY `train.py`. The evaluation metric (`val_bpb`) is defined in `prepare.py` and cannot be gamed.

## Hardware

- **Target**: g7e.24xlarge (4x RTX PRO 6000 Blackwell, 96GB GDDR7 each)
- **Minimum**: Any CUDA GPU (T4, L4, A100, etc.)
- **Multi-GPU**: Agent may discover DDP/FSDP as an optimization (use Gloo backend, not NCCL)

## Key Files

| File | Purpose |
|------|---------|
| `program.md` | Agent loop instructions |
| `scripts/run-loop.sh` | Remote setup script |
| `lessons.md` | Operational lessons (append-only) |
| `results/experiments.jsonl` | Structured experiment log |

## References

- [Karpathy's autoresearch](https://www.latent.space/p/ainews-autoresearch-sparks-of-recursive) — 118 experiments, 11% speedup
- [Shopify/Liquid autoresearch](https://github.com/Shopify/liquid/pull/2056) — 53% faster, 61% fewer allocations
- [autoresearch-colab repo](https://github.com/aigorahub/autoresearch-colab)
