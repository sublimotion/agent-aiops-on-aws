---
blueprint: "finetuning-recipes-1.7b"
domain: "autoresearch"
spec: "domains/autoresearch/specs/finetuning-recipes-1.7b.md"
status: "completed"
last_updated: "2026-04-02T06:00:00Z"
last_stage: "stage-9"

stages:
  - id: "stage-1"
    name: "Read Spec"
    status: "completed"
  - id: "stage-2"
    name: "Validate Environment"
    status: "completed"
  - id: "stage-3"
    name: "Setup Codebase"
    status: "completed"
  - id: "stage-4"
    name: "Configure Loop"
    status: "completed"
  - id: "stage-5"
    name: "Run Baseline"
    status: "completed"
  - id: "stage-6"
    name: "Execute Grid Sweep"
    status: "completed"
  - id: "stage-7"
    name: "Execute Random Mutation"
    status: "completed"
  - id: "stage-8"
    name: "Analyze Results"
    status: "completed"
  - id: "stage-9"
    name: "Capture Lessons"
    status: "completed"

phases:

artifacts:
  lessons: true
  readiness_audit: []
  deployment_log: []
  compound: []
  benchmark_report: false
---

# Progress: finetuning-recipes-1.7b

## Summary

**48 experiments completed** (18 grid sweep + 30 random mutations) on g5.xlarge (A10G 24GB).

| Metric | Value |
|--------|-------|
| Best macro F1 | **0.8954** |
| Target | 0.932 |
| Gap to target | -3.7pp |
| Improvement over Run 1 (0.6B) | +1.2pp |
| Total experiments | 48 |
| Total compute time | ~26 hours |
| Compute cost | ~$26 (g5.xlarge @ $1.01/hr) |

## Instance
- **Type**: g5.xlarge (1x A10G, 24GB GDDR6)
- **ID**: i-09c93b84546b6e29e (us-west-2a)
- **SSH**: `ssh -i ~/.ssh/g7e-bench.pem ubuntu@34.217.78.214`
- **Cost**: ~$1.01/hr

## Best Config (Exp 18)

```
learning_rate: 5e-4
lora_r: 32
lora_alpha: 64
num_epochs: 5
per_device_batch: 16
grad_accum_steps: 1
warmup_ratio: 0.03
weight_decay: 0.01
lr_scheduler: cosine
optim: adamw_torch
lora_dropout: 0.05
max_seq_length: 512
```

## Phase 1: Grid Sweep (18 experiments)

| Exp | F1 | LR | Rank | Alpha | Epochs | Time | Status |
|-----|------|----|------|-------|--------|------|--------|
| 0 | 0.6861 | 5e-5 | 8 | 16 | 1 | 14m | keep |
| 1 | 0.8330 | 5e-5 | 8 | 16 | 3 | 30m | keep |
| 2 | 0.7287 | 5e-5 | 16 | 32 | 1 | 13m | discard |
| 3 | 0.8517 | 5e-5 | 16 | 32 | 3 | 30m | keep |
| 4 | 0.7656 | 5e-5 | 32 | 64 | 1 | 13m | discard |
| 5 | 0.8656 | 5e-5 | 32 | 64 | 3 | 30m | keep |
| 6 | 0.8166 | 2e-4 | 8 | 16 | 1 | 13m | discard |
| 7 | 0.8788 | 2e-4 | 8 | 16 | 3 | 30m | keep |
| 8 | 0.8353 | 2e-4 | 16 | 32 | 1 | 13m | discard |
| 9 | 0.8797 | 2e-4 | 16 | 32 | 3 | 30m | keep |
| 10 | 0.8499 | 2e-4 | 32 | 64 | 1 | 13m | discard |
| 11 | 0.8706 | 2e-4 | 32 | 64 | 3 | 30m | discard |
| 12 | 0.8602 | 5e-4 | 8 | 16 | 1 | 13m | discard |
| 13 | 0.8748 | 5e-4 | 8 | 16 | 3 | 30m | discard |
| 14 | 0.8645 | 5e-4 | 16 | 32 | 1 | 13m | discard |
| 15 | 0.8851 | 5e-4 | 16 | 32 | 3 | 30m | keep |
| 16 | 0.8653 | 5e-4 | 32 | 64 | 1 | 13m | discard |
| 17 | **0.8864** | 5e-4 | 32 | 64 | 3 | 30m | **grid best** |

**Grid sweep best**: 0.8864 (lr=5e-4, rank=32, 3 epochs)

## Phase 2: Random Mutation (30 experiments)

| Exp | F1 | Key mutations from best | Status |
|-----|------|------------------------|--------|
| 18 | **0.8954** | 5 epochs, adamw_torch | **NEW BEST** |
| 19 | 0.8850 | grad_accum=2, constant_with_warmup | discard |
| 20 | 0.8883 | rank=8, 3 epochs | discard |
| 21 | 0.8680 | lr=1e-4, warmup=0.05 | discard |
| 22 | 0.8953 | rank=16, seq=256 | discard (ties) |
| 23 | 0.8873 | lr=2e-4, paged_adamw | discard |
| 24 | 0.8580 | lr=5e-5, batch=32 | discard |
| 25 | 0.8944 | wd=0.1, linear sched | discard |
| 26 | 0.8936 | batch=8, warmup=0.1 | discard |
| 27 | 0.8821 | grad_accum=4, seq=256 | discard |
| 28 | 0.8574 | 1 epoch, grad_accum=2 | discard |
| 29 | 0.8654 | lr=1e-4, paged_adamw | discard |
| 30 | 0.8891 | warmup=0, wd=0.1 | discard |
| 31 | 0.8830 | lr=1e-4, constant_with_warmup | discard |
| 32 | 0.8814 | alpha=32, seq=256 | discard |
| 33 | 0.8784 | constant_with_warmup, paged_adamw | discard |
| 34 | 0.8948 | batch=8, seq=256 | discard |
| 35 | 0.8802 | wd=0, seq=768 | discard |
| 36 | 0.8888 | rank=16, 3 epochs | discard |
| 37 | 0.8409 | 1 epoch, grad_accum=4 | discard |
| 38 | 0.8812 | rank=64, 3 epochs | discard |
| 39 | 0.8513 | lr=2e-4, 1 epoch | discard |
| 40 | 0.8873 | lr=2e-4, paged_adamw | discard |
| 41 | 0.8795 | wd=0.1, dropout=0 | discard |
| 42 | 0.8688 | lr=5e-5, rank=4 | discard |
| 43 | 0.8836 | wd=0.05, dropout=0.1 | discard |
| 44 | 0.8837 | alpha=16, seq=768 | discard |
| 45 | 0.8876 | warmup=0.05, wd=0.05 | discard |
| 46 | 0.8809 | rank=16, batch=32 | discard |
| 47 | 0.8944 | linear sched, paged_adamw | discard |

## Bugs Fixed

1. **Unsloth generate() shape mismatch on A10G**: Switched to logit-based classification (forward pass + argmax over label tokens)
2. **torchvision version conflict**: Let Unsloth manage torch version
3. **python3.10-venv missing**: `apt install python3.10-venv`
4. **set_seed(42) resets mutation RNG**: SFTTrainer's `seed=42` calls `set_seed()` which resets Python's `random` module, causing identical mutations every time. Fixed by using a separate `random.Random(os.urandom(4))` instance for mutations.
