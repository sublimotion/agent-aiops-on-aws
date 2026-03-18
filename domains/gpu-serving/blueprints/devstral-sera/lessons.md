---
model: devstral-small-2
engine: vllm
hardware: g7e.24xlarge
gpu_arch: sm_120
deployment_date: "2026-03-04"

outcome: success
failure_categories:
  - nccl
  - tool_timeout

cards_used:
  mdc: [devstral-2-vllm]
  gpu_infra: [g7e]

card_helped: partial

benchmark:
  throughput_toks_s: 52.9
  ttft_p50_ms: null
  ttft_p99_ms: null
  concurrent_users: 1
  gpu_util_pct: null

ralph_iterations: 4

mdc_learn_commands:
  - 'mdc learn devstral-2 vllm "--tool-call-parser mistral --enable-auto-tool-choice required for correct tool call parsing"'
  - 'mdc learn devstral-2 vllm "model_type=mistral3, must import Mistral3ForConditionalGeneration directly"'

gpu_infra_learn_commands:
  - 'gpu-infra learn -c nccl "NCCL 2.25.1 broken on Blackwell sm_120 PCIe — ALL collective ops fail with Cuda failure 1. Fixed in 2.26.2 (NGC 25.03+). vLLM inference unaffected (uses custom allreduce)."'
  - 'gpu-infra learn -c platform "g7e containers need --network host (no CNI plugin on bare metal)"'
---

# Lessons Learned: Devstral Small 2 SERA Fine-Tuning

## Baseline (from qwen3-next-sglang benchmarks, 2026-03-03)

| Eval | Score | Notes |
|------|-------|-------|
| D2 BFCL | 91.7% | Tool calling, multi-turn |
| D5 Functional | 100% task, 97% SVG recall | Multi-turn bug fixing |
| D4 Swarm (4 rep) | 0% fail, 161 tok/s @ 32 agents | Round-robin across 4 GPUs |
| D1 Throughput | 52.9 tok/s (1 GPU) | Single-stream decode |

## Phase 0: Environment Setup (2026-03-04)

### What Works
- **Instance**: Feb 25 g7e.24xlarge (35.94.217.100), 4x RTX PRO 6000, 6.5 TB NVMe free
- **BF16 base weights**: Downloaded (49 GB) at `/mnt/nvme/models/devstral-small-2-bf16`
- **vLLM serving**: 4 replicas via nerdctl containers (vllm/vllm-openai:v0.15.0), ports 8000-8003
- **Python 3.11**: Installed via `sudo dnf install python3.11`, sera-venv at `/mnt/nvme/sera-venv`
- **Scripts staged**: All SERA scripts at `/mnt/nvme/sera-scripts/`
- **LB proxy**: Round-robin on port 9000 across 4 replicas

### Critical Fix: --tool-call-parser mistral

**Root cause**: vLLM replicas without `--tool-call-parser mistral --enable-auto-tool-choice` return Devstral's tool calls as raw `[TOOL_CALLS]` text in content with `finish_reason=stop`, instead of parsed `tool_calls` array. This caused the agent loop to exit after 1 turn.

**Impact**: 0/3 tests passed (model never used tools). With the fix, model does 20+ turns of proper tool use.

**All vLLM containers must include**: `--tool-call-parser mistral --enable-auto-tool-choice`

### Critical Fix: edit_file Tool

**Root cause**: With only `write_file`, the model tries to write entire files but uses placeholders like `# ... (previous content remains the same)`. Large files (20K+ chars) can't be reliably rewritten by a 24B model.

**Solution**: Added `edit_file` tool (search-and-replace) matching Claude Code's Edit pattern. Model uses it naturally for targeted changes.

**Also added**: `read_file` with `start_line`/`end_line` parameters for reading specific ranges.

### Fixed: Python Version Mismatch

**Root cause**: Host Python 3.9 can't run SWE-bench repo tests (Django needs 3.10+, `datetime.UTC` is 3.11).

**Solution**: Installed Python 3.11 (`sudo dnf install python3.11`), created sera-venv. `_install_repo_deps()` now uses `python3.11 -m venv` for workspace venvs + creates `python` → `python3` symlink.

### SVG Pipeline Validation (5-issue Django test)

| Metric | Value |
|--------|-------|
| Issues processed | 5 |
| Fixes generated | 3 |
| Tests passed | 1 (django__django-11039) |
| SVG accepted | 1 (recall=0.857) |
| Training examples | 1 |
| Avg fix turns | 20.0 |
| Avg latency | 112s |
| Pass rate | 20% |

**Rejection reasons**: 4x `fix_tests_fail`. Model hits 20-turn limit on harder issues. Increased to 30 turns for production run.

**Known issue**: Model uses `python` not `python3` in run_command — fixed with symlink in venv.

### Infrastructure State
- 4x vLLM containers (nerdctl) with `--tool-call-parser mistral --enable-auto-tool-choice`
- LB proxy on port 9000
- Python 3.11 sera-venv at `/mnt/nvme/sera-venv`
- FP8 weights at `/mnt/nvme/models/devstral-small-2-fp8`
- BF16 weights at `/mnt/nvme/models/devstral-small-2-bf16`

## Phase 1: SVG Data Generation — Production Run 1 (2026-03-04)

### COMPLETE (03:59–08:43 UTC, 4h 44m)

**Config**: SWE-bench Lite (300 issues), concurrency=4, max_turns=30, recall_threshold=0.8

### Final Results

| Metric | Value | Rate |
|--------|-------|------|
| Issues processed | 300 | — |
| Fixes generated | 246 | 82% |
| Tests PASS | 53 | 17.7% |
| **SVG accepted** | **28** | **9.3%** |
| Training examples | 28 | — |
| Avg recall (of passed) | 74.8% | — |
| Avg fix turns | 29.6 | — |
| Avg latency | 224s | — |
| Errors | 0 | 0% |

### All 28 Accepted Examples

| Instance ID | Recall | Repo |
|-------------|--------|------|
| django__django-11001 | 0.86 | django |
| django__django-11039 | 0.86 | django |
| django__django-11049 | 1.00 | django |
| django__django-11179 | 0.83 | django |
| django__django-11797 | 0.86 | django |
| django__django-11999 | 1.00 | django |
| django__django-12453 | 0.94 | django |
| django__django-12497 | 0.89 | django |
| django__django-12747 | 1.00 | django |
| django__django-13033 | 1.00 | django |
| django__django-13315 | 1.00 | django |
| django__django-13660 | 1.00 | django |
| django__django-13710 | 0.86 | django |
| django__django-13964 | 1.00 | django |
| django__django-14238 | 1.00 | django |
| django__django-14855 | 1.00 | django |
| django__django-14915 | 1.00 | django |
| django__django-14999 | 0.89 | django |
| django__django-15498 | 0.85 | django |
| django__django-15814 | 1.00 | django |
| django__django-15851 | 1.00 | django |
| django__django-16255 | 1.00 | django |
| django__django-16379 | 1.00 | django |
| django__django-16527 | 1.00 | django |
| django__django-16595 | 1.00 | django |
| django__django-17087 | 0.88 | django |
| pylint-dev__pylint-7993 | 0.88 | pylint |
| pytest-dev__pytest-11148 | 0.89 | pytest |

### Key Observations
- **Django dominance**: 26/28 accepted are Django. Other repos (astropy, sympy, matplotlib, scikit-learn) have dep install failures or complex build requirements on Python 3.11.
- **High recall quality**: 17/28 have perfect 1.00 recall — model very consistently reproduces fixes.
- **Rejection funnel**: 300 → 246 (fixes) → 53 (tests pass) → 28 (recall≥0.8). Biggest drop is tests-pass.
- **25 near-misses**: 25 issues passed tests but rejected by recall filter. Lowering threshold to 0.7 would yield ~40 examples.
- **Turn exhaustion**: Avg 29.6 turns = nearly all issues use all 30 turns. Model needs more budget.
- **Zero errors**: Pipeline is robust across 300 issues, 4h runtime.

## Phase 1.5: Production Run 2 — Partial (2026-03-04)

Ran on SWE-bench Verified (500 issues). Stopped at 131/500 (30 accepted, 47 tests pass) to start training. Dataset saved to S3 (`qwen3-next-bench-results-*` bucket).

Required adding inline IAM policy `sera-s3-upload` to `g7e-bench-role` for PutObject.

## Phase 2: LoRA Fine-Tuning (2026-03-04 – 03-05)

### Dataset

Mixed AI2 SERA + CoderForge-Preview:
- **AI2 SERA**: 34,127 examples (from `allenai/Sera-4.6-Lite-T2`)
- **CoderForge**: 155,144 examples (from `togethercomputer/CoderForge-Preview`, `filtered_reward1` split, 31 GB JSONL)
- **Mixed**: 180,445 train / 3,682 val (SERA 18%, CoderForge 82%), shuffled, max 200 messages/example

Scripts: `sera-mix-datasets.py`, `sera-normalize-content.py`, `sera-pretokenize.py`

### Training Container Setup

**Container**: `nvcr.io/nvidia/pytorch:25.02-py3` (PyTorch 2.7, CUDA 12.8)

**Critical**: Must use 25.02+, not 24.12. PyTorch 2.6 (24.12) lacks sm_120 (Blackwell) support. Also, 24.12 triggers `TransformGetItemToIndex` import error with transformers 5.2.0.

**Container launch**: Must use `--network host` (CNI plugin not available on bare-metal g7e).

```bash
sudo nerdctl run -d --name sera-training --gpus all --network host \
    --shm-size=64g -v /mnt/nvme:/mnt/nvme \
    nvcr.io/nvidia/pytorch:25.02-py3 sleep infinity
```

**Deps installed**: `trl==0.29`, `peft==0.18.1`, `deepspeed==0.18.6`, `bitsandbytes`, `transformers==5.2.0`

### Critical Lesson: Devstral Small 2 Model Loading

1. **model_type=mistral3**: `AutoModelForCausalLM` does NOT recognize `Mistral3Config`. Must import directly:
   ```python
   from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
   ```

2. **FP8 weights masquerading as BF16**: The "bf16" directory actually contained FP8-quantized weights (`quantization_config.quant_method: fp8` in config.json). Loading with `BitsAndBytesConfig` causes conflict. Fix: delete `quantization_config` from `config.json` on disk, use LoRA directly (no QLoRA needed).

3. **UNEXPECTED weight keys**: FP8 scale_inv and activation_scale keys show as UNEXPECTED during load. These are harmless — the model loads BF16 copies of the weights, ignoring the FP8 quantization metadata.

### Critical Lesson: Content Normalization for Mixed Datasets

AI2 SERA uses list-of-dicts content format `[{"type": "text", "text": "..."}]` while CoderForge uses plain strings. Mixing causes `ArrowInvalid: changed from string to array` when HuggingFace datasets tries to build a unified schema.

**Fix**: Run `sera-normalize-content.py` to flatten all content fields to strings before training. Produces `train_norm.jsonl` / `val_norm.jsonl`.

### Critical Lesson: Pre-Tokenize Before Multi-GPU Training

SFTTrainer tokenizes inside `main_process_first()` — rank 0 tokenizes while other ranks wait at a distributed barrier. With 180K examples at ~70 examples/s, tokenization takes ~45 min. Default barrier timeout is 1800s (30 min) → **timeout crash**.

Even with `ddp_timeout=7200`, the barrier uses NCCL under the hood, which hits the Blackwell NCCL bug (see below).

**Fix**: Pre-tokenize in a separate single-process step (`sera-pretokenize.py`), save to Arrow format, then load with `load_from_disk()` + `skip_prepare_dataset: True` in SFTTrainer.

### Critical Lesson: trl 0.29 API Quirks

- `max_seq_length` → `max_length` (renamed in trl 0.29)
- `trl sft` CLI: Not usable via `torchrun -m trl` (no `__main__`). `accelerate launch $(which trl) sft` also fails with argument parsing. **Use Python script approach instead.**
- `skip_prepare_dataset: True` via `dataset_kwargs` skips tokenization but requires pre-tokenized data with `input_ids`, `attention_mask`, `labels` columns
- Must set `remove_unused_columns=False` when using pre-tokenized data

### BLOCKING: NCCL Broken on Blackwell PCIe-Only Topology

**This is the most important lesson from Phase 2.**

**Symptom**: Every NCCL collective operation (all_reduce, broadcast, barrier) fails with:
```
NCCL WARN Cuda failure 1 'invalid argument' (enqueue.cc:1500)
ncclUnhandledCudaError: Call to CUDA function failed.
```

**Scope**: Affects ALL multi-GPU distributed training on g7e.24xlarge with RTX PRO 6000 Blackwell (sm_120). NCCL init succeeds, but operations fail. `NCCL_P2P_DISABLE=1` does not help.

**Environment**: NCCL 2.25.1, CUDA 12.8, PyTorch 2.7 (NGC 25.02), 4x RTX PRO 6000 Blackwell Server Edition (compute 12.0), PCIe Gen5 x16, no NVLink.

**Root cause**: NCCL 2.25.1 has a shared memory bug on Blackwell GPUs. Fixed in NCCL 2.26.2 release notes: *"Fixed shared memory usage on recent Blackwell GPUs."*

**Note**: vLLM inference serving works fine because it uses custom allreduce for PCIe >2 GPUs, bypassing NCCL.

**Workaround**: Single-GPU training. 24B FP8 model + LoRA fits in one 96 GB GPU (91 GB used). Training config:
- `CUDA_VISIBLE_DEVICES=0`, no accelerate/deepspeed
- `gradient_accumulation_steps=64` (effective batch=64)
- ~716s/step × 2,820 steps = ~23 days (4x slower than 4-GPU)

**Proper fix**: Upgrade to NGC PyTorch 25.03+ (NCCL ≥ 2.26.2).

### Would p5e Have the Same NCCL Issue?

**No.** p5e instances are completely different:

| Factor | g7e.24xlarge (broken) | p5e.48xlarge (OK) |
|--------|----------------------|-------------------|
| GPU | RTX PRO 6000 Blackwell | H200 SXM |
| Architecture | Blackwell (sm_120) | Hopper (sm_90) |
| Interconnect | PCIe Gen5 only | NVSwitch (900 GB/s) |
| NCCL support | New (2.25.1), buggy | Mature (since 2.18+) |
| Memory | 96 GB GDDR7 | 141 GB HBM3e |
| Bandwidth | ~1.5 TB/s | ~4.8 TB/s |

The bug requires both (a) Blackwell sm_120 AND (b) the broken shared memory kernel path. Hopper (sm_90) on NVSwitch is the most battle-tested NCCL configuration. Multi-GPU training on p5e would work out of the box.

**Training time estimate on p5e (8x H200)**:
- 8 GPUs, grad_accum=8 → effective batch=64
- ~3x faster per-step (HBM3e bandwidth + NVSwitch)
- Estimated ~2-3 days for 1 epoch on 180K examples

### Current Training State (2026-03-05)

Single-GPU training running on GPU 0:
- **GPU**: 91 GB / 96 GB VRAM, 100% util, 516W, 60°C
- **Config**: LoRA r=64 alpha=128, batch=1, grad_accum=64, lr=5e-5, max_length=16384
- **Speed**: ~716s/step (11.9 min), 2,820 total steps
- **ETA**: ~23 days at current rate
- **Log**: `/mnt/nvme/sera-checkpoints/mixed-run-001/train-v6.log`

### Training Config Summary

```python
# Model: Mistral3ForConditionalGeneration (FP8→BF16, 24B)
# LoRA: r=64, alpha=128, targets=[q,k,v,o,gate,up,down]_proj
# Data: 180K examples (AI2 SERA 18% + CoderForge 82%)
# Optimizer: AdamW via SFTTrainer defaults
# Schedule: cosine, warmup_ratio=0.03, lr=5e-5
# Batch: 1 × 64 grad_accum = 64 effective
# Sequence: max_length=16384, gradient_checkpointing=True
# Saves: every 500 steps, keep last 3
```

### Recommendations

1. **Short-term**: Let single-GPU run continue, reduce to grad_accum=16 for 4x faster training (~6 days) if smaller effective batch acceptable
2. **Medium-term**: Upgrade to NGC 25.03+ container (NCCL 2.26.2) to unlock 4-GPU training → ~6 days at full batch
3. **Best option**: Run on p5e.48xlarge with 8x H200 → ~2-3 days, no NCCL issues
4. **Post-training**: Merge LoRA adapter → quantize to FP8 → evaluate on SWE-bench
