# CoderForge Eval — Progress

## Status: PHASE 1 TRAINING (LoRA)

## Infrastructure

| Resource | Instance | IP | Status |
|----------|----------|-----|--------|
| g7e.24xlarge (inference/eval) | i-03955d59a22d67ad1 | 35.93.254.135 | Idle (Phase 0 complete) |
| p6-b200.48xlarge (training) | i-06af825a7d467dda7 | 54.68.90.65 | Spot, us-west-2b, ~$25.60/hr |

### B200 Setup
- 8x B200, 183GB HBM3e each (1.46TB total), 2TB RAM
- 28TB NVMe RAID0 at `/mnt/nvme`
- Model: Qwen3.5-122B-A10B BF16 (234GB, 39 safetensors shards)
- Dataset: CoderForge-Preview (22GB, 155K trajectories)
- Training env: Python 3.12, torch 2.11.0+cu128 (sm_100), transformers 5.4, peft 0.18.1

## Phase 0: Quick Signal — LAION CoderForge-8B (2026-03-28)

### Result: NEGATIVE — Community 8B fine-tune is non-functional

| Metric | CoderForge-8B × SERA | SERA-32B × SERA (ref) | Devstral × OpenCode (ref) |
|--------|----------------------|----------------------|--------------------------|
| Fix rate | **~8%** (3/37 interim) | 64% | 88% |
| Test pass | **~3%** (1/37 interim) | — | 22% |
| Avg turns | 9.8 | — | — |

### Root Cause Analysis
1. **Parkinson's Law (extreme)**: Model reads files in tiny 20-line windows for 26+ turns, never calls `edit_file`
2. **Tool call parser mismatch**: Initially used `qwen3_xml` parser — model generates `<tool_call>` XML but parser didn't extract them. Fixed by switching to `hermes` parser.
3. **JSON errors**: ~10% of tool calls have malformed arguments (HTTP 400)
4. **Context overflow**: 8B model with 32K context overflows on complex issues (`Can not write request body`)

### Assessment
- LAION's fine-tune used only 31.6K of 155K trajectories on Qwen3-8B (not 32B)
- Community training recipe likely differs from Together AI's reference
- **Does NOT invalidate CoderForge dataset** — Phase 0 designed to test community weights, not the dataset itself

## Phase 1: Training — Qwen3.5-122B-A10B LoRA (2026-03-28)

### Why LoRA (not full fine-tune)
Full fine-tuning of 122B MoE with FSDP1 **OOMs on 8x B200** (178GB/GPU):
1. FSDP1 gathers full decoder layer (including all 256 experts) during forward/backward
2. 256-expert MoE layers are ~2.5B params each — too large to unshard to single GPU
3. Built-in `load_balancing_loss_func` allocates `(seq_len × num_layers, 256)` float tensors → 11GB OOM
4. Even at 4K context, backward pass uses 175GB/178GB per GPU
5. Qwen3.5's hybrid attention (Gated DeltaNet + full attention) adds memory overhead

**References**: Unsloth docs say 256GB VRAM for bf16 LoRA on 122B. NeMo uses FSDP2 + Expert Parallelism (EP) for full fine-tune.

### Current Config
- **Method**: LoRA r=64, α=128, dropout=0.05
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Trainable params**: 66.8M / 122.2B (0.055%)
- **Model distribution**: device_map="auto" (pipeline parallel across 8 GPUs)
- **Seq length**: 8K tokens
- **Batch**: 4 per device × 8 grad accum = 32 effective
- **LR**: 1e-5, cosine, warmup 3%
- **Steps**: 4,849 (155K examples / batch 32)
- **Router layers**: Frozen (stability per Unsloth recommendation)

### Training Status
- Step 14/4,849 at ~254s/step
- GPU memory: 44-148GB/GPU (fits comfortably)
- Est. time: ~14.2 days at 254s/step
- Est. cost: ~$8,700 at $25.60/hr spot

### Issues Resolved During Setup
1. **PyTorch cu126 → cu128**: B200 sm_100 requires cu128+
2. **FSDP config YAML → JSON**: transformers 5.x breaking change
3. **`use_cache` kwarg**: Moved to `model.config.use_cache = False`
4. **`tokenizer=` → `processing_class=`**: transformers 5.x deprecation
5. **`Qwen3_5MoeVisionBlock` FSDP error**: Used `Qwen3_5MoeForCausalLM` directly
6. **NCCL OFI double-free on B200**: Renamed `/opt/amazon/ofi-nccl/lib64/libnccl-net*.so`
7. **wandb not configured**: Set `WANDB_DISABLED=true`, `report_to=none`
8. **Chat template "No user query"**: Added `user` message check + try/except in label loop
9. **Full fine-tune OOM**: Switched to LoRA + device_map="auto"
10. **32K seq len OOM**: 248K vocab × 32K = 30.6GB logits tensor → reduced to 8K

### Speed Concern
254s/step pipeline parallel is slow (only 1 GPU active at a time). Options:
- **Accept**: 14 days, $8,700 — over spec break-even ($2,200-3,500)
- **Switch to Qwen3-32B dense**: Fits full fine-tune on 8x B200 with FSDP, much faster
- **Install DeepSpeed ZeRO-3**: Better CPU offloading for MoE, enables higher batch
- **Use Unsloth**: Claims 12x faster MoE training, 35% less VRAM

### Key Decision: No Qwen3.5-32B Dense
Qwen3.5 family has no 32B dense variant. Options:
- **Qwen3.5-122B-A10B MoE** (current) — LoRA running, slow
- **Qwen3.5-27B** (dense alternative) — would fit full fine-tune
- **Qwen3-32B** (CoderForge reference base) — exact base Together AI used, ungated

## Phase 1b: Unsloth MoE Training (2026-03-28)

### v1: Loss Collapse (FAILED)
Unsloth confirmed **14.2x faster** than vanilla LoRA (17.9s/step vs 254s/step).
However, loss collapsed to 8e-5 by step 500 — model memorized training data.

**Root causes** (MoE-specific):
1. **No input masking**: Trained on ALL tokens (system, user, assistant). Model memorized conversations.
2. **Too many trainable params**: r=64 with MoE expert layers → 7.3B trainable (5.63%). Unsloth targets `mlp.experts.gate_up_proj` + `mlp.experts.down_proj` across all 256 experts.
3. **High LR for MoE**: 1e-5 destabilizes the router → "lazy routing" to few experts.
4. **No auxiliary loss**: Load balancing disabled → router collapse.

| Step | Loss | Grad Norm | LR |
|------|------|-----------|-----|
| 10 | 0.690 | 2.57 | 6e-7 |
| 50 | 0.414 | 0.94 | 3.3e-6 |
| 100 | 0.030 | 0.14 | 6.6e-6 |
| 200 | 0.002 | 0.02 | 1e-5 |
| 500 | 0.00008 | 0.001 | 1e-5 |

### v2: MoE-Stabilized Training (IN PROGRESS)
All mitigations applied per Unsloth docs + MoE best practices:

| Fix | v1 | v2 |
|-----|----|----|
| Loss masking | All tokens | Assistant-only (custom collator) |
| LoRA rank | r=64, α=128 | r=16, α=16 (Unsloth recommended) |
| Learning rate | 1e-5 | 2e-6 |
| Router aux loss | Disabled | Enabled (coef=0.01) |
| Optimizer | AdamW | AdamW 8-bit |
| S3 checkpoint sync | Broken (no creds) | Working (IAM profile attached) |
| Output dir | coderforge-unsloth-output/ | coderforge-unsloth-output-v2/ |

### Infrastructure Fixes
11. **Root disk full (200GB)**: HF cache filled root → symlinked `~/.cache/huggingface` → `/mnt/nvme/hf-cache/`
12. **S3 credentials missing**: No IAM profile on B200 → attached `ecr-staging-ec2-profile` with S3 write policy
13. **TRL 0.24 collator crash**: Extra columns in dataset → stripped all columns except tokenized outputs
14. **Unsloth returns Processor not Tokenizer**: `Qwen3VLProcessor` wraps tokenizer → unwrap via `.tokenizer`
15. **MoE loss collapse**: Multiple mitigations applied (see v2 table above)
16. **Spot reclaim**: B200 instance terminated before v2 reached checkpoint 500. No S3 uploads (v1 had no creds, v2 too early). Need new spot instance.

### Spot Reclaim (2026-03-29)
B200 spot instance `i-06af825a7d467dda7` terminated during v2c training iteration.
- v1 checkpoint-500 (collapsed loss) was on local NVMe only — never uploaded (no AWS creds)
- IAM profile was attached minutes before reclaim but no v2 checkpoint reached save_steps=500
- **Lesson**: Save more frequently (save_steps=100 for first 1000 steps) and verify S3 sync on first save

### Ready to Relaunch (v2 fast path)
Script `train_unsloth.py` is finalized with all MoE stability mitigations:
- r=16, α=16, LR=2e-6, adamw_8bit, router aux loss
- Unsloth fast path via SFTTrainer + dataset_text_field="text"
- S3 checkpoint sync (IAM profile required)
- LossCollapseDetector callback alerts if loss < 0.01 before step 500
- Expected: ~18s/step, 9,690 steps, ~48 hrs, ~$1,200
