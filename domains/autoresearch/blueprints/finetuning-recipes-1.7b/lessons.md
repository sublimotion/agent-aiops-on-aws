# Fine-Tuning Recipes 1.7B — Lessons Learned

## Run 2: 48 Experiments on g5.xlarge A10G (2026-04-01/02)

### Environment
- **Hardware**: g5.xlarge, 1x NVIDIA A10G (24 GB GDDR6, sm_86)
- **Software**: PyTorch 2.11.0+cu130, Unsloth 2026.3.18, Transformers 5.3.0, TRL, PEFT
- **Base Model**: Qwen3-1.7B (1.74B params, 1.00% trainable with LoRA r=32 = 17.4M params)
- **Dataset**: dair-ai/emotion (20K samples, 6 classes: joy/sadness/anger/fear/love/surprise)
- **Eval**: Macro F1 on full 2K validation set, logit-based classification (forward pass + argmax)
- **Time per experiment**: 13 min (1 epoch) to 50 min (5 epochs)
- **Total compute**: ~26 hours, ~$26

### Results Summary

| Metric | Run 1 (0.6B) | Run 2 (1.7B) | Delta |
|--------|-------------|-------------|-------|
| Best macro F1 | 0.883 | **0.895** | +1.2pp |
| Experiments | 34 | 48 | +14 |
| Eval set size | 300 | 2,000 | +5.7x |
| Best LR | 2e-4 | 5e-4 | higher |
| Best rank | 16 | 32 | higher |
| Best epochs | 3 | 5 | more |
| Time/experiment | ~70 min | ~30 min | 2.3x faster |
| Hardware | g7e.24xlarge | g5.xlarge | 8x cheaper |

### What Matters (Ranked by Impact)

1. **Epochs** (largest effect, +0.15-0.20 F1): 5 > 3 >> 1. Every 3-epoch experiment beats its 1-epoch counterpart by 0.10-0.15. Going from 3 to 5 adds another +0.009. This was the single biggest lever.

2. **Learning rate** (+0.02-0.05 F1): 5e-4 > 2e-4 > 5e-5. Higher LR consistently better for this task/model. The 5e-4 tier won the grid sweep.

3. **LoRA rank** (+0.005-0.01 F1): rank=32 > 16 > 8, but diminishing returns. rank=64 was slightly worse (0.881 vs 0.895). The sweet spot is 32 for 1.7B.

4. **Optimizer** (+0.005 F1): adamw_torch slightly beats adamw_8bit. paged_adamw_8bit is consistently worse (-0.01 to -0.02).

5. **Everything else** (<0.005 F1): Schedule type, weight decay, warmup, batch size, seq length, dropout, grad accumulation — none of these meaningfully moved the needle. Cosine is slightly better than linear/constant but within noise.

### What Doesn't Matter

- **Sequence length**: 256 vs 512 vs 768 — no significant difference (emotion texts are short)
- **Batch size**: 8, 16, 32 all similar (at same effective batch with grad_accum)
- **Weight decay**: 0.0, 0.01, 0.05, 0.1 all within noise
- **Warmup**: 0.0, 0.03, 0.05, 0.1 all within noise
- **Dropout**: 0.0, 0.05, 0.1 all within noise
- **Alpha/rank ratio**: Varies from 0.5 to 4x with minimal effect once rank is set

### Critical Bug: set_seed() Resets Mutation RNG

SFTTrainer's `seed=42` config calls `transformers.set_seed(42)` which resets ALL random state including Python's `random` module. If you use `random.sample()` or `random.choice()` for mutation, you'll get the same mutation every single time.

**Fix**: Use a separate `random.Random()` instance initialized with `os.urandom()`:
```python
_mutation_rng = random.Random(struct.unpack("I", os.urandom(4))[0])
# Use _mutation_rng.sample() and _mutation_rng.choice() instead of random.*
```

This cost ~3 wasted experiments before detection.

### Grid Sweep Was Worth It

Run 1's lesson was "random mutation is too narrow" — only 2/13 params explored. The grid sweep in Run 2 systematically covered the top-3 parameters (lr, rank, epochs) in 18 experiments, establishing that 5e-4/32/3ep was optimal before mutations began. Without this, the mutations would have started from a suboptimal base.

### QLoRA Ceiling on Emotion Classification

The F1 plateaued at 0.895 despite 30 diverse mutations. The ceiling appears to be a property of QLoRA on this task/model, not a search failure. Evidence:
- 30 mutations explored all 12 hyperparameter dimensions
- Multiple near-ties (0.8948, 0.8953, 0.8954) from very different configs
- Higher rank (64), more epochs (5), different optimizers — none broke through

To reach 0.932, likely need: full fine-tuning (not LoRA), larger model (4B+), or different quantization (FP16 instead of 4-bit).

### Logit-Based Eval Works Better Than Generation

Unsloth's patched attention has a shape mismatch bug during `model.generate()` on A10G (sm_86). Instead of generating text, we evaluate by:
1. Forward pass on the prompt
2. Extract logits at the last position
3. Compare logit scores for each label token ("sadness", "joy", etc.)
4. Pick highest logit as prediction

This is also ~3x faster than generation (no autoregressive decoding) and deterministic (no sampling noise).

### g5.xlarge Is Sufficient for 1.7B QLoRA

Peak memory: 12.6 GB (with memory leak) out of 24 GB available. The A10G has more than enough headroom. At $1.01/hr vs g7e.24xlarge at ~$8/hr, this saves 87% on compute cost. The A10G's lower memory bandwidth doesn't matter for QLoRA training.

### Memory Leak Pattern

GPU memory grows ~0.6 GB per experiment (2.1 GB → 12.6 GB after 18 experiments). This is the same Unsloth/PEFT bug from Run 1. The process restart after the grid sweep (bug fix deployment) incidentally reclaimed all leaked memory (reset to 2.5 GB). Recommend: restart every 20 experiments or use subprocess isolation.

### Benchmark Comparison (Updated)

| Model | Params | Method | Macro F1 |
|-------|--------|--------|----------|
| CJT Ensemble (BERT+RoBERTa+DistilBERT) | ~300M | Full FT + Jury Voting | 0.937 (SOTA) |
| Qwen-1.8B | 1.8B | LoRA (r=8) | 0.932 |
| Falcon-7B | 7.0B | LoRA (r=8) | 0.915 |
| Phi-2 | 2.7B | LoRA (r=8) | 0.913 |
| **Qwen3-1.7B (Ours, Run 2)** | **1.7B** | **QLoRA (r=32, Unsloth)** | **0.895** |
| Qwen3-0.6B (Run 1) | 0.6B | QLoRA (r=16, Unsloth) | 0.883 |
| BERT-base | 110M | Full Fine-Tune | 0.882 |
| DistilBERT-base | 66M | Full Fine-Tune | 0.883 |
| Mistral-7B | 7.1B | LoRA (r=8) | 0.880 |

**Key**: 3x model size (0.6B→1.7B) only gained +1.2pp (0.883→0.895). The Qwen-1.8B benchmark at 0.932 likely uses full-precision LoRA (not 4-bit quantized), which could explain the 3.7pp gap. QLoRA's 4-bit quantization may be the bottleneck, not model size.

### Next Steps (if continuing)

1. **Full-precision LoRA** (not QLoRA): Load model in FP16/BF16 instead of 4-bit. Needs ~7 GB for 1.7B — fits A10G.
2. **Qwen3-4B**: 4B model with QLoRA. Needs ~16-20 GB — still fits A10G.
3. **Longer training**: 7-10 epochs with the best config (diminishing returns expected).
4. **Different task**: Try a harder classification task where model capacity matters more.
