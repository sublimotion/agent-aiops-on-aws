# Training Recipes Autoresearch — Lessons Learned

## Run 1: 16 Experiments on g7e Blackwell (2026-03-14)

### Environment
- **Hardware**: g7e.24xlarge, single RTX PRO 6000 Blackwell (96 GB GDDR7, sm_120)
- **Software**: PyTorch 2.9.1+cu128, Python 3.10.20 (uv managed)
- **Model**: 50.3M params, 8 layers, 4 heads, 512 embed dim
- **Time budget**: 300s (5 min) per experiment
- **Agent**: Claude Sonnet 4.6 via Bedrock (--print mode, single response)
- **Codebase**: autoresearch-colab (Karpathy GPT-2 training recipes)

### Blackwell Compatibility Fix

FA3 (`kernels-community/flash-attn3`) has no sm_120 binary — crashes with `CUDA error: no kernel image available`. FA4 (`BACKEND: "FLASH"`) also fails (Triton compilation error on sm_120 in PyTorch 2.9.1).

**Fix**: Replace FA3 import with `_FA3Compat` wrapper using:
- `torch.nn.attention.flex_attention` (compiled Triton, supports sliding window via `create_block_mask`) for windowed attention
- `F.scaled_dot_product_attention` (SDPA) for full causal attention
- Layout conversion: FA3 `(B,T,H,D)` ↔ flex/SDPA `(B,H,T,D)` via transpose
- Block mask cache for repeated sequence lengths

This is a Blackwell-specific issue. Hopper (sm_90) uses FA3 from `varunneal/flash-attention-3` without issues.

### Baseline

| Metric | Value |
|--------|-------|
| val_bpb | 1.0803 |
| Steps | 340 |
| Throughput | 574K tok/sec |
| MFU | 13.9% |
| Total tokens | 178.3M |
| Final loss | 3.039 |
| Warmup time | ~26s (torch.compile JIT) |

### Experiment Log

| # | Hypothesis | Parameter | Before → After | val_bpb | Delta | Status |
|---|-----------|-----------|---------------|---------|-------|--------|
| 1 | LR warmup stabilizes early training | WARMUP_RATIO | 0.0 → 0.05 | ~1.080 | ~0.000 | REVERTED |
| 2 | Deeper model has more capacity | DEPTH | 8 → 10 | >1.080 | regression | REVERTED |
| 3 | Higher Muon LR for faster convergence | MATRIX_LR | 0.04 → 0.06 | ~1.076 | -0.004 | KEPT |
| 4 | Higher unembedding LR | UNEMBEDDING_LR | 0.004 → 0.008 | ~1.074 | -0.002 | KEPT |
| 5 | Higher embedding LR | EMBEDDING_LR | 0.6 → 0.7 | ~1.073 | -0.001 | KEPT |
| 6-15 | Various (LR sweeps, architecture, etc.) | — | — | ≥1.073 | regression/neutral | REVERTED |
| 16 | Lower weight decay | WEIGHT_DECAY | 0.2 → 0.15 | 1.0727 | +0.00002 | NEUTRAL |

**Best val_bpb: 1.0726** (improvement: -0.0077 / -0.71% from baseline)

### Transferable Improvements

Three hyperparameter changes survived the 16-experiment sweep:

1. **MATRIX_LR: 0.04 → 0.06 (+50%)** — Most impactful. The Muon optimizer's learning rate was too conservative for the 5-minute training window. Higher LR enables faster convergence within the fixed time budget.

2. **UNEMBEDDING_LR: 0.004 → 0.008 (2x)** — The lm_head projection was underfit. Doubling its LR improved output distribution learning.

3. **EMBEDDING_LR: 0.6 → 0.7 (+17%)** — Modest gain from slightly faster token embedding adaptation.

### What Didn't Work
- **Deeper model (8→10 layers)**: More parameters couldn't converge in 5 minutes — underfitting beats overfitting under time pressure
- **LR warmup**: Baseline already trains with no warmup; adding warmup wastes steps in the fixed budget
- **Weight decay changes**: 0.2 → 0.15 was neutral at this training duration

### Meta-Observations on the Autoresearch Pattern

1. **Sonnet explored conservatively**: All 16 experiments were hyperparameter sweeps — no architecture, multi-GPU, or torch.compile changes. The model stuck to what it knew was safe. Karpathy's Opus run discovered ~20 improvements across diverse categories in 118 experiments.

2. **--print mode limits loop depth**: Claude Code's `--print` flag returns a single response, capping at ~16 experiments before context limits. A wrapper script or interactive mode is needed for 100+ experiment runs.

3. **13.9% MFU is the real opportunity**: The model only uses 13.9% of the GPU's theoretical FLOPS. Multi-GPU (DDP on 4 GPUs), larger batch sizes, or torch.compile optimizations could 4-7x the throughput, enabling 4-7x more training steps per experiment.

4. **Blackwell compatibility is a real obstacle**: The FA3→flex_attention patch was necessary before any experiments could run. On Hopper, the repo works out of the box. This infrastructure gap consumed ~30 min of setup time.

5. **5-minute budget favors optimization over architecture**: At 340 steps per run, there's no room for architecture exploration (deeper/wider models underfit). All improvements came from making the existing architecture train faster within the budget.
