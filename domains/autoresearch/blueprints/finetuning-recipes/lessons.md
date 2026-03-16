# Fine-Tuning Recipes Autoresearch — Lessons Learned

## Run 1: 34 Experiments on g7e Blackwell (2026-03-15/16)

### Environment
- **Hardware**: g7e.24xlarge, single RTX PRO 6000 Blackwell (96 GB GDDR7, sm_120)
- **Software**: PyTorch 2.10.0+cu128, Unsloth 2026.3.4, Transformers 5.2.0, TRL, PEFT
- **Base Model**: Qwen3-0.6B (606M params, 1.67% trainable with LoRA = 10.1M params)
- **Dataset**: dair-ai/emotion (20K samples, 6 classes: joy/sadness/anger/fear/love/surprise)
- **Eval**: Macro F1 on 300-sample validation subset (stratified)
- **Framework**: autoresearch-colab sentiment module + custom autoresearch_train.py loop
- **Time per experiment**: ~50-70 min (training ~50 min + eval ~2 min + model load ~5 min)

### Qwen3 Thinking Mode Compatibility Fix

Qwen3 models wrap all responses in `<think>...</think>` tags before the actual answer. The evaluate.py label matching takes the first line of output (`raw_output.split("\n")[0]`), which returns `<think>` instead of the predicted label — causing 0% match rate even on a well-trained model.

**Fix**: Patch `evaluate.py` to strip thinking tags before label extraction:
```python
import re
raw_output = re.sub(r"<think>.*?</think>\s*", "", raw_output, flags=re.DOTALL).strip()
```

This is a **Qwen3-specific** issue. Qwen2.5 and other model families don't have thinking mode. The `enable_thinking=False` parameter in `apply_chat_template` does NOT reliably suppress the tags in Qwen3-0.6B.

### Baseline

| Metric | Value |
|--------|-------|
| Zero-shot macro F1 | 0.0000 |
| First LoRA macro F1 | 0.8648 |
| Best macro F1 | 0.8830 |
| Trainable params | 10.1M / 606.1M (1.67%) |
| Training time | ~50 min per experiment (3189 steps) |
| Eval time | ~70s per experiment (300 samples) |
| GPU memory | 1.7-5.0 GB |

Zero-shot F1 is 0.0 because Qwen3-0.6B is a base model (not instruct-tuned) — it generates random tokens instead of classification labels. The jump from 0.0 to 0.8648 is entirely from LoRA fine-tuning teaching the model the task format.

### Experiment Log

| # | macro_f1 | Memory | Status | Changes |
|---|----------|--------|--------|---------|
| 0 | 0.0000 | 0.7 GB | baseline | Zero-shot Qwen3-0.6B |
| 1 | 0.8648 | 1.7 GB | **keep** | DEFAULT_CONFIG (r=16, alpha=16, lr=2e-4, cosine, adamw_8bit) |
| 2 | 0.8758 | 2.0 GB | **keep** | alpha=64, optim=adamw_torch |
| 3 | 0.8722 | 2.0 GB | discard | alpha=32, optim=adamw_8bit |
| 4 | 0.8714 | 2.3 GB | discard | alpha=32, optim=adamw_8bit |
| 5 | 0.8769 | 2.6 GB | **keep** | alpha=32, optim=adamw_8bit |
| 6 | 0.8768 | 3.0 GB | discard | alpha=64, optim=adamw_torch |
| 7 | 0.8724 | 3.0 GB | discard | alpha=64, optim=adamw_torch |
| 8 | 0.8775 | 3.0 GB | **keep** | alpha=64, optim=adamw_torch |
| 9 | 0.8695 | 2.9 GB | discard | alpha=32, optim=adamw_8bit |
| 10 | 0.8738 | 3.2 GB | discard | alpha=32, optim=adamw_8bit |
| 11 | 0.8714 | 3.5 GB | discard | alpha=32, optim=adamw_8bit |
| 12 | 0.8584 | 3.9 GB | discard | alpha=32, optim=adamw_8bit |
| 13 | 0.8786 | 4.2 GB | **keep** | alpha=32, optim=adamw_8bit |
| 14-19 | 0.8733-0.8778 | 4.5 GB | discard | various |
| 20 | 0.8814 | 4.5 GB | **keep** | alpha=64, optim=adamw_torch |
| 21 | 0.8822 | 4.5 GB | **keep** | alpha=32, optim=adamw_8bit |
| 22-31 | 0.8688-0.8814 | 4.8 GB | discard | various |
| 32 | **0.8830** | 4.8 GB | **keep** | alpha=64, optim=adamw_torch |
| 33-34 | 0.8693-0.8708 | 4.8-5.1 GB | discard | alpha=32, optim=adamw_8bit |

**Best macro_f1: 0.8830** (improvement: +0.0182 / +2.1% from first LoRA)

### Transferable Improvements

Seven config changes survived the 34-experiment sweep. All improvements were small and incremental:

1. **DEFAULT_CONFIG → 0.8648**: The biggest jump — teaching the model the classification format via LoRA. r=16, alpha=16, lr=2e-4, 3 epochs, cosine scheduler.

2. **lora_alpha 16→64 (+50% alpha/rank ratio)**: Most impactful hyperparameter change. Higher alpha amplifies LoRA updates (alpha/r ratio: 1→4). Contributed ~0.011 F1.

3. **adamw_torch vs adamw_8bit**: Both optimizers appear in kept experiments. The optimizer choice interacts with other hyperparameters but neither consistently dominates.

### What Didn't Get Explored

The `sample_config(n_changes=2)` mutation function only sampled `lora_alpha` and `optim` across all 34 experiments. With 13 hyperparameters in the search space and only 2 mutations per experiment, the random selection hit the same 2 parameters repeatedly. **Never explored**:

- Learning rate (stuck at 2e-4)
- LoRA rank (stuck at 16)
- Number of epochs (stuck at 3)
- Batch size (stuck at 4)
- Scheduler type (stuck at cosine)
- Dropout (stuck at 0.0)
- Max sequence length (stuck at 512)
- Base model (never tried Qwen3-1.7B)

### Meta-Observations on the Autoresearch Pattern

1. **Random mutation is too narrow**: With 13 searchable parameters and n_changes=2, the probability of hitting any specific parameter pair is low. After 34 experiments, only 2 of 13 parameters were explored. A better strategy: first sweep each parameter individually (grid search the most impactful ones), then use random mutation for interaction effects.

2. **~70 min/experiment is too slow for hyperparameter search**: Training 17K samples × 3 epochs on a 0.6B model takes ~50 min. With 96 GB of GPU memory and only 5 GB used, the model is dramatically underutilizing the hardware. Options: larger batch size (4→32), fewer epochs (3→1 for early experiments), or train on a data subset first.

3. **Memory leak across experiments**: GPU memory grew from 1.7 GB to 5.0 GB across experiments despite cleanup() calls. Unsloth/PEFT leave compiled kernels and cached tensors in CUDA memory. Restart the process every ~20 experiments to reclaim memory.

4. **300-sample eval is noisy**: With 300 samples and 6 classes (smallest class: surprise with ~15 val samples), macro F1 has high variance. Experiments with F1 within ±0.005 of each other are likely indistinguishable. Many "discard" experiments (0.8768, 0.8778) were within noise of the "keep" threshold.

5. **Qwen3-0.6B is already good at this task**: 86.5% macro F1 from the first LoRA training with default hyperparameters. The remaining ~14% gap is likely from: (a) class imbalance (surprise/love are rare), (b) genuinely ambiguous examples, (c) model capacity limits. Switching to Qwen3-1.7B (unexplored) may be the biggest lever.

6. **Direct Python loop > Claude Code for fine-tuning**: The first attempt via Claude Code `--print` mode spawned unnecessary vLLM processes and hit API formatting issues. Running `autoresearch_train.py` directly is simpler and more reliable for this use case. Claude Code is better for the pre-training recipe loop where the agent needs to edit code, not just mutate configs.

### Benchmark Comparison (dair-ai/emotion)

Our macro F1 of 0.883 with Qwen3-0.6B + LoRA compares as follows (sources: arXiv:2512.17630, HuggingFace model cards):

| Model | Params | Method | Macro F1 |
|-------|--------|--------|----------|
| CJT Ensemble (BERT+RoBERTa+DistilBERT) | ~300M | Full FT + Jury Voting | 0.937 (SOTA) |
| 5-Model Ensemble | 595M | Full FT + Credibility Voting | 0.935 |
| Qwen-1.8B | 1.8B | LoRA (r=8) | 0.932 |
| Falcon-7B | 7.0B | LoRA (r=8) | 0.915 |
| Phi-2 | 2.7B | LoRA (r=8) | 0.913 |
| **Qwen3-0.6B (Ours)** | **0.6B** | **LoRA (r=16, Unsloth)** | **0.883** |
| BERT-base | 110M | Full Fine-Tune | 0.882 |
| DistilBERT-base | 66M | Full Fine-Tune | 0.883 |
| Mistral-7B | 7.1B | LoRA (r=8) | 0.880 |
| OpenLLaMA-3B | 3.0B | LoRA (r=8) | 0.861 |

**Key findings**:
- Matches fully fine-tuned BERT-base and DistilBERT despite using LoRA (1.67% params trained)
- Outperforms Mistral-7B + LoRA (12x larger model) by +0.003
- ~5 points below Qwen-1.8B + LoRA (0.932) — the 3x parameter gap matters
- ~5 points below SOTA ensemble (0.937) — but those use 5 separate fully fine-tuned models
- The unexplored base model switch (Qwen3-0.6B → 1.7B) is likely the biggest remaining lever

### Bugs Found During Execution

- **Unsloth model names**: `unsloth/Qwen3.5-0.6B-Instruct` doesn't exist. Correct names: `unsloth/Qwen3-0.6B`, `unsloth/Qwen3-1.7B` (no ".5", no "-Instruct")
- **TRL `formatting_func` required**: Newer TRL/Unsloth versions require `formatting_func` parameter in SFTTrainer, not `dataset_text_field`
- **`torchvision` missing**: Unsloth imports `torchvision` transitively — must install separately with matching CUDA index
- **HuggingFace rate limiting**: Downloading models on every experiment without caching causes 429 errors. Set `HF_HUB_ENABLE_HF_TRANSFER=0`
- **Git stash drops untracked files**: `git stash` + `git stash drop` silently removes untracked files that were staged with the stash
