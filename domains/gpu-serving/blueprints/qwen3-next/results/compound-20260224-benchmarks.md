# Compound Learning: Qwen3-Next Benchmark Session
**Date**: 2026-02-24
**Session**: P0-P1 benchmark execution on p5en.48xlarge (8x H200)

## New Cross-Cutting Rules Elevated

### Rule 1: Budget for JIT compilation startup time
**Source**: Lesson #5 (SGLang DeepGEMM 640s startup)
**Elevated to**: `.claude/steering/tech-stack.md` → "Budget for JIT compilation startup time on first-run serving stacks"
**Generalization**: Applies beyond SGLang — any serving framework with JIT compilation (TensorRT-LLM, DeepGEMM, CUDA graph capture) will have first-run penalties. Critical for capacity block planning.

### Rule 2: TP > DP+EP for MoE at single-node scale
**Source**: Lesson #7 (DP+EP underperforms TP=4)
**Elevated to**: `.claude/steering/tech-stack.md` → "For MoE models, favor tensor parallelism over data parallelism with expert parallelism at single-node scale"
**Generalization**: The expert parallelism routing overhead dominates at single-node scale. This likely applies to any high-expert-count MoE model (Mixtral, DeepSeek, Qwen-MoE family). The rule may not hold for multi-node deployments where TP cross-node communication is the bottleneck.

## Previously Elevated Rules (from earlier compound run)

1. Scale-to-0 before GPU config changes (Lesson #3)
2. Air-gapped tokenizer paths for benchmarking (Lesson #2)
3. Document benchmark execution location (new rule)
4. FP8 MoE TP compatibility check (Lesson #1)

## Key Benchmark Findings Summary

| Finding | Impact | Generalizability |
|---------|--------|-----------------|
| TP=4 is optimal for Qwen3-Next FP8 on p5en | High — production config decision | Model-specific (MoE dimensions) |
| DP+EP with 8 GPUs worse than TP=4 with 4 GPUs | High — saves 4 GPUs per replica | Likely generalizable to high-expert MoE |
| vLLM outperforms SGLang at all QPS levels | Medium — engine selection | Version-specific (vLLM 0.16 vs SGLang 0.5.9) |
| MTP blocked on vLLM 0.16 V1 engine | Medium — latency optimization blocked | Version-specific, fixed in 0.17+ |
| TPOT stable across 4K-64K context | High — predictable cost model | Likely generalizable for hybrid attention |
| Cost: ~$0.032/1M output tokens at QPS=4 | High — competitive pricing | Config-specific |

## Lessons Not Elevated (Blueprint-Specific)

- Lesson #4 (vLLM MTP warmup bug): Version-specific, will be fixed in vLLM 0.17+
- Lesson #6 (nerdctl -d/--rm): Already captured in general container runtime knowledge
- Lesson #2 (air-gapped tokenizer): Already elevated in prior compound run
