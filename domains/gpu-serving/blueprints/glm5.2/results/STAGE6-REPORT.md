# GLM-5.2-FP8 on B200 — Stage 6 Benchmark Report

**Model**: `zai-org/GLM-5.2-FP8` (glm_moe_dsa, 753B MoE / ~40B active, MLA+DSA, native MTP, 1M ctx)
**Hardware**: p6-b200.48xlarge (8× B200 sm_100 NVSwitch), us-east-2, cluster qwen3-next-bench-eks-cluster
**Engine**: SGLang v0.5.13.post1-cu130, TP8 (forced — 753GB FP8 doesn't fit TP4 on 180GB B200)
**Date**: 2026-06-23/24

---

## Executive summary
GLM-5.2-FP8 serves well on a single 8-GPU B200 at TP8. The dominant serving lever is **prefix caching**
(+73% throughput, 8.5× TTFT on the coding-agent workload). It is a **capable coding agent** (89% fix rate
via OpenCode on SWE-bench Lite, just under frontier Claude tiers). Two notable negative findings: **NVFP4
regresses batched throughput vs FP8** on this model, and **torch.compile is incompatible** with the
glm_moe_dsa path on SGLang 0.5.13.

## Tier stack (coding-agent, 12K shared prompt, c=64 unless noted)
| Tier | Config | agg tok/s | TTFT p99 | verdict |
|------|--------|-----------|----------|---------|
| T0 | FP8 floor, no cache | 1708 | 29.7s | reference (BF16 doesn't fit) |
| T1 | +fp8 KV | 3004 | 3.33s | best high-throughput |
| T2 | +prefix cache | 2950 | 3.50s | **dominant lever: +73%/8.5×TTFT** (92% hit) |
| T3 | +EAGLE/MTP | 2331 (c64); **139 @ c1 (+40%)** | — | latency lever: +40% @ c1, −22% @ c64 |
| T5 | +torch.compile | — | — | **BLOCKED** (cuda graph capture fails on glm_moe_dsa) |

CUDA graphs run default-ON in all tiers. MoE tile tuning N/A (flashinfer_trtllm backend, vendor-precompiled).

## Quantization: NVFP4 vs FP8 (T1) — community lukealonso/GLM-5.2-NVFP4, full 753B, quality-gate PASSED
| workload | FP8 | NVFP4 | delta |
|----------|-----|-------|-------|
| coding c1 | 99 | 114 | +15% (NVFP4 wins single-stream) |
| coding c64 | 3004 | 1219 | **−59%** |
| 16k c64 | 2200 | 1624 | −26% |
| 31k c32 | 831 | 779 | −6% |
**NVFP4 REGRESSES batched throughput 55-60% vs FP8** (gap narrows with context). NVFP4's win is
weight-bandwidth (single-stream/VRAM: 56 vs 94 GB/GPU), but batched decode is compute-bound on the NVFP4
trtllm MoE kernel here. **Use FP8 for throughput; NVFP4 only for c=1 latency or VRAM-constrained.** Not a
quality issue (gate passed: fixed bug correctly, tool-calls work) nor a backend fallback (same trtllm runner).

## Long-context (MNBT=8192)
| ctx | c=32 | c=64 | c=128 | SLO knee (15s TTFT) |
|-----|------|------|-------|---------------------|
| 16k | 1645 tok/s, 5.5s | 2200, 11.0s | 2865, 21.9s | ~c64-80 |
| 31k | 831, 11.0s | 986, 22.3s | 1139, 44.6s | ~c32-48 |
- **Regime: PREFILL-COMPUTE-BOUND** (throughput plateaus, TTFT explodes, 0 errors/OOM to c=256). NOT
  KV-capacity-bound.
- **MNBT sub-sweep**: 8192 best at the SLO knee (5.5s vs 7.9s default @ 16-conc); 32768 trades knee-TTFT
  for high-conc throughput. fp8 KV + HiCache don't help (compute-bound, not capacity-bound).

## B300 verdict
**B300 would NOT relieve the long-context knee** — it's prefill-compute-bound, and B300's advantage is KV
capacity. B300 IS worth testing only to unlock the **TP4+DP2 layout** (753GB FP8 fits at 275GB/GPU but not
180GB) for the coding-agent throughput regime — that's the layout that won +19-25% on kimi-k2.6-nvfp4.

## Kimi comparison (16k) — NOT apples-to-apples
GLM-5.2-FP8/B200-TP8 ~2.3× lower throughput at 16k c64 than Kimi-NVFP4/B300-TP4+DP2 (2200 vs 5161 tok/s),
but 3 confounded variables (hardware B200-vs-B300, model 753B-FP8-vs-1T-NVFP4, layout TP8-vs-TP4+DP2). The
forced-TP8 on B200 is a real structural disadvantage. A B300 TP4+DP2 run would make it fair.

## Agent capability — SWE-bench Lite (46 issues, OpenCode harness)
- **Fix rate 89%** (41/46) — capable agent, just under frontier Claude tiers (100% edit-attempt on same issues).
- Gold pass 13% (lower bound — in-pod gold-eval test-env-limited; needs full Docker harness for clean number).
- Median 404K tokens/issue (reasoning-first). Codex + Claude Code arms blocked by LiteLLM thinking-block
  translation (OpenCode-only comparison, which is the cleanest match to the baseline traces).

## GLM-5.2 operational traits (for the deployment card)
- **Reasoning-first**: emits reasoning_content before content/tool_calls. NEVER use small max_tokens
  (tight budget starves the answer → empty output). Harnesses need >=2048; ideally serve at 131072 ctx.
- SGLang v0.5.13.post1-cu130 supports GlmMoeDsaForCausalLM. Cold start ~8 min (DeepGEMM JIT + graph capture);
  fp8-KV configs ~10 min. attention_backend='dsa' (DSA, not classic MLA — FLASHINFER_MLA N/A).
- Parsers: --tool-call-parser glm47 --reasoning-parser glm45 (both work). DCGM PROF unavailable (driver-580).

## Production recommendation
- **Throughput**: T1 (FP8, prefix cache + fp8 KV), TP8, MNBT default → 3004 tok/s @ c64, TTFT 3.33s.
- **Low-latency**: add EAGLE/MTP (+40% @ c1).
- **Long-context**: MNBT=8192; SLO-safe to ~c64 (16k) / ~c32 (31k); scale out, not up (compute-bound).
- **Avoid**: NVFP4 (batched regression), HiCache (compute-bound, −12%), torch.compile (blocked).

## Deferred / follow-up
- vLLM/MTP engine arm (SGLang well-characterized; vLLM head-to-head not run).
- B300 TP4+DP2 layout (unlocks the kimi-comparable config).
- Codex + Claude Code agent arms (need a LiteLLM thinking-block-stripping adapter).
- Full Docker gold-eval for a clean SWE-bench pass-rate number.
