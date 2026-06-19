---
name: cherry-pick-eval
description: Evaluate upstream PRs or commits individually against the K2.6 baseline. Tests each change in isolation using the same L0-L4 pipeline, identifying which are helpful, harmful, or neutral for our specific workload. Use when new upstream PRs land (Alpha-MoE, Mega MoE, FP8 fusion) or when evaluating a bundle of commits. Never assume a bundle is net-positive — the EFA harness found bundled expert commits regressed LL performance by 10%.
---

# Cherry-Pick Evaluation

## Overview

Evaluates upstream PRs, commits, or patch bundles **individually** against the K2.6 serving baseline. Inspired by the EFA harness finding: 12 hand-crafted expert commits applied together regressed low-latency performance by 10%, even though 6 of the 12 were individually helpful (−15% to −25% each).

The principle: **commit-level granularity matters**. A bundle of optimizations can be net-negative if harmful commits outweigh helpful ones for a specific workload profile.

## Usage

### Input

1. **PR/commit list**: GitHub PR numbers or commit SHAs to evaluate
2. **Target workload**: Which K2.6 workload profile to measure against (W1-W6, or custom)
3. **Baseline**: Current champion configuration (default: vLLM FLASHINFER_MLA)

### Procedure

For each PR/commit in the list:

1. **Isolate**: Apply ONLY this single change to the baseline image
2. **Build**: Compile with the change (record build time, any warnings)
3. **Verify**: Run L2 correctness check (catch accuracy regressions early)
4. **Benchmark**: Run L4 e2e benchmark at c=128 and c=512
5. **Classify**:
   - **Helpful**: Speedup CI excludes 1.0 (improvement confirmed)
   - **Harmful**: Slowdown CI excludes 1.0 (regression confirmed)
   - **Neutral**: CI includes 1.0 (no measurable effect)
6. **Record**: Classification + telemetry for each change

### Output

```json
{
  "evaluation_id": "cherry_pick_001",
  "baseline": "vllm_v0.19.1_flashinfer_mla",
  "workload": "W3_agentic_multiturn",
  "commits_evaluated": [
    {
      "ref": "vllm#35474",
      "title": "Dynamic MLA/MHA Routing for Sub-1024 Token Prefill",
      "verdict": "helpful",
      "throughput_delta": "+8.3%",
      "ttft_delta": "-62%",
      "ci": [1.04, 1.13],
      "notes": "Significant TTFT improvement on short agentic prompts"
    },
    {
      "ref": "vllm#30078",
      "title": "Alpha MoE integration",
      "verdict": "helpful",
      "throughput_delta": "+11.2%",
      "ci": [1.07, 1.16],
      "notes": "16% kernel speedup translates to 11% e2e on K2.6 384-expert"
    },
    {
      "ref": "vllm#36297",
      "title": "Fused BMM+FP8 quant for MLA v_up_proj",
      "verdict": "neutral",
      "throughput_delta": "+0.4%",
      "ci": [0.98, 1.03],
      "notes": "Kernel-level 1.7x but <1% e2e — v_up_proj is not the bottleneck"
    }
  ],
  "composition_recommendation": "Apply #35474 + #30078. Skip #36297 (neutral). Test composition for interactions."
}
```

### Composition Testing

After individual evaluation, test **pairwise and full compositions** of helpful commits:
- Do two helpful changes interact positively (super-linear) or negatively (contention)?
- The EFA finding: helpful changes can conflict when combined

### Constraint Database Integration

- **Helpful PRs**: Extract the optimization idea, inject as positive constraint into `/generate-candidate`
- **Harmful PRs**: Record as soft constraint ("this approach regresses K2.6 because...")
- **Neutral PRs**: Record with evidence — prevents re-evaluation of the same approach

### Priority PRs to Evaluate (Phase 1)

| PR | What | Why Test on K2.6 |
|----|------|-----------------|
| vLLM #35474 | Dynamic MLA/MHA routing (<1024 tokens) | ~3x TTFT for agentic workloads |
| vLLM #30078 | Alpha-MoE (16% over DeepGEMM on DS V3) | Unknown on 384-expert layout |
| SGLang #23167 | DeepGEMM Mega MoE integration | Tuned for 256 experts, test on 384 |
| vLLM #35879 | RoPE+KV cache fusion (+16% TPOT) | Tested on H100, not B300 |
| vLLM #36297 | Fused BMM+FP8 for MLA decode | 1.7x kernel-level, unknown e2e |

### When NOT to Use

- For agent-generated kernels (use `/verify-kernel` instead)
- When evaluating ideas from PRs (inject as constraints, don't cherry-pick code)
- For PRs that require major refactoring to apply (out of scope for A/B testing)
