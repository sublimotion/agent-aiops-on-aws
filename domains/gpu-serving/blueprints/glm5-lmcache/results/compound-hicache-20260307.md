# Compound Summary — GLM-5 LMCache (HiCache Focus) — 2026-03-07

## Sources reviewed
- lessons.md entries: 15 total, 3 new since last compound run (lessons #13-#15 about HiCache)
- Readiness audits: 0 files
- Deployment logs: 0 files
- Benchmark reports: 3 files (baseline-20260306, extended-20260307, hicache-20260307)

## Benchmark signal
| Report date | Key findings | Notes |
|-------------|--------------|-------|
| 2026-03-06 | Baseline: 1,530 tok/s peak (32 conc, shared-prefix), RadixAttention effective | DeepGEMM JIT ~15 min warmup |
| 2026-03-07 | Extended baseline: 909 tok/s peak (64 conc, diverse prompts), LMCache blocked by NSA/MLA | Per-request 48-50 tok/s decode |
| 2026-03-07 | HiCache CPU offload: 2,602 tok/s peak (128 conc), 2.86x vs baseline at same concurrency | 800 GB host memory pool, zero errors |

## Elevated to steering
| Rule | Source (lesson) | Target file | Section |
|------|-----------------|-------------|---------|
| SGLang HiCache works with NSA/MLA attention where LMCache fails — use --enable-hierarchical-cache for MLA models | Lesson #13 | `.claude/steering/tech-stack.md` | GPU Serving Conventions → Deployment Conventions |
| HiCache --hicache-size must exceed device KV pool size to pass initialization assertion | Lesson #14 | `.claude/steering/tech-stack.md` | GPU Serving Conventions → Deployment Conventions |
| For memory-constrained models, CPU KV cache offloading fundamentally changes the concurrency ceiling | Lesson #15 | `.claude/steering/tech-stack.md` | GPU Serving Conventions → Deployment Conventions |

## Kept local
| Lesson (summary) | Source | Reason kept local |
|------------------|--------|-------------------|
| B200 NVL5+ requires AL2023 AMI | Lesson #1 | Already elevated in prior compound run (2026-03-07) |
| AL2023 EKS uses nodeadm MIME format | Lesson #2 | Already elevated in prior compound run (2026-03-07) |
| GLM-5 requires specialized SGLang image | Lesson #3 | Model-specific image requirement |
| SGLang defaults to host 127.0.0.1 | Lesson #4 | Configuration detail |
| DeepGEMM JIT ~15 min warmup | Lesson #5 | Already elevated in prior compound run (2026-03-07) |
| p6-b200 termination delay ~10 min | Lesson #6 | Already elevated in prior compound run (2026-03-07) |
| hf_xet deadlocks on macOS | Lesson #7 | Platform-specific bug, not infrastructure constraint |
| GLM-5-FP8 memory profile (175/183 GB per GPU) | Lesson #8 | Model-specific observation |
| GLM-5 throughput scales 17x with batching | Lesson #9 | Model-specific performance characteristic |
| LMCache NSA/MLA incompatibility | Lesson #10 | Already elevated in prior compound run (2026-03-07) |
| PYTHONPATH NVMe trick | Lesson #11 | Already elevated in prior compound run (2026-03-07) |
| LMCache dev branch overrides transformers version | Lesson #12 | Dependency version conflict |

## No action needed
Lessons #1, #2, #5, #6, #10, #11 were already captured in steering files during the first compound run (2026-03-07). No duplicates created.

## Analysis

### Why lessons #13-#15 are cross-cutting

**Lesson #13 (HiCache works with MLA/NSA)** solves a systemic gap affecting multiple models. LMCache is blocked on all NSA/MLA models (GLM-5, DeepSeek V3, and any future models using fused `kv_buffer` attention). HiCache's native `NSATokenToKVPoolHost` support provides an immediate alternative that works today, not dependent on external PR merge timelines. This is a framework capability difference that applies across all MLA/NSA models on SGLang, making it a general deployment rule: when LMCache fails, HiCache is the fallback.

**Lesson #14 (HiCache size assertion)** is a hard framework constraint. The `host_memory > device_memory` assertion in HiCache initialization applies to all HiCache deployments, regardless of model architecture. The lesson provides the formula for calculating required host memory (`num_tp_ranks × hicache_size`) and warns against relying on the default `--hicache-ratio 2.0` which can OOM on memory-constrained instances. This is a pre-flight validation rule that prevents deployment failures after GPU capacity is reserved.

**Lesson #15 (CPU offload changes concurrency ceiling)** captures a general performance principle. When model weights consume most GPU VRAM, the KV cache becomes the throughput bottleneck, not compute. The superlinear scaling pattern (baseline plateaus at N concurrent while CPU offload continues scaling to 2N+) is a diagnostic signal that KV cache eviction was limiting throughput. This applies across frameworks (HiCache, LMCache once MLA support merges, vLLM PagedAttention with CPU offload) and model architectures. The lesson provides a decision rule: always benchmark both baseline and CPU offload for large models to identify the true bottleneck.

### Benchmark evidence for elevation

The HiCache benchmark report provides quantitative evidence for lesson #15:

- **Baseline peak**: 909 tok/s at 64 concurrent, could not scale to 128 concurrent
- **HiCache peak**: 2,602 tok/s at 128 concurrent (2.86x improvement)
- **71% improvement at 64 concurrent**: 909 → 1,556 tok/s, confirming KV cache was the bottleneck even at moderate concurrency
- **Single-request throughput unchanged**: 48 tok/s baseline vs 48 tok/s HiCache, confirming no hot-path overhead

This performance delta is not GLM-5-specific. Any model with tight KV cache memory (high weight-to-VRAM ratio) will exhibit the same pattern. The lesson elevates the diagnostic approach (look for superlinear scaling as a KV cache bottleneck signal) rather than the specific numbers.

### Why lessons #3, #4, #7, #8, #9, #12 remain local

- **Lesson #3**: `lmsysorg/sglang:glm5-blackwell` image requirement is specific to GLM-5's `glm_moe_dsa` architecture
- **Lesson #4**: SGLang `--host 0.0.0.0` flag is a configuration detail, not a workflow sequence or platform constraint
- **Lesson #7**: macOS hf_xet deadlock is a client-side bug, not an AWS/Kubernetes infrastructure issue
- **Lesson #8**: 175 GB / 183 GB memory usage is a GLM-5-FP8 model profile, not a general MoE or Blackwell characteristic
- **Lesson #9**: 17x throughput scaling with batching is a GLM-5 observation; MoE scaling varies by expert count, routing policy, and active parameter ratio
- **Lesson #12**: LMCache dev branch overriding transformers is a transient dependency conflict, not a design pattern

These lessons are valuable for operators returning to the GLM-5 blueprint but do not generalize to other deployments.

## Recommendations

1. **Add HiCache pre-flight check to readiness audit template** — for models with tight KV cache memory (>85% GPU VRAM used by weights), verify that `num_tp_ranks × hicache_size ≤ available_system_ram` before reserving capacity. This prevents OOM during HiCache initialization.

2. **Update MLA model deployment playbook** — for models using NSA or MLA attention, the decision tree is now: (1) try SGLang HiCache first (works today), (2) wait for LMCache PR #2629 to merge if external KV offload library is required for compliance/licensing reasons, (3) fall back to device-only KV cache (RadixAttention) if neither is viable.

3. **Benchmark both baseline and CPU offload for all large models** — the 2.86x improvement from HiCache on GLM-5 justifies adding CPU offload benchmarks to the standard test matrix for any model where weights consume >75% of GPU VRAM. The cost of the extra benchmark (one additional capacity block hour) is negligible compared to the potential throughput gain.

4. **Document HiCache memory layout options** — the benchmark used `--hicache-mem-layout layer_first` (default). For models with many layers and small per-layer KV, `token_first` may perform better. Add a brief note to the steering rule recommending operators test both layouts during benchmarking.

## Next steps

- Steering file updated with three new rules (see "Elevated to steering" table)
- Consider adding a HiCache configuration template to `domains/gpu-serving/blueprints/_template/configs/hicache-cpu.sh` for future blueprints
- Track LMCache PR #2629 merge status — once merged, re-benchmark GLM-5 with LMCache to compare against HiCache baseline
- File an issue to track "HiCache vs LMCache feature parity" — some teams may have compliance requirements for external libraries vs built-in framework features

## Summary

This compound run elevates three HiCache lessons that apply across MLA/NSA models and memory-constrained serving scenarios. The key insight: CPU KV cache offloading is not a "nice to have" optimization — it fundamentally changes the concurrency ceiling for models where weights dominate GPU VRAM. HiCache provides this capability today for MLA models where LMCache is blocked, making it the recommended path forward until LMCache PR #2629 merges.

All three elevated rules are now in `.claude/steering/tech-stack.md` under "GPU Serving Conventions → Deployment Conventions". Future blueprints deploying MLA models or large MoE models on high-VRAM GPUs will benefit from these lessons without repeating the discovery process.
