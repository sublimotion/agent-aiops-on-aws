# Steering Rule Updates for Kimi K2.6 Deployment

## Instructions

Append the following rules to `.claude/steering/tech-stack.md` after line 547 (after the "HyperPod EKS GPU pods require three tolerations" section).

---

## Rules to Append

```markdown
#### B200 and B300 Blackwell NVSwitch require -cu130 container tags (sm_100f and sm_103)
<!-- stack: vllm=0.19.1, sglang=0.5.10 | validated: 2026-04-22 -->

B200 GPUs (compute capability sm_100f) and B300 GPUs (compute capability sm_103) require CUDA 13.0 (`-cu130`) container images. Standard container images compiled for sm_80/sm_90 will fail. Use:
- vLLM: `vllm/vllm-openai:v0.19.1-cu130`
- SGLang: `lmsysorg/sglang:v0.5.10.post1-cu130`

This applies to all p6-b200 and p6-b300 deployments. Do not use `-cu128` or older CUDA tags — they lack the Blackwell architecture support.

#### vLLM --disable-log-requests flag removed in v0.19.1
<!-- stack: vllm=0.19.1 | validated: 2026-04-22 -->

The `--disable-log-requests` flag that was commonly used in vLLM v0.15-v0.18 is no longer recognized in v0.19.1. Launching with this flag causes an immediate crash: `vllm: error: unrecognized arguments: --disable-log-requests`. Remove the flag from pod specs and serving configs when upgrading to vLLM v0.19+. This is a breaking change, not a deprecation.

#### SGLang HiCache sizing per hardware platform — B300 needs ~250 GB/rank max, B200 uses 100-200 GB/rank
<!-- stack: sglang=0.5.10 | validated: 2026-04-22 -->

SGLang `--hicache-size` allocates host memory **per TP rank**. Total host memory requirement is `num_tp_ranks × hicache_size`. For p6-b300.48xlarge (4 TB system RAM, TP=8): max safe `--hicache-size` is ~250 GB/rank (2 TB total). Setting `--hicache-size 500` would require 4 TB, which exceeds available system RAM after model weights and OS, causing the pod to hang indefinitely at "Allocating host memory for hierarchical KV cache" with no error. For p6-b200.48xlarge (2 TB system RAM, TP=8): use 100-200 GB/rank. Always calculate total requirement before launching and verify it fits within system RAM. This is a platform constraint, not a model-specific limitation.

#### HiCache improves single-stream throughput but not under load — bottleneck shifts to compute at high concurrency
<!-- stack: sglang=0.5.10 | validated: 2026-04-22 -->

SGLang HiCache (CPU KV cache offloading) provides significant single-stream improvements: +58% TPS, -37% ITL, -40% TTFT compared to device-only KV cache. However, at high concurrency (qps=8, c=16+), HiCache performance matches or slightly trails base SGLang. The bottleneck shifts from KV cache capacity to compute at high load, making the host memory tier irrelevant. HiCache is most valuable for single-stream or low-concurrency workloads where KV eviction is the bottleneck. For high-concurrency serving, it adds cold start overhead without throughput benefit. This pattern applies to all HiCache deployments across models and hardware.

#### vLLM FLASHINFER_MLA dominates SGLang on B300 NVSwitch — 2-3x lower TTFT, 3.1x higher throughput at scale
<!-- stack: vllm=0.19.1, sglang=0.5.10 | validated: 2026-04-22 -->

On B300 NVSwitch hardware (p6-b300.48xlarge), vLLM v0.19.1 with FLASHINFER_MLA backend (block size 32) provides 2-3x lower TTFT across all workloads and 3.1x higher aggregate throughput at 512 concurrent (10,437 vs 3,400 tok/s) compared to SGLang v0.5.10. vLLM also achieves near-linear throughput scaling to 512 concurrent while SGLang saturates at ~128 concurrent. SGLang's advantages are limited to 2.8x faster cold start (3 min vs 8.3 min) and slightly higher single-stream TPS. Default to vLLM for latency-sensitive or high-concurrency MLA models on B300. Only use SGLang when cold start time is critical (spot instances with frequent interruptions). This pattern applies to Kimi K2.x, DeepSeek V3, and similar MLA architectures on B300.

#### SGLang RadixAttention requires exact prefix match — no benefit from query variations
<!-- stack: sglang=0.5.10 | validated: 2026-04-22 -->

SGLang RadixAttention (automatic prefix caching) requires exact token-level prefix match to trigger radix tree reuse. Benchmark workloads with slight query variations (e.g., RAG with different retrieval contexts) show ~1.0x improvement (no caching benefit). vLLM's prefix caching implementation is more tolerant — it achieved 103x TTFT improvement on first cold→warm hit and consistent 1.5-1.65x improvements on subsequent queries in the same workload. For RAG/multi-turn workloads where exact prefix reuse is uncertain, prefer vLLM prefix caching over SGLang RadixAttention. This applies to all SGLang deployments where prefix caching is a key optimization target.

#### EKS managed node groups unreliable for spot B300 — use self-managed nodes with direct EC2 launch
<!-- stack: eks=1.32 | validated: 2026-04-22 -->

EKS managed node groups for p6-b300.48xlarge spot instances can get stuck in CREATING state for 20+ minutes with no ASG provisioned. Multiple attempts fail. The launch template InstanceType field conflicts with EKS overrides. Bypass managed node groups entirely. Launch spot instances directly via `aws ec2 run-instances` with nodeadm UserData for self-managed EKS joining. Fix CA cert via SSM if needed. For GPU spot instances, prefer self-managed nodes over EKS managed node groups. The control plane overhead is not worth the unreliability. This applies to all large GPU instance types on spot market.

#### scripts/benchmark-serving.py SGLang requires reasoning_content field fallback
<!-- stack: sglang=0.5.10, vllm=0.19.1 | validated: 2026-04-22 -->

The shared benchmark script `scripts/benchmark-serving.py` must handle both `reasoning_content` (OpenAI convention, used by SGLang and vLLM v0.19+) and `reasoning` (vLLM v0.15-v0.18) when capturing reasoning tokens from streaming responses. Use `delta.get("reasoning_content", "") or delta.get("reasoning", "")` to handle both field names. Without this fallback, SGLang benchmark results show 0ms TTFT and 0.0 TPS — the script cannot detect any tokens being generated. This applies to all benchmarking scripts that consume streaming responses from multiple engines.
```

---

## Verification

After appending, verify with:
```bash
grep -n "B200 and B300 Blackwell NVSwitch require -cu130" .claude/steering/tech-stack.md
```

Should return a line number greater than 547.
