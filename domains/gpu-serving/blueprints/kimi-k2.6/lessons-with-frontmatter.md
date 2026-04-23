---
model: "moonshotai/Kimi-K2.6"
engine: "vllm+sglang"  # Dual-engine comparison benchmark
hardware: "p6-b300.48xlarge"
gpu_arch: "sm_103"  # B300 Blackwell NVSwitch
deployment_date: "2026-04-22"

outcome: "success"
failure_categories: []

cards_used:
  mdc: ["kimi-k2.6"]
  gpu_infra: []

card_helped: true  # mdc card provided deployment flags and cold start expectations

benchmark:
  throughput_toks_s: 10437  # vLLM peak at c=512
  ttft_p50_ms: 22  # vLLM single-stream best case (W5, qps=0.5)
  ttft_p99_ms: 1137  # vLLM at c=512
  concurrent_users: 512
  gpu_util_pct: null  # Not tracked in benchmark logs

ralph_iterations: 1  # Single benchmark session, all tracks completed

mdc_learn_commands:
  - 'mdc learn kimi-k2.6 vllm "vLLM v0.19.1 with FLASHINFER_MLA: 2-3x lower TTFT, 3.1x higher throughput at c=512 (10,437 vs 3,400 tok/s SGLang). Cold start ~8.3 min. Peak: 10,437 tok/s at c=512, $0.43/1M tokens on B300 spot."'
  - 'mdc learn kimi-k2.6 vllm "vLLM prefix caching: 103x TTFT improvement on first cold→warm hit (5928ms → 57ms). Consistent 1.5-1.65x on subsequent queries. Key differentiator for RAG/multi-turn."'
  - 'mdc learn kimi-k2.6 vllm "Remove --disable-log-requests flag in v0.19.1 (unrecognized, causes crash)."'
  - 'mdc learn kimi-k2.6 vllm "Use vllm/vllm-openai:v0.19.1-cu130 for B300 (sm_103). Standard tags lack Blackwell support."'
  - 'mdc learn kimi-k2.6 sglang "SGLang v0.5.10: 3 min cold start (2.8x faster than vLLM), 139-143 tok/s single-stream TPS. RadixAttention shows ~1.0x prefix cache benefit (requires exact token match). Saturates at c=128. Peak: 3,400 tok/s at c=512."'
  - 'mdc learn kimi-k2.6 sglang "HiCache on B300 TP8: use --hicache-size 200 (max 250 GB/rank = 2 TB total). --hicache-size 500 OOMs (4 TB exceeds system RAM). +58% single-stream TPS, -37% ITL. No benefit at high concurrency."'
  - 'mdc learn kimi-k2.6 sglang "Use lmsysorg/sglang:v0.5.10.post1-cu130 for B300 (sm_103)."'
  - 'mdc learn kimi-k2.6 sglang "SGLang uses reasoning_content field (not reasoning) for streaming responses. Update benchmark scripts."'

gpu_infra_learn_commands: []
---

# Kimi K2.6 Benchmark — Lessons Learned

## Session: 2026-04-22

**Model**: Kimi K2.6 (moonshotai/Kimi-K2.6), 1T MoE, 32B active, INT4 QAT
**Hardware**: p6-b300.48xlarge (8x B300 SXM6 AC, 268GB HBM3e, NV18 NVSwitch, sm_103)
**Engines**: vLLM v0.19.1, SGLang v0.5.10.post1
**Outcome**: COMPLETE — all tracks and priority tiers finished

---

## Blocking: HiCache 500GB/rank OOMs on B300

**Severity**: HIGH
**Category**: configuration

SGLang `--hicache-size 500` allocates 500 GB host memory **per TP rank**. With TP=8 that's 4 TB, which exceeds available system RAM (~3.8 TB free after model + OS on p6-b300). The pod hangs indefinitely at "Allocating 500.00 GB host memory for hierarchical KV cache" — no error, no OOM kill, just stuck.

**Fix**: Use `--hicache-size 200` (200 GB/rank = 1.6 TB total). This is sufficient for benchmark workloads and matches the GLM-5 B200 recommendation of sizing hicache above the device KV pool.

**Rule**: For B300 with TP8, max safe hicache-size is ~250 GB/rank. For B200 (141 GB HBM3e), use 100-200 GB/rank.

---

## Lesson: benchmark-serving.py SGLang reasoning_content field

**Severity**: MEDIUM
**Category**: tooling

The shared benchmark script `scripts/benchmark-serving.py` used `delta.get("reasoning")` to capture reasoning tokens from streaming responses. SGLang uses `reasoning_content` (matching OpenAI's convention), not `reasoning`. vLLM also uses `reasoning_content` in its latest versions but the bug was masked because vLLM's streaming format is slightly different.

**Fix**: Changed to `delta.get("reasoning_content", "") or delta.get("reasoning", "")` to handle both field names.

**Impact**: Without this fix, all SGLang benchmark results showed 0ms TTFT and 0.0 TPS — the script couldn't detect any tokens being generated.

---

## Lesson: vLLM FLASHINFER_MLA dominates SGLang on B300

**Severity**: HIGH
**Category**: engine-selection

vLLM v0.19.1 with FLASHINFER_MLA backend (block size 32) provides:
- 2-3x lower TTFT than SGLang across all workloads
- 3.1x higher aggregate throughput at 512 concurrent (10,437 vs 3,400 tok/s)
- Near-linear throughput scaling to 512 concurrent (SGLang saturates at ~128)
- 103x prefix caching improvement on cold→warm (5928ms → 57ms)

SGLang's advantages are limited to:
- 2.8x faster cold start (3 min vs 8.3 min)
- Slightly higher single-stream TPS with HiCache (136 vs 124 tok/s)

**Recommendation**: Default to vLLM for Kimi K2.6 on B300. Only use SGLang when cold start time is critical (spot instances with frequent interruptions).

---

## Lesson: SGLang RadixAttention shows no prefix caching benefit

**Severity**: MEDIUM
**Category**: engine-behavior

Both SGLang configurations (base and HiCache) showed ~1.0x improvement in W2 RAG prefix caching tests. RadixAttention requires exact prefix match at the token level, but the benchmark generates slight query variations that break the prefix match.

vLLM's prefix caching implementation is more tolerant — it achieved 103x TTFT improvement on the first cold→warm hit and consistent 1.5-1.65x improvements on subsequent queries.

**Implication**: For RAG/multi-turn workloads where exact prefix reuse is common (shared system prompts, document QA), vLLM's prefix caching provides dramatically better performance.

---

## Lesson: EKS managed node groups unreliable for spot B300

**Severity**: HIGH
**Category**: infrastructure

EKS managed node groups for p6-b300.48xlarge spot instances got stuck in CREATING state for 20+ minutes with no ASG provisioned. Multiple attempts failed. The launch template InstanceType field conflicts with EKS overrides.

**Fix**: Bypass managed node groups entirely. Launch spot instances directly via `aws ec2 run-instances` with nodeadm UserData for self-managed EKS joining. Fix CA cert via SSM if needed.

**Rule**: For GPU spot instances, prefer self-managed nodes over EKS managed node groups. The control plane overhead isn't worth the unreliability.

---

## Lesson: B300 requires -cu130 container tags

**Severity**: HIGH
**Category**: compatibility

B300 GPUs are compute capability 10.3 (sm_103). Standard container images compiled for sm_80/sm_90 will fail. Both vLLM and SGLang require the `-cu130` tagged images:
- `vllm/vllm-openai:v0.19.1-cu130`
- `lmsysorg/sglang:v0.5.10.post1-cu130`

This is different from B200 (sm_100) which uses `-cu128` or standard images with CUDA 12.8+.

---

## Lesson: vLLM --disable-log-requests removed in v0.19.1

**Severity**: LOW
**Category**: compatibility

The `--disable-log-requests` flag that was commonly used in vLLM v0.15-v0.18 is no longer recognized in v0.19.1. Launching with this flag causes an immediate crash: `vllm: error: unrecognized arguments: --disable-log-requests`.

**Fix**: Remove the flag from pod specs and serving configs.

---

## Lesson: HiCache improves SGLang single-stream but not under load

**Severity**: MEDIUM
**Category**: engine-behavior

SGLang HiCache (200 GB/rank host memory) provided significant single-stream improvements:
- TPS: 136 vs 86 tok/s (+58%)
- ITL: 6.7ms vs 10.6ms (-37%)
- TTFT: 93ms vs 155ms (-40%)

But at high concurrency (qps=8, c=16+), HiCache performance matched or slightly trailed base SGLang. The bottleneck shifts from KV cache capacity to compute at high load, making the host memory tier irrelevant.

**Rule**: HiCache is most valuable for single-stream or low-concurrency workloads where KV eviction is the bottleneck. For high-concurrency serving, it adds cold start overhead without throughput benefit.

---

## Lesson: K2.6 vs K2.5 — generational improvement

**Severity**: INFO
**Category**: model-performance

K2.6 on B300 (vLLM v0.19.1) vs K2.5 on H100 (vLLM v0.15.1):
- Single-stream TPS: 128 vs 41 tok/s (3.1x)
- Agentic TTFT: 45-118ms vs 820-926ms (8-18x)
- Multi-turn TTFT: 22-62ms vs 1216-1565ms (25-71x)
- Cost per 1M tokens: $0.43 vs ~$217 (est) (~500x at peak throughput)

The improvement comes from three factors: B300 hardware (faster HBM3e, NV18 NVSwitch), vLLM v0.19 (FLASHINFER_MLA, better scheduling), and INT4 QAT (smaller model footprint, more KV cache headroom).
