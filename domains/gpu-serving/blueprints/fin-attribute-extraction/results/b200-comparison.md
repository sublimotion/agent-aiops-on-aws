# Qwen3.6-35B-A3B — B200 vs g7e per-GPU efficiency (2026-06-15)

**Question**: does B200 raise the per-replica ceiling for this launch-bound 3B-active hybrid MoE, or stay flat (as the g7e launch-bound analysis predicted)?

**Answer: B200 delivers ~1.8–2× the per-replica SLO-safe throughput — my pre-run "stays flat" prediction was WRONG.** The reason is a specific, measured optimization that fires on B200 but not g7e.

## Setup
- B200: p6-b200.48xlarge spot (us-east-2), 1 GPU, vLLM 0.22.1 FP8, TP1, mnbt=8192, `--limit-mm-per-prompt '{"image":0,"video":0}'`. EKS `qwen3-next-bench-eks-cluster`, ns ml-inference.
- g7e: RTX PRO 6000 Blackwell sm_120 (Tokyo), same vLLM/FP8/TP1/mnbt=8192 config (from earlier session).
- Workload: random 2500-in / 22-out (fin-attribute-extraction shape).

## Per-replica SLO sweep (E2E p90 ms, FP8, graph capture, mnbt=8192)
| Rate | g7e p90 | B200 p90 | g7e SLO | B200 SLO |
|------|---------|----------|---------|----------|
| 10   | 1401    | 863      | PASS    | PASS |
| 12   | 1991    | 916      | PASS(edge) | PASS |
| 14   | 3489    | 948      | **FAIL** | PASS |
| 18   | (sat ~14) | 1439   | —       | PASS |
| 22   | —       | 1829     | —       | **PASS (knee)** |
| 26   | —       | 2670     | —       | FAIL |
| 30   | —       | 3876     | —       | FAIL |

- **g7e**: SLO-safe ~10–12 RPS, saturates ~14 req/s.
- **B200**: SLO-safe **~22 RPS**, saturates **~27 req/s**. **~1.8–2× per replica.**
- Unloaded latency identical (B200 TTFT 97ms/E2E 181ms vs g7e 98/237) — the win is throughput ceiling, not per-request latency.

## WHY — the deciding optimization (measured, not assumed)
Two B200-specific levers were checked vs g7e:
1. **FP8 MoE backend = TRITON on BOTH** (`fp8.py:405 Using TRITON Fp8 MoE backend`). So the MoE kernel path is the *same* — NOT the differentiator. (Refutes the hypothesis that B200 picks FlashInfer TRTLLM for this model.)
2. **Full CUDA-graph capture: SUCCEEDS on B200, CRASHED on g7e.** B200 logged `Capturing CUDA graphs (decode, FULL): 100%|...| 35/35`; g7e crashed full-decode capture with `1 != ISL` shape assertion (had to run effectively without it). **This is the differentiator.** Full decode-graph capture eliminates per-step kernel-launch overhead — exactly the bottleneck that capped g7e.

## Bottleneck class flipped (DCGM)
- **g7e at saturation**: SM ~43%, HBM ~0% → **launch/scheduling-bound** (SMs idle waiting on dispatch).
- **B200 at knee (rate 26)**: SM **~99%**, HBM mem-util ~31%, 786W → **compute/SM-bound**. The working CUDA graphs let the SMs actually fill; the launch wall is gone, so the box runs until it's genuinely compute-limited.

## Corrected guidance
- My earlier report claim "a bigger GPU won't help per-replica because it's launch-bound, and kernel tuning was a no-op" was **correct for g7e but does NOT transfer to B200**. The launch bound was *engine/graph-capture-specific*, not hardware-fundamental. On B200 where full graph capture works, the launch wall lifts and you get ~2× per replica.
- **Capacity for 70 RPS**: B200 ~22 RPS/replica → **~4 replicas** (half a p6-b200 node, 4 GPUs) vs g7e's ~6× g7e.48xl GPUs. Cost comparison still TODO (B200 spot ~$19-30/hr/8-GPU node vs g7e).
- **Caveat**: vLLM 0.22.1 has a CUTLASS-DSL ICE on B200 in the *vision encoder* path (`vit_flash_attn_wrapper`→`cute.compile`); must serve text-only with `--limit-mm-per-prompt '{"image":0,"video":0}'`. The g7e full-graph crash (`1 != ISL`) may be fixable on a newer vLLM — if so, g7e's ceiling could rise too.

## Infra lessons (B200 spot nodegroup on this EKS cluster)
- **The `ai-infra-use2-b200-spot` nodegroup launches WITHOUT the node security group** (only the cluster SG) → ALL pod networking (cross-node, FSx, DNS) silently broken. Fix: attach the node SG (`qwen3-next-bench-eks-cluster-node-*`) to the instance ENI. This was the root cause of a long failure cascade.
- B200 node also needs the `nvidia.com/gpu.present=true` label or the device plugin won't schedule (GPUs show 0 allocatable).
- B200 node has **no internet egress** and **FSx-Lustre CSI mount fails** (NID parse error, Lustre kmod mismatch on the AMI). Workaround: stage weights to FSx from a CPU node, serve them over HTTP from a CPU pod, fetch to B200 emptyDir via pure-Python (no apt/egress needed).
