# GLM-5.2-FP8 B200 TP8 — Tier Stack Table (coding-agent workload, 12K shared prompt)

SGLang v0.5.13.post1-cu130, single 8-GPU B200, --context-length 65536. All [measured] client-side agg tok/s.
DCGM PROF unavailable (driver-580, kimi L8) → regime via engine gauges. CUDA graphs default-ON in all tiers.

| Tier | Config (cumulative) | c=1 | c=4 | c=16 | c=32 | c=64 | c=64 TTFT p99 | Verdict |
|------|---------------------|-----|-----|------|------|------|---------------|---------|
| T0 | FP8 floor, radix OFF | 98.5 | 314.7 | 883.4 | 1305.3 | 1708.2 | 29.7s | reference (BF16 doesn't fit; FP8 is floor) |
| T2 | +prefix cache | 99.3 | 317.6 | 1101.6 | 1884.6 | 2950.1 | 3.50s | **+73% tok/s, 8.5× TTFT — dominant lever** (12K shared prompt @ 92% hit) |
| T1 | +fp8 KV cache | 99.3 | 364.4 | 1135.1 | 1931.2 | 3004.2 | 3.33s | +2% tok/s on coding-agent (capacity lever — real value at long ctx, see 31k sweep) |
| T3 | +EAGLE/MTP spec | 138.7 | 480.8 | 1239.3 | 1650.7 | 2330.7 | 3.78s | **latency lever: +40% @ c1, −22% @ c64.** accept_len ~2.25. Uses native NextN/MTP head. ON for low-conc, OFF for high-conc |
| T5 | +torch.compile | — | — | — | — | — | — | **BLOCKED — "Capture cuda graph failed" on GLM-5.2/DSA path; engine says disable torch.compile.** CUDA graphs already default-ON (the achievable kernel gain is in the baseline) |

## Key findings
- **T2 (prefix cache) is the headline coding-agent lever** — the 12K shared system prompt caches at 92%, collapsing TTFT 8.5× and lifting throughput 73%.
- **T1 (fp8 KV) is near-null on coding-agent** but is a capacity lever; its payoff is in the long-input-31k regime (KV-bound).
- **T3 (spec decode) is regime-dependent**: per-user latency win at low concurrency (+40% c1), aggregate loss at high (−22% c64). EAGLE drove the model's native MTP/NextN head; accept_len ~2.25 on coding-agent shape (not synthetic-inflated).
- **T5 torch.compile BLOCKED** on the glm_moe_dsa + SGLang 0.5.13 path (cuda graph capture fails). CUDA graphs run by default regardless, so the kernel tier's achievable gain is already in T0-T3.

## Best operating configs
- **High-concurrency throughput**: T1 (prefix + fp8 KV, no spec) — 3004 tok/s @ c64, TTFT 3.33s.
- **Low-latency / interactive**: T3 (+EAGLE) — 139 tok/s/user @ c1 (+40%).

## long-input-31k sweep (T1: prefix + fp8 KV), B200 TP8 — DSA long-context + KV-capacity test

| c | agg tok/s | TTFT p99 | cache hit | err |
|---|-----------|----------|-----------|-----|
| 16 | 560.2 | 7.9s | 0.70 | 0 |
| 32 | 830.1 | 11.2s | 0.74 | 0 |
| 64 | 985.7 | 22.3s | 0.74 | 0 |
| 128 | 1139.4 | 44.6s | 0.74 | 0 |
| 256 | 1166.2 | 110.1s | 0.74 | 0 |

**Regime = PREFILL-COMPUTE-BOUND (not KV-capacity-bound) [gauge-inferred].** Evidence: throughput
plateaus (+2% c128→c256) while TTFT explodes (44.6s→110s); ZERO errors/OOM/preemptions to c=256
(capacity-bound would error out). Requests queue on prefill compute for the 31K-token inputs, not KV room.
~74% cache hit reproduced (matches kimi customer profile, byte-identical shared prefix).

**SLO knee** (kimi 15s TTFT p99 ceiling): ~c=32-48 (11.2s @ c32, breaches by c64). Per-request decode
holds (well above the 34.8 tok/s/req floor at the knee).

**B300 decision: B300 would NOT help this workload.** B300's advantage is KV capacity (275 vs 180 GB/GPU),
but 31k is prefill-COMPUTE-bound, so the TP4+DP2 B300 arm wouldn't raise this knee. The TP4+DP2 layout
remains worth testing for the coding-agent throughput regime, but NOT to relieve the 31k prefill wall.
fp8 KV (T1) was near-null on coding-agent AND doesn't bind here either — KV was never the constraint
on a single B200 for this workload. The real long-context lever would be prefill parallelism (chunked
prefill tuning / more compute), not more KV.

## MNBT (chunked-prefill-size) sweep on long-input-31k — the prefill-bound lever

| MNBT | c=16 TTFT | c=32 TTFT | c=64 agg | c=128 agg | verdict |
|------|-----------|-----------|----------|-----------|---------|
| 8192 | 5.5s | 11.0s | 1063 | — | **best at SLO knee** (smaller chunks → faster TTFT) |
| 16384 (default) | 7.9s | 11.2s | 986 | 1139 | baseline |
| 32768 | — | 13.7s | 1032 | 1242 | worse TTFT at knee, +throughput past ceiling |

**MNBT=8192 wins at the latency-sensitive knee** (c16: 5.5s vs 7.9s default); larger chunks (32768) trade
knee-TTFT for high-conc throughput. For the long-context SLO operating point, use 8192. MoE tile tuning
N/A (moe_runner_backend=flashinfer_trtllm, vendor-precompiled — doesn't read Triton tile JSON).

## 16k context tier (kimi comparison) — GLM-5.2-FP8 B200 TP8, MNBT=8192

| c | agg tok/s | TTFT p99 | cache |
|---|-----------|----------|-------|
| 16 | 1039 | 3.1s | 0.74 |
| 32 | 1645 | 5.5s | 0.74 |
| 64 | 2200 | 11.0s | 0.74 |
| 128 | 2865 | 21.9s | 0.74 |

SLO-safe (15s TTFT) to ~c64-80 at 16k. 0 errors.

**Kimi comparison — NOT apples-to-apples (flag loudly):** kimi-k2.6-nvfp4 16k numbers (5,161 tok/s @ c64,
SLO-safe to c512) were on **B300 + TP4+DP2 + 1T-NVFP4**, vs GLM-5.2 here on **B200 + TP8 + 753B-FP8**.
Three confounded variables (hardware, model, parallelism). GLM-5.2-FP8/B200-TP8 is ~2.3× lower throughput
at 16k c64 than Kimi-NVFP4/B300-TP4+DP2, but a clean comparison requires matched HW+layout. The forced-TP8
on B200 (753GB FP8 doesn't fit TP4) is a real structural disadvantage vs kimi's TP4+DP2. NVFP4 (in progress,
lukealonso/GLM-5.2-NVFP4, full-size 753B) + a B300 TP4+DP2 run would close the gap to a fair comparison.

### 16k high-concurrency (FP8, MNBT=8192) — full ceiling
| c | agg tok/s | TTFT p99 | err |
|---|-----------|----------|-----|
| 256 | 1464 | 162s | 0 |
| 512 | 2162 | 220s | 0 |
GLM-5.2-FP8/B200-TP8 16k: throughput climbs to ~2160 @ c512 but TTFT is far past SLO (220s). The 15s-SLO
ceiling is ~c64-80. Contrast kimi-NVFP4/B300-TP4+DP2 held SLO to c512 — but that's B300+NVFP4+TP4+DP2 (3
confounds). 0 errors throughout = NOT capacity-bound; pure prefill-compute saturation at TP8 on B200.

## T2+ HiCache (CPU KV tiering) — measured NEGATIVE/NULL, FP8 sweep COMPLETE
SGLang --enable-hierarchical-cache --hicache-ratio 2 (105GB host/rank × 8, pod mem 1600Gi, no hang).
| workload | T1 (no HiCache) | T2+HiCache | delta |
|----------|-----------------|------------|-------|
| coding-agent c64 agg | 3004 | 2629 | **−12%** |
| coding-agent c16 TTFT | — | 2.96s (cache 0.89) | — |
| 31k c32 TTFT | 11.0s | 12.3s | ~flat |
| 31k c64 agg | 986 | 1113 | +13% (TTFT unchanged 22s) |
**Verdict: HiCache does NOT help GLM-5.2 here.** Workloads are prefill-compute-bound (not KV-capacity-
bound), so CPU tiering has no capacity wall to relieve — it adds overhead (−12% on coding-agent). Matches
the kimi/GLM-5 lesson: HiCache only pays when capacity-bound. **FP8 tier sweep COMPLETE.**

## FP8 SWEEP — FINAL SUMMARY (coding-agent, B200 TP8)
Best high-throughput config: **T1 (prefix cache + fp8 KV, MNBT default)** = 3004 tok/s @ c64, TTFT 3.33s.
Best low-latency: **T3 (+EAGLE/MTP)** = 139 tok/s/user @ c1 (+40%).
Best long-context knee: **MNBT=8192** (TTFT 5.5s @ 16-conc on 31k vs 7.9s default).
Levers that DON'T help GLM-5.2: HiCache (−12%, compute-bound), torch.compile (BLOCKED — cuda graph fail),
MoE tile tuning (N/A — trtllm backend). Dominant lever: prefix cache (T2, +73%/8.5×TTFT).

## NVFP4 vs FP8 (T1, coding-agent) — community lukealonso/GLM-5.2-NVFP4, full 753B
Quality gate PASSED (fixed the bug correctly, glm47 tool-call works). Weights 462GB, 56GB/GPU at TP8
(vs FP8's ~94GB). Same moe_runner_backend=flashinfer_trtllm as FP8 (NOT a fallback).

| c | FP8 (T1) tok/s | NVFP4 tok/s | delta |
|---|----------------|-------------|-------|
| 1 | 99 | 114 | **+15%** (NVFP4 wins single-stream — less weight BW) |
| 16 | 1135 | 451 | **−60%** |
| 32 | 1931 | 915 | **−53%** |
| 64 | 3004 | 1219 | **−59%** |

**SURPRISING: NVFP4 REGRESSES batched throughput ~55-60% vs FP8** on this SGLang 0.5.13/glm_moe_dsa
path, despite the optimization-stack expectation of +15-25%. Pattern (single-stream win → batch loss)
indicates the NVFP4 trtllm MoE GEMM is less batch-efficient than the FP8 path here — NVFP4's advantage
is weight-bandwidth (helps low-conc/memory-bound), but batched decode is compute-bound on this kernel.
**NVFP4 is NOT a free win for GLM-5.2 serving today**: use FP8 for throughput; NVFP4 only helps at c=1
(latency) or where VRAM is the constraint (56 vs 94 GB/GPU → frees KV room, relevant on smaller GPUs).
Caveat: community modelopt quant, not official; a tuned NVFP4 MoE kernel could change this. CONFIRMED not
a quality issue (gate passed) and not a backend fallback (same trtllm runner).

### NVFP4 long-context (vs FP8) — gap narrows with context
| workload | FP8 | NVFP4 | delta |
|----------|-----|-------|-------|
| 16k c32 | 1645 | 1130 | −31% |
| 16k c64 | 2200 | 1624 | −26% |
| 31k c32 | 831 | 779 | **−6%** |
NVFP4's batch-throughput deficit SHRINKS as context grows (−60% coding-agent → −6% @ 31k): at long
context the run is prefill/DSA-attention-bound, so MoE-GEMM efficiency matters less. NVFP4 still never
beats FP8 at batch on any workload. NET: FP8 is the production choice for GLM-5.2 throughput; NVFP4 only
for c=1 latency or VRAM-constrained deployments. **NVFP4 comparison COMPLETE.**
