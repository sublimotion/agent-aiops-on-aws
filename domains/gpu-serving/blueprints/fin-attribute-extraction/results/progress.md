# Fin Attribute Extraction — Qwen3.6-35B-A3B on g7e — Benchmark Results

**Date**: 2026-06-15
**Hardware**: g7e.12xlarge, Tokyo (ap-northeast-1a), 2× RTX PRO 6000 Blackwell (sm_120, 96 GB, PCIe). Instance `i-0de1456111eb3b30e`.
**Engine**: vLLM 0.22.1 (staged image), model `Qwen/Qwen3.6-35B-A3B` (bf16 ~72 GB; FP8 via `--quantization fp8` online).
**Workload**: `fin-attribute-extraction` — ISL p50 1654 / mean 2500, OSL ~22. SLO: E2E p50 ≤1s, **p90 ≤2s @ 70 RPS peak**.

> us-west-2 24xl (4 GPU) was the first choice but **InsufficientInstanceCapacity even on restart** (pinned to us-west-2d). Fell back to the Tokyo 12xl (2 GPU) — the cheaper instance the spec's decision gate actually targets.

## Headline result: g7e CANNOT hold the 70 RPS SLO; it's launch-bound, ~5 replicas short

| Config | SLO-safe RPS (E2E p90 ≤2s) | Saturation RPS | vs 70 peak |
|--------|---------------------------|----------------|-----------|
| **FP8, 1 replica** | **~10** | ~14 | need ~7 replicas |
| **bf16, 1 replica** | **~5–6** | ~11 | need ~12 replicas |
| bf16, 2 replicas (DP=2, measured) | ~17 saturation total | ~17 | ~4× short |

**To hold 70 RPS within SLO on FP8 you need ~7 TP1 replicas → g7e.48xlarge (8 GPU)**, NOT the 12xl/24xl. The spec's "start on g7e, escalate to B200" gate resolves to: **g7e is viable but only at 48xl size**; a B200 slice is the alternative if a single node must hold peak with margin.

## Stage 0 (all PASS)
- **Arch**: registers as `qwen3_5_moe` / `Qwen3_5MoeForConditionalGeneration` — same family as Qwen3.5, hybrid **linear-attention (Gated-DeltaNet, 3 of every 4 layers) + full-attention** MoE, 256 experts (8 routed), `moe_intermediate_size=512`, 40 layers. Multimodal (vision encoder present, unused).
- **vLLM 0.22.1 loads it cleanly on sm_120** — weights 25s, fits one 96 GB GPU (bf16 65 GiB, FP8 32 GiB).
- **Thinking disable WORKS** (the headline risk): `chat_template_kwargs={"enable_thinking": False}` → output `negative`, **2 tokens, no `<think>`**. With thinking on: 64 tokens of deliberation (would blow the 1s SLO). Verified.
- **FP8 TP-divisibility**: `moe_intermediate_size 512 / TP % 128 == 0` → TP1/2/4 OK, TP8 forbidden. Moot on 2-GPU box.

## Bottleneck class — LAUNCH-BOUND (DCGM, confirmed)
FP8 single-replica DCGM at saturation: **SM ~43%, HBM mem-util ~0%.** Neither compute- nor bandwidth-bound. The 256 tiny experts (`moe_intermediate_size=512`) + Gated-DeltaNet decode fire many small kernels → SMs busy with dispatch, not math. Same signature as Nemotron-3-Super and the bottleneck-migration thesis. **TPOT is anomalously high for a 3B-active model (123–168 ms loaded)** — the tell of launch overhead, not FLOPs. A bigger GPU (B200/B300) would NOT help per-replica throughput; more replicas is the only lever.

## Per-replica precision comparison (single TP1 replica, 2500 in / 22 out)
| Rate | bf16 E2E p90 | FP8 E2E p90 | bf16 SLO | FP8 SLO |
|------|-------------|-------------|----------|---------|
| 5    | 1272 ms     | 667 ms      | PASS     | PASS    |
| 10   | 6096 ms     | **1394 ms** | FAIL     | **PASS** |
| 15   | 9885 ms     | 5012 ms     | FAIL     | FAIL    |
| 20   | —           | 8933 ms     | —        | FAIL    |

- **FP8 ~doubles SLO-safe per-replica capacity** (~10 vs ~5-6 RPS) and saturates higher (~14 vs ~11 req/s). Unlike Nemotron (where FP8 bought only ~1.25-1.4× throughput), here it matters more because the binding constraint is the admission ceiling and FP8's lighter kernels raise it.
- **Unloaded both are fast**: bf16 TTFT 98 ms / E2E 237 ms; FP8 TTFT 93 ms / E2E 203 ms at the real p50 ISL. **Per-request latency is NOT the problem — capacity is.**

## Engineering gotchas (g7e / vLLM 0.22.1 / qwen3_5_moe)
1. **`max_num_seqs` default 1024 > Mamba cache blocks (754)** → CUDA-graph capture refuses to start: `max_num_seqs exceeds available Mamba cache blocks`. Fix: `--max-num-seqs 256`.
2. **FP8 + `--data-parallel-size 2` is BROKEN on 0.22.1 for this hybrid GDN MoE**: `AssertionError: 1 != <ISL>` in the all-reduce/sequence-parallel path → EngineCore dies on first real request. bf16 DP=2 worked. Workaround: run independent single-GPU replicas (no DP coordinator) behind a load balancer. (Consistent with the 0.22-avoid flag in memory.)
3. **First-batch JIT is brutal**: `fused_moe_kernel`, `_causal_conv1d_fwd_kernel`, GDN kernels JIT-compile on first unseen shape → 32 s TTFT on a cold batch, then 93 ms warm. ALWAYS warm up before measuring; cold-start makes the server look wedged.
4. **g7e needs `--network host`** (no CNI on bare metal); container runtime is `nerdctl`; start `containerd` first.
5. **No official FP8** — used vLLM online `--quantization fp8` (per-tensor dynamic). Community quant selection (spec Stage 0) not needed for the benchmark, but quality vs a real corpus is unverified.

## Tuning follow-up (2026-06-15, second session) — mnbt helps ~20%, kernel tuning is a no-op

Re-ran on Tokyo g7e (had to re-acquire an ODCR — ODCR was cancelled after run 1, so restart hit `ReservationCapacityExceeded`; **a stopped instance does NOT auto-attach to an open ODCR — must `modify-instance-capacity-reservation-attributes ... CapacityReservationTarget` while stopped, then start**).

### mnbt sweep (FP8 single replica, WITH graph capture)
| Rate | mnbt=2048* | mnbt=4096 | mnbt=8192 | default(~) |
|------|-----------|-----------|-----------|-----------|
| 8    | 2615 (eager) | 1256 | **1124** | — |
| 10   | 3515 (eager) | 1746 | **1401** | 1394 |
| 12   | 4830 (eager) | 2450 FAIL | **1991 PASS** | (fails) |
| 14   | 7072 (eager) | 3318 | 3489 | (sat) |

(E2E p90 ms. *mnbt=2048 run was eager-mode — discarded as confounded, but directionally confirms sub-ISL chunking hurts.)

- **mnbt=8192 WINS: extends SLO-safe ceiling from ~10 → ~12 RPS (~20% per-replica capacity), one-line change.** Monotonic: bigger mnbt = better.
- **Mechanism**: ISL is ~2500, so mnbt < ISL (2048, 4096) *forces* prefill chunking → more scheduler iterations → worse on a launch-bound box. mnbt ≥ ISL (8192) runs each prefill in one step → fewer launches. Same lesson as Nemotron (size mnbt UP when launch-bound). The spec's hypothesis (sweep {2048,4096,8192} to *pack* requests) had the direction backwards for this prefill-heavy shape.
- **Methodology note**: first attempt used `--enforce-eager` to dodge slow cold starts — WRONG for a launch-bound study (eager disables the CUDA graphs that are the key launch lever). Re-ran with graph capture. Eager data kept only as `mnbt_eager_throwaway.log`.

### MoE kernel tile tuning — NO-OP (but for a different reason than Nemotron)
Generated the missing tile config (`E=256,N=512,fp8_w8a8` on RTX PRO 6000) with the ray-free tuner (`fin-rag-answer/scripts/tune-moe-rayfree.py`, batches 8/12/16/24/32, pruned). Mounted via `VLLM_TUNED_CONFIG_FOLDER`, **confirmed loaded** (`Using configuration from /tuned/E=256,N=512,...json for MoE layer` — the "default sub-optimal" warning gone).
| Rate | untuned p90 | tuned p90 |
|------|------------|-----------|
| 10   | 1401 | 1921 |
| 12   | 1991 PASS | 2531 FAIL |
| 14   | 3489 | 5102 |

- **Tuned config did NOT help (slightly worse, within variance).** Crucially: unlike Nemotron/B200 (where tuning was inert because the kernel was FlashInfer and never read the JSON), here the kernel **IS Triton and DID load the tuned JSON** — yet no gain.
- **Conclusion**: the launch-bound bottleneck is the *number* of kernel launches + scheduling across the 256-expert + Gated-DeltaNet path, NOT per-GEMM tile efficiency. Tile tuning optimizes each GEMM's throughput; it cannot reduce dispatch count. So even where the tuner is *live* (g7e Triton path), it doesn't move a launch-bound knee.

### Net for the original question (B200/H200 per-GPU efficiency)
The two software levers settle it: **mnbt=8192 is the real lever (~20%), kernel tuning is dead.** A bigger GPU still won't help per-replica (launch-bound, and tuning proved the kernel isn't the issue). Updated capacity math with mnbt=8192: ~12 RPS SLO-safe/replica → **~6 FP8 replicas for 70 RPS → still g7e.48xlarge**, marginally better than the ~7 from the untuned baseline.

## Recommendation
- **If single-node peak required**: g7e.48xlarge (8× FP8 replicas @ mnbt=8192 ≈ ~96 RPS SLO-safe headroom) OR a B200 slice. The 12xl/24xl cannot hold 70 RPS.
- **Use FP8 + `--max-num-batched-tokens 8192`** (online per-tensor FP8) — FP8 gives ~2× SLO-safe capacity vs bf16 and fits 32 GB (1 replica/GPU, no TP collective); mnbt=8192 adds another ~20% (~10 → ~12 RPS/replica). Do NOT set mnbt below the ISL (~2500) — it forces chunking and hurts.
- **Do NOT bother tuning the MoE tile config** — measured no-op here even though the Triton path loads it (the bound is launch count, not GEMM tiles).
- **Run independent single-GPU replicas behind a router**, not vLLM DP=2 (FP8 DP broken on 0.22.1).
- Per-request latency is excellent (~200 ms unloaded); the deployment question is purely replica count for the 70 RPS admission ceiling.

## Raw data
`results/raw/`: `sweep_bf16.log` (DP=2 full), `sweep_fp8_single.log`, `bf16_single_sweep.txt`, `dmon_r*.txt` (DCGM), `sweep_fp8_dp2_crash.log` (DP-FP8 assertion), `mnbt_sweep_graphcapture.log` (mnbt 4096/8192), `mnbt_eager_throwaway.log` (discarded eager run), `moetune.log` + `E=256,N=512,...fp8_w8a8.json` (tuned config), `tuned_moe_rebench.txt` (no-op result).
