# Fin RAG Answer Extraction — Final Benchmark Report

**Model**: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B (hybrid Mamba-2 + LatentMoE + Select-Attention, 120B/12B-active)
**Engine**: vLLM **0.18.1** (see Image Honesty below) · **Hardware**: 1× p6-b200.48xlarge (8× B200 183 GB, NVSwitch) SPOT
**Cluster**: qwen3-next-bench-eks-cluster (EKS 1.32, us-east-2) · **Workload**: `fin-support` (RAG, ~9K ISL / ~300 OSL, prefill-dominated)
**SLO**: E2E p50 ≤ 6,500 ms, p90 ≤ 9,500 ms at 130 concurrent · **Spot price**: $32/hr
**Date**: 2026-06-11

---

## Headline verdict

**Winner: FP8, agg-tp2-x4 (4 replicas × TP=2), `--max-num-batched-tokens 16384`, TRITON_ATTN, fp8 KV cache.**
E2E **p50 4,685 ms / p90 8,147 ms** at 130 concurrent, 0 errors → **PASS** both SLO gates with comfortable headroom.

FP8 beats BF16: BF16 fails the p90 gate (10,496 > 9,500) at the target concurrency; FP8 passes and is faster across the board (lower TPOT, tighter tail, faster floor).

---

## FP8 vs BF16 + full config comparison @ conc=130 (with cost at spot $32/hr)

| Config | E2E p50 | E2E p90 | TPOT p50 | tot tok/s | **$/1M total tok** | $/1M output tok | SLO |
|--------|--------:|--------:|---------:|----------:|-------------------:|----------------:|-----|
| **FP8 mnbt=16384 (winner)** | 4,685 | 8,147 | 55.0 | 221,168 | **$0.0402** | $4.70 | PASS |
| FP8 mnbt=8192 | 5,105 | 8,878 | 55.8 | 203,489 | $0.0437 | $4.98 | PASS |
| FP8 mnbt=4096 | 7,065 | 11,933 | 83.0 | 150,348 | $0.0591 | $6.91 | FAIL |
| BF16 mnbt=8192 | 6,128 | 10,496 | 72.5 | 169,386 | $0.0525 | $6.30 | **p90 FAIL** |
| Leg 1 — KV-auto (fp8 anyway) | 4,598 | 7,366 | 49.8 | 236,860 | $0.0375 | $4.44 | PASS |
| Leg 2 — FlashInfer attn | 4,197 | 7,598 | 44.9 | 245,831 | **$0.0362** | $4.28 | PASS |
| Leg 3 — agg-tp4-x2 | 6,755 | 10,593 | 74.5 | 167,241 | $0.0532 | $6.37 | **FAIL** |
| Leg 4 — TP1 (conc=64) | 10,559 | 16,940 | 129.0 | 50,606 | $0.1756 | $20.83 | FAIL |

- **$/1M total tokens** (prefill in + decode out) is the right cost metric for this prefill-dominated workload (~9K in / ~76–300 out). The winner serves at **~$0.040 / 1M total tokens** on spot.
- **$/1M output tokens** is shown for reference but is inflated/noisy because the model under-generates (out p50 ~76 vs telemetry OSL p50 243 at temp=0) — treat output-token cost as an upper bound.
- **Cheapest config that holds SLO**: Leg 2 (FlashInfer) at $0.0362/1M total, then the TRITON winner at $0.0402/1M. See FlashInfer caveat below before adopting.

---

## Optimization-grid legs

### Leg 1 (P1) — FP8 KV cache on vs off → NO-OP
Dropping `--kv-cache-dtype fp8` does **not** disable fp8 KV: vLLM 0.18.1 auto-selects `kv_cache_dtype=fp8_e4m3` for the ModelOpt FP8 checkpoint (it ships k/v scales). Available KV memory unchanged at 95.02 GiB. The flag is effectively a confirmation of the default. The 4,598 vs 4,685 delta is run-to-run noise on the identical fp8 KV path. **There is no real "KV-off" config on the FP8 checkpoint** short of loading BF16 (which fails p90). KV footprint is not a lever here (507× conc headroom either way).

### Leg 2 (P1) — FlashInfer vs TRITON_ATTN → FlashInfer faster, but accuracy caveat
FlashInfer attention **loaded** on this hybrid Mamba2 model (contrary to the spec's caution) and is **~10% faster** on E2E p50 (4,197 vs 4,685) and **~18% lower TPOT** (44.9 vs 55.0), 0 errors. **Caveat**: with `--kv-cache-dtype fp8`, fp8 attention runs with **uncalibrated q_scale/prob_scale = 1.0** (the FP8 checkpoint lacks q/prob scales) — vLLM warns "may cause accuracy issues". (This warning is from fp8 KV, present on TRITON too — not unique to FlashInfer.) **Recommendation**: FlashInfer is a viable latency/cost win *if* a coherence smoke confirms no quality regression; otherwise keep TRITON_ATTN (hybrid-safe default). The FP8 MoE GEMM backend stayed `FLASHINFER_TRTLLM` in both legs (independent subsystem).

### Leg 3 (P2) — agg-tp4-x2 vs agg-tp2-x4 → tp2-x4 wins
2 replicas × TP=4 **FAILS both SLO gates** at conc=130 (p50 6,755 / p90 10,593). Root cause: 2 replicas means ~65 concurrent/replica vs ~32 for the 4-replica TP2 layout → deeper prefill queue → higher TTFT/E2E. Extra per-replica KV headroom (635× vs 507×) is wasted; NVSwitch TP4 all-reduce comms lift TPOT (74.5 vs 55.0). **More replicas beats more TP for this prefill-dominated, high-concurrency RAG workload.** TP4 NCCL initialized cleanly on B200 NVSwitch (no Blackwell-PCIe-class issue). TP4 only wins the conc=8 single-stream TPOT floor (9.0 ms), irrelevant at conc=130.

### Leg 4 (P2, key science) — TP1 prefix-cache probe → hit rate STILL 0
**The interesting result.** Prefix-cache hit rate is **0 at TP1 too**, not just TP>1. A single TP1 replica loaded on one B200 (no OOM; KV pool 1.94M tok / 116× headroom) and served 1,805,903 prefix queries against the byte-identical ~3,050-token shared header with:
```
vllm:prefix_cache_queries_total{engine="0"} 1,805,903
vllm:prefix_cache_hits_total{engine="0"}    0.0
```
So the 0% hit rate is **NOT** the upstream #26201 "TP>1 not yet tested" caveat — **automatic prefix caching is non-functional for Nemotron-3-Super (hybrid Mamba2) on vLLM 0.18.1 at ANY TP.** `--enable-prefix-caching --mamba-cache-mode all` are accepted and `enable_prefix_caching=True` shows in the engine config, but the Mamba-2 'all'-mode path never registers a hit. This points at the Mamba2 prefix-caching implementation (vLLM #26201/#25752 merged but not yet effective for this architecture on 0.18.1), not the parallelism layout. TP1 also fails SLO (single-GPU throughput; 116× < 130 target) — it was a science probe, not a serving candidate. **The RAG prefix-cache win does not materialize on this build; SLO is met purely on raw B200 prefill throughput + KV headroom.** Re-validate on a newer vLLM with mature Mamba2 PC.

### Leg 5 (disagg) — DEFERRED
4p4d / 2p6d KV-transfer disagg not run — the session was consumed by the four cold-start cycles above (each ~4–6 min on Recreate). Given the prefill-dominated shape and that aggregated tp2-x4 already passes SLO with headroom, disagg is low-priority. Defer to a future session if sub-4s p50 is required.

---

## Speculative-decode verdict (CLOSED — not deployable)

Spec-decode is **not deployable for Nemotron-3-Super on vLLM 0.18.1 at TP2**. Both paths are blocked by the Mamba2 backend:
- **Native MTP**: crashes in the MTP draft forward (`size of tensor a (2) must match b (3)`) — shape mismatch under the hybrid Mamba2 + select-attention layout. Pods exit 137 in a restart loop.
- **n-gram / prompt-lookup**: crashes in CUDA-graph capture (`mamba_attn.py:498 _update_metadata_for_cudagraph_capture`, tensor 2 vs 6) — the Mamba2 backend assumes a fixed single-token decode shape and cannot reconcile the multi-token speculative query.

The `--enforce-eager` workaround (which sidesteps graph capture to *measure* acceptance) was **explicitly rejected by the operator as unrealistic** for a benchmark of record (eager latency is not production-representative). No acceptance number is reported. Consistent with the workload being prefill-dominated (~300-token output), where spec-decode was always the secondary lever. Re-test only on a vLLM build that fixes Mamba2 graph-capture + spec-decode.

---

## Image honesty

Serving ran on vLLM **0.18.1** (ECR `615299764834.dkr.ecr.us-east-2.amazonaws.com/vllm-openai:v0.18.1`). The benchmark.yaml sidecar lists 0.22.1 as the *primary candidate* with 0.18.1 as fallback and a Stage 0 smoke test as the arbiter. **There is no on-disk evidence that 0.22.1 was ever deployed, smoke-tested, or that it failed** — the 0.18.1 image was skopeo-copied to ECR, passed the Stage 0 coherence smoke (5/5 coherent grounded answers, model loads as `NemotronHForCausalLM`), and was used for all runs. The 0.22.1 candidacy appears to have been **skipped**, not tried-and-failed. The engine config line in every pod log confirms `v0.18.1`. No 0.22.1 failure is claimed.

---

## Prefix-cache finding (summary)

Hit rate is **0 at TP2 and at TP1** (authoritative `vllm:prefix_cache_hits_total` = 0 after 1.8M queries). Mamba2 automatic prefix caching is non-functional on vLLM 0.18.1 for this architecture regardless of TP. The ~3,050-token shared system header is re-prefilled every request. SLO still passes on B200 prefill throughput. The reliability gate (hit rate must stay ≤ ~30% corpus ceiling) is trivially satisfied (0%). This is the single most actionable finding for a future vLLM upgrade.

---

## Campaign cost & GPU-hours

- **Node**: p6-b200.48xlarge SPOT, joined ~12:25Z, this report ~18:15Z → **~5.8 GPU-node-hours** (8× B200).
- **Cost at spot $32/hr**: ~**$186** for the full campaign (Stage 0 → all P0/P1/P2 legs).
- At on-demand $98/hr the same campaign would be ~$568. Spot saved ~67%.
- Per-config serving cost at the operating point: winner **~$0.040 / 1M total tokens** (prefill+decode) on spot — extremely cost-efficient for a 120B hybrid MoE on a frontier GPU.

---

## Recommendation to customer

1. **Deploy FP8 agg-tp2-x4, mnbt=16384, TRITON_ATTN** (the validated winner): p50 4,685 / p90 8,147 at 130 concurrent, ~$0.040/1M total tokens on spot, 0 errors, ample headroom to the 25-RPS peak.
2. **Optionally adopt FlashInfer attention** for a ~10% latency / cost improvement, *after* a coherence smoke confirms the uncalibrated-fp8-scale path produces no quality regression on real Fin prompts.
3. **Do not use** BF16 (fails p90), TP4x2 (fails SLO), spec-decode (blocked on 0.18.1), or TP1 (single-GPU throughput too low).
4. **Prefix caching is currently unavailable** for this model — budget for full re-prefill of the shared header, or upgrade vLLM and re-validate Mamba2 PC before relying on it.
5. **Re-test on a newer vLLM** when one lands working Mamba2 prefix caching AND Mamba2 graph-capture + spec-decode — both could meaningfully lower TTFT / cost.


## Max-concurrency-at-SLO sweep (2026-06-11)

Pressure-tested the winner (FP8 agg-tp2-x4, mnbt=16384) past the 130 anchor to find the SLO ceiling. Gates: E2E p50 <= 6500, p90 <= 9500.

| conc | E2E p50 | E2E p90 | TPOT p50 | errors | verdict |
|------|--------:|--------:|---------:|-------:|---------|
| 130 (anchor) | 4899 | 7909 | 60.9 | 0 | PASS |
| 160 | 5351 | 7837 | 65.6 | 0 | PASS |
| 200 | 6395 | 9188 | 77.6 | 0 | PASS (marginal, ~105ms p50 / ~312ms p90 headroom) |
| 256 | 7711 | 11101 | 94.5 | 0 | FAIL (both gates) |

- **SLO ceiling = conc ~200** (true edge just above 200; 256 is first failure). Bracket 200->256.
- **0 errors at every level including 256** — failure is pure latency (decode-batch interference lifting TPOT), not capacity/errors. TTFT p50 stayed <500ms throughout.
- **~1.5x headroom** from the validated 130-concurrent operating point to the ceiling; 25-RPS production peak is comfortably inside.
- Anchor reproduced within ~5% (campaign c130 4685/8147 vs sweep c130 4899/7909). No drift.
- Raw: `/tmp/fin-rag-conc-sweep-20260611.md` + `/tmp/fin-rag-conc-sweep-c{130,160,200,256}.json`; in-pod `/tmp/sweep/`.
