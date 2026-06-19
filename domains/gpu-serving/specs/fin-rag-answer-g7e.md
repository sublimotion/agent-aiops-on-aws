# Fin RAG Answer Extraction on g7e — Cost-Comparison Addendum

## Status: DRAFT (2026-06-11)

## Parent Spec

See [`fin-rag-answer.md`](./fin-rag-answer.md) for the model, workload, SLO,
optimization grid, and success criteria. This addendum documents **only the
deltas** for re-running the benchmark on **g7e (RTX PRO 6000 Blackwell, PCIe)**
and answering one question:

> **Can the much cheaper g7e hold the 6.5s/9.5s E2E SLO at 130 concurrent, and
> what is the $/1M-token delta vs the B200 baseline?**

The customer SLO, traffic, and workload card (`fin-support.yaml`) are
hardware-independent and carry over **unchanged**. Everything below is a
hardware delta. **Run the B200 parent benchmark first** — this addendum is a
cost-comparison against those numbers, not a standalone hunt.

---

## Compute Delta

| Property | p6-b200.48xlarge (parent) | g7e.48xlarge (this spec) |
|----------|---------------------------|--------------------------|
| GPU | 8× B200 (183 GB HBM3e) | 8× RTX PRO 6000 (96 GB GDDR7) |
| Architecture | Blackwell sm_100 | Blackwell **sm_120** |
| Interconnect | **NVSwitch NVL5+** | **PCIe Gen5 only — no NVLink/NVSwitch** |
| Aggregate VRAM | 1,464 GB | 768 GB |
| EFA | — | 4 interfaces (SRD, kernel-bypass; NOT true RDMA) |
| Spot price | ~$32/hr (us-east-2b, 2026-06-11) | ~$33/hr on-demand / verify spot |
| AMI | AL2023 NVIDIA (Fabric Manager) | AL2023 NVIDIA |
| Container runtime | containerd | **nerdctl** (containerd) |

> The g7e is NOT obviously cheaper per hour than B200 *spot* right now — the
> cost case rests on **on-demand/committed** pricing and on whether g7e can hold
> the SLO at all. If B200 spot stays at ~$32/hr, the headline becomes
> **$/1M-token at SLO**, dominated by g7e's lower throughput, not a raw $/hr win.

---

## What transfers from the parent grid (and what does NOT)

| Axis | Transfers? | g7e delta |
|------|-----------|-----------|
| **A — Precision** (bf16/fp8) | **Axis yes, values no** | FP8 is **sm_120**, not sm_100 — DeepGEMM/FlashInfer FP8 MoE kernels must be re-verified on Blackwell PCIe. Both checkpoints fit (FP8 124 GB, BF16 240 GB on 768 GB). |
| **B — Kernel/batching** (chunked-prefill, max-num-seqs, FP8 KV, CUDA graph) | **Flags yes, optima no** | GDDR7 has far less bandwidth than HBM3e → **must re-sweep**; the B200 winning values will not be optimal. Prefill throughput is the gating resource here. |
| **C — Prefix caching** (`mamba_cache_mode=all`, ~3050-tok header) | **Yes** | Engine feature, hardware-independent. Biggest RAG win; carries. Re-validate at TP>1 on PCIe. |
| **D — Spec-decode** (n-gram + MTP) | **Acceptance yes, net speedup no** | Acceptance is workload-driven (transfers); the net E2E/TPOT benefit changes on weaker compute — **re-measure at temp=1.0**. |
| **E — Parallelism / disagg** | **NO — redesign** | g7e is **PCIe-only**. Optimization pressure **inverts**: favor *lower* TP to cut cross-PCIe comms (opposite of B200). NVSwitch-dependent disagg (4p4d/2p6d KV transfer) is out of scope on g7e; EFA is CPU-bounce, a different regime. |

**Net**: reuse the grid *structure*, drop Axis E, re-sweep B, re-verify A on sm_120.

---

## Stage 0 — g7e Pre-Flight (fail-closed, in addition to parent Stage 0)

Parent Stage 0 (checkpoint resolution, FP8-not-NVFP4, version smoke) already
passed and is hardware-independent. Additional g7e gates:

- [ ] **FP8 MoE kernels work on sm_120**: load the FP8 checkpoint on g7e and
      generate ~10 coherent Fin outputs. FP8 is per-tensor static on this
      checkpoint (parent Stage 0 verified) → **no block_n=128 divisibility
      constraint**, so TP1/2/4 are all arithmetically valid; the question is
      kernel support, not divisibility.
- [ ] **NCCL version ≥ 2.26.2 in the serving image** — NCCL 2.25.1 is BROKEN on
      Blackwell sm_120 + PCIe (all collectives fail at `enqueue.cc:1500`). vLLM
      inference uses custom allreduce so is unaffected, but **verify the image's
      NCCL** before any SGLang/TP-collective path. (See MEMORY.md.)
- [ ] **VRAM fits the chosen TP** (see budget below).
- [ ] Serving-config resolver (parent Stage 0c) re-run with g7e infrastructure
      block — must exit 0.

---

## Upstream evidence — sm_120 bring-up is ALREADY characterized (skip the trial-and-error)

A community benchmark of **this exact model on this exact hardware** exists:
SGLang issue [#20541](https://github.com/sgl-project/sglang/issues/20541) —
Nemotron-3-Super-120B-A12B-**FP8** on **8× RTX PRO 6000 (sm_120), g7e.48xlarge**,
SGLang 0.5.6.post2, 3-run validated. It resolves the Stage 0 kernel-support
questions below, so **G0 collapses to a smoke confirmation, not exploration.**
CAVEAT: it is **SGLang, not vLLM**, and its shapes (200/200 burst, 16K-in
long-ctx) are NOT our fin-support 9K×130 SLO shape — so it removes the bring-up
risk but does NOT answer the SLO question (still our run).

**SM120 FP8 backend matrix (from #20541, "most SM120-friendly model tested", 2 required flags):**

| Component | sm_120 status | Action |
|-----------|:-------------:|--------|
| FP8 GEMM — DeepGEMM | **FAIL** (`kernel_runtime.hpp:45` assert) | disable: `SGLANG_DISABLE_DEEP_GEMM=1` |
| FP8 GEMM — CUTLASS | **FAIL** (SM100/SM90-only gate) | n/a |
| FP8 GEMM — **Triton** | **WORKS** | force `--fp8-gemm-backend triton` |
| MoE runner — **Triton** | **WORKS** (512 experts / 8 GPU) | force `--moe-runner-backend triton` |
| Attention — **FlashInfer** | **WORKS** at defaults (GQA 32Q/2KV) | no triton-attn workaround needed |
| Attention — trtllm_mha | **FAIL** (`nemotron_h.py:408`, not ported) | avoid |
| CUDA graphs | **WORKS** (GQA, no SMEM overflow) | NO `--enforce-eager` needed |
| KV cache — fp8_e4m3 | **WORKS** | use it |
| Mamba-2 layers — Triton | **WORKS** (auto-selected) | — |
| DeepEP / HiCache | **FAIL** (PCIe P2P — universal g7e blocker) | out of scope on g7e |

**Model-specific flags confirmed on sm_120:** `--mamba-scheduler-strategy no_buffer`
(`extra_buffer` asserts unsupported for NemotronHForCausalLM), `--reasoning-parser
nemotron_3` (NOT `nano_v3`/`super_v3`). **First FP8 ModelOpt load ~45 min** (one-time
quantization processing), ~14s thereafter — budget the cold start. KV capacity
17.9M tokens. 5/5 coherence at temp=0.

**Spec-decode on g7e (SGLang EAGLE/MTP) WORKS** — accept rate **0.81**, length 3.2 —
but ONLY with `--speculative-moe-runner-backend triton` (else draft MoE OOM-kills),
and it forces `--disable-radix-cache` + lower mem-fraction (0.80). NOTE: this is
the OPPOSITE of the B200/vLLM finding (0% accept / OOM at TP2) — spec-decode
viability is engine+hardware-specific, so re-measure on the vLLM g7e path; do not
assume the SGLang result transfers to vLLM.

**Reference perf (SGLang, for sanity-checking our vLLM numbers — NOT the SLO answer):**
online 376 tok/s @ 4rps / 150ms TTFT / 20ms ITL; 692 @ 8rps (unsaturated);
16K-in long-ctx cold TTFT ~43s, warm 871ms–3.8s via radix cache (no HiCache on g7e).

---

## TP layout — PCIe inverts the parent's pressure

On B200/NVSwitch the parent can afford high TP. On g7e every TP shard crosses
PCIe, so the sweep should **start low and only raise TP if VRAM forces it**.

| Layout | GPUs used | Note |
|--------|-----------|------|
| **agg-tp2-x4** | 8 | Mirror of B200 baseline — direct comparison point. Each TP=2 pair crosses one PCIe hop. |
| **agg-tp1-x8** | 8 | 8 independent FP8 replicas, **zero TP collectives** — likely best on PCIe IF the model fits TP1 (FP8 124 GB > 96 GB → does NOT fit one GPU; TP1 only viable with smaller footprint, so probably N/A for this 120B model). |
| **agg-tp4-x2** | 8 | 2 replicas × TP=4 — more comms, more KV headroom per replica. Only if tp2 is VRAM-bound at 130 concurrent. |

> FP8 weights are ~124 GB → cannot fit a single 96 GB GPU, so **TP≥2 is
> mandatory**; agg-tp2-x4 is the natural g7e baseline. Sweep tp2 vs tp4 for the
> comms-vs-KV-headroom tradeoff. **No disagg.**

---

## VRAM Budget (96 GB/GPU)

| Component (FP8, TP=2) | Estimate |
|-----------------------|----------|
| Model weights (FP8, TP=2) | ~62 GB/GPU |
| KV cache + Mamba state (prefix-cache on) | ~26 GB/GPU |
| Activations + chunked-prefill buffers | ~6 GB/GPU |
| **Total** | ~94 GB/GPU (tight) |

BF16 (240 GB) at TP=2 = ~120 GB/GPU → **does NOT fit**; BF16 needs **TP≥4** on
g7e (~60 GB/GPU weights). So the precision comparison on g7e is
**FP8 @ tp2-x4 vs BF16 @ tp4-x2** — not the same-layout comparison the B200 run
does. Document this asymmetry in results.

---

## Benchmark Phases (staged, cheap-first)

```
G0 ~0.5 hr · G1 ~2 hr · G2 ~1 hr
```

### G0 — Smoke (MUST pass before anything else) — REDUCED to confirmation
The sm_120 FP8 kernel matrix is already known from upstream #20541 (see "Upstream
evidence" above), so G0 is no longer an exploration — boot with the **known-good
backends pre-set** (`SGLANG_DISABLE_DEEP_GEMM=1 --fp8-gemm-backend triton
--moe-runner-backend triton`, FlashInfer attn at defaults; vLLM equivalents on the
vLLM path) and just CONFIRM: FP8 loads on sm_120, 10/10 coherent Fin output, prefix
cache engages on the shared header, custom-allreduce healthy. If the smoke matches
upstream, proceed straight to G1. If it DIVERGES from #20541 (e.g. a vLLM-vs-SGLang
kernel gap), that divergence is itself the finding — capture it and fall back to the
matrix above. Budget the **~45 min first-load** FP8 ModelOpt processing.

### G1 — SLO feasibility @ production shape
- G1a: FP8 agg-tp2-x4, prefix-cache on, `fin-support` @ conc 8 & 130. **Does g7e
  hold E2E p50 6.5s / p90 9.5s at 130 concurrent?** This is the whole question.
- G1b: best chunked-prefill value (re-sweep {2048,4096,8192,16384}) at conc 130.
- G1c: BF16 agg-tp4-x2 for the precision/$ comparison.

**Gate**: report sustained-concurrency-at-SLO. If SLO breaks well before 130,
g7e is a capacity finding (cheaper but can't hold peak), not a pass.

### G2 — Knobs that earn their place (only if G1 holds or is close)
- Full `fin-support` concurrency sweep [8,32,128,512].
- Spec-decode acceptance @ temp=1.0 — net effect on weaker compute. NOTE upstream
  #20541 shows SGLang EAGLE/MTP **works on g7e** (accept 0.81, len 3.2) — opposite
  of the B200/vLLM result. Re-measure on the vLLM g7e path; the SGLang viability
  does NOT transfer across engines. Requires `--speculative-moe-runner-backend triton`.
- FP8 KV on/off — already confirmed working on sm_120 (#20541); just measure delta.
  Attention: FlashInfer works at defaults on sm_120 (GQA) per #20541, so no need to
  force TRITON_ATTN here (unlike the B200 hybrid-safe default) — confirm, then compare.

---

## Infrastructure Changes

- **Separate blueprint**: `domains/gpu-serving/blueprints/fin-rag-answer-g7e/`
  (own Terraform state; do not entangle with the B200 blueprint).
- **Reuse staged weights**: same FSx Lustre bucket as the B200 run — no re-upload.
- **nerdctl**, not docker, for any host-level container ops (g7e convention).
- **`--network host`** for bare-metal g7e pods (no CNI on some g7e setups; verify).

---

## Success Criteria

1. G0 smoke passes (FP8 works on sm_120 Blackwell PCIe, prefix cache engages).
2. **SLO verdict**: does g7e hold E2E p50 6.5s/p90 9.5s at 130 concurrent? Report
   sustained-concurrency-at-SLO with cold-vs-warm prefix hit rate.
3. **Headline cost table**: $/1M-token at the SLO operating point, **g7e vs B200**,
   using each platform's measured throughput and its real price (B200 spot vs g7e
   on-demand/spot). Include the FP8-tp2 vs BF16-tp4 asymmetry note.
4. `lessons.md` captures sm_120 / PCIe-specific findings (FP8 kernels, TP comms
   overhead, NCCL version, VRAM fit).

---

## Known Limitations / Risks

1. **PCIe TP comms** — cross-PCIe allreduce is slower than NVSwitch; the B200
   "best TP" finding does NOT transfer. Re-sweep tp2 vs tp4.
2. **BF16 needs TP4 on g7e** (doesn't fit tp2) — precision comparison is
   cross-layout, not apples-to-apples. State it explicitly.
3. **NCCL 2.25.1 broken on sm_120 PCIe** — verify ≥2.26.2 in the image (vLLM
   inference unaffected via custom allreduce, but any NCCL collective path fails).
4. **FP8 kernel support on sm_120** — DeepGEMM/FlashInfer FP8 MoE paths must be
   re-verified; do not assume B200 sm_100 behavior.
5. **g7e may simply not hold the SLO** — prefill-dominated 9K×130 against 6.5s is
   demanding on GDDR7 bandwidth. A clean "cheaper but can't hold peak" result is
   a valid, useful outcome — surface it as a capacity finding, not a failure.
6. **Spot parity** — if B200 spot stays ~$32/hr, g7e's cost edge is small unless
   on-demand/committed pricing is the customer's reality. Frame the $/1M-token
   comparison against the customer's actual purchasing model.

---

> Operational artifacts live in `blueprints/fin-rag-answer-g7e/`, not in this spec.
> Compare results directly against `blueprints/fin-rag-answer/results/` (B200).
