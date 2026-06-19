# Fin RAG Answer Extraction on H200 — Cost-Comparison Addendum

## Status: DRAFT (2026-06-11)

## Parent Spec

See [`fin-rag-answer.md`](./fin-rag-answer.md) for the model, workload, SLO,
optimization grid, and success criteria. This addendum documents **only the
deltas** for re-running the benchmark on **H200 (p5e.48xlarge, Hopper sm_90,
NVSwitch)** and answering one question:

> **How does H200 (the mature, no-surprises Hopper baseline with NVSwitch)
> compare to B200 on $/1M-token and reliability for this RAG workload?**

The customer SLO, traffic, and workload card (`fin-support.yaml`) are
hardware-independent and carry over **unchanged**. Everything below is a
hardware delta. **Run the B200 parent benchmark first** — this addendum is a
cost-comparison against those numbers, not a standalone hunt.

---

## Compute Delta

| Property | p6-b200.48xlarge (parent) | p5e.48xlarge (this spec) |
|----------|---------------------------|--------------------------|
| GPU | 8× B200 (183 GB HBM3e) | 8× H200 (141 GB HBM3e) |
| Architecture | Blackwell **sm_100** | Hopper **sm_90** |
| Interconnect | **NVSwitch NVL5+** | **NVSwitch NVL5** |
| Aggregate VRAM | 1,464 GB | 1,128 GB |
| EFA | — | 32× 400 Gbps (optional single-node) |
| Spot price | ~$32/hr (us-east-2b, 2026-06-11) | TODO — verify live us-east-2a spot rate |
| AMI | AL2023 NVIDIA (Fabric Manager) | AL2023 NVIDIA (Fabric Manager) |
| NCCL maturity | mature (sm_100) | **mature (sm_90) — gold standard** |

> H200 is the **mature, no-surprises baseline** — it sits between B200 (faster,
> more VRAM, newer sm_100) and g7e (cheaper, less VRAM, PCIe-only sm_120).
> The key advantage over g7e: **NVSwitch + mature NCCL** (no sm_120 shared-mem
> bug, no PCIe TP comms bottleneck). The question vs B200 is cost-efficiency:
> does H200's compute difference materially change throughput/$?

---

## What transfers from the parent grid (and what does NOT)

| Axis | Transfers? | H200 delta |
|------|-----------|-----------|
| **A — Precision** (bf16/fp8) | **Yes** | FP8 is **sm_90**, mature ecosystem. DeepGEMM/FlashInfer FP8 MoE kernels are standard-path on Hopper (unlike sm_120 workarounds). Both checkpoints fit (FP8 124 GB, BF16 240 GB on 1,128 GB). |
| **B — Kernel/batching** (chunked-prefill, max-num-seqs, FP8 KV, CUDA graph) | **Flags yes, optima no** | H200 HBM3e bandwidth between B200 and g7e → **re-sweep chunked-prefill** to find the H200 sweet spot; do not assume B200 mnbt=16384 is optimal. Prefill throughput is the gating resource. |
| **C — Prefix caching** (`mamba_cache_mode=all`, ~3050-tok header) | **Yes** | Engine feature, hardware-independent. Biggest RAG win; carries. Re-validate at TP>1 on H200/vLLM (B200 0% hit rate finding must be confirmed). |
| **D — Spec-decode** (n-gram + MTP) | **Acceptance yes, net speedup no** | Acceptance is workload-driven (transfers); the net E2E/TPOT benefit changes on weaker compute — **re-measure** (B200 spec-decode blocked on vLLM 0.18.1; re-check H200). |
| **E — Parallelism / disagg** | **Partial** | H200 has **NVSwitch like B200**, so TP comms are cheap (unlike g7e PCIe). VRAM per-GPU is 141 GB (< B200's 183 GB), so TP1 vs TP2 breakpoint may differ. Re-sweep tp1/tp2/tp4 layouts; disagg feasible. |

**Net**: reuse the grid *structure*, re-sweep B for H200 bandwidth, re-verify C/D
on H200/vLLM to confirm B200 findings, re-sweep E to account for 141 GB VRAM.

---

## Stage 0 — H200 Pre-Flight (fail-closed, in addition to parent Stage 0)

Parent Stage 0 (checkpoint resolution, FP8-not-NVFP4, version smoke) already
passed and is hardware-independent. Additional H200 gates:

- [ ] **FP8 MoE kernels work on sm_90**: load the FP8 checkpoint on H200 and
      generate ~10 coherent Fin outputs. FP8 is per-tensor static on this
      checkpoint (parent Stage 0 verified) → **no block_n=128 divisibility
      constraint**, so TP1/2/4 are all arithmetically valid; the question is
      H200 kernel maturity (should be gold-standard on sm_90).
- [ ] **NCCL version ≥ 2.23 in the serving image** — H200 Hopper sm_90 is
      **mature NCCL territory** (unlike sm_120 Blackwell PCIe). No known NCCL
      collectives bug on Hopper NVSwitch. Confirm anyway for completeness.
- [ ] **VRAM fits the chosen TP** (see budget below).
- [ ] Serving-config resolver (parent Stage 0c) re-run with H200 infrastructure
      block — must exit 0.

---

## Upstream evidence — sm_90 is standard-path (skip the trial-and-error)

Hopper sm_90 is the **most mature serving architecture** in vLLM and SGLang —
DeepGEMM FP8 GEMM, CUTLASS FP8 MoE, FlashInfer/FlashDecoding attention are all
standard-path on sm_90. This is the **OPPOSITE** of g7e's sm_120 Triton-only
workaround matrix (g7e issue #20541). On H200:

| Component | sm_90 (H200) status | g7e sm_120 (for contrast) |
|-----------|:------------------:|--------------------------|
| FP8 GEMM — DeepGEMM | **WORKS** (standard) | FAIL (requires `SGLANG_DISABLE_DEEP_GEMM=1`) |
| FP8 GEMM — CUTLASS | **WORKS** (standard) | FAIL (SM100/SM90-only gate) |
| FP8 GEMM — Triton | works (fallback) | WORKS (forced via `--fp8-gemm-backend triton`) |
| MoE runner — CUTLASS | **WORKS** (standard) | n/a |
| MoE runner — Triton | works (fallback) | WORKS (forced via `--moe-runner-backend triton`) |
| Attention — FlashInfer | **WORKS** (GQA 32Q/2KV) | WORKS (GQA) |
| Attention — FlashDecoding | **WORKS** | suspect (not tested on sm_120) |
| CUDA graphs | **WORKS** | WORKS |
| KV cache — fp8_e4m3 | **WORKS** | WORKS |
| Mamba-2 layers — Triton | **WORKS** (auto) | WORKS (auto) |
| NCCL collectives | **WORKS** (mature) | FAIL on NCCL ≤2.25.1 (sm_120+PCIe bug) |
| DeepEP / HiCache | **WORKS** (NVSwitch P2P) | FAIL (PCIe P2P blocker) |

**Model-specific flags confirmed on sm_90**: `--mamba-scheduler-strategy no_buffer`
(same as B200/g7e; `extra_buffer` asserts unsupported for NemotronHForCausalLM),
`--reasoning-parser nemotron_3` (NOT `nano_v3`/`super_v3`). **First FP8 ModelOpt
load ~15 min** (one-time quantization processing), ~14s thereafter — same as B200.

**Expectation**: H200 should be the **most reliable** hardware to bring up (no
sm_120 workarounds, no sm_100 bleeding-edge surprises, mature NCCL). If H200
has bring-up trouble that B200/g7e didn't, that's a red flag worth capturing.

---

## TP layout — NVSwitch permits both high-TP AND TP1

On B200/NVSwitch the parent used agg-tp2-x4 as the winner. H200 has the same
NVSwitch interconnect (cheap TP comms) but **141 GB VRAM per GPU** (vs B200's
183 GB). The FP8 checkpoint is ~124 GB → **TP1 is VIABLE on a single H200**
(124 < 141 GB, unlike g7e's 96 GB). So the sweep should test **both** TP1-x8
(zero TP collectives, maximum replica count) and the B200 winner agg-tp2-x4.

| Layout | GPUs used | Note |
|--------|-----------|------|
| **agg-tp2-x4** | 8 | Direct mirror of B200 winner — apples-to-apples comparison. |
| **agg-tp1-x8** | 8 | 8 independent FP8 replicas, **zero TP collectives** — the TP1-on-141GB advantage H200 has over g7e (96 GB forced TP≥2). Test this first. |
| **agg-tp4-x2** | 8 | 2 replicas × TP=4 — more KV headroom per replica. Only if tp2 is VRAM-bound at 130 concurrent. |

> **TP1 is the interesting science question here** — can H200 serve this 120B
> FP8 model at TP1 × 8 replicas and hold the SLO, avoiding TP collectives
> entirely? If yes, the zero-comms-overhead + max-replica-count layout might
> beat TP2. Test tp1-x8 **first**, then tp2-x4 for the B200 mirror.

---

## VRAM Budget (141 GB/GPU)

| Component (FP8, TP=1) | Estimate |
|-----------------------|----------|
| Model weights (FP8, TP=1) | ~124 GB/GPU |
| KV cache + Mamba state (prefix-cache on) | ~12 GB/GPU |
| Activations + chunked-prefill buffers | ~3 GB/GPU |
| **Total** | ~139 GB/GPU (tight but fits) |

| Component (FP8, TP=2) | Estimate |
|-----------------------|----------|
| Model weights (FP8, TP=2) | ~62 GB/GPU |
| KV cache + Mamba state (prefix-cache on) | ~70 GB/GPU |
| Activations + chunked-prefill buffers | ~6 GB/GPU |
| **Total** | ~138 GB/GPU (fits with large KV headroom) |

BF16 (240 GB) at TP=2 = ~120 GB/GPU → **fits comfortably** on H200's 141 GB.
BF16 needs TP≥2 (240 > 141). So the precision comparison on H200 is
**FP8 @ tp1-x8 or tp2-x4 vs BF16 @ tp2-x4** — closer to same-layout than g7e
(where BF16 needed tp4).

---

## Two engines under test (vLLM + SGLang)

The parent B200 run used **vLLM 0.18.1 only**. On H200, test **both** engines
to answer the SGLang question that g7e #20541 raised:

### Engine A — vLLM (B200 mirror, apples-to-apples)

- **Version**: **0.22.1** as PRIMARY candidate (latest stable), with **0.18.1**
  as fallback (parent's validated version).
- **Stage 0 smoke arbiter**: load the FP8 checkpoint on H200, generate 10
  coherent Fin outputs. If 0.22.1 passes smoke (no garble, no crash), use it;
  if it fails, fall back to 0.18.1 and note the regression. Do NOT skip 0.22.1
  this time (B200 appears to have skipped it per final-report "Image Honesty").
- **Standard H200 path**: use DeepGEMM or CUTLASS FP8 backends (sm_90 defaults),
  FlashInfer attention. No Triton-only flags required.

### Engine B — SGLang (the g7e/sm_120 winner, re-tested on standard H200)

- **Version**: latest stable cu13 nightly (or pinned SGLang 0.5.6.post2+ if
  nightly is unstable).
- **Why test SGLang on H200**: g7e #20541 showed SGLang with radix cache +
  Triton FP8 backends worked cleanly on sm_120 Blackwell PCIe. On H200/sm_90,
  SGLang gets the **standard DeepGEMM/CUTLASS path** (no workarounds) — does
  SGLang's radix cache + standard FP8 kernels beat vLLM on this hybrid Mamba2
  model where vLLM prefix-cache was 0% on B200?
- **Specific SGLang questions** (from g7e #20541 findings):
  - Does SGLang **radix cache** engage on the ~3,050-token shared header and
    deliver the TTFT win that vLLM's automatic prefix-cache missed on B200?
  - Does SGLang's **DeepGEMM FP8 MoE** path on sm_90 (clean, not Triton-forced)
    match or beat vLLM throughput on this latent-MoE architecture?
  - Does SGLang's **HiCache** (CPU offload) work on H200 NVSwitch (it's blocked
    on g7e PCIe) and materially improve concurrency headroom?

**Engine priority**: run **vLLM first** (parent mirror, direct B200 comparison).
Run SGLang **second** if time permits, as a science question — if SGLang radix
cache solves the 0% prefix-hit problem vLLM had on B200, that's actionable.

---

## Benchmark Phases (staged, cheap-first)

```
H0 ~0.5 hr · H1 ~2 hr · H2 ~1 hr · H3 ~1 hr (SGLang leg, optional)
```

### H0 — Smoke (MUST pass before anything else)

H200 sm_90 is standard-path; H0 is a **confirmation, not exploration**.

- [ ] **vLLM 0.22.1 smoke** (primary): FP8 loads on H200, 10/10 coherent Fin
      output, no garble, no crash. If PASS → use 0.22.1. If FAIL → fall back
      to 0.18.1 and note the 0.22.1 regression (version sensitivity is real).
- [ ] **vLLM 0.18.1 fallback smoke** (if 0.22.1 fails): same test.
- [ ] **Prefix-cache flag accepted**: `--enable-prefix-caching --mamba-cache-mode all`
      do not crash. Confirm `enable_prefix_caching=True` in engine config.
- [ ] **FP8 ModelOpt first-load time**: budget ~15 min (same as B200).
- [ ] **H200 NVSwitch topology**: `nvidia-smi topo -m` confirms NVL5 NVSwitch,
      not PCIe-only. All 8 GPUs on the fabric.

### H1 — SLO feasibility @ production shape (vLLM, mirroring B200)

- H1a: **FP8 agg-tp1-x8** (the H200 TP1-viable hypothesis), prefix-cache on,
       `fin-support` @ conc 8 & 130. **Does H200 hold E2E p50 6.5s / p90 9.5s
       at 130 concurrent with TP1 × 8 replicas (zero TP comms)?**
- H1b: **FP8 agg-tp2-x4** (B200 winner mirror), prefix-cache on, `fin-support`
       @ conc 8 & 130. Apples-to-apples comparison to B200 final-report
       (p50 4,685 / p90 8,147).
- H1c: best chunked-prefill value — re-sweep {2048,4096,8192,16384} at conc 130
       on the better of tp1-x8 or tp2-x4. Do NOT assume B200 mnbt=16384 is optimal.
- H1d: **BF16 agg-tp2-x4** for the precision/$ comparison (BF16 fits tp2 on H200).

**Gate**: report sustained-concurrency-at-SLO. Compare H200 numbers directly to
B200 final-report. Quantify prefix-cache hit rate (B200 was 0% at TP1/TP2).

### H2 — Knobs that earn their place (vLLM, only if H1 holds)

- Full `fin-support` concurrency sweep [8,32,128,512].
- Spec-decode (n-gram + MTP) acceptance @ temp=1.0 — re-test on H200/vLLM to
  confirm whether the B200 "blocked on 0.18.1" finding still holds on 0.22.1
  or if the Mamba2 graph-capture bug is fixed. If blocked, skip with note.
- FP8 KV on/off — measure (B200 found kv-dtype flag is a no-op on ModelOpt FP8).
- Attention backend: TRITON_ATTN vs FlashInfer (B200 found FlashInfer ~10% faster).

### H3 — SGLang leg (OPTIONAL, science question)

**Only if H1/H2 vLLM complete and time permits.**

- H3a: **SGLang FP8 agg-tp2-x4**, radix cache on, `fin-support` @ conc 130.
       **Does SGLang radix cache deliver >0% hit rate on the shared header?**
- H3b: **SGLang HiCache** (CPU offload) on/off — measure concurrency headroom delta.
       HiCache requires NVSwitch P2P (works on H200, blocked on g7e PCIe).
- H3c: SGLang spec-decode (EAGLE/MTP) acceptance — g7e #20541 showed 0.81
       accept rate on SGLang/sm_120; re-measure on H200/sm_90.

**Gate**: if SGLang radix cache solves the 0% prefix-hit problem, quantify the
TTFT improvement and cost delta vs vLLM. If SGLang is slower or same hit rate,
note and move on.

---

## Infrastructure Changes

- **Separate blueprint**: `domains/gpu-serving/blueprints/fin-rag-answer-h200/`
  (own Terraform state; do not entangle with B200 or g7e blueprints).
- **Reuse cluster + FSx**: same EKS cluster `qwen3-next-bench-eks-cluster`
  (us-east-2, `fin-rag-b200` kubectl context), same FSx PVC `vllm-qwen3-fsx-pvc`
  holding the staged FP8 ~124 GB + BF16 ~240 GB weights. FSx is in us-east-2c
  → **cross-AZ bulk copy to local NVMe** (same pattern as B200/g7e).
- **H200 launch**: p5e.48xlarge SPOT in **us-east-2a** (subnet-0fced510ea62b874e).
  Verify live spot rate before launching (marked TODO above).
- **AMI**: amazon-eks-node-al2023-x86_64-nvidia-1.32 (AL2023, Fabric Manager
  for NVSwitch).
- **Container runtime**: containerd (standard EKS path, not nerdctl bare-metal).

---

## Success Criteria

1. H0 smoke passes (FP8 works on sm_90, prefix-cache flags accepted, 0.22.1 or
   0.18.1 validated).
2. **SLO verdict**: does H200 hold E2E p50 6.5s/p90 9.5s at 130 concurrent?
   Report for both tp1-x8 and tp2-x4. Compare directly to B200 final-report
   (winner: p50 4,685 / p90 8,147).
3. **Headline cost table**: $/1M-token at the SLO operating point, **H200 vs
   B200 vs g7e (if g7e ran)**, using each platform's measured throughput and
   its real spot price. H200 comparison baseline = B200 final-report $0.040/1M
   total tokens @ spot $32.
4. **Prefix-cache hit rate**: measure on H200/vLLM (B200 was 0%). If SGLang
   ran, compare SGLang radix-cache hit rate to vLLM automatic prefix-cache.
5. `lessons.md` captures H200-specific findings (sm_90 maturity, TP1 viability,
   vLLM 0.22.1 vs 0.18.1 delta, SGLang radix cache if tested).

---

## Known Limitations / Risks

1. **H200 141 GB VRAM** — less than B200's 183 GB but still enough for TP1
   (FP8 124 GB fits). TP1 KV headroom is tighter than TP2; may need TP2 for
   high concurrency. Re-sweep.
2. **vLLM 0.22.1 unknown** — B200 appears to have run 0.18.1 only (skipped
   0.22.1 smoke). H200 should TRY 0.22.1 first (latest stable) but must
   fall back to 0.18.1 if 0.22.1 fails smoke. Version sensitivity is real.
3. **Prefix-cache may still be 0%** — B200 automatic prefix-cache was
   non-functional on vLLM 0.18.1 at any TP. Re-validate on H200; do not
   assume it works. If still 0%, that's consistent (Mamba2 PC not yet
   effective), not an H200-specific issue.
4. **Spec-decode blocked on vLLM 0.18.1** — B200 MTP crashed (shape mismatch),
   n-gram crashed (graph-capture). Re-test on H200/vLLM 0.22.1 to see if
   Mamba2 graph-capture + spec-decode is fixed. If still blocked, skip with note.
5. **H200 may simply match B200 throughput** — sm_90 vs sm_100 is a modest
   gap; the headline may be "H200 and B200 tie on throughput/$" rather than
   a clear winner. A clean "H200 is reliable but not cheaper" result is valid.
6. **SGLang leg is optional** — only run if time permits after vLLM H1/H2. If
   skipped, note in lessons.md that SGLang radix-cache question remains open.

---

> Operational artifacts live in `blueprints/fin-rag-answer-h200/`, not in this spec.
> Compare results directly against `blueprints/fin-rag-answer/results/` (B200)
> and `blueprints/fin-rag-answer-g7e/results/` (g7e, if it ran).
