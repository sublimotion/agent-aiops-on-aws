# Kimi K2.6 NVFP4 — CuTe-DSL MoE Kernel A/B Requirements

## Status: DRAFT

## Overview

Measure the serving-level impact of the **FlashInfer CuTe-DSL NVFP4 MoE kernel** on
`nvidia/Kimi-K2.6-NVFP4`, single-node Blackwell, holding everything else constant. The only
variable is the MoE GEMM backend: **baseline `flashinfer_trtllm`** (what `kimi-k2.6-nvfp4`
already ran) **vs. the new CuTe-DSL NVFP4 path** (FlashInfer PRs #3448 "CuTe DSL NVFP4 + 4over6
FP16 scoring" and #3645 "per-token NVFP4 for CuTe-DSL MoE", consumed by SGLang #28354).

### Why this experiment exists

A SemiAnalysis post (2026-06, "CUDA MOAT ALERT") claimed **GB200 NVL72 serving cost for the
Kimi architecture dropped 2.5× in <70 days through software alone**, crediting two levers:
1. **A CuTe-DSL rewrite of the NVFP4 MoE kernel** (FlashInfer #3448 reports **2.24–4.12× geomean**
   GEMM speedup vs the prior CUDA NVFP4 kernel).
2. **Wide expert-parallelism across the NVL72 72-GPU copper backplane.**

We can reproduce **lever 1 on hardware we have** (single-node B200/B300). **Lever 2 we cannot**:
forming a 72-rank NVLink EP domain needs NVL72 hardware we don't have, and SGLang's EP+NVFP4
path is upstream-blocked regardless (`kimi-k2.6-nvfp4/lessons.md` "EP on NVFP4" → SGLang #24502
OPEN: `flashinfer_trtllm` MoE runner has no DeepEP fused all-to-all). **This spec scopes lever 1
only** and reports the kernel's contribution to the 2.5× as a standalone, hardware-attainable number.

### Core question

**How much of the claimed serving-cost win comes from the kernel rewrite alone, at our scale?**
A 2–4× *GEMM* speedup is not a 2–4× *serving* speedup — the MoE GEMM is one term in a request
that is **prefill-compute / KV-capacity / BW co-bound** (`kimi-k2.6-nvfp4/lessons.md`: B200 c512
was partly KV-capacity-throttled, B300 lifted peak +36–43%). The kernel only moves the needle to
the extent MoE-linear GEMM is on the critical path. **This spec measures that fraction.** A null
or small serving delta despite a large microbenchmark delta is a valid, publishable result.

## Components

### 1. Compute
- **Reuse the `kimi-k2.6-nvfp4` substrate verbatim** — same cluster, nodegroup, AZ, AMI, bring-up.
  - **Cluster**: existing EKS `qwen3-next-bench-eks-cluster` (us-east-2). kubectl context already set.
  - **Instance**: **p6-b200.48xlarge** spot (8× B200, sm_100, NVSwitch) via nodegroup
    `ai-infra-use2-b200-spot` (max=1, desired=0; scaling to 1 is the ~$18/hr billable gate).
  - **AZ**: us-east-2b (use2-az2, subnet-03d03f1fb8d62d6a5).
  - **AMI**: managed by the nodegroup (EKS-optimized AL2023+NVIDIA, `nodeadm` bootstrap).
  - **Node targeting**: after scale-up add `blueprint=kimi-k2.6-cutedsl-moe` AND
    `nvidia.com/gpu.present=true` (no GFD/NFD on this cluster — `kimi-k2.6-nvfp4` L1).
- **The winning parallelism layout is already known — do NOT re-sweep it.** `kimi-k2.6-nvfp4`
  established **TP4+DP2** as the single-node winner (+19–25% over TP8). Run the kernel A/B on
  **TP4+DP2** (the operating config) and additionally on **TP8×1** (the simplest, most-isolated
  layout) to confirm the kernel delta is layout-independent. Do not re-explore TP2/disagg/EP.
- **B200 is primary.** Optionally repeat the single A/B point on **B300** (us-west-2) only if the
  B200 result is ambiguous (kernel delta within noise of the KV-capacity regime) — B300's 5.6×
  larger KV pool can move the run off the capacity wall so a compute-side kernel delta becomes
  visible. Decide from the B200 bottleneck classification, do not pre-provision.
- **Scaling**: single node, fixed. This is an A/B, not an autoscaling deployment.

### 1a. GPU & NCCL Pre-Flight
Standard Stage 4a. TP4/TP8 on B200 NVSwitch — mature path (NOT the broken Blackwell-PCIe sm_120
topology from `devstral-sera`). **NVFP4 requires the `-cu130` SGLang image on BOTH B200 (sm_100)
AND B300 (sm_103)** — cu129 crashes `ModuleNotFoundError: cutlass` on the FP4 path (`kimi-k2.6-nvfp4`
L7). The `-cu130` requirement is driven by the NVFP4 cutlass DSL, not the GPU arch. Arm B may need
an even newer image/branch carrying SGLang #28354 — pin at deploy (see §2).

### 2. Model
- **Model ID**: `nvidia/Kimi-K2.6-NVFP4` (modelopt ckpt). **NOT** `RedHatAI/...` (load bug, SGLang
  #25331 OPEN). Identical weights to `kimi-k2.6-nvfp4` — reuse the staged `/mnt/nvme/models/...`
  copy if the node is warm.
- **Format**: NVFP4 (MoE-linear weights+activations, modelopt v0.44.0; group_size=16, fp8 KV by default).

#### THE SINGLE VARIABLE — MoE kernel backend (A vs B)
This is the entire experiment. Everything else (model, TP layout, workload, context, cache,
SLOs) is held byte-identical between the two arms.

- **Arm A — baseline `flashinfer_trtllm`**: the vendor-precompiled TRT-LLM-gen NVFP4 MoE path
  that `kimi-k2.6-nvfp4` ran. This is the control; its numbers should reproduce that blueprint's
  (TP4+DP2 ≈ 3,138 tok/s @ c512 on B200) — **confirm reproduction before trusting any delta.**
- **Arm B — FlashInfer CuTe-DSL NVFP4 MoE**: the rewritten kernel (FlashInfer #3448 + #3645),
  surfaced in SGLang via #28354, which routes `flashinfer.nvfp4_quantize(..., backend="cute-dsl")`
  and the per-token NVFP4 / "4over6" flow into the CuTe-DSL MoE wrapper.
  - **DEPLOY-TIME PREREQUISITE — the exact toggle is not yet pinned.** As of 2026-06-23 SGLang
    #28354 is a **DRAFT** with a failing reload test (post-`update_weights` output diverges from
    baseline) and depends on FlashInfer #3645. **Before benchmarking, determine the current
    activation mechanism** — likely a combination of: a FlashInfer version that includes #3448+#3645,
    an SGLang build/branch carrying #28354 (or its merged successor), and an env-var / flag selecting
    `backend="cute-dsl"` plus the `NVFP44Over6Config` env vars (#3448 gates 4over6 + MAE/MSE scoring
    by env). Run `mdc prs kimi-k2.6` and check the FlashInfer/SGLang PRs for the merged form. If
    #28354 is still draft/broken at deploy, **smoke-test correctness first** (see Stage 5) and treat
    Arm B as experimental.
  - **4over6 scoring mode is a sub-lever, not the headline.** #3448 exposes strict full-dequant
    (fp32) scoring vs FP16-domain candidate scoring with MAE/MSE metrics. The headline A/B is
    backend (trtllm vs cute-dsl); 4over6 mode is a secondary sweep IF Arm B shows a serving delta
    AND output quality holds (see correctness gate).
- **Deployment Card**: `mdc get kimi-k2.6 --engine sglang` + `mdc prs kimi-k2.6` before deploy.

### 3. Networking
- Existing cluster VPC. Serving + bench-runner pods: `hostNetwork: true`, tolerate
  `ai-infra/b200=true:NoSchedule` (NOT `nvidia.com/gpu` — `kimi-k2.6-nvfp4` L2). In-cluster only,
  no public ingress. Internet-facing pods (HF/pip) need `dnsPolicy: Default` (L4).

### 4. Storage
- **Weights**: reuse `/mnt/nvme/models/kimi-k26-nvfp4` (RAID-0 the 8× instance-store disks on a
  fresh node; root EBS too small — `kimi-k2.6-nvfp4` L4). `hf download` with `HF_HUB_DISABLE_XET=1`
  + `HF_HUB_ENABLE_HF_TRANSFER=0` + resilient retry loop as PID 1 (L5).
- **No HiCache tiering needed for the A/B** — KV tiering is held OFF (or identically ON) in both
  arms so it isn't a confound. The kernel A/B is about GEMM, not KV residency.

### 5. Development Environment
- None. Headless benchmark.

## Non-Requirements (explicitly out of scope)
- **Wide-EP / NVL72 reproduction** — the other half of the SemiAnalysis claim. Not attainable on
  single-node B200/B300, and SGLang EP+NVFP4 is upstream-blocked (#24502). This spec is kernel-only.
- **Parallelism re-sweep** — TP4+DP2 already won (`kimi-k2.6-nvfp4`). Run the A/B on the known
  winner + TP8 control; do not re-explore TP2/EP/disagg.
- **P/D disaggregation** — `kimi-k2.6-nvfp4` measured 3.8× regression single-node; out of scope.
- **Spec decode (EAGLE3)** — orthogonal decode-latency lever; would confound the MoE-GEMM A/B. OFF in both arms.
- **vLLM arm** — the CuTe-DSL kernel lands in SGLang via FlashInfer; this A/B is SGLang-only. (vLLM
  consumes FlashInfer too, but #28354 is the SGLang wiring under test. A vLLM CuTe-DSL A/B is a
  possible follow-up, not this spec.)
- **Custom kernel authoring** — we measure the upstream kernel, we don't write one.
- **g7e / H200** — cannot run NVFP4 Blackwell kernels / cannot hold 520 GB weights.
- HA/DR, multi-region, production monitoring.

## Security Requirements
- Spot, ephemeral. Verify node IAM **S3 write** before relying on result upload (`kimi-k2.6-nvfp4`
  L10 — `kubectl cp` to workstation as the fallback).

## Cost Considerations
- B200 spot ~$18/hr. The A/B (two arms × {TP4+DP2, TP8} × knee-region concurrency) should fit in
  well under a day of node time, especially if the node is warm from `kimi-k2.6-nvfp4`.
- **The output number**: Δ $/Mtok between Arm A and Arm B at the operating point. Frame against the
  SemiAnalysis 2.5×: "the kernel alone delivers X× of the serving throughput at our scale; the
  remainder of their 2.5× is attributable to wide-EP we can't reproduce here."

## Known Limitations
- **Microbenchmark speedup ≠ serving speedup.** #3448's 2.24–4.12× is a GEMM-level geomean over
  scanned M/K shapes. At serving time the MoE GEMM shares the request with attention/MLA, prefill
  chunking, KV access, and collectives. Expect the serving delta to be a *fraction* of the GEMM
  delta — quantifying that fraction is the point. Do not present the microbenchmark number as the
  serving result.
- **Arm B may be unstable.** SGLang #28354 is draft with a known post-reload correctness failure as
  of 2026-06-23. The reload bug is about weight updates (not relevant to a static benchmark), but it
  signals the path is fresh. **Correctness-gate Arm B** (output parity vs Arm A) before trusting perf.
- **Regime-dependence.** If the B200 run is KV-capacity-bound at the knee (as `kimi-k2.6-nvfp4`
  found at c512), a compute-side kernel win may be invisible — the bottleneck is elsewhere. The DCGM
  roofline classification (below) decides whether a null is "kernel doesn't help" vs "kernel help
  masked by a capacity wall." This is exactly why the optional B300 point exists.
- **`-cu130` image required on BOTH B200 and B300** (NVFP4 cutlass DSL; cu129 fails
  `ModuleNotFoundError: cutlass` — `kimi-k2.6-nvfp4` L7). Driven by NVFP4, not GPU arch. Arm B may
  need an even newer image/branch carrying #28354 — pin at deploy.
- **FlashInfer cubin symlink race** on every cold start — pre-clear in the launch wrapper (L-Stage5 below).
- Check `mdc prs kimi-k2.6` for NVFP4 MoE / CuTe-DSL PRs that supersede #3448/#3645/#28354.

## Verification Criteria

### Stage 0 — Carryover Audit (spec-design gate)
- [ ] Ran `carryover-auditor` against this spec, scanning `kimi-k2.6-nvfp4/lessons.md` (the parent),
      `qwen3-235b-speculative/lessons.md`, `kimi-k2.6/lessons.md`, and B200/B300 infra memory.
- [ ] Carried lessons reflected as requirements: TP4+DP2 is the known winner (don't re-sweep);
      EP+NVFP4 blocked (#24502) so wide-EP is out of scope not "to try"; `-cu130` image (L7);
      FlashInfer cubin race (L-Stage5); RadixAttention byte-identical prefix for cache reproduction;
      ECC gate on `volatile.total` not aggregate (L3); node needs both labels + `ai-infra/b200`
      toleration + RAID NVMe + `dnsPolicy: Default` (L1/L2/L4); SGLang `--enable-metrics` else /metrics
      404 (L8); verify S3 write early (L10).
- [ ] No P0 carryover gap remains.

### Stage 0c — Serving-Config Resolver (fail-closed)
- [ ] `python3 standards/serving-commons/resolver/validate-serving-config.py --sidecar blueprints/kimi-k2.6-cutedsl-moe/benchmark.yaml --corpus-root .` exits 0
- [ ] NVFP4 group_size=16 (not block_n=128) → fp8-moe-tp-divisibility N/A (parent benchmark.yaml confirmed).
- [ ] Every `prior-failure:*` finding reviewed and noted.

### Stage 4-pre — Node bring-up gate (run after scale-up, BEFORE serving)
Identical to `kimi-k2.6-nvfp4` Stage 4-pre (this cluster has no GFD/NFD + custom taint):
- [ ] Node labeled BOTH `blueprint=kimi-k2.6-cutedsl-moe` AND `nvidia.com/gpu.present=true`; verify
      `allocatable.nvidia.com/gpu == 8`.
- [ ] Pods tolerate `ai-infra/b200=true:NoSchedule`.
- [ ] NVMe RAID-0 mounted at `/mnt/nvme` with `mountPropagation: HostToContainer` on every pod (L12).
- [ ] Internet-facing pods use `hostNetwork: true` + `dnsPolicy: Default`.
- [ ] ECC gate: `volatile.total==0` + clean `remapped_rows` (NOT lifetime aggregate).

### Stage 4a — GPU Health
- [ ] ECC enabled, 0 *volatile* uncorrected; `remapped_rows.{pending,failure}==No`; no Xid; thermals < 85°C idle.
- [ ] NCCL all-reduce BW recorded for TP4 and TP8 (NVSwitch path).

### Stage 5-pre — Arm B activation mechanism (resolve BEFORE any launch)
- [ ] **DEPLOY-TIME GATE — determine the current CuTe-DSL activation mechanism as of deploy date:**
      - `mdc prs kimi-k2.6` + check FlashInfer #3448/#3645 merge status and SGLang #28354 (DRAFT as of
        2026-06-23). If merged, record the merge SHA + the exact flags. If still draft/broken, look for
        a successor PR or SGLang release notes mentioning "CuTe-DSL" / "NVFP4 4over6".
      - Document the exact toggle (likely a `backend="cute-dsl"` selector + `NVFP44Over6Config` env vars
        per #3448). The A/B is invalid if Arm B silently falls back to the trtllm path.
- [ ] Smoke-test that Arm B actually launches, serves a single request, AND is provably running the
      CuTe-DSL backend (log line / env confirming `backend=cute-dsl`) — not a silent trtllm fallback.

### Stage 5 — Serving Stack + Arm B correctness gate
- [ ] Both arms: `/health` 200; single `/v1/completions` returns valid output; no CUDA OOM; TP4+DP2 and TP8 both load.
- [ ] Cold start recorded per arm (Arm B may JIT-compile CuTe-DSL kernels on first run — note the delta).
- [ ] **FlashInfer cubin symlink race pre-clear** in the launch wrapper (fires every cold start):
      `find .../flashinfer_cubin/cubins -name trtllmGen_bmm_export -exec rm -rf {} + 2>/dev/null; exec <server>`.
- [ ] **SGLang `--enable-metrics` set**; `curl localhost:30000/metrics` returns `sglang:*` (else TTFT data lost — L8).
- [ ] **ARM B CORRECTNESS GATE (blocks the perf comparison) — with concrete pass criteria.** The
      CuTe-DSL path is a numerics change (4over6 scoring, FP16-domain candidates), so it MUST be proven
      output-faithful before any throughput claim. A faster-but-wrong kernel is not a win — this is the
      exact failure mode #28354's draft reload test surfaces. Protocol:
      - Run **50 fixed prompts** (coding-agent shapes: tool calls, reasoning, 31K context) at
        **temperature 0.0** through Arm A and Arm B. Compute per-pair token-level exact-match fraction.
      - **PASS**: ≤3 / 50 outputs diverge beyond ~2% token mismatch (allows for expected FP noise).
      - **BORDERLINE (4–10 divergences)**: report both arms with a loud caveat ("Arm B showed N%
        output divergence; any throughput gain may be a quality tradeoff, not a free win").
      - **FAIL (>10 / 50, >20%)**: Arm B is numerically unstable → report Arm A baseline only, note
        "CuTe-DSL rejected at correctness gate (cf. SGLang #28354 known divergence)". Do NOT report Arm B perf.
      - **Manually spot-check divergences** — triage cosmetic (whitespace/casing) vs semantic (wrong
        tool call, broken reasoning). Semantic divergences are higher severity than token-level noise.

### Stage 6a — MoE GEMM Microbenchmark (kernel-level baseline, REQUIRED)
The GEMM-level delta is the numerator; the Stage 6 serving delta is the denominator; their ratio
is the headline finding. Run this BEFORE the serving sweep so a serving null can be attributed
("kernel didn't win at GEMM level" vs "GEMM win masked by a non-MoE bottleneck").
- [ ] Run FlashInfer's #3448 benchmark harness (if shipped in-container) OR a minimal MoE GEMM script
      on the **actual Kimi K2.6 MoE shapes** (K=hidden, N=`moe_intermediate_size`=2048, 384 experts,
      top_k=8) at the operating batch (~c512 → M≈16K–32K tokens).
- [ ] Record Arm A (`flashinfer_trtllm`) vs Arm B (CuTe-DSL) GEMM-level speedup (TFLOPS or tok/s).
- [ ] Cross-check against #3448's reported **2.24–4.12× geomean** — if our measured GEMM delta is
      outside that band, flag "kernel config mismatch" (wrong block size / wrong 4over6 mode) and
      revalidate Arm B setup before the serving sweep.

### Stage 6 — Benchmark (the A/B)

**Held constant across both arms** (reuse the parent blueprint's sidecar workloads verbatim):
- Workload cards: `concurrency-sweep` (knee), `coding-agent` (primary shape), `shared-prefix-multitenant`
  (74% cache reproduction) — same `catalog_id` + overrides as `kimi-k2.6-nvfp4/benchmark.yaml`.
- Context 31,404 avg, output 1,024, 74.1% target prefix hit, byte-identical shared prefix.
- TP layout: run the full A/B on **TP4+DP2** (operating) and a confirmation point on **TP8×1**.
- KV tiering OFF (or identically configured) in both arms. Spec decode OFF. Same context-length 65536.

**Required measurements (per arm, per layout):**
- [ ] **Arm A reproduces `kimi-k2.6-nvfp4`** (TP4+DP2 ≈ 3,138 tok/s @ c512 ± noise). If it doesn't,
      the environment drifted — fix before trusting any A/B delta (verify-before-assert).
- [ ] Concurrency sweep across the knee region (c128–c1024) at fixed 31,404 context, both arms.
- [ ] Δ aggregate tok/s and Δ TTFT (p50/p99) between Arm A and Arm B at c256 and c512,
      client-measured (NOT the engine `gen_throughput` gauge — it overstates; `kimi-k2.6-nvfp4` Stage 6c).
- [ ] **HEADLINE: serving speedup (Arm B / Arm A) vs GEMM speedup (Stage 6a)** — e.g. "GEMM 3.2×,
      serving 1.4× → 56% of the kernel win reached serving; 44% eaten by non-MoE bottlenecks (regime: ___)."
      This ratio is the deliverable; a bare Δ tok/s without the GEMM baseline can't answer the core question.
- [ ] Δ $/Mtok (input and output) between arms at the operating point.
- [ ] Error rate < 0.1% both arms; no timeouts.

**Bottleneck classification at the knee (decides if a null is real or masked):**
- [ ] DCGM exporter with PROF CSV + `privileged: true`; `curl localhost:9400/metrics | grep DCGM_FI_PROF_DRAM_ACTIVE` non-empty BEFORE the sweep (else classification is `[inferred]` — `kimi-k2.6-nvfp4` L8).
- [ ] At the knee, capture `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` (compute), `DCGM_FI_PROF_DRAM_ACTIVE`
      (HBM BW), `sglang:token_usage` (KV residency), `sglang:num_queue_reqs`. Classify:
  - **Prefill-compute-bound** (`PIPE_TENSOR_ACTIVE` high) → the MoE GEMM IS on the critical path; Arm B should help here. This is where the kernel win is real.
  - **KV-capacity-bound** (`token_usage` ~100% + queue rising) → kernel win is masked; flag the null as capacity-masked, consider the B300 point.
  - **HBM-BW-bound** (`DRAM_ACTIVE` >~80%) → kernel changes GEMM compute, not BW; small serving delta expected.
- [ ] **TTFT-share check (mandatory before any bound claim — `benchmark-analysis.md`):** compute
      TTFT share = median(TTFT) / median(E2E) from the Prometheus histograms. High share (>20%) means
      prefill is on the critical path (where the MoE GEMM lives) — never call a run "decode/BW-bound"
      without checking TTFT share first. The parent found ~26% prefill-bound only after this check.
- [ ] Record the regime alongside the A/B delta in lessons — a kernel that wins the GEMM but not the
      serving run because the run is capacity-bound is the expected, honest outcome to document.

**Optional B300 point (only if B200 result is capacity-masked):**
- [ ] Repeat the single A/B point (TP4+DP2, c512) on B300 (us-west-2), where the 5.6× larger KV pool
      moves the run off the capacity wall → a compute-side kernel delta becomes visible if it exists.

### Stage 7 — Readiness Audit
- [ ] Arm A reproduced the parent baseline; Arm B passed the correctness gate.
- [ ] Δ tok/s, Δ TTFT, Δ $/Mtok recorded per layout, with the bottleneck regime that explains the magnitude.
- [ ] **The framed conclusion**: "of SemiAnalysis's claimed 2.5× Kimi serving-cost win, the CuTe-DSL
      MoE kernel alone accounts for ___× at single-node scale (regime: ___); the remainder is
      attributable to NVL72 wide-EP, which is not reproducible on the hardware available."
- [ ] All criteria above checked and recorded.

---

> Operational artifacts (lessons, results, deployment notes) belong in
> `blueprints/kimi-k2.6-cutedsl-moe/`, not in this spec. The parent
> `blueprints/kimi-k2.6-nvfp4/` is the control-arm reference — reuse its manifests and sidecar.
