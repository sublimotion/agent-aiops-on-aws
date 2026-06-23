# Kimi K2.6 NVFP4 — CuTe-DSL MoE Kernel A/B

Spec: `domains/gpu-serving/specs/kimi-k2.6-cutedsl-moe.md`
Parent (control reference): `domains/gpu-serving/blueprints/kimi-k2.6-nvfp4/`

## What this measures

A single-variable A/B on the MoE GEMM backend for `nvidia/Kimi-K2.6-NVFP4`, single-node B200:
- **Arm A** — `flashinfer_trtllm` (the parent's control; reproduce 3,138 tok/s @ c512 TP4+DP2 first).
- **Arm B** — FlashInfer CuTe-DSL NVFP4 MoE (FlashInfer #3448 + #3645; SGLang #28354).

Motivated by the SemiAnalysis "CUDA moat" claim of a 2.5× Kimi serving-cost drop on GB200 NVL72.
This reproduces the **kernel half only** — wide-EP/NVL72 is out of scope (no NVL72 hardware; SGLang
EP+NVFP4 upstream-blocked, #24502). **Headline deliverable: the GEMM-level speedup (Stage 6a) vs
the serving-level speedup (Stage 6) ratio** — how much of a 2–4× GEMM win survives to serving.

## Run order

1. **Stage 4-pre** — scale `ai-infra-use2-b200-spot` to 1; label the node
   `blueprint=kimi-k2.6-cutedsl-moe` AND `nvidia.com/gpu.present=true`; RAID-0 NVMe at `/mnt/nvme`;
   reuse staged weights at `/mnt/nvme/models/kimi-k26-nvfp4` if warm (parent L1/L2/L4).
2. **Observability FIRST** — `kubectl apply -f k8s/observability.yaml`, then SMOKE-TEST the PROF
   scrape before any benchmark:
   ```
   curl -s localhost:9400/metrics | grep DCGM_FI_PROF_DRAM_ACTIVE     # MUST be non-empty
   curl -s localhost:30000/metrics | grep sglang:                      # after serving is up
   ```
   If PROF is empty (driver-580 limitation, parent L8), fall back to engine gauges
   (`sglang:token_usage`, `num_queue_reqs`) and label the regime verdict `[gauge-inferred]`.
3. **Stage 5-pre (Arm B gate)** — `mdc prs kimi-k2.6`; confirm the real CuTe-DSL toggle + a build
   carrying #28354; update the Arm B manifest image + env. Prove the backend is actually CuTe-DSL,
   not a silent trtllm fallback.
4. **Arm A** — `kubectl apply -f k8s/sglang-cutedsl-armA-tp4dp2.yaml`; reproduce the parent baseline.
5. **Correctness gate** — 50 prompts @ t=0 through Arm A and Arm B (`correctness_diff.py`).
   PASS ≤3/50 divergent; FAIL >10/50 → report Arm A only.
6. **Stage 6a (microbenchmark, REQUIRED)** — GEMM-level Arm A vs Arm B on Kimi shapes.
7. **Stage 6 (serving A/B)** — concurrency sweep c128–c1024, both arms, TP4+DP2 then TP8.
   Classify the bottleneck regime at the knee (PROF or gauges) — a serving null under a
   capacity/BW-bound regime is the expected honest outcome.
8. **B300 (optional)** — only if the B200 result is KV-capacity-masked.

## Manifests

| File | Purpose |
|------|---------|
| `k8s/observability.yaml` | DCGM (PROF CSV wiring **fixed** vs parent) + Prometheus |
| `k8s/sglang-cutedsl-armA-tp4dp2.yaml` | Arm A control, operating layout |
| `k8s/sglang-cutedsl-armA-tp8.yaml` | Arm A control, layout-independence point |
| `k8s/sglang-cutedsl-armB-tp4dp2.yaml` | Arm B treatment, operating layout (DEPLOY-TIME toggle gate) |
| `k8s/sglang-cutedsl-armB-tp8.yaml` | Arm B treatment, layout-independence point |
| `k8s/bench-runner.yaml` | Concurrency driver + correctness-diff helper |

> Operational artifacts (lessons, results) land here after the run.
