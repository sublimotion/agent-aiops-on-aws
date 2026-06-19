# Spec C — Compile-Cache Strategies (JIT/AOT)

## Status: DRAFT

## Hypothesis

For frontier-model serving where JIT/compile dominates cold start (DeepGEMM autotune, `torch.compile` Inductor cache, Triton kernel cache, FlashInfer kernel selection):

- **Image-baked compile caches** match Modal's measured AOT-cache-HIT floor (~22 s combined torch.compile + warmup on H200 for a 26B-class MoE — see `references/modal-gemma4-aot-h200.md`). Target on EKS: **≤30 s for 26B-class, ≤60 s for Kimi K2.6 FP8 (745 GB MoE TP=8)**.
- **PVC-mounted compile caches** (FSx Lustre, populated by an init job) are within 20% of image-baked. Filesystem latency for cache-file deserialization is the variable.
- **Runtime warmup + node-local promotion** (Smart Cache DaemonSet pattern applied to compile caches) is within 30% of baked for replica N≥2, and matches the no-cache baseline for replica 1.
- **TensorRT-LLM offline engine compilation** is in a different class — engine artifact replaces JIT entirely, but ties cache to engine version and shape set.

## Falsification criteria

- Image-baked caches don't bring 26B-class JIT to under 60 s on EKS (vs Modal's 22 s on the same model) → cache hit path is broken on K8s+PVC, mechanism doesn't transfer.
- Cache portability fails across hardware SKUs the deployment actually uses → cache becomes per-SKU, multiplies storage cost, may not be worth it.
- vLLM cache invalidation is too aggressive (every minor version bump invalidates) → operational tax outweighs cold-start benefit.

## Reference floor (measured on Modal H200, 2026-05-23)

For Gemma-4-26B-A4B-it (26B MoE, TP=1, MTP speculative, vLLM 0.21.0) with AOT compile cache HIT on Modal H200:

| Stage | Time |
|---|---|
| Backbone Dynamo bytecode | 5.02 s |
| Backbone AOT compiled-graph load | **8.52 s** |
| Backbone profiling/warmup | 5.34 s |
| Drafter Dynamo + AOT load | 2.82 s |
| **Compile + warmup combined** | **~22 s** |
| CUDA graph capture (51 sizes) | ~100+ s (NOT cached, runtime-only) |

Cache file paths captured: `/root/.cache/vllm/torch_compile_cache/<config_hash>/rank_0_0/{backbone,eagle_head}` + `/root/.cache/vllm/torch_compile_cache/torch_aot_compile/<aot_hash>/rank_0_0/model`. Format: `compile_cache_save_format='binary'`.

See full breakdown in `references/modal-gemma4-aot-h200.md`.

## Why this matters

Memory: GLM-5 first-start is ~15 min (DeepGEMM JIT + torch.compile + CUDA graphs). This stage is **completely missing** from every published cold-start taxonomy (AWS Labs, ScaleOps, Modal). For any frontier model, JIT is now the dominant stage and the lab's most novel contribution is a rule for when each cache strategy wins.

This is also the spec where the Lila critique applies in reverse: they don't mention compile caching at all, despite serving 49 models including DeepSeek V4 (13B-A284B MoE) and Nemotron Super 120B which both have substantial JIT cost.

## Stage-budget claim

Anchored against Modal's H200 measured floor (22 s compile+warmup, see Reference floor above). EKS targets are within 20-30% bands of Modal floor due to PVC mount overhead and CUDA graph capture variance.

| Stage | Baseline (no cache) | Image-baked | PVC-mounted | Runtime warmup + promote (replica N≥2) | Why |
|---|---|---|---|---|---|
| Node provision | 60-120 | 60-120 | 60-120 | 60-120 | unchanged |
| Image pull | 60-120 | 90-180 (image grew by 1-5 GB cache) | 60-120 | 60-120 | bake adds cache GB |
| Container start | 5-10 | 5-10 | 10-15 | 10-15 | PVC mount or daemon hook |
| Model load | unchanged | unchanged | unchanged | unchanged | Spec B's domain |
| **JIT compile + warmup** (26B-class) | 90-180 | **22-30** (target Modal-equiv) | **22-40** | 90-180 (1st), 22-40 (Nth) | Modal-floor anchored |
| **JIT compile + warmup** (Kimi K2.6 FP8 TP=8) | 600 | **40-60** (target) | **50-90** | 600 (1st), 50-90 (Nth) | extrapolated; needs measurement |
| **CUDA graph capture** | 30-100+ | 30-100+ (NOT cached) | 30-100+ (NOT cached) | 30-100+ (NOT cached) | runtime-keyed by shapes |
| First token | 1-5 | 1-5 | 1-5 | 1-5 | unchanged |

Replica index: 1st replica for first three columns; **Nth replica** for warmup-and-promote.

**CUDA graph capture is the un-cached residual.** Even with AOT cache HIT, capturing 51 cudagraph_capture_sizes adds 30-100+ s. A practical optimization independent of caching is reducing `cudagraph_capture_sizes` to actual production batch sizes. Worth a separate cell.

## Matrix

| Axis | Values |
|------|--------|
| Models | (1) Qwen3-8B (low JIT cost — control), (2) GLM-5-FP8 (high DeepGEMM JIT), (3) Kimi K2.6 (large MoE, torch.compile-heavy), (4) Nemotron Super 120B |
| Cache type | (a) `TORCHINDUCTOR_CACHE_DIR`, (b) `~/.triton/cache`, (c) DeepGEMM autotune table, (d) all combined |
| Delivery | (1) baseline (no cache), (2) image-baked, (3) PVC-mounted from FSx, (4) runtime warmup with node-local promotion via DaemonSet, (5) TensorRT-LLM offline engine (only where supported) |
| Hardware | g7e (Blackwell PCIe), p6-b300 (Blackwell NVSwitch). Cache portability is hardware-keyed; both must be tested. |

~24 prioritized cells. Models 2 and 3 are the headline cells — they're the ones with multi-minute JIT today.

## Baseline

`--enforce-eager=false`, no compile cache, vLLM defaults. Time from pod-Running to first-100-tokens-streamed (catches CUDA graph capture too).

## Measurement

Reuse `shared/cold_start_harness.py`. Additional instrumentation in vLLM:
- `VLLM_LOGGING_LEVEL=DEBUG` to capture `torch.compile` and Inductor stage times.
- Triton cache hit/miss counters (Triton 3.x exposes these).
- DeepGEMM tuning JSON: record whether tuning ran (cache miss) or loaded (cache hit).
- CUDA graph capture time (vLLM logs `Capturing CUDA graphs`).

For PVC-mounted variants: also record PVC mount latency and read throughput during cache load.

## Fixtures

- Reuse `glm5-fp8`, `kimi-k2.6-speculative`, `nemotron-super-120b` blueprints.
- Build a parallel "cache-baker" CI job that runs a model once on each hardware SKU and snapshots the cache directories.
- For PVC variant: deploy an FSx Lustre filesystem in the `ai-infra` namespace, populated by an init job.
- For TRT-LLM variant: only Qwen3-8B (small, well-supported in TRT-LLM); skip GLM-5/Kimi where TRT-LLM support is limited.

## Rule the experiment would produce

> **Compile-cache strategy by deployment shape**:
> - **Shape-stable, single-SKU serving** (e.g., always B300, fixed batch sizes): bake compile caches into image. Accept +1-5 GB image bloat; cold start drops to within 30% of Modal's measured 22 s floor.
> - **Multi-SKU fleet**: per-SKU cache directories on a shared FSx PVC, mounted read-only. CI populates per-SKU on cache miss. No image bloat; cold start drops to within 50% of Modal floor.
> - **High image churn** (frequent vLLM bumps): runtime warmup + node-local promotion via DaemonSet. Replica-1 pays full JIT; replica-N matches PVC-mounted timing. DaemonSet syncs node-local → FSx → S3 on a schedule.
> - **Latency-critical, narrow shape set**: TensorRT-LLM offline engine. Loses vLLM's flexibility; gains predictable startup.
> - **Always trim `cudagraph_capture_sizes`** to actual production batch sizes. vLLM default = 51 sizes (1..512), capture takes 30-100+ s and is NOT cached. Cutting to ~10 sizes reclaims 60-80% of this stage independent of cache strategy.
> - **All cache artifacts must be KMS-signed** before mount/load (cache poisoning is a real attack surface).
> - **Cache key includes**: vLLM version, CUDA toolkit version, Python version, model config hash, hardware SKU, attention/MoE backend, compile_ranges_endpoints. Modal hash example: `a6f06ab438` (vLLM compile cache) + per-submodel AOT hashes (`39e7378ed5...`, `a5fe6da851...`). Mismatch = rebuild, never silently use stale.

## Operational lessons (captured from Modal H200 reference run)

These transfer to any vLLM deployment, regardless of cache strategy:

1. **`compile_cache_save_format='binary'`** is the format AOT cache files use. Image-baked / PVC variants must preserve binary format (no text re-encoding).
2. **`--safetensors-load-strategy=prefetch`** is a vLLM flag we should explicitly enable on FSx Lustre. Modal's 9P auto-disabled it; FSx is recognized so it should activate, but verify.
3. **`fast_moe_cold_start: False`** is in vLLM `compilation_config` — unexplored knob that may further reduce MoE cold start. Worth a separate variant cell.
4. **flashinfer sampling JIT** (specifically `top_k_top_p_sampling_from_logits`) is a known papercut — Modal hits it, our slim image hits it. Pre-warm strategy: send a dummy completion request during cache-baker CI to populate flashinfer cache, then bake or PVC-mount it.
5. **Modal's `startup_timeout=600s` was insufficient** for production-config vLLM with default 51 cudagraph sizes. EKS readinessProbe `failureThreshold` should be sized for ≥1200 s on frontier MoE.
6. **9P-equivalent on AWS = FSx Lustre** at ~1.9 GB/s. Don't bother trying to beat this with NFS/EFS for the model-weight load stage.

## Out of scope

- Image-pull stage (Spec A).
- Model-weight load (Spec B).
- The first-token warmup itself (irreducible CUDA context init).
- vLLM internals beyond cache directories — we don't modify vLLM, only configure it.

## Cost estimate

~$1,500-3,000. The cache-baker CI runs are full cold starts on B300 (expensive), but each run produces a reusable artifact.

## References

- **Modal H200 measured AOT-cache-HIT floor** (~22 s combined): `references/modal-gemma4-aot-h200.md`. The publishable anchor for "what's reachable" on this stage.
- vLLM `torch.compile` cache: `VLLM_TORCH_COMPILE_CACHE_DIR` (vLLM docs)
- vLLM compilation_config (CUDAGraphMode, AOT cache, MoE knobs): observed in Modal log as `compilation_config={...}` in EngineCore init line
- DeepGEMM autotune: https://github.com/deepseek-ai/DeepGEMM
- Triton kernel cache: `~/.triton/cache` (Triton 3.x)
- Our memory: GLM-5 ~16 min cold start (`b300_kimi_k26_benchmark.md`)
- Our spec-0 measurements: Qwen3-1.7B B300 (~85 s JIT), Kimi K2.6 FP8 B300 (~10 min JIT)
- TensorRT-LLM engine builder: https://nvidia.github.io/TensorRT-LLM/
