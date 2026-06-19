# Reference: Modal Gemma-4-26B-A4B-it on H200 — AOT compile cache HIT measurement

**Date captured**: 2026-05-23
**App**: `ap-AVsLFbflcy7OBtVot58kh1` (`example-vllm-inference`)
**Source**: `modal app logs` snapshot, saved as `modal-gemma4-aot-h200.log`
**Why this matters**: this is the empirical ceiling for what an AOT compile cache HIT achieves on a frontier-class MoE model. Spec C's hypothesis ("compile cache cuts JIT stage by ≥80%") is anchored against Modal's measured **22 s combined compile+warmup floor**.

## Setup

- **Model**: `google/gemma-4-26B-A4B-it` (revision pinned)
- **Drafter**: `google/gemma-4-26B-A4B-it-assistant` (MTP, num_speculative_tokens=4)
- **Hardware**: 1× H200 (NVIDIA H200, ~141 GB HBM)
- **vLLM**: 0.21.0
- **Filesystem for checkpoints**: 9P (Modal proprietary network FS)
- **vLLM args**:
  ```
  --async-scheduling --no-enforce-eager
  --tensor-parallel-size 1
  --reasoning-parser gemma4 --tool-call-parser gemma4
  --speculative-config '{"method":"mtp", num_spec_tokens=4}'
  ```
- **Compilation config** (default): `mode=VLLM_COMPILE`, `backend=inductor`, `cudagraph_mode=FULL_AND_PIECEWISE`, `cudagraph_capture_sizes=[1,2,4,...,512]` (51 sizes), `compile_ranges_endpoints=[8192]`

## App-level timeline (Modal-side, EDT)

| Event | Time | Δ from app create |
|---|---|---|
| App created | 08:12:47 | 0:00 |
| First boot — flashinfer-ninja crash | ~08:25:59 | 13:12 |
| Second boot — APIServer banner | 08:26:51 | 14:04 |
| App stopped (Modal `startup_timeout=600` expired) | 08:30:03 | 17:16 |

**Modal cold-start overhead from app-create to first vLLM banner: ~13-14 min.** This includes container build/pull, 9P mount setup, Python startup. Modal's L0/L1/L2 lazy-loading stack handles this; we don't see breakdown inside it.

## Server-side stage budget (UTC, second boot)

T0 = 12:26:51 UTC = APIServer banner (`vllm version 0.21.0`)

| Stage | Boundary | Duration | Cumulative |
|---|---|---|---|
| **APIServer init** | 0:00 → 0:32 | **32 s** | 0:32 |
| | banner, arg parsing, arch resolution (Gemma4 + Gemma4MTP), MTP wiring, EngineCore process spawn | | |
| **Engine bootstrap** | 0:32 → 0:41 | **9 s** | 0:41 |
| | parallel_state, sampler init, attention backend selection (TRITON_ATTN forced for heterogeneous head dims) | | |
| **Model weight load** | 0:41 → 1:09 | **28 s** | 1:09 |
| | Backbone 48.07 GiB over 9P: 25.36 s (~1.9 GB/s) | | |
| | Drafter 0.78 GiB: 0.73 s | | |
| | MTP rewiring (4 draft layers → backbone layers 28-29): ~1 s | | |
| **torch.compile (backbone, AOT cache HIT)** | 1:15 → 1:24 | **13.96 s** | 1:24 |
| | Dynamo bytecode transform: 5.02 s | | |
| | AOT compiled graph load (compile range 1-8192): **8.52 s** | | |
| **Initial profiling/warmup (backbone)** | 1:24 → 1:29 | **5.34 s** | 1:29 |
| **torch.compile (drafter, AOT cache HIT)** | 1:30 → 1:32 | **2.82 s** | 1:32 |
| | Dynamo bytecode transform: 0.58 s | | |
| | AOT compiled graph load: 2.20 s | | |
| **Drafter warmup** | 1:32 → 1:32 | **0.01 s** | 1:32 |
| **CUDA graph capture** (truncated) | 1:32 → ≥3:13 | **≥100 s** | ≥3:13 |
| | "Profiling CUDA graph memory: PIECEWISE=48 (largest=500), FULL=48 (largest=500)" at 12:30:04 — capture in flight when timeout fired | | |
| **STOP** (Modal startup_timeout) | 3:12 | — | — |

## Headline numbers — the AOT cache HIT floor

| What | Value |
|---|---|
| **Backbone torch.compile (cached, monitored)** | **13.96 s** |
| **Drafter torch.compile (cached, monitored)** | **2.82 s** |
| **Combined Dynamo + AOT load** | **16.78 s** |
| **+ Initial profiling/warmup** | **+5.35 s** |
| **= Spec C target floor (compile + warmup)** | **~22 s** |

**Bytes-per-second for 9P checkpoint load**: 48.07 GiB / 25.36 s ≈ **1.94 GB/s**. Same speed class as FSx Lustre for AWS-native deployments.

## AOT cache file layout (load-bearing for Spec C variants)

vLLM compile cache:
```
/root/.cache/vllm/torch_compile_cache/<config_hash>/rank_0_0/backbone/
/root/.cache/vllm/torch_compile_cache/<config_hash>/rank_0_0/eagle_head/
```
Hash observed: `a6f06ab438` (config-keyed: model, dtype, attention backend, MoE backend, compile config)

AOT compilation cache (Inductor compiled graphs):
```
/root/.cache/vllm/torch_compile_cache/torch_aot_compile/<aot_hash>/rank_0_0/model/
```
Two distinct hashes (one per submodel):
- backbone: `39e7378ed502eca6caf75d7ea0e86bbac7ed017f786813f4293c9ff6534baa59`
- eagle_head: `a5fe6da851901b54b777a6dc3c5826a34f6c75bd6f3bec85b2d6b95e0ab5194c`

**Cache key dimensions**: model architecture + revision + dtype + attention/MoE backend + compile config + shapes (compile_ranges_endpoints).

**Save format**: `compile_cache_save_format='binary'`.

## What this proves for Spec C

1. **The hypothesis is reachable**: AOT cache HIT compresses backbone JIT from minutes to **8.5 s of cache deserialization**. Spec C's "≥80% reduction" claim is achievable in principle.
2. **Compile + warmup combined floor: ~22 s** for a 26B-class MoE on H200. Spec C variants on EKS should match or come within 20% of this number, otherwise something is wrong with the cache mount path.
3. **CUDA graph capture is NOT in the cache**. The ~100+ s for graph capture remains; it's keyed on dynamic shapes seen at runtime. Smaller `cudagraph_capture_sizes` lists would shorten this stage. Modal default = 51 sizes, which is excessive for most workloads.
4. **Cache hit is FAST when on local disk** (Modal's case). On EKS, cache mounted via PVC over FSx Lustre would add filesystem latency on top of the 13.96 s. Worth measuring.

## What it does NOT prove

- **Cold-cache (first run) compile cost** — we only saw the cache-HIT path. The MISS path (build the AOT cache) is what we need to populate the cache initially. That's a one-time cost paid in CI / a warmup pod.
- **Multi-shape generalization** — this run captured `compile_ranges_endpoints=[8192]`, single endpoint. Real production with varied prompt lengths needs more endpoints, larger cache.
- **Cross-vLLM-version cache portability** — vLLM bumps invalidate the AOT cache key. Modal hides this behind their image build.

## Operational lessons captured

1. **`compile_cache_save_format='binary'` is the format we'd ship** in image-baked / PVC-mounted variants.
2. **`--safetensors-load-strategy=prefetch`** is a vLLM flag that auto-disabled here because 9P isn't recognized. On AWS FSx Lustre we should explicitly enable it.
3. **`fast_moe_cold_start: False`** appears in vLLM compilation_config — *unexplored knob* that may further reduce cold start. Worth a separate cell in Spec C.
4. **flashinfer sampling JIT crashes the same way Modal hits it** as our slim image. Confirms this is upstream papercut, not our environment. Workaround: pre-warm sampling kernels during build, or use a different sampler backend.
5. **CUDA graph capture sizes default to 51** — many of those sizes won't be hit in practice. Trimming to actual production batch sizes is a quick win that doesn't need a cache at all.
6. **Modal default `startup_timeout=600s` is too short** for production-config vLLM with 51 cudagraph sizes. Anything serving frontier MoE needs ≥1200 s.

## Comparison to our own measurements

| Hardware / model | Source | Compile + warmup | Notes |
|---|---|---|---|
| H200 / Gemma4-26B-A4B-it | **Modal (this log, AOT HIT)** | **~22 s** | reference floor |
| B300 / Qwen3-1.7B | our spec-0 (cold cache) | ~85 s | small model, full JIT |
| B300 / Kimi K2.6 FP8 (TP=8) | our spec-0 (cold cache) | ~10 min | 745 GiB MoE, full JIT |
| EKS + PVC AOT cache (planned) | spec-c | TBD | target ~25-30 s match |
| EKS + image-baked AOT cache (planned) | spec-c | TBD | target ~22-25 s match |

## Files

- `modal-gemma4-aot-h200.log` — raw log
- this file — annotated walkthrough
