# Slim Serving Images

Slim variants of the vLLM and SGLang serving containers, used as the **measurement floor** for every spec in this domain. Stock upstream images are documented as a "bloat baseline" but are not the comparison target — measuring against bloated defaults would inflate lazy-loading and decoupling wins beyond what's achievable in production.

## Why these exist

Upstream slim variants don't exist (yet). Recent tag sizes (May 2026):

| Image | Compressed | Notes |
|---|---|---|
| `vllm/vllm-openai:v0.21.0` | 9.4 GB | default |
| `vllm/vllm-openai:latest-x86_64-ubuntu2404` | 8.3 GB | smallest published x86 CUDA tag |
| `lmsysorg/sglang:v0.5.12-cu130` | 12.4 GB | default |
| `lmsysorg/sglang:v0.5.12-cu130-runtime` | 11.3 GB | upstream's existing runtime variant |

Both upstream projects have publicly acknowledged the bloat:

- **vLLM Issue [#28656](https://github.com/vllm-project/vllm/issues/28656)** (open) — image size > 30 GB tracking.
- **vLLM PR [#22377](https://github.com/vllm-project/vllm/pull/22377)** (closed inactive Apr 2026) — community attempt at a slim Dockerfile, claimed 47% size cut. Our `Dockerfile.vllm-slim` picks up where this left off.
- **vLLM PR [#41134](https://github.com/vllm-project/vllm/pull/41134)** (merged May 2026) — defers `flashinfer download-cubin` to remove ~2.5 GB cross-layer duplication. Adopted here.
- **SGLang Issue [#19160](https://github.com/sgl-project/sglang/issues/19160)** (closed) — forensic breakdown of the 76.5 GB ROCm image. Names every bloat source we strip.
- **SGLang Issue [#22231](https://github.com/sgl-project/sglang/issues/22231)** (open) — redundant `nvidia-*` pip wheels. Our `Dockerfile.sglang-slim` strips them.

## Targets

| Image | Target compressed | Target uncompressed |
|---|---|---|
| `vllm-slim:0.21.0-cu128` | ≤ 7 GB | ≤ 18 GB |
| `vllm-slim:0.21.0-cu130` | ≤ 7 GB | ≤ 18 GB |
| `sglang-slim:0.5.12-cu128` | ≤ 8 GB | ≤ 20 GB |
| `sglang-slim:0.5.12-cu130` | ≤ 8 GB | ≤ 20 GB |

Source PR #22377 hinted at ~6.5 GB compressed achievable for vLLM. Treat 7 GB as the conservative target; if a build comes in cleaner, document the diff and lock in.

## What was stripped

Both Dockerfiles share a strip pattern. Specifics differ slightly per engine.

| Category | Removed | Reason |
|---|---|---|
| Base image flavor | `-devel` → `-runtime` for stage 2 | -devel ships toolchain (~6 GB) we only need at build time |
| Torch extras | `torchvision`, `torchaudio` | unused by serving path |
| HuggingFace | `datasets` | unused by serving path |
| Test/dev | `tests/`, `test/`, `examples/`, `docs/`, `*.cu`, `*.cpp.o`, `*.a` | not load-bearing at runtime |
| Python bytecode artifacts | `__pycache__`, `*.pyc`, `*.pyo` | regenerated at runtime if needed |
| Pip caches | `~/.cache/pip` via `--no-cache-dir` | leftover wheel/HTTP caches |
| Symbol tables | `strip --strip-unneeded` on `*.so` | debug symbols, not needed at runtime |
| FlashInfer cubin | deferred to runtime | ~2.5 GB bundled by default (PR #41134) |
| Redundant nvidia-* wheels (SGLang) | `nvidia-cuda-runtime-cu*`, `nvidia-cublas-cu*`, `nvidia-cufft-cu*`, `nvidia-curand-cu*`, `nvidia-cusolver-cu*`, `nvidia-cusparse-cu*`, `nvidia-cuda-cupti-cu*`, `nvidia-cuda-nvrtc-cu*`, `nvidia-cuda-nvcc-cu*` | already provided by `nvidia/cuda:*-runtime` base |

What was **kept**:

- `flash-attn`, `flashinfer-python`, `triton`, `torch` core — load-bearing kernels.
- `nvidia-nccl` and `nvidia-nvjitlink` (SGLang case) — multi-GPU + lazy-load ABI.
- `transformers`, `tokenizers`, `huggingface_hub` — model loading.
- `fastapi`, `uvicorn`, `openai` — HTTP serving surface.
- `prometheus-client` — observability.

## Variant strategy: slim base + per-blueprint overlay

Per-blueprint patches (e.g. our memory has GLM-5 chat-template fixes, Kimi K2.6 specific imports, Qwen3 tool-call parser) live in a thin overlay Dockerfile in each `gpu-serving` blueprint:

```dockerfile
# domains/gpu-serving/blueprints/glm5-fp8/Dockerfile.overlay
FROM ai-infra/vllm-slim:0.21.0-cu130
COPY patches/anthropic-messages-api-fix.patch /tmp/
RUN cd /opt/venv/lib/python3.12/site-packages/vllm && \
    patch -p1 < /tmp/anthropic-messages-api-fix.patch
```

This keeps the slim base **upstreamable as-is** while per-deployment quirks stay in their respective blueprints.

## CUDA version pinning

Two variants:

- `cu128` — for current g7e/p4d/p5e fleet on CUDA 12.8.
- `cu130` — for B300 (sm_103) which requires CUDA 13.0+ per our memory.

When the fleet shifts hardware, add a new variant; don't reuse old tags.

## Building

```bash
# all engines, all variants
./build.sh

# one engine, one variant
./build.sh vllm cu128
./build.sh sglang cu130
```

Pinned versions live in `build.sh` as defaults; override with env vars:

```bash
VLLM_VERSION=0.22.0 ./build.sh vllm cu128
```

Builds are local-only (`--load`). To push to a registry, retag and push manually so we don't accidentally ship a bad build.

## Comparing sizes

```bash
./compare-sizes.sh
```

Prints a table of upstream vs slim, compressed (registry) vs uncompressed (local). Run after a build to verify size targets were met.

## Spec 0 integration

The profiler validation spec (`specs/profiler-validation.md`) now requires running a baseline cold-start measurement against the slim image *before* declaring the profiler validated. This establishes:

- Image-pull stage time on slim baseline (the floor for Spec A).
- Stage-budget shape on slim (so spec D's stacking math doesn't include stripped-bloat as a measurement win).

## Upstream contribution path

When the slim Dockerfiles are stable and proven on at least three blueprints (qwen3-next, glm5-fp8, kimi-k2.6), open contribution PRs:

1. **vLLM**: PR against `vllm-project/vllm` adding `Dockerfile.slim`, citing Issue #28656 and resurrecting the closed PR #22377's intent. Author the PR with `fe contribute` and the lessons frontmatter from this domain.
2. **SGLang**: PR against `sgl-project/sglang` addressing Issue #22231 directly with the redundant-wheel strip pattern, plus the multi-stage refactor from Issue #10784/#19160.

Both upstreams have explicitly asked for this; political risk is low.

## Maintenance

- **Bump trigger**: when either upstream releases a major (vLLM y in 0.x.y, SGLang minor) or when CUDA version changes, rebuild + re-run Spec 0.
- **Regression check**: if compressed size grows by >20% after a bump, dive into the new dependencies before accepting (`docker run --rm -v /var/run/docker.sock:/var/run/docker.sock wagoodman/dive:latest <tag>`).
- **Spec 0 re-validation**: every bump invalidates the log_patterns.yaml regex set; CI must re-validate before downstream specs trust the new image.
