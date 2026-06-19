# Spec 0 Profiler Validation — Session Lessons

## Status: PARTIAL VALIDATION COMPLETE

Spec 0 ran on 2026-05-22 against:
- **Small fixture**: Qwen3-1.7B (TP=1) on p6-b300.48xlarge spot, 5 cold-start measurements.
- **Large fixture**: Kimi K2.6 FP8 (TP=8, 585 GB) on p6-b300.48xlarge spot, 1 cold-start measurement.

Results in `results/` (small) and `../spec-0-kimi/results/` (large). Profiler artifact format validated. Stage attribution math holds across all 5 small-fixture runs.

## Headline numbers

### Qwen3-1.7B cold-start stage budget (median of 5 runs)

| Stage | Median ms |
|---|---|
| Node provision (T0→T1) | ~5,000 |
| Image pull (T2→T3) | ~120 (cached on node) |
| Container start (T4→T6) | ~3,700 |
| Weights load start (T7) | ~16,000 |
| CUDA graphs done (T11) | **~85,000** |
| Health 200 (T12) | ~98,000 |
| **First token (T13)** | **~102,000-118,000** |

**JIT/CUDA-graph capture is ~70 of the 102 sec total** even for a 1.7B model.

### Kimi K2.6 FP8 cold-start (single run, 1.18 TB → 585 GB FP8)

| Stage | Duration |
|---|---|
| S3 sync (init container, 585 GB → NVMe) | **32:20 min** |
| Container start | 4.8 sec |
| Model load + first JIT pass | **10:19 min** |
| CUDA graph capture | 31 sec |
| **TOTAL pod-create → pod-ready** | **43:19 min** |
| First completion request after ready | 12.2 sec |

## What was learned (planned vs actual)

### Confirmed hypotheses

1. **JIT compile dominates cold start even for small models** (Spec C's premise). Qwen3-1.7B: 85s of 102s = 83%.
2. **S3 sync is the dominant stage for frontier models** (Spec B's premise). Kimi: 32:20 of 43:19 = 75%.
3. **Image pull from ECR can be sub-second when cached** (Spec A baseline confirmed).
4. **Stock vLLM 0.21.0 supports Kimi K2.6** (`kimi_k25` model class works) — the speculative-decoding fork is only needed for that specific feature.

### Surprises

1. **Slim images don't work without GPU dev toolchain** — flashinfer JITs at runtime via ninja, needs `nvcc`. Stripping `-devel` for `-runtime` base broke flashinfer. Trade-off must be made: ship full toolchain (~2 GB) or pre-warm the JIT cache during build (needs GPU-at-build-time). Neither is in our slim Dockerfile yet.
2. **EBS root + 500GB is not enough** for Kimi K2.6 (585 GB) plus container layers. Needed to mount instance NVMe (which the AL2023 NVIDIA AMI doesn't auto-mount).
3. **B300 (p6-b300.48xlarge) has 8x 3.5 TB instance NVMe** = 28 TB local. That's where serving fixtures should put their data.
4. **vLLM v1 engine emits `Loading model weights` and similar markers as DEBUG, not INFO** — log_patterns.yaml regexes need DEBUG-prefix support or `VLLM_LOGGING_LEVEL=DEBUG`.
5. **Cross-region SSH from spot CIDR is unreliable**: my IP changed mid-session, and SSH port-22 latency would intermittently 30s timeout. SSM is much more stable. Future build hosts should default to SSM-only access.

### Bugs found in our own scaffold

1. **`build.sh` had wrong CUDA tags** (`12.8.0`, `13.0.0` don't exist on docker hub — current is `12.8.2`, `13.0.3`). Fixed.
2. **Dockerfile base was Ubuntu 22.04** which lacks Python 3.12. Bumped to Ubuntu 24.04. Fixed.
3. **Dockerfile didn't `ARG CUDA_TAG` inside builder stage** — value reset to empty after FROM. Fixed.
4. **Torch version pin was 2.5.1 (vLLM 0.21.0 wants 2.11.0)**. Fixed.
5. **`outlines==0.1.11` pin downgraded torch to 2.4** — replaced manual dep list with vLLM's own dep resolver. Fixed.
6. **Build-host bootstrap shell tried `disown` builtin**, but SSM uses `/bin/sh`. Switched to `setsid`.
7. **Bootstrap default repo URL was a fake** — not a real public repo. Worked around with S3 staging tarball.
8. **Profiler `log_patterns.yaml` matches not validated** — T8 (model loaded) didn't fire on any of the 5 runs because the regex doesn't catch vLLM 0.21.0's actual log line shape.

## What's next (concrete TODOs)

1. **Update `log_patterns.yaml`** with vLLM 0.21.0 log lines actually observed. Specifically:
   - T7: `(EngineCore .*) Starting model loading|Loading safetensors checkpoint shards`
   - T8: `Model loading took .*GiB|finished loading.*model in`
   - T11: `Capturing CUDA graphs|Application startup complete`
2. **Decide slim image strategy**: ship `gcc`/`g++`/`nvcc` (~2 GB more) or pre-warm flashinfer cache during build with a GPU runner.
3. **Add NVMe instance store auto-mount** to B300 nodegroup userData. Current manual mount is fragile.
4. **Run Spec B on Kimi**: 5 cold starts × variants {baseline, init+s5cmd-only, +Run:ai Streamer, +ModelExpress same-node}. Each adds <5 min on top of the 43-min baseline. Budget ~$60 of B300 time.
5. **Run Spec C**: same, but with compile-cache strategies.
6. **Extend small fixture to 5 Kimi K2.6 runs** for proper variance statistics, not just one.

## Cost summary

- Build host (c7i.4xlarge on-demand, ~2 hr): ~$1.50
- B300 (p6-b300.48xlarge spot, ~3:15 hr): ~$85
- ECR storage: <$1
- S3 transfer: free in-region
- **Total spend so far: ~$87 of $500 budget**

## Files

- `results/qwen3-prof-{1..5}.json` — Qwen3 cold-start artifacts (this dir)
- `../spec-0-kimi/results/kimi-prof-1.json` — Kimi K2.6 cold-start artifact
- `../../shared/images/Dockerfile.vllm-slim` — slim image (broken at runtime, see TODO 2)
- `../../staging/manifests/{qwen3-next,kimi-k2.6}-fixture.yaml` — fixture pod manifests
- `../../staging/scripts/run-plan.sh` — phase-driven runner (didn't end up using; ran phases manually)
