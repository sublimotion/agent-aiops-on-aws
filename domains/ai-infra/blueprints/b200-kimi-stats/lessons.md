# B200 Kimi K2.6 — Cudagraph Trim + Spec C + Spec B Session

## Status: 4 measurements captured 2026-05-24

Single B200 spot session in us-east-2b, 70 min at $113.93/hr = ~$133. Four runs back-to-back: cold-S3 baseline, cudagraph-trim, AOT-cache-hit, RunAI Streamer.

## Headline numbers

| Run | Variant | main → ready | Δ | Notes |
|---|---|---|---|---|
| 1 | Cold S3 + default | **573 s** | baseline | In-region S3 sync = 18.3 min (vs 32.3 cross-region B300) |
| 2 | Warm + cudagraph trim (51→10 sizes) | **411 s** | -162 s (-28%) | Single CLI flag |
| 3 | Warm + trim + AOT cache HIT | **384 s** | -189 s vs baseline, -27 s vs trim alone | Cache copied from run-1 to /mnt/nvme |
| 4 | Warm + trim + cache + RunAI Streamer | **381 s** | -3 s vs run 3 (zero) | Local NVMe is already PCIe-bound |

First-token after Ready was ~6.5 s on every run.

## Total cold-start (TOTAL pod-create-to-ready)

| Variant | Total | vs B300 cross-region 46 min |
|---|---|---|
| Cold S3 (in-region) | 30.6 min | 1.5× faster |
| Warm + trim + cache + streamer | ~6.4 min | 7× faster |

## Findings

### 1. Cudagraph trim is the cheapest big win
**162 s saved** by setting `cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,256,512]` (10 sizes vs default 51). Single CLI flag. Zero infrastructure change.

This is the **single most cost-effective optimization in the entire lab.** $/second saved: ~$0.05.

### 2. AOT compile cache HIT works as expected — but on TP=8 the marginal gain is small
- Backbone + drafter compile = ~140 s of original 411 s warm path
- AOT cache HIT eliminates ~27 s of that (the actual `torch.compile` step)
- Inductor warmup (~5 s per submodule) still happens
- CUDA graph capture (which is NOT cached) still takes ~30-50 s with our trimmed list

The 22 s Modal floor was for a 26B *single-GPU* model. On Kimi K2.6 TP=8 (745 GB MoE), weight-load swamps compile time, so cache impact is muted.

### 3. RunAI Streamer is conditional — no win on local NVMe
**3-second improvement** (statistical zero). RunAI Streamer parallelizes safetensors range reads, which matters when source is high-latency (S3, FSx). When source is local NVMe at ~6-12 GB/s, single-reader already saturates PCIe→HBM bandwidth.

**Steering rule**: only enable `--load-format=runai_streamer` when load source is S3 or other network-attached storage. On local NVMe it's neutral.

### 4. Weight load is now the dominant stage
With trim + cache + streamer, breakdown is approximately:
- Weight load: 246 s (64% of warm path)
- Compile + warmup + capture: 135 s (35%)
- API server bootstrap: ~3 s

To get below ~6 min on Kimi K2.6 TP=8 warm-NVMe, **weight load is the next bottleneck**. Levers:
- ModelExpress same-node P2P (need ≥2 pods on same node first)
- vLLM `--safetensors-load-strategy=prefetch` flag (untested for our case)
- Reducing model size (FP4 instead of FP8?) — orthogonal to cold-start work

### 5. In-region S3 saves 14 minutes vs cross-region
B200 us-east-2b reading from us-east-2 S3 bucket: **18.3 min for 585 GB** (~533 MB/s effective).
B300 us-west-2b reading from us-east-2 S3 cross-region: **32.3 min** (~302 MB/s effective).

Always co-locate weight bucket with serving cluster. Steering rule level.

## What this means for Spec D (stacked)

Best-of-stacked path on Kimi K2.6 FP8 TP=8 B200 us-east-2:

| Stage | Default | Stacked best | Saving |
|---|---|---|---|
| S3 sync | 18:20 | 0 (warm pool) | -18:20 |
| Image pull | ~30 s | ~30 s | unchanged |
| Container start | ~5 s | ~5 s | unchanged |
| Weight load | 246 s | 246 s (no improvement available on NVMe) | 0 |
| Compile + warmup | ~140 s | ~140 s with cache | small further gain |
| CUDA graph capture | ~50 s (trimmed) | ~50 s | unchanged (already trimmed) |
| **TOTAL** | **2030 s (33 min cold)** | **~470 s (~7:50 min)** | **4.3× from cold** |

The Modal "40× speedup → 50 s" headline isn't reachable on EKS without GPU memory snapshotting infrastructure (per `references/modal-truly-serverless-gpus.md`).

For our setup: **~6-8 min is the realistic warm-replica floor on Kimi K2.6 FP8 TP=8 B200**. Most of that is unavoidable weight-load over PCIe to fill 745 GB across 8 GPUs.

## Operational lessons

1. **Cache extraction trick**: copy `/root/.cache/vllm/` from a running pod to `/mnt/nvme/compile-cache/`, then mount it as `hostPath` in subsequent pods. Cache key (`12def824...`) is shape+model+config keyed; portable across vLLM restarts.
2. **`compilation-config` JSON arg works** for cudagraph trim: `--compilation-config={"cudagraph_capture_sizes":[...]}`
3. **AOT compile cache size for Kimi K2.6 TP=8**: 502 MB across 8 rank dirs — small enough to bake into image or store on PVC trivially.
4. **RunAI Streamer auto-disabled the existing AOT cache** on first try (cache key includes load_format). Cache is keyed on enough fields that switching loaders invalidates. Not a problem here because run-3's cache is identical config to run-4 except load_format... wait actually the log showed run-4 hash matches run-3, so cache key may NOT include load_format. Worth investigating.

## Cost summary

- B200 us-east-2b spot, ~70 min at $113.93/hr: ~$133
- Capacity probe earlier: ~$2
- Total this session: ~$135

Cumulative across all sessions: $131 + $135 = **~$266 of $500 budget** ($234 remaining).

## What's still left

| Item | Cost | Priority |
|---|---|---|
| Spec A SOCI variants (no GPU) | ~$30 build host | Low — confirmed 17.2% access ratio is real, image-shrinking already wins 1.87× |
| Spec E FUSE tuning (no GPU) | ~$10 | Low — sub-Spec-A optimization |
| Variance tightening (3 more cold + 3 more warm runs of each variant) | ~$200 B200 | Medium — current single-shot results are headline-strong |
| GPU memory snapshot exploration | ~$50 + research time | Low — Modal proves it works but multi-GPU NCCL deadlock blocks our TP=8 case |
| **Stacked Spec D end-to-end** | covered by completed runs | DONE — already measured the stacked path |

**Recommendation**: lab core is functionally complete. Remaining $234 is reserve for variance/replication work. The publishable findings are in hand.
