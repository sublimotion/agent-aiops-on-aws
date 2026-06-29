---
model: "GLM-5.2"
engine: "sglang"
hardware: "p6-b300.48xlarge"
gpu_arch: "sm_103"
deployment_date: "2026-06-27"

outcome: "success"
failure_categories: []

cards_used:
  mdc: ["glm-5-sglang"]
  gpu_infra: ["p6-b300"]

card_helped: true

benchmark:
  throughput_toks_s: 9271
  ttft_p50_ms: null
  ttft_p99_ms: 8360
  concurrent_users: 320
  gpu_util_pct: null

ralph_iterations: 1

mdc_learn_commands:
  - 'mdc learn GLM-5.2 sglang "fp8-KV is auto-default on SGLang v0.5.13+ — quantization=fp8 auto-resolves kv_cache_dtype=fp8_e4m3"'
  - 'mdc learn GLM-5.2 sglang "Official nvidia/GLM-5.2-NVFP4 regresses ~50% batched throughput vs FP8 on B300 sm_103 (experimental MoE kernel, deferred-finalize disabled). Use FP8 for high-concurrency goodput; NVFP4 only for VRAM-constrained fit or single-stream latency."'
  - 'mdc learn GLM-5.2 sglang "TP4+DP2 layout beats TP8 by +28% @ c256 on B300 8-GPU (7,728→9,271 tok/s @ c320 knee, TTFT p95 8.1s). Batch-scheduling win for high-concurrency MoE."'
  - 'mdc learn GLM-5.2 sglang "Prefix cache is the dominant lever (+347% vs cache-off baseline @ c256). Coding-agent workload with 12K byte-identical prefix → 92% cache hit → regime flips prefill-bound → decode-bound."'
  - 'mdc learn GLM-5.2 sglang "NEXTN MTP spec-decode REGRESSES −12% at the c256 knee (accept-len ~1.6, draft overhead at batch). MTP is a low-QPS latency lever, not for high-concurrency goodput."'
  - 'mdc learn GLM-5.2 sglang "--chunked-prefill-size (MNBT) is regime-dependent: default-16384 wins at the c320 knee WITH prefix cache (+9400 vs 9157 @ c320), opposite of B200 no-cache long-context result where 8192 won. Prefix cache removes prefill pressure MNBT addressed on B200; smaller chunks add overhead when cache already handles the repeated prefix."'

gpu_infra_learn_commands:
  - 'gpu-infra learn -c platform "DCGM PROF metrics unavailable on B300 sm_103 / driver 580.159.03 (dcgmi confirms: DCP module loads but fields 1004/1005 read N/A/0.000, no counter movement). Same class as B200/driver-580 kimi-k2.6-nvfp4 L8. Driver-580 profiling path not functional on Blackwell yet. Use nsys --gpu-metrics-device for HBM-BW/SM/tensor-active sampling at the knee, or fall back to engine gauges + nvidia-smi dmon for coarse bottleneck classification."'
  - 'gpu-infra learn -c platform "B300 cluster EKS node has no GFD/NFD — device plugin needs hand-labeling nvidia.com/gpu.present=true after scale-up before GPUs advertise (same as kimi-k2.6-nvfp4 L1/L7)."'
---

# GLM-5.2 — Lessons (B300 optimization-loop run, us-west-2)

> B200 run lessons live in `results/STAGE6-REPORT.md`. This file tracks the fresh B300 loop
> (cluster `qn-sglang-eks-cluster`, nodegroup `ai-infra-b300-spot`, p6-b300.48xlarge, driver 580.159.03).

## L1 — kubectl context silently flips back to the B200 cluster; pin `--context` on every call
Mid-session, `kubectl config current-context` reverted from `qn-sglang-usw2` (B300/us-west-2) to
`qn-bench-use2` (B200/us-east-2) on its own. Symptom: `node not found` + a just-created pod "vanished",
which looked exactly like a spot reclaim. It was NOT — the EC2 instance (`i-0df2c4807b80b98a5`,
10.2.27.125) was `running` the whole time; I was querying the wrong cluster. **Rule:** confirm a
suspected spot reclaim against EC2 (`aws ec2 describe-instances`) before believing kubectl, and pin
`kubectl --context=qn-sglang-usw2` on every command this session. (This is the prior-session
stale-context trap recurring.)

## L2 — B300 cluster has no GFD/NFD: device plugin needs the node hand-labeled
`nvidia-device-plugin` daemonset has `nodeSelector: nvidia.com/gpu.present=true` and already tolerates
the `ai-infra/b300` taint, but nothing labels the node (no GFD/NFD). After scale-up the node shows
`nvidia.com/gpu: <none>` until you `kubectl label node <n> nvidia.com/gpu.present=true blueprint=glm5.2`.
Plugin then schedules and advertises 8 GPUs within ~20s. (Same condition as kimi-k2.6-nvfp4 L1/L7.)

## L3 — DCGM PROF metrics unavailable on B300 + driver-580.159.03 (HBM-BW/compute regime = gauge-inferred)
The spec's "re-verify PROF on the actual node" gate fired correctly and caught this. Findings, in order:
- **CSV wiring fix works**: exporter loaded `/etc/dcgm-prof/metrics.csv` and read all 7 PROF fields
  (no longer the kimi-parent wiring bug). So the empty result is NOT our config.
- **v3.3.9 exporter**: DCP module "not currently loaded" → every PROF field "metric not enabled".
  Predates Blackwell.
- **v4.2.3 exporter**: DCP module loads ("Collecting DCP Metrics"), but still emits only the 4 basic
  device fields (`FB_USED`, `GPU_TEMP`, `POWER_USAGE`, `XID_ERRORS`).
- **dcgmi probe (definitive)**: `dcgmi profile --list -i 0` LISTS fields 1004 `tensor_active` / 1005
  `dram_active` as present, but `dcgmi dmon -e 1005,1004,1002` reads **N/A then 0.000** with no counter
  movement — the profiling sampler doesn't actually engage on this B300/driver-580 combo.
- **Conclusion**: same class of limitation as kimi-k2.6-nvfp4 L8 on B200/driver-580. Driver-580 profiling
  path is not functional on Blackwell (sm_100/sm_103) yet. Not worth a driver upgrade on a spot node.
- **Resolution (chosen 2026-06-27): nsys GPU-metrics sampling, not the gauge-only fallback.**
  `nsys profile --gpu-metrics-device=all` samples the SAME SM-active / tensor-active / DRAM-bandwidth
  hardware counters DCGM exposes, but via the Nsight sampling path that DOES work on Blackwell/driver-580
  (independent of the broken DCP module). Plan: at the identified knee, `nsys profile` the running SGLang
  serving process for ~15–30s under steady load, then read the GPU-metrics row for DRAM throughput % vs
  SM/tensor active % → a true [measured] HBM-BW-vs-compute regime call. Engine gauges (`token_usage`,
  `num_queue_reqs`) still classify KV-capacity/admission; nsys covers the BW-vs-compute axis DCGM can't.

## L5 — bench-standard.py SGLang metric map was wrong on more than just the colon (permanent fix)
The P0 TTFT gate is real and the script needed a permanent patch (prior sed-only fix didn't persist).
Prometheus sanitizes `:`→`_` on ingestion (colon reserved for recording rules), so the engine's
`sglang:time_to_first_token_seconds_bucket` is stored as `sglang_...` — a colon selector is invalid
PromQL and silently returns empty. But verifying against a live SGLang v0.5.13 scrape, several names
differ beyond the colon swap: TPOT → `sglang_inter_token_latency_seconds_bucket` (no
`time_per_output_token`); KV usage → `sglang_token_usage` (not `_ratio`); cache hit →
`sglang_cache_hit_rate`; and there's NO native success/error counter — derive from
`sglang_http_responses_total{endpoint=~"/generate|/v1/chat/completions",status_code=~"2.."}`. All fixed
in `.claude/skills/benchmark-runner/scripts/bench-standard.py`. **P0 gate verified closed**: smoke run
gave client-side TTFT p50 0.97s / p95 2.55s, server-side `histogram_quantile` p95 3.67s — both non-zero,
distinct from E2E p50 21.6s. This is the failure that plagued Kimi for 2 sessions; caught before any paid sweep.

## L6 — name collision with the parked minimax bench-runner; do NOT mutate it
A `bench-runner` pod from the minimax-m2 work sits `Pending` in the same namespace (its
`blueprint: minimax-m2` node is scaled to 0 — harmless, not on my B300 node). `kubectl apply` of the
glm5.2 bench-runner FAILED trying to mutate that immutable pod (pod specs are largely immutable).
Per the "leave minimax alone" constraint: ran mine as `glm52-bench-runner` (distinct name) instead of
touching the parked pod. Reused the same `glm52-bench-scripts` ConfigMap (created cleanly).

## L7 — fp8 KV is the AUTO-DEFAULT for GLM-5.2 DSA; not a separate lever (verify-before-assert win)
Building T4 (TP4+DP2) WITHOUT `--kv-cache-dtype` still logged `kv_cache_dtype='fp8_e4m3'` and allocated
KV as `torch.float8_e4m3fn`. So on SGLang v0.5.13 a quantization='fp8' GLM-5.2 model defaults its KV to
fp8 — T0/T2 were already running fp8 KV. My "T1 = +fp8 KV" config was therefore a near no-op (+1.7% =
noise), NOT evidence that "fp8 KV is a weak lever." Don't credit it as a tunable tier for this model.
Flags this for the B200 STAGE6-REPORT too: re-examine whether its 1708→3004 "fp8 KV" gain was
default-vs-explicit confound or a genuine no-cache-regime effect.

## L8 — c384+ "error wall" was a CLIENT harness artifact (connector limit + cross-run connection reuse)
The 27-32% request failures at c384/c512 were NOT a server limit. Root cause, pinned down by elimination:
bench.py's `TCPConnector(limit=conc*2)` PLUS running multiple concurrencies back-to-back in ONE python
process — closing/reused connections from the prior run poisoned the next. Evidence:
- c384, total=768, fresh process, limit=0 → 768/768, 0 err
- c384, total=1152, fresh process, limit=0 → 1152/1152, 0 err
- c384, total=1152, limit=0 but immediately AFTER another run in the same process → ~267 err
Server `num_retracted_reqs=0` throughout (never server-side). **Rules**: (1) confirm client failures
against server metrics before calling an SLO breach; (2) one fresh client process PER concurrency point —
never chain concurrencies in a single process; (3) `TCPConnector(limit=0)` — the semaphore bounds
concurrency, not the connector. Fixed in `k8s/bench-runner.yaml`. This nearly cost a WRONG knee (c256
recorded as T4's ceiling when it actually scales to >=c512) → wrong fleet sizing. Re-swept clean.

## L9 — official NVFP4 ALSO regresses ~50% vs FP8 (batched) — the B200 finding transfers to B300
The B200 STAGE6 report measured the COMMUNITY lukealonso NVFP4 checkpoint at -55-60% batched vs FP8 and
explicitly flagged the OFFICIAL nvidia/GLM-5.2-NVFP4 (modelopt, shared-expert-preserved recipe) as an OPEN
question — "could change the batched-throughput gap." It does not. Measured on B300, official NVFP4,
TP4+DP2, same prefix-cache layout as FP8 T4: **c256 -56%, c320 -52%, c384 -53%** (NVFP4 ~half FP8's tok/s).
- NOT a fallback: logs confirm the SAME `flashinfer_trtllm` MoE backend as FP8 (`ModelOptNvFp4FusedMoEMethod`).
- Root cause is the kernel: log warns `nvfp4 checkpoint ... format is EXPERIMENTAL`, and `FlashInfer TRTLLM
  MoE deferred finalize is DISABLED` for the NVFP4 quant method (an optimization FP8 retains).
- The 2.4x larger KV pool (433GB weights -> 2.29M KV tokens vs FP8's 956K) does NOT help — the bottleneck is
  MoE GEMM compute, not KV residency. Output is correct (no quant garbage).
**Rule**: for GLM-5.2 high-concurrency *serving goodput*, use FP8, not NVFP4, on SGLang as of 2026-06
(both community AND official recipes lose). NVFP4's remaining value: VRAM-constrained fit (433 vs 707GB) and
possibly c1 single-stream latency (untested). Re-test when SGLang NVFP4 leaves 'experimental'. The spec's
Stage 6c "official recipe might be different" hypothesis is now CLOSED (negative).

## L10 — MNBT/chunked-prefill is a ceiling-tuner here, opposite of the B200 result
On B200 (no prefix cache, long-context) MNBT 8192 beat the 16384 default at the SLO knee. On B300 WITH
prefix cache, default-16384 wins at the c320 knee (9400 vs 9157 tok/s); 8192 only helps by extending the
clean ceiling to c384 (10246 tok/s, p95 10s vs default's p99 breach); 4096 is strictly dominated. Reason:
prefix cache already removes the repeated-prefill pressure that MNBT tuning addressed on B200, so smaller
chunks just add scheduling overhead at the knee. Lever value is regime-dependent — don't carry the B200
MNBT setting forward blindly.

## L4 — NVMe RAID-0 must be built by hand (8× 3.5T data disks, raw on boot)
Node boots with `nvme1n1`–`nvme8n1` (3.5T each) raw + `nvme0n1` (500G root). No RAID, `/mnt/nvme` not
mounted. Built `/dev/md0` RAID-0 over the 8 data disks via `nsenter -t 1 -m` from a privileged pod →
mkfs.xfs → mount `/mnt/nvme` (28T). **Verified `df` shows the mount BEFORE proceeding** (prior trap:
tearing down the staging pod before confirming the mount persisted).
