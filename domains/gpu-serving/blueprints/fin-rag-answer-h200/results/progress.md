# Fin-RAG Answer — H200 (p5e.48xlarge) campaign

Model: NVIDIA-Nemotron-3-Super-120B-A12B-FP8. Hardware: 8× H200 141GB (Hopper sm_90, NVSwitch NVL5).
Goal: 3-way $/1M-token cost comparison vs B200 (done) and g7e (pending capacity).
Workload: `fin-support` (RAG answer-extraction, prefill-dominated, ISL≈9.1K, OSL≈75).
SLO: e2e p50 ≤ 6500ms, p90 ≤ 9500ms @ c130; ttft_p99 ≤ 6000ms; tpot_p99 ≤ 50ms; err ≤ 0.001.

## Stage status
- Node Ready: DONE (p5e.48xlarge spot, us-east-2b, i-04f13551a5a18ec27).
- NVMe + weight copy: DONE. FSx Lustre client absent on AL2023 NVIDIA AMI → `dnf install lustre-client` +
  `modprobe lustre` + `lctl network up`, then host-direct mount (CSI driver's bundled mount.lustre can't
  parse the DNS-style NID). 120G FP8 weights on `/mnt/nvme/models`, 26 safetensors verified.
- vLLM 0.22.1 Stage 0 smoke (TP1, 1 replica): PASS — 5/5 coherent grounded answers. This validates the
  PRIMARY image that the B200 campaign skipped (B200 ran on 0.18.1).
- SGLang (Engine B): deployed `serving-sglang-agg-tp2x4-fp8.yaml` (cu130 v0.5.12.post1, standard sm_90
  path — DeepGEMM enabled, no Triton FP8 workarounds; radix cache ON). FP8 loaded cleanly, server
  "fired up". **Probe gotcha:** SGLang's `/health` runs a real generation and exceeded the 1s probe
  timeout → pods stuck 0/1 Ready despite serving fine. Fixed: readiness/liveness → `/get_model_info`
  with timeoutSeconds=10. Bench pending ready.

## Bench results — vLLM 0.22.1 FP8

### Leg A: agg-tp2-x4 (B200 mirror config — 4 replicas × TP2)
| conc | e2e p50 | e2e p90 | $/1M total tok | SLO |
|------|---------|---------|----------------|-----|
| 8    | 1287.0  | 1613.4  | 0.1506         | PASS |
| 130  | 7038.2  | 11095.4 | 0.0531         | FAIL |
| 200  | 10296.0 | 16222.9 | 0.0495         | FAIL |
| 256  | 12662.4 | 20743.0 | 0.0479         | FAIL |
| 512  | 24147.6 | 43183.0 | 0.0457         | FAIL |

**Finding:** On the identical config B200 passed c130 at p50 5334 / p90 8055, $0.0402/1M total.
H200 misses SLO at c130 (p50 7038 > 6500, p90 11095 > 9500) and is ~30% slower + ~32% costlier.
For this prefill-heavy workload, B200 (sm_100, higher FP8 FLOPS + memory BW) wins on both latency and cost.
No prefix-cache hit (0%) — same Mamba2 PC non-functionality seen on B200.

### Leg B: agg-tp1-x8 (H200 TP1-VIABLE hypothesis — 8 replicas × TP1, zero TP collectives)
Deployed: 8/8 replicas Ready. FP8 weights (~124G) fit a single H200's 141GB → 8 independent replicas,
no NCCL on serving path. Tests whether max-replica TP1 beats tp2x4 on this latency-bound workload.

**Gotcha:** the `fin-rag-h200-vllm-fp8-tp1` *service DNS name* failed to resolve from the bench pod
(`gaierror Name or service not known`) — first sweep got 100% errors in <1s. The ClusterIP (172.20.221.2)
resolves fine and `/v1/models` serves correctly. Re-ran the sweep against the ClusterIP directly (same
approach used for the working tp2x4 leg). Lesson: bench against ClusterIP, not service DNS, on this cluster.

| conc | e2e p50 | e2e p90 | $/1M total tok | SLO |
|------|---------|---------|----------------|-----|
| 8    | 1218.5  | 2334.9  | 0.1601         | PASS |
| 130  | 5863.0  | 9961.7  | 0.0458         | FAIL (p50 PASS 5863≤6500; p90 9962>9500, ttft_p99 7013>6000) |
| 200  | 8012.7  | 12944.2 | 0.0403         | FAIL |
| 256  | 9410.8  | 16199.5 | 0.0376         | FAIL |
| 512  | 17619.0 | 28490.6 | 0.0384         | FAIL |

**Finding — TP1-VIABLE confirmed:** 8×TP1 beats 4×TP2 at every concurrency on this latency-bound,
prefill-dominated workload. At c130: p50 5863 vs 7038 (-17%), p90 9962 vs 11095, $0.0458 vs $0.0531/1M
(-14%). Zero TP collectives (no NCCL on the serving path) + max replica count is the right H200 layout
when FP8 weights fit in a single GPU's 141GB. TP1x8 *passes* the p50 SLO at c130 but narrowly misses
p90 and ttft_p99 — H200 is close but can't quite hit the B200-tuned SLO at full c130 load.

### Leg C: SGLang agg-tp2-x4 (Engine B — the radix-cache science question)
cu130 v0.5.12.post1, standard sm_90 path (DeepGEMM, no Triton FP8 workarounds), radix cache ON.
| conc | e2e p50 | e2e p90 | tpot p99 | $/1M total | SLO |
|------|---------|---------|----------|------------|-----|
| 8    | 2956    | 4253    | —        | 0.358      | FAIL (p50 2956<6500 PASS, but tail) |
| 130  | 15655.3 | 25196.5 | 322.7    | 0.1201     | FAIL |
| 200  | (slower)| —       | —        | —          | FAIL |

**Headline finding — radix cache WORKS but SGLang still loses ~2.7x:**
- **The science question is answered YES:** SGLang's radix cache *does* engage on this hybrid Mamba2 model
  where vLLM's automatic prefix-cache was 0% on B200. Server logs during the c130 bench show
  `#cached-token` consistently 1544–4764 (the ~1.5K–3.1K shared RAG header, sometimes multiples) —
  i.e. the shared header IS being reused. (Bench's `prefix_cache.hit_rate_measured` reads vLLM-style
  metric names → shows null for SGLang; confirmed via serving logs instead.)
- **But it doesn't help end-to-end:** SGLang c130 p50 15655 vs vLLM tp1x8 5863 (~2.7x slower), tpot_p99
  322.7 vs 106 (~3x), $0.1201 vs $0.0458/1M (~2.6x costlier). Root cause: the Mamba2 linear-attention
  path runs on **Triton kernels** (`decode=triton, prefill=triton`) and `mamba usage: 0.63` shows the
  Mamba *state* cache — not prefix reuse — is the bottleneck. vLLM's Mamba2 kernels are simply faster
  on this architecture. Radix cache solving the prefix-hit gap is real but second-order here because
  this workload is decode/Mamba-bound, not prefill-prefix-bound, at c130.
- **Actionable:** for Nemotron-3-Super hybrid-Mamba serving, prefer vLLM over SGLang on H200/sm_90.
  SGLang's radix-cache advantage only pays off on attention-heavy (non-Mamba) models or much higher
  shared-prefix ratios than this RAG workload's ~3K/9K header.

## H200 winner: agg-tp1-x8 (vLLM). Cross-platform @ c130 (SLO anchor)
| Platform | Config | e2e p50 | e2e p90 | $/1M total | c130 SLO |
|----------|--------|---------|---------|------------|----------|
| B200 (sm_100) | agg-tp2x4 mnbt16384 | 4685 | 8147 | 0.0402 | PASS |
| H200 (sm_90)  | agg-tp1x8           | 5863 | 9962 | 0.0458 | p50 only |
| H200 (sm_90)  | agg-tp2x4           | 7038 | 11095| 0.0531 | FAIL |

B200 remains the latency + cost winner; H200-tp1x8 is the closest H200 layout (~17% slower p50,
~14% costlier) and only just misses the p90/ttft tail. **g7e leg CAPACITY-BLOCKED** (2026-06-12):
g7e on-demand `InsufficientInstanceCapacity` across us-east-2a/2b for >2h; comparison ships as
B200 vs H200, g7e deferred (watch loop will launch it opportunistically if capacity returns).

## Cost basis
H200 p5e.48xlarge spot ≈ $30.10/hr (us-east-2). Cost metric is $/1M TOTAL tokens (prefill+decode);
output-only also emitted in enriched envelopes.

## Artifacts
Enriched v1 envelopes: `results/standard/` (schema-valid, symlinked into results-vault, indexed).
Enrichment shim: `../fin-rag-answer/scripts/enrich-to-standard.py --platform h200`.
