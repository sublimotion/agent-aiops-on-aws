# GLM-5.2: our sweep vs Artificial Analysis providers

Source (AA): https://artificialanalysis.ai/models/glm-5-2/providers (P50 over 72h, snapshot 2026-07-09)
Source (ours): `optimization-trajectory-2026-06-27.json` — SGLang v0.5.13.post1-cu130,
GLM-5.2-FP8, 1× p6-b300.48xlarge (8× B300 SXM6, TP4+DP2), coding-agent workload
(12K byte-identical shared prefix + ~1K suffix, 2048 out).

## The two benchmarks measure DIFFERENT axes — don't lay headline numbers side by side

| | Artificial Analysis | Our sweep |
|---|---|---|
| Metric | Per-**user** median output speed (single stream) + TTFT | **Aggregate** goodput (tok/s across all concurrent reqs) at an SLO |
| Workload | AA generic prompt mix (independent prompts) | Coding-agent, 12K **shared** prefix (92% cache-hit) |
| Price | **Blended** 7:2:1 cache:input:output | **Pure output-token** cost |
| Scope | Hosted endpoints, provider-dependent quant | One 8-GPU B300 node, FP8, us-west-2 |

## Per-user speed (AA's axis) — hosted providers win 4–5×

- AA field: 38–455 tok/s per user (Together 455, Fireworks 371, Databricks 333, median ~150).
- Ours (aggregate ÷ concurrency): ~85 tok/s/user at c8 (w/ MTP), ~30 tok/s/user at c256.
- Expected: we optimized goodput, not single-stream latency. MTP was only a modest win here
  (accept rate grounded on production mix, not synthetic).

## Throughput / density (our axis) — AA does not report it

- AA page explicitly: throughput is *not* a separate concurrent metric.
- Ours: ~7,728 tok/s at TTFT p95 6.7s (T4 c256); **9,895 tok/s** peak @ SLO (T4 c512, TTFT 13.6s).

## $/1M output tokens — the one comparable cell

`$/1M = hourly × 1e6 / (tok/s × 3600)`. Pure output-token cost (conservative vs AA's blend).

| Config (SLO) | tok/s | OD $142.42/hr | Spot $27.63/hr (Vantage) | Spot $15/hr (observed us-west-2 az2) |
|---|---:|---:|---:|---:|
| T2 c256 (TTFT 10.7s) | 6,025 | $6.57 | $1.27 | $0.69 |
| T4 c256 (TTFT 6.7s) | 7,728 | $5.12 | $0.99 | $0.54 |
| **T4 c512 (peak @ SLO, 13.6s)** | 9,895 | **$4.00** | **$0.78** | **$0.42** |

**AA blended price:** $0.61–$1.70/1M, most providers at **$0.90** (floor DeepInfra FP4 $0.61).

### Reading it
- **On-demand loses** (4–7× hosted). Don't self-host OD for cost — only for residency/custom quant/guaranteed capacity.
- **Spot flips it.** $27.63/hr → $0.78/1M (inside AA band, beats $0.90 median). $15/hr observed → $0.42/1M (below AA floor).
- Contingent on: tolerating spot reclaim, landing the low spot price, AND running c256–512 (per-user speed only ~30 tok/s there).

## One-liner
Our sweep can undercut hosted GLM-5.2 pricing on spot at high concurrency, but only by trading
away the single-stream latency AA's leaderboard actually ranks on. Prices are also
apples-to-oranges: AA blends in cheap cached-input; ours is pure (expensive) output tokens.
