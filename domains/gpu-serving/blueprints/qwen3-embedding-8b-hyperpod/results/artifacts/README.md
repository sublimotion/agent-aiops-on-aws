# Common Benchmark Artifacts — Qwen3-Embedding-8B on HyperPod A10G

Seven benchmark result files in [Common Benchmark Artifact v1.0.0](../../../../../../standards/benchmark-commons/PROPOSAL.md) format. All pass `standards/benchmark-commons/container/validate-artifact.py` against `schema/enriched-artifact.json`.

| Artifact | Workload | Peak throughput | SLO |
|----------|----------|----------------:|:---:|
| `…_concurrency-sweep_…json` | smoke-bench c=1→32 | 119.95 req/s @ c=32 | PASS |
| `…_rag-qa_…json` | 2-10K char RAG contexts, c=1→32 | 114.27 req/s @ c=32 | PASS |
| `…_production-mix_…json` | 40/40/20 short/medium/long char mix, c=1→32 | 113.69 req/s @ c=32 | PASS |
| `…_long-context-sweep_…json` | context axis 1K/2K/4K/8K tokens × c=1/4/16 | 80.30 req/s @ 1K·c=16 | PASS |
| `…_burn-in_…json` | 1h sustained load at c=28 (12 × 5-min slices) | 124.7 req/s avg, +2.49% drift | FAIL (drift gate 2.0%) |
| `…_rag-long-context_tier-t0_…json` | T0 baseline (eager, no prefix cache) | 6.66 req/s @ c=2 | FAIL (100 req/s floor) |
| `…_rag-long-context_tier-t5_…json` | T5 optimized (FLASH_ATTN + torch.compile + CUDA graphs) | 122.96 req/s @ c=32 | PASS |

## Embedding-specific conventions

Embedding workloads don't have streaming or generated tokens, so the schema fields map as follows:

| Schema field | Embedding interpretation |
|-------------|---------------------------|
| `workload.api.type` | `"embeddings"` |
| `workload.api.streaming` | `false` |
| `workload.api.endpoint` | `/v1/embeddings` |
| `metrics.ttft_ms` / `tpot_ms` / `itl_ms` | omitted — not applicable |
| `metrics.e2e_ms` | total request latency (send → vector returned) |
| `metrics.output_toks_per_s` | `0.0` — output is a 4096-dim vector, not tokens |
| `metrics.request_throughput` | headline metric (req/s) |

## SLO targets

Per `specs/qwen3-embedding-8b-hyperpod.md` §Stage-6 "Per-workload success criteria":

- `request_throughput_min`: **100 req/s** at peak concurrency
- `error_rate_max`: **0.001** (0.1%)

Burn-in adds a stability gate: `drift_pct_max: 2.0` over 1h post-warmup — the run logged **+2.49% positive drift** (not degradation), so the strict gate fails but the throughput actually improved over time.

## Regenerating

All seven files are produced deterministically (minus UUIDs) from the raw results by:

```bash
python3 scripts/emit_common_artifacts.py
```

The script reads:
- `results/smoke-bench.json`
- `results/workload-rag-qa.json`
- `results/workload-production-mix.json`
- `results/workload-long-context.json`
- `results/burn-in/burn-in-final.json`
- `results/tier-comparison/workload-rag-qa-t0-baseline.json`
- `results/tier-comparison/workload-rag-qa-t5-optimized.json`

and writes into `results/artifacts/`. Each file also carries an `extensions.raw_tool_output.uri` pointer back to its source.

## Why these exist

The proposal (`standards/benchmark-commons/PROPOSAL.md`) calls out this blueprint as a reference dataset for the Common Benchmark Artifact v1.0.0 spec. Publishing these files enables:

- Cross-team comparability (same `catalog_id` + `infrastructure.gpu` → directly comparable)
- Ingest by `standards/benchmark-commons/runner/compare.py`
- Publication to `awslabs/ai-on-eks` / `aws/sagemaker-hyperpod-recipes` as reference inference benchmarks for embedding workloads on A10G HyperPod
