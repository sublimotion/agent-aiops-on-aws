# Qwen3-Reranker 4B — CTO Benchmark Model #4

## Status: DRAFT (2026-05-13)

## Overview

Second-stage retrieval cross-encoder. ~4B parameters, high QPS, latency-critical. Pairs with Qwen3-Embedding for two-stage retrieval over banking document corpora. Small enough to be a natural MIG tenant.

Parent: `cto-benchmark-engagement.md`.

## Components

### 1. Compute

- **Platform**: EKS 1.32
- **Primary instance**: `g6.xlarge` (1× L4 24GB, Ada). 4B BF16 weights ~8 GB leaves ~3× VRAM headroom for KV cache and concurrent pairs. L4 wins on $/M-tokens and tokens/joule for this model class (TDP 72 W vs 300 W on L40S).
- **FP8-cell instance**: `g6e.xlarge` (1× L40S 48GB, Ada). Required only for the O3 FP8 precision row — L4 lacks FP8 tensor cores. Other precisions (BF16/INT8/INT4) still run on `g6.xlarge`.
- **MIG-cell instance**: `p5en.48xlarge` (1× H200 in a single 1g.10gb MIG slice) for the O4 MIG sweep. MIG is not available on Ada (L4/L40S).
- **Region**: us-east-2

### Cost / performance rationale

On-demand price delta at engagement time (update at run start): `g6.xlarge` ~$0.80/hr vs `g6e.xlarge` ~$1.86/hr. At 4B parameters, L4 per-stream decode throughput is ~70–80% of L40S; given the price delta, L4 wins on `$/M tokens` and `tokens/joule`. Running non-FP8 cells on L4 saves ~57% of the instance bill for this model's share of the matrix.

### 2. Model

- **Model ID**: `Qwen/Qwen3-Reranker-4B`
- **Modality**: cross-encoder scoring (query + candidate → relevance score)
- **Format**: BF16 baseline; FP8, INT8, INT4 Pareto
- **Serving**: vLLM classifier mode or custom rerank handler
- **Deployment card**: run `mdc get qwen3-reranker-4b --engine vllm` before deploying; the card's `tiers:` block carries canonical configs. Use `mdc tiers:refresh` to validate against latest upstream.

## Benchmark matrix

| O# | Workload card | Sidecar axes | Expected cells |
|----|---------------|--------------|----------------|
| O1 | `concurrency-sweep` | Query×candidate concurrent; sweep batch size vs pair-length | TBD |
| O2 | `cohost-isolation` | Reranker role in full-ensemble; small model fits every slice | 4 topologies × 5 roles = 20 |
| O3 | `quantization-pareto` | `--quality-eval fiqa`; one sidecar per precision | 4 precisions (BF16/FP8/INT8/INT4) |
| O4 | `mig-partitioning` | **Primary O4 target**; one sidecar per `infrastructure.partition_profile` | 5 NVIDIA MIG profiles |
| O9 | `cold-start` | source × format × fs_cache; expect sub-10s cold start | TBD |
| O11 | `power-efficiency` | `--load-fraction` × precision; reranker anchors idle-power baseline | 4 × 4 = 16 |

### Practitioner workloads (beyond the CTO matrix)

Reranker is second-stage retrieval — most shape-specific cards don't apply. Relevant ones:

| Workload | Card | Reranker-specific notes |
|----------|------|-------------------------|
| RAG Document Q&A | `rag-qa` | Runs in "rerank mode" — measures query+candidate-pair throughput |
| Shared System Prompt | `shared-prefix-multitenant` | Not applicable — reranker has no system prompt |
| Production Traffic Mix | `production-mix` | Trace replay of production retrieval queries |
| Long Context Scaling | `concurrency-sweep` with `context_lengths: [512, 1024, 2048, 4096]` | Pair-length sweep for document-chunk reranking |

## Quality baselines

```yaml
quality_baselines:
  fiqa:
    bf16: TBD           # MTEB main_score for reranking
    tolerance: 0.015
```

## Verification criteria

Standard template Stages 4a–7 apply. Engagement-specific additions:

- [ ] O4 MIG sweep covers 1g.10gb (7 slices) through 7g.80gb (1 slice); aggregate QPS recorded per profile
- [ ] Reconfiguration time between MIG profiles measured (pod evict + reschedule + health-check)
- [ ] FiQA MTEB reranking gate passes per precision
- [ ] Tier Stack Table filled (T0–T5) per `docs/optimization-stack.md`

## Known limitations

- MIG is p5/p5en only on AWS; no B200 MIG support today.
- Cross-encoder throughput depends on candidate count per query; workload card fixes this at k=50.

## Links

- Parent: `cto-benchmark-engagement.md`
- Companion: `qwen3-embedding-8b.md`
