# Qwen3-Embedding 8B — CTO Benchmark Model #3

## Status: DRAFT (2026-05-13)

## Overview

Retrieval backbone for banking RAG. ~8B embedding model; latency-critical. First-stage retrieval over ~100M vectors. Exercises high-QPS embedding throughput, MTEB-gated quantization, and multi-model co-tenancy.

Parent: `cto-benchmark-engagement.md`.

## Components

### 1. Compute

- **Platform**: EKS 1.32
- **Instance**: `g6e.xlarge` (1× L4-24GB) primary; `inf2.xlarge` for ROCm/Neuron comparison if required by engagement
- **Region**: us-east-2

### 2. Model

- **Model ID**: `Qwen/Qwen3-Embedding-8B`
- **Modality**: text embeddings (chunked input → fixed-dim vector)
- **Format**: BF16 baseline; FP8 and INT8 Pareto (INT4 skipped — quality collapse typical for embeddings)
- **Serving**: vLLM embeddings endpoint (`--task embed`) or TEI (Text Embeddings Inference) — decided per engine-comparison cell
- **Deployment card**: run `mdc get qwen3-embedding-8b --engine vllm` before deploying; the card's `tiers:` block carries canonical configs. Use `mdc tiers:refresh` to validate against latest upstream.

## Benchmark matrix

| O# | Workload card | Sidecar axes | Expected cells |
|----|---------------|--------------|----------------|
| O1 | `concurrency-sweep` | No KV cache for embeddings — sweep batch size vs input length; record `peak_vram_gb` | TBD — depends on batch ceiling |
| O2 | `cohost-isolation` | Embedding role; sidecar declares ensemble | 4 topologies × 5 roles = 20 |
| O3 | `quantization-pareto` | `--quality-eval banking77,fiqa`; MTEB main_score tolerance = 0.01; one sidecar per precision | 3 precisions (BF16/FP8/INT8) |
| O9 | `cold-start` | source × format × fs_cache (small model; expect safetensors load to dominate) | TBD |
| O11 | `power-efficiency` | `--load-fraction` sweep × precision; extension reports vectors/joule too | 4 × 3 = 12 |

### Practitioner workloads (beyond the CTO matrix)

Most of the multi-turn / tool-calling cards don't apply to embeddings (no generation). The shape-relevant ones:

| Workload | Card | Embedding-specific notes |
|----------|------|--------------------------|
| RAG Document Q&A | `rag-qa` | Runs in "embed mode" — measures chunk throughput at retrieval-time document lengths |
| Shared System Prompt | `shared-prefix-multitenant` | Not applicable — embeddings don't have a system prompt |
| Production Traffic Mix | `production-mix` | Trace replay of document-ingest workload |
| Long Context Scaling | `concurrency-sweep` with `context_lengths: [1K, 2K, 4K, 8K]` | Measures chunk-size vs throughput for ingest pipelines |

## Quality baselines

```yaml
quality_baselines:
  banking77:
    bf16: TBD           # MTEB main_score
    tolerance: 0.01
  fiqa:
    bf16: TBD           # MTEB main_score
    tolerance: 0.015
```

## Verification criteria

Standard template Stages 4a–7 apply. Engagement-specific additions:

- [ ] Throughput reported in both tokens/s and embeddings/s
- [ ] MTEB gate passes per precision before throughput row is recorded
- [ ] Max batch size cell reached without quality drift > tolerance
- [ ] Tier Stack Table filled (T0–T5) per `docs/optimization-stack.md`

## Known limitations

- The enriched artifact schema is LLM-token-centric; for embeddings, `metrics.total_output_tokens` will represent output **dimensions written**, not generated tokens. Flagged in `workload.use_case = "embeddings"`.
- INT4 typically degrades retrieval quality severely — plan to skip rather than fail through the gate.

## Links

- Parent: `cto-benchmark-engagement.md`
- Companion: `qwen3-reranker-4b.md` (second-stage retrieval, same MTEB scaffolding)
