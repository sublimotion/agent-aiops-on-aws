# DeepSeek-OCR-2 — CTO Benchmark Model #2

## Status: DRAFT (2026-05-13)

## Overview

Document-understanding VLM for bank-scale OCR (invoices, statements, KYC packets). Exercises vision-pipeline throughput, long-image latency, and DocVQA-gated quantization.

Parent: `cto-benchmark-engagement.md`.

## Components

### 1. Compute

- **Platform**: EKS 1.32
- **Instance**: `g6e.12xlarge` (1× L40S-96GB) primary; `p5en.48xlarge` for multi-model co-host cell
- **Region**: us-east-2

### 2. Model

- **Model ID**: `deepseek-ai/DeepSeek-OCR-2`
- **Modality**: vision-language (image in, tokens out)
- **Format**: BF16 baseline; FP8 and INT8 for Pareto
- **Serving**: vLLM with VLM support (`--max-image-tokens`); fallback SGLang if vLLM path unstable
- **Deployment card**: run `mdc get deepseek-ocr-2 --engine vllm` before deploying; the card's `tiers:` block carries canonical per-tier configs. Use `mdc tiers:refresh` to validate against latest upstream.

### 3. Networking / Storage

Standard; S3 mirror of DocVQA validation split for O3 quality gate.

## Benchmark matrix

| O# | Workload card | Sidecar axes | Expected cells |
|----|---------------|--------------|----------------|
| O1 | `concurrency-sweep` | `context_lengths` = image-token budget (VLM); precision {BF16, FP8, INT8} × paged-KV {on, off} | TBD — depends on VLM context ranges |
| O2 | `cohost-isolation` | Co-host with Qwen3.5 LLM + embedding + reranker + voxtral (sidecar supplies ensemble) | 4 topologies × 5 noisy-neighbour roles = 20 |
| O3 | `quantization-pareto` | `--quality-eval docvqa`; DocVQA tolerance = 0.03 ANLS; one sidecar per precision | 3 precisions; failing omits throughput row |
| O9 | `cold-start` | source × format × fs_cache; record image-encoder vs LLM load times separately | TBD — VLM format matrix |
| O11 | `power-efficiency` | `--load-fraction {0.25, 0.50, 0.75, 1.0}` × precision | 4 × 3 = 12 |

### Practitioner workloads (beyond the CTO matrix)

Cards to run beyond the engagement matrix for real deployment characterization:

| Workload | Card | VLM-specific notes |
|----------|------|--------------------|
| RAG Document Q&A | `rag-qa` | Image-as-context variant of RAG; record image token count in sidecar |
| Shared System Prompt | `shared-prefix-multitenant` | OCR persona shared across 4-16 tenants |
| Production Traffic Mix | `production-mix` | Trace replay — needs redacted document images |
| Long Context Scaling | `concurrency-sweep` with `context_lengths: [2K, 4K, 8K, 16K]` (image-token budget) | Subset of O1 focused on pre-production range |

## Quality baselines

```yaml
quality_baselines:
  docvqa:
    bf16: TBD           # ANLS on DocVQA validation
    tolerance: 0.03
```

## Verification criteria

Standard template Stages 4a–7 apply. Engagement-specific additions:

- [ ] O3 DocVQA gate passes for every precision before throughput row is recorded
- [ ] Image-encoder cold-start broken out from LLM cold-start in `cold_start.weights_load_s`
- [ ] Tier Stack Table filled (T0–T5) per `docs/optimization-stack.md`

## Known limitations

- VLM KV cache scales with image resolution; O1 cell uses a fixed image size from Appendix A calibration set.
- DeepSeek-OCR-2 may not be supported by vLLM main at engagement start — fallback to SGLang or deepseek-VL-chat container documented in `lessons.md`.

## Links

- Parent: `cto-benchmark-engagement.md`
- No prior repo evidence for this model
