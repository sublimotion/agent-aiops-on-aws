# CTO Infrastructure Benchmark Engagement

## Status: DRAFT (2026-05-13)

## Overview

Umbrella spec for the CTO Infrastructure team's structured GPU benchmark engagement. Covers five production models × eleven test objectives across AWS, on-prem, and CSP-equivalent environments, producing a standard JSON manifest per run.

**Source**: `AWS_WorkDay/Benchmarks/GPU Benchmarking - Test Cases.docx` (CTO Infrastructure team brief).

**Scope**: Define the matrix, pin the harness/driver/image SHAs, point each cell at either (a) an existing blueprint's artifacts or (b) the per-model spec where the work still needs to run. Execution happens inside the per-model specs (Layer 2) using the workload cards (Layer 3).

**This spec does not deploy anything itself** — it is the contract the per-model specs satisfy.

## Models under test

| # | Model | Parameters | Spec | Primary AWS instance | Closest existing evidence |
|---|-------|------------|------|----------------------|---------------------------|
| 1 | Qwen3.5-125B-A10B MoE | 125B / 10B active | `domains/gpu-serving/specs/qwen3-5-125b-a10b.md` (new) | p5en.48xl (H200) or p6-b200.48xl | `qwen3-235b-b300` (larger), `qwen3-next` (smaller) |
| 2 | DeepSeek-OCR-2 | VLM | `domains/gpu-serving/specs/deepseek-ocr-2.md` (new) | g6e.12xl (L40S 48GB) | none |
| 3 | Qwen3-Embedding | ~8B | `domains/gpu-serving/specs/qwen3-embedding-8b.md` (new) | g6e.xlarge (L40S 48GB; FP8 needed for T1) | none |
| 4 | Qwen3-Reranker | ~4B | `domains/gpu-serving/specs/qwen3-reranker-4b.md` (new) | g6.xlarge (L4 24GB); g6e.xlarge only for FP8 cell | none |
| 5 | Mistral Voxtral | ~4B (speech) | `domains/gpu-serving/specs/mistral-voxtral-4b.md` (new) | g6.xlarge (L4 24GB); g6e.xlarge only for FP8 cell | `voice-agent-hyperpod` (DRAFT, different model) |

**Instance sizing rationale**: for 4B-class models (reranker, Voxtral), the L4-based `g6.xlarge` wins on `$/M tokens` and `tokens/joule` over the L40S-based `g6e.xlarge` (roughly 57% cheaper $/hr, ~24% of the TDP). FP8 tensor cores are L40S-only on the G-family, so `g6e.xlarge` is reserved for the single FP8 precision row in each model's O3 Pareto. The 8B embedding and VLM stay on `g6e.xlarge` because FP8 is the primary throughput lever for embeddings and image tokens inflate KV memory.

## Test objectives (O1–O11)

Each objective names its **executor** (workload card, runner module, or new investigation spec) and the **current coverage** in this repo.

| ID | Name | Workload card | Axes enumerated in sidecars | Current coverage |
|----|------|---------------|-----------------------------|------------------|
| O1 | KV cache memory scaling | `concurrency-sweep` (with `context_lengths` + `stop_condition.p99_ttft_ms`) | paged-KV on/off (engine config) × precision × context | **Substantial**: `qwen3-next`, `qwen3-235b-b300`, `glm5-lmcache`, `kimi-k2.5`, `kimi-k2.6` sweeps |
| O2 | Multi-model co-hosting | `cohost-isolation` | topology (full-share / MIG / CSP-native / siloed); sidecar supplies the 5-model banking ensemble | None |
| O3 | Quantization Pareto | `quantization-pareto` + `run-quality-eval.py` | precision (one sidecar per BF16/FP8/INT8/INT4); evals picked by modality | Partial: FP8/INT4 throughput measured; **no quality gates run yet** |
| O4 | MIG / GPU partitioning | `mig-partitioning` | MIG profile (sidecar.infrastructure.partition_profile) | None |
| O5 | 72-hour burn-in stability | `burn-in` (card defaults to 1h pre-prod soak; CTO O5 sidecar sets `duration_hours: 72`, `rate_fraction: 0.85`, banking `mix`) + `analyze-burn-in.py` | None at 72h; the 1h default can run against every existing blueprint as a standard pre-prod check |
| O6 | CUDA → ROCm migration effort | `specs/cuda-to-rocm-port.md` (new investigation spec) | N/A — not a benchmark | None |
| O7 | GPU failover and resilience | `specs/gpu-failover-resilience.md` (new investigation spec) | N/A — fault injection plan | Partial: `ray-serve-ft` for YOLOv8, not vLLM/SGLang |
| O8 | Confidential compute / TEE | `specs/tee-confidential-compute.md` (new investigation spec) | TEE on/off (two sidecars); attestation captured separately | None |
| O9 | Model cold-start time | `cold-start` | source × format × fs_cache_state (all sidecar-level) | Partial: per-blueprint lessons scattered |
| O10 | ECC / SDC sentinel | `power-efficiency` (sentinel block) + `specs/sdc-sentinel.md` (new) | sentinel prompt set (sidecar) | None |
| O11 | Power efficiency (tokens/joule) | `power-efficiency` + `scrape-power.py` | load fraction (0.25/0.50/0.75/1.0) × precision | None |

See **Evidence Reuse Table** below for the exact file paths we can lift from already-run blueprints to populate O1, O3 (partial), and O9 (partial) without re-running.

## Statistical and reproducibility rules (Appendix A)

From the doc, applied globally to every manifest in this engagement:

1. **Warm-up**: 5 minutes ignored before capture.
2. **Steady state**: 15 minutes of data per cell.
3. **Repeats**: minimum 3 runs in separate scheduling windows.
4. **Variance gate**: coefficient of variation ≤ 5% on the headline metric or the cell is re-run.
5. **Version pinning**: driver, firmware, container image SHAs frozen at engagement start. Any change → config change → affected cells re-run.
6. **No live downloads during measurement**: all datasets, weights, harness images mirrored to the CSP-side artifact store before runs begin.
7. **Quality gate precedes throughput**: if the O3 quality eval fails, no throughput row is produced for that configuration.
8. **Tier Stack Table required in every Stage 6 report**: every blueprint under this engagement records which of the six optimization tiers (T0–T5) landed, which were blocked, and the delta each delivered. Per-tier configuration recommendations come from the model deployment card's `tiers:` block (see `mdc get <model> --engine <engine>`) rather than being duplicated in the spec. Framework: `docs/optimization-stack.md`.

Rules 1–3 are enforced by `workloads/*.yaml` (`warmup_requests`, `steady_state_duration_s`, `repeats`). Rule 4 is enforced by the runner's variance check in `runner/compare.py`. Rules 5–6 are captured in the sidecar `version_pins` block. Rule 7 is enforced by the runner's Step 0a (quality gate runs before throughput capture and exits non-zero on failure unless `--no-quality-gate`). Rule 8 is a reporting requirement checked at manifest review.

## Pinned harness versions (Appendix A)

Update **both** this section and `container/run-quality-eval.py::PINNED_COMMITS` together. Any change re-qualifies every affected cell.

| Component | Pin | Source |
|-----------|-----|--------|
| MLCommons loadgen | v4.1 | GitHub `mlcommons/inference` v4.1 tag |
| MLPerf Inference reference | v4.1 | Same repo; custom extensions labeled `non-MLPerf` in manifest |
| lm-evaluation-harness | v0.4.7 | `EleutherAI/lm-evaluation-harness` v0.4.7 tag |
| MTEB | 1.38.22 | `embeddings-benchmark/mteb` pip version 1.38.22 |
| DCGM exporter | 3.3.7-3.5.0 | `nvcr.io/nvidia/k8s/dcgm-exporter:3.3.7-3.5.0-ubuntu22.04` |

## Manifest schema (Appendix B)

Every run produces one JSON artifact per `standards/benchmark-commons/container/schema/enriched-artifact.json`. The engagement extends the core schema with these optional top-level blocks, all of which are now populated by the runner when the corresponding data is available:

- `quality` — O3 quality-gate results (array of eval scores vs baselines with pass/fail).
- `power` — O11 tokens/joule, average fleet power, per-GPU power distribution.
- `hardware_errors` — O10 ECC SBE/DBE deltas, NVLink CRC, PCIe replay, sentinel divergences.
- `stability` — O5 burn-in drift percentage, thermal events, unrecoverable errors.
- `cold_start` — O9 per-phase breakdown (weights load, allocator warmup, kernel compile, CUDA graph capture).
- `cost` — Intelligence-Adjusted Cost fields (pre-existing).

Manifest submission cadence (Appendix B rules): within 24 hours of run completion; weekly readout consolidates prior week and flags cells where CV > 0.05 or `quality.gate_passed == false`.

## CSP equivalence map (Appendix C)

| On-prem behavior | AWS equivalent | Substitution note |
|------------------|----------------|-------------------|
| MIG partition | EKS node + MIG profile on p5/p5en/p6 | Record `mig_profile` in sidecar |
| MIG profile reconfiguration | ~1 min direct nvidia-smi; on EKS add pod evict + reschedule = 5–10 min | Include reconfig overhead in O4 total time; record in `extensions.partition.reconfig_time_s` |
| SR-IOV virtual function | N/A | Document as "unavailable on AWS; AMD-only cell" |
| Bare-metal host | EC2 bare-metal equivalents (`*.metal`) where available | Otherwise virtualized instance, flagged in manifest |
| PDU wall-socket measurement | DCGM + published SKU TDP proxy | `power.source = "cloud-proxy"`, delta logged |
| TEE (NVIDIA CC / SEV-SNP) | p5 with NVIDIA CC enabled; AMD not available | O8 cell skipped for AMD track on AWS |

Any substitution outside this table requires approval before the run — captured in `cloud_equivalence_note` on the manifest.

## Evidence reuse table

Before kicking off new runs, harvest these existing artifacts, re-wrap in the manifest format, and mark them `reused`:

| Objective | Model | Path | What to extract |
|-----------|-------|------|-----------------|
| O1 | Qwen3-235B-A22B | `domains/gpu-serving/blueprints/qwen3-235b-b300/results/` | Concurrency sweep JSONs (c=1→512) — maps directly to O1's concurrency ramp |
| O1 | Qwen3-Next-80B-A3B | `domains/gpu-serving/blueprints/qwen3-next/results/` | 32K/64K/128K context sweeps |
| O1 | GLM-5 | `domains/gpu-serving/blueprints/glm5-lmcache/results/` + lessons.md | HiCache vs baseline KV scaling |
| O1 | Kimi K2.6 | `domains/gpu-serving/blueprints/kimi-k2.6/results/` | Long-context TTFT curves |
| O3 (partial) | Qwen3-235B FP8 | `domains/gpu-serving/blueprints/qwen3-235b-b300/lessons.md` | FP8 throughput — **still needs MMLU/GSM8K run against the same endpoint** |
| O3 (partial) | Kimi K2.6 INT4 | `domains/gpu-serving/blueprints/kimi-k2.6/lessons.md` | INT4 QAT — **still needs quality eval** |
| O9 (partial) | GLM-5 | `domains/gpu-serving/blueprints/glm5-lmcache/lessons.md` §5 | 15 min DeepGEMM JIT startup — lift into cold_start block |
| O9 (partial) | Kimi K2.6 | `domains/gpu-serving/blueprints/kimi-k2.6/lessons.md` | 3 min vs 8.3 min (vLLM vs SGLang INT4 QAT) |
| O9 (partial) | Nemotron | `domains/gpu-serving/blueprints/nemotron-super/lessons.md` | 31s weights + 14s init + 43s compile + 9s CUDA graphs |
| O7 (partial) | Ray Serve FT | `domains/gpu-serving/blueprints/ray-serve-ft/lessons.md` | Head/worker failover primitives — **CV workload, not LLM**; use as reference, re-run for vLLM/SGLang |

## Execution plan

**Phase 1 — Quality-gate wiring and dry run** (days 1–3):
- Stand up lm-evaluation-harness against an existing endpoint (Qwen3-235B or Kimi K2.6).
- Record BF16/FP16 baselines for MMLU, GSM8K; FP8 and INT4 deltas.
- Invocation: `run-benchmark.sh --quality-eval mmlu,gsm8k` against the endpoint captures `quality.json` and gates throughput.
- This unblocks O3 for every subsequent cell.

**Phase 2 — Evidence harvest** (days 4–7):
- Re-wrap existing O1 / partial O3 / partial O9 artifacts into the manifest schema.
- This populates ~30% of the matrix before any new hardware time.

**Phase 3 — Greenfield models** (weeks 2–4):
- Deploy and O1-benchmark: DeepSeek-OCR-2, Qwen3-Embedding, Qwen3-Reranker, Mistral Voxtral.

**Phase 4 — Platform studies** (weeks 4–8):
- O5 burn-in on one node per primary model using the `burn-in` workload card (default 1h; override to 72h via sidecar `workload_overrides.duration_hours`).
- O7 failover: inject faults on the Qwen3-235B or Qwen3.5-125B endpoint.
- O11 power sweep: re-run O1 at 25/50/75/100% load fractions with `run-benchmark.sh --power-sampler auto --load-fraction <f>`.
- O4 MIG only on H200/B200; AMD / SR-IOV cells flagged as AWS-unavailable.
- O10 sentinel canary: `run-benchmark.sh --sentinel <golden-prompts.json>` alongside the O11 sweep (same sampler infrastructure, different prompt injection).

**Phase 5 — Optional / defer** (outside critical path):
- O6 (CUDA → ROCm): no AMD on AWS critical path; defer to external environment.
- O8 (TEE): p5 with NVIDIA CC; single model, single cell.

## Non-requirements

- O6 ROCm deployments (no AMD GPUs on AWS critical path).
- MLPerf Inference reference implementations are **referenced**, not re-run from scratch — we use MLCommons loadgen v4.1 but generate our own workload traces.
- Any on-prem or non-AWS CSP execution — those are the peer engagements this matrix will be compared against.

## Verification criteria (engagement-level)

- [ ] Every cell in the 5 × 11 matrix has either a manifest or an "unavailable-with-justification" note.
- [ ] Every manifest emits CV ≤ 0.05 on its headline metric.
- [ ] Every quality-gated precision row has a passing `quality.gate_passed`.
- [ ] Every driver/firmware/image SHA listed in the sidecar matches the pinned values in this spec at run time.
- [ ] Every manifest includes a Tier Stack Table stating which of T0–T5 landed, which were blocked, and the delta each delivered vs T0 (framework: `docs/optimization-stack.md`).
- [ ] Weekly readout produced, flagging cells that failed variance or quality gate.

## Known limitations

- Qwen3.5-125B-A10B is not yet released to Hugging Face as of 2026-05-13; Qwen3-Next-80B-A3B is the closest stand-in if timing forces an early start.
- DCGM power numbers on cloud bare-metal are telemetered, not wall-socket — the delta is logged but not eliminated.
- MIG cells require physical partition reconfiguration between profile changes; each reconfig costs ~5–10 min of GPU time plus pod reschedule.

## Links

- Source doc: `AWS_WorkDay/Benchmarks/GPU Benchmarking - Test Cases.docx`
- Per-model specs: Layer 2 files listed in the model table above
- Workload cards: `standards/benchmark-commons/workloads/`
- Runner CLI: `standards/benchmark-commons/runner/run-benchmark.sh` — flags `--quality-eval`, `--power-sampler`, `--burn-in-hours`, `--load-fraction`, `--sentinel` populate the quality/power/stability/hardware_errors blocks
- Runner scripts: `standards/benchmark-commons/container/scrape-power.py`, `run-quality-eval.py`, `analyze-burn-in.py`
- Seven workload cards used by this engagement: `concurrency-sweep`, `cohost-isolation`, `quantization-pareto`, `mig-partitioning`, `burn-in`, `cold-start`, `power-efficiency` — all generic; CTO-specific axes live in per-model sidecars, not in the cards
- Four practitioner-shape cards per-model specs reference beyond the CTO matrix: `multi-turn-chat`, `rag-qa`, `shared-prefix-multitenant`, `production-mix` (existing `coding-agent` covers Agent Tool Calling, extended `concurrency-sweep` covers Long Context Scaling)
- Optimization tier framework: `docs/optimization-stack.md` (T0–T5; Tier Stack Table requirement for every Stage 6 report)
- Theory + Pareto methodology: `docs/inference-optimization-guide.md` (sections 9–12: cost, methodology, KV cache, kernels)
- Model deployment card tier block: `../model-deployment-card/TIERS-PROPOSAL.md` (run `mdc get <model> --engine <engine>` for per-model tier recommendations)
- Schema: `standards/benchmark-commons/container/schema/enriched-artifact.json`
