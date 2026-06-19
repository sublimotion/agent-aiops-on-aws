# DeepSeek-OCR-2 on EKS (g6e / L40S) — Deployment Spec

## Status: DRAFT v2 (2026-05-14)

## Overview

Deploy `deepseek-ai/DeepSeek-OCR-2` (document-understanding VLM) on managed EKS in `us-east-2`, targeting the **g6e instance family** (NVIDIA L40S, 48 GB per GPU, sm_89, FP8 tensor cores). This is the **EKS variant** of the CTO `deepseek-ocr-2.md` spec; an earlier draft targeted HyperPod in us-east-1 but g6e capacity was unavailable in `use1-az1` for 25+ min, so the engagement pivoted to us-east-2 EKS where capacity landed in ~2 min.

Parent CTO spec: `deepseek-ocr-2.md`. **This spec is the EKS-infrastructure row only** — same model + engine + workload axes; different substrate.

### Optimization Objective

```
Primary (latency):    VLM p99 TTFT and e2e for single-page OCR, streaming chat.
Primary (throughput): Pages/s and $/1K pages for offline batch document ingest.
Secondary:            Quantization Pareto (BF16, FP8, INT8) gated by olmOCR-bench score.
                      Tier Stack Table T0-T5 delta per cell.
Metric axes:          request_throughput, ttft_p99_ms, tokens_per_joule, $/1K pages.
```

**Why two headline axes**: an OCR VLM has two distinct operating modes that optimize differently. Latency mode (user-facing document scanner, chat interface) cares about TTFT and e2e per request. Throughput mode (nightly PDF backfill, KYC batch pipeline) cares about pages/$ and can accept multi-second request latency. Workload cards and tier choices diverge between them (see Workload Selection below).

## Compute (actual substrate)

| Field | Value |
|-------|-------|
| Platform | Amazon EKS (managed nodegroup) |
| Region / AZ | `us-east-2` / `us-east-2a` (landed), `2b/2c` available |
| Cluster | `qwen3-next-bench-eks-cluster` (reused; hosts p5en nodegroups for other blueprints) |
| Nodegroup | `g6e-ocr` - instance mix `[g6e.2xlarge, g6e.xlarge]`, ON_DEMAND, min/max/desired = 0/1/1 |
| VPC | `vpc-0490a5031a96f53dd` |
| Subnets | `subnet-0fced510ea62b874e` (az1), `subnet-03d03f1fb8d62d6a5` (az2), `subnet-04be09c7bf104edb8` (az3) - all 3 AZs added to nodegroup to maximise capacity hit |
| AMI | `AL2_x86_64_GPU` (managed, driver-bundled) |
| Disk | 200 GB gp3 |
| Labels | `role=gpu`, `nvidia.com/gpu.present=true`, `workload=deepseek-ocr` |
| Taint | `nvidia.com/gpu=true:NoSchedule` |
| Node (first) | `ip-10-0-10-217.us-east-2.compute.internal` (g6e.2xlarge, us-east-2a, Ready) |

### Instance options (g6e family)

| Instance | GPUs x VRAM | vCPU / RAM | Aggregate VRAM | Role in this spec |
|----------|:-----------:|-----------:|---------------:|-------------------|
| `g6e.xlarge` | 1x L40S 48GB | 4 / 32 GiB | 48 GB | Smallest-cell economics (capacity fallback) |
| **`g6e.2xlarge`** | 1x L40S 48GB | 8 / 64 GiB | 48 GB | **Primary single-GPU cell** - CPU headroom for image preprocessing |
| `g6e.4xlarge` | 1x L40S 48GB | 16 / 128 GiB | 48 GB | Single-GPU, larger batch preprocessing |
| `g6e.8xlarge` | 1x L40S 48GB | 32 / 256 GiB | 48 GB | CPU-heavy preprocessing (PDF rasterization, multi-page fan-out) |
| `g6e.12xlarge` | 4x L40S 48GB PCIe | 48 / 384 GiB | 192 GB | TP=4 cell - long image-token budgets + O2 co-host |
| `g6e.16xlarge` | 1x L40S 48GB | 64 / 512 GiB | 48 GB | Optional, high host-RAM edge case |
| `g6e.24xlarge` | 4x L40S 48GB PCIe | 96 / 768 GiB | 192 GB | TP=4, CPU-heavy variant |
| `g6e.48xlarge` | 8x L40S 48GB PCIe | 192 / 1536 GiB | 384 GB | TP=8 scaling ceiling + O11 power sweep at scale |

**Sizing rules**:
- `g6e.xlarge` through `g6e.8xlarge` all carry **one** L40S 48GB. Scaling up within the single-GPU rungs trades $/hr for CPU/RAM headroom, not GPU capacity.
- `g6e.12xl/24xl/48xl` are the multi-GPU rungs - 4x, 4x, 8x L40S - all **PCIe** (no NVSwitch on the g-family).
- Multi-GPU TP efficiency on g6e caps ~0.65-0.75 of linear scale (PCIe-only interconnect; document as measured floor, not a bug).

### Primary cell selection (minimum viable benchmark)

If capacity forces a reduced run, execute in this order:
1. `g6e.2xlarge` - BF16 single-GPU latency + smoke
2. `g6e.xlarge` - smallest-cell economics (skip if capacity-tight)
3. `g6e.12xlarge` - TP=4 long-context + throughput ceiling
4. `g6e.8xlarge` - CPU-bound regression detection (PDF pre-processing)
5. `g6e.48xlarge` - TP=8 throughput ceiling (only if engagement matrix demands it)

## Model

- **Model ID**: `deepseek-ai/DeepSeek-OCR-2`
- **Modality**: vision-language (image in -> tokens out)
- **Format**: BF16 baseline (~8 GB weights + image-encoder overhead); FP8 + INT8 for O3 Pareto
- **Serving**: vLLM >= 0.19.1 (stable VLM path); SGLang fallback documented in `lessons.md` if vLLM VLM path breaks
- **Context**: image-token budget `[1K, 2K, 4K, 8K, 16K]` - swept in O1 / Long Context
- **`--max-image-tokens`**: set per cell in the sidecar
- **Trust-remote-code**: yes
- **Deployment card**: run `mdc get deepseek-ocr-2 --engine vllm` before first deploy; contribute an EKS section once spec completes

### Prompt template (mandatory)

The HF repo does **not** ship a chat template. vLLM loads a generic fallback that produces degenerate output (`"1. 1. 1. ..."` loop). Use one of the two upstream prompt shapes, injected as raw user-message text content (alongside the `image_url` content item):

| Mode | Prompt text | When to use |
|------|-------------|-------------|
| Plain OCR | `<image>\nFree OCR. ` | Flat transcription, no layout structure |
| Layout-aware | `<image>\n<\|grounding\|>Convert the document to markdown. ` | Document understanding — headers, tables, reading order. Required for olmOCR-bench + OmniDocBench. |

Both prompts must include the literal trailing space. The `<image>` sentinel is replaced by vLLM with the image-token expansion when an `image_url` content item is present in the same message.

**Client integration**: benchmark harnesses and production clients must send these as the `text` content part of a `/v1/chat/completions` user message. Do **not** rely on a server-side `--chat-template` flag — Path B testing showed prompt-as-text works cleanly with the default fallback template and avoids ConfigMap + Deployment restart.

## Networking / Storage

- Reuse EKS cluster VPC + security groups
- Internal ClusterIP service; benchmark via `kubectl port-forward` or cluster-internal client
- **Image ingress**:
  - olmOCR-bench corpus mirrored to `s3://agent-aiops-bench-us-east-2/datasets/olmocr-bench/` (1,400-doc standard set; primary O3 gate)
  - OmniDocBench mirrored to `s3://agent-aiops-bench-us-east-2/datasets/omnidocbench/` (includes `gt.json` + source PDFs; periodic deeper eval)
  - No live downloads during measurement (Appendix A rule 6)
- HF cache in emptyDir (80 GiB) on the 200 GB root EBS
- `/dev/shm` 8 GiB for vLLM worker IPC + image-token tensors

## Non-Requirements

- **O2 multi-model co-hosting** scoped only to `g6e.12xl+` (4x or 8x L40S) - single-GPU cells don't have VRAM headroom for the 5-model ensemble.
- **No HyperPod** - moved to stock EKS managed nodegroup after us-east-1 g6e capacity failure. Auto-recovery = EKS replace, not HyperPod deep-health-check.
- **No MIG** (MIG cells run on H200/B200 per parent spec).
- **No TEE** (O8) - NVIDIA CC is p5-only.
- **No CUDA -> ROCm port** (O6).
- **Virtualized only** - g6e isn't offered as bare-metal.
- **No DocVQA**: dropped from the O3 quality gate as of v2. DocVQA is a VQA benchmark (answer a question about a document) rather than an OCR fidelity benchmark — it confounds OCR accuracy with instruction-following. olmOCR-bench + OmniDocBench measure what we actually care about (character accuracy, reading order, table / formula structure) and map directly to published DeepSeek-OCR-2 numbers.

---

## Workload Selection: Batch vs Latency Split

An OCR VLM has two operating modes with genuinely different optimal configs. Every workload card below is tagged as **L** (latency-critical), **B** (batch-throughput), or **Both**.

### Why two modes matter for DeepSeek-OCR-2

| Dimension | Latency mode | Batch mode |
|-----------|--------------|------------|
| Typical client | Document scanner app, chat UI, KYC real-time check | Nightly PDF backfill, historical archive ingest |
| Concurrency | 1-32 | 256-max |
| Batch size | 1-4 pages/req | 32-128 pages/req |
| Headline metric | p99 TTFT, p99 e2e | pages/s, $/1K pages |
| TTFT target | < 500 ms @ 4K image tokens | best-effort |
| e2e target | < 5 s per page | < 30 s per request |
| Tier that dominates | T2 (prefix cache for shared OCR persona), T5 (compile + CUDA graphs) | T1 (FP8 quant), T4 (TP=4 on 12xl) |
| Output mode | Streaming | Non-streaming (batched) |

### Workload cards applied to DeepSeek-OCR-2

From `standards/benchmark-commons/workloads/*.yaml` - 17 cards exist; the subset that maps to an OCR VLM:

| Workload card | Mode | Cell(s) | What it measures for OCR | CTO O-code |
|---------------|:----:|---------|--------------------------|:----------:|
| `concurrency-sweep` | **Both** | g6e.2xl (latency range c=1->32); g6e.12xl (throughput c=16->256) | Operating point for each mode; context axis via `context_lengths` for image-token budget | O1 |
| `chatbot-short` | L | g6e.2xl | Interactive OCR chat - ~256 image-tokens in, 128 out, 2 QPS; p99 TTFT gate | O1 (latency slice) |
| `rag-qa` | L | g6e.2xl | RAG document Q&A with image-as-context - 2K-10K retrieved context + vision tokens | - |
| `batch-throughput` | **B** | g6e.2xl, g6e.12xl, g6e.48xl | Offline PDF backfill - max-rate open loop; error < 0.1% @ load | O1 (throughput slice) |
| `qps-sweep` | Both | g6e.2xl | Production mix between batch and latency, sweep 0.5->16 QPS | O1 |
| `production-mix` | Both | g6e.2xl | Trace replay - realistic distribution of short scans / long invoices | - |
| `quantization-pareto` | Both | g6e.2xl | BF16 / FP8 / INT8; **olmOCR-bench** gated per row (primary); OmniDocBench periodic | **O3** |
| `cold-start` | - | g6e.xl, g6e.2xl, g6e.12xl | Image-encoder vs LLM decoder startup time breakdown | **O9** |
| `burn-in` | B | g6e.2xl | 1h at 85% of peak; drift <= 2%, 0 unrecoverable errors | **O5** |
| `power-efficiency` | Both | g6e.2xl, g6e.48xl | tokens/joule at 4 load fractions x precision | **O11** |
| `cohost-isolation` | - | g6e.12xl | Co-host OCR with Qwen3.5 LLM + embedding + reranker + voxtral | O2 (12xl+ only) |

### Cards explicitly **not** run for this spec

- `chatbot-long` - OCR doesn't do 32K-token outputs; long-context is an **image-token** axis, covered by `concurrency-sweep` with `context_lengths`
- `coding-agent` - no code tool-calling shape for OCR
- `multi-turn-chat` - not the production pattern (document scans are stateless)
- `shared-prefix-multitenant` - interesting but deferred to the parent EKS spec; not on critical path for this engagement
- `mig-partitioning` - MIG not in scope on g6e (H200/B200 only)

### Execution order

1. **Smoke** - chatbot-short BF16 single-GPU on g6e.2xl. Confirms serving pipeline end-to-end.
2. **Latency sweep** - concurrency-sweep c=1->32 on g6e.2xl, BF16, 4K image-tokens. Establishes TTFT / e2e baseline.
3. **O3 quality gate** - quantization-pareto BF16 -> FP8 -> INT8 on g6e.2xl. **olmOCR-bench must pass before any throughput row is published** per Appendix A rule 7. OmniDocBench runs on BF16 + the winning FP8 row for deeper structural validation.
4. **Batch throughput** - batch-throughput on g6e.2xl (BF16 + FP8 after gate); extend to g6e.12xl TP=4 if capacity allows.
5. **Cold-start** - image-encoder vs LLM decoder breakdown on xl / 2xl / 12xl.
6. **Burn-in** - 1h on g6e.2xl at 85% peak (72h deferred per engagement Phase 4).
7. **Power sweep** - O11 at 0.25/0.50/0.75/1.00 load fractions on g6e.2xl.

## Optimization Tier Stack (required per engagement Rule 8)

Every Stage 6 report for this spec must fill out the Tier Stack Table below using deltas measured against T0 on the **same cell**. Framework reference: `docs/optimization-stack.md`.

### Tier plan (batch vs latency)

| Tier | Config | Batch mode plan | Latency mode plan |
|------|--------|-----------------|-------------------|
| **T0** Baseline | eager, no prefix cache, default attention | Required reference | Required reference |
| **T1** Quantization | FP8 (L40S sm_89 FP8 TC); INT8 fallback | **Primary lever** - biggest throughput / $ gain; O3 gated on olmOCR-bench | Secondary - FP8 delta smaller at low concurrency; still gate on olmOCR-bench |
| **T2** KV / prefix cache | vLLM `--enable-prefix-caching` + optional LMCache | Minor - each page is a fresh prompt | **Primary lever** - system prompt / OCR persona shared; large TTFT win |
| **T3** Speculative decode | Draft-model (not MTP - model-specific) | **Skip** - page content is novel, low acceptance rate | Evaluate only if a lightweight draft <= 1B is available; likely skipped |
| **T4** Parallelism | TP / PP / DP | **TP=4 on g6e.12xl** - throughput scaling when single-GPU saturates | TP=1 preferred; PP has no benefit for single-step VLM decode |
| **T5** Kernel / compile | torch.compile (inductor) + CUDA graphs + FLASH_ATTN | Required (compounds with T1) | Required - landed T5 is the "last 15%" for single-GPU latency |

### Required Tier Stack Table (per benchmark report)

```
| Tier | Config landed | Metric @ target cell | Delta vs T0 | Notes |
|------|---------------|----------------------|-------------|-------|
| T0   |                                                              |
| T1   |                                                              |
| T2   |                                                              |
| T3   |                                                              |
| T4   |                                                              |
| T5   |                                                              |
```

Record deltas on the mode-appropriate headline metric:
- **Batch rows**: `pages/s` (or tokens/s equivalent) at fixed concurrency
- **Latency rows**: `p99 ttft_ms` and `p99 e2e_ms` at c=1 and c=32

Blocked tiers must be called out explicitly - "T3 blocked: no draft model published; acceptance rate on novel page tokens < 0.3 in pilot."

### Known tier conflicts on g6e L40S

- **T1 x T3**: FP8 draft models are scarce for VLMs; skipping T3 is the default
- **T2 x T1**: LMCache + FP8 KV is validated for LLMs (see MEMORY); VLM KV tier support is untested - measure before enabling
- **T4 x g6e interconnect**: PCIe-only TP=4 on g6e.12xl caps at ~0.65-0.75 linear; document as expected floor
- **T5 compile time**: DeepGEMM / torch.compile can add 10-15 min to cold-start (see GLM-5 on B200 in MEMORY) - measure in O9 breakdown

## Verification criteria

### Stage 4a - GPU Health
- [ ] Node in nodegroup `g6e-ocr` reports `nvidia.com/gpu=<count>` allocatable
- [ ] `nvidia-smi` on the node (via debug pod) shows L40S 48GB, ECC on, 0 uncorrected
- [ ] For 12xl/24xl/48xl cells: `nvidia-smi topo -m` confirms PCIe-only (no NVSwitch)
- [ ] `dmesg | grep Xid` empty

### Stage 5 - Serving Stack
- [ ] vLLM deployment applied; pod reaches Ready within 15 min (weights + image-encoder first-boot download)
- [ ] `/health` returns 200
- [ ] Test `/v1/chat/completions` with one image attached **and the DeepSeek-OCR prompt shape injected as user-message text** returns non-degenerate transcription (not a repeated-token loop)

### Stage 6 - Benchmark
- [ ] One enriched artifact per cell under `blueprints/deepseek-ocr-2-eks/results/artifacts/`, validated against `schema/enriched-artifact.json`
- [ ] Every O3 row has a passing `quality.gate_passed` (olmOCR-bench) before the throughput row is emitted
- [ ] Every manifest's sidecar pins driver / container_image / DCGM SHAs per Appendix A
- [ ] **Tier Stack Table** filled per cell (T0-T5 landed/blocked/delta)
- [ ] Batch cells report `pages/s` and `$/1K pages`; latency cells report `p99 TTFT / e2e` at c=1 and c=32

### Stage 7 - Readiness Audit
- [ ] All Stage 6 cells have artifacts committed
- [ ] `lessons.md` captures cross-cell lessons (cold-start split, FP8 olmOCR-bench delta, TP=4 PCIe efficiency, batch/latency tier split)
- [ ] Nodegroup `g6e-ocr` scaled to desired=0 after benchmarks complete

## Cost considerations

Approximate us-east-2 ON_DEMAND (May 2026):

| Instance | $/hr | Role |
|----------|-----:|------|
| g6e.xlarge | ~1.86 | smallest-cell economics |
| **g6e.2xlarge** | **~2.24** | **primary** - full suite ~4 h = $9 |
| g6e.4xlarge | ~3.00 | CPU headroom |
| g6e.8xlarge | ~4.54 | CPU-heavy preprocessing |
| g6e.12xlarge | ~10.49 | TP=4 |
| g6e.48xlarge | ~30.13 | TP=8 ceiling |

**Expected burn**: ~$75-100 full matrix end-to-end. 1h burn-in on g6e.2xl adds ~$2.25. O11 power sweep ~4 x 15 min = $2.25.

**Cost hygiene**: nodegroup `g6e-ocr` scales to desired=0 immediately after cells complete. Cluster control plane (~$0.10/hr) + m6i system node pair (~$0.19/hr) are shared costs already borne by the cluster.

## Known limitations

- **vLLM VLM path stability**: DeepSeek-OCR-2 may lag vLLM mainline; pin image + commit in sidecar. SGLang fallback documented in `lessons.md`.
- **Chat template absent upstream**: HF repo ships no `chat_template.jinja`; vLLM fallback produces degenerate output. Mitigation is client-side prompt-as-text (see Model / Prompt template section). Do not rely on server-side `--chat-template` — it works but requires a Deployment restart and a ConfigMap, and the client-side path has zero downsides.
- **g6e capacity volatility**: us-east-1 was unavailable at spec-draft time (2026-05-13); us-east-2 landed in ~2 min. EKS nodegroup retries ASG, so transient capacity blips are handled; persistent shortage requires switching AZ or region.
- **PCIe-only TP**: multi-GPU cells cap at ~0.65-0.75 linear; 4x L40S on g6e.12xl does **not** match 4x H100 on p5 at TP=4.
- **olmOCR-bench / OmniDocBench mirror**: both must be pre-staged to S3 per Appendix A rule 6; cold runs invalidate the O3 gate.
- **Image-encoder cold-start**: expected to dominate O9 (~30-60s typical for vision encoders).
- **EKS vs HyperPod**: no deep-health-checks, no auto-recovery with replacement node; this is the trade-off of pivoting off HyperPod. For the engagement's O5 burn-in the 1h window is achievable; 72h is deferred to a HyperPod cluster.

## Appendix A — Quality harnesses

Two harnesses gate the O3 quantization Pareto. Both operate on pre-mirrored S3 corpora and record one row per (precision, harness) into the benchmark sidecar's `quality_baselines` block.

### olmOCR-bench (primary gate)

- **Source**: `allenai/olmocr` on HF + `pip install olmocr[bench]`
- **Corpus**: 1,400-doc standard set across 7 subscores: `arxiv_math`, `old_scans`, `tables`, `headers_footers`, `multi_column`, `long_tiny_text`, `base`
- **Published baseline**: DeepSeek self-reports **76.3** overall on this benchmark — direct comparability
- **Prompt**: grounding mode (`<image>\n<|grounding|>Convert the document to markdown. `)
- **Gate**: weighted overall score within `tolerance: 0.02` of the seeded BF16 baseline
- **Runner**: `scripts/run-olmocr-bench.sh`
- **S3 mirror**: `s3://agent-aiops-bench-us-east-2/datasets/olmocr-bench/`

### OmniDocBench (periodic deeper eval)

- **Source**: `opendatalab/OmniDocBench`
- **Runner image**: `ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204` (structured metrics)
- **Components**: `text_edit` (edit distance), `formula_cdm` (CDM), `table_teds` (TEDS), `layout`, `reading_order`
- **Published baseline**: DeepSeek-OCR-2 community score **90.25** on the leaderboard
- **Cadence**: runs on BF16 + the winning FP8 row — not every row. Used to confirm structural fidelity (tables, formulae, reading order), which flat character-accuracy scores miss.
- **Runner**: `scripts/run-omnidocbench.sh`
- **S3 mirror**: `s3://agent-aiops-bench-us-east-2/datasets/omnidocbench/` (contains `gt.json` + source documents)

### Harnesses explicitly NOT in scope

- **DocVQA**: VQA task (answer questions about a document), not OCR fidelity. Confounds instruction-following with transcription accuracy and does not map to published DeepSeek-OCR-2 numbers. Removed from the O3 gate in v2; kept out of VQA smoke runs as well.

## Links

- Parent CTO spec: `cto-benchmark-engagement.md`
- Parent model spec: `deepseek-ocr-2.md` (generic EKS)
- Sibling EKS spec with comparable sizing logic: `qwen3-embedding-8b-hyperpod.md`
- Workload cards: `standards/benchmark-commons/workloads/*.yaml`
- Artifact schema: `standards/benchmark-commons/container/schema/enriched-artifact.json`
- Optimization tier framework: `docs/optimization-stack.md`
- Blueprint dir: `domains/gpu-serving/blueprints/deepseek-ocr-2-eks/`
