# Proposal 001: Common Benchmark Artifact for LLM Inference

**Status**: PROPOSAL
**Version**: 1.1
**Date**: 2026-04-22 (v1.0), 2026-05-07 (v1.1 tool-agnostic rewrite), 2026-05-14 (v1.1 repo alignment)
**Scope**: EKS, HyperPod

---

## Problem

Multiple teams benchmark LLM inference using different tools — vLLM `bench serve`, SGLang `bench_serving`, NVIDIA AIPerf, genai-bench, guidellm, LLMPerf, recon-perf — each producing incompatible output formats. Results sit in local JSON files that nobody else can read. There is no way to:

- Compare results across teams or substrates (EKS vs HyperPod)
- Publish reproducible artifacts to community repos (ai-on-eks, RECON, SMHP recipes)
- Build cross-team dashboards, a compatibility matrix, or regression detection
- Evaluate SLO compliance using a common contract

Rather than building yet another benchmark tool or maintaining adapters for every existing tool, we define a **common artifact format** — a context envelope and workload catalog — that wraps any benchmark tool's output with the deployment metadata needed for cross-team sharing. Teams keep their preferred tool; the enrichment layer normalizes their output into a portable, comparable artifact.

**Load generator gaps**: External load generators cannot observe engine-internal metrics like KV cache utilization, prefix cache hit rates, speculative decoding acceptance rates, or LMCache tier stats. These require companion scraping scripts that pull from the engine's Prometheus `/metrics` endpoint during the benchmark run. The enrichment wrapper attaches these as optional extensions alongside the tool's core output.

**Why cross-team sharing matters**: Each team benchmarking LLM inference has different goals — cost optimization, capacity sizing, latency SLO validation, feasibility comparison against managed APIs — but they all run on the same scarce GPU fleet. A single benchmark session on a p6-b200 produces data valuable to multiple teams simultaneously. Without a common format, that data stays siloed. The enrichment layer makes one team's benchmark results consumable by every other team, regardless of their specific purpose.

## Landscape

| Repo | Role | Workflow | Benchmark maturity |
|------|------|----------|-------------------|
| **agent-aiops-on-aws** (this repo) | Spec author, reference data, adapters | **Agent-native** — RALPH loops, deployer agents, benchmark-analyst, compound-learner | High — months of production results across 15+ model/engine/instance combos |
| **recon-perf** | Benchmark orchestration tool | CLI-driven Python pipeline | New initiative — well-architected but unproven at scale |
| **awslabs/ai-on-eks** | Community publication target (EKS) | Static files, manual READMEs | Near-zero benchmark data |
| **aws/sagemaker-hyperpod-recipes** | Community publication target (HyperPod) | Hydra YAML + launcher scripts | Training-only. Zero inference benchmarks |

## Strategy: Common Artifact Format + Enrichment Layer

The bottleneck is not how benchmarks are run — it's that results can't be shared. Every major tool already produces quality metrics (TTFT, ITL, E2E, throughput). What's missing is a portable **deployment context** — what was benchmarked (model, hardware, engine config, substrate) — in a format that enables cross-team comparison regardless of which tool generated the data.

**Keep your tool. Standardize the output.**

```
Phase 1:  Define enrichment spec + workload catalog
          → common artifact schema, 7 standard workloads, enrichment wrapper

Phase 2:  Build enrichment wrapper (benchmark.yaml sidecar → enriched artifact)
          → any tool's output + model/infra/engine context = publishable artifact

Phase 3:  Agent integration
          → benchmark-runner, benchmark-analyst, spec-writer consume/produce enriched artifacts

Phase 4:  Publish benchmark output to a common repo
          → ai-on-eks, hyperpod-recipes, awslabs/ai-infra-benchmark get real reproducible artifacts
```

**Customization points**: The workload catalog defines standard inputs (7 workload cards) that provide comparable baselines. Teams customize by:
1. Overriding workload parameters for their specific use case (`catalog_id: null` with custom dataset/load),
2. Adding extensions for engine-specific metrics they care about (cache stats for cost optimization, speculative decode stats for latency tuning),
3. Attaching their own SLO targets.

The core contract (metrics + context envelope) stays stable for cross-team comparison while each team extracts the signal relevant to their purpose.

The **7 standard workload cards** are: `chatbot-short` (interactive), `chatbot-long` (long-context), `batch-throughput` (max throughput), `rag-long-context` (shared-prefix RAG), `coding-agent` (agentic with tool calls), `qps-sweep` (open-loop sweep), and `concurrency-sweep` (closed-loop sizing). See Workload Catalog for full definitions and Appendix A for a custom workload example.

## Publication Model

Enriched artifacts are published to a shared repository (public or private, depending on sensitivity) where any team can contribute and consume. Each team publishes:

- **Enriched artifacts** (benchmark output + context envelope) — the portable, comparable benchmark results
- **Reproducibility pointers** — container image (stock or custom), engine launch args, workload config
- **Dockerfile** (only when custom image) — so others can rebuild the exact serving environment in EKS or HyperPod-on-EKS

The publication target can be:
- **Public**: `awslabs/ai-infra-benchmark` that each AWS team or customer can consume
- **Private**: Internal repo or S3 bucket — for cross-team sharing within an organization where benchmark data is sensitive

The key constraint: **the enriched artifact format is the same regardless of publication target.** A team publishing to a private repo produces the same JSON as one publishing to ai-on-eks.

## Why This Repo

### Agent-Native Architecture

This repo is **agent-native** — the entire workflow is driven by AI agents, not shell scripts or manual runbooks:

- **RALPH loops** orchestrate end-to-end deploy → benchmark → learn cycles
- **infra-deployer agent** handles Terraform, GPU pre-flight, serving stack deployment
- **benchmark-analyst agent** reads raw results and produces structured reports
- **compound-learner agent** elevates cross-cutting lessons to steering rules
- **benchmark-runner skill** plans and executes benchmark sweeps

None of the other repos have this. recon-perf is a Python CLI with an 8-phase pipeline — powerful, but human-driven. ai-on-eks and hyperpod-recipes are static file collections.

The agent-native architecture means this repo can **automatically**:
1. Run a benchmark suite against a catalog workload
2. Produce the common artifact via adapter
3. Evaluate SLOs and flag regressions
4. Feed lessons back to steering rules
5. Publish to community repos

The common artifact spec is designed to slot into this loop. The adapter runs as a post-processing step in the benchmark-runner skill. The benchmark-analyst agent reads common artifacts instead of parsing tool-specific formats. The compound-learner extracts operational patterns from artifact metadata (which models on which hardware hit which SLOs).

This is the difference between "here's a JSON schema" and "here's a JSON schema embedded in an autonomous workflow that produces, validates, analyzes, and publishes artifacts without human intervention."

### Reference Data

This repo is also the **only place with real, battle-tested benchmark data** spanning:

- **Engines**: vLLM, SGLang, Ray Serve, llm-d, Dynamo
- **Hardware**: g7e (Blackwell PCIe), p5e (H200 NVSwitch), p6-b200 (B200 NVSwitch)
- **Models**: Qwen3.5 MoE, GLM-5 744B, Kimi K2.5, Devstral, Nemotron, Llama
- **Patterns**: Single-request, concurrent QPS sweeps, multi-turn, fault tolerance, agent workloads

The spec was extracted from these real results, not designed in a vacuum. The adapters are tested against actual tool output files already in `domains/gpu-serving/blueprints/*/results/`.

### Role of each repo

| Concern | Owner |
|---------|-------|
| Spec definition + versioning | **agent-aiops-on-aws** (`proposals/001-common-benchmark-commons/`) |
| Adapters (vLLM, SGLang, GenAI-Perf, etc.) | **agent-aiops-on-aws** (`scripts/adapters/`) |
| Reference dataset (retroconverted results) | **agent-aiops-on-aws** (`domains/*/blueprints/*/results/`) |
| Workload catalog governance | **agent-aiops-on-aws** (this proposal) |
| Agent-driven artifact production + analysis | **agent-aiops-on-aws** (benchmark-runner skill, benchmark-analyst agent) |
| Autonomous publish loop (produce → validate → PR) | **agent-aiops-on-aws** (RALPH loop + publish script) |
| Benchmark orchestration + RECON layers | **recon-perf** (premium CLI producer) |
| Community publication (EKS) | **awslabs/ai-on-eks** (consumer) |
| Community publication (HyperPod) | **aws/sagemaker-hyperpod-recipes** (consumer) |

---

## Artifact Design

### Three-Layer Structure

```
┌─────────────────────────────────────────────────┐
│  Envelope                                       │
│  schema_version, artifact_id, created_at,       │
│  source_tool                                    │
├─────────────────────────────────────────────────┤
│  Context (reproducibility)                      │
│  ┌──────────┐ ┌──────────────┐ ┌─────────┐     │
│  │  model   │ │infrastructure│ │ engine  │     │
│  └──────────┘ └──────────────┘ └─────────┘     │
│  ┌──────────────────────────────────────┐       │
│  │  workload (dataset + load + api)    │       │
│  └──────────────────────────────────────┘       │
├─────────────────────────────────────────────────┤
│  Core Metrics (the contract)                    │
│  ttft_ms, tpot_ms, itl_ms, e2e_ms              │
│  output_toks_per_s, request_throughput          │
│  completed, failed, error_rate                  │
├─────────────────────────────────────────────────┤
│  SLO Evaluation (optional)                      │
│  targets + actuals + pass/fail                  │
├─────────────────────────────────────────────────┤
│  Extensions (optional)                          │
│  recon_layers, gpu_telemetry, cache_stats,      │
│  speculative_decode, per_request, raw_output    │
└─────────────────────────────────────────────────┘
```

### Envelope

```json
{
  "schema_version": "1.0.0",
  "artifact_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2026-04-22T14:30:00Z",
  "source_tool": {
    "name": "vllm-bench-serve",
    "version": "0.17.2",
    "adapter_version": "1.0.0"
  }
}
```

`source_tool.name` enum: `vllm-bench-serve`, `sglang-bench-serving`, `genai-perf`, `genai-bench`, `guidellm`, `llmperf`, `recon-perf`, `inference-perf`, `custom`.

### Model

```json
{
  "model": {
    "name": "Qwen3.5-MoE-A10B-Instruct",
    "id": "Qwen/Qwen3.5-MoE-A10B-Instruct",
    "architecture": "moe",
    "parameters_total": "122B",
    "parameters_active": "10B",
    "quantization": "fp8",
    "max_model_len": 131072
  }
}
```

| Field | Required | Values |
|-------|----------|--------|
| `name` | Yes | Human-readable model name |
| `id` | Yes | HuggingFace ID or path |
| `architecture` | No | `dense`, `moe` |
| `quantization` | No | `fp16`, `bf16`, `fp8`, `int8`, `int4`, `awq`, `gptq`, `none` |
| `max_model_len` | No | Context length in tokens |

### Infrastructure — EKS + HyperPod

Designed so the **same workload and engine config can be compared across EKS and HyperPod** with only the substrate fields differing.

```json
{
  "infrastructure": {
    "substrate": "eks",
    "instance_type": "p5en.48xlarge",
    "region": "us-east-1",
    "gpu": {
      "name": "H200",
      "arch": "sm_90",
      "count": 8,
      "vram_gb": 141,
      "interconnect": "nvswitch"
    },
    "eks": {
      "cluster_version": "1.32",
      "node_count": 1,
      "ami": "al2023-nvidia"
    },
    "hyperpod": null
  }
}
```

HyperPod variant:

```json
{
  "infrastructure": {
    "substrate": "hyperpod-eks",
    "instance_type": "p5en.48xlarge",
    "region": "us-east-1",
    "gpu": {
      "name": "H200",
      "arch": "sm_90",
      "count": 8,
      "vram_gb": 141,
      "interconnect": "nvswitch"
    },
    "eks": {
      "cluster_version": "1.32",
      "node_count": 1
    },
    "hyperpod": {
      "cluster_name": "hp-inference-01",
      "instance_group": "gpu-workers",
      "deep_health_checks": true,
      "auto_recovery": true
    }
  }
}
```

| Field | Required | Values |
|-------|----------|--------|
| `substrate` | Yes | `eks`, `hyperpod`, `hyperpod-eks`, `sagemaker-endpoint`, `bare-metal` |
| `instance_type` | Yes | AWS instance type |
| `gpu.interconnect` | Yes | `nvswitch`, `pcie`, `efa`, `none` |
| `eks` | When substrate includes EKS | Cluster metadata |
| `hyperpod` | When substrate includes HyperPod | HyperPod metadata |

**Why this matters**: Same model on `p5en.48xlarge` EKS vs HyperPod-EKS with `deep_health_checks: true` can show different tail latency due to HyperPod's automatic bad-node replacement. The substrate section makes this visible.

### Engine

```json
{
  "engine": {
    "name": "vllm",
    "version": "0.17.2",
    "tensor_parallel": 4,
    "pipeline_parallel": 1,
    "prefix_caching": true,
    "gpu_memory_utilization": 0.92,
    "max_num_seqs": 256,
    "speculative_decoding": null,
    "extra_args": {
      "enable-chunked-prefill": true,
      "tool-call-parser": "qwen3_coder"
    }
  }
}
```

| Field | Required | Values |
|-------|----------|--------|
| `name` | Yes | `vllm`, `sglang`, `trt-llm`, `dynamo`, `llmd`, `ray-serve`, `custom` |
| `version` | Yes | Semver string |
| `tensor_parallel` | Yes | Integer |
| `extra_args` | No | Pass-through for engine-specific flags |

### Workload — Reproducibility Contract

Everything needed to reproduce the benchmark with any tool.

```json
{
  "workload": {
    "use_case": "chatbot",
    "catalog_id": "chatbot-short",
    "dataset": {
      "type": "synthetic",
      "input_tokens": {"mean": 256, "std_dev": 64},
      "output_tokens": {"mean": 128, "std_dev": 32}
    },
    "load": {
      "type": "constant",
      "request_rate": 2.0,
      "duration_s": 120,
      "num_prompts": 100,
      "warmup_requests": 30,
      "max_concurrency": null
    },
    "api": {
      "type": "chat",
      "streaming": true,
      "endpoint": "/v1/chat/completions"
    }
  }
}
```

`catalog_id` references a standard workload from the workload catalog. When two artifacts share the same `catalog_id`, results are directly comparable. Custom workloads use `catalog_id: null`.

| Field | Required | Values |
|-------|----------|--------|
| `use_case` | Yes | `chatbot`, `batch`, `rag`, `coding-agent`, `production-mix`, `stress`, `custom` |
| `catalog_id` | No | References workload catalog |
| `dataset.type` | Yes | `synthetic`, `sharegpt`, `generated-shared-prefix`, `trace-replay`, `custom` |
| `load.type` | Yes | `constant`, `poisson`, `staged`, `open-loop`, `sweep` |
| `api.streaming` | Yes | Boolean |

---

## Core Metrics — The Contract

This is the heart of the spec. Every adapter MUST fill these fields.

```json
{
  "metrics": {
    "duration_s": 103.5,
    "completed": 100,
    "failed": 0,
    "error_rate": 0.0,

    "ttft_ms": {"mean": 110.96, "p50": 98.5, "p90": 165.0, "p95": 180.2, "p99": 194.97},
    "tpot_ms": {"mean": 6.61, "p50": 6.67, "p90": 7.20, "p95": 7.45, "p99": 7.63},
    "itl_ms":  {"mean": 6.59, "p50": 6.36, "p90": 6.96, "p95": 7.30, "p99": 7.65},
    "e2e_ms":  {"mean": 953.2, "p50": 920.0, "p90": 1150.0, "p95": 1280.0, "p99": 1450.0},

    "output_toks_per_s": 247.25,
    "request_throughput": 0.97,
    "total_toks_per_s": 741.75,
    "total_input_tokens": 25600,
    "total_output_tokens": 12800,
    "max_concurrent_requests": 5
  }
}
```

### Latency metric shape

Every latency metric uses the same structure:

```json
{"mean": float, "p50": float, "p90": float, "p95": float|null, "p99": float}
```

| Stat | Required | Rationale |
|------|----------|-----------|
| `mean` | Yes | Every tool outputs mean |
| `p50` | Yes | Median — core operating point |
| `p90` | Yes | Tail latency threshold |
| `p95` | **Nullable** | vLLM/SGLang default to p50/p90/p99; p95 is best-effort |
| `p99` | Yes | SLO target metric |

`min`/`max` are in extensions (outlier-sensitive, not all tools emit them).

### Metric definitions

| Metric | Definition | Unit |
|--------|-----------|------|
| `ttft_ms` | Time from request send to first token received | ms |
| `tpot_ms` | `(e2e - ttft) / (output_tokens - 1)` — excludes first token | ms |
| `itl_ms` | Mean time between consecutive token arrivals | ms |
| `e2e_ms` | Total request latency (send to last token) | ms |
| `output_toks_per_s` | `total_output_tokens / duration_s` | tok/s |
| `request_throughput` | `completed / duration_s` | req/s |
| `total_toks_per_s` | `(total_input + total_output) / duration_s` | tok/s |

---

## SLO Evaluation (Optional)

```json
{
  "slo": {
    "targets": {
      "ttft_p99_ms": 300,
      "tpot_p99_ms": 50,
      "e2e_p99_ms": 15000,
      "error_rate_max": 0.001
    },
    "results": {
      "ttft_p99_ms": {"target": 300, "actual": 194.97, "pass": true},
      "tpot_p99_ms": {"target": 50, "actual": 7.63, "pass": true},
      "e2e_p99_ms": {"target": 15000, "actual": 4022.83, "pass": true},
      "error_rate_max": {"target": 0.001, "actual": 0.0, "pass": true}
    },
    "overall_pass": true
  }
}
```

---

## Workload Catalog

Standardized workload definitions that any tool can execute. When two artifacts share the same `catalog_id`, results are directly comparable.

| Catalog ID | Use case | Input tokens | Output tokens | Load | Key SLO |
|------------|----------|-------------|---------------|------|---------|
| `chatbot-short` | chatbot | 256 +/- 64 | 128 +/- 32 | 2 QPS constant | TTFT p99 < 300ms |
| `chatbot-long` | chatbot | 32K fixed | 512 fixed | 0.5 QPS constant | TTFT p99 < 2s |
| `batch-throughput` | batch | 2048 fixed | 512 fixed | max rate | error < 0.1% |
| `rag-long-context` | rag | 16K shared prefix | 256 fixed | 1 QPS constant | TTFT p99 < 2s |
| `coding-agent` | coding-agent | 4096 +/- 1024 | 2048 +/- 512 | 4 QPS constant | TTFT p99 < 500ms |
| `stress-saturation` | stress | 1024 fixed | 512 fixed | staged 1/4/16/64 | error < 5% |
| `qps-sweep` | production-mix | 2048 fixed | 512 fixed | sweep 0.5-16 | TTFT p99 < 300ms |

Full YAML definitions in [workloads/](./workloads/).

---

## Extensions (Optional)

Well-known extension keys. Tools populate what they can; consumers ignore what they don't need.

| Extension | Source | Description |
|-----------|--------|-------------|
| `recon_layers` | recon-perf | RECON framework R/E/C/O/N layer metrics |
| `gpu_telemetry` | GenAI-Perf, Prometheus | GPU util, memory, power, thermals, ECC |
| `speculative_decode` | vLLM, SGLang | Acceptance rate, tokens per step, per-position rates |
| `cache_stats` | vLLM, SGLang | Prefix cache hit rate, KV usage, LMCache tiers |
| `latency_detail` | All tools | min/max/std/p25/p75 for each latency metric |
| `per_request` | vLLM, SGLang | Pointer to detailed JSONL (keeps artifact < 10KB) |
| `raw_tool_output` | All tools | URI + format of original tool output |

---

## Adapter Pattern

Each tool writes a thin adapter that projects native output into this schema.

```
┌──────────────────┐     ┌────────────────────┐
│  Raw tool output │     │  Sidecar config     │
│  (JSON from CLI) │     │  (benchmark.yaml)   │
└────────┬─────────┘     └──────────┬──────────┘
         │                          │
         └──────────┬───────────────┘
                    │
              ┌─────▼─────┐
              │  Adapter   │
              │  (Python)  │
              └─────┬──────┘
                    │
         ┌──────────▼──────────┐
         │  Common artifact    │
         │  (validated JSON)   │
         └─────────────────────┘
```

**Sidecar config** (`benchmark.yaml`): Authored once per session. Contains model, infra, engine, SLO — things the benchmark tool doesn't output.

### Tool mapping cheat sheet

| Source field | Common artifact field |
|-------------|----------------------|
| vLLM `p50_ttft_ms` | `metrics.ttft_ms.p50` |
| vLLM `output_throughput` | `metrics.output_toks_per_s` |
| SGLang `mean_ttft_ms` | `metrics.ttft_ms.mean` |
| SGLang `output_throughput` | `metrics.output_toks_per_s` |
| GenAI-Perf `time_to_first_token.p99` | `metrics.ttft_ms.p99` |
| genai-bench `stats.ttft.p99` | `metrics.ttft_ms.p99` |
| recon-perf `ttft_p99_ms` | `metrics.ttft_ms.p99` |

---

## Cross-Tool Compatibility

| Capability | vLLM bench | SGLang bench | GenAI-Perf | genai-bench | guidellm | recon-perf |
|-----------|-----------|-------------|-----------|------------|---------|-----------|
| TTFT p50/p90/p99 | Native | p99 only* | Native | Native | Native | Native |
| TPOT p50/p90/p99 | Native | p99 only* | Native | Native | Native | Native |
| ITL p50/p90/p99 | Native | p95/p99 | Native | N/A | Native | Native |
| E2E p50/p90/p99 | Native | p90/p99 | Native | Native | Native | Native |
| Throughput | Native | Native | Native | Native | Native | Native |
| SLO evaluation | `--goodput` | N/A | `--goodput` | N/A | N/A | Scenario SLO |
| GPU telemetry | N/A | N/A | `--verbose` | N/A | N/A | Prometheus |
| RECON layers | N/A | N/A | N/A | N/A | N/A | **Native** |
| Per-request | `--save-detailed` | `--output-details` | Export | JSON | N/A | Logs |

*SGLang: p50 = median, p90 computed from `--output-details` arrays.

---

## File Naming Convention

```
{model}_{substrate}_{instance}_{engine}_{workload}_{timestamp}.json
```

Examples:
```
qwen3.5-moe_eks_p5en-48xl_vllm_chatbot-short_20260422T143000Z.json
glm5-744b_hyperpod-eks_p5e-48xl_sglang_batch-throughput_20260305T091500Z.json
llama4-scout_eks_g6e-48xl_vllm_rag-long-context_20260410T120000Z.json
```

---

## Publication Paths

### awslabs/ai-on-eks

```
blueprints/inference/{blueprint-name}/
  benchmarks/
    {artifact}.json          # common artifact
    benchmark.yaml           # sidecar config (reproducibility)
```

### aws/sagemaker-hyperpod-recipes

```
inference/{model-name}/
  benchmarks/
    {artifact}.json          # common artifact
    benchmark.yaml           # sidecar config
```

### Cross-team dashboards

Common artifacts are directly ingestable by Grafana (JSON datasource), CloudWatch (flatten metrics), S3 + Athena (JSONL aggregation), static HTML (compatibility matrix, leaderboards).

---

## Publishable Artifact Inventory

This repo isn't just benchmark JSON — it has **complete, production-tested deployment artifacts** ready to publish. The community repos are nearly empty; we have the content to fill them.

### What we have vs what they need

| Artifact type | This repo | ai-on-eks has | hyperpod-recipes has |
|--------------|-----------|---------------|---------------------|
| Dockerfiles | **13** (vLLM, SGLang, Dynamo, HiCache, LMCache) | Scattered, often outdated | 0 for inference |
| K8s YAMLs | **23+** (Ray Serve, deployments, services, stunnel, redis, InferencePool, EPP) | Some, no benchmarks attached | 0 for inference |
| Terraform | **30+** (.tf + .tfvars for EKS, capacity blocks, FSx, EFA) | Lives in separate `infra/` dir | 0 for inference |
| Benchmark results | **95+** JSON files across vLLM, SGLang, Dynamo | 0 stored results | 0 |
| Scripts | **130+** (setup, benchmark, pre-flight, model staging) | Basic client scripts | Training launchers only |
| Lessons / operational knowledge | **19** lessons.md files | 0 | 0 |
| READMEs | **17+** with deployment walkthroughs | Varies | Training READMEs only |

### Direct publication mapping

**To ai-on-eks** (`blueprints/inference/`):

| This repo blueprint | Target ai-on-eks blueprint | Artifacts to publish |
|---------------------|---------------------------|---------------------|
| `ray-serve-ft/` | `ray-serve-fault-tolerance/` | 5 K8s YAMLs (Ray, stunnel, redis, NLB), Terraform, 6 result JSONs, deploy script, fault-inject script, lessons.md |
| `ray-serve-video/` | `ray-serve-video-pipeline/` | 3 K8s YAMLs (Kafka, Ray, stunnel), benchmark script, video pipeline scripts |
| `glm5-llmd/` | `vllm-llmd-glm5/` | 8 K8s manifests (deployment, InferencePool, EPP, HTTPRoute, redis), Dockerfile, Terraform, lessons.md |
| `glm5-lmcache/` | `sglang-lmcache-glm5/` | 5 K8s manifests, Dockerfile, Terraform, LMCache configs, lessons.md |
| `qwen3-32b-eks/` | `vllm-qwen3-32b-eks/` | Setup script, benchmark script, 2 result JSONs (cache vs no-cache), README |
| `nemotron-super/` | `vllm-nemotron-super/` | 2 K8s manifests (Dynamo, disagg), Terraform, **14 result JSONs** (vLLM vs SGLang, TP2x1 vs TP2x4, c1-c256), lessons.md |
| `dynamo-hyperpod/` | `nvidia-dynamo/` (enhance existing) | 4 K8s manifests (namespace, FSx PVC, etcd, worker), smoke test |

**To sagemaker-hyperpod-recipes** (`inference/`):

| This repo blueprint | Target recipe | Artifacts to publish |
|---------------------|--------------|---------------------|
| `kimi-k2.5/` | `inference/kimi-k2.5/` | 3 Dockerfiles, Terraform, **10 result JSONs** (multi-turn, RAG, multi-tenant, memory pressure), benchmark script, setup scripts, lessons.md |
| `glm5-hyperpod/` | `inference/glm5/` | Dockerfile, Terraform, model staging scripts, pre-flight script |
| `mistral-small-4-hyperpod/` | `inference/mistral-small-4/` | K8s YAMLs, **12 result JSONs** (QPS sweep + context sweep + prefix cache), lessons.md |
| `gemma4-4b-hyperpod/` | `inference/gemma4-4b/` | K8s YAMLs, 2 result JSONs, lessons.md |
| `gemma4-hyperpod/` | `inference/gemma4/` | K8s YAMLs |

### What a published blueprint looks like

Taking `nemotron-super` as an example — this is what the ai-on-eks contribution would look like:

```
blueprints/inference/vllm-nemotron-super-b200/
  README.md                          # from our README + lessons
  docker/
    # (uses stock vllm/sglang images, no custom Dockerfile)
  manifests/
    dynamo-infra.yaml                # from nemotron-super/manifests/
    disagg-sglang-4p4d.yaml          # P/D disaggregation config
  terraform/
    main.tf                          # EKS + capacity block + FSx
    variables.tf
    nemotron-super-b200.tfvars
  scripts/
    bench.sh                         # benchmark runner
    pd_router.py                     # P/D routing proxy
  benchmarks/
    benchmark.yaml                   # sidecar config (common artifact)
    # 14 common artifact JSONs converted from results/
    nemotron-super_eks_p6b200-48xl_vllm_chatbot-short_20260312T101546Z.json
    nemotron-super_eks_p6b200-48xl_sglang_batch-throughput_20260312T124224Z.json
    ...
```

Every blueprint we publish comes with **infrastructure + deployment + results + operational knowledge**. That's the full stack — not just "here's a YAML, good luck."

### The agent advantage for publication

Because this repo is agent-native, we can **automate the conversion**:

1. Agent reads a blueprint directory (Docker, K8s, Terraform, results, lessons)
2. Agent converts benchmark results to common artifacts via adapters
3. Agent restructures files to match the target repo's convention
4. Agent generates README from lessons.md + spec + results
5. Agent creates PR to target repo

This is a one-time script per target repo, then every new blueprint we complete can be published with a single command. No other repo can do this because no other repo has the agent infrastructure.

---

## Reference Data

The benchmark results that validate the spec:

| Blueprint | Engine | Instance | Results path |
|-----------|--------|----------|-------------|
| qwen3-next | vLLM | g7e.24xlarge | `domains/gpu-serving/blueprints/qwen3-next/results/session-20260224/` |
| qwen3-next-sglang | SGLang | g7e.24xlarge | `domains/gpu-serving/blueprints/qwen3-next-sglang/results/session-20260303/` |
| kimi-k2.5 | vLLM + Dynamo | p5e.48xlarge | `domains/gpu-serving/blueprints/kimi-k2.5/results/` |
| glm5-hyperpod | SGLang | p6-b200.48xlarge | `domains/gpu-serving/blueprints/glm5-hyperpod/results/` |
| nemotron-super | vLLM | g7e.24xlarge | `domains/gpu-serving/blueprints/nemotron-super/results/` |
| ray-serve-ft | Ray Serve | g7e.24xlarge | `domains/gpu-serving/blueprints/ray-serve-ft/results/` |

These span 3 engines, 3 instance types, 2 GPU architectures (Blackwell sm_120, Hopper sm_90), and both dense and MoE models. If the spec works for all of them, it works for anyone.

---

## Implementation Phases

### Phase 1: Enrichment Spec + Workload Catalog (this repo)
- Finalize enriched artifact schema (JSON Schema draft-2020-12)
- Write workload catalog YAML definitions (17 cards shipped)
- Build `enrich-benchmark.py` with adapters for vLLM, SGLang, AIPerf
- Build `scrape-engine-metrics.py` (Prometheus scraper for engine-internal metrics)
- Build `validate-artifact.py` (schema + semantic validation)
- Dockerfile (standardized container: enrichment wrapper + workloads + schema)
- `benchmark-job.yaml` (K8s Job manifest with optional metrics-scraper sidecar)
- Reference example (`examples/nemotron-super/` — sidecar input + enriched artifact output)
- Re-run 2–3 existing benchmark sessions to validate end-to-end

### Phase 2: Agent Integration (this repo)
- Update `benchmark-runner` skill to invoke enrichment wrapper after any tool run
- Update `benchmark-analyst` agent to read enriched artifacts
- Update `spec-writer` agent to accept enriched artifacts as input for spec generation
- Validate: enriched artifacts pass schema validation, benchmark-analyst produces correct reports

### Phase 3: Community Publication
- First PR to `ai-on-eks` with enriched artifacts for 2–3 blueprints
- `publish-benchmark.sh` automates PR creation from enriched artifacts
- Publish to `hyperpod-recipes` with reproducibility pointers
- First PR to `awslabs/ai-infra-benchmark` (public shared repo) when available

---

## Supported Tool Adapters

The enrichment wrapper ships adapters for each supported tool. Each adapter maps the tool's native output format to the common metric contract:

| Tool | Output format | Adapter maps |
|------|---------------|--------------|
| vLLM `bench serve` | JSON (`benchmark_result.json`) | TTFT, ITL, E2E, throughput, per-request |
| SGLang `bench_serving` | JSON (stdout) | TTFT, ITL, E2E, throughput |
| AIPerf | JSON + CSV (`profile_export`) | TTFT, ITL, E2E, throughput, GPU telemetry, HTTP traces |
| GenAI-Perf | JSON (`profile_export`) | TTFT, ITL, E2E, throughput |
| guidellm | JSON | TTFT, E2E, throughput |
| LLMPerf | JSON | TTFT, E2E, throughput |
| recon-perf | JSON | TTFT, ITL, E2E, throughput + RECON layers |
| `bench-standard.py` (this repo) | v1 envelope (native) | All fields — Prometheus-first, emits envelope directly |

Teams using tools not listed above can use the `custom` adapter with a mapping config that specifies which fields in their output correspond to which contract metrics.

---

## Open Questions

1. **Workload catalog governance**: Who approves additions? This repo via PR review?
2. **Schema evolution**: Semver on `schema_version`. Breaking changes = major bump.
3. **Storage**: Git-committed per blueprint? S3 for aggregation? Both?
4. **Multi-run aggregation**: How to combine 3 runs of the same workload? Mean of means? Separate artifacts?
5. **Historical data migration**: Re-run existing benchmarks (gold standard, expensive) or write one-time converters for legacy results (cheaper, less precise)? **Recommendation**: re-run the top 3–5 blueprints, keep the rest as historical reference.
6. **AA-AgentPerf alignment**: Artificial Analysis's agent benchmark (real coding traces, 200 turns, >100K sequences) is the only standardized agent workload. Revisit in 6 months as agent deployments grow — may warrant a catalog entry.

---

## Appendix A: Custom Workload Example — Customer Document OCR

Customer-specific benchmark engagements often don't map to catalog workloads. These use `catalog_id: null` with parameters tuned to the customer's production traffic. Here's a worked example from a document-processing sizing engagement.

**Workload config**:

```json
{
  "workload": {
    "use_case": "custom",
    "catalog_id": null,
    "modality": "vision",
    "dataset": {
      "type": "custom",
      "description": "Bank document OCR — scanned forms, ID cards, financial statements",
      "input_tokens": {"mean": 4096, "std_dev": 1024},
      "output_tokens": {"mean": 512, "std_dev": 128},
      "image_resolution": "1024x768",
      "images_per_request": 1
    },
    "load": {
      "type": "concurrency-sweep",
      "levels": [1, 2, 4, 8, 16, 32],
      "num_prompts_per_level": 100,
      "warmup_requests": 20
    },
    "api": {
      "type": "chat",
      "streaming": false,
      "endpoint": "/v1/chat/completions"
    }
  }
}
```

**What makes this different from catalog workloads**: vision modality, custom dataset description, production-traffic-tuned I/O lengths. `catalog_id: null` signals that these results are only comparable to other runs of the same custom workload (e.g., same customer's workload on B200 vs H200, or vLLM vs TRT-LLM). Cross-customer comparison requires a shared catalog workload.

**What the sizing team concludes** (from the full artifact with all concurrency levels):

- **Operating point**: Concurrency 16 is the sweet spot — TTFT p99 stays under 3s and error rate is 0%.
- **Concurrency 32 breaks SLOs**: TTFT p99 exceeds 3s target, error rate hits 2%. The customer needs 2 replicas to serve 32 concurrent document streams.
- **Vision overhead**: TTFT is ~2× higher than text-only at equivalent concurrency (image preprocessing + cross-attention). The `modality: vision` label makes this visible when comparing against text-only catalog workloads.

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-05-14 | 1.1 | Repo alignment: `latency_metric` percentiles allow null (document missing data honestly); expanded `source_tool.name`, engine name, substrate, interconnect, quantization, speculative_decode method enums to cover real artifacts; marked `ttft_ms`/`tpot_ms`/`itl_ms`/`e2e_ms`/`request_throughput` as required on `metrics` block (with null percentiles permitted); added field-level descriptions to schema. |
| 2026-05-07 | 1.1 | Made tool-agnostic: removed AIPerf mandate, added Supported Tool Adapters table, reframed strategy around common artifact format. |
| 2026-05-07 | 1.1 | Removed "Why AIPerf over other tools" decision tree; replaced with adapter reference table. |
| 2026-05-07 | 1.1 | Added Changelog section. |
| 2026-04-22 | 1.0 | Initial proposal. |
