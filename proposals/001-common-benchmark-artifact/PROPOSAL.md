# Proposal 001: Common Benchmark Artifact for LLM Inference

**Status**: PROPOSAL
**Date**: 2026-04-22
**Scope**: EKS, HyperPod

---

## Problem

Multiple teams benchmark LLM inference using different tools — vLLM `bench serve`, SGLang `bench_serving`, NVIDIA GenAI-Perf, genai-bench, guidellm, LLMPerf, recon-perf — each producing incompatible output formats. Results sit in local JSON files that nobody else can read. There is no way to:

- Compare results across teams or tools
- Publish reproducible artifacts to community repos
- Build cross-team dashboards or regression detection
- Evaluate SLO compliance using a common contract

## Landscape

| Repo | Role | Workflow | Benchmark maturity |
|------|------|----------|-------------------|
| **agent-aiops-on-aws** (this repo) | Spec author, reference data, adapters | **Agent-native** — RALPH loops, deployer agents, benchmark-analyst, compound-learner | High — months of production results across 15+ model/engine/instance combos |
| **recon-perf** | Benchmark orchestration tool | CLI-driven Python pipeline | New initiative — well-architected but unproven at scale |
| **awslabs/ai-on-eks** | Community publication target (EKS) | Static files, manual READMEs | Near-zero benchmark data |
| **aws/sagemaker-hyperpod-recipes** | Community publication target (HyperPod) | Hydra YAML + launcher scripts | Training-only. Zero inference benchmarks |

## Strategy: Spec-First, Not Tool-First

Asking teams to adopt a new tool (recon-perf or anything else) is slow. Teams already have tools that work. The bottleneck is not how benchmarks are run — it's that results can't be shared.

**Lead with the output contract, not the tool.**

```
Phase 1:  Ship the artifact spec + adapters for existing tools
          → teams publish today with zero tool changes

Phase 2:  recon-perf adds native --export-artifact support
          → becomes the richest producer (core + RECON layers + GPU telemetry)

Phase 3:  recon-perf becomes the default for new teams
          → not because mandated, but because it produces the most
            complete artifact with the least effort
```

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
| Spec definition + versioning | **agent-aiops-on-aws** (`proposals/001-common-benchmark-artifact/`) |
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

### Phase 1: Spec + Validation (this repo)
- Finalize artifact schema (JSON Schema draft-2020-12)
- Write workload catalog YAML definitions
- Retroconvert 2-3 existing benchmark sessions to validate end-to-end

### Phase 2: Adapters (this repo)
- vLLM adapter (most existing data)
- SGLang adapter (second most data)
- Validate: adapter output passes schema validation

### Phase 3: Integration
- recon-perf adds `--export-artifact` (native, richest output)
- `publish-benchmark.sh` creates PRs to ai-on-eks and hyperpod-recipes
- benchmark-analyst agent updated to read common artifacts

### Phase 4: Community
- First PR to ai-on-eks with benchmark artifacts for 2-3 blueprints
- First PR to hyperpod-recipes with inference benchmark section
- Remaining adapters (GenAI-Perf, genai-bench, guidellm)

---

## Open Questions

1. **Workload catalog governance**: Who approves additions? This repo via PR review?
2. **Schema evolution**: Semver on `schema_version`. Breaking changes = major bump.
3. **Storage**: Git-committed per blueprint? S3 for aggregation? Both?
4. **Multi-run aggregation**: How to combine 3 runs of the same workload? Mean of means? Separate artifacts?
5. **recon-perf alignment**: Native export vs adapter? Recommendation: native export (recon-perf is closest to the spec already).
