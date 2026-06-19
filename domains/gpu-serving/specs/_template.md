# [Blueprint Name] Requirements

## Status: DRAFT | IN_PROGRESS | DEPLOYED | COMPLETED

## Overview
Brief description of what this deployment does.

## Components

### 1. Compute
- **Platform**: EKS / SageMaker Endpoints / Lambda / etc.
- **Instance Types**: (specify with fallbacks for GPU)
- **Scaling**: Min/Max/Desired

### 1a. GPU & NCCL Pre-Flight (required for multi-GPU)
Before any multi-GPU workload, run these diagnostics and record results:

1. **GPU inventory**: `nvidia-smi` — driver version, CUDA version, GPU names, memory, ECC status
2. **Topology**: `nvidia-smi topo -m` — verify NVLink vs PCIe-only, identify GPU groupings
3. **PCIe link**: `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current` — verify expected gen/width under load
4. **ECC/Xid errors**: `nvidia-smi --query-gpu=ecc.errors.*` + `dmesg | grep "NVRM: Xid"` — zero tolerance for uncorrected errors
5. **NCCL collective test**: Run `nccl_diag.py` (all_reduce, broadcast, barrier across all GPUs) — must pass before deploying distributed training or tensor-parallel serving
6. **NCCL transport**: Check `NCCL_DEBUG=INFO` output for P2P support, transport type (NVLink/SHM/NET), channel count

**Known blockers** (see `devstral-sera/lessons.md`):
- NCCL ≤ 2.25.1 has shared memory bug on Blackwell (sm_120) PCIe-only topology — upgrade to ≥ 2.26.2
- g7e instances (RTX PRO 6000 Blackwell) are PCIe-only, no NVLink — affects multi-GPU training
- vLLM inference unaffected (uses custom allreduce, not NCCL)

### 2. Model
- **Model ID**: HuggingFace model ID or path
- **Format**: safetensors / GGUF / etc.
- **Serving**: vLLM / TGI / Triton / etc.
- **Required Args**: Any model-specific arguments
- **Deployment Card**: Run `mdc get <model> --engine <engine>` before deploying. If no card exists, run `mdc sync` and create one from upstream docs.

### 3. Networking
- **VPC**: CIDR, AZs
- **Access**: Public / Private / VPN
- **Endpoints**: Required VPC endpoints

### 4. Storage
- **Model Storage**: S3 / EFS / Local
- **Caching**: PVC size if needed

### 5. Development Environment
- **IDE**: SageMaker Studio / Cloud9 / None
- **Connectivity**: To compute cluster

## Non-Requirements
List what's explicitly out of scope:
- Multi-region?
- HA/DR?
- Production monitoring?

## Security Requirements
- Encryption at rest
- Network isolation
- IAM/RBAC

## Cost Considerations
Rough estimates or cost-saving recommendations.

## Known Limitations
Known issues or constraints to be aware of before deployment.

Check `mdc prs <model>` for recently merged upstream PRs that may affect this deployment.

## Verification Criteria

Concrete, mechanically checkable conditions for each stage. The deployer agent checks each criterion and records pass/fail in the readiness audit. A criterion is either deterministic (command returns expected output) or metric-bounded (value within threshold).

### Stage 0 — Carryover Audit (spec-design gate)
- [ ] Ran the `carryover-auditor` agent against this spec (or did the equivalent self-check): scanned every `domains/**/lessons.md` whose stack (`model`/`engine`/`gpu_arch`/`hardware`/`failure_categories`) overlaps this deployment.
- [ ] Every applicable prior lesson — especially `outcome: failure`/`partial` — is reflected here as a component requirement, required arg, or verification criterion, OR explicitly noted as not applicable. Each carried lesson cites its source (`<blueprint>/lessons.md` #N).
- [ ] No P0 carryover gap remains (an applicable, non-codified prior failure-lesson absent from this spec).

### Stage 0b — Optimization Coverage (lever ledger)

Predict the bottleneck regime, then account for every optimization tier — so a high-leverage lever is never silently skipped (the failure mode that loses throughput with no error). Fill the ledger below; a tier left blank is treated as an unreviewed gap, a tier marked `deferred` with a reason is fine.

1. **Regime prediction** (from `.claude/steering/inference-first-principles.md`): this deployment should be `____`-bound on `____` (hardware) at target concurrency. One line of reasoning: `____`.
2. **Lever ledger** — for each tier in `docs/optimization-stack.md`, mark `applied` (name the config) or `deferred — <reason>`:

| Tier | Lever | applied / deferred — reason |
|------|-------|------------------------------|
| T0 | Baseline (honest reference) | |
| T1 | Quantization (weight + KV bytes) | |
| T2 | KV / prefix cache | |
| T3 | Speculative decode | |
| T4 | Parallelism (TP/EP/DP shape) | |
| T5 | Kernel / compile | |

- [ ] Regime predicted with one-line reasoning
- [ ] Every tier is either `applied` (with config) or `deferred` (with reason) — no blank rows
- [ ] Any tier whose `docs/optimization-stack.md` priority is high for the predicted regime is `applied`, OR its deferral reason explicitly addresses why the high-priority lever doesn't pay here
- [ ] **Any deferral that cites an engine blocker ("BLOCKED by PR #X", "incompatible with …", "not supported in <engine> yet") is re-verified against the live tracker** — `mdc prs <model>` plus `gh pr view <N> --repo <vllm-project/vllm|sgl-project/sglang>` / `gh issue list --repo <repo> --search "<feature> in:title" --state all`. A merged PR or newer release may have lifted the blocker. Record the blocker's `validated: YYYY-MM-DD` next to the deferral; a blocker carried from a card/lessons.md without re-check is not a valid deferral reason for a high-priority lever
- [ ] The same tier list will be filled with measured deltas in the Stage 6 Tier Stack Table (this ledger is the *plan*; Stage 6 is the *result*)

### Stage 0c — Serving-Config Resolver (fail-closed)
- [ ] `python3 standards/serving-commons/resolver/validate-serving-config.py --sidecar blueprints/<name>/benchmark.yaml --corpus-root .` exits 0 (no hard-rule FAILs)
- [ ] If model is FP8 MoE: `model.moe_intermediate_size` is present in the sidecar (or mdc card) so `fp8-moe-tp-divisibility` verifies rather than WARNs
- [ ] Every `prior-failure:*` finding from the corpus is reviewed and noted in the deployment log

### Stage 4a — GPU Health
- [ ] All GPUs report ECC enabled, 0 uncorrectable errors: `nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader` returns all zeros
- [ ] No pending row remaps: `nvidia-smi --query-gpu=retired_pages.pending --format=csv,noheader` returns `No`
- [ ] GPU thermals < 85°C under idle: `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader`
- [ ] NCCL all-reduce bandwidth > _____ GB/s for TP=_____ (fill in from deployment card)
- [ ] No Xid errors in dmesg: `dmesg | grep "NVRM: Xid"` returns empty

### Stage 5 — Serving Stack
- [ ] Health endpoint responds: `curl -s -o /dev/null -w '%{http_code}' localhost:8000/health` returns `200`
- [ ] Test completion succeeds: single request to `/v1/completions` returns valid output
- [ ] Model loads without OOM: container logs show no `CUDA out of memory` errors
- [ ] Startup time < _____ minutes (fill in; account for DeepGEMM JIT if applicable)

### Stage 6 — Benchmark

**Workload selection**: Pick from the [standard workload catalog](../../../standards/benchmark-commons/workloads/) or define custom workloads. Each benchmark run produces an **enriched artifact** (AIPerf output + deployment context envelope).

| Workload | When to use | Catalog file |
|----------|-------------|-------------|
| `concurrency-sweep` | Always — find SLO-max operating point | `workloads/concurrency-sweep.yaml` |
| `chatbot-short` | Interactive serving validation | `workloads/chatbot-short.yaml` |
| `coding-agent` | Agentic / tool-calling workloads | `workloads/coding-agent.yaml` |
| `batch-throughput` | Max throughput ceiling | `workloads/batch-throughput.yaml` |
| `rag-long-context` | Prefix caching / long-doc workloads | `workloads/rag-long-context.yaml` |

**Required measurements** (minimum for any deployment):
- [ ] Concurrency sweep completed (1 → saturation, power-of-2 steps)
- [ ] TTFT P50 < _____ ms, P99 < _____ ms at target concurrency
- [ ] Throughput > _____ tok/s at target concurrency
- [ ] No OOM at max concurrent requests = _____
- [ ] No request timeouts during benchmark run
- [ ] Error rate < 0.1% at all concurrency levels

**KV cache validation** (required for multi-turn or agentic workloads):
- [ ] Prefix cache hit rate measured (disable → enable, compare TTFT)
- [ ] KV cache utilization % at max target concurrency < 95%
- [ ] If using HiCache/CPU offload: validated net-positive vs baseline

**Engine-internal metrics** (scraped from Prometheus `/metrics` during run):
- [ ] KV cache utilization (`vllm:gpu_cache_usage_perc` or equivalent)
- [ ] Running requests (`vllm:num_requests_running`)
- [ ] Speculative decode acceptance rate (if MTP/spec decode enabled)

**Tier Stack Table** (required — closes the Stage 0b ledger): fill the table from `docs/optimization-stack.md` with the **measured** Δ each tier delivered vs T0, and mark any tier blocked. This is the *result* half of the Stage 0b *plan*; `compound-learner` reads it to refresh the catalog's delta cells.
- [ ] One row per tier T0–T5: config landed, Δ tok/s vs T0, Δ TTFT p99 vs T0, blocked? (with reason)
- [ ] Every tier marked `deferred` in Stage 0b is reconciled here (still deferred, or applied + measured)
- [ ] Any tier that underperformed its `optimization-stack.md` typical-Δ range is noted in `lessons.md`

**Enriched artifact output**: Store in `blueprints/<name>/results/` following the schema from `standards/benchmark-commons/PROPOSAL.md`. Each artifact includes: model metadata, engine config, infrastructure, workload, core metrics, SLO evaluation, and optional extensions.

> **SLO targets**: Use thresholds from the workload catalog YAML, or override per-spec based on `mdc get <model>` deployment card recommendations.

### Stage 7 — Readiness Audit
- [ ] All readiness audit categories pass
- [ ] No unresolved lessons with severity >= HIGH in `lessons.md`
- [ ] All verification criteria above are checked and recorded
- [ ] Deployment card recommendations are either followed or explicitly overridden with justification

> **How to fill in thresholds**: Run `mdc get <model> --engine <engine>` and use the card's recommended performance targets. If no card exists, leave blank and establish baselines from the first deployment.

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes,
> design evaluations) belong in the blueprint directory, not in this spec.
> See `blueprints/<name>/lessons.md`, `blueprints/<name>/results/`, etc.
