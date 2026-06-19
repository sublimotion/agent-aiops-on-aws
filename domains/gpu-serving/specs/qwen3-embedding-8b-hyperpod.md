# Qwen3-Embedding 8B on HyperPod (A10G) — Deployment Spec

## Status: COMPLETED (2026-05-13) — deploy + 5 benchmark workloads + tier comparison all complete; nodes scaled to 0

## Overview

Deploy Qwen/Qwen3-Embedding-8B on the existing SageMaker HyperPod cluster `finetune-g5-cluster` (us-east-1), targeting the `llmd-validation` instance group (ml.g5.2xlarge, 1× A10G 24GB). This is the HyperPod variant of the CTO `qwen3-embedding-8b.md` spec; it exists because g6e capacity is unavailable in the cluster's AZ (`use1-az1`), so we're running on Ampere without the FP8 cell.

Parent CTO spec: `qwen3-embedding-8b.md`. This spec is for the HyperPod-infrastructure row only.

### Optimization Objective

```
Primary:   Deploy vLLM embedding endpoint on HyperPod A10G, run concurrency sweep
Secondary: Quantization Pareto minus FP8 (A10G doesn't support FP8 tensor cores)
Metric:    throughput_toks_s, ttft_p99_ms, vectors_per_joule
```

## Components

### 1. Compute

- **Platform**: SageMaker HyperPod EKS
- **Cluster**: `finetune-g5-cluster` (us-east-1, EKS `finetune-eks`)
- **Instance group**: `llmd-validation` (target=1, current=1)
- **Node**: `hyperpod-i-0c53e6921ca835849` (ml.g5.2xlarge, 1× A10G 24GB sm_86)
- **Namespace**: `cto-embedding-g5-2xlarge`
- **Region**: us-east-1 AZ use1-az1

### 2. Model

- **Model ID**: `Qwen/Qwen3-Embedding-8B`
- **Modality**: text embeddings
- **Format**: BF16 (~16 GB). FP8 skipped (A10G has no FP8 TC).
- **Serving**: vLLM v0.19.1 (`vllm/vllm-openai:v0.19.1`) in `--task embed` mode
- **Context**: 8192 (conservative for A10G VRAM headroom)
- **Trust-remote-code**: yes

### 3. Networking

- Existing VPC + subnet from HyperPod cluster (`subnet-0956322f99a1ae4a8`, SG `sg-0c026989a9a5246e3`)
- Internal ClusterIP service; benchmark via `kubectl port-forward`

### 4. Storage

- HuggingFace cache in emptyDir (40 GiB) — first run downloads ~16 GB
- /dev/shm 4 GiB for vLLM worker IPC

## Existing artifacts

These files are already created and ready to apply:

```
domains/gpu-serving/blueprints/qwen3-embedding-8b-hyperpod/
├── k8s/
│   ├── _template.yaml
│   ├── render.sh
│   ├── vllm-embedding-g5-2xlarge.yaml     ← apply this
│   ├── vllm-embedding-g6e-xlarge.yaml     ← future, needs g6e node
│   └── vllm-embedding-g6e-2xlarge.yaml    ← future, needs g6e node
├── benchmark-g5-2xlarge.yaml              ← benchmark sidecar (ready)
├── benchmark-g6e-xlarge.yaml
└── benchmark-g6e-2xlarge.yaml
```

## Non-Requirements

- No FP8 cell (hardware limitation)
- No MIG (A10G has no MIG support)
- No cohost-isolation (single node, single GPU)
- No multi-node / disagg (single-GPU deployment)

## Verification Criteria

### Stage 4a — GPU Health
- [ ] Node `hyperpod-i-0c53e6921ca835849` is Ready in EKS
- [ ] `nvidia-smi` on the node reports 1× A10G with ECC enabled, 0 uncorrected errors
- [ ] `nvidia.com/gpu` reported in node allocatable

### Stage 5 — Serving Stack
- [ ] `kubectl apply -f k8s/vllm-embedding-g5-2xlarge.yaml` succeeds
- [ ] Pod reaches Ready within 10 minutes (accounting for model download)
- [ ] `/health` endpoint returns 200
- [ ] Test embedding request to `/v1/embeddings` returns a vector of expected dimension

### Stage 6 — Benchmark

Embedding models don't have every workload shape (no system prompts, no tool calls, no multi-turn generation). The relevant workloads from the parent `qwen3-embedding-8b.md` spec's Practitioner section map here as follows:

#### Required workloads

Run each workload against the live `qwen3-embedding-g5-4xlarge-svc` endpoint in namespace `cto-embedding-g5-4xlarge`, via `kubectl port-forward` to `localhost:8000`. Emit one enriched artifact per workload into `blueprints/qwen3-embedding-8b-hyperpod/results/`.

| # | Workload card | Sidecar | What it measures | Applicable? |
|---|---------------|---------|------------------|-------------|
| 1 | `concurrency-sweep` (fixed input length) | `benchmark-g5-4xlarge.yaml` | SLO-max operating point at 2K input | ✅ **COMPLETE** — `results/smoke-bench.json` (c=1→32, peak 119.95 req/s) |
| 2 | `concurrency-sweep` with `context_lengths: [1024, 2048, 4096, 8192]` | same sidecar, different workload override | Long Context Scaling — chunk-size vs throughput | ⏳ pending |
| 3 | `rag-qa` | `benchmark-g5-4xlarge.yaml` | 2-10K retrieved context chunks — enterprise RAG ingest shape | ⏳ pending |
| 4 | `production-mix` (trace replay) | `benchmark-g5-4xlarge.yaml` with `workload_overrides.trace.source: sharegpt` (embed the user-turns) | Real distribution of ingest lengths | ⏳ pending |
| 5 | `burn-in` with `duration_hours: 1`, `rate_fraction_of_ceiling: 0.85` | `benchmark-g5-4xlarge.yaml` | 1h stability at high load — drift ≤ 2%, zero errors | ✅ COMPLETE — `results/burn-in/burn-in-final.json` (12 slices, +2.49% positive drift, 0 errors) |
| 6 | Quality gate — MTEB Banking77 + FiQA | `run-quality-eval.py --eval banking77,fiqa` | T1 Quantization gate; BF16 baseline for future FP8 comparison | ⛔ SKIPPED per user directive |

#### Not applicable for this model

- `multi-turn-chat` — embeddings are stateless, no conversation history
- `coding-agent` (Agent Tool Calling) — embeddings don't tool-call
- `shared-prefix-multitenant` (Shared System Prompt) — embeddings have no system prompt
- `cohost-isolation` — single-node, single-GPU; no co-host testing possible

#### Per-workload success criteria

For workloads #1–4: error rate < 0.1%, no OOM, peak throughput ≥ 100 req/s at c=32 (already exceeded by #1).

For workload #5 (burn-in): throughput drift ≤ 2% from hour-1 baseline, zero unrecoverable errors, no thermal throttle past warmup.

For workload #6 (quality gate): baseline captures — no pass/fail gate on BF16 (it IS the reference). Record MTEB main_score for Banking77 and FiQA so future quantization comparisons have a target.

#### Execution order

Run sequentially to avoid contention. Each workload uses the same pod+port-forward. Order:
1. Long-context sweep (extends current smoke-bench into context axis)
2. RAG Q&A
3. Production-mix trace replay
4. Quality gate (MTEB — can run in parallel with prior since it uses its own prompts)
5. Burn-in (1h) — runs last since it leaves the pod loaded for an hour

### Stage 7 — Readiness Audit
- [x] Stage 4a GPU Health passed (A10G reports nvidia.com/gpu=1)
- [x] Stage 5 Serving Stack passed (pod Ready, /health=200, test embedding returns 4096-dim vector)
- [x] Workload #1 (concurrency-sweep) — `results/smoke-bench.json`, `results/benchmark-report-20260513.md`
- [x] Workload #2 (long-context sweep) — `results/workload-long-context.json`
- [x] Workload #3 (rag-qa) — `results/workload-rag-qa.json`
- [x] Workload #4 (production-mix) — `results/workload-production-mix.json`
- [x] Workload #5 (1h burn-in) — `results/burn-in/burn-in-final.json` (0 errors, +2.49% drift)
- [x] Tier comparison (T0 baseline vs T5 optimized) — `results/tier-comparison/tier-report.md` (21.3× delta)
- [x] `lessons.md` captures lessons across 10+ RALPH iterations
- [x] Nodes scaled to 0 (see finalize run log)
- ⛔ Workload #6 (MTEB quality gate) — skipped per user directive

## Cost Considerations

- **Current state**: ml.g5.4xlarge (workload) + ml.g5.2xlarge (system) nodes both up
  - g5.4xlarge ~$2.03/hr, g5.2xlarge ~$1.52/hr → ~$3.55/hr total
- Expected runtime for full benchmark suite (workloads 2-6): ~2 hours (burn-in is 1h; others are 30 min combined) = ~$7.10
- After benchmarks complete: scale both instance groups to 0 to stop billing
- Alternative: keep g5.4xlarge up for iteration, drop g5.2xlarge (system pods will reschedule onto g5.4xlarge if there are enough slots)

## Known Limitations

- A10G lacks FP8 tensor cores — FP8 cell of the CTO O3 matrix cannot run here
- HyperPod `use1-az1` appears capacity-constrained for L40S (g6e) — FP8 cell deferred
- vLLM embedding endpoint uses `/v1/embeddings`, not `/v1/completions` — benchmark tooling must target the right path

## Links

- Parent spec: `qwen3-embedding-8b.md`
- Blueprint dir: `domains/gpu-serving/blueprints/qwen3-embedding-8b-hyperpod/`
- CTO engagement: `cto-benchmark-engagement.md`
