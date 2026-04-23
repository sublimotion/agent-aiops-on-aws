# Gemma 4 4B on SageMaker HyperPod — Small Multimodal Model Serving Spec

## Status: DRAFT (2026-04-06)

## Overview

Deploy **Gemma 4 4B** (google/gemma-4-E4B-it) on the existing SageMaker HyperPod cluster (`llmd-inference-cluster`) using a single ml.g5.4xlarge instance (1× A10G, 24GB VRAM). This is a lean evaluation of Google's smallest Gemma 4 variant — a 4B dense multimodal (vision+language) model that fits comfortably on modest hardware with significant VRAM headroom for KV cache.

**Why Gemma 4 4B:**
- First 4B model with multimodal (text+image) support in Gemma family
- Hybrid attention (sliding window 1024 + global every 6th layer)
- head_dim=512 (requires SM 8.0+, works on A10G Ampere)
- 128K native context (capped to 32K for practical serving on 24GB VRAM)
- Tool calling support via `pythonic` parser

**Primary goals:**
1. Validate Gemma 4 4B loads and serves on A10G (ml.g5.4xlarge)
2. Measure throughput and latency for a small multimodal model
3. Test tool calling accuracy (BFCL subset)
4. Smoke test vision inputs (image+text prompts)
5. Establish baseline quality for code generation

**Sister spec**: `gemma4-hyperpod.md` (31B variant on ml.p5.48xlarge with 8× H100)

---

## Components

### 1. Compute — HyperPod EKS

- **Platform**: SageMaker HyperPod with EKS 1.34 orchestrator
- **Cluster**: `llmd-inference-cluster` (pre-provisioned)
  - EKS cluster: `llmd-inference-eks`
  - Region: **us-east-1**
  - VPC: vpc-0c5f8a4e2b9d37a1f
  - Private subnet: subnet-0a1b2c3d4e5f67890 (us-east-1a)
- **Node Group**: `gpu-workers` (already exists)
  - Instance type: **ml.g5.4xlarge** (1× A10G, 24GB VRAM, 16 vCPU, 64GB RAM)
  - Current count: 1 node running
  - Min/Max: 1/3 (do not scale beyond 1 for this eval)
- **Availability Zone**: us-east-1a
- **Auto-recovery**: Enabled via HyperPod deep health checks
- **Scaling**: Fixed at 1 node for this evaluation

### 1a. GPU Pre-Flight (A10G Ampere)

| Check | Expected Result | Command |
|-------|-----------------|---------|
| GPU count | 1× A10G (24GB) | `nvidia-smi` |
| GPU driver | 535+ (CUDA 12.1+) | `nvidia-smi` |
| ECC errors | 0 uncorrected | `nvidia-smi --query-gpu=ecc.errors.uncorrected.*` |
| PCIe link | Gen4 x16 | `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current` |
| SM architecture | 8.6 (Ampere) | `nvidia-smi --query-gpu=compute_cap` |

**Note**: A10G is PCIe-only (no NVLink). Multi-GPU not required — single TP1 replica.

### 2. Model

- **Model ID**: `google/gemma-4-E4B-it` (ungated — no HF token required)
- **Architecture**: Dense decoder-only transformer, ~4B params (E4B = effective 4B)
- **Context**: 131K native (using 32K max for practical serving on 24GB VRAM)
- **Attention**: Hybrid — local sliding window + full attention every 6th layer
  - **head_dim=256** — works on A10G SM 8.6 (Ampere)
- **Multimodal**: Text + image (Gemma 4 vision tower)
- **VRAM Requirements**:
  - BF16: ~8 GB model weights
  - Total with KV cache @ 32K context: ~14 GB (10 GB headroom on A10G)
- **License**: Gemma Terms of Use (research + commercial allowed with attribution)

### Parallelism Strategy

| Precision | Model Size | TP | GPUs | KV Headroom | Use Case |
|-----------|-----------|----|----|-------------|----------|
| **BF16** | ~8 GB | 1 | 1 | ~16 GB | Standard serving, full accuracy |

Single GPU (TP1) is sufficient. No expert parallelism (dense model). No disaggregated inference (colocated on A10G).

### Serving Configuration

**Container Image**: `vllm/vllm-openai:latest` (or v0.6.6+ with Gemma 4 support)

```bash
vllm serve google/gemma-4-E4B-it \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --tool-call-parser pythonic \
  --port 8000
```

**vLLM Details**:
- `model_type: gemma4`
- `class: Gemma4ForConditionalGeneration`
- `--tool-call-parser pythonic` (Gemma 4 tool calling format)
- `--max-model-len 32768` (conservative limit for 24GB VRAM)
- `--gpu-memory-utilization 0.90` (leave 10% headroom for batch expansion)

### Known Issues

1. **Reasoning mode broken**: vLLM #38945 — channel tokens stripped during decoding. Cannot test `--reasoning-parser gemma4` until fixed. Use tool calling only.
2. **Tool calling streaming bugs**: vLLM #38945, #38910 — may require non-streaming mode for tool calls. Test both streaming and non-streaming.
3. **Vision input format**: Gemma 4 multimodal requires specific image encoding (base64 or URL). Follow vLLM's vision API format.
4. **LMCache incompatible**: Heterogeneous head_dim across layers breaks LMCache. Do not use LMCache with Gemma 4.

### 3. Networking

- **VPC**: vpc-0c5f8a4e2b9d37a1f (pre-provisioned)
- **Private Subnet**: subnet-0a1b2c3d4e5f67890 (us-east-1a)
- **Security Group**: sg-0d9e8f7c6b5a43210 (allow 8000 from EKS pods)
- **VPC Endpoints**: S3, ECR API, ECR DKR, STS, CloudWatch Logs, SSM, EC2
- **Access**: SSM Session Manager to nodes, direct HTTP to pod IP (no external LB for eval)

### 4. Storage

**Model Weights — S3**:
- Bucket: `s3://hyperpod-eks-bucket-495365983931-us-east-1`
- Path: `s3://.../models/gemma-4-4b-it/` (~16 GB safetensors)

**Pre-session setup**:

```bash
huggingface-cli download google/gemma-4-E4B-it \
  --local-dir ./gemma-4-4b-it/ \
  --local-dir-use-symlinks False

aws s3 sync ./gemma-4-4b-it/ \
  s3://hyperpod-eks-bucket-495365983931-us-east-1/models/gemma-4-4b-it/ \
  --region us-east-1
```

**Node Storage**: EBS: 200 GB (system + logs), model staged from S3 to local disk via init container or mounted via S3 CSI driver.

**Staging Strategy**: S3 → EBS init container → mount to pod `/models/` (NVMe not required — model is small).

### 5. Development Environment

- **IDE**: Not required (command-line eval only)
- **Connectivity**: SSM Session Manager to HyperPod nodes, kubectl to EKS cluster

---

## Benchmark Design

Time budget: ~2-3 hours total (lean evaluation for a small model).

### Priority Tiers

| Priority | Phase | Time | Deliverable |
|----------|-------|------|-------------|
| **P0** | Smoke test + tool calling | 20 min | Model serves, tools work, BFCL ≥60% |
| **P1** | Standard workload sweep (W1-W6) | 1.5 hrs | Comprehensive serving profile via bench-runner pod |
| **P2** | Vision smoke test + code quality | 30 min | Multimodal validation, code generation pass rate |

### P0: Smoke Test + Tool Calling

| Step | Test | Expected |
|------|------|----------|
| 1 | Health check | `/health` returns 200 |
| 2 | Basic generation | Prompt: "Hello" -> coherent response |
| 3 | Context length | 4K input + 512 output |
| 4 | Tool call (single) | `read_file(path="/etc/hosts")` -> correct tool call JSON |
| 5 | Multi-turn | 3-turn conversation with memory |

**Tool Calling Accuracy** — BFCL subset (50 scenarios):

| Category | Scenarios | What It Tests |
|----------|-----------|---------------|
| Simple function call | 10 | Single tool, correct arguments |
| Multi-tool selection | 10 | Choose correct tool from 5+ options |
| Parallel tool calls | 10 | Call 2+ tools in one response |
| Multi-turn tool use | 10 | Tool result -> follow-up call |
| Structured output | 10 | JSON schema compliance |

**Gate**: BFCL accuracy >= 60% to proceed. Lower threshold than 31B variant (60% vs 70%) — this is a 4B model.

### P1: Standard Workload Sweep (W1-W6) — bench-runner Pod

Run the standard `benchmark-serving.py` workloads via in-cluster bench-runner pod. This provides the common benchmark baseline comparable across all models in the repo.

**Pod Setup**:

```bash
# Create ConfigMap with benchmark script
kubectl create configmap benchmark-scripts \
  --from-file=benchmark-serving.py=/scripts/benchmark-serving.py \
  --namespace default

# Deploy bench-runner pod (edit env vars first)
# Set BENCHMARK_API_URL=http://<gemma4-pod-ip>:8000
# Set BENCHMARK_MODEL=google/gemma-4-E4B-it
kubectl apply -f scripts/bench-runner-pod.yaml

# Run all workloads
kubectl exec bench-runner -- python /scripts/benchmark-serving.py \
  --api-url $BENCHMARK_API_URL \
  --model $BENCHMARK_MODEL \
  --config gemma4-4b-bf16-tp1 \
  --workloads w1,w2,w3,w4,w5,w6 \
  --output-dir /results
```

**W1: Multi-Turn Chat** — Sweep rounds (1/5/10) x concurrency (1/4/8):

| Rounds | Concurrency | TTFT p50 (ms) | ITL p50 (ms) | Throughput (tok/s) |
|--------|-------------|--------------|-------------|-------------------|
| 1 | 1 | TBD | TBD | TBD |
| 5 | 4 | TBD | TBD | TBD |
| 10 | 8 | TBD | TBD | TBD |

**W2: RAG / Long Document QA** — Shared document prefix (2K/5K/10K tokens) with cache warmup:

| Doc Tokens | Warmup:Query | Concurrency | TTFT p50 (ms) | Cache Benefit |
|-----------|-------------|-------------|--------------|--------------|
| 2000 | 2:2 | 4 | TBD | TBD |
| 5000 | 3:1 | 8 | TBD | TBD |
| 10000 | 4:1 | 8 | TBD | TBD |

**W3: Agentic Tool Calling** — Multi-turn with tool latency pauses:

| Turns | Tool Latency | TTFT p50 (ms) | E2E (s) | Throughput (tok/s) |
|-------|-------------|--------------|---------|-------------------|
| TBD | TBD | TBD | TBD | TBD |

**W4: Shared System Prompt** — Prefix caching benefit under load (2K/8K/16K prompt):

| Prompt Len | Concurrency | TTFT p50 (ms) | Cache Hit Rate |
|-----------|-------------|--------------|----------------|
| 2000 | 4 | TBD | TBD |
| 8000 | 8 | TBD | TBD |
| 16000 | 8 | TBD | TBD |

**W5: ShareGPT-style Conversations** — QPS sweep to find capacity ceiling:

| QPS | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | Throughput (tok/s) | Success Rate |
|-----|--------------|--------------|-------------|-------------------|-------------|
| 1.0 | TBD | TBD | TBD | TBD | TBD |
| 5.0 | TBD | TBD | TBD | TBD | TBD |
| 10.0 | TBD | TBD | TBD | TBD | TBD |
| 15.0 | TBD | TBD | TBD | TBD | TBD |
| 20.0 | TBD | TBD | TBD | TBD | TBD |

**W6: Long Context Scaling** — Input length sweep (1K-16K tokens):

| Input Len | Output | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | Throughput (tok/s) |
|----------|--------|--------------|--------------|-------------|-------------------|
| 1024 | 512 | TBD | TBD | TBD | TBD |
| 4096 | 512 | TBD | TBD | TBD | TBD |
| 8192 | 512 | TBD | TBD | TBD | TBD |
| 16384 | 512 | TBD | TBD | TBD | TBD |

**Output**: JSON results at `/results/benchmark_gemma4-4b-bf16-tp1_<timestamp>.json`. Copy to `blueprints/gemma4-4b-hyperpod/results/`.

### P2: Vision Smoke Test + Code Quality

**P2a: Vision Input** (5 tasks):

| Task | Input | Expected |
|------|-------|----------|
| Image description | Cat photo + "What's in this image?" | "A cat sitting on..." |
| OCR | Screenshot with text + "Read the text" | Accurate transcription |
| Chart analysis | Bar chart + "What's the highest value?" | Correct value extraction |
| Meme understanding | Image + text overlay + "Explain this" | Coherent explanation |
| Multi-turn vision | Image + follow-up questions | Context retained |

**P2b: Code Generation** (5 tasks):

| Task | Complexity | LOC |
|------|-----------|-----|
| Parse JSON with error handling | Easy | 20 |
| Binary search implementation | Medium | 30 |
| LRU cache class | Medium | 50 |
| Depth-first search on graph | Medium | 40 |
| Rate limiter with sliding window | Hard | 60 |

**Evaluation**: Manual review for correctness. Pass = code runs without errors and produces correct output.

---

## Success Criteria

| Metric | Target | Phase | Notes |
|--------|--------|-------|-------|
| BFCL accuracy | >= 60% | P0 | Lower bar for 4B model |
| W5 max QPS (TTFT p99 < 1s) | >= 15 | P1 | Small model should be fast |
| W6 TTFT p50 at 16K input | < 400ms | P1 | Low latency on A10G |
| W1 multi-turn 10 rounds TTFT p50 | < 800ms | P1 | |
| W2 RAG cache hit TTFT improvement | >= 30% | P1 | |
| W3 agentic tool calling E2E | < 8s per turn | P1 | |
| W4 prefix caching speedup | >= 2x | P1 | |
| ITL p50 | < 30ms | P1 | Fast decoding |
| Vision tasks correct | >= 3/5 | P2 | Multimodal validation |
| Code generation pass rate | >= 60% | P2 | (3/5 tasks) |

---

## Non-Requirements

- Multi-GPU / TP > 1 (single A10G sufficient)
- Production autoscaling (eval session, fixed capacity)
- Full 128K context evaluation (using 32K max)
- vLLM KV cache offloading tests
- LMCache integration (incompatible with head_dim heterogeneity)
- SageMaker endpoint creation (direct pod access)
- Distributed inference across nodes
- Fine-tuning or training workloads
- Production-grade TLS / auth
- Comprehensive multimodal benchmark suite (only smoke tests)

---

## Security Requirements

- **Encryption**: S3 SSE-S3, EBS KMS encryption
- **Network**: Private subnets only, no public IPs
- **VPC Endpoints**: All AWS service access via endpoints
- **Node Access**: SSM Session Manager only
- **IAM**: Least privilege roles (HyperPod execution role)

---

## Cost Considerations

| Resource | Estimated Cost | Notes |
|----------|---------------|-------|
| ml.g5.4xlarge (3 hrs on-demand) | ~$4.50 | $1.51/hr |
| EKS control plane | $0.10/hr (~$0.30) | Already running (shared) |
| S3 model storage | ~16 GB (~$0.37/month) | Delete after eval |
| **Total evaluation session** | ~$5 | |

Extremely low cost due to small instance size and short eval window. Cluster infrastructure is pre-provisioned and shared.

---

## Deployment Sequence

```
1. Pre-session: Download gemma-4-4b-it weights (~16 GB) -> S3
2. Verify HyperPod cluster status, node group Ready
3. Stage weights: S3 -> EBS init container
4. Deploy: vLLM pod (TP1, port 8000), wait for model load (~3 min)
5. Deploy: bench-runner pod with benchmark-serving.py ConfigMap
6. P0: Smoke test + BFCL (20 min) -> GATE
7. P1: Standard workload sweep W1-W6 via bench-runner pod (1.5 hrs)
8. P2: Vision smoke test + code generation (30 min)
9. Export results (JSON + logs) to S3 and blueprints/gemma4-4b-hyperpod/results/
10. Teardown or leave running for ad-hoc testing
```

---

## Known Limitations

- **4B parameter size**: Lower quality than 31B variant — expect less sophisticated reasoning and potentially lower tool calling accuracy
- **A10G 24GB VRAM**: Limits context to 32K (vs 128K native) for practical batch sizes
- **PCIe-only interconnect**: No NVLink — not relevant for single-GPU TP1 deployment
- **Vision capabilities**: Limited compared to specialized vision models (GPT-4V, Claude 3) — use for basic multimodal tasks only
- **Reasoning mode broken**: vLLM #38945 prevents testing extended thinking mode
- **Small batch size**: 24GB VRAM limits concurrent requests compared to larger instances

Check `mdc prs gemma-4` for recently merged upstream PRs that may affect this deployment.

---

> **Note**: Operational artifacts (lessons learned, benchmark results, deployment notes)
> belong in the blueprint directory: `blueprints/gemma4-4b-hyperpod/`.
