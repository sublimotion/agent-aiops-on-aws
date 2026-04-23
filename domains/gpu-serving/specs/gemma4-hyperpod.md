# Gemma 4 31B on SageMaker HyperPod — Serving Evaluation Spec

## Status: DRAFT (2026-04-06)

## Overview

Evaluate **Gemma 4 31B** (dense) on SageMaker HyperPod with Flexible Training Plan (FTP) capacity. Time-bounded evaluation session to validate serving feasibility, measure baseline performance, and test tool calling on ml.p5.48xlarge (8x H100 80GB).

**Why Gemma 4:** First Gemma with head_dim=512 (Hopper+ only), hybrid attention (sliding window + global), multimodal (text/image/video), 256K context. Tool calling requires dedicated parser.

**Test window:** April 6, 2026, 12:55 PM ET -> April 7, 2026, ~8:00 AM ET (~19 hours). Shared with Mistral Small 4 eval (see `mistral-small-4-hyperpod.md`).

**Primary goals:**
1. Validate Gemma 4 31B loads and serves correctly on H100
2. Measure single-request and concurrent throughput/latency
3. Test tool calling accuracy (gemma4 parser)
4. Optionally test Gemma 4 26B-A4B MoE variant
5. Identify blockers for production deployment

---

## Shared Infrastructure

This spec shares compute, networking, storage, and monitoring with `mistral-small-4-hyperpod.md`. Both models run on the same cluster during the same FTP window.

### Compute — HyperPod EKS with Flexible Training Plan

- **Platform**: SageMaker HyperPod with EKS 1.34 orchestrator
- **Region**: us-east-2
- **GPU Instance**: ml.p5.48xlarge (8x H100 80GB = 640 GB total VRAM)
- **Availability Zone**: us-east-2a (determined by FTP offering)
- **Training Plan**: `gemma4-mistral4-test-20260406`
  - Duration: ~19 hours (April 6, 12:55 PM ET -> April 7, ~8:00 AM ET)
  - Cost: $636 (split across both model evals)
  - Instance type: ml.p5.48xlarge
  - Instance count: 1

**IMPORTANT: All AWS commands MUST use `--profile agent` (account 495365983931).**

**EKS Cluster** (pre-provisioned):
- Cluster name: `gemma4-mistral4-eks`
- Version: 1.34
- VPC: vpc-0ac42dd6bad805ebf
- Private subnet: subnet-096f79eef468898e4 (10.1.0.0/16, us-east-2a)
- EKS private subnets: subnet-08c53d1be961f805c (us-east-2a), subnet-09b67bce9fe17f238 (us-east-2b)
- Public subnets: subnet-0d4f5917c3e0010e3 (us-east-2a), subnet-09905305a2d91da26 (us-east-2b)
- Security group: sg-0b38a70db0a6a994f
- AWS profile: **`agent`** (account 495365983931)
- Region: **us-east-2**

**HyperPod Cluster**:
- Name: `gemma4-mistral4-hp`
- Orchestrator: EKS (`gemma4-mistral4-eks`)
- Instance group: `p5-hopper` -> ml.p5.48xlarge x 1
- Training plan: `gemma4-mistral4-test-20260406`
- Execution role: arn:aws:iam::495365983931:role/service-role/AmazonSageMaker-ExecutionRole-20260212T145083
- Lifecycle S3: s3://hyperpod-eks-bucket-495365983931-us-east-2
- Deep health checks: enabled (GPU, NVLink, EFA, NCCL validation)
- Auto-recovery: enabled
- Node provisioning mode: Continuous

### GPU Pre-Flight Validation (H100 NVSwitch)

| Check | Expected Result | Source |
|-------|-----------------|--------|
| GPU count | 8x H100 80GB | Deep health check + `nvidia-smi` |
| GPU driver | 535+ (CUDA 12.1+) | `nvidia-smi` |
| NVLink topology | All 8 GPUs via NVSwitch | `nvidia-smi topo -m` |
| ECC errors | 0 uncorrected | `nvidia-smi --query-gpu=ecc.errors.*` |
| NCCL all-reduce | > 450 GB/s bus bandwidth | Deep health check |
| EFA adapters | 32 active | `fi_info -p efa` |

**Gemma 4 compatibility:** head_dim=512 requires SM 9.0+ (Hopper). Works on H100. Broken on Blackwell PCIe (g7e).

### Networking

- **VPC**: vpc-0ac42dd6bad805ebf (pre-provisioned)
- **Private Subnet**: subnet-096f79eef468898e4 (10.1.0.0/16, us-east-2a)
- **Security Group**: sg-0b38a70db0a6a994f (EFA-enabled, self-referencing)
- **VPC Endpoints**: S3, ECR API, ECR DKR, STS, CloudWatch Logs, SSM, EC2
- **Access**: SSM Session Manager only (no public SSH)
- **Model Invocation**: Direct HTTP to pod IP (no load balancer for eval session)

**HyperPod K8s Tolerations** (required in pod manifest):
```yaml
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  - key: sagemaker.amazonaws.com/RestrictedNode
    operator: Exists
    effect: NoSchedule
  - key: node.kubernetes.io/disk-pressure
    operator: Exists
    effect: NoSchedule
```
**Note**: HyperPod can apply stale disk-pressure taints even with 50% free space. Fix: restart kubelet via `kubectl debug node/... -- chroot /host systemctl restart kubelet`.

### Storage

**Model Weights — S3**:
- Bucket: s3://hyperpod-eks-bucket-495365983931-us-east-2
- Path: `s3://.../models/gemma-4-31B-it/` (~126 GB)

**Pre-session setup**:

```bash
huggingface-cli download google/gemma-4-31B-it \
  --local-dir ./gemma-4-31B-it/ \
  --local-dir-use-symlinks False

aws s3 sync ./gemma-4-31B-it/ \
  s3://hyperpod-eks-bucket-495365983931-us-east-2/models/gemma-4-31B-it/ \
  --profile agent --region us-east-2
```

**Node Storage**: NVMe: 8x 3.84 TB SSDs (~30 TB total), EBS: 500 GB (system + logs).

**Staging Strategy**: Download to NVMe via SSM session (`/opt/dlami/nvme/models/gemma-4-31B-it/`), then mount via `hostPath` in the pod manifest. This matches the Mistral SM4 approach and avoids init container complexity.

```bash
# Via SSM session on the HyperPod node:
aws s3 sync s3://hyperpod-eks-bucket-495365983931-us-east-2/models/gemma-4-31B-it/ \
  /opt/dlami/nvme/models/gemma-4-31B-it/ --profile agent --region us-east-2
```

### Monitoring

Shared Prometheus deployment (see `mistral-small-4-hyperpod.md` for full config). Scrape vLLM `/metrics` on port 8002.

### Node Access — SSM

```bash
aws sagemaker list-cluster-nodes \
  --cluster-name gemma4-mistral4-hp \
  --region us-east-2 --profile agent

aws ssm start-session \
  --target sagemaker-cluster:<cluster-id>_<instance-id> \
  --region us-east-2 --profile agent
```

### AWS Profile Requirement

All CLI commands MUST use `--profile agent --region us-east-2`. The FTP, EKS cluster, and HyperPod cluster are in account **495365983931** (agent profile), NOT the default account (615299764834).

---

## Model

### Gemma 4 31B (Dense)

- **Model ID**: `google/gemma-4-31B-it`
- **Architecture**: Dense decoder-only transformer, 30.7B params
- **Context**: 256K native (sliding window 1024 + global attention every 6th layer)
- **Attention**: Hybrid — local sliding window (1024) + full attention every 6th layer
  - **head_dim=512** (not 128 like most models) — requires SM 9.0+
- **Multimodal**: Text + image + video (Paligemma vision tower)
- **VRAM Requirements**:
  - BF16: ~63 GB (TP2 on 8x H100)
  - INT4 (GPTQ): ~17 GB (TP1 possible)
- **License**: Gemma Terms of Use (research + commercial allowed with attribution)

### Parallelism Strategy

| Variant | Total VRAM | TP | GPUs | Headroom | Use Case |
|---------|-----------|----|----|----------|----------|
| **Gemma 4 31B BF16** | 63 GB | 2 | 2 | 97 GB | Dense, low latency |
| **Gemma 4 26B-A4B BF16** | 52 GB | 2 | 2 | 108 GB | MoE comparison (optional) |

Both variants fit comfortably on ml.p5.48xlarge with TP2. Uses 2 of 8 GPUs, leaving 6 for Mistral SM4.

### Serving Configuration

**Container Image**: `vllm/vllm-openai:latest` (v0.19.0+) — requires `pip install git+https://github.com/huggingface/transformers.git` at startup (PyPI transformers lacks `gemma4` model_type). Also needs `apt-get install git` first.

**IMPORTANT**: The `vllm/vllm-openai:gemma4` tag fails on H100 (subprocess NVML error). Use `latest` + git transformers instead.

```bash
# Startup preamble (in pod command)
apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1
pip install -q git+https://github.com/huggingface/transformers.git 2>&1 | tail -3

vllm serve google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --tool-call-parser pythonic \
  --enable-auto-tool-choice \
  --host 0.0.0.0 \
  --port 8002 \
  --trust-remote-code
```

**Port 8002** avoids conflict with colocated Gemma 4 E4B (8000) and Mistral SM4 (8001).

**vLLM Details**:
- `model_type: gemma4`
- `class: Gemma4ForConditionalGeneration`
- `--tool-call-parser pythonic` (battle-tested on E4B with 100% BFCL; `gemma4` parser available as alternative)
- `--enable-auto-tool-choice` (required — without this, tool calling is silently disabled)
- `--reasoning-parser gemma4` (KNOWN BROKEN — channel tokens stripped in vLLM #38945)

### Known Issues

1. **Reasoning mode broken**: vLLM #38945 — channel tokens stripped during decoding. Cannot test `--reasoning-parser gemma4` until fixed. Use tool calling only.
2. **Tool calling streaming bugs**: vLLM #38945, #38910 — may require non-streaming mode for tool calls. Test both streaming and non-streaming.
3. **head_dim=512 requirement**: Requires SM 9.0+ (Hopper, H100). Would break on Blackwell PCIe (g7e, sm_120). Not an issue here (using H100).
4. **TRITON_ATTN forced on H100**: Heterogeneous head_dim (256 local + 512 global) forces vLLM to use TRITON_ATTN instead of FlashAttention. From E4B results, TTFT at 16K was **worse** on H100 (379ms) than A10G (148ms) due to this. Throughput under concurrency is dramatically higher, but single-request prefill is slower. The W6 TTFT target of <500ms at 16K may be tight for the larger 31B model.
5. **NVFP4 MoE weight loading fails**: vLLM #38912 — affects 26B-A4B variant, not 31B dense.
6. **LMCache incompatible**: Heterogeneous head_dim across layers breaks LMCache.
7. **gemma4 image broken on H100**: `vllm/vllm-openai:gemma4` fails with subprocess NVML error. Use `latest` + git transformers.
8. **PyPI transformers lacks gemma4**: Must install from git (`pip install git+https://github.com/huggingface/transformers.git`). Requires `apt-get install git` in the container first.
9. **Colocated Mistral vLLM 0.19 bug**: If sharing the cluster with Mistral SM4, the Mistral pod needs the `reasoning_effort` sed patch in `tokenizers/mistral.py` (see `mistral-small-4-hyperpod.md`).

### Secondary Model: Gemma 4 26B-A4B MoE (Optional)

- MoE variant: 25.2B total, 3.8B active per token, 128 experts (top-8)
- VRAM: ~52 GB BF16 (TP2)
- Test if time permits after 31B validation (P3c)

```bash
vllm serve google/gemma-4-26B-A4B-it \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --tool-call-parser pythonic \
  --enable-auto-tool-choice \
  --host 0.0.0.0 \
  --port 8003 \
  --trust-remote-code
```

---

## Benchmark Design

Time budget: Shared 19-hour FTP window. Gemma 4 benchmarks target ~6 hours total (increased for standard workload sweep).

### Priority Tiers

| Priority | Phase | Time | Deliverable |
|----------|-------|------|-------------|
| **P0** | Smoke test + tool calling | 30 min | Model serves, tools work |
| **P1** | Standard workload sweep (W1-W6) | 2 hrs | Comprehensive serving profile via bench-runner pod |
| **P1v** | vllm bench serve sweep | 1.5 hrs | QPS sweep, context scaling, prefix caching |
| **P2** | Code + reasoning tasks | 45 min | Quality benchmarks |
| **P3** (optional) | Gemma 4 26B-A4B MoE | 1 hr | Dense vs MoE within Gemma family |

### P0: Smoke Test + Tool Calling

| Step | Test | Expected |
|------|------|----------|
| 1 | Health check | `/health` returns 200 |
| 2 | Basic generation | Prompt: "Hello" -> coherent response |
| 3 | Context length | 8K input + 512 output |
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

**Note**: Tool calling may take significant time per scenario due to multi-turn parsing and potential streaming bugs. Budget 30 min for 50 scenarios. If P0 takes >45 min, reduce to 30 scenarios and proceed.

**Gate**: BFCL accuracy >= 70% to proceed. If < 70%, skip tool-use benchmarks (P2c).

### P1: Standard Workload Sweep (W1-W6) — bench-runner Pod

Run the standard `benchmark-serving.py` workloads via in-cluster bench-runner pod. This provides the common benchmark baseline comparable across all models in the repo.

**Pod Setup**:

```bash
# Create ConfigMap with benchmark script
kubectl create configmap benchmark-scripts \
  --from-file=benchmark-serving.py=/scripts/benchmark-serving.py

# Deploy bench-runner pod (edit env vars first)
# Set BENCHMARK_API_URL=http://<gemma4-pod-ip>:8002
# Set BENCHMARK_MODEL=google/gemma-4-31B-it
kubectl apply -f scripts/bench-runner-pod.yaml

# Run all workloads
kubectl exec bench-runner -- python /scripts/benchmark-serving.py \
  --api-url $BENCHMARK_API_URL \
  --model $BENCHMARK_MODEL \
  --config gemma4-31b-bf16-tp2 \
  --workloads w1,w2,w3,w4,w5,w6 \
  --output-dir /results
```

**W1: Multi-Turn Chat** — Sweep rounds (1/5/10) x concurrency (1/4/8) x QPS (1.0/4.0):

| Rounds | Concurrency | QPS | TTFT p50 (ms) | ITL p50 (ms) | Throughput (tok/s) |
|--------|-------------|-----|--------------|-------------|-------------------|
| 1 | 1 | 1.0 | TBD | TBD | TBD |
| 5 | 4 | 4.0 | TBD | TBD | TBD |
| 10 | 8 | 4.0 | TBD | TBD | TBD |

**W2: RAG / Long Document QA** — Shared document prefix (2K/5K/10K tokens) with cache warmup:

| Doc Tokens | Warmup:Query | Concurrency | TTFT p50 (ms) | Cache Benefit |
|-----------|-------------|-------------|--------------|--------------|
| 2000 | 2:2 | 4 | TBD | TBD |
| 5000 | 3:1 | 8 | TBD | TBD |
| 10000 | 4:1 | 8 | TBD | TBD |

**W3: Agentic Tool Calling** — Multi-turn with tool latency pauses (simulates real agent loop):

| Turns | Tool Latency | TTFT p50 (ms) | E2E (s) | Throughput (tok/s) |
|-------|-------------|--------------|---------|-------------------|
| TBD | TBD | TBD | TBD | TBD |

**W4: Shared System Prompt** — Prefix caching benefit under load (2K/8K/16K prompt):

| Prompt Len | Concurrency | TTFT p50 (ms) | Cache Hit Rate |
|-----------|-------------|--------------|----------------|
| 2000 | 4 | TBD | TBD |
| 8000 | 8 | TBD | TBD |
| 16000 | 8 | TBD | TBD |

**W5: ShareGPT-style Conversations** — QPS sweep (0.5-8.0):

| QPS | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | Throughput (tok/s) | Success Rate |
|-----|--------------|--------------|-------------|-------------------|-------------|
| 0.5 | TBD | TBD | TBD | TBD | TBD |
| 1.0 | TBD | TBD | TBD | TBD | TBD |
| 2.0 | TBD | TBD | TBD | TBD | TBD |
| 4.0 | TBD | TBD | TBD | TBD | TBD |
| 8.0 | TBD | TBD | TBD | TBD | TBD |

**W6: Long Context Scaling** — Input length sweep (1K-16K tokens):

| Input Len | Output | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | Throughput (tok/s) |
|----------|--------|--------------|--------------|-------------|-------------------|
| 1024 | 512 | TBD | TBD | TBD | TBD |
| 4096 | 512 | TBD | TBD | TBD | TBD |
| 8192 | 512 | TBD | TBD | TBD | TBD |
| 16384 | 512 | TBD | TBD | TBD | TBD |

**Output**: JSON results at `/results/benchmark_gemma4-31b-bf16-tp2_<timestamp>.json`. Copy to `blueprints/gemma4-hyperpod/results/`.

### P1v: vllm bench serve Sweep

Standard `vllm bench serve` phases for direct comparison with other blueprint results.

**P1v-a: QPS Sweep** (find max QPS meeting SLO):

```bash
for QPS in 0.5 1.0 2.0 4.0 8.0; do
  vllm bench serve \
    --model google/gemma-4-31B-it \
    --base-url http://localhost:8002 \
    --dataset-name random \
    --random-input-len 2048 --random-output-len 512 \
    --num-prompts 100 --request-rate $QPS \
    --warmup 30 --save-result --save-detailed \
    --result-dir /results --result-filename "p1v_qps${QPS}.json"
done
```

**SLO targets**:
- TTFT p99 < 1000ms at 32K context
- ITL p50 < 50ms

| QPS | TTFT p50 (ms) | TTFT p99 (ms) | TPOT (ms) | Output tok/s | Pass SLO? |
|-----|--------------|--------------|-----------|-------------|-----------|
| 0.5 | TBD | TBD | TBD | TBD | TBD |
| 1.0 | TBD | TBD | TBD | TBD | TBD |
| 2.0 | TBD | TBD | TBD | TBD | TBD |
| 4.0 | TBD | TBD | TBD | TBD | TBD |
| 8.0 | TBD | TBD | TBD | TBD | TBD |

**P1v-b: Context Scaling**:

```bash
for CTX in 1024 4096 8192 16384 32768; do
  vllm bench serve \
    --model google/gemma-4-31B-it \
    --base-url http://localhost:8002 \
    --dataset-name random \
    --random-input-len $CTX --random-output-len 512 \
    --num-prompts 50 --request-rate 1.0 \
    --warmup 30 --save-result --save-detailed \
    --result-dir /results --result-filename "p1v_ctx${CTX}.json"
done
```

| Context | TTFT p50 (ms) | TTFT p99 (ms) | TPOT (ms) | Output tok/s |
|---------|--------------|--------------|-----------|-------------|
| 1024 | TBD | TBD | TBD | TBD |
| 4096 | TBD | TBD | TBD | TBD |
| 8192 | TBD | TBD | TBD | TBD |
| 16384 | TBD | TBD | TBD | TBD |
| 32768 | TBD | TBD | TBD | TBD |

**P1v-c: Prefix Caching with Shared Prefix**:

```bash
for CTX in 4096 16384 32768; do
  vllm bench serve \
    --model google/gemma-4-31B-it \
    --base-url http://localhost:8002 \
    --dataset-name generated-shared-prefix \
    --gsp-system-prompt-len $((CTX - 256)) --gsp-question-len 256 --gsp-output-len 512 \
    --num-prompts 50 --request-rate 1.0 \
    --warmup 30 --save-result --save-detailed \
    --result-dir /results --result-filename "p1v_gsp_ctx${CTX}.json"
done
```

| Prefix Len | TTFT p50 cold (ms) | TTFT p50 warm (ms) | Cache Speedup |
|-----------|-------------------|-------------------|--------------|
| ~4K | TBD | TBD | TBD |
| ~16K | TBD | TBD | TBD |
| ~32K | TBD | TBD | TBD |

### P2: Code + Reasoning Quality

**P2a: Code Generation** (5 tasks):

| Task | Complexity | LOC |
|------|-----------|-----|
| Parse JSON with error handling | Easy | 20 |
| Binary search implementation | Medium | 30 |
| LRU cache class | Medium | 50 |
| Depth-first search on graph | Medium | 40 |
| Rate limiter with sliding window | Hard | 60 |

**P2b: Reasoning Chain Quality** (3 tasks):

| Task | Type |
|------|------|
| Math word problem (multi-step) | Arithmetic reasoning |
| Logic puzzle (knights/knaves) | Deductive reasoning |
| Algorithm optimization | Algorithmic reasoning |

**P2c: Tool-Use Agent Workflow** — Realistic coding agent scenario (read -> analyze -> fix -> test). Skipped if P0 BFCL < 70%.

### P3: Gemma 4 26B-A4B MoE (Optional)

Run P0c (tool calling) + W5 QPS sweep + P1v-a QPS sweep to compare dense vs MoE within Gemma family. Deploy on port 8003 with TP2.

---

## Success Criteria

| Metric | Target | Phase |
|--------|--------|-------|
| BFCL accuracy | >= 70% | P0 |
| W5 max QPS (TTFT p99 < 2s) | >= 4.0 | P1 |
| W6 TTFT p50 at 16K input | < 500ms | P1 |
| W1 multi-turn 10 rounds TTFT p50 | < 1000ms | P1 |
| W2 RAG cache hit TTFT improvement | >= 30% | P1 |
| W3 agentic tool calling E2E | < 10s per turn | P1 |
| vllm bench TTFT p99 at 32K | < 1000ms | P1v |
| vllm bench ITL p50 | < 50ms | P1v |
| QPS at SLO (vllm bench) | >= 2.0 | P1v |
| Prefix cache speedup (vllm bench) | >= 2x | P1v |
| Code generation pass rate | >= 60% | P2 |
| Reasoning correctness | >= 2/3 tasks | P2 |
| Agent workflow correctness | >= 80% | P2 |

---

## Non-Requirements

- Multi-node distributed inference (single ml.p5.48xlarge)
- Production autoscaling (eval session, fixed capacity)
- Full 256K context evaluation (using 32K max)
- vLLM KV cache offloading tests
- LMCache integration (incompatible with head_dim heterogeneity)
- SageMaker endpoint creation (direct pod access)

---

## Security Requirements

- **Encryption**: S3 SSE-S3, EBS KMS encryption
- **Network**: Private subnets only, no public IPs
- **VPC Endpoints**: All AWS service access via endpoints
- **Node Access**: SSM Session Manager only
- **IAM**: Least privilege roles

---

## Cost Considerations

FTP cost ($636) shared with Mistral Small 4 eval. Gemma 4 portion estimated at ~$300 (roughly half the time budget). S3 model storage: ~126 GB (~$3/month pro-rated). Delete after session.

---

## Deployment Sequence

```
1. Pre-session: Download gemma-4-31B-it weights (~126 GB) -> S3
2. FTP start: HyperPod cluster active, deep health checks pass
3. Stage weights: S3 -> NVMe init container
4. Deploy: vLLM pod (TP2, port 8000), wait for model load (~5 min)
5. Deploy: bench-runner pod with benchmark-serving.py ConfigMap
6. P0: Smoke test + BFCL (30 min) -> GATE
7. P1: Standard workload sweep W1-W6 via bench-runner pod (2 hrs)
8. P1v: vllm bench serve sweep — QPS, context scaling, prefix caching (1.5 hrs)
9. P2: Code + reasoning (45 min)
10. P3: 26B-A4B MoE if time permits (1 hr)
11. Export results (JSON + logs) to S3 and blueprints/gemma4-hyperpod/results/
12. Teardown (shared with Mistral SM4 eval)
```

---

> **Note**: Operational artifacts (lessons, results, deployment notes)
> belong in the blueprint directory: `blueprints/gemma4-hyperpod/`.
