# Mistral Small 4 119B on SageMaker HyperPod — Serving Evaluation Spec

## Status: DRAFT (2026-04-06)

## Overview

Evaluate **Mistral Small 4 119B** (MoE, 6.5B active) on SageMaker HyperPod with Flexible Training Plan (FTP) capacity. Time-bounded evaluation session to validate serving feasibility, measure baseline performance, test tool calling, and explore EAGLE speculative decoding + NVFP4 quantization on ml.p5.48xlarge (8x H100 80GB).

**Why Mistral Small 4:** First Mistral with MLA (Multi-head Latent Attention, like DeepSeek V3), MoE with 128 routed experts + 1 shared, FP8/NVFP4 native quantization, EAGLE speculative decode support. Apache 2.0 licensed.

**Test window:** April 6, 2026, 12:55 PM ET -> April 7, 2026, ~8:00 AM ET (~19 hours). Shared with Gemma 4 eval (see `gemma4-hyperpod.md`).

**Primary goals:**
1. Validate Mistral SM4 loads and serves correctly on H100 with MLA backend
2. Measure single-request and concurrent throughput/latency (FP8 TP2)
3. Test tool calling accuracy (Mistral parser + auto tool choice)
4. Test EAGLE speculative decoding speedup
5. Test NVFP4 quantization (TP1 feasibility)
6. Identify blockers for production deployment

---

## Shared Infrastructure

This spec shares compute, networking, storage, and monitoring with `gemma4-hyperpod.md`. Both models run on the same cluster during the same FTP window.

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

**Mistral SM4 compatibility:** MLA requires FlashAttention 3 or TRITON_MLA backend. Both available on H100. NCCL 2.26.2+ recommended.

### Networking

- **VPC**: vpc-0ac42dd6bad805ebf (pre-provisioned)
- **Private Subnet**: subnet-096f79eef468898e4 (10.1.0.0/16, us-east-2a)
- **Security Group**: sg-0b38a70db0a6a994f (EFA-enabled, self-referencing)
- **VPC Endpoints**: S3, ECR API, ECR DKR, STS, CloudWatch Logs, SSM, EC2
- **Access**: SSM Session Manager only (no public SSH)
- **Model Invocation**: Direct HTTP to pod IP (no load balancer for eval session)

### Storage

**Model Weights — S3**:
- Bucket: s3://hyperpod-eks-bucket-495365983931-us-east-2
- Path: `s3://.../models/Mistral-Small-4-119B-2603/` (~238 GB FP8)

**Pre-session setup**:

```bash
huggingface-cli download mistralai/Mistral-Small-4-119B-2603 \
  --local-dir ./Mistral-Small-4-119B-2603/ \
  --local-dir-use-symlinks False

aws s3 sync ./Mistral-Small-4-119B-2603/ \
  s3://hyperpod-eks-bucket-495365983931-us-east-2/models/Mistral-Small-4-119B-2603/ \
  --profile agent --region us-east-2
```

**EAGLE draft model** (optional, for P3a):

```bash
huggingface-cli download mistralai/Mistral-Small-4-119B-2603-eagle \
  --local-dir ./Mistral-Small-4-119B-2603-eagle/

aws s3 sync ./Mistral-Small-4-119B-2603-eagle/ \
  s3://hyperpod-eks-bucket-495365983931-us-east-2/models/Mistral-Small-4-119B-2603-eagle/ \
  --profile agent --region us-east-2
```

**Node Storage**: NVMe: 8x 3.84 TB SSDs (~30 TB total), EBS: 500 GB (system + logs).

**Staging Strategy**: S3 -> NVMe init container -> mount to pod `/models/`.

### Monitoring

**Prometheus** (shared deployment):
- Scrape interval: 15s
- Targets: vLLM `/metrics` on port 8001 (FP8), 8002 (NVFP4 if tested)
- Retention: 4 hours

**Key Metrics**:

| Metric | Purpose |
|--------|---------|
| `vllm:num_running_requests` | Current load |
| `vllm:num_waiting_requests` | Queue depth |
| `vllm:time_to_first_token_seconds` | Prefill latency |
| `vllm:inter_token_latency_seconds` | Decode latency |
| `vllm:kv_cache_usage_percent` | Memory pressure |
| `vllm:num_preemptions_total` | OOM indicator |
| `vllm:prefix_cache_hit_rate` | Cache effectiveness |
| `DCGM_FI_DEV_GPU_UTIL` | GPU compute utilization |
| `DCGM_FI_DEV_FB_USED` | GPU memory usage |

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

### Mistral Small 4 (119B MoE)

- **Model ID**: `mistralai/Mistral-Small-4-119B-2603`
- **Architecture**: MoE decoder-only transformer
  - 119B total params, 6.5B active per token
  - 128 routed experts + 1 shared expert, top-4 routing
- **Attention**: MLA (Multi-head Latent Attention) — first Mistral model with this
  - Like DeepSeek V3, compresses KV cache via latent projection
  - Uses `FLASH_ATTN_MLA` backend on H100 for FP8
  - Uses `TRITON_MLA` backend for NVFP4
- **Context**: 256K native (YaRN RoPE), 1M max positions
- **Multimodal**: Text + image (Idefics vision encoder)
- **VRAM Requirements**:
  - FP8 (default shipped): ~119 GB (TP2 on 8x H100)
  - NVFP4: ~60 GB (TP1 possible, but TP2 safer for performance)
- **Speculative Decoding**: EAGLE available (`mistralai/Mistral-Small-4-119B-2603-eagle`)
- **License**: Apache 2.0

### Parallelism Strategy

| Variant | Total VRAM | TP | GPUs | Headroom | Use Case |
|---------|-----------|----|----|----------|----------|
| **FP8** | 119 GB | 2 | 2 | 41 GB | Primary eval, TP2 required |
| **NVFP4** | 60 GB | 1 | 1 | 20 GB | TP1 test if time permits (P3b) |

FP8 requires TP2 (119 GB > 80 GB single GPU). Uses 2 of 8 GPUs, leaving 6 for Gemma 4.

### Serving Configuration — FP8 (Primary)

**Container Image**: `vllm/vllm-openai:v0.6.6` or `mistralllm/vllm-ms4:latest` (if official vLLM lacks Mistral 4 support)

```bash
vllm serve mistralai/Mistral-Small-4-119B-2603 \
  --tensor-parallel-size 2 \
  --dtype float8 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --reasoning-parser mistral \
  --port 8001
```

**vLLM Details**:
- `model_type: mistral4`
- `class: Mistral3ForConditionalGeneration` (Mistral 4 inherits from 3 class)
- `--tool-call-parser mistral --enable-auto-tool-choice`
- `--reasoning-parser mistral` (supports `reasoning_effort` param: none/high)

### Serving Configuration — NVFP4 (Optional, P3b)

```bash
vllm serve mistralai/Mistral-Small-4-119B-2603 \
  --tensor-parallel-size 1 \
  --dtype nvfp4 \
  --attention-backend triton_mla \
  --max-model-len 32768 \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --port 8002
```

### Serving Configuration — EAGLE Speculative Decode (Optional, P3a)

```bash
vllm serve mistralai/Mistral-Small-4-119B-2603 \
  --tensor-parallel-size 2 \
  --dtype float8 \
  --max-model-len 32768 \
  --speculative-model mistralai/Mistral-Small-4-119B-2603-eagle \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --port 8001
```

### Known Issues

1. **LMCache incompatible with MLA**: Multi-group KV cache mismatch. PR #2951 may fix, but unvalidated. Do not attempt LMCache in this session.
2. **Container availability**: Official vLLM may not have Mistral 4 support yet. Fall back to `mistralllm/vllm-ms4:latest`.
3. **vLLM KV offloading untested with MLA**: `--kv-offloading-size` behavior unknown.
4. **EAGLE speculative decode unvalidated**: Draft model download + serving config not tested. P3a is optional.
5. **NVFP4 attention backend**: Requires `--attention-backend triton_mla`. Untested on H100. P3b is optional.

---

## Benchmark Design

Time budget: Shared 19-hour FTP window. Mistral SM4 benchmarks target ~7.5 hours total (increased for standard workload sweep + EAGLE + NVFP4).

### Priority Tiers

| Priority | Phase | Time | Deliverable |
|----------|-------|------|-------------|
| **P0** | Smoke test + tool calling | 30 min | Model serves, tools work |
| **P1** | Standard workload sweep (W1-W6) | 2 hrs | Comprehensive serving profile via bench-runner pod |
| **P1v** | vllm bench serve sweep | 1.5 hrs | QPS sweep, context scaling, prefix caching |
| **P2** | Code + reasoning tasks | 45 min | Quality benchmarks |
| **P3a** (optional) | EAGLE speculative decode | 1.5 hrs | Decode speedup + re-run W5/P1v-a for comparison |
| **P3b** (optional) | NVFP4 quantization TP1 | 1.5 hrs | Quality/latency vs FP8 TP2 + W5/P1v-a |

### P0: Smoke Test + Tool Calling

| Step | Test | Expected |
|------|------|----------|
| 1 | Health check | `/health` returns 200 |
| 2 | Basic generation | Prompt: "Hello" -> coherent response |
| 3 | Context length | 8K input + 512 output |
| 4 | Tool call (single) | `read_file(path="/etc/hosts")` -> correct tool call JSON |
| 5 | Multi-turn | 3-turn conversation with memory |
| 6 | Reasoning effort | Test `reasoning_effort: high` param |

**Tool Calling Accuracy** — BFCL subset (50 scenarios):

| Category | Scenarios | What It Tests |
|----------|-----------|---------------|
| Simple function call | 10 | Single tool, correct arguments |
| Multi-tool selection | 10 | Choose correct tool from 5+ options |
| Parallel tool calls | 10 | Call 2+ tools in one response |
| Multi-turn tool use | 10 | Tool result -> follow-up call |
| Structured output | 10 | JSON schema compliance |

**Note**: Tool calling may take significant time per scenario. Budget 30 min for 50 scenarios. If P0 takes >45 min, reduce to 30 scenarios and proceed.

**Gate**: BFCL accuracy >= 75% to proceed (higher bar than Gemma 4 — Mistral has established tool calling track record). If < 75%, skip P2c.

### P1: Standard Workload Sweep (W1-W6) — bench-runner Pod

Run the standard `benchmark-serving.py` workloads via in-cluster bench-runner pod. This provides the common benchmark baseline comparable across all models in the repo.

**Pod Setup**:

```bash
# Create ConfigMap with benchmark script
kubectl create configmap benchmark-scripts \
  --from-file=benchmark-serving.py=/scripts/benchmark-serving.py

# Deploy bench-runner pod (edit env vars first)
# Set BENCHMARK_API_URL=http://<mistral-sm4-pod-ip>:8001
# Set BENCHMARK_MODEL=mistralai/Mistral-Small-4-119B-2603
kubectl apply -f scripts/bench-runner-pod.yaml

# Run all workloads
kubectl exec bench-runner -- python /scripts/benchmark-serving.py \
  --api-url $BENCHMARK_API_URL \
  --model $BENCHMARK_MODEL \
  --config mistral-sm4-fp8-tp2 \
  --workloads w1,w2,w3,w4,w5,w6 \
  --output-dir /results
```

**W1: Multi-Turn Chat** — Sweep rounds (1/5/10) x concurrency (1/4/8) x QPS (1.0/4.0):

| Rounds | Concurrency | QPS | TTFT p50 (ms) | ITL p50 (ms) | Throughput (tok/s) |
|--------|-------------|-----|--------------|-------------|-------------------|
| 1 | 1 | 1.0 | TBD | TBD | TBD |
| 5 | 4 | 4.0 | TBD | TBD | TBD |
| 10 | 8 | 4.0 | TBD | TBD | TBD |

**Expected**: MoE (6.5B active) should show lower TTFT than Gemma 4 31B (30.7B dense) at same concurrency.

**W2: RAG / Long Document QA** — Shared document prefix (2K/5K/10K tokens) with cache warmup:

| Doc Tokens | Warmup:Query | Concurrency | TTFT p50 (ms) | Cache Benefit |
|-----------|-------------|-------------|--------------|--------------|
| 2000 | 2:2 | 4 | TBD | TBD |
| 5000 | 3:1 | 8 | TBD | TBD |
| 10000 | 4:1 | 8 | TBD | TBD |

**Note**: MLA compresses KV cache — prefix caching benefit may differ from standard attention models.

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

**Output**: JSON results at `/results/benchmark_mistral-sm4-fp8-tp2_<timestamp>.json`. Copy to `blueprints/mistral-small-4-hyperpod/results/`.

### P1v: vllm bench serve Sweep

Standard `vllm bench serve` phases for direct comparison with other blueprint results.

**P1v-a: QPS Sweep** (find max QPS meeting SLO):

```bash
for QPS in 0.5 1.0 2.0 4.0 8.0; do
  vllm bench serve \
    --model mistralai/Mistral-Small-4-119B-2603 \
    --base-url http://localhost:8001 \
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
    --model mistralai/Mistral-Small-4-119B-2603 \
    --base-url http://localhost:8001 \
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
    --model mistralai/Mistral-Small-4-119B-2603 \
    --base-url http://localhost:8001 \
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

**Note**: MLA latent projection compresses KV cache differently — prefix caching effectiveness may be lower than standard GQA models. This is a key finding to capture.

### P2: Code + Reasoning Quality

Same tasks as Gemma 4 spec (see `gemma4-hyperpod.md` P2a/P2b/P2c) for direct comparison.

Additionally test Mistral-specific features:
- `reasoning_effort: high` on reasoning tasks (P2b)
- Compare reasoning quality with/without effort param
- Compare `reasoning_effort: none` vs `high` on code generation (P2a)

### P3a: EAGLE Speculative Decoding (Optional)

Restart vLLM with EAGLE draft model, then re-run key benchmarks for direct comparison.

**Re-run with EAGLE**:

```bash
# W5 QPS sweep (bench-runner pod)
kubectl exec bench-runner -- python /scripts/benchmark-serving.py \
  --api-url $BENCHMARK_API_URL \
  --model $BENCHMARK_MODEL \
  --config mistral-sm4-fp8-tp2-eagle \
  --workloads w5 \
  --output-dir /results

# P1v-a QPS sweep (vllm bench serve)
for QPS in 0.5 1.0 2.0 4.0 8.0; do
  vllm bench serve \
    --model mistralai/Mistral-Small-4-119B-2603 \
    --base-url http://localhost:8001 \
    --dataset-name random \
    --random-input-len 2048 --random-output-len 512 \
    --num-prompts 100 --request-rate $QPS \
    --warmup 30 --save-result --save-detailed \
    --result-dir /results --result-filename "p3a_eagle_qps${QPS}.json"
done
```

| Config | W5 QPS=4.0 TTFT p50 | W5 QPS=4.0 ITL p50 | W5 QPS=4.0 tok/s | vllm bench ITL p50 |
|--------|---------------------|--------------------|-----------------|--------------------|
| Baseline (no EAGLE) | TBD | TBD | TBD | TBD |
| + EAGLE draft | TBD | TBD | TBD | TBD |
| **Speedup** | TBD | TBD | TBD | TBD |

**Expected**: 1.3-1.5x speedup on decode (ITL) with negligible TTFT overhead.

### P3b: NVFP4 Quantization (Optional)

Restart vLLM with NVFP4 TP1, then re-run key benchmarks for quality + perf comparison.

```bash
# W5 QPS sweep
kubectl exec bench-runner -- python /scripts/benchmark-serving.py \
  --api-url http://<nvfp4-pod-ip>:8002 \
  --model $BENCHMARK_MODEL \
  --config mistral-sm4-nvfp4-tp1 \
  --workloads w5,w6 \
  --output-dir /results

# P1v-a QPS sweep
for QPS in 0.5 1.0 2.0 4.0 8.0; do
  vllm bench serve \
    --model mistralai/Mistral-Small-4-119B-2603 \
    --base-url http://localhost:8002 \
    --dataset-name random \
    --random-input-len 2048 --random-output-len 512 \
    --num-prompts 100 --request-rate $QPS \
    --warmup 30 --save-result --save-detailed \
    --result-dir /results --result-filename "p3b_nvfp4_qps${QPS}.json"
done
```

| Metric | FP8 TP2 | NVFP4 TP1 | Delta |
|--------|---------|-----------|-------|
| W5 max QPS at SLO | TBD | TBD | TBD |
| W6 TTFT p50 at 16K | TBD | TBD | TBD |
| vllm bench ITL p50 | TBD | TBD | TBD |
| vllm bench max QPS | TBD | TBD | TBD |
| BFCL accuracy | TBD | TBD | TBD |
| Code gen pass rate | TBD | TBD | TBD |

**Expected**: 5-10% quality degradation, 20-30% latency increase, but 2x lower VRAM. Key question: is NVFP4 TP1 viable for latency-sensitive use cases?

---

## Success Criteria

| Metric | Target | Phase |
|--------|--------|-------|
| BFCL accuracy | >= 75% | P0 |
| W5 max QPS (TTFT p99 < 2s) | >= 4.0 | P1 |
| W6 TTFT p50 at 16K input | < 300ms | P1 (MoE should be faster than dense) |
| W1 multi-turn 10 rounds TTFT p50 | < 800ms | P1 |
| W2 RAG cache hit TTFT improvement | >= 20% | P1 (lower bar — MLA cache may behave differently) |
| W3 agentic tool calling E2E | < 8s per turn | P1 (MoE should be faster) |
| vllm bench TTFT p99 at 32K | < 1000ms | P1v |
| vllm bench ITL p50 | < 50ms | P1v |
| QPS at SLO (vllm bench) | >= 2.0 | P1v |
| Prefix cache speedup (vllm bench) | >= 1.5x | P1v (lower bar — MLA uncertainty) |
| Code generation pass rate | >= 60% | P2 |
| Reasoning correctness | >= 2/3 tasks | P2 |
| Agent workflow correctness | >= 80% | P2 |
| EAGLE decode speedup | >= 1.3x | P3a |
| NVFP4 quality retention | >= 90% of FP8 BFCL | P3b |

---

## Non-Requirements

- Multi-node distributed inference (single ml.p5.48xlarge)
- Production autoscaling (eval session, fixed capacity)
- Full 256K context evaluation (using 32K max)
- vLLM KV cache offloading tests
- LMCache integration (incompatible with MLA)
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

FTP cost ($636) shared with Gemma 4 eval. Mistral SM4 portion estimated at ~$336 (slightly more time due to EAGLE/NVFP4 tests). S3 model storage: ~238 GB FP8 + EAGLE draft (~$6/month pro-rated). Delete after session.

---

## Deployment Sequence

```
1. Pre-session: Download Mistral-Small-4-119B-2603 weights (~238 GB) -> S3
   Optional: Download EAGLE draft model -> S3
2. FTP start: HyperPod cluster active, deep health checks pass
3. Stage weights: S3 -> NVMe init container
4. Deploy: vLLM pod (FP8 TP2, port 8001), wait for model load (~5 min)
5. Deploy: bench-runner pod with benchmark-serving.py ConfigMap (shared with Gemma 4)
6. P0: Smoke test + BFCL (30 min) -> GATE
7. P1: Standard workload sweep W1-W6 via bench-runner pod (2 hrs)
8. P1v: vllm bench serve sweep — QPS, context scaling, prefix caching (1.5 hrs)
9. P2: Code + reasoning (45 min)
10. P3a: Restart with EAGLE -> re-run W5 + P1v-a (1.5 hrs)
11. P3b: Restart with NVFP4 TP1 -> re-run W5 + W6 + P1v-a + BFCL (1.5 hrs)
12. Export results (JSON + logs) to S3 and blueprints/mistral-small-4-hyperpod/results/
13. Teardown (shared with Gemma 4 eval)
```

---

> **Note**: Operational artifacts (lessons, results, deployment notes)
> belong in the blueprint directory: `blueprints/mistral-small-4-hyperpod/`.
