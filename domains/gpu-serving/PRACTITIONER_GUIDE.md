# GPU Serving Practitioner Guide

> Operational playbook distilled from 19 blueprint deployments across g5, g6e, g7e, p5, p5e, p5en, and p6-b200 instances. Covers vLLM, SGLang, Ray Serve, llm-d, and NVIDIA Dynamo on EKS and SageMaker HyperPod.

---

## Table of Contents

0. [First-Principles Preflight](#0-first-principles-preflight) — *the generating function behind §§1, 7, 11*
1. [Hardware Selection](#1-hardware-selection)
2. [Pre-Flight Validation](#2-pre-flight-validation)
3. [Infrastructure Provisioning](#3-infrastructure-provisioning)
4. [Model Staging](#4-model-staging)
5. [Serving Engine Configuration](#5-serving-engine-configuration)
6. [Model-Specific Requirements](#6-model-specific-requirements)
7. [KV Cache and Memory Optimization](#7-kv-cache-and-memory-optimization)
8. [Routing and Load Balancing](#8-routing-and-load-balancing)
9. [Benchmarking](#9-benchmarking)
10. [Troubleshooting](#10-troubleshooting)
11. [Cost Optimization](#11-cost-optimization)
12. [Quick Reference Commands](#12-quick-reference-commands)

---

## 0. First-Principles Preflight

> **Read this before §1.** §§1–11 are *memorized answers* — "model size X → instance Y," tuned flags, hard-won gotchas. This section is the **generating function** that produces them. When a blueprint hits a model the tables didn't anticipate, reason from here. The whole section is ~6 equations; everything downstream is a special case.

### How to read this guide: three knowledge tiers

Every claim in this playbook has a **half-life**. Mistaking a short-lived fact for a durable one is the single most common way a spec goes wrong (it's the same defect class the `carryover-auditor` guards against). So each heuristic is tagged:

| Tier | Half-life | Holds across… | Example |
|------|-----------|---------------|---------|
| **[T1]** universal | ~never | any cloud, any model, any year | `T = max(t_compute, t_mem)`; decode is bandwidth-bound |
| **[T2]** environment | years | until the provider/platform changes | EFA is SRD, not true RDMA; g7e uses `nerdctl` |
| **[T3]** release | weeks–months | one version window — **must carry a `validated:` date** | NCCL 2.25.1 broken on sm_120; vLLM 0.18.1 has no draft-MoE knob |

T1 is asserted plainly. **T2/T3 always carry their qualifier** ("on AWS…", "as of vLLM 0.18.1…") so neither a human nor an agent mistakes a decaying fact for a law. T3 rules reuse the repo's version-stamp convention (`<!-- stack: … | validated: YYYY-MM-DD -->`, see `tech-stack.md`). **The reasoner below is pure T1.** It only *consults* T2/T3 to confirm the clean prediction survives contact with the stack — which is the whole point of the closing rule: **theory predicts the regime; measurement confirms it.** Measurement is exactly where T2/T3 reality intrudes on T1 math.

### The two rooflines [T1]

Every inference efficiency question descends from one fact: a transformer's two phases sit on **opposite sides of the roofline**.

```
forward-pass time   T = max( t_compute , t_mem )

  t_compute = (B · N_active) / FLOPs            ← batch × active params, over chip FLOPs
  t_mem     = (N_total + B · L_ctx · kv_bytes) / mem_bw
              \________/   \__________________/
              weight fetch   KV-cache fetch
```

- **Prefill** (process the prompt): compute-bound. One big matmul over all prompt tokens at once → high arithmetic intensity. **FLOPs are the constraint.**
- **Decode** (generate one token at a time): memory-bandwidth-bound. Each step reads the *entire* model + KV cache to emit one token → arithmetic intensity ~1–2 FLOP/byte. **Bandwidth and latency are the constraint; tensor cores sit idle.**

A **latency floor** exists for any hardware: you must read all params from HBM at least once, so `T ≥ N_total / mem_bw`. No batching or kernel trick beats it.

*Origin: Pope et al., "Efficiently Scaling Transformer Inference" (arXiv:2211.05102). Re-derived first-principles in the Dwarkesh × Reiner Pope blackboard lecture — see vault note `Reiner Pope - How LLMs Are Trained and Served`.*

### Optimal batch size [T1]

Set `t_compute = t_mem` (weight term) and the model size cancels out:

```
B*  ≈  300 × sparsity          sparsity = N_total / N_active
```

- The **~300** is a dimensionless hardware constant (FLOP-per-byte at the precision used), remarkably stable across GPU generations because FLOPs and bandwidth scaled together. **[T1]**, but spot-check per precision: it's FP4 vs FP8 vs FP16 dependent, and the exact number drifts with hardware **[T3]**.
- DeepSeek-V3 activates 32/256 experts → sparsity ≈ 8 → **B\* ≈ 2,400 sequences.** In practice run 2–3× higher (roofline overstates real efficiency).
- **Batch optimum depends only on sparsity, not model scale.** More sparsity → less compute but bigger batches needed (more HBM for KV). "Keep increasing sparsity until you run out of users."

### The decision procedure

Given a model (`N_total`, `N_active`, `kv_bytes/token`) and target hardware (`FLOPs`, `mem_bw`, HBM/GPU, scale-up domain size), answer in order:

**1 · Which roofline am I on?** Compute the arithmetic intensity of your target workload and compare to the machine balance (`FLOPs ÷ mem_bw`, ~300–630 FLOP/byte at low precision on modern GPUs **[T3]**).

| Regime | Symptom | The lever (what to optimize) | Wrong lever (no-op) |
|--------|---------|------------------------------|---------------------|
| **bandwidth-bound** | decode; AI ≪ machine balance; HBM util high, tensor low | more HBM bandwidth (newer gen / HBM4), shrink bytes/token (quant, MLA/GQA), bigger batch to amortize weights | more FLOPs; bigger FP4 numbers |
| **compute-bound** | prefill, large-batch decode; tensor cores hot | more FLOPs, FP4/FP8, better GEMM kernels | more bandwidth |
| **capacity-bound** | model + KV won't fit; OOM at target concurrency | more HBM/GPU, more GPUs, quantize weights, pipeline across racks | faster kernels |
| **launch/scheduling-bound** | small model; SM ~50%, HBM ~15%, tensor ~11% at SLO knee | kernel fusion, CUDA graphs, megakernels — **software, not silicon** | **a bigger chip** (B300 over B200 adds capacity + FP4, not the bandwidth you lack) |

> **The decode-bytes decomposition that prevents the most common error [T1]:** `decode bytes/token = active-weight read (N_active × dtype) + KV read (kv_bytes × dtype)`. Attention innovations (GQA, MLA, Mamba, sparse) **only shrink the second term.** A 32B-active MoE in FP8 reads ~16 GB of weights/token — even with MLA crushing KV 20×, the **weight term is ~95% of decode bytes**, so it stays firmly bandwidth-bound. "It's sparse, so it's not the bottleneck" is the trap. Check which term dominates *before* concluding anything.

**2 · What's my B\*, and does target concurrency reach it?** Below B\* you're wasting the GPU on un-amortized weight fetches; the cost/token curve is still falling. At/above B\* you've hit the compute floor. This sets your throughput target and tells you if "slow mode / low QPS" will ever be cost-efficient (it won't — compute and KV don't amortize).

**3 · Does the model + KV for target concurrency fit and saturate ONE node?** This is the disagg test (full version in §7 and the inference-bottleneck report). The screen:

```
fits + saturates one node  →  replicas + chunked prefill   (≈95% of deployments; disagg is over-engineering)
forced onto a 2nd node     →  only THEN consider disaggregation
  (forced by: weights too big, 100K+ context whose prefill alone exhausts a node, or QPS beyond one box)
```

Parameter count is **not** the axis — a 1T-param MoE (32B active) fits one node and needs no disagg; "sounds frontier" predicts nothing. "Forced onto a second node" predicts everything. **[T1] in logic; the cross-node tax that makes disagg lose is [T2]:** on AWS, EFA is SRD not true RDMA — a TCP fallback turned a 355 ms TTFT into 10+ s in our testing. The moment you're forced cross-node is often the moment you *lack* the fabric that makes disagg pay.

**4 · Is pipelining worth it? (almost never, for inference) [T1]** Pipeline parallelism solves weight **capacity** (each rack holds 1/P of weights) — but a Blackwell rack already has tens of TB and a 1T model needs ~1 TB, so the benefit is usually nil. Crucially it **does not shard the KV cache**: keeping P stages busy needs P micro-batches in flight, so concurrent sequences scale with P and the saving cancels. Inference recipe (matches DeepSeek's published setup): **max expert parallelism to the scale-up domain size, then little-to-no pipelining.**

**5 · Why does scale-up domain size matter? Bandwidth, not capacity. [T1]** Weight-load bandwidth = scale-up size × per-GPU bandwidth. A bigger NVLink domain lets you load weights in parallel → lower decode latency and longer feasible context. This is *why* NVL72 matters, and why interconnect is a lever to exploit, not a bottleneck to fear.

### The closing rule — theory predicts, measurement confirms

Every heuristic above outputs a **predicted regime**, never a deploy command. The prediction is T1 physics; the stack is full of T2/T3 reality that can flip it (a kernel that ignores your tile config, an engine without the flag you need, an AMI that won't boot). So every preflight conclusion ends the same way:

> *"This model should be `<regime>`-bound on `<hardware>`, so optimize `<lever>` — **now confirm with `nvidia-smi dmon` / a benchmark sweep before trusting it.**"*

If measurement contradicts the prediction, you've found a T2/T3 quirk worth a lessons.md entry — not a reason to distrust the physics. See §2 (Pre-Flight Validation) for the measurement commands, §9 (Benchmarking) for the sweep, and §10 (Troubleshooting) for regime-confirmation queries.

---

## 1. Hardware Selection

### Instance Comparison Matrix

| Instance | GPU | Count | VRAM/GPU | Interconnect | Cost/GPU/hr | Best For |
|----------|-----|-------|----------|--------------|-------------|----------|
| g5.4xlarge | A10G | 1 | 24 GB GDDR6 | PCIe | ~$1.51 | Small models (<8B), validation |
| g6e.2xlarge | L40S | 1 | 48 GB GDDR6 | PCIe | ~$1.86 | Medium models (<32B), EKS baseline |
| g7e.24xlarge | RTX PRO 6000 | 4 | 96 GB GDDR7 | PCIe Gen5 | $2.07 | Cost-optimized MoE serving |
| p5.48xlarge | H100 | 8 | 80 GB HBM3 | NVSwitch | ~$7.00 | Dense models, multi-GPU TP |
| p5e.48xlarge | H200 | 8 | 141 GB HBM3e | NVSwitch | ~$7.91 | Large MoE, long context |
| p5en.48xlarge | H200 | 8 | 141 GB HBM3e | NVLink5 | ~$7.91 | Production baseline |
| p6-b200.48xlarge | B200 | 8 | 183 GB HBM3e | NVSwitch | ~$18.90 | Largest models (700B+), max throughput |

### Decision Tree

```
Is the model < 8B?
  YES → g5.4xlarge (A10G) or g6e.2xlarge (L40S)
  NO → Is it a dense model < 80B?
    YES → g7e.24xlarge TP4 (4.6x cheaper than H200)
    NO → Is it an MoE > 100B?
      YES → Does it need NVLink for NCCL training?
        YES → p5e/p5en (H200) or p6-b200 (B200)
        NO → g7e.24xlarge TP4 (inference uses custom allreduce, not NCCL)
      NO → p5e/p5en for production, g7e for cost optimization
```

### Key Hardware Constraints

**g7e (Blackwell PCIe)**:
- NCCL 2.25.1 broken on sm_120 + PCIe. Fixed in NCCL 2.26.2 (NGC 25.03+). vLLM inference unaffected.
- MTP speculative decoding degrades throughput 2-41%. Only use MTP on NVLink platforms.
- Container runtime is `nerdctl`, not `docker`. Use `sudo nerdctl`.
- EFA supported (2 interfaces on 24xl) but is kernel-bypass (SRD), not true RDMA.
- No capacity blocks. Launch via bare EC2; shotgun across regions.

**p6-b200 (Blackwell NVSwitch)**:
- Must use AL2023 AMI (`ami-02bb9f913067dadb1`). AL2 kernel lacks `ib_umad` for Fabric Manager.
- Instance termination takes ~10 min. Plan for gaps when replacing.
- DeepGEMM JIT compilation: ~15 min cold start. Set readiness probe `initialDelaySeconds: 900`.
- Launch with `--instance-market-options '{"MarketType":"capacity-block"}'`.

**p5/p5e (Hopper)**:
- Mature NCCL support. No PCIe bugs.
- EKS bootstrap on AL2023 uses `nodeadm` (MIME multipart), not `/etc/eks/bootstrap.sh`.

---

## 2. Pre-Flight Validation

Run these checks before deploying any serving workload. Order matters.

### Step 1: GPU Inventory

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
  --format=csv,noheader
```

Expected output examples:
- B200: `NVIDIA B200, 183308 MiB, 580.126.09, 10.0`
- RTX PRO 6000: `NVIDIA RTX PRO 6000, 98304 MiB, 570.x.x, 12.0`
- H200: `NVIDIA H200, 143859 MiB, 555.x.x, 9.0`

### Step 2: Topology

```bash
nvidia-smi topo -m
```

Look for: `NV18` (NVSwitch), `SYS` (PCIe cross-socket), `PHB` (same PCIe bridge).
- NVSwitch: multi-GPU TP works at full bandwidth
- PCIe-only: custom allreduce for inference, NCCL broken on Blackwell

### Step 3: ECC and Health

```bash
nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader
# Zero tolerance for uncorrected errors

dmesg | grep "NVRM: Xid"
# Any Xid errors indicate hardware issues. Use gpu-infra explain_xid for lookup.
```

### Step 4: PCIe Link Width

```bash
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current \
  --format=csv,noheader
```

Expected: Gen5 x16 on g7e/B200, Gen4/5 x16 on H100/H200.

### Step 5: NCCL Collective Test (Multi-GPU Only)

```python
# nccl_diag.py — test all_reduce, broadcast, barrier
import torch, torch.distributed as dist
dist.init_process_group("nccl")
t = torch.ones(1024, device=f"cuda:{dist.get_rank()}")
dist.all_reduce(t)  # If this hangs or crashes, NCCL is broken
```

### Step 6: NVMe Storage

```bash
# Check NVMe RAID (g7e, p5e, p6-b200)
lsblk | grep nvme
df -h /mnt/nvme
# Should show RAID0 array. If not mounted:
mdadm --create /dev/md0 --level=0 --raid-devices=N /dev/nvme*n1
mkfs.xfs /dev/md0 && mount /dev/md0 /mnt/nvme
```

---

## 3. Infrastructure Provisioning

### EKS Cluster (Standard Path)

```bash
cd domains/gpu-serving/blueprints/<name>/terraform
terraform init -upgrade
terraform plan -var-file=<env>.tfvars
terraform apply -var-file=<env>.tfvars
```

**Critical Terraform rules**:
- `var.project_name` must be <= 12 characters (IAM role 64-char limit)
- Default optional features to `false`: `variable "enable_waf" { default = false }`
- GPU node groups need 50+ GB EBS root (container images are 10-16 GB)
- System nodes (m6i/m7i) need 50 GB+ EBS for pip installs and image layers

**State management pitfalls**:
- Never run parallel `terraform apply` — state lock deadlock. Check `ps aux | grep terraform` first.
- If state lock persists: `terraform force-unlock <lock-id>` (not `pkill`).
- Import conflicts (e.g., `bootstrap_self_managed_addons`): Edit state JSON directly with `terraform state pull/push`.

### HyperPod Cluster

```bash
# HyperPod creates EKS cluster + managed node groups
# Use awsome-distributed-training module as baseline
helm repo add sagemaker-hyperpod-cli <url>
helm install hyperpod-helm-chart ...
```

**HyperPod-specific rules**:
- K8s 1.32 required (RestrictedInstanceGroups not supported on 1.33+)
- GPU pods need 3 tolerations:
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
- Inference Operator `EnableFailed` is non-blocking for self-managed llm-d deployments

### Capacity Block Instances

```bash
aws ec2 run-instances \
  --instance-type p6-b200.48xlarge \
  --instance-market-options '{"MarketType":"capacity-block"}' \
  --placement 'AvailabilityZone=us-east-2c' \
  --capacity-reservation-specification \
    'CapacityReservationTarget={CapacityReservationId=cr-XXXXX}' \
  ...
```

To join a capacity block instance to EKS:
```bash
aws eks create-access-entry \
  --cluster-name <cluster> \
  --principal-arn <instance-role-arn> \
  --type EC2_LINUX
```

### Bare EC2 (Fast Path for Benchmarking)

When EKS is overkill or capacity blocks are unavailable:
```bash
aws ec2 run-instances --instance-type g7e.24xlarge --region us-west-2 ...
# SSH in, install nerdctl, mount NVMe, run containers directly
```

---

## 4. Model Staging

### Download to NVMe (Fastest Serving)

```bash
# On the GPU instance
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('org/model-name', local_dir='/mnt/nvme/models/model-name')
"
```

### Stage via S3 + FSx (Production)

```bash
# Upload to S3
aws s3 sync /mnt/nvme/models/model-name s3://bucket/models/model-name

# FSx Data Repository Association auto-imports
# OR manual import:
aws fsx create-data-repository-association \
  --file-system-id fs-XXXXX \
  --data-repository-configuration "S3={BucketPath=s3://bucket/models/}"
```

### Large Model Streaming (700B+ MoE)

For models like GLM-5 (733 GB, 142 shards):
```bash
# Download shard-by-shard to avoid filling local disk
for i in $(seq -f "%05g" 1 142); do
  huggingface-cli download org/model --include "model-${i}-of-00142.safetensors" \
    --local-dir /tmp/shard
  aws s3 cp /tmp/shard/model-${i}-of-00142.safetensors s3://bucket/models/model/
  rm /tmp/shard/model-${i}-of-00142.safetensors
done
```

### Pre-download Tokenizer (Air-Gapped)

vLLM benchmark CLI needs the tokenizer locally:
```bash
# Pass --tokenizer pointing to local model path
vllm bench serve --tokenizer /mnt/nvme/models/model-name --model served-model-name
```

---

## 5. Serving Engine Configuration

### vLLM

**Standard launch (EKS / nerdctl)**:
```bash
# EKS: via Kubernetes Deployment
# Bare metal: via nerdctl
sudo nerdctl run -d --name vllm \
  --gpus '"device=0,1,2,3"' --ipc=host --network=host \
  -v /mnt/nvme:/mnt/nvme:ro \
  vllm/vllm-openai:v0.19.0 \
  --model /mnt/nvme/models/<model> \
  --tensor-parallel-size 4 \
  --quantization fp8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 131072 \
  --enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 256 \
  --port 8000
```

**nerdctl vs Docker syntax differences**:
- nerdctl: `--gpus 4` (count), not `--gpus '"device=0,1,2,3"'` (list)
- nerdctl: cannot combine `-d` with `--rm`
- g7e bare metal: always `--network host` (no CNI plugin)

**Version selection guide**:

| vLLM Version | Use When |
|---|---|
| v0.15.0 | Devstral, Mistral models, stable tool calling |
| v0.18.0+ | GLM-5 MTP speculative decode, FlashMLASparse |
| v0.19.0+ | Gemma 4, Kimi K2, latest model support |
| `glm5` tag | GLM-5 specific (DeepGEMM + glm47 parser) |
| `gemma4` tag | Gemma 4 on A10G (but fails on H100 — use `latest` + git transformers) |
| `qwen3_5-x86_64-cu130` | Qwen3.5 MoE hybrid attention |

**Critical flags by scenario**:

| Flag | When Required | Why |
|---|---|---|
| `--tool-call-parser <name>` | Always for tool-using models | Without it, raw `[TOOL_CALLS]` text returned instead of JSON |
| `--enable-auto-tool-choice` | Mistral models | Enables automatic tool detection |
| `--reasoning-parser <name>` | Models with chain-of-thought | Extracts reasoning into separate field |
| `--attention-backend triton` | Hybrid attention on Blackwell | FlashInfer doesn't support hybrid GDN |
| `--fp8-gemm-backend cutlass` | FP8 on Blackwell (SGLang) | DeepGEMM crashes with non-ue8m0 scale formats |
| `--enforce-eager` | LMCache, small models | Disables CUDA graphs (required by LMCache) |
| `--disable-cuda-graph` | HiCache + hybrid attention | CUDA graph conflicts with dynamic HiCache memory |
| `--kv-cache-dtype fp8` | Memory-constrained | Halves KV cache memory (slight quality trade-off) |

### SGLang

**Standard launch**:
```bash
python3 -m sglang.launch_server \
  --model-path /mnt/nvme/models/<model> \
  --tp-size 8 --dtype bfloat16 \
  --context-length 131072 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --mem-fraction-static 0.90 \
  --tool-call-parser <parser> \
  --served-model-name <name> \
  --host 0.0.0.0 --port 30000
```

**SGLang-specific gotchas**:
- Default host is `127.0.0.1`. Always add `--host 0.0.0.0` for external access.
- `qwen3_coder` parser outputs tool calls as `<tool_call>` XML in content, not structured `tool_calls` array. Scripts need XML fallback parsing.
- GLM-5 requires specialized image: `lmsysorg/sglang:glm5-blackwell` (or `glm5-hopper`).
- HiCache requires `--enable-hierarchical-cache --hicache-size <N>` where N > device KV pool size.

### Ray Serve

**Deployment via KubeRay**:
```bash
kubectl apply -f ray-cluster.yaml
kubectl apply -f ray-service.yaml
```

**Ray Serve specific rules**:
- Ray C++ Redis client does NOT support TLS. ElastiCache Serverless enforces TLS. Solution: stunnel sidecar (`alpine:3.20`) on every pod, proxy `localhost:6380` to ElastiCache TLS endpoint.
- Pin `numpy<2` in runtime_env (Ray CPU image has pyarrow compiled against numpy 1.x).
- `libGL.so.1` missing in Ray CUDA image. Fix: init container copies from `debian:bookworm-slim`.
- Head pod needs 20Gi+ ephemeral-storage for pip installs.
- Worker proxies provide zero-downtime failover when head crashes (route NLB to workers only).

### Tool Call Parser Reference

| Model Family | vLLM Parser | SGLang Parser | Notes |
|---|---|---|---|
| Mistral / Devstral | `mistral` | N/A | Must add `--enable-auto-tool-choice` |
| GLM-5 | `glm47` | `glm47` | Reasoning: `glm45` |
| Qwen3 / Qwen3.5 | `qwen3_xml` | `qwen3_coder` | SGLang outputs XML in content |
| Gemma 4 | `pythonic` | N/A | `gemma4` parser also available |
| Kimi K2 | `kimi_k2` | N/A | Reasoning: `kimi_k2` |

---

## 6. Model-Specific Requirements

### GLM-5 (744B MoE)

```
Architecture: glm_moe_dsa (inherits DeepSeekV2)
Attention: Multi-Latent Attention (MLA) with NSA
Active params: ~40B (top-8 of 256 routed + 1 shared)
Disk: 733 GB (142 shards)
GPU memory: ~175 GB/GPU at TP8
```

- **Image**: `lmsysorg/sglang:glm5-blackwell` or `vllm/vllm-openai:glm5`
- **Cold start**: ~15-16 min (DeepGEMM JIT + torch.compile + CUDA graphs)
- **LMCache**: BLOCKED by NSA/MLA incompatibility (PR #2629). Use SGLang HiCache instead.
- **HiCache**: `--hicache-size 100` (must exceed device KV pool ~82 GB/rank)
- **Peak throughput**: 2,602 tok/s at 128 concurrent (HiCache), 2,374 tok/s (vLLM baseline)
- **Tool calling**: 100% BFCL (vLLM glm47 parser)

### Qwen3.5 MoE (122B-A10B)

```
Architecture: qwen3_5_moe with hybrid linear+full attention
Active params: ~10B
Disk: ~127 GB (FP8)
GPU memory: ~29 GiB/GPU at TP4
```

- **Image**: `vllm/vllm-openai:qwen3_5-x86_64-cu130`
- **GPTQ-Int4 produces garbage**. Use FP8 only.
- **TP8 blocked**: FP8 block_k=128 incompatible with shared expert down_proj partition at TP8. Use TP4.
- **Tool calling**: Use `--tool-call-parser qwen3_xml --reasoning-parser qwen3`
- **Proper tool_calls output**: Unlike Qwen 2.5 (bare JSON), Qwen3.5 outputs structured `tool_calls` with `finish_reason: tool_calls`.

### Qwen3-Next (80B Hybrid MoE)

```
Architecture: Hybrid attention (Mamba + GDN)
FP8 TP8: BLOCKED (block_k=128 divisibility). Use TP4.
```

- **Attention backend**: `--attention-backend triton` required on Blackwell
- **MTP speculative decode**: Only viable on NVLink. Hurts 2-41% on PCIe.
- **CPU offload**: Blocked on vLLM 0.16 V1 for hybrid attention (HMA incompatibility)
- **Prefix caching**: 82% TTFT reduction with shared prefixes at QPS 2
- **Sweet spot**: QPS 4-8 on g7e (1,500-2,200 tok/s), QPS 8 on H200 (2,280 tok/s)

### Gemma 4 (4B E4B / 31B)

```
Architecture: Hybrid sliding window + full attention
head_dim: 256 (local) + 512 (global) — forces TRITON_ATTN
```

- **PyPI transformers lacks `gemma4`**: Must `pip install git+https://github.com/huggingface/transformers`
- **`gemma4` vLLM tag fails on H100**: Use `latest` + git transformers instead
- **TRITON_ATTN impact**: Superlinear TTFT scaling. But prefix caching compensates: 11.1x speedup at 16K on 31B.
- **Tool calling**: 100% BFCL with `tool_choice: "required"` (auto mode puts tools in content text)

### Mistral Small 4 (119B MoE)

```
Architecture: MoE with MLA (Multi-head Latent Attention)
Active params: 6.5B
Attention: FLASH_ATTN_MLA backend
```

- **vLLM 0.19 bug**: `reasoning_effort` kwarg crashes `MistralCommonTokenizer`. Patch in pod startup:
  ```bash
  sed -i 's/version_kwargs\["reasoning_effort"\]/pass  # patched/' "$TOKFILE"
  ```
- **Needs**: `pip install mistral_common>=1.10.0`
- **Prefix caching**: 5.23x speedup at 16K context
- **Peak throughput**: 2,160 output tok/s at QPS 8 (TP2 on H100)

### Nemotron-Super (120B Mamba Hybrid)

```
Architecture: Mamba-2 hybrid (not standard transformer)
KV cache: Hybrid (Mamba state + attention KV)
```

- **Attention backend**: `--attention-backend TRITON_ATTN` required
- **Disaggregated P/D**: NOT supported (Mamba state incompatible with NIXL transfer)
- **Sweet spot**: conc=64 with sub-1s TTFT p50, 1,081 tok/s throughput

### Devstral Small 2 (24B)

```
Architecture: mistral3 (Mistral3ForConditionalGeneration)
Must import directly, not via AutoModelForCausalLM
```

- **Flags**: `--tool-call-parser mistral --enable-auto-tool-choice`
- **Single GPU**: Fits on 1x RTX PRO 6000 (96 GB) with FP8 at `--gpu-memory-utilization 0.95`
- **4-replica pattern**: 4 isolated vLLM instances on 4 GPUs with round-robin proxy for swarm workloads

### Kimi K2 (1T MoE)

```
Architecture: 1T MoE, 32B active, native INT4 QAT
Context: 256K with always-on thinking (96-128K tokens)
```

- **INT4 Marlin**: Works on H200, untested on B200
- **SGLang**: Incompatible (INT4 packed weight format)
- **Reasoning parser adds ~10s hidden latency**: All TTFT benchmarks must account for thinking time
- **KV capacity on H200**: 610K tokens — most workloads fit in GPU. FSx value is persistence + multi-node sharing, not offloading.

---

## 7. KV Cache and Memory Optimization

### Strategy Decision Tree

```
Does the model use MLA/NSA attention?
  YES → Is LMCache PR #2629 merged?
    YES → LMCache (L1 CPU + L2 Redis + L3 FSx)
    NO → SGLang HiCache (CPU offload, works with MLA)
  NO → Standard model
    → Does workload have shared prefixes?
      YES → Enable prefix caching (biggest single optimization: 50-82% TTFT reduction)
      NO → Consider KV cache dtype FP8 for capacity

Is GPU VRAM sufficient for target concurrency?
  YES → Prefix caching alone is sufficient
  NO → Add CPU offload tier
    → H200/B200: HiCache or LMCache CPU tier (abundant DRAM)
    → g7e: Multi-tier (GPU → CPU → NVMe via GDS)
```

### Prefix Caching

The single highest-impact optimization across all blueprints:

| Model | Prefix Length | Speedup | Blueprint |
|---|---|---|---|
| Gemma 4 31B | 16K | **11.1x** | gemma4-hyperpod |
| Mistral SM4 119B | 16K | **5.23x** | mistral-small-4-hyperpod |
| Gemma 4 E4B | 16K | **3.9x** | gemma4-4b-hyperpod |
| Qwen3-Next 80B | 30K prefix + 2K suffix | **2.4x** (58% reduction) | qwen3-next |
| Kimi K2.5 | Multi-turn 20 rounds | **1.85x** | kimi-k2.5 |

Enable with: `--enable-prefix-caching` (vLLM) or RadixAttention (SGLang default).

### SGLang HiCache (MLA-Compatible CPU Offload)

```bash
python3 -m sglang.launch_server \
  --enable-hierarchical-cache \
  --hicache-size 100 \             # GB per TP rank, MUST exceed device KV pool
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --disable-cuda-graph \           # Required for hybrid attention
  ...
```

Results from GLM-5 on B200:
- Baseline: 909 tok/s peak at 64 concurrent
- HiCache: **2,602 tok/s** peak at 128 concurrent (2.86x improvement)
- Zero overhead at low concurrency (48 tok/s both configs at conc=1)

**Critical**: `--hicache-size` must exceed device KV pool. GLM-5 uses ~82 GB/rank KV; set to 100. Do NOT use `--hicache-ratio 2.0` (calculates 2x device pool = 165 GB x 8 = 1,325 GB total, causes OOM).

### LMCache Tiered Architecture

```
L0: GPU prefix cache (vLLM native)
L1: CPU DRAM (LMCACHE_LOCAL_CPU=True)
L2: Shared memory / Redis (cross-replica sharing)
L3: FSx Lustre (persistent, multi-node)
```

**Configuration (llm-d on HyperPod)**:
```yaml
# vLLM args
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# LMCache config
LMCACHE_LOCAL_CPU: "True"
LMCACHE_REMOTE_URL: "sagemaker-hyperpod://$(NODE_IP):9200"  # L2 via ai-toolkit
LMCACHE_REMOTE_SERDE: "cachegen"
```

**Known limitations**:
- LMCache 0.3.15 incompatible with NSA/MLA models (GLM-5, DeepSeek V3)
- GDS mode serializes requests under moderate pressure (synchronous in scheduler thread)
- CPU bounce path (`LMCACHE_USE_EXPERIMENTAL=False`) is async and often faster
- ABI mismatch with vLLM nightly: build LMCache from source at startup

### NVIDIA Dynamo KVBM

```
G1: GPU VRAM (fastest, limited)
G2: CPU DRAM (configured via DYN_KVBM_CPU_CACHE_SIZE)
G3: Disk/FSx (configured via DYN_KVBM_DISK_CACHE_PATH)
```

**FSx compatibility flags**:
```bash
DYN_KVBM_DISK_ZEROFILL_FALLBACK=true   # Lustre lacks fallocate
DYN_KVBM_DISK_DISABLE_O_DIRECT=true    # Lustre strict alignment
```

**Key finding**: Dynamo's in-memory prefix cache outperforms LMCache even without tiered offloading active. 1.82x speedup vs LMCache's 1.31x on API gateway workload (Kimi K2.5).

---

## 8. Routing and Load Balancing

### Simple Round-Robin (Bare Metal)

For multi-replica setups on a single node (e.g., 4x Devstral on g7e):
```python
# nginx or aiohttp proxy on port 9000 → vLLM ports 8000-8003
upstream vllm { server localhost:8000; server localhost:8001; ... }
```

### Gateway API + llm-d EPP (EKS Production)

**Components**:
1. **Envoy Gateway**: GatewayClass + Gateway (NLB)
2. **EPP v1.3.1**: Endpoint Picker Plugin for prefix-cache-aware routing
3. **InferencePool**: Defines backend pod selection

**EPP image**: Use `registry.k8s.io/gateway-api-inference-extension/epp:v1.3.1` (staging images return 403).

**InferencePool v1 API** (GA, not v1alpha2):
```yaml
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
spec:
  endpointPickerRef:           # NOT extensionRef
    name: epp
  targetPorts:
    - number: 8000             # NOT targetPortNumber
  selector:
    matchLabels:               # NOT flat selector
      app: vllm
```

**EPP configuration**:
- Flags are kebab-case: `--pool-name`, `--grpc-port`, `--secure-serving`
- `--config-file` or `--config-text` is mandatory
- RBAC: SA needs list/watch on pods, inferencepools (GA group), inferencemodelrewrites, inferenceobjectives, inferencepoolimports

**EnvoyExtensionPolicy for ext-proc**:
```yaml
apiVersion: gateway.envoyproxy.io/v1alpha1
kind: EnvoyExtensionPolicy
spec:
  extProc:
    - backendRef:
        name: epp
        port: 9002
      messageTimeout: 30s      # Default 200ms too short for LLM!
      failOpen: true            # Bypass EPP on timeout
```

**Known issues**:
- `--skip-crds` skips GatewayClass creation. Apply manually.
- EnvoyExtensionPolicy CRD not installed with `--skip-crds`. Install from `helm show crds`.
- EPP with empty scheduler config accepts gRPC but never responds. Needs proper scorer plugins.

### Ray Serve (Zero-SPOF)

Route NLB to worker proxies only (exclude head):
- Worker proxies take over instantly when head crashes (0% error rate, 0s downtime)
- Head crash recovery: ~3 min (stunnel + GCS restore from ElastiCache)
- GCS FT: `ray.io/ft-enabled: "true"` annotation + ElastiCache Redis backend

---

## 9. Benchmarking

### Standard Workload Suite

| ID | Workload | Parameters | What It Measures |
|---|---|---|---|
| W1 | Multi-Turn Chat | rounds (1/5/10) x conc (1/4/8) | Prefix reuse over turns |
| W2 | RAG / Long Doc | shared prefix (2K/5K/10K) | Cache warmup + hit rates |
| W3 | Agentic Tool Calling | multi-turn with pauses (0.5/2/5s) | Interleaved compute |
| W4 | Shared System Prompt | prefix (2K/8K/16K) | APC effectiveness |
| W5 | ShareGPT Conversations | QPS sweep (0.5-20) | Capacity ceiling |
| W6 | Long Context Scaling | input (1K-16K), fixed 512 out | Prefill scaling |

### Key Metrics

| Metric | Target | Notes |
|---|---|---|
| TTFT p50 | < 500ms | Prefill latency. Dominated by context length |
| TTFT p99 | < 2s | Tail latency under load |
| ITL p50 | < 30-50ms | Decode latency per token |
| Output tok/s | Workload-dependent | Throughput at target QPS |
| Error rate | 0% | Any errors indicate capacity issue |

### Benchmark Execution

**Server-side (recommended for clean measurements)**:
```bash
kubectl exec -it <pod> -- python3 -m vllm.entrypoints.openai.bench_serve \
  --model <served-name> \
  --tokenizer /mnt/nvme/models/<model> \
  --dataset-name sharegpt \
  --num-prompts 100 \
  --request-rate 4.0
```

**Client-side (measures full E2E including network)**:
```bash
kubectl port-forward svc/<service> 8000:8000 &
python3 benchmark.py --endpoint http://localhost:8000 --qps 4.0
```

Document both — they measure different things.

### Benchmark Results Summary (Peak Throughput)

| Model | Hardware | Engine | Peak tok/s | Sweet Spot QPS | Blueprint |
|---|---|---|---|---|---|
| GLM-5 FP8 | B200 x8 | SGLang HiCache | 2,602 | 128 conc | glm5-lmcache |
| GLM-5 FP8 | B200 x16 | vLLM 2-replica | 2,374 | 64 conc | glm5-llmd |
| Qwen3-Next FP8 | H200 x4 (TP4) | vLLM | 2,280 | QPS 8 | qwen3-next |
| Mistral SM4 FP8 | H100 x2 (TP2) | vLLM | 2,160 | QPS 8 | mistral-small-4 |
| Qwen3-Next FP8 | g7e x4 (TP4) | vLLM | 2,172 | QPS 8 | qwen3-next-g7e |
| Gemma 4 31B | H100 x2 (TP2) | vLLM | 1,188 | QPS 4 | gemma4-hyperpod |
| Nemotron 120B | B200 x8 (TP2x4) | vLLM | 1,081 | 64 conc | nemotron-super |
| Gemma 4 E4B | H100 x1 | vLLM | 5,158 | QPS 20 | gemma4-4b |
| Ministral 3B | g6e x1 | vLLM + LMCache | 61 | QPS 1 | ministral-3b |

---

## 10. Troubleshooting

### Common Issues and Fixes

**Symptom: Pod stuck in Pending**
```bash
kubectl describe pod <pod>
# Look for: Insufficient nvidia.com/gpu
```
Fix: Scale down other GPU deployments first:
```bash
kubectl scale deployment <other> --replicas=0
# Then scale back after new pod starts
```

**Symptom: GPU memory not released after pod deletion**
```bash
# Force-delete doesn't cleanly terminate GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
```

**Symptom: DiskPressure taint persists**
```bash
kubectl debug node/<name> --image=busybox -- \
  sh -c "chroot /host systemctl restart kubelet"
```

**Symptom: IMDS credentials fail in pod**
```bash
aws ec2 modify-instance-metadata-options \
  --instance-id <id> \
  --http-put-response-hop-limit 2
```

**Symptom: NCCL `Cuda failure 1 'invalid argument'`**
- Platform: Blackwell (sm_120) + PCIe topology
- Root cause: NCCL 2.25.1 shared memory bug
- Fix: Upgrade to NGC 25.03+ (NCCL >= 2.26.2)
- Note: vLLM inference unaffected (custom allreduce)

**Symptom: DeepGEMM JIT takes 15+ minutes**
- Expected on first startup for B200/Blackwell
- Cache location: `/root/.cache/vllm/deep_gemm/cache/`
- Persist cache on NVMe volume to speed subsequent starts
- Set readiness probe: `initialDelaySeconds: 900`

**Symptom: `model type 'gemma4' not recognized`**
- PyPI transformers lacks gemma4 support
- Fix: `pip install git+https://github.com/huggingface/transformers`

**Symptom: `AttributeError: 'NSATokenToKVPool' has no attribute 'k_buffer'`**
- LMCache incompatible with NSA/MLA models
- Fix: Use SGLang HiCache instead of LMCache

**Symptom: vLLM tool calls return raw text instead of JSON**
- Missing `--tool-call-parser <parser>` flag
- Mistral also needs `--enable-auto-tool-choice`

**Symptom: `broken pipe` / `ClientOSError: [Errno 32]`**
- Context overflow — vLLM rejected the request
- Fix: Context trimming in client, not connection fixes

**Symptom: FSx mount permission denied**
- FSx root is 755 owned by root:root, vLLM runs as uid 2000
- Fix: Init container to create subdirectory with 777 permissions

**Symptom: Terraform state lock**
- Check: `ps aux | grep terraform`
- Fix: `terraform force-unlock <lock-id>` or kill stale process

**Symptom: EKS bootstrap fails on AL2023**
- AL2023 uses `nodeadm`, not `bootstrap.sh`
- User data must be MIME multipart with `application/node.eks.aws` content type

**Symptom: AL2 on B200 — cuda error 802**
- AL2 kernel 5.10 lacks `ib_umad` module for Fabric Manager
- Fix: Use AL2023 AMI

**Symptom: Changing TP size causes scheduling deadlock**
- Old pod holds GPUs while new pod requests more
```bash
kubectl scale deployment <name> --replicas=0
# Wait for termination
terraform apply  # Change TP config
kubectl scale deployment <name> --replicas=1
```

**Symptom: Redis OOM on system nodes**
- System nodes (m5/m6i) have limited resources
- Fix: Add GPU node toleration to Redis pod — GPU nodes have abundant free CPU/RAM

**Symptom: `Unknown recipe` crash with FP8 on SGLang Blackwell**
- DeepGEMM broken with non-ue8m0 scale formats
- Fix: `--fp8-gemm-backend cutlass`

**Symptom: Root partition full on AL2 GPU AMI**
- 500 GB EBS but only 20 GB partitioned
- Fix: `growpart /dev/nvme0n1 1 && xfs_growfs /`

---

## 11. Cost Optimization

### Cost Per Million Output Tokens

| Config | Instance | TP | Cost/M Tokens | Relative |
|---|---|---|---|---|
| Blackwell g7e TP4 | g7e.24xlarge | 4 | **$1.06** | 1.0x (cheapest) |
| H200 capacity block dual TP4 | p5en.48xlarge | 4+4 | $2.54 | 2.4x |
| H200 on-demand TP1 | p5en.48xlarge | 1 | $4.88 | 4.6x |

### Cost Reduction Strategies

1. **Instance selection dominates**: g7e at $2.07/GPU/hr vs H200 at $7.91/GPU/hr. Per-GPU throughput is comparable (455 vs 451 tok/s for Qwen3-Next).

2. **TP4 over DP+EP**: TP=4 consistently beats DP+EP — 5.5x lower TTFT on half the GPUs, 9% higher throughput. Run 2x TP=4 replicas on 8-GPU node.

3. **Prefix caching is free performance**: 50-82% TTFT reduction for shared-prefix workloads. No cost, no downside.

4. **FP8 quantization**: Halves model memory, enabling smaller instances. Quality trade-off minimal for most models. Exception: GPTQ-Int4 produces garbage for some MoE models.

5. **Capacity blocks**: 50% discount vs on-demand for H200/B200. Plan 10-min termination gaps.

6. **Bare EC2 for benchmarking**: Skip EKS overhead when testing. Faster capacity acquisition.

### Memory Budget Planning

```
Total GPU VRAM
  - Model weights (check safetensors size / TP)
  - Activation memory (~2-4 GB)
  = Available for KV cache

KV cache tokens = Available_GB / (num_layers * head_dim * num_heads * 2 * dtype_bytes / TP)

Example: H200 141 GB, Qwen3-Next FP8 ~29 GB/GPU at TP4
  Available: ~104.5 GiB
  Capacity: ~34.6x concurrency at 262K context
```

---

## 12. Quick Reference Commands

### Serving Launch Templates

```bash
# vLLM on EKS (standard)
kubectl apply -f deployment.yaml

# vLLM on bare metal (nerdctl)
sudo nerdctl run -d --gpus all --ipc=host --network=host \
  -v /mnt/nvme:/mnt/nvme:ro \
  vllm/vllm-openai:<tag> \
  --model /mnt/nvme/models/<model> \
  --tensor-parallel-size <N> \
  --enable-prefix-caching \
  --tool-call-parser <parser>

# SGLang on bare metal
sudo nerdctl run -d --gpus all --ipc=host --network=host \
  -v /mnt/nvme:/mnt/nvme:ro \
  lmsysorg/sglang:<tag> \
  python3 -m sglang.launch_server \
    --model-path /mnt/nvme/models/<model> \
    --tp-size <N> --host 0.0.0.0 --port 30000
```

### Health Checks

```bash
# vLLM
curl http://localhost:8000/health
curl http://localhost:8000/v1/models

# SGLang
curl http://localhost:30000/health
curl http://localhost:30000/v1/models

# Quick chat test
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<name>","messages":[{"role":"user","content":"Say hello"}],"max_tokens":50}'
```

### GPU Monitoring

```bash
# Live GPU utilization
watch -n1 nvidia-smi

# Detailed metrics
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv -l 5

# Find orphaned GPU processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### Kubernetes Operations

```bash
# Check pod GPU allocation
kubectl describe node <node> | grep -A5 "Allocated resources"

# Force release GPUs (after stuck pod)
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

# Scaling for config changes
kubectl scale deployment <name> --replicas=0
# ... change config ...
kubectl scale deployment <name> --replicas=1

# Port forward for benchmarking
kubectl port-forward svc/<service> 8000:8000 &
```

### Model Staging

```bash
# Download from HuggingFace
huggingface-cli download <org>/<model> --local-dir /mnt/nvme/models/<model>

# Copy to FSx
aws s3 sync /mnt/nvme/models/<model> s3://<bucket>/models/<model>

# Verify model files
ls -lh /mnt/nvme/models/<model>/*.safetensors | wc -l
du -sh /mnt/nvme/models/<model>
```

### Pre-Flight Validation (One-Liner)

```bash
nvidia-smi && \
nvidia-smi topo -m && \
nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader && \
dmesg | grep -c "NVRM: Xid" && \
df -h /mnt/nvme && \
echo "Pre-flight OK"
```

---

## Appendix: Blueprint Index

| Blueprint | Model | Hardware | Engine | Status |
|---|---|---|---|---|
| devstral-sera | Devstral 24B | g7e.24xl | vLLM 0.15 | SVG training |
| ministral-3b | Ministral 3B | g6e.xl | vLLM + LMCache | Complete |
| qwen3-next | Qwen3-Next 80B | p5en.48xl | vLLM/SGLang | Complete |
| qwen3-next-custbench | Qwen3-Next 80B | p5en.48xl | vLLM | Complete |
| qwen3-next-g7e | Qwen3-Next 80B | g7e.24xl | vLLM | Complete |
| qwen3-next-sglang | Qwen3-Next 80B | g7e.24xl | SGLang | Complete |
| qwen3-32b-eks | Qwen3-32B | g6e.2xl | vLLM | Baseline |
| glm5-hyperpod | GLM-5 744B | p5e.48xl | SGLang | Template |
| glm5-lmcache | GLM-5 744B | p6-b200.48xl | SGLang HiCache | Complete |
| glm5-llmd | GLM-5 744B | p6-b200.48xl x2 | vLLM + llm-d | Partial EPP |
| nemotron-super | Nemotron 120B | p6-b200.48xl | vLLM | Complete |
| kimi-k2.5 | Kimi K2.5 1T | p5e.48xl | vLLM + Dynamo | Complete |
| ray-serve-ft | YOLOv8n | g5.xl x2 | Ray Serve | Complete |
| ray-serve-video | YOLO+MobileNet | g5.xl x2 | Ray Serve | Complete |
| dynamo-hyperpod | Qwen3-0.6B | g5.4xl | vLLM (Dynamo) | Smoke test |
| llmd-hyperpod | Qwen3-0.6B | g5.4xl | vLLM + llm-d | Complete |
| gemma4-hyperpod | Gemma 4 31B | p5.48xl (2xH100) | vLLM 0.19 | Complete |
| gemma4-4b-hyperpod | Gemma 4 E4B | g5.4xl / p5.48xl | vLLM | Complete |
| mistral-small-4-hyperpod | Mistral SM4 119B | p5.48xl (2xH100) | vLLM 0.19 | Complete |
