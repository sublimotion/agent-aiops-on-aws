# Nemotron-3-Ultra-550B-A55B-NVFP4 — Serving Benchmark (Blackwell B200/B300)

## Status: DRAFT (2026-06-05)

## Overview

Deploy NVIDIA **Nemotron-3-Ultra-550B-A55B** in its native **NVFP4** quantization (`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`) on Blackwell and benchmark serving performance against the published commercial endpoint. This is NVIDIA's flagship open-weights model (Artificial Analysis Intelligence Index **48** — current US open-weights leader, announced June 1 2026, full release June 4 2026), and the largest member of the hybrid **Mamba-2 + LatentMoE + Select-Attention** family we serve.

**Primary goal — beat the DeepInfra endpoint on single-stream output speed.** Artificial Analysis measured **~300 tok/s single-stream output** on the pre-release DeepInfra endpoint ("leading speed for its intelligence"; peer DeepSeek/Kimi models run 50–100 tok/s). **300 tok/s is the headline number to beat.** Secondary cost reference: DeepInfra retail price is **$0.50 / 1M input** and **$2.50 / 1M output** (262K context, $0.15/M cached input) — we report self-hosted blended $/1M against it, but the *speed* target drives the config choices (native MTP spec-decode is the lever).

**Secondary goals:**
- Characterize the hybrid Mamba-2/LatentMoE architecture across the full standard workload suite (chat, RAG, coding-agent, concurrency sweep, long-context to 1M).
- Validate native **Multi-Token Prediction (MTP)** speculative decoding — this model ships shared-weight MTP heads designed for stable drafting at long draft lengths.
- Dual-engine comparison: **vLLM** (primary) vs **SGLang**, with **TRT-LLM** as a stretch leg.

**Why this model:**
- 550B total / **55B active** (90% sparsity) — LatentMoE projects tokens into a smaller latent dimension for expert routing ("accuracy per byte").
- Native **NVFP4** pre-training recipe (quantization-aware) — most linear layers NVFP4; latent projections, MTP layers, QKV/attention, embeddings kept BF16/MXFP8 for stability.
- **Up to 1M context** (256K default cap) — interleaved Mamba-2 keeps long-context prefill tractable.
- Native MTP speculative decoding (5 draft tokens) — free acceleration with no separate draft model.
- Reasoning toggle (`enable_thinking`) + `reasoning_budget` trace control + `medium_effort` mode.
- OpenMDW-1.1 license (commercial OK).

**Why Blackwell B200/B300 spot:**
- NVFP4 weights ~335 GB (safetensors header) fit **4×B200 single-node** — NVIDIA's documented minimum unit and the only TP layout with a verbatim launch command (TP4). On an 8-GPU node this means **2× TP4 replicas**; TP8 is an *un-documented* layout we test as an ablation, not the validated baseline.
- NVFP4 is a **native Blackwell tensor-core format** — sm_100 (B200) / sm_103 (B300) execute it without dequant overhead. This is the model-hardware pairing NVFP4 was designed for.
- B200 spot ~$18/hr (us-east-2), B300 spot ~$15/hr (us-west-2) — far below the implied cost of the DeepInfra retail endpoint.

**Comparison targets:**

| Model | Active | Format | Weights | Instance | $/1M out (target) |
|---|---|---|---|---|---|
| **Nemotron-3-Ultra-550B** (this spec) | 55B | NVFP4 | ~335 GB | p6-b200 / b300 spot | **< $2.50 (beat DeepInfra)** |
| Nemotron-3-Super-120B (sibling) | 12B | FP8 | ~124 GB | p6-b200 | baseline (peak 1,449 tok/s @ c256) |
| Qwen3-235B-A22B | 22B | FP8 | ~235 GB | p6-b300 | $0.39/M @ c512 |
| Kimi K2.6 (1T) | 32B | INT4 | ~594 GB | p6-b300 | — |

---

## Components

### 1. Compute

**Primary — p6-b200.48xlarge (matches NVIDIA's tested recipe):**

- **Platform**: EKS on EC2 (spot)
- **Region/AZ**: us-east-2 (B200 spot abundant in **AZ1 and AZ2** per current capacity)
- **GPU Node**: p6-b200.48xlarge — 8× B200 (183 GB HBM3e each = **1,464 GB**), NVSwitch NVL5+
- **AMI**: `amazon-eks-node-al2023-x86_64-nvidia-1.32` (AL2023 required for Fabric Manager / `ib_umad` on NVL5+)
- **vCPUs**: 192 · **System RAM**: ~2 TB · **NVMe**: ~28 TB RAID0
- **System Nodes**: m6i.xlarge (cluster workloads)

**Long-context leg — p6-b300.48xlarge (256K→1M tiers only):**

- **Region/AZ**: us-west-2 **az2** (B300 spot)
- **GPU Node**: p6-b300.48xlarge — 8× B300 (275 GB HBM3e each = **2,200 GB**), NVLink 5 / NVSwitch, sm_103
- **Why B300 for 1M**: extra ~736 GB total VRAM is the difference between OOM and completion at the 512k/1m KV tiers. Run `rag-1m-context` 512k+1m tiers here; everything else on B200.
- **Image caveat**: B300 sm_103 requires **`-cu130` image tags** (standard tags are CUDA 12.x and will not load).

### 1a. GPU & NCCL Pre-Flight

Both B200 and B300 are NVSwitch topologies — mature NCCL on sm_100/sm_103.

| Check | Expected (B200) | Expected (B300) |
|---|---|---|
| GPU count | 8× B200 (sm_100) | 8× B300 (sm_103) |
| Topology | All 8 via NVSwitch | All 8 via NVSwitch |
| Driver / CUDA | 570.x+ / 12.6+ | 580.x+ / 13.0+ |
| ECC uncorrected | 0 | 0 |
| NCCL all_reduce bus BW | > 450 GB/s | > 450 GB/s |
| Thermals (idle) | < 85 °C | < 85 °C |

Use the `gpu-infra` MCP tools (`discover_cluster`, `check_gpu_health`, `run_nccl_test`) for Stage 4a.

### 2. Model

- **Model ID**: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`
- **Architecture**: Nemotron Hybrid LatentMoE — interleaved **Mamba-2** + **MoE** + select **Attention** layers, with shared-weight **MTP** heads.
  - 550B total / **55B active** per token (90% sparsity)
  - LatentMoE: tokens projected into a smaller latent dim for expert routing/compute
  - Mamba-2 layers for efficient sequential processing (sub-quadratic long-context prefill)
  - Select Attention (not every layer carries full attention)
- **Context Length**: up to **1M tokens** (default deployments cap at 262,144; 1M requires explicit override env var — see below)
- **Quantization**: **NVFP4** (weights/activations/gradients on most linear layers); BF16/MXFP8 on latent projections, MTP layers, QKV/attention, embeddings
- **Tensor types**: F32 · BF16 · F8_E4M3 · U8 · NVFP4
- **Disk footprint**: ~335 GB (NVFP4, safetensors header)
- **Tokenizer**: `AutoTokenizer.from_pretrained(...)`; reasoning trace closed with `</think>`
- **License**: OpenMDW-1.1 (commercial + non-commercial)
- **Deployment card**: run `mdc get nemotron-3-ultra --engine vllm` (and `--engine sglang`) before deploying; `mdc prs nemotron-3-ultra` for upstream changes. If no card exists, `mdc sync` then create one from the HF model card.

#### Memory Budget (NVFP4 on p6-b200.48xlarge, 1,464 GB total)

| Layout | Weights/GPU | KV + Mamba state/GPU | Notes |
|---|---|---|---|
| **TP4 × 2 replicas** *(documented)* | ~84 GB | ~91 GB | **NVIDIA's validated single-node unit** — the only layout with a verbatim launch command. Two independent replicas on the 8-GPU node; doubles request-routing parallelism. **Primary config.** |
| **TP8 × 1 replica** *(ablation)* | ~42 GB | ~133 GB | Un-documented for this model — no NVIDIA TP8 command exists. Max KV headroom per replica + lowest single-stream latency (fewer cross-GPU hops on decode). Test as the single-stream-speed ablation since 300 tok/s is the target. |

On B300 (2,200 GB): TP8 leaves ~233 GB/GPU for KV — the configuration for the 512k/1m context tiers (where single-replica KV headroom matters more than replica count).

#### Container Images

| Engine | B200 image | B300 image (sm_103) |
|---|---|---|
| vLLM | `vllm/vllm-openai:v0.22.0` | `vllm/vllm-openai:v0.22.0-cu130` |
| SGLang | `lmsysorg/sglang:v0.5.12.post1` | `lmsysorg/sglang:v0.5.12.post1-cu130` |
| TRT-LLM (stretch) | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc17` | same (verify cu130) |

> Pin to the **exact versions on the HF model card** — Nemotron-3-Ultra arch support, `nemotron_v3`/`nemotron_3` reasoning parsers, and `nemotron_h_mtp` spec-decode landed in these releases. Older builds will not recognize the architecture.

#### Serving Configuration — Track A: vLLM (primary, TP4)

**Verbatim from the HF model card** (4×B200 single node, NVFP4). Run **two replicas** on the 8-GPU node (GPUs 0–3 and 4–7) behind a round-robin proxy for the throughput configs:

```bash
docker run -d --name nemotron-ultra-vllm \
  --gpus '"device=0,1,2,3"' \
  --ipc=host --network=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $MODEL_CKPT:/model:ro \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e SAFETENSORS_FAST_GPU=1 \
  -e NVIDIA_TF32_OVERRIDE=1 \
  -e VLLM_LOGGING_LEVEL=INFO \
  vllm/vllm-openai:v0.22.0 \
  /model \
  --host 0.0.0.0 --port 8000 \
  --served-model-name nvidia/nemotron-3-ultra \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --mamba-ssm-cache-dtype float16 \
  --mamba-backend flashinfer \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --speculative-config '{"method": "nemotron_h_mtp", "num_speculative_tokens": 5}' \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
```

> **Every flag above is verbatim from the card — do not alter spellings.** Note `--reasoning-parser nemotron_v3` (with `v`) is vLLM-specific; SGLang uses `nemotron_3` (no `v`). `nemotron_h_mtp` MTP is **native** (no separate draft model) — measure acceptance rate (Stage 6). `--model-loader-extra-config` 96-thread load is what keeps the ~335 GB cold start tractable.

**Our additional tuning env vars (NOT on the card — validate each before trusting):**

```bash
# Blackwell FlashInfer paths — our prior B200/B300 tuning. Card does not set these;
# confirm they don't conflict with --mamba-backend flashinfer before benchmarking.
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export VLLM_FLASHINFER_MOE_BACKEND=latency      # min-latency (TRTLLM-Gen); =throughput (CUTLASS) at high conc

# AOT compile cache on NVMe — persist across restarts/spot reclaim
export VLLM_TORCH_COMPILE_CACHE=/mnt/nvme/vllm-cache
export TRITON_CACHE_DIR=/mnt/nvme/triton-cache

# 1M context (B300 long-context leg only) — card-documented override
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1   # then --max-model-len 1048576
```

For the **TP8 single-stream-speed ablation**, change `--tensor-parallel-size 4` → `8`, drop to one container with `--gpus all`, and keep all other flags. This is un-documented for the model — treat as experimental.

`--disable-log-requests` / `--enable-reasoning` are removed in current vLLM — not present in the card command either; do not add them (K2.6 + Qwen3-235B lessons).

#### Serving Configuration — Track B: SGLang (TP4)

**Verbatim from the HF model card** (4×B200 single node, NVFP4):

```bash
docker run -d --name nemotron-ultra-sglang \
  --gpus '"device=0,1,2,3"' \
  --cap-add SYS_NICE \
  --ipc=host --network=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $MODEL_CKPT:/model:ro \
  -e SAFETENSORS_FAST_GPU=1 \
  -e NVIDIA_TF32_OVERRIDE=1 \
  -e SGLANG_DISABLE_DEEP_GEMM=1 \
  lmsysorg/sglang:v0.5.12.post1 \
  python3 -m sglang.launch_server \
  --model-path /model \
  --host 0.0.0.0 --port 8000 \
  --served-model-name nvidia/nemotron-3-ultra \
  --tp-size 4 --ep-size 4 \
  --context-length 262144 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 32768 \
  --fp8-gemm-backend triton \
  --moe-runner-backend triton \
  --mamba-scheduler-strategy no_buffer \
  --disable-piecewise-cuda-graph \
  --reasoning-parser nemotron_3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --log-level info
```

> **`SGLANG_DISABLE_DEEP_GEMM=1` is card-mandated** — do NOT add `SGLANG_JIT_DEEPGEMM_FAST_WARMUP` (the Qwen3-235B warmup trick); DeepGEMM is explicitly disabled for this model, so the warmup flag is meaningless/contradictory here. SGLang's MTP is configured via the four `--speculative-*` EAGLE flags (not a single config object as in vLLM).
> 1M context (B300 leg): `export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` and `--context-length 1048576`.

> **SGLang tool + reasoning gotcha (model card):** with tools enabled, the request body **must** include
> `"chat_template_kwargs": {"enable_thinking": true, "force_nonempty_content": true}`
> or reasoning + tool-call parsing fails. Bake this into the benchmark client for SGLang.

#### Serving Configuration — Track C: TRT-LLM (stretch)

```bash
trtllm-serve $MODEL_CKPT --backend pytorch \
  --tp_size 8 --ep_size 8 --max_num_tokens 16384 \
  --reasoning_parser nano-v3 --tool_parser qwen3_coder \
  --chat_template $MODEL_CKPT/chat_template.jinja \
  --extra_llm_api_options extra-llm-api-config.yml --trust_remote_code
```

`extra-llm-api-config.yml` — max-throughput preset: MoE backend `CUTEDSL`, `ep_size 8`; min-latency: MoE backend `TRTLLM`, `ep_size 1`, MTP `max_draft_len: 5`. On rc16+ `max_draft_len` is authoritative (`num_nextn_predict_layers` deprecated). NGC `trtllm-runtime` images may 401 (nemotron-super lesson #9) — verify NGC org access before committing to this leg.

> **Reasoning control (all engines):** `extra_body={"chat_template_kwargs": {"enable_thinking": True|False}}`; `medium_effort: True` for reduced trace length; `reasoning_budget` caps trace tokens (model closes at next newline before budget, else abruptly at budget+500). Coding agents: add `force_nonempty_content: True`. Sampling: **temperature=1.0, top_p=0.95** (model-mandated — no greedy decoding).

### 3. Networking

- **VPC**: existing EKS VPC per region (us-east-2 for B200, us-west-2 for B300)
- **Access**: `kubectl port-forward` for benchmarks; `hostNetwork: true` for GPU pods
- **VPC Endpoints**: S3, ECR
- **NVSwitch**: intra-node TP8 collective communication

### 4. Storage

- **Model weights**: S3 → init container → NVMe hostPath (`/mnt/nvme/models/nemotron-3-ultra-nvfp4/`, ~335 GB)
- **NVMe**: RAID0 local SSD at `/mnt/nvme` — serving tier + compile caches
- **Ephemeral storage**: GPU pod needs **400 GB+** (weights + runtime + caches)
- **KV cache strategy**:

| Tier | B200 | B300 |
|---|---|---|
| GPU VRAM | prefix caching (attention layers only) | prefix caching |
| Mamba recurrent state | per-sequence SSM cache (float16) — **not** prefix-cacheable | same |

> **No HiCache / LMCache / CPU-offload** for this model. The Mamba-2 hybrid recurrent state is incompatible with all standard KV-transfer/offload connectors (nemotron-super lessons #1, #2, #17, #21 — same architecture family). Prefix caching only helps the attention layers; the Mamba state is not prefix-cacheable. Do not waste a config slot on HiCache here.

### 5. Monitoring

- **Prometheus**: scrape vLLM/SGLang `/metrics` (1s interval)
- **Key metrics**: `kv_cache_usage_perc`, prefix hit rate, `num_requests_running`, `num_preemptions_total`, **MTP/spec-decode acceptance rate**, TTFT/ITL histograms
- **GPU metrics**: `gpu-infra` `get_gpu_metrics` (DCGM via Prometheus) or `nvidia-smi dmon` DaemonSet

---

## Benchmark Design

Standardized runner (`benchmark-runner` skill / `standards/benchmark-commons`). Each run emits an **enriched artifact** (AIPerf output + deployment-context envelope) into `blueprints/nemotron-ultra/results/`. Workloads reference canonical cards by `catalog_id` — do not inline divergent params.

### Controlled Variables

| Parameter | Fixed value | Why |
|---|---|---|
| Model | Nemotron-3-Ultra-550B-A55B-NVFP4 | single model under test |
| Quantization | NVFP4 | native format |
| Temperature / top-p | 1.0 / 0.95 | model-mandated |
| Mamba SSM cache | float16 + stochastic rounding (philox=5) | model-mandated |
| KV cache dtype | fp8 | all configs |
| Reasoning | per-request `enable_thinking` | both states measured |

### Configurations

| Config ID | Engine | Layout | Spec-decode | Platform | Purpose |
|---|---|---|---|---|---|
| `vllm-tp4x2-mtp` | vLLM v0.22.0 | **TP4 ×2** | MTP (5 tok) | B200 | **Primary** — NVIDIA's documented unit; headline throughput + cost |
| `vllm-tp4-mtp` | vLLM v0.22.0 | TP4 ×1 | MTP (5 tok) | B200 | Single-replica baseline (4 idle GPUs) |
| `vllm-tp4-nomtp` | vLLM v0.22.0 | TP4 ×1 | off | B200 | **MTP gain ablation** — the 300 tok/s lever |
| `vllm-tp8-mtp` | vLLM v0.22.0 | TP8 ×1 *(undocumented)* | MTP (5 tok) | B200 | **Single-stream-speed ablation** — fewer decode hops → chase 300 tok/s |
| `sglang-tp4-eagle` | SGLang v0.5.12 | TP4 ×1 | EAGLE (5) | B200 | Engine shootout |
| `vllm-tp8-1m` | vLLM v0.22.0 | TP8 ×1 | MTP (5 tok) | **B300** | 512k/1m long-context tiers |
| `trtllm-tp4` *(stretch)* | TRT-LLM 1.3.0rc17 | TP4 | MTP (3–5) | B200 | Third-engine validation |

### Priority Tiers

```
P0 (must-have): Smoke + tool-call + reasoning toggle + MTP gate     ~1 hr
P1 (must-have): Standard workload suite on vllm-tp4x2-mtp          ~2.5 hrs
P2 (must-have): Concurrency sweep + engine/layout/MTP ablations     ~2.5 hrs
P3 (must-have): Cost analysis vs DeepInfra (the headline)           ~0.5 hr
P4 (should-have): 1M long-context on B300                           ~1.5 hrs
Total: ~8 hrs compute (B200) + ~1.5 hrs (B300)
```

### P0 — Smoke Test + Gate

| Step | Test | Pass criteria |
|---|---|---|
| 0a | Health | `/health` → 200, `/v1/chat/completions` responds |
| 0b | Basic inference | 1K in / 512 out, coherent output, no repetition |
| 0c | Reasoning ON | `enable_thinking=True` → `</think>`-terminated trace present |
| 0d | Reasoning OFF | `enable_thinking=False` → no trace |
| 0e | `medium_effort` | trace present but materially shorter than 0c |
| 0f | Tool-call (BFCL subset, 50) | `qwen3_coder` parser → valid `tool_calls` |
| 0g | Tool + reasoning | SGLang: `force_nonempty_content=True` — both parse together |
| 0h | **MTP acceptance** | spec-decode acceptance rate reported & > 0 (sanity that MTP is active) |

**Gate**: BFCL ≥ 75% on ≥1 engine AND MTP acceptance > 0 → proceed. Else STOP and diagnose.

### P1 — Standard Workload Suite (`vllm-tp4x2-mtp`, B200)

Canonical cards (run each, store enriched artifact):

| Card | `catalog_id` | Focus |
|---|---|---|
| Interactive chat | `chatbot-short` | TTFT/ITL latency floor |
| Multi-turn chat | `chatbot-long` | prefix reuse across turns |
| RAG long-context | `rag-long-context` | prefix cache hit (≤16K) |
| Coding agent | `coding-agent` | 12K system prompt + tool loop, prefix caching, KV retention |
| Production mix | `sharegpt-production-mix` | real trace distribution, blended in/out |
| Concurrency sweep | `concurrency-sweep` | SLO-max operating point (drives P2 + cost) |

Reasoning measured **on and off** for `chatbot-short`, `coding-agent`, and `sharegpt-production-mix` (the reasoning-overhead delta directly affects blended $/1M).

### P2 — Concurrency Sweep + Ablations

**Concurrency sweep** (power-of-2: 1, 4, 16, 32, 64, 128, 256; ramp until TTFT p99 > SLO or OOM), at 4K/1K shape, on:
- `vllm-tp4x2-mtp` (primary — documented unit)
- `vllm-tp8-mtp` (layout ablation — does single-replica TP8 KV headroom + fewer decode hops beat 2× TP4 routing parallelism?)
- `sglang-tp4-eagle` (engine ablation)

**Single-stream speed (the 300 tok/s chase)**: at **c=1**, compare `vllm-tp4-mtp`, `vllm-tp8-mtp`, `vllm-tp4-nomtp`, `sglang-tp4-eagle`. This is where the DeepInfra 300 tok/s number lives. TP8 (fewer cross-GPU hops per decode step) + native MTP is our best shot at matching/beating it.

**MTP ablation**: `vllm-tp4-mtp` vs `vllm-tp4-nomtp` at c=1, 16, 64 — quantify spec-decode throughput gain and acceptance-rate decay under batching. (Hypothesis: MTP shines at low concurrency / single-stream; gain compresses as the batch fills the GPU. Single-stream is exactly the regime the 300 tok/s target measures.)

**Reasoning-under-load**: sweep with `enable_thinking=True` at c=16, 64, 256 — thinking-token overhead on aggregate throughput.

Find SLO-max: the concurrency where TTFT p99 < 2s (4K) and ITL p99 < 50ms hold. This is the operating point for P3 cost math.

### P3 — Speed + Cost Analysis (the headline: beat DeepInfra's 300 tok/s)

**Primary verdict — single-stream output speed.** Report c=1 output tok/s for every config from the P2 single-stream sweep against the **300 tok/s** DeepInfra reference. State the winning config and whether self-hosted Blackwell + native MTP matches/beats it.

Then, at SLO-max from P2, compute blended $/1M tokens and compare to DeepInfra retail price.

```
$/1M output  = (instance_$ / hr) / (output_tok/s at SLO-max) × (1e6 / 3600)
$/1M input   = (instance_$ / hr) / (input_tok/s prefill-bound) × (1e6 / 3600)
blended      = weight by production-mix in:out ratio (from sharegpt-production-mix)
```

| Source | $/1M in | $/1M out | Context |
|---|---|---|---|
| **DeepInfra (target to beat)** | $0.50 | $2.50 | 262K |
| DeepInfra cached input | $0.15 | — | 262K |
| **Self-hosted B200 spot** (this spec) | TBD | TBD | 262K |
| **Self-hosted B300 spot** (long-ctx leg) | TBD | TBD | up to 1M |

B200 spot ≈ $18/hr; B300 spot ≈ $15/hr. **Win condition:** self-hosted blended $/1M-out < $2.50 at an operating point that also satisfies the latency SLOs. Report the break-even concurrency (the output tok/s at which self-hosted crosses the DeepInfra retail line) and the utilization (% of day at SLO-max) required to realize it.

> Caveat to state plainly in the report: DeepInfra retail price ≠ DeepInfra cost. We are comparing self-hosted cost to a competitor's *price*; the honest framing is "at what utilization does owning the box beat renting tokens."

### P4 — 1M Long-Context (`vllm-tp8-1m`, B300, us-west-2 az2)

Run `rag-1m-context` (`catalog_id: rag-1m-context`) — context tiers 64k → 128k → 256k → 512k → 1m, shared-prefix RAG pattern. Per the card's cross-check policy, **also** report `rag-long-context` (16K) and `chatbot-long` (32K) so the 16K→1M curve is complete (a single 1M datapoint is unfalsifiable).

Required per-tier metrics (card validation gates):
- `ttft_cold_vs_warm_ratio` — if < 2×, flag prefix caching broken (expected partial: Mamba state isn't prefix-cached, only attention layers are — **predict ratio between 1× and 2×, and report it as an architecture finding, not a bug**).
- `kv_cache_usage_perc_per_tier` — the efficiency curve. For a Mamba-2 hybrid, attention-KV growth should be sub-linear vs a pure-transformer baseline; this is the long-context efficiency story.
- `per_tier_completion` — record the highest tier that completes without OOM. OOM at 1m on B300 means "does not support 1M in this config" regardless of card claim.
- `degenerate_output_tokens` — sample 5 outputs/tier; reject if any token repeats >5× (spec-decode + extreme context repetition guardrail — our `feedback_synthetic_specdec_repetition` lesson).

---

## Verification Criteria

### Stage 0c — Serving-Config Resolver (fail-closed)
- [ ] `python3 standards/serving-commons/resolver/validate-serving-config.py --sidecar blueprints/nemotron-ultra/benchmark.yaml --corpus-root .` exits 0
- [ ] Sidecar declares `model.moe_intermediate_size` so any FP8/NVFP4 MoE TP-divisibility rule verifies rather than WARNs
- [ ] Every `prior-failure:*` finding (esp. nemotron-super Mamba-hybrid entries) reviewed and noted in the deployment log

### Stage 4a — GPU Health (via `gpu-infra` MCP)
- [ ] `check_gpu_health`: all GPUs ECC enabled, 0 uncorrectable, no pending row remaps, < 85 °C idle
- [ ] `discover_cluster`: 8× B200 (sm_100) / 8× B300 (sm_103) all visible via NVSwitch
- [ ] `run_nccl_test`: all_reduce bus BW > 450 GB/s
- [ ] `dmesg | grep "NVRM: Xid"` empty

### Stage 5 — Serving Stack
- [ ] `/health` → 200 on vLLM and SGLang
- [ ] Test completion succeeds; no `CUDA out of memory` in logs (NVFP4 TP8)
- [ ] Reasoning toggle (`enable_thinking` True/False) behaves correctly
- [ ] **MTP active**: spec-decode acceptance rate present in `/metrics` and > 0
- [ ] Cold start < 8 min with AOT compile cache warm (record cold-cache time separately)

### Stage 6 — Benchmark

| Metric | Target | Phase |
|---|---|---|
| **Single-stream output (c=1)** | **≥ 300 tok/s — beat DeepInfra** *(stretch; ≥ 150 tok/s = competitive)* | **P2/P3** |
| BFCL tool-call accuracy | ≥ 75% (≥1 engine) | P0 |
| MTP acceptance rate (single-stream) | report; expect > 0.5 | P0/P2 |
| Peak throughput (SLO-max) | ≥ 1,500 tok/s (≥ Nemotron-Super peak) | P2 |
| MTP throughput gain (c=1) | ≥ 1.3× vs `nomtp` | P2 |
| TTFT p99 @ 4K, SLO-max | < 2,000 ms | P2 |
| ITL p99 (streaming) | < 50 ms | P1/P2 |
| Error rate | < 0.1% all levels | P1/P2 |
| **Blended $/1M out @ SLO-max** | **< $2.50 (beat DeepInfra)** | **P3** |
| Max completed context tier | ≥ 512k on B300 | P4 |
| `ttft_cold_vs_warm_ratio` @ 1m | reported (Mamba-hybrid finding) | P4 |

Engine-internal (Prometheus): KV cache utilization, running requests, prefix hit rate, MTP acceptance — captured in the `extensions` block of each enriched artifact.

### Stage 7 — Readiness Audit
- [ ] All criteria above checked and recorded
- [ ] No unresolved `lessons.md` entries severity ≥ HIGH
- [ ] Enriched artifacts written for all P0–P3 configs (+ P4 on B300)
- [ ] Cost comparison table vs DeepInfra completed with break-even concurrency + required utilization
- [ ] Deployment-card recommendations followed or explicitly overridden with justification

---

## Non-Requirements

- BF16 deployment (NVFP4 is native and smaller — only relevant if NVFP4 quality is questioned)
- HiCache / LMCache / CPU KV offload (Mamba-2 hybrid state incompatible — see §4)
- Disaggregated prefill/decode (NIXL KV transfer fails on Mamba hybrid — nemotron-super #17/#21; revisit only if Dynamo ships native hybrid-KV transfer)
- Multi-node distributed inference (single 8-GPU node; PP=2 multi-node is a future stretch only if NVFP4 weights ever exceed single-node VRAM)
- Vision/multimodal (text-only)
- Production autoscaling / multi-region HA

## Security Requirements

- EKS RBAC, least-privilege IRSA; private subnets, no public endpoints on GPU pods
- Model weights on local NVMe (ephemeral); S3 source bucket KMS-encrypted
- GPU node group scaled to 0 after each benchmark session

## Cost Considerations

| Resource | Spot rate | Notes |
|---|---|---|
| p6-b200.48xlarge (us-east-2) | ~$18/hr | primary, P0–P3 (~8 hrs) |
| p6-b300.48xlarge (us-west-2 az2) | ~$15/hr | P4 long-context (~1.5 hrs) |
| Model download from S3 | ~$3 | ~335 GB NVFP4 |
| **Total benchmark session** | **~$170–185** | GPU dominates |

Checkpoint enriched artifacts to S3 between P-tiers (spot reclaim only loses the in-flight run; ~335 GB re-stage from S3→NVMe ≈ 10 min).

## Known Limitations

1. **Mamba-2 hybrid blocks KV offload/disagg** — recurrent state incompatible with HiCache/LMCache/NIXL. Prefix caching helps attention layers only (nemotron-super #1/#2/#17/#21, same family).
2. **NVFP4 is brand-new for this model** — verify the pinned vLLM/SGLang versions actually load NVFP4 + the `nemotron_v3`/`nemotron_3` parser + `nemotron_h_mtp` before committing benchmark hours. Older builds silently lack arch support.
3. **Mamba SSM cache must use stochastic rounding** (`--enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5`) — dropping it risks numerical drift on the recurrent state.
4. **Temperature 1.0 mandatory** — no greedy decoding; affects determinism of benchmark reruns (seed where possible).
5. **SGLang tool+reasoning requires `force_nonempty_content: true`** in `chat_template_kwargs` or parsing fails (model card).
6. **1M context needs explicit override env var** (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` / `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`) — default cap is 262K. Likely B300-only due to KV footprint.
7. **B300 requires `-cu130` image tags** (sm_103); standard CUDA-12.x tags will not load (K2.6/Qwen3-235B lesson).
8. **MTP acceptance decays under batching** — single-stream gain won't hold at high concurrency; report acceptance per concurrency level.
9. **Spec-decode + extreme context can collapse to repetition** — apply the degenerate-output guardrail at every 1M tier (`feedback_synthetic_specdec_repetition`).
10. **NGC TRT-LLM images may 401** (nemotron-super #9) — TRT-LLM leg is stretch-only; verify org access first.
11. **Spot reclaim** — 2-min warning; checkpoint between P-tiers.
12. **AL2023 NVIDIA AMI Lustre client gap** — if FSx is used instead of NVMe, install `lustre-client` via privileged init container (nemotron-super #10). This spec uses S3→NVMe to avoid it.

Check `mdc prs nemotron-3-ultra` for upstream PRs (arch support, parser fixes, MTP) before deploying.

---

## Deployment Sequence

```
1. Pre-session
   ├── mdc get nemotron-3-ultra --engine vllm  (+ sglang); mdc prs nemotron-3-ultra
   ├── Pre-stage NVFP4 weights to S3 (~335 GB)
   ├── Verify container images load NVFP4 + nemotron_v3 parser + nemotron_h_mtp (smoke on 1 node)
   └── Confirm B200 spot (us-east-2 AZ1/AZ2) + B300 spot (us-west-2 az2)
2. EKS GPU node (self-managed spot, AL2023 NVIDIA AMI, NVMe RAID0)
3. Model staging: init container S3 → /mnt/nvme/models/nemotron-3-ultra-nvfp4/
4. GPU pre-flight (gpu-infra MCP: discover_cluster, check_gpu_health, run_nccl_test)
5. Track A vLLM: P0 → P1 → P2 → P3, checkpoint artifacts to S3
6. Track B SGLang: P0 + engine-ablation sweep (P2)
7. P4 long-context: launch B300 (us-west-2 az2), run rag-1m-context curve
8. Teardown: sync results, write lessons.md, scale node groups to 0
```

> **Note**: Operational artifacts (lessons, benchmark results, deployment notes) belong in
> `blueprints/nemotron-ultra/` (lessons.md, results/, benchmark.yaml), not in this spec.
