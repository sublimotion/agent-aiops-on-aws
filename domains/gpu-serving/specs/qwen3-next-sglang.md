# Coding Agent Feasibility Benchmark — Qwen3-Next vs Devstral Small 2

## Status: DRAFT (2026-03-03)

## Parent Spec

See [`qwen3-next.md`](./qwen3-next.md) for full model details and workload definitions.

This spec benchmarks **two candidate models** on g7e.24xlarge for coding agent feasibility:

**Track S — Qwen3-Next-80B on SGLang** (4× GPU, TP=4):
1. SGLang as the only framework supporting hybrid attention KV offloading
2. HiCache tiered KV cache (GPU → CPU → NVMe) for hybrid DeltaNet+GQA architecture
3. Tool-use quality (BFCL) to determine coding agent viability
4. Swarm concurrency scaling to validate agent economics

**Track D — Devstral Small 2 (24B) on vLLM** (1× GPU, single-GPU baseline):
1. SWE-bench Verified 68.0% — strongest open-weight coding model in its class
2. Standard GQA — no framework compatibility issues, KV offloading works everywhere
3. Single-GPU serving (~$2/hr) vs Qwen3-Next's 4-GPU requirement (~$17/hr)
4. Head-to-head comparison on identical BFCL, swarm, and functional eval tasks

---

## Why This Test Matters

From the Coding Agent Feasibility Analysis:
- Qwen3-Next-80B is **blocked on vLLM, Dynamo, and LMCache** for KV offloading (HMA incompatibility)
- SGLang is the **only framework** with native `MambaRadixCache` handling heterogeneous per-layer state
- HiCache L2 KV offload is working for hybrid models (PR #19663: cache hit 0.29 → 0.81)
- Qwen3-Next published BFCL-v3 = 70.3 (below our ≥75 gate — risk flag)
- Devstral Small 2 has **68.0% SWE-bench Verified** on 24B params (no published BFCL — this test fills the gap)
- Devstral fits on **1 GPU** (~$2/hr) vs Qwen3-Next on 4 GPUs (~$17/hr)
- **Cost to test both: ~$130** (7-8 hours on g7e.24xlarge at $16.57/hr)

---

## Compute

Two instance options, switchable via `gpu_instance_types` variable:

| Property | g7e.24xlarge (default) | g7e.48xlarge (scale-up) |
|----------|----------------------|------------------------|
| GPUs | 4× RTX PRO Server 6000 | 8× RTX PRO Server 6000 |
| VRAM per GPU | 96 GB GDDR7 | 96 GB GDDR7 |
| Total VRAM | 384 GB | 768 GB |
| Interconnect | PCIe | NVLink |
| CPU / RAM | 96 vCPU / 384 GB | 192 vCPU / 1.5 TB |
| NVMe | Instance store | Instance store |
| On-demand cost | **$16.57/hr** | $110.30/hr |
| Region / AZ | us-east-2a | us-east-2a |
| TP configs | TP=4 (single replica) | TP=8 or 2× TP=4 replicas |

**Start on g7e.24xlarge** for cost-efficient validation (S0-S4). Scale to g7e.48xlarge for 2-replica benchmarks (still PCIe — RTX PRO 6000 has no NVLink).

### GDS / EFA Considerations

**g7e instances support EFA but not GDS.** This affects HiCache L3 storage backends:
- L3 NVMe (file backend): Uses standard kernel I/O. Still fast (~3-5 GB/s read on NVMe RAID0), but not GPU-direct.
- L3 FSx: Would require GDS for zero-copy GPU→FSx transfers. Without GDS, FSx-backed L3 falls back to CPU bounce buffer — viable but slower.
- **For GDS + EFA KV offloading to FSx**, use p5en.48xlarge ($71.47/hr on-demand, ~$41.61/hr capacity block). This is a production-path follow-up, not needed for initial validation.

The g7e HiCache benchmark validates the **architectural path** (SGLang + MambaRadixCache + tiered KV). The p5en benchmark would validate the **production I/O path** (GDS + EFA + FSx L3/L4).

---

## VRAM Budget

### g7e.24xlarge (TP=4, FP8)

| Component | Per GPU | Total |
|-----------|---------|-------|
| Model weights (FP8, TP=4) | ~20 GB | ~80 GB |
| KV cache (0.90 mem_fraction) | ~66 GB | ~264 GB |
| Activations + overhead | ~6 GB | ~24 GB |
| **Available for KV** | **~66 GB/GPU** | **~264 GB** |

With HiCache L2, an additional ~300 GB of CPU RAM is available for cold KV cache eviction.

**KV headroom: ~264 GB GPU + ~300 GB CPU = ~564 GB total** — supporting 100-300+ concurrent 32K-context agents.

### g7e.48xlarge (TP=8, FP8)

| Component | Per GPU | Total |
|-----------|---------|-------|
| Model weights (FP8, TP=8) | ~10 GB | ~80 GB |
| KV cache (0.90 mem_fraction) | ~76 GB | ~608 GB |
| Activations + overhead | ~6 GB | ~48 GB |
| **Available for KV** | **~76 GB/GPU** | **~608 GB** |

With HiCache L2, an additional ~1.4 TB of CPU RAM is available.

**KV headroom: ~608 GB GPU + ~1.4 TB CPU = ~2 TB total** — massive swarm capacity.

### g7e.48xlarge (2× TP=4 replicas)

Each replica has the same budget as g7e.24xlarge above (~264 GB KV each). Combined: 2× throughput at 1.71x efficiency (sub-linear due to shared NVMe bandwidth, validated on p5en custbench).

---

## Models

### Track S: Qwen3-Next-80B-A3B

| Property | Value |
|----------|-------|
| Model | Qwen3-Next-80B-A3B (FP8) |
| Architecture | Hybrid MoE: 75% Gated DeltaNet (linear attn) + 25% GQA |
| Total / Active params | 80B / 3B |
| FP8 weight size | ~80 GB |
| Quantization | FP8 (official `Qwen3-Coder-Next-FP8`) |
| Context | 131,072 native (start with 65,536 if VRAM tight) |
| Tool-call parser | `qwen3_coder` (via `--tool-call-parser`) |
| Published BFCL-v3 | 70.3 |
| Published SWE-bench | Not reported |

### Track D: Devstral Small 2 (24B)

| Property | Value |
|----------|-------|
| Model | Devstral-Small-2-24B-Instruct-2512 (FP8) |
| Architecture | Dense transformer, standard GQA (32Q/8KV heads) |
| Total / Active params | 24B / 24B (dense — all params active) |
| FP8 weight size | ~26 GB (fits on 1× RTX PRO 6000) |
| Quantization | FP8 native (official checkpoint ships in FP8) |
| Context | 256K native (YaRN, factor=48) |
| Tool-call parser | `mistral` (via `--tool-call-parser mistral --enable-auto-tool-choice`) |
| Published SWE-bench Verified | **68.0%** (OpenHands scaffold) |
| Published BFCL | Not reported (this test fills the gap) |
| License | Apache 2.0 |

### Head-to-Head Summary

| Dimension | Qwen3-Next-80B | Devstral Small 2 |
|-----------|----------------|-------------------|
| GPUs required | 4× (TP=4) | **1×** |
| Serving cost | ~$17/hr | **~$2/hr** (or $17/hr for fair TP=4 comparison) |
| Framework | SGLang only (hybrid attention) | **vLLM or SGLang** (standard GQA) |
| KV offloading | SGLang HiCache only | **All frameworks** |
| SWE-bench Verified | Unknown | **68.0%** |
| BFCL-v3 | 70.3 | Unknown (tested here) |
| Architecture risk | High (hybrid DeltaNet+MoE) | **Low** (standard dense) |

---

## Serving Engines

### Track S: SGLang (Qwen3-Next)

SGLang is the only framework with:
- Native `MambaRadixCache` for per-layer heterogeneous state management
- `HybridLinearAttnBackend` dispatching per-layer to GDN or standard attention
- HiCache cascading eviction (GPU → CPU → NVMe) with `MambaRadixCache` support (PR #19663)
- Native GDN kernels (Triton, CuteDSL, FlashInfer) — no external `flash-linear-attention` dependency

**Minimum SGLang version**: v0.5.9+ (Qwen3-Next day-0 support, HiCache improvements)

### Track D: vLLM (Devstral Small 2)

vLLM is the natural choice for Devstral:
- Standard GQA architecture — no exotic attention handling needed
- Native FP8 checkpoint support (official weights ship in FP8)
- Automatic Prefix Caching (APC) for agentic workloads with shared system prompts
- Chunked prefill for long-context interleaving
- `--tool-call-parser mistral --enable-auto-tool-choice` for tool calling

**Known caveat**: vLLM has streaming tool-call parsing bugs with the Mistral parser (issues #23180, #33916). Use non-streaming mode for tool-calling evaluation (S2/S5). Streaming is fine for throughput benchmarks.

**vLLM configs**:
- `configs/devstral-vllm-baseline.sh` — 1× GPU (port 8000), cost-efficiency baseline
- `configs/devstral-vllm-4gpu.sh` — 4× GPU TP=4 (port 8000), apples-to-apples throughput comparison

---

## Benchmark Phases

### Track S: Qwen3-Next on SGLang

### Phase S0: Smoke Test (30 min)

**Goal**: SGLang loads Qwen3-Next FP8 on g7e.24xl, generates correct output.

| Config | Details |
|--------|---------|
| Engine | SGLang TP=4, FP8, `--context-length 65536` |
| Test | 10 requests, 1024 input / 512 output, random data |
| Pass criteria | Model loads, output is coherent, no CUDA errors |

If 65536 works, retest with `--context-length 131072`.

### Phase S1: Throughput Baseline (1 hr)

**Goal**: Establish SGLang throughput on g7e.24xl for comparison with vLLM on p5en.

| Config | QPS | Requests | Input/Output |
|--------|-----|----------|-------------|
| S1a | 0.5 | 100 | 1024 / 512 |
| S1b | 2.0 | 200 | 1024 / 512 |
| S1c | 4.0 | 400 | 1024 / 512 |
| S1d | 8.0 | 400 | 1024 / 512 |

**Metrics**: TTFT (p50/p95/p99), ITL (p50/p99), throughput (output tok/s), GPU utilization.

**Comparison target**: vLLM on p5en achieves 230 tok/s at TP=4. g7e.24xl target: ≥150 tok/s (sufficient for swarm use).

### Phase S2: BFCL Tool-Use Evaluation (1 hr)

**Goal**: Determine if Qwen3-Next can reliably call tools — the gate for coding agent viability.

| Test | Details |
|------|---------|
| Dataset | BFCL-V4 subset (100-200 multi-turn tool-use scenarios) |
| Endpoint | SGLang with `--tool-call-parser qwen3_coder` |
| Metrics | First-call success rate, multi-turn completion rate, structured output accuracy |

**Decision gates**:
- BFCL < 70: **Stop** — model is not viable for coding agents
- BFCL 70-75: **Proceed with caution** — viable for swarms (retry at machine speed), not for interactive
- BFCL ≥ 75: **Proceed** — viable for both swarm and interactive coding agent use
- BFCL ≥ 80: **Strong** — competitive with Claude Sonnet for tool orchestration

### Phase S3: HiCache Tiered KV Cache (1.5 hr)

**Goal**: Validate SGLang HiCache on hybrid attention model, measure concurrency expansion.

| Config | HiCache Tier | Details |
|--------|-------------|---------|
| S3a | L1 only (baseline) | No HiCache — GPU KV cache only |
| S3b | L1 + L2 (CPU) | `--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through` |
| S3c | L1 + L2 + L3 (NVMe) | S3b + `--hicache-storage-backend file --hicache-storage-backend-extra-config '{"path": "/mnt/nvme/kv-cache"}'` |

**Test per config**: 500 requests with shared 8K system prompt + 128-token unique suffix. Measure:
- Cache hit rate (target: >80% with shared prefix)
- Max concurrent requests before TTFT > 30s
- KV cache utilization (GPU tier vs CPU tier vs NVMe tier)
- TTFT distribution at 50, 100, 200 concurrent requests

**Key comparison**: S3a vs S3b shows the value of CPU offloading. S3b vs S3c shows marginal value of NVMe tier.

**Known risks**:
- PR #19663 (HiCache + MambaRadixCache) may not be merged yet — test with latest SGLang nightly
- SSM recurrent state offloading to L3 is TODO in the PR — only KV cache from attention layers offloads
- `--disable-cuda-graph` may be required (CUDA graph + HiCache conflict for hybrid models)
- MoE + HiCache kernel bug on SXM (#19737) — g7e uses PCIe, may not be affected

### Phase S4: Swarm Concurrency Simulation (1 hr)

**Goal**: Validate how many concurrent coding agents g7e.24xl can sustain.

Simulate agentic workload: each "agent" sends a request, waits for response, idles during tool execution (simulated 5-30s delay), then sends next request.

| Test | Concurrent Agents | Tool Exec Delay | Duration |
|------|------------------|----------------|----------|
| S4a | 4 agents | 15s avg | 15 min |
| S4b | 8 agents | 15s avg | 15 min |
| S4c | 16 agents | 15s avg | 15 min |
| S4d | 32 agents | 15s avg | 15 min |
| S4e | 64 agents | 15s avg | 15 min (if S4d GPU util < 90%) |

**Metrics per test**:
- GPU utilization (target: >80% at saturation)
- Per-agent TTFT (should stay <10s even at high concurrency with prefix caching)
- Aggregate throughput (tok/s across all agents)
- Memory pressure (GPU KV %, CPU KV % with HiCache)
- Zero failures (no OOM, no timeouts)

**Expected outcome**: GPU saturation at 3-8 agents (from duty cycle analysis). KV headroom supports 70-140+ agents on GPU alone, 200-400+ with HiCache L2. The limiting factor should be GPU compute, not memory.

### Phase S5: Functional Coding Evaluation — SERA-Inspired (30 min)

**Goal**: Close the loop — validate the model can actually fix real bugs through multi-turn tool use, not just call tools with correct syntax.

Inspired by SERA (Soft-Verified Efficient Repository Agents, Dettmers et al., Allen AI). Uses two evaluation modes:

1. **Direct Task Completion**: Model receives a bug report + codebase. It must read files, identify the bug, write a fix, and run tests — all via real tool execution against a workspace filesystem. Scored by test pass rate.

2. **SVG Reproduction** (Soft Verified Generation): The model's fix from phase 1 is converted into a PR description. A fresh agent run attempts to reproduce the fix from only the PR description. Scored by line-level patch recall (SERA metric).

| Test | Task | Bug Type |
|------|------|----------|
| S5a | parse_date returning None | Stub implementation |
| S5b | Off-by-one in pagination | Logic error |
| S5c | Missing JWT expiration check | Security bug |
| S5d | Thread-unsafe counter | Concurrency bug |
| S5e | CSV encoding crash on Unicode | Encoding error |

**Metrics**:
- Task completion rate (model produces a fix that passes tests)
- Test pass rate (all tests green, not just the target test)
- SVG reproduction rate (can reproduce its own fix from PR description)
- SVG line-level recall (SERA metric: patch overlap between runs)
- Average turns per task (efficiency)

**Decision gates**:
- Test pass rate ≥ 80%: **STRONG** — model reliably fixes bugs through multi-turn tool use
- Test pass rate 60-80%: **VIABLE** — retry strategy recommended for production
- Test pass rate 40-60%: **MARGINAL** — swarm-only viability (retry at machine speed)
- Test pass rate < 40%: **NOT VIABLE** — model cannot reliably fix bugs

**Why this matters**: S2 (BFCL) tests tool-call syntax. S4 tests throughput under load. S5 tests whether the full stack actually works for coding — the model reads code, reasons about bugs, writes correct patches, and verifies them. This is the final feasibility gate.

### Track D: Devstral Small 2 on vLLM

Mirrors the S-track phases for head-to-head comparison. All evaluation scripts (bfcl-eval.py, swarm-simulator.py, functional-eval.py) are shared — only the endpoint and model name change.

#### Phase D0: Smoke Test (15 min)

Same as S0 but against vLLM with `--tool-call-parser mistral`.

#### Phase D1: Throughput Baseline (1.5 hr)

Two sub-tracks:
- **D1a-d**: Single GPU — measures cost-efficiency (1× RTX PRO 6000, ~$2/hr equivalent)
- **D1e-h**: TP=4 — apples-to-apples throughput comparison with S1 (same 4 GPUs)

| Config | GPUs | QPS | Requests | Input/Output |
|--------|------|-----|----------|-------------|
| D1a | 1 | 0.5 | 100 | 1024 / 512 |
| D1b | 1 | 2.0 | 200 | 1024 / 512 |
| D1c | 1 | 4.0 | 400 | 1024 / 512 |
| D1d | 1 | 8.0 | 400 | 1024 / 512 |
| D1e | 4 | 0.5 | 100 | 1024 / 512 |
| D1f | 4 | 2.0 | 200 | 1024 / 512 |
| D1g | 4 | 4.0 | 400 | 1024 / 512 |
| D1h | 4 | 8.0 | 400 | 1024 / 512 |

**Key comparison**: D1a-d vs S1a-d answers "is 1-GPU Devstral competitive with 4-GPU Qwen3-Next?"

#### Phase D2: BFCL Tool-Use Evaluation (30 min)

Same evaluation as S2. Fills the missing BFCL data for Devstral (not published by Mistral).

#### Phase D3: Prefix Caching Effectiveness (30 min)

Devstral uses standard GQA — no HiCache needed. vLLM's Automatic Prefix Caching (APC) handles prefix sharing natively. Same shared-prefix workload and concurrency sweep as S3a for direct comparison.

#### Phase D4: Swarm Concurrency Simulation (1 hr)

Same as S4 but on single GPU. Tests 4/8/16/32 concurrent agents (capped at 32 — single GPU saturates earlier). Key question: how many coding agents can $2/hr sustain?

#### Phase D5: Functional Coding Evaluation (30 min)

Same SERA-inspired tasks as S5. Devstral has 68% SWE-bench Verified — this phase validates whether that translates to our specific bug-fixing tasks with real tool execution.

---

## Infrastructure

### Reusable Components

- **S3 model bucket**: Same as parent blueprint (qwen3-next-fp8 already staged)
- **FSx Lustre**: SCRATCH_2, 1200 GiB (model staging only, cheaper than PERSISTENT_2)
- **Container image**: `lmsysorg/sglang:latest` or nightly build with PR #19663

### New Components

- **g7e.24xlarge instance**: On-demand (no capacity blocks for g7e)
- **EKS cluster**: Reuse or create minimal cluster in us-east-2a
- **NVMe RAID0**: Instance store, detect and format at boot

### Storage Flow

```
S3 → FSx SCRATCH_2 (1.2 TB) → NVMe RAID0 (~3.8 TB) → GPU VRAM
                                    ↑
                              HiCache L3 (kv-cache dir)
```

---

## Container Image

SGLang nightly with hybrid model HiCache support:

```
lmsysorg/sglang:v0.5.9-cu131    # If PR #19663 is merged
lmsysorg/sglang:nightly-cu131   # If testing pre-merge
```

**Fallback**: Build from source with PR #19663 cherry-picked:
```dockerfile
FROM lmsysorg/sglang:v0.5.9-cu131
RUN pip install --upgrade sglang  # Or build from PR branch
```

Blackwell (sm_100) requires cu131+. Verify CUDA compute capability in smoke test.

### Track D: Devstral Small 2

```
vllm/vllm-openai:latest           # Official vLLM OpenAI-compatible image
mistralai/Devstral-Small-2-24B-Instruct-2512  # HuggingFace model (ships in FP8)
```

Model download: `huggingface-cli download mistralai/Devstral-Small-2-24B-Instruct-2512 --local-dir /mnt/nvme/models/devstral-small-2-fp8`

---

## Success Criteria

| # | Criterion | Threshold |
|---|-----------|-----------|
| 1 | S0 smoke test passes | Model loads, coherent output on g7e.24xl |
| 2 | S1 throughput | ≥ 150 tok/s at TP=4 (≥65% of vLLM p5en baseline) |
| 3 | S2 BFCL tool-use | ≥ 75 (viable for coding agents) |
| 4 | S3 HiCache cache hit | ≥ 80% with shared prefix workload |
| 5 | S3 HiCache concurrency | ≥ 2x max concurrent requests vs S3a baseline |
| 6 | S4 swarm GPU saturation | ≥ 80% GPU utilization at 8+ concurrent agents |
| 7 | S4 zero failures | No OOM, no timeouts at peak tested concurrency |
| 8 | Cost validation | Effective $/task ≤ $3.00 at swarm scale |
| 9 | S5 functional test pass rate | ≥ 60% (viable with retry strategy) |
| 10 | S5 SVG reproduction rate | ≥ 50% (model is consistent in its fixes) |
| 11 | D1 Devstral 1-GPU throughput | ≥ 100 tok/s (sufficient for single-agent use) |
| 12 | D2 Devstral BFCL | ≥ 75 (fills missing benchmark data) |
| 13 | D5 Devstral functional pass rate | ≥ 60% (expect higher given 68% SWE-bench) |
| 14 | Cost comparison | Winner identified: $/task at swarm scale for both models |

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PR #19663 not merged | Medium | Blocks S3 (HiCache) | Test S0-S2 first; use nightly or cherry-pick |
| cu131 image lacks sm_100 | Medium | Blocks all | Build from source with sm_100 target |
| FP8 on Blackwell (sm_100) | Low | Model loads wrong | Fallback to BF16 (2x weight size, still fits TP=4) |
| HiCache + MoE kernel crash | Medium | Blocks S3 | Use `--disable-cuda-graph` and `write_through` policy |
| 65536 context insufficient for BFCL | Low | S2 limited | BFCL tasks are short-context; 65536 is sufficient |
| PCIe bottleneck (no NVLink) | Low | Lower throughput | TP=4 on PCIe is standard; 80B FP8 has low communication |
| SGLang BFCL tool parser | Medium | Tool calls malformed | Test with `--tool-call-parser qwen3_coder` first, fall back to manual parsing |

---

## Non-Requirements

- **vLLM comparison**: Already benchmarked on p5en (230 tok/s). This spec is SGLang-only.
- **MTP / speculative decoding**: Out of scope for initial validation. Can add in follow-up.
- **Multi-replica**: Single TP=4 replica only. 2x replica topology is a production follow-up.
- **Fine-tuning**: Out of scope. This validates base model capability.
- **Production deployment**: This is a benchmark spec, not production infrastructure.

---

## Estimated Cost

### g7e.24xlarge path (recommended start)

| Phase | Duration | Instance Cost | Total |
|-------|----------|-------------|-------|
| S0: Smoke | 0.5 hr | $16.57/hr | ~$8 |
| S1: Throughput | 1.0 hr | $16.57/hr | ~$17 |
| S2: BFCL | 1.0 hr | $16.57/hr | ~$17 |
| S3: HiCache | 1.5 hr | $16.57/hr | ~$25 |
| S4: Swarm | 1.0 hr | $16.57/hr | ~$17 |
| S5: Functional | 0.5 hr | $16.57/hr | ~$8 |
| D0-D5: Devstral track | 4.0 hr | $16.57/hr | ~$66 |
| Setup + teardown | 1.0 hr | $16.57/hr | ~$17 |
| **Total (g7e.24xl, both tracks)** | **10.5 hrs** | | **~$174** |

### g7e.48xlarge add-on (if g7e.24xl results are promising)

| Phase | Duration | Instance Cost | Total |
|-------|----------|-------------|-------|
| S1-48xl: TP=8 throughput | 1.0 hr | $110.30/hr | ~$110 |
| S3-48xl: HiCache TP=8 | 1.0 hr | $110.30/hr | ~$110 |
| S5: 2-replica TP=4 | 1.0 hr | $110.30/hr | ~$110 |
| Setup + teardown | 0.5 hr | $110.30/hr | ~$55 |
| **Total (g7e.48xl)** | **3.5 hrs** | | **~$385** |

### Combined total: ~$485 (both instances)

---

## References

- [[Coding-Agent-Feasibility-Analysis]] — Parent analysis this test validates
- [`qwen3-next.md`](./qwen3-next.md) — Full model spec
- [`qwen3-next-g7e.md`](./qwen3-next-g7e.md) — g7e compute reference
- SGLang PR #19663 — HiCache + MambaRadixCache support
- SGLang PR #17373 — RadixLinearAttention abstraction
- SGLang PR #18489 — Qwen3.5 model support
- SERA (Dettmers et al., Allen AI) — Soft-Verified Efficient Repository Agents, SVG pipeline for coding agent evaluation
