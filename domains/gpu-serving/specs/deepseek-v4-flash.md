# DeepSeek V4 Flash Serving Benchmark Spec

## Status: COMPLETE (2026-05-19)

Full T0 + R2 (prefix + long-context) + R3 (MTP) sweep landed on B300 in us-west-2b.
Results: `domains/gpu-serving/blueprints/deepseek-v4-flash/results/FINAL-report.md`
Lessons: `domains/gpu-serving/blueprints/deepseek-v4-flash/lessons.md`

## Overview

Standalone serving benchmark for **DeepSeek V4 Flash** — a 284B / 13B-active MoE with novel **CSA + HCA hybrid attention**, native FP4+FP8 mixed precision, and 1M context. Goal: characterize the throughput-latency-cost frontier and quantify the upstream "10% KV cache, 27% FLOPs" efficiency claims under our standard W0-W6 + vllm bench workload sweep.

This spec was written **after** the Gemma 4 spec was deferred for upstream stability reasons, and uses the same template (W0 sharegpt cross-check, optimization tiers, MTP repetition guardrail, upstream-issue-tracker verification rule).

### Optimization Objective

```
Primary:   Quantify V4 Flash's efficiency story on H200 / B200 / B300 vs other MoE
           in repo (Kimi K2.6, GLM-5 744B, Qwen3-235B-B300)
Secondary: Validate 79% SWE-bench claim by feeding outputs into the
           autoresearch/blueprints/verification-primitives-swebench/ harness
Metric:    TTFT p50/p99, ITL p50/p99, output tok/s at 1M context, $/M tokens,
           KV cache footprint vs DeepSeek V3.2 baseline
```

### Why this model

- **Frontier-tier capability on open weights**: SWE-bench Verified 79%, LiveCodeBench 91.6, GPQA Diamond 88.1
- **Industry-leading active-params/capability ratio**: 13B active for ~Kimi-K2.6-class scores (which uses 32B active of 1T)
- **Novel attention architecture** (CSA + HCA) — not yet characterized in this repo's coverage; bug surface is fresh code, not Gemma-4-style dead-end paths
- **1M native context** with FP4 expert weights — uniquely positioned for long-context production workloads
- **API price reference**: $0.14/M input, $0.28/M output (Artificial Analysis) — sets the $/M-token target for self-hosted deployment

---

## Components

### 1. Compute

The **CSA + HCA architecture's 10% KV cache footprint** is the headline efficiency claim — to measure it cleanly we need a hardware target where we can hit the long-context / high-concurrency regime that exposes the win. Three options:

| Hardware | Fit | Recommendation |
|---|---|---|
| **p6-b300.48xlarge** (B300 NVSwitch, 8× 275GB) | ✅ **Best** | Native FP4 compute on weights; 2.2 TB total HBM enables 1M context at high concurrency; matches `qwen3-235b-b300` benchmark conditions for cross-comparison |
| **p6-b200.48xlarge** (B200 NVSwitch, 8× 183GB) | ✅ **Second** | Native FP4; 1.46 TB HBM; matches `kimi-k2.6` and `glm5-lmcache` baselines |
| **p5e.48xlarge** (H200 NVSwitch, 8× 141GB) | ⚠️ Workable | FP4 dequantizes to FP8/BF16 at matmul (no SM 9.0 native FP4 cores). Memory win still real (~10% KV) but compute win lost. Useful only as a "cost-of-no-Blackwell" data point. |

**Primary**: p6-b300.48xlarge (B300). Falls back to p6-b200 if B300 capacity unavailable.

**Avoid** (per upstream issue tracker, 2026-05-19):
- **g7e (RTX PRO 6000 / SM_120 PCIe)**: vLLM #40821, #40802, #42432 — fails to load
- **A100 / A800 (SM_80)**: vLLM #40851 — feature request open, no support
- **L20**: vLLM #40903 — fails to load
- **AMD MI300X / MI325X / MI350X**: vLLM #41962, #41963, #42876, #25118 — multiple OPEN bugs; "v0.21.0 release notes claim AMD support" but actually broken

### 2. Model

- **Primary**: `deepseek-ai/DeepSeek-V4-Flash` — 284B total / 13B active, MoE
- **Fallback FP8**: `sgl-project/DeepSeek-V4-Flash-FP8` (SGLang #24111 — pre-converted FP8 checkpoint, OPEN/expected)
- **Architecture**: CSA (Compressed Sparse Attention) + HCA (Heavily Compressed Attention) hybrid + Manifold-Constrained Hyper-Connections (mHC) + MoE
- **Native precision**: FP4 expert weights + FP8 elsewhere (mixed)
- **Context**: 1,048,576 tokens native; "Think Max" recommends ≥384K window
- **Reasoning modes**: Non-think / Think High / Think Max — must benchmark all three
- **License**: MIT
- **Tool calling**: Card silent. vLLM #42878 reports DSML tool call format ("fake-stream arguments"); needs parser investigation in T0.

### 3. Serving Image

- **vLLM**: `vllm/vllm-openai:v0.21.0+` minimum — but verify PR #42701 (MTP `post_mix` regression) is patched. Use nightly if not.
- **SGLang**: per the SGLang #23602 V4 roadmap — image not yet pinned. Check `lmsysorg/sglang` for `dsv4` or `v4flash` tag.
- **transformers ≥ 4.57** required (PR #42806 OPEN); but #42741 reports breakage with 4.57+. Pin transformers carefully, validate on T0.

> **Stability check (synced 2026-05-19 from upstream issue trackers — model deployment card not yet authoritative)**:
>
> **vLLM merged fixes** (verify in image):
> - PR #41694 (merged 2026-05-10): PP support for V4
> - PR #42930 (merged 2026-05-18): MTP after ROCm mHC integration
> - PR #41006 / #41171: base model + decoder layer integration
>
> **CRITICAL OPEN issues that affect this spec**:
> - **vLLM #42948**: Prefix-cache **0% hit rate** on re-sent requests on V4 Flash (hybrid groups lose first-block cache keys). **Directly invalidates T1 numbers** — gate the tier on this issue's status.
> - **vLLM #43093**: Two crashes preventing V4 Flash + OffloadingConnector — affects KV-offload tiers if added.
> - **vLLM #41331**: Garbled output with CUDA Graph + concurrent identical inputs → must run repetition-detection guardrail (per Gemma 4 RCA).
> - **vLLM #40987 / #41604 / #40952**: MTP / decode crashes; #42701 specifically: v0.21.0 release missing #42320 → MTP fails with `TypeError: missing required positional argument: post_mix`. **MTP is gated until verified working.**
> - **vLLM #42769**: `UnboundLocalError: 'name_mapped'` when expert mapping has no match — affects load step.
> - **vLLM #42878 / #40801**: Tool-calling parser instability (DSML format leaks, fake-stream args). T-W3 (tool calling) gate must verify parser before measuring.
> - **vLLM #42741**: Fails to load with transformers ≥ 4.57 — pin to 4.56.x or apply PR #42806 manually.
> - **SGLang #25662**: "Precision issues encountered in DeepSeek V4" (updated 2026-05-19, OPEN) — quality regression; **run quality smoke test before trusting any throughput number**.
> - **SGLang #23896**: TP4+DP4+DeepEP throughput **lower** than plain TP4 on H20-3e — challenges intuition that DP+EP wins; matches our practitioner-guide finding (TP > DP+EP).
> - **SGLang #23657**: No SM120 fallback for Lightning Indexer (compressed attention) — confirms g7e exclusion.
> - **SGLang #25704**: V4-Pro NVFP4 on B200 produces NaN (TP=8) or garbage (DP+DeepEP); only EAGLE works. **Watch this path for the V4-Flash B200 NVFP4 case** — may be the same bug class.
> - **SGLang #23769**: V4-Pro-FP8 official serving config OOMs during CUDA Graph capture on B200.
>
> **HF model card**: comprehensive on architecture and capability scores, **silent on tool calling, multimodal, vLLM/SGLang serving flags**. Use upstream tracker as source of truth.

### 4. Networking

Inherits standard B200/B300 capacity-block patterns from `kimi-k2.6` and `qwen3-235b-b300` blueprints. Private subnets, EFA, VPC endpoints. AL2023 AMI required (per memory: AL2 lacks `ib_umad`).

---

## Optimization Tiers

Same lever priority as `gemma4-benchmark.md`, with V4-Flash-specific adjustments. Each tier adds **one** lever on top of the prior config.

| Tier | Lever | Configs | Workloads | Why this tier |
|------|-------|---------|-----------|---------------|
| **T0** | Baseline (no opts) | A | W0, W5, W6, P1v-a, P1v-b | Establishes floor. **Must use Non-think mode** for clean comparison. |
| **T1** | Prefix caching | B vs A | W0, W2, W4, P1v-c | **HIGH RISK — vLLM #42948 reports 0% hit rate on V4 Flash hybrid groups.** Run anyway to characterize current state. |
| **T2** | Reasoning mode sweep | C, D vs A | W0, W1, P1v-a | V4 Flash is reasoning-tuned; Non-think / Think High / Think Max have very different decode footprints. **No analog in Gemma 4 spec — V4-Flash-specific.** |
| **T3** | Long context (CSA+HCA win) | E vs A | W6 extended, W0 long | **The headline efficiency claim**: 10% KV vs V3.2, 27% FLOPs at 1M. Sweep context 64K → 256K → 1M. |
| **T4** | KV cache FP8 | F vs A | W5, P1v-a | Halves KV memory on top of CSA+HCA's 10× compression — compounding wins. |
| **T5** | Higher batch ceiling | G vs F | W5 high-QPS, P1v-a | Probes throughput knee at the now-massive effective KV pool. |
| **T6** | MTP / EAGLE speculative | H vs G | W1, W5 (with guardrail) | **GATED on vLLM #42701 / SGLang #24650 / #25704 — high failure risk.** |
| **T7** (opt) | Pipeline parallelism | I vs G | W0, W5 | Tests PR #41694's PP support (merged 2026-05-10). Skip unless TP=8 hits OOM in T3 long-context. |
| **T8** (opt) | DP+EP (SGLang) | J vs G | W5, P1v-a | Verifies SGLang #23896 finding (TP > DP+EP) on H200/B200 instead of H20-3e. |

### Skipped Tiers (with rationale)

| Lever | Why skipped |
|-------|-------------|
| **CPU offload** | V4 Flash + OffloadingConnector crashes (#43093). Reconsider when fixed. |
| **LMCache** | Not yet validated for CSA+HCA layout; would replicate the Kimi K2.6 / GLM-5 NSA-MLA situation. |
| **NVFP4 on B200** | SGLang #25704 reports V4-Pro NaN/garbage on B200 NVFP4. V4-Flash may share the bug class — gate at T0 with a precision smoke test before promoting. |
| **AMD ROCm** | Multiple OPEN bugs (#41962, #41963, #42876, #25118). |
| **A100 / A800 / L20 / g7e** | Hardware unsupported per upstream. |

---

## Benchmark Configs

### Config A — T0: Baseline (no opts, Non-think)

```yaml
extra_vllm_args:
  - --tensor-parallel-size
  - "8"
  - --max-model-len
  - "32768"
  - --gpu-memory-utilization
  - "0.90"
  - --no-enable-prefix-caching
  - --trust-remote-code
  # transformers pin: 4.56.x until #42741 / #42806 resolved
```

Non-think mode (no special flag — default). Single GPU per attention head; TP=8 because 284B total weights with FP4+FP8 mix sit at ~150 GB equivalent — fits on 8× B200 with KV headroom for 1M context.

### Config B — T1: + Prefix Caching

Config A plus `--enable-prefix-caching`. **Expected to fail** per vLLM #42948 — record hit rate from `vllm:prefix_cache_hit_rate`. If 0% on shared-prefix workload, document and move on.

### Config C — T2a: Think High mode

Config A but configure reasoning mode = `think_high` via API parameter or env (DeepSeek V4-specific; check serving image docs). Sample longer outputs (>4K).

### Config D — T2b: Think Max mode

Config C with `think_max` mode. Increase `--max-model-len` to 524288 (≥384K per card recommendation). Output budget 8K-16K to capture full reasoning trace.

### Config E — T3: Long-Context Sweep (driven by `rag-1m-context` card)

Config A with `--max-model-len 1048576`. Workload definition is canonical: see `proposals/001-common-benchmark-artifact/workloads/rag-1m-context.yaml`. Sweeps {64K, 128K, 256K, 512K, 1M} shared prefixes with cold/warm measurement protocol.

Goal: the **CSA+HCA efficiency curve**. Measure `vllm:gpu_cache_usage_perc` and per-token decode time at each tier; compare to DeepSeek's published V3.2 baseline. The card's `validation.kv_growth_superlinear` rule auto-flags violations.

### Config F — T4: + KV Cache FP8

Config E plus `--kv-cache-dtype fp8`. Halves the (already-compressed) KV pool. Run quality smoke test (per SGLang #25662 precision-issue caveat).

### Config G — T5: + Higher Batch Ceiling

Config F plus `--max-num-seqs 256` and `--max-num-batched-tokens 65536`.

### Config H — T6: + MTP / EAGLE

Config G plus `--speculative-config '{"method":"deepseek_v4_mtp","num_speculative_tokens":2}'` OR EAGLE v3 (PR #42413 OPEN). **Pre-run gate**:
1. Verify vLLM PR #42701 / #42320 fix is in the image — abort if `post_mix` error appears.
2. Sample 20 outputs; reject if any token repeats >5× consecutively (Gemma 4 RCA guardrail).
3. Measure draft acceptance rate; if <30%, record and skip rest of T6.

### Config I — T7 (optional): Pipeline Parallelism

Config G with `--pipeline-parallel-size 2 --tensor-parallel-size 4` (PR #41694). Skip unless T3 long-context hits OOM with TP=8.

### Config J — T8 (optional): DP+EP on SGLang

SGLang server with `--dp-size 4 --ep-size 4 --tp-size 2`. Tests #23896's H20-3e finding on B200/B300.

---

## Benchmark Workload

Same workload card catalog as `gemma4-benchmark.md` — references `proposals/001-common-benchmark-artifact/workloads/*.yaml` by `catalog_id`.

| catalog_id | Use case | Cards-driven tier |
|------------|---------|-------------------|
| `chatbot-short` | Latency-sensitive interactive | T0 sanity |
| `chatbot-long` | 32K input prefill | T3 long-context entry |
| `qps-sweep` | SLO-max QPS finder | T0, T4, T5 |
| `rag-long-context` | Shared 16K prefix | T1 prefix caching, mid-context anchor |
| **`rag-1m-context`** | **Shared prefix sweep 64K → 1M** | **T3 — the architectural validation tier** |
| `coding-agent` | Tool calls, longer outputs | T2 reasoning, T6 MTP |
| `sharegpt-production-mix` | **Real conversation distribution** | **All tiers — production cross-check** |

### W0: ShareGPT Production-Mix Cross-Check (mandatory all tiers)

Same `sharegpt-production-mix` workload card and rules as Gemma 4 spec. Synthetic-vs-W0 deltas >25% → flag synthetic as misleading; W0 is the headline number.

### W6 (extended) — `rag-1m-context` workload card

The canonical 1M context test case. **References `proposals/001-common-benchmark-artifact/workloads/rag-1m-context.yaml`** — do not inline tier definitions or override parameters here.

**Tier sweep** (per the card): 64K, 128K, 256K, 512K, 1M shared-prefix prompts with 256-token unique question + 1024-token output. Cold-vs-warm split (first 2 requests as warmup, p50 reported on warm) — first request pays the full prefill cost, rest must hit prefix cache to be useful.

**Why this tier matters most for V4 Flash**: this is the **only** tier where CSA+HCA's "10% KV, 27% FLOPs vs V3.2" claim is even falsifiable. Below 64K, the architectural compression is masked by per-request overhead. The card's `validation.kv_growth_superlinear` rule explicitly checks the sub-linear claim — `kv @ 1M / kv @ 100K ≤ 12x`. Failing this rule is itself a publishable finding.

**Pre-emptive caveats from upstream tracker**:
- vLLM **#42948**: 0% prefix cache hit on hybrid groups → expect `ttft_cold_vs_warm_ratio < 2x` (reliability flag fires); document and proceed.
- SGLang **#24153**: SWARadixCache assertion under V4 long-context workload — gate SGLang version.
- SGLang **#23842**: V4 prefill may not capture piecewise CUDA graphs — affects cold prefill timing reproducibility.
- vLLM **#41331**: Garbled output with CUDA Graph + concurrent identical inputs → run the card's `degenerate_output_tokens` validation (5 outputs per tier, no >5× repetition).

**Vendor-claim validation handoff**: per the card's `cross_check.vendor_claim_validation` policy, after the sweep completes, compute the ratios `kv @ 1M / kv @ 100K` and `flops_per_token @ 1M / flops_per_token @ 100K` and compare directly to DeepSeek's published "10% / 27%" numbers in `lessons.md`.

### vllm bench phases (P1v)

| Phase | Sweep | Workload |
|-------|-------|----------|
| **P1v-a** | QPS {0.5, 1.0, 2.0, 4.0, 8.0, 16.0} | random 4K input / 1K output, 100 prompts |
| **P1v-b** | Context {1K, 4K, 16K, 64K, 256K, 1M} | random, 50 prompts at 1.0 QPS |
| **P1v-c** | Shared prefix {16K, 64K, 256K} | `generated-shared-prefix`, 50 prompts at 1.0 QPS |
| **P1v-d** | Reasoning mode | Non-think / Think High / Think Max at QPS 1.0 |

---

## Metrics

Same client + server-side metric set as `gemma4-benchmark.md`. V4-Flash-specific additions:

| Metric | Source | Why for V4 Flash |
|--------|--------|------------------|
| `vllm:gpu_cache_usage_perc` at each context length | gauge | **Validates the 10% KV claim**. Compare curves across context sweep. |
| Per-token decode time at 1M context | timing | **Validates the 27% FLOPs claim**. Compare to V3.2 published baseline. |
| Tokens-per-second per active param | derived | V4 Flash 13B active vs Kimi 32B active vs Qwen3-235B-A22B — efficiency frontier. |
| `$/M tokens` self-hosted | derived | Compare to API: $0.14 in / $0.28 out. |

### Targets

| Metric | Target | Phase |
|--------|--------|-------|
| TTFT p50 at 32K | < 800ms | P1v-b |
| TTFT p99 at 1M | < 30s | P1v-b |
| ITL p50 (Non-think) | < 30ms | P1v-a |
| ITL p50 (Think Max) | < 50ms | P1v-d |
| Output tok/s (single-stream Non-think) | ≥ 96 (match API) | P1v-a |
| QPS at SLO (Non-think) | ≥ 8.0 | P1v-a |
| KV @ 1M / KV @ 100K ratio | ≤ 12× | P1v-b (validates sub-linear claim) |
| W0 prefix cache hit rate | ≥ 5% (or 0% expected per #42948) | W0 |
| MTP draft acceptance | ≥ 30% (or skip) | T6 |
| Error rate | 0% | All |
| Tool-call accuracy (BFCL) | ≥ 70% | T0 with tool parser bake-off |

---

## Tooling & Standards Compliance

Identical to `gemma4-benchmark.md`. Sources `.claude/skills/benchmark-runner/scripts/benchmark-helpers.sh`, scaffold from `run-benchmarks.sh.tmpl`, in-cluster bench-runner pod, Prometheus pre/post per run, 30 warmup, 3 reps median-of-medians, 60s cooldown.

Per-tier execution rules from the skill apply with no modifications.

### Result handoff

```
run-benchmarks.sh
   └→ results/session-YYYYMMDD/*.json + pre/post_*_metrics.txt
       └→ benchmark-analyst agent → results/benchmark-report.md
           └→ visual-explainer skill → results/benchmark-visual-YYYYMMDD.html
               └→ reports/benchmark-results.md (cross-blueprint comparison row)
```

After all tiers complete, also feed Config G's outputs (Non-think + Think Max) into `domains/autoresearch/blueprints/verification-primitives-swebench/` to validate the **79% SWE-bench claim** independently. This is a unique cross-domain handoff for V4 Flash; flag it in the success criteria.

---

## RCA Inheritance from Gemma 4 Spec

This spec inherits the **synthetic + speculative-decoding repetition guardrail** from `gemma4-benchmark.md`:

- T6 (MTP) pre-run gate samples 20 outputs and rejects on token repetition >5× consecutive
- W0 sharegpt-production-mix is mandatory on every tier
- Synthetic-vs-W0 deltas >25% flag synthetic as misleading

Additional V4-Flash-specific RCA inputs (per upstream tracker):

| vLLM/SGLang issue | What it means for our benchmarks |
|---|---|
| **vLLM #41331**: Garbled output with CUDA Graph + concurrent identical inputs | Run output-validation guardrail under concurrency, not just for spec-decode |
| **vLLM #42948**: 0% prefix cache hit on V4 Flash hybrid groups | Treat T1 as documentation, not optimization |
| **SGLang #25662**: Precision issues in V4 | Mandatory quality smoke test on T0 (5-prompt deterministic check vs reference outputs) |
| **vLLM #42878 / #40801**: Tool-calling parser DSML format leaks | T0 tool-call gate must verify parser before measuring W3 |

---

## Hardware Decision (you said it would answer itself)

**Recommendation: p6-b300.48xlarge (B300).** The CSA + HCA efficiency story is the headline; B300 is the only target that gives us:

1. **Native FP4 compute** on weights (matches the model's design point)
2. **2.2 TB total HBM** (8× 275GB) — comfortably handles 1M context at the high concurrency we need to expose the KV-cache compression win
3. **Direct cross-comparison** with `qwen3-235b-b300` blueprint on identical hardware
4. **NVSwitch** — required because TP=8 across the FP4+FP8 MoE has high collective traffic; PCIe topology is a non-starter

**Fallback: p6-b200.48xlarge (B200).** Loses some HBM but keeps native FP4 + NVSwitch. Cross-comparison anchor becomes `kimi-k2.6` and `glm5-lmcache` instead.

**H200 (p5e) is not recommended.** Without native FP4 compute we measure only the memory side of the efficiency claim, missing the headline. Useful only as a separate "what V4 Flash looks like without Blackwell" data point — defer to a follow-up spec.

---

## Success Criteria

1. All target metrics in the Metrics table met or characterized as not met with root-cause citation
2. CSA+HCA long-context efficiency claim (10% KV / 27% FLOPs vs V3.2) **measured directly** at 256K and 1M context, not just stated
3. **Every tier reports both synthetic and W0 sharegpt numbers side-by-side**; deltas >25% flagged
4. Reasoning mode comparison (Non-think / Think High / Think Max) produces a Pareto curve in `reports/benchmark-results.md`
5. Cross-model row added: V4 Flash 13B-active alongside Kimi K2.6 32B-active, Qwen3-235B-A22B, GLM-5 ~40B-active — efficiency frontier table
6. SWE-bench 79% claim validated (or refuted) via `verification-primitives-swebench` cross-domain handoff
7. Tool calling: parser identified, BFCL ≥ 70% verified before any T6 (tool-using) MTP measurement
8. T6 outputs sampled and validated (no degenerate token repetition); if degenerate, synthetic MTP throughput is discarded and only W0 is reported
9. Lessons captured to `blueprints/deepseek-v4-flash/lessons.md` with severity ratings; HIGH-severity lessons fed to compound-learner for steering elevation

---

## Non-Requirements

- DeepSeek V4 **Pro** (862B) — separate spec; needs B200/B300 multi-node
- DeepSeek V4 **base** (non-chat) — eval-only, not a serving target
- Multimodal (V4 Flash is text-only)
- ROCm / AMD — multiple OPEN bugs; defer
- A100 / Ampere / SM_80 — feature request OPEN, no support (#40851)
- g7e (RTX PRO 6000 Blackwell) — fails to load (#40821, #40802, #42432)
- Pipeline parallelism unless TP=8 OOMs in T3 (T7 optional)
- LMCache integration — not yet validated on CSA+HCA
- CPU offload — vLLM #43093 crashes
- Reasoning-trace-quality benchmarks — separate evaluation; this spec is serving-only

---

## Verification Criteria

### Stage 4a — GPU Health

- [ ] All B300 GPUs report ECC enabled, 0 uncorrectable errors
- [ ] NVLink topology: all 8 GPUs via NVSwitch
- [ ] Driver 580+ / CUDA 13+ (per memory: B200 driver baseline)
- [ ] AL2023 AMI confirmed (`ib_umad` present)
- [ ] No Xid errors in dmesg

### Stage 5 — Serving Stack

- [ ] Image pulled with vLLM ≥ 0.21 + PR #42701 patch (or nightly post-2026-05-18)
- [ ] transformers pinned (4.56.x or 4.57+ with PR #42806)
- [ ] Model loads without expert-mapping errors (#42769)
- [ ] `/health` returns 200 within 20 minutes (DeepGEMM JIT + 1M context allocation)
- [ ] Single completion against `/v1/completions` returns coherent output (precision smoke test, per SGLang #25662)
- [ ] Non-think / Think High / Think Max modes all selectable
- [ ] Tool-call parser identified; BFCL smoke test passes

### Stage 6 — Benchmark

- [ ] All T0-T5 runs complete with 0% error rate
- [ ] T6 either completes with sampled-output validation OR is documented as gated on upstream fixes
- [ ] CSA+HCA efficiency claim measured at 256K and 1M context
- [ ] Results JSON written to `/results/` and synced to blueprint
- [ ] Cross-model comparison table updated in `reports/benchmark-results.md`

### Stage 7 — Readiness Audit

- [ ] All success criteria met or characterized
- [ ] No unresolved lessons with severity ≥ HIGH
- [ ] `mdc learn deepseek-v4 vllm --from blueprints/deepseek-v4-flash/lessons.md` executed (creates the card if it doesn't exist)
- [ ] Spec status updated to COMPLETE with bench-report timestamp

---

> **Note**: Operational artifacts (lessons, results, deployment notes) belong in
> `blueprints/deepseek-v4-flash/`, not in this spec.
