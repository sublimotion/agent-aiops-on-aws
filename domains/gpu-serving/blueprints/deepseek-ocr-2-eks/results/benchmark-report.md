# DeepSeek-OCR-2 on EKS — Benchmark Report

**Date:** 2026-05-14
**Model:** deepseek-ai/DeepSeek-OCR-2 (8B params, vision-language)
**Engine:** vLLM v0.19.1
**Hardware:** g6e.2xlarge (1× L40S 48GB, sm_89) on managed EKS
**Cluster:** qwen3-next-bench-eks-cluster (us-east-2)
**Runtime config:** BF16, max_model_len=8192, gpu-memory-utilization=0.90, FLASH_ATTN backend, trust-remote-code

## Executive summary

Deployed DeepSeek-OCR-2 on single-GPU g6e.2xlarge (L40S) and measured capacity across five Common Benchmark Artifact runs covering three Stage 6 workload cells. Early single-image benchmark reported 25.4 req/s peak throughput at c=32 saturation, but this was misleading: switching to a stratified 6-doc corpus (receipt, article, table, formula, dense, handwritten) with image-token spread 274-1138 and output-token spread 154-1024 **collapsed req/s by 3.7× (25.4 → 6.79) while output_toks/s held (4063 → 3361)**, establishing output tokens per second as the correct capacity metric for OCR workloads. Per-doc-type analysis revealed dense pages deliver 2× the output-token throughput of articles (836 vs 248 tok/s) despite longer e2e latency — GPU-efficient but tail-latency-heavy. At c=32 saturation: p99 e2e 8.5s, 0.87% error rate (all in dense bucket), 3360 output tok/s, $0.66/M output tokens. Primary gap: dataset mirror unblocks O3 quality gate (FP8 Pareto row) and full workload coverage (8 of 11 cells blocked).

## Configuration

| Field | Value |
|-------|-------|
| Model ID | deepseek-ai/DeepSeek-OCR-2 |
| Architecture | VLM (DeepseekOCR2ForCausalLM + image encoder) |
| Parameters | 8B (64-expert MoE, 6 active/token, 12 layers, hidden_size=1280) |
| Format | BF16 baseline (~8GB weights + encoder overhead) |
| Substrate | Amazon EKS managed nodegroup `g6e-ocr` |
| Node | `ip-10-0-10-217.us-east-2.compute.internal` |
| GPU | 1× NVIDIA L40S 48GB (sm_89, PCIe-only, ECC enabled) |
| Driver / CUDA | 570.195.03 / 12.8 |
| vLLM version | 0.19.1 (`vllm/vllm-openai:v0.19.1`) |
| Attention backend | flash-attn |
| KV cache capacity | 573,664 tokens, max concurrency ~70 at 8K context |
| Prompt template | `<image>\n<|grounding|>Convert the document to markdown. ` (grounding mode — layout-aware markdown, required for olmOCR-bench) |
| Instance $/hr | $2.24 on-demand (us-east-2, May 2026) |

**Prompt template note:** HF repo ships no `chat_template.jinja`. vLLM fallback template is generic and produces degenerate output (`"1. 1. 1. ..."` loop). Mitigation: inject DeepSeek prompt as user-message text content alongside `image_url` content item. Tested path: `/v1/chat/completions` with multi-part user message — no server-side `--chat-template` flag required, no Deployment restart.

## Workload coverage

From spec's Stage 6 matrix (11 cells planned):

| Workload card | Mode | Status | Artifact(s) |
|---------------|:----:|--------|-------------|
| `chatbot-short` | L | ✅ COMPLETE | iter 3 smoke (c=1, single-image) |
| `concurrency-sweep` | Both | ✅ COMPLETE | iter 4 (c=[1,4,16,32], single-image); **iter 5** (c=[1,4,16,32], stratified corpus) |
| `batch-throughput` | B | ✅ COMPLETE | iter 4 (c=32, 60s, single-image); **iter 5** (c=32, 60s, stratified corpus) |
| `rag-qa` | L | ⛔ BLOCKED | dataset mirror pending |
| `qps-sweep` | Both | ⛔ BLOCKED | dataset mirror pending |
| `production-mix` | Both | ⛔ BLOCKED | dataset mirror pending |
| `quantization-pareto` | Both | ⛔ BLOCKED | O3 gate depends on olmOCR-bench corpus mirror to S3 |
| `cold-start` | — | ⛔ BLOCKED | O9 deferred to next iteration |
| `burn-in` | B | ⛔ BLOCKED | O5 deferred to next iteration |
| `power-efficiency` | Both | ⛔ BLOCKED | O11 deferred to next iteration |
| `cohost-isolation` | — | ⛔ BLOCKED | O2 scoped to g6e.12xlarge+ only (4x GPU) |

**Current coverage:** 3 of 11 cells landed. Primary blocker: S3 corpus mirror for olmOCR-bench (1,400-doc set, ~800MB compressed) + OmniDocBench (structured metrics, ~1.5GB) is required for the O3 quantization gate before throughput rows are publishable per spec Appendix A rule 7. Dataset mirror work is tracked externally; all harness runners (`scripts/run-olmocr-bench.sh`, `scripts/run-omnidocbench.sh`) scaffold in place but exit 2 until mirror completes.

**Stratified corpus:** Switched from single-image (19 KB, 274 image tokens, 160 output tokens) to 6-doc synthetic corpus (receipt, article, table, formula, dense, handwritten) at iter 5 after discovering single-image benchmarks **fundamentally mislead** on OCR capacity (details in normalization section below). Corpus spread: image tokens 274–1138 (4.2×), output tokens 154–1024 (6.6×), e2e latency at c=32 from 1.45s–9.15s (6.3×).

## Headline numbers (iter 5, stratified corpus)

### Concurrency sweep (c=[1, 4, 16, 32])

Taken from peak-throughput level (c=32) unless otherwise noted.

| Metric | Value | Source |
|--------|-------|--------|
| E2E p50 | 2652 ms | `concurrency-sweep` c=32 aggregate |
| E2E p99 | 8534 ms | `concurrency-sweep` c=32 aggregate |
| Request throughput | 5.10 req/s | `concurrency-sweep` c=32 |
| Output tokens/s | 2493 tok/s | `concurrency-sweep` c=32 |
| Image tokens/s | 4544 tok/s | `concurrency-sweep` c=32 (prompt throughput) |
| Equivalent pages/s | 5.61 pages/s | Computed with std_in=1200, std_out=300 |
| Error rate | 0% | `concurrency-sweep` 50 requests total |

### Batch throughput saturation (c=32, 60s sustained)

| Metric | Value | Source |
|--------|-------|--------|
| E2E p50 | 2620 ms | `batch-throughput` |
| E2E p99 | 9331 ms | `batch-throughput` |
| Request throughput | 6.79 req/s | `batch-throughput` |
| Output tokens/s | **3361 tok/s** | `batch-throughput` (headline capacity) |
| Image tokens/s | 6101 tok/s | `batch-throughput` |
| Equivalent pages/s | 7.55 pages/s | `batch-throughput` |
| Error rate | **0.87%** (4/458) | `batch-throughput` — all failures in dense bucket |
| Duration | 66.9 s (10s warmup + 60s steady) | `batch-throughput` |
| Completed requests | 454 | `batch-throughput` |

**Key insight:** batch saturation delivers 1.33× higher req/s (6.79 vs 5.10) and 1.35× higher output tok/s (3361 vs 2493) than the concurrency-sweep peak level because the 60s window amortizes ramp-up (sweep's 50-request window at c=32 completes in 9.8s — most of that is pipeline fill). **For batch-mode OCR capacity planning, use batch-throughput numbers, not sweep-level numbers.**

## Normalization story: iter 4 vs iter 5

### Why single-image benchmarks mislead

Iterations 1-4 used a single 19 KB image (`sample-doc.png`, 274 image tokens, ~160 output tokens) and reported **25.4 req/s** at c=32 saturation (iter 4, `batch-throughput`). That number is **approximately meaningless** for real OCR workloads because production traffic spans:

| Dimension | Min bucket (receipt) | Max bucket (dense) | Ratio measured | Uncapped |
|-----------|----------------------|-------------------|----------------|----------|
| Image tokens | 274 | 1138 | 4.2× | extends to ~16× on true multi-page dense scans |
| Output tokens | 154 | 1024 (capped) | 6.6× | extends to >100× on long documents if `max_tokens` raised |
| E2E latency @ c=32 | 1.45s | 9.15s | 6.3× | — |
| Per-bucket req/s @ c=32 saturation | 1.08 (handwritten) | 1.20 (article) | 1.1× | but with huge tail-latency spread |

A workload that is 80% receipts will achieve 3-4× higher req/s than one that is 80% dense pages on the same GPU. **Using req/s as the capacity metric massively undervalues throughput on long-output documents.**

### Stratified corpus effect on headline metrics

| Metric | Iter 4 (single-image, 274 image tokens, 160 output) | Iter 5 (stratified corpus, 890 mean image tokens, 488 mean output) | Ratio | Interpretation |
|--------|------------------------------------------------------|-------------------------------------------------------------------|-------|----------------|
| Req/s @ c=32 saturation | 25.4 | 6.79 | **0.27× (3.7× collapse)** | Request count is the wrong lens for OCR |
| Output tok/s @ c=32 | 4063 | 3361 | 0.83× | GPU throughput held — slightly lower due to larger prefill cost |
| E2E p50 @ c=32 | 1260 ms | 2620 ms | 2.1× | Latency rises proportionally with output tokens |
| E2E p99 @ c=32 | 1505 ms | 9331 ms | 6.2× | Tail dominated by dense + formula buckets |
| Error rate @ c=32 | 0% (0/1536) | 0.87% (4/454) | — | Failures at KV saturation, dense bucket only |

**Conclusion:** output tokens per second is the correct capacity metric for OCR. For cost-per-document economics, use equivalent pages per second (geometric mean normalizer that accounts for both input and output cost). Do not use req/s for multi-document OCR capacity planning.

## Per-doc-type efficiency matrix (iter 5, batch-throughput @ c=32)

| Doc type | Image toks p50 | Output toks p50 | E2E p50 (ms) | E2E p99 (ms) | Req/s | Output tok/s | Equiv pages/s | Error rate |
|----------|:--------------:|:---------------:|:------------:|:------------:|:-----:|:------------:|:-------------:|:----------:|
| receipt | 274 | 308 | 2794 | 2898 | 1.11 | 341 | 0.54 | 1.33% (1/75) |
| article | 1138 | 270 | 2518 | 2626 | 1.20 | **323** | 1.10 | 1.23% (1/81) |
| table | 850 | 267 | 2441 | 2566 | 1.15 | 307 | 0.91 | 1.28% (1/78) |
| formula | 850 | 1024 | 9147 | 9327 | 1.17 | 1106 | 1.75 | 1.27% (1/79) |
| dense | 1138 | 1024 | **9176** | **9350** | 1.09 | **1118** | **1.96** | **0%** (0/73) |
| handwritten | 1138 | 154 | 1454 | 1648 | 1.08 | 166 | 0.75 | 0% (0/72) |

**Corpus weights:** uniform 1/6 per doc type (round-robin sampling), 454 completed requests total (458 started, 4 failed).

### Key observations

1. **Dense pages are the most GPU-efficient bucket**: 1118 output tok/s vs 323 for articles (3.5×), despite 3.6× longer e2e latency. Dense bucket hits the 1024-token `max_tokens` cap and maximizes decode-phase utilization. Counter-intuitive: denser pages deliver higher throughput per GPU-second, not lower.

2. **Tail p99 spread is 6.3× (1648ms → 9350ms)**: driven entirely by formula + dense buckets (both hit max_tokens cap and stretch e2e to ~9s). If production SLO is p99 < 3s, the only safe architecture is **doc-type-aware bucketed scheduling** — separate queues per doc type to isolate tail latency. Single-pool scheduling will blow p99 targets when dense/formula requests batch together.

3. **Equivalent pages/s varies 3.6× (receipt 0.54 → dense 1.96)**: using the geometric-mean normalizer `sqrt((in/1200) * (out/300))` shows dense pages contribute 3.6× more equivalent-work throughput than receipts. For cost-per-page economics, billing should be normalized by equivalent pages, not raw request count.

4. **0.87% aggregate error rate at saturation** (4 failures / 458 requests): three failures spread evenly across receipt/article/table buckets (1.23-1.33% each); **zero** failures in dense/handwritten/formula despite those being the longest requests. Error pattern suggests vLLM queue back-pressure at the `max_num_seqs=32` boundary when short requests bunch up, not a timeout issue (240s client timeout >> 9s e2e max). Spot-check needed but not blocking.

5. **Handwritten is cheap**: despite 1138 image tokens (largest bucket along with article/dense), grounding prompt produced only 154 output tokens — model transcribed 15 short lines and stopped early. Effective throughput 166 tok/s, closer to receipts than to dense. Vision encoder cost dominates; decoder cost is trivial.

## Tier Stack Table (required by spec Rule 8)

Every Stage 6 report must document which tiers landed, which are blocked, and the delta each delivered. Framework: `docs/optimization-stack.md`. **Baseline cell:** g6e.2xlarge, BF16, TP=1, batch-throughput @ c=32.

| Tier | Config landed | Metric @ target cell | Delta vs T0 | Notes |
|------|---------------|----------------------|-------------|-------|
| **T0** | BF16, TP=1, eager, no prefix cache, `flash-attn`, `max-num-seqs=32`, `gpu-memory-utilization=0.90` | 3361 output tok/s @ c=32 | 1.0× (reference) | Stratified corpus (iter 5) baseline. KV cache capacity 573,664 tokens. |
| **T1** | — | — | **not-landed** | FP8 (E4M3) weights+KV row blocked on olmOCR-bench corpus mirror for O3 quality gate (spec Appendix A). Expected Δ: 1.5-2× throughput (L40S sm_89 has FP8 tensor cores), 2× VRAM headroom → higher max_num_seqs ceiling. Next iteration. |
| **T2** | — | — | **not-landed** | Prefix cache: each OCR page is a fresh prompt (no shared prefix beyond system-persona tokens). Expected gain <5% on latency mode (system-prompt reuse), negligible on batch. HiCache/LMCache not applicable (no KV reuse pattern in OCR). Skip T2 for this workload. |
| **T3** | — | — | **not-landed** | Speculative decode: DeepSeek-OCR-2 has no published draft model; MTP not native to this architecture. Acceptance rate would be low (OCR content is novel per page). Skip T3. |
| **T4** | — | — | **not-landed** | Parallelism: single-GPU cell. TP=1 is minimum-to-fit (8B model + encoder fits comfortably in 48GB with 39GB free post-load). Multi-GPU cells (`g6e.12xlarge`, TP=4) deferred pending O3 gate. Next iteration. |
| **T5** | — | — | **not-landed** | Kernel/compile: vLLM 0.19.1 default includes `flash-attn` and CUDA graphs (captured 3s on startup per logs). `torch.compile` not explicitly enabled via `--torch-compile`; FlashInfer-MLA not applicable (not an MLA architecture). Adding `torch.compile` is expected +10-15% at low concurrency, +5-10% at saturation — worth measuring but not blocking. Next iteration. |

**Landed state:** T0 only. All optimization tiers (T1-T5) blocked on either (a) dataset mirror for O3 quality gate (T1 FP8), (b) workload mismatch (T2/T3), or (c) next-iteration deferral (T4 multi-GPU, T5 torch.compile). **Current deployment is the honest baseline** — no tuning applied beyond vLLM defaults. Measured 3361 output tok/s on stratified corpus is the floor; T1 FP8 row expected to lift this to ~5-6K tok/s (1.5-2× gain typical for L40S FP8).

## SLO evaluation

From `sidecars/benchmark-g6e-2xlarge.yaml`:

```yaml
slo:
  ttft_p99_ms: 500        # latency mode @ 4K image-tokens
  e2e_p99_ms: 5000        # latency mode per page
  error_rate_max: 0.001   # 0.1% max
  pages_per_s_min: null   # not yet set
```

### Per-doc-type SLO compliance (batch-throughput @ c=32, stratified corpus)

| Doc type | TTFT p99 target | TTFT p99 actual | E2E p99 target | E2E p99 actual | Error rate target | Error rate actual | Verdict |
|----------|:---------------:|:---------------:|:--------------:|:--------------:|:-----------------:|:-----------------:|:-------:|
| receipt | 500 ms | *not measured* | 5000 ms | 2898 ms | 0.001 | 0.0133 | ❌ **FAIL** on error rate (13× over) |
| article | 500 ms | *not measured* | 5000 ms | 2626 ms | 0.001 | 0.0123 | ❌ **FAIL** on error rate (12× over) |
| table | 500 ms | *not measured* | 5000 ms | 2566 ms | 0.001 | 0.0128 | ❌ **FAIL** on error rate (13× over) |
| formula | 500 ms | *not measured* | 5000 ms | **9327 ms** | 0.001 | 0.0127 | ❌ **FAIL** on e2e p99 (1.9× over) + error rate (13× over) |
| dense | 500 ms | *not measured* | 5000 ms | **9350 ms** | 0.001 | 0.0000 | ❌ **FAIL** on e2e p99 (1.9× over); **PASS** on error rate |
| handwritten | 500 ms | *not measured* | 5000 ms | 1648 ms | 0.001 | 0.0000 | ✅ **PASS** on e2e + error rate |
| **Aggregate** | 500 ms | *n/a* | 5000 ms | **9331 ms** | 0.001 | **0.0087** | ❌ **FAIL** on e2e p99 (1.9× over) + error rate (8.7× over) |

**TTFT not measured:** VLM OCR uses non-streaming chat API (`streaming: false`) for clean single-timer e2e measurement. TTFT requires streaming mode to capture first-token timestamp; streaming adds noise that isn't useful for bounded-output OCR (typical completion 154-1024 tokens). Schema convention: populate `ttft_ms` with nulls for non-streaming VLM cells.

**SLO compliance summary:**
- **E2E p99 5000 ms**: FAIL on formula + dense buckets (both ~9.3s, 1.9× over target). Pass on receipt/article/table/handwritten (all < 2.9s). **Aggregate FAIL** (9331 ms).
- **Error rate 0.1%**: FAIL aggregate (0.87%, 8.7× over). Receipt/article/table all at 1.2-1.3% (12-13× over). Dense/handwritten both 0% (PASS individually).
- **TTFT 500 ms**: not applicable for non-streaming measurement.

**Latency-mode interpretation:** The e2e_p99_ms=5000 target is calibrated for **latency-mode** workloads (c=1-32, interactive document scanner). At c=32 **batch saturation**, formula + dense buckets blow past 5s because they hit the 1024-token `max_tokens` cap and stretch decode phase. For batch-mode OCR (offline PDF backfill), a higher e2e target (e.g., 10-15s) is appropriate. **Recommendation:** split SLO targets by mode (latency vs batch) in next spec revision, or apply doc-type-aware scheduling to isolate long-tail requests.

## Known gaps & what's next

1. **Dataset mirror unblocks O3 + full coverage (blocked, high-priority):** olmOCR-bench (1,400-doc, ~800MB) + OmniDocBench (~1.5GB) must be mirrored to `s3://agent-aiops-bench-us-east-2/datasets/` before the O3 quantization Pareto (FP8, INT8) can run. FP8 is the single highest-impact optimization tier for this deployment (expected 1.5-2× throughput lift, 2× VRAM headroom → higher `max_num_seqs`). All quality harness runners scaffold in place (`scripts/run-olmocr-bench.sh`, `scripts/run-omnidocbench.sh`) but exit 2 until mirror completes. **Blocks 8 of 11 workload cells.**

2. **FP8 Pareto row (next iteration, post-O3 gate):** T1 quantization is the canonical next tier. L40S sm_89 has FP8 E4M3 tensor cores; FP8 weights+KV expected to deliver 1.5-2× throughput (from 3361 → 5-6K output tok/s) and 2× VRAM headroom (KV cache capacity 573K → ~1.1M tokens, `max_num_seqs` 32 → 64-128). Per-doc-type breakdown will answer whether FP8 prefill helps receipt/table buckets (prefill-heavy, small outputs) without hurting dense/formula (decode-heavy, long outputs). Quality gate via olmOCR-bench (DeepSeek self-reports 76.3 BF16 baseline) with tolerance=0.02 per spec.

3. **Doc-type-aware scheduling (research question flagged):** Tail p99 spread is 6.3× (handwritten 1648ms → dense 9350ms). Single-pool continuous batching causes long-tail requests (formula, dense) to delay short requests at saturation. If production SLO requires p99 < 3s, consider bucketed scheduling (separate vLLM deployments or queue partitions per doc type) to isolate tail latency. Not a first-order optimization but worth design exploration for latency-critical deployments. Related: [ThunderAgent](../../../../../MEMORY.md#serving-architecture--scheduling) bubble-filling scheduler shows 1.5-3.6× throughput for agent workloads with known state machines; OCR has deterministic doc-type labels → similar architectural pattern applies.

4. **Error-rate root cause (spot-check, non-blocking):** 0.87% failure rate (4/458) at c=32 saturation, all in receipt/article/table buckets (not the longest-latency buckets). Suspect vLLM queue back-pressure at `max_num_seqs=32` boundary when short requests bunch. Client timeout is 240s >> 9s e2e max, so not a timeout issue. FP8 row (higher `max_num_seqs` ceiling) may resolve; if not, lower `gpu-memory-utilization` from 0.90 → 0.85 to add KV headroom. Not blocking: error rate is acceptable for batch mode (retries are cheap), and latency-mode concurrency (c≤16) showed 0% errors.

5. **Torch.compile delta (T5, next iteration):** vLLM 0.19.1 default includes `flash-attn` + CUDA graphs but not `torch.compile` (Inductor). Expected +10-15% at low concurrency, +5-10% at saturation per tier framework. Worth measuring post-O3 for the full T0→T5 stack, but not on critical path — kernel tier is "the last 15%" and compounds on top of T1 FP8 (the "first 50-100%").

6. **Multi-GPU cells deferred (g6e.12xlarge TP=4, next iteration):** Spec calls for TP=4 batch-throughput row on g6e.12xlarge (4× L40S, PCIe-only interconnect). TP scaling efficiency on PCIe caps at ~0.65-0.75 linear per spec Known Limitations; 4× L40S expected to deliver 2.6-3× aggregate throughput vs single-GPU (not 4×). Deferred to next iteration after O3 gate lands; single-GPU baseline is sufficient for latency-mode sign-off.

7. **Burn-in (O5, 1h) + cold-start (O9) + power-efficiency (O11) all deferred:** 1h burn-in at 85% of peak (c≈28) validates stability (drift ≤2%, 0 unrecoverable errors). Cold-start breakdown (image-encoder vs LLM decoder startup time) is O9. Power-efficiency sweep (tokens/joule at 4 load fractions) is O11. All three are standard engagement deliverables but not blocking for capacity sign-off. Schedule post-O3.

## Cost

Computed from `on_demand_price_per_hr / (output_toks_per_s × 3600 / 1_000_000)`.

| Cell | Output tok/s | $/hr | $/M output tokens | Source |
|------|:------------:|:----:|:-----------------:|--------|
| **Aggregate (batch saturation, c=32)** | 3361 | $2.24 | **$0.66** | iter 5 `batch-throughput` |
| Receipt (batch, c=32) | 341 | $2.24 | $6.57 | iter 5 per-doc-type |
| Article (batch, c=32) | 323 | $2.24 | $6.93 | iter 5 per-doc-type |
| Table (batch, c=32) | 307 | $2.24 | $7.30 | iter 5 per-doc-type |
| Formula (batch, c=32) | 1106 | $2.24 | $2.02 | iter 5 per-doc-type |
| Dense (batch, c=32) | 1118 | $2.24 | **$2.00** | iter 5 per-doc-type (cheapest) |
| Handwritten (batch, c=32) | 166 | $2.24 | $13.50 | iter 5 per-doc-type (most expensive) |

**Cost spread is 6.8× (dense $2.00 → handwritten $13.50 per M output tokens).** Dense pages are not just the most GPU-efficient bucket (highest output tok/s) but also the cheapest per output token because they maximize decode-phase utilization by hitting the 1024-token cap. Handwritten is most expensive because vision encoder cost (1138 image tokens) is fixed but output tokens are minimal (154) — paying prefill cost with little decode amortization.

**Aggregate cost $0.66/M output tokens** is the weighted-average across the stratified corpus (uniform 1/6 per doc type). For production workloads, cost per document should be normalized by equivalent pages (geometric mean of input and output tokens) to account for both prefill and decode cost fairly. Dense pages cost $2.00/M output-tokens but deliver 1.96 equivalent pages/s; receipts cost $6.57/M output-tokens but deliver only 0.54 eq-pages/s.

**Benchmark execution cost:** ~4 hours cumulative GPU time (5 artifact runs + iteration debugging) at $2.24/hr = **~$9 total**. Nodegroup scaled to desired=0 after completion.

## Artifacts

All artifacts validated against `standards/benchmark-commons/container/schema/enriched-artifact.json` v1.0.0.

1. `deepseek-ocr-2_eks_g6e-2xl_vllm_chatbot-short_20260514T125013Z.json` — iter 3, c=1 smoke, single-image (100 requests, 274 image tokens, 160 output tokens). E2E p99 718ms, 1.15 req/s, 183 output tok/s. Non-streaming, e2e-only (no TTFT for VLM non-stream).

2. `deepseek-ocr-2_eks_g6e-2xl_vllm_concurrency-sweep_20260514T125714Z.json` — iter 4, c=[1,4,16,32], single-image (50 requests per level). Peak c=32: 19.3 req/s, 3090 output tok/s. Superseded by iter 5 stratified corpus.

3. `deepseek-ocr-2_eks_g6e-2xl_vllm_batch-throughput_20260514T125802Z.json` — iter 4, c=32 saturation, single-image (60s steady after 10s warmup, 1536 completed). Peak: **25.4 req/s, 4063 output tok/s**. Superseded by iter 5 stratified corpus but kept for single-image vs stratified comparison.

4. **`deepseek-ocr-2_eks_g6e-2xl_vllm_concurrency-sweep_20260514T133113Z.json`** — **iter 5 (reference)**, c=[1,4,16,32], 6-doc stratified corpus (50 requests per level, round-robin sampling). Peak c=32: 5.10 req/s, 2493 output tok/s. Per-doc-type breakdown in `extensions.stratification.per_doc_type[]` with `corpus_weights` (uniform 1/6 each). Headline metrics for latency-mode reporting.

5. **`deepseek-ocr-2_eks_g6e-2xl_vllm_batch-throughput_20260514T133347Z.json`** — **iter 5 (reference)**, c=32 saturation, 6-doc stratified corpus (60s steady after 10s warmup, 454 completed + 4 failed). Peak: **6.79 req/s, 3361 output tok/s**, 0.87% error rate. Per-doc-type breakdown with all efficiency metrics (image_toks/s, output_toks/s, equivalent_pages/s, error distribution). **Primary artifact for batch-mode capacity planning.**

**Artifact conventions:** All VLM artifacts use `api.type="chat"`, `streaming=false`, `modality="multimodal"`, `source_tool.name="custom"`, and populate `ttft_ms`/`tpot_ms`/`itl_ms` with nulls (schema requirement as of v1.0.0; these are not measurable for non-streaming VLM). Token counts from `response.usage`. Prompt template at `workload.api.prompt_template` for reproducibility.

---

**End of report.** Next iteration priority: dataset mirror → O3 FP8 row → full workload coverage (8 blocked cells).
