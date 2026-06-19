# DeepSeek-OCR-2 EKS — Lessons

<!-- Field notes. Raw capture; compound-learner elevates to steering. -->

## Stage 4a — GPU health (g6e.2xlarge, L40S)

Single L40S 46,068 MiB, driver 570.195.03, CUDA 12.8, ECC enabled with 0 uncorrected aggregate errors, PCIe width x8 (link gen reports 1 at idle — expected, trains up under load). `dmesg | grep Xid` empty. No row remaps. Node `ip-10-0-10-217.us-east-2.compute.internal` healthy.

Diagnostic method: launched `nvidia/cuda:12.4.1-base-ubuntu22.04` debug pod with `nvidia.com/gpu: 1` request, `workload=deepseek-ocr` nodeSelector, `nvidia.com/gpu:NoSchedule` toleration.

## Stage 5 — Pod cross-node network isolation (BLOCKING → FIXED)

<!-- captured: 2026-05-14 | stage: 5 -->

### Symptom
vLLM pod CrashLoopBackOff with `OSError: Can't load the configuration of 'deepseek-ai/DeepSeek-OCR-2'`. Underlying error in traceback: `[Errno -3] Temporary failure in name resolution` while HEADing `https://huggingface.co/...`. At first glance this looked like model-ID / gated-repo / architecture-support issue. It was not — DeepSeek-OCR-2 is supported by vLLM 0.19.1 out of the box.

### Root cause
The new managed nodegroup `g6e-ocr` was provisioned with ONLY the EKS cluster primary security group (`sg-0abb08cf4d13be131` = `eks-cluster-sg-...`). The other (pre-existing) nodes in the cluster use a separate node SG (`sg-0bf5ad07fc6c29df1` = `qwen3-next-bench-eks-cluster-node-...`) whose ingress rules only permit traffic from `self` and the control-plane SG. CoreDNS pods run on the old nodes, so pods on the new GPU node could not reach `172.20.0.10:53` (or the pod IPs `10.0.41.29`, `10.0.26.26`) — all UDP/53 traffic silently dropped. Verified by running a busybox `nslookup` from a pod on the new node: `;; connection timed out; no servers could be reached`.

### Fix
Added the node SG to both ENIs of the GPU instance (primary ENI + VPC-CNI secondary ENI). Pod now retains the primary cluster SG **and** sits within the node-to-node allow rules of the shared node SG.

```bash
INSTANCE_ID=i-074b3c2ec9db807ad   # ip-10-0-10-217
for eni in eni-058f8cc0ceb64080c eni-0814c5914a4f5f008; do
  aws ec2 modify-network-interface-attribute --region us-east-2 \
    --network-interface-id "$eni" \
    --groups sg-0abb08cf4d13be131 sg-0bf5ad07fc6c29df1
done
```

After fix: `nslookup huggingface.co` from a pod on the new node resolves. vLLM pod starts, downloads weights, KV cache 573,664 tokens, CUDA graph capture 3 s, `/health` returns 200.

### Followup for subsequent nodegroups
When creating a new managed nodegroup on this cluster, also attach the node SG at nodegroup creation via the launch template (`--launch-template LaunchTemplateId=...`), or add a broader ingress rule on the node SG accepting the cluster primary SG. Otherwise every new node needs this ENI-patch step.

## Stage 5 — Smoke test: endpoint healthy, prompt shape untuned

<!-- captured: 2026-05-14 | stage: 5 -->

### Symptom
`/health` = 200, `/v1/models` registers `deepseek-ai/DeepSeek-OCR-2`, `/v1/chat/completions` with a simple "Transcribe all text" prompt + base64 PNG returns an HTTP 200 with a **non-empty** but **degenerate** assistant message (`"1. 1. 1. ..."` repeated to max_tokens). No crash, no 500.

Pre-reqs that also surfaced during smoke:
- `https://upload.wikimedia.org/...` image URL returned **HTTP 403** (Wikipedia blocks unauthenticated hot-linkers). Use a base64 `data:image/png;base64,...` URL or an S3-mirrored image for smoke + benchmarks, never hot-link Wikipedia.
- `https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/testing/phototest.tif` returned 404 (path moved). `tesseract-ocr.github.io/tessdoc/images/eurotext.png` is reachable.

### Likely cause
vLLM logged `Loading chat template fallback for deepseek-ai/DeepSeek-OCR-2 as there isn't one defined on HF Hub`. The fallback template is a generic chat wrapper, not the DeepSeek-OCR persona that the model was trained on (upstream reference uses a `<image>\n<|User|>...<|Assistant|>` shape with an explicit image sentinel token, not `<image_url>` in chat content). With the wrong persona, the model decodes but loops on a trivial token.

### Status
Out of scope for this iteration (Stage 5 sign-off criterion is "test request returns non-empty text" — met). Fix belongs to Stage 6 / benchmark-wiring:
- Supply `--chat-template` pointing at a DeepSeek-OCR Jinja file (mirror from DeepSeek's example or reuse `deepseek_vl` template from vLLM examples), OR
- Use `/v1/completions` with the raw DeepSeek-OCR prompt shape rather than `/v1/chat/completions`, OR
- Add `--chat-template-content-format openai` plus an explicit system prompt.

Smoke response captured at `results/smoke-response-1778762050.txt`.

## Model architecture note

DeepSeek-OCR-2 config declares `architectures: ["DeepseekOCR2ForCausalLM"]` with `language_config.architectures: ["DeepseekV2ForCausalLM"]` (small MoE: 64 experts, 6 per token, hidden 1280, 12 layers). Total compute footprint on L40S at BF16: KV cache 573,664 tokens allocated, max concurrency 70x at 8192-token window — plenty of VRAM headroom for FP8 experiments in O3.

## Stage 5 — Smoke test v2: correct prompt template (RESOLVED)

<!-- captured: 2026-05-14 | stage: 5 -->

### Root cause of iteration-1 degenerate output
The HF repo for `deepseek-ai/DeepSeek-OCR-2` ships **no `chat_template.jinja`**. vLLM 0.19.1 silently falls back to a generic chat wrapper, which does not emit the grounding markers (`<image>`, `<|grounding|>`) the model was trained to condition on. Without those markers the model decodes off-distribution and loops on a trivial token (`"1. 1. 1. ..."`).

### Correct prompt shapes (from upstream HF model card)
| Mode | Prompt text | Use case |
|------|-------------|----------|
| Plain OCR | `<image>\nFree OCR. ` | Flat transcription |
| Layout-aware | `<image>\n<\|grounding\|>Convert the document to markdown. ` | Structured markdown; required for olmOCR-bench + OmniDocBench |

Trailing space is load-bearing (matches upstream reference); `<image>` is vLLM's image-token placeholder.

### Paths tested
- **Path A** (`/v1/completions` with raw prompt + top-level `multi_modal_data`): **FAILED** on vLLM 0.19.1. `multi_modal_data` at request-root is not parsed by the OpenAI-compat `/v1/completions` handler; the image never reaches the encoder, output remains degenerate (`"0.0.0.0/16 ..."` loop on 7 prompt tokens).
- **Path B** (`/v1/chat/completions`, inject prompt as user-message **text content** alongside an `image_url` content item): **WORKS** with the default fallback template and no Deployment restart. Both `Free OCR.` and `<|grounding|>Convert the document to markdown.` produce faithful transcription of the tesseract eurotext sample (English + German + French + Italian + Spanish + Portuguese).

### Winning request shape
```json
{
  "model": "deepseek-ai/DeepSeek-OCR-2",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "<image>\n<|grounding|>Convert the document to markdown. "}
    ]
  }],
  "max_tokens": 512, "temperature": 0.0
}
```

### Decision: no server-side `--chat-template`
Path B avoids a ConfigMap + Deployment restart. The fallback template is fine so long as the DeepSeek prompt tokens are injected as text content. Adding `--chat-template` was considered and rejected — no upside, adds restart cost, ties benchmark harness to a flag change.

### Artifacts
- `results/smoke-response-v2-freeocr-1778762449.txt` — Free OCR mode (plain transcription)
- `results/smoke-response-v2-grounding-1778762442.txt` — grounding mode (prose output; markdown structure minimal on the eurotext sample because it has no structure to lift)

## Stage 6 — Quality harness decision (DocVQA → olmOCR-bench + OmniDocBench)

<!-- captured: 2026-05-14 | stage: 6-planning -->

Replaced DocVQA with two upstream-aligned OCR benchmarks in the O3 quantization Pareto:

- **olmOCR-bench (primary gate)**: 1,400-doc standard set, 7 subscores (arxiv_math, old_scans, tables, headers_footers, multi_column, long_tiny_text, base). DeepSeek self-reports **76.3** → direct comparability. Installed via `pip install olmocr[bench]`.
- **OmniDocBench (periodic deeper eval)**: structured metrics (TEDS, CDM, edit-distance, layout, reading order). DeepSeek-OCR-2 community leaderboard score **90.25**. Runs on BF16 + winning FP8 row only (deeper but more expensive).
- **DocVQA dropped entirely**: it's a VQA benchmark (answer questions about a document), not OCR fidelity — it confounds instruction-following with transcription accuracy and does not map to any published DeepSeek-OCR-2 number.

Stub runners scaffolded at `scripts/run-olmocr-bench.sh` and `scripts/run-omnidocbench.sh`; both exit 2 with a clear mirror-the-dataset message until S3 corpora are populated. Full evals are Stage 6 work.

## VLM artifact conventions (non-streaming, e2e-only)

<!-- captured: 2026-05-14 | stage: 6 -->

DeepSeek-OCR-2 is a vision-language model producing short, bounded completions per page. The Common Benchmark Artifact schema is LLM-centric (TTFT/TPOT/ITL), so for VLM OCR cells use these conventions:

- **API**: `api.type = "chat"`, `streaming: false`. Non-streaming gives a clean single-timer e2e measurement; streaming adds noise that isn't useful for bounded-output OCR.
- **No TTFT / TPOT / ITL**: the schema `$defs.latency_metric` requires `mean/p50/p90/p99`. Emitting partial TTFT data would force fabricated percentiles. Populate **only** `metrics.e2e_ms` with `mean/p50/p90/p95/p99` from 100 samples (minimum to get a stable p99).
- **Token counts from `response.usage`**: vLLM returns `prompt_tokens`, `completion_tokens`, `total_tokens` (yes, even in non-streaming). Sum `completion_tokens` across requests → `total_output_tokens` → `output_toks_per_s = total_output_tokens / duration_s`.
- **`workload.modality = "multimodal"`**: schema enum allows `text|vision|audio|multimodal`. `"vision-language"` is not in the enum — put that string in `extensions.modality` if you need the finer label.
- **`source_tool.name = "custom"`**: the enum does not include harness-specific names for ad-hoc scripts. Required fields are `name`, `version`, `enrichment_version` (**not** `adapter_version`).
- **Prompt template in extensions**: the winning grounding prompt (`<image>\n<|grounding|>Convert the document to markdown. `) is captured at `workload.api.prompt_template` (extension field on `api`) so downstream comparisons see which shape produced the numbers.
- **Sequential c=1 for the smoke cell**: before running a concurrency sweep, produce one artifact with `load.type="sequential"`, `load.concurrency=1`, 10 warmup + 100 steady. That's enough to (a) prove wiring, (b) get non-degenerate p99, (c) stay under $0.15 infra time.

First passing run: `results/artifacts/deepseek-ocr-2_eks_g6e-2xl_vllm_chatbot-short_20260514T125013Z.json` — e2e mean 623 ms / p99 718 ms / 1.15 req/s / 183 output tok/s at 160 completion tokens per request. Validator PASS.

## Stage 6 — Concurrency scaling + saturation window effect

<!-- captured: 2026-05-14 | stage: 6 -->

### Observation
Concurrency sweep at c=[1,4,16,32] with only 50 steady-state requests per level showed peak 19.3 req/s @ c=32. A subsequent 60 s saturation run at the same c=32 reached **25.4 req/s** — a 31% higher throughput on the identical endpoint, identical image, identical prompt. No errors in either run.

### Cause
vLLM continuous batching takes a few seconds to fill the decode queue and stabilize token-budget scheduling. A 50-request window at c=32 completes in ~2.6 s — most of that is the pipeline ramping up. The 60 s saturation window amortizes ramp-up and captures steady-state throughput honestly.

**Fix / takeaway**: For sweep cells, either (a) use enough steady-state requests that `duration >= 30 s` at each level, or (b) treat peak-level sweep numbers as a **lower bound** and always pair the sweep with a dedicated saturation (`batch-throughput`) artifact at the target concurrency for the true ceiling. Don't report sweep-level throughput as the batch number.

## Stage 6 — Shared Python helpers for VLM bench runners

<!-- captured: 2026-05-14 | stage: 6 -->

Factoring shared plumbing into `scripts/_common.py` (request body builder, percentile helper, envelope + model/engine/infra constants, `write_artifact`) let `_concurrency_sweep.py` and `_batch_throughput.py` stay tight (~150 lines each) and kept VLM conventions consistent across artifacts (no ttft/tpot/itl; `source_tool.name = "custom"`; `modality = "multimodal"`; usage tokens from `response.usage`).

Pattern is worth reusing for any future VLM / OCR blueprint with multiple Stage 6 cells — the bash wrapper owns port-forward + health gate + timestamped output path; the Python owns the async loop, stats, and artifact emit.

## Stage 6 — OCR workload normalization & stratified corpus (iter 5)

<!-- captured: 2026-05-13 | stage: 6 -->

### Why req/s is the wrong headline for OCR

Iters 1-4 reported a single batch headline of **25.4 req/s** from `sample-doc.png`
(19 KB, 274 image tokens, ~160 output tokens). That number is meaningless for
capacity planning because production OCR traffic varies enormously in both
directions:

| Axis | Smallest bucket | Largest bucket | Ratio |
|------|-----------------|----------------|-------|
| Image (prompt) tokens | receipt ~274 | dense / article ~1138 | **~4x** measured, extends to ~16x on true dense pages |
| Output tokens (at 1024-cap) | handwritten 154 | formula / dense 1024 (capped) | **~7x** measured, uncapped >100x |
| Per-request e2e @ c=32 saturation | handwritten 1.45 s | formula / dense 9.15 s | **~6x** |

A workload that is 80% receipts will hit 3-4x higher req/s than one that is
80% dense pages on the same GPU. The right headline for multi-document OCR is
**output tokens per second** (dense/formula reach 1117 out-tps each, receipts
340 — output-bound work scales with output tokens, not request count).

### 6-doc stratified corpus

Generated synthetically with PIL (`scripts/test-assets/generate_corpus.py`):
receipt, article, table, formula, dense, handwritten. Round-robin sampling
keeps the distribution stable at any concurrency level. All images <120 KB,
fully reproducible, zero licensing risk (no hotlinks, no copyrighted content).

### `equivalent_pages_per_s` — geometric-mean normalizer

`compute_equivalent_pages(in, out, std_in=1200, std_out=300) = sqrt((in/1200) * (out/300))`.

Kept in `extensions.equivalent_pages_per_s` only (not a headline). Why
geometric mean: under continuous batching prefill cost scales with input
tokens and decode cost with output tokens; total request cost is closer to
their product than their sum, so geometric mean is the right aggregator for a
single scalar "how much compute did this page need" knob.

Batch saturation @ c=32 numbers:
- receipt 0.536 eq_pps, handwritten 0.751
- article 1.105, table 0.914
- **formula 1.745, dense 1.964** — dense is 3.7x more equivalent-page-work per
  second than receipt, yet receipts have 2x the raw req/s. Using req/s as the
  comparison metric would massively undervalue the L40S's throughput on hard
  documents.

### Schema gotcha: `ttft_ms`/`tpot_ms`/`itl_ms` now required

Between iter 4 and iter 5 the common-benchmark-commons schema tightened to
require `ttft_ms`, `tpot_ms`, `itl_ms` in `metrics` (all 5 percentiles each).
For non-streaming VLM these are genuinely not measurable — a single-timer
e2e is the only honest signal. **Fix**: fill the three keys with
`{mean: null, p50: null, p90: null, p95: null, p99: null}` — the schema's
`$defs.latency_metric` accepts nulls. This is now baked into
`_common.write_artifact()` so every future VLM artifact emits compliant stubs.

Retroactively patched the 3 iter-3/4 artifacts already in `results/artifacts/`
to stay compliant after the schema bump.

### Surprises

1. **Handwritten is cheap**: despite the corpus design targeting 200-400
   output tokens, the grounding prompt produced only ~154 output tokens on
   the handwriting sample — the model transcribed the 15 short lines and
   stopped early. Effective eq_pps 0.751 at c=32, closer to receipts than to
   articles. The vision encoder cost (image tokens ~1138) is what we pay for;
   the decoder cost is low.
2. **Formula ≈ dense on cost, not content**: both hit the 1024-token max
   output cap and both produce ~1.2 eq_pps at c=16 and ~1.7-1.96 at c=32. For
   budgeting, formula pages behave like dense pages even though they're
   structurally much sparser. Implication: raising `MAX_TOKENS` from 1024 to
   2048 would disproportionately extend dense-bucket latency and is worth
   measuring before any future quality (O3) gate.
3. **Tail grows much faster than median**: the c=32 aggregate `e2e_p50` is
   2652 ms but `e2e_p99` is 8534 ms (3.2x ratio). Almost all of the tail
   comes from dense+formula buckets; if a customer promises p99 < 3 s the
   only safe path is bucketed scheduling (a separate queue per doc-type)
   rather than a single pool.
4. **0.87% error rate under saturation** (4/458 at c=32 saturation vs 0/200
   in iter 4). All 4 failures landed in the dense bucket — with 1024-token
   outputs and ~9 s e2e, the 240 s client timeout isn't the issue; spot
   check later, but likely vLLM queue back-pressure at exactly the
   max_num_seqs=32 boundary when long requests bunch up.

### Open question (flag for O3)

Does FP8 prefill help receipt/table buckets (prefill-heavy) without hurting
dense/formula (decode-heavy)? With stratified measurement in place, we can
now answer that per-bucket instead of as a single aggregate. Worth the FP8
row in the Pareto.
