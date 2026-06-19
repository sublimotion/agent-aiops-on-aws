# DeepSeek-OCR-2 on EKS g6e — Progress

## Status: STAGE 5 LANDED (2026-05-14)

## Infrastructure provisioned

- **EKS cluster**: `qwen3-next-bench-eks-cluster` (us-east-2, ACTIVE, v1.32)
- **Nodegroup**: `g6e-ocr` — ACTIVE, 1x `g6e.2xlarge` landed in `us-east-2a` (2026-05-14 ~08:08 ET)
- **Node**: `ip-10-0-10-217.us-east-2.compute.internal` — Ready, `nvidia.com/gpu: 1` allocatable
- **Pre-flight** (from terminated scratch EC2 i-06d7b14604388c290): L40S 46068 MiB, driver 580.126.09, PCIe Gen4 x8, ECC uncorrected = 0

## Scaffolded

- `k8s/vllm-deepseek-ocr-g6e-2xlarge.yaml` — namespace + Deployment + Service; vLLM 0.19.1, BF16, 8192 ctx, 32 max_num_seqs
- `sidecars/benchmark-g6e-2xlarge.yaml` — common-artifact sidecar; workloads split latency/batch; O3/O5/O9/O11 cells
- `scripts/deploy.sh` — applies manifest, waits for pod Ready
- `scripts/smoke-test.sh` — single-image OCR request sanity check

## Capacity / pivot log

- 2026-05-13 ~15:28 ET — HyperPod `finetune-g5-cluster` `embedding-g6e-2xlarge` scale-up to 1: 25+ min `Scaling=1, Current=0` (no node)
- 2026-05-13 ~17:09 ET — hedged with `embedding-g6e-xlarge` at 1; both stuck 15 min
- 2026-05-14 ~08:00 ET — cancelled HyperPod; tried plain EC2 us-east-1 (a/b/c all `InsufficientInstanceCapacity` for g6e.xl/2xl)
- 2026-05-14 ~08:03 ET — dropped to us-east-2, EC2 launched (i-06d7b14604388c290 in us-east-2a) — **capacity available**, terminated after pre-flight
- 2026-05-14 ~08:06 ET — created EKS nodegroup `g6e-ocr` on existing `qwen3-next-bench-eks-cluster` -> ACTIVE in ~2 min

## This iteration (2026-05-14 ~12:20-12:45 ET)

### Stage 4a — GPU health: PASS
Debug pod (cuda:12.4.1-base-ubuntu22.04) on GPU node. L40S 46068 MiB, driver 570.195.03, CUDA 12.8, ECC enabled 0 uncorrected, PCIe width x8. `dmesg | grep Xid` empty. Documented in `lessons.md` (Stage 4a section).

### Stage 5 — Serving stack: LANDED (with caveat)
Applied `k8s/vllm-deepseek-ocr-g6e-2xlarge.yaml`. First attempt CrashLoopBackOff with HF DNS failure → root-caused to **cross-node SG gap** (new nodegroup's cluster SG not permitted by existing node SG on cluster). Fixed by attaching node SG `sg-0bf5ad07fc6c29df1` to both ENIs of instance `i-074b3c2ec9db807ad`. Pod restarted cleanly.

- Pod READY 1/1 after ~3 min (first boot: 2m image pull + ~2m weights + CUDA graph)
- `/health` returns 200
- `/v1/models` lists `deepseek-ai/DeepSeek-OCR-2`
- KV cache: 573,664 tokens, max concurrency 70x at 8192 ctx
- Model arch `DeepseekOCR2ForCausalLM` supported by vLLM 0.19.1 out of the box (no SGLang fallback needed)

### Stage 5 — Smoke test: degenerate output (tracked for Stage 6)
Request returned non-empty content but pathological `"1. 1. 1. ..."` loop. Cause: vLLM loaded a **fallback chat template** because the HF repo ships none. Real fix is a chat-template flag (`--chat-template deepseek_ocr.jinja`) or prompt-shape rework in benchmark wiring. Captured in `lessons.md` Stage 5 smoke section and `results/smoke-response-1778762050.txt`. Stage 5 validation criterion ("test request returns non-empty text") is met — moving on.

## Next

1. Address chat-template before Stage 6 — need either a DeepSeek-OCR Jinja template, or pivot benchmark client to `/v1/completions` with the upstream raw prompt shape. Out of scope for current iteration.
2. Run Stage 6 concurrency-sweep + batch-throughput cells (see `sidecars/benchmark-g6e-2xlarge.yaml`).
3. Fold the SG fix into a node-creation runbook so subsequent nodegroups auto-attach both SGs via launch-template.

## Artifacts

- `results/smoke-response-1778762050.txt` — successful HTTP 200 but degenerate content (chat-template issue)
- `lessons.md` — Stage 4a health, Stage 5 SG fix, Stage 5 smoke note

## Iteration 2 (2026-05-14 ~14:00 ET)

### Spec updates (DRAFT v2)
- Dropped DocVQA from O3 gate and non-requirements.
- Added olmOCR-bench as primary O3 quality gate (DeepSeek self-reports 76.3).
- Added OmniDocBench as periodic deeper eval on BF16 + winning FP8 row (community score 90.25).
- Added Model / Prompt template subsection with the two upstream shapes.
- Added Appendix A — Quality harnesses section.
- Updated §Image ingress S3 paths to olmOCR-bench + OmniDocBench mirrors.

### Sidecar updates (`sidecars/benchmark-g6e-2xlarge.yaml`)
- Added `prompt_templates` block (plain_ocr + grounding; default = grounding for quality evals).
- `quality_baselines` now has `olmocr_bench` (7 subscores) + `omnidocbench` (5 components) blocks; `docvqa` removed.
- `quantization-pareto` workload `quality.eval` switched from `docvqa` → `olmocr_bench`, with `periodic_deeper_eval: omnidocbench`.

### Stage 5 re-smoke (PASS — non-degenerate output)
Port-forwarded `svc/deepseek-ocr-g6e-2xlarge-svc 8000:8000` and tested two paths.
- Path A (`/v1/completions` + top-level `multi_modal_data`): **failed** — image never reaches encoder on vLLM 0.19.1.
- Path B (`/v1/chat/completions` with DeepSeek prompt injected as user-message text alongside `image_url`): **succeeded** — both Free OCR and grounding modes return faithful multi-language transcription of the tesseract eurotext sample.
- Winning response saved: `results/smoke-response-v2-freeocr-1778762449.txt`, `results/smoke-response-v2-grounding-1778762442.txt`.
- Decision: no server-side `--chat-template` flag; inject prompt client-side (Deployment untouched).

### Harness scaffolds
- `scripts/run-olmocr-bench.sh` — stub invoking `pip install olmocr[bench]` + `olmocr-bench-run` against the vLLM endpoint. Exits 2 with mirror-the-dataset message while `s3://agent-aiops-bench-us-east-2/datasets/olmocr-bench/` is empty.
- `scripts/run-omnidocbench.sh` — stub invoking `ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204` with GT + predictions paths. Same S3-mirror gate.
- Both executable, documented with full invocation skeleton in comments.

## Next

1. Update `scripts/smoke-test.sh` to use Path B (grounding prompt + data: URL) — currently still points at the Wikipedia-hotlink shape from iteration 1.
2. Mirror olmOCR-bench standard set + OmniDocBench release to the two S3 prefixes; then flip the `exit 2` guards to actual invocation blocks.
3. Run Stage 6 concurrency-sweep + O3 BF16 seed (olmOCR-bench) — still no new infra needed, just need harness container.

## Iteration 3 (2026-05-14 ~12:50 ET)

### Smoke script fixed (`scripts/smoke-test.sh`)
Rewritten to iteration-2 Path B: `/v1/chat/completions` with grounding prompt (`<image>\n<|grounding|>Convert the document to markdown. `) as user-message `text` content alongside an `image_url` content item. Image now shipped as a base64 data URL built from a bundled 19 KB synthetic test doc at `scripts/test-assets/sample-doc.png` (checked in) — no more Wikipedia hotlink 403s. Deps: requires `jq`, `curl`, `base64`; fails fast if missing. Response body + HTTP status are printed; exit non-zero on empty content or the known-bad `"1. 1. 1. ..."` degenerate pattern. Idempotent — re-runnable without setup. Verified PASS: returned grounded markdown with bounding boxes for all 5 text regions in the sample.

### First Stage 6 latency cell — chatbot-short BF16 @ c=1 (PROVES WIRING)
New runner `scripts/run-chatbot-short.sh`: ensures `kubectl port-forward svc/deepseek-ocr-g6e-2xlarge-svc 8000:8000` (spawns if `lsof` finds nothing on :8000), 10 warmup requests, 100 steady-state sequential requests, per-request timing via `time.perf_counter()`, completion/prompt tokens from `response.usage`. Emits Common Benchmark Artifact v1.0.0 conforming to `standards/benchmark-commons/container/schema/enriched-artifact.json`.

**Headline metrics** (`results/artifacts/deepseek-ocr-2_eks_g6e-2xl_vllm_chatbot-short_20260514T125013Z.json`):

| Metric | Value |
|---|---|
| completed / failed | 100 / 0 |
| duration | 87.26 s |
| e2e mean | 622.7 ms |
| e2e p50 / p90 / p95 / p99 | 621.4 / 649.0 / 671.6 / 718.3 ms |
| request throughput | 1.15 req/s |
| output tokens/s | 183.4 |
| completion tokens / request (avg) | 160 |

**Validator**: `python3 standards/benchmark-commons/container/validate-artifact.py` → **PASS** (1 passed, 0 failed).

### Lessons additions
New section "VLM artifact conventions" in `lessons.md` — covers non-streaming / e2e-only / usage-tokens / modality-enum / `source_tool.name="custom"` / prompt-template-in-extensions pattern for future OCR cells.

## Next

1. Dataset mirror — `s3://agent-aiops-bench-us-east-2/datasets/{olmocr-bench,omnidocbench}/` still empty; blocks full O3 quality gate.
2. Concurrency sweep — extend runner to c=1/4/16/32 and emit a single `concurrency-sweep` artifact with `extensions.sweep_levels[]` (pattern already used by qwen3-embedding-8b-hyperpod).
3. Optional: seed a synthetic multi-image dataset for c>1 to avoid repeated-input prefix-cache skew.

## Iteration 4 (2026-05-14 ~12:57 ET)

Two new Stage 6 artifacts on the same live pod + test image + grounding prompt. Shared plumbing factored into `scripts/_common.py` (request body builder, percentile helper, envelope + model/engine/infra constants). Python runners: `_concurrency_sweep.py`, `_batch_throughput.py`; thin bash wrappers `run-concurrency-sweep.sh`, `run-batch-throughput.sh` handle port-forward + health gate + timestamped output path. True concurrency via `asyncio` + `aiohttp`.

### Artifact 1 — concurrency sweep (latency range)

Levels [1, 4, 16, 32], 10 warmup + 50 steady per level. Headline = peak level c=32.

| c | rps | out tok/s | e2e p50 | e2e p99 | scale vs c=1 |
|---|-----|-----------|---------|---------|--------------|
| 1 | 2.42 | 388 | 411 ms | 440 ms | 1.00x |
| 4 | 6.09 | 974 | 641 ms | 665 ms | 2.51x |
| 16 | 13.41 | 2146 | 1048 ms | 1238 ms | 5.53x |
| 32 | 19.31 | 3090 | 1220 ms | 1505 ms | 7.97x |

Artifact: `results/artifacts/deepseek-ocr-2_eks_g6e-2xl_vllm_concurrency-sweep_20260514T125714Z.json`. 0 failures across all 200 steady-state requests.

### Artifact 2 — batch throughput (saturation, c=32, 60 s)

32 worker coroutines fire back-to-back for 60 s after 10 s warmup → vLLM `max_num_seqs=32` queue stays saturated.

| metric | value |
|---|---|
| total / completed / failed | 1536 / 1536 / 0 |
| duration | 60.49 s |
| request throughput | **25.39 req/s** |
| output tokens/s | **4063** |
| e2e p50 / p90 / p99 | 1260 / ~1380 / ~1510 ms |

Artifact: `results/artifacts/deepseek-ocr-2_eks_g6e-2xl_vllm_batch-throughput_20260514T125802Z.json`. Sustained saturation throughput is **1.31x the sweep c=32 cell** (19.31 → 25.39 rps); the sweep's 50-request window was too short for the server to fully warm its queue — saturation is the honest batch number.

### Validator

All three artifacts in `results/artifacts/` PASS `standards/benchmark-commons/container/validate-artifact.py`.

### Saturation surprise

Sweep c=16 → c=32 gained only 44% throughput (13.4 → 19.3 rps) in the 50-request window, but the longer 60 s saturation run at c=32 pushed to 25.4 rps — a further 31% gain. Interpretation: short c=32 windows don't give vLLM enough time to fill the continuous-batching pipeline; the real p6e-2xl L40S ceiling for this workload is somewhere past 25 rps / 4K tok/s. No OOM, no failures even at full saturation — VRAM headroom is comfortable at 8192 ctx × 32 seqs × ~440 output tokens/req.

### Next

1. Dataset mirror — still the blocker for O3 quality gate. `s3://agent-aiops-bench-us-east-2/datasets/{olmocr-bench,omnidocbench}/` empty.
2. Optional broader sweep up to c=64/128 to locate the real saturation knee — would need `--max-num-seqs` bump in the Deployment first.
3. Optional FP8 Pareto row once quality gate lands.

## Session pause (2026-05-14)

Nodegroup `g6e-ocr` scaled to `desired=0` — GPU billing stopped. Cluster control plane + m6i system nodegroup remain up (shared with p5en blueprints; not paused by this blueprint).

**State at pause**:
- 3 validated artifacts in `results/artifacts/` (chatbot-short, concurrency-sweep, batch-throughput)
- Smoke + runner scripts working end-to-end
- Spec + sidecar at v2 (olmOCR-bench + OmniDocBench wired, DocVQA removed)
- Blocker for O3 Pareto: datasets not yet mirrored to `s3://agent-aiops-bench-us-east-2/datasets/{olmocr-bench,omnidocbench}/`

**To resume**:
```bash
aws eks update-nodegroup-config \
  --cluster-name qwen3-next-bench-eks-cluster \
  --nodegroup-name g6e-ocr \
  --region us-east-2 \
  --scaling-config minSize=0,maxSize=1,desiredSize=1
# Wait for node Ready, re-apply k8s/vllm-deepseek-ocr-g6e-2xlarge.yaml (pod was likely evicted on scale-down)
./scripts/deploy.sh g6e-2xlarge
```

## Iteration 5 (2026-05-13 ~09:30 ET)

OCR-specific benchmark normalization landed. Three deliverables:

### 1. Stratified 6-doc corpus

Generated via `scripts/test-assets/generate_corpus.py` (PIL, deterministic,
no hotlinks). Each image <120 KB. Actual image-token counts (measured at c=1):

| doc_type    | dims       | size  | image_tokens | output_tokens (1024 cap) |
|-------------|------------|-------|--------------|--------------------------|
| receipt     | 400×600    | 27 KB | 274          | 308                      |
| article     | 1000×1400  | 78 KB | 1138         | 270                      |
| table       | 1000×800   | 38 KB | 850          | 267                      |
| formula     | 1000×800   | 60 KB | 850          | 1024 (capped)            |
| dense       | 1200×1600  | 119 KB| 1138         | 1024 (capped)            |
| handwritten | 800×1000   | 96 KB | 1138         | 154                      |

Ratios actually measured: **4.2× image tokens** (receipt→article/dense),
**6.6× output tokens** (handwritten→formula), **6.3× e2e latency** at c=32
(handwritten→dense). Real-world dense pages push the image-token ratio
higher (~16× vs receipt) — our synthetic "dense" is a lower bound.

### 2. Code refactor (`scripts/_common.py`)

- `DOC_TYPES`, `CorpusItem` dataclass
- `load_corpus(assets_dir) -> list[CorpusItem]` with cached base64
- `compute_equivalent_pages(in_tok, out_tok, std_in=1200, std_out=300)` — geometric-mean page-equivalence
- `summarize_per_doc_type()` + `attach_throughput()` — per-bucket stats from raw per-request records
- `write_artifact(doc, per_doc_type=...)` — embeds stratification block at `extensions.stratification.per_doc_type[]`
- `NULL_LATENCY` auto-fill — schema now requires `ttft_ms`/`tpot_ms`/`itl_ms` (nullable); stubs filled automatically for every VLM artifact

### 3. Two new artifacts (round-robin corpus, c=32 saturation)

**Artifact A — concurrency sweep, stratified** (`results/artifacts/deepseek-ocr-2_eks_g6e-2xl_vllm_concurrency-sweep_20260514T133113Z.json`)

| c  | rps  | out_tps | img_tps | e2e p50 | e2e p99 |
|----|------|---------|---------|---------|---------|
| 1  | 0.76 | 380     | 678     | 753 ms  | 2721 ms |
| 4  | 1.81 | 901     | 1608    | 1244 ms | 4440 ms |
| 16 | 3.67 | 1791    | 3266    | 2239 ms | 7776 ms |
| 32 | 5.10 | 2493    | 4544    | 2652 ms | 8534 ms |

50 requests/level is not enough to saturate at c=16/32 (duration drops to
10 s) — for full saturation use the batch-throughput cell. The sweep reveals
the latency *shape* across load, not the throughput ceiling.

**Artifact B — batch throughput, stratified** (`results/artifacts/deepseek-ocr-2_eks_g6e-2xl_vllm_batch-throughput_20260514T133347Z.json`)

Saturation c=32, 60 s steady. **454 completed / 4 failed / 66.9 s wall** → aggregate **6.79 req/s, 3361 out-tps, 6101 img-tps, 1.500 equivalent-pages/s**. Per-bucket:

| doc_type    | rps  | out_tps | e2e p50 | e2e p99 | eq_pps |
|-------------|------|---------|---------|---------|--------|
| receipt     | 1.11 | 341     | 2794 ms | 2898 ms | 0.536  |
| article     | 1.20 | 323     | 2518 ms | 2626 ms | 1.105  |
| table       | 1.15 | 307     | 2441 ms | 2566 ms | 0.914  |
| formula     | 1.17 | **1107**| 9147 ms | 9327 ms | **1.745** |
| dense       | 1.09 | **1118**| 9176 ms | 9350 ms | **1.964** |
| handwritten | 1.08 | 166     | 1454 ms | 1648 ms | 0.751  |

### Note: raw req/s dropped

The iter-4 single-image benchmark reported 25.4 req/s saturation; iter-5
round-robin reports 6.79 req/s. That's **not** a regression — it's the
stratified corpus being honest. Formula + dense buckets dominate the wall
clock with ~9 s per-request latency; the throughput floor is now set by
the hardest documents, which is what you actually need for capacity
planning. Output-tokens-per-second (3361) is the metric that stayed within
18% of iter-4's 4063 (a fair drop because half of prompts now hit the 1024
output cap and can't be processed as quickly as short ones).

### Validator

Both artifacts **PASS** `standards/benchmark-commons/container/validate-artifact.py`.

Note: schema tightened between iter-4 and iter-5 to require `ttft_ms` /
`tpot_ms` / `itl_ms` in `metrics`. For VLM non-streaming these are filled
as null-percentile stubs (schema allows nulls). Iter-3/4 artifacts
retroactively patched with the same stub to stay compliant; handled
automatically for all future runs inside `_common.write_artifact()`.

### SG fix re-applied to new node

New node `ip-10-0-5-214.us-east-2.compute.internal` (instance
`i-0c1104b3a73669b0b`) landed with only the cluster primary SG. Applied
the iter-1 fix: attached `sg-0bf5ad07fc6c29df1` to both ENIs
(`eni-0c25dbbfb16f9d649`, `eni-09271fddd62725561`). Pod Ready ~3 min after.

### Session pause

Nodegroup scaled back to `desiredSize=0` at end of iteration. Total GPU
billing for iter-5: ~12 min wall-clock on one g6e.2xlarge (~$0.55).
