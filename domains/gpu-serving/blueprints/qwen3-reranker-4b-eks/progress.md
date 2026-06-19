# Qwen3-Reranker-4B on EKS g6e — Progress

## Status: SCAFFOLDED (2026-05-14)

## Substrate decision

Running on the existing `g6e-ocr` nodegroup (L40S 48GB, g6e.2xlarge) instead of the spec-preferred `g6.xlarge` (L4 24GB). Rationale:

- Reuses the running node — no nodegroup-add latency (~5 min) or separate billing cycle
- Model + KV fit easily on L40S (4B BF16 ~8 GB; L40S has 48 GB, ~6× headroom)
- Accepted cost: cost-efficiency row in the final report must note that L4 is spec-preferred and would be ~57% cheaper at ~75% of per-stream throughput. Measurement on L40S is a valid upper bound.

## Scaffolded

- `k8s/vllm-reranker-g6e-2xlarge.yaml` — Namespace + Deployment + Service; vLLM 0.19.1, `--task classify`, BF16, 4096 ctx, 64 max_num_seqs
- `sidecars/benchmark-g6e-2xlarge.yaml` — sidecar with latency/batch workload split + O3/O5/O9 cells; O4 MIG + O11 power deferred (MIG requires p5en, O11 needs DCGM)

## Known spec gaps (flagged but not blocking launch)

Per the next-model recommendation research, `domains/gpu-serving/specs/qwen3-reranker-4b.md` has these TBDs that we're resolving inline in the blueprint for this run:

- **Serving mode**: pinned to `--task classify` (vLLM cross-encoder mode)
- **O1 concurrency ladder**: 1/4/16/64/256 (reranker pairs scale higher than chat)
- **O1 pair-length axis**: 512 / 1024 / 2048 / 4096
- **O3 SLO**: p99 pair-score latency ≤ 50 ms at k=50
- **FiQA harness**: MTEB, task `FiQA2018`
- **O4 MIG**: **deferred** to later session (needs p5en nodegroup)

## Cluster state

- EKS: `qwen3-next-bench-eks-cluster` (us-east-2, ACTIVE, v1.32)
- Nodegroup: `g6e-ocr` (ACTIVE, 1× g6e.2xlarge, shared with deepseek-ocr but deepseek deployment is removed)
- Node SG fix already applied to current node's ENIs (both `eni-0e8aed8351584d0aa` and `eni-0d4dade610810f1b2` carry `sg-0bf5ad07fc6c29df1` alongside cluster SG)

## Iteration 1 deploy (2026-05-14)

### Stage 5 — serving ✓
- vLLM 0.19.1 launch flags required TWO fixes vs scaffold:
  1. `--task classify` → `--runner pooling --convert classify` (flag renamed in 0.19)
  2. Added `--hf-overrides` for `Qwen3ForSequenceClassification` + `classifier_from_token:["no","yes"]` — checkpoint has no `score.weight` tensor so vLLM must synthesize the head from lm_head rows
- API shape: `/v1/score` winning shape (request-ordered scores). `/v1/rerank` also works (Cohere-shape, sorted). Both validated in `scripts/smoke-test.sh`.
- Smoke response saved: `results/smoke-response-20260514T162745Z.txt`
- Semantic sanity: query "capital of France" → Paris docs 0.986/0.987, Berlin 0.469 ✓

### Stage 6 — validated artifact ✓
- Script: `scripts/run-concurrency-sweep.sh` + `scripts/_concurrency_sweep.py` + `scripts/_common.py`
- Levels [1, 4, 16, 64], k=50 candidates/req, pair_length_target=1024, 10 warmup + 50 steady/level
- Results (200/200 requests successful, zero failures):
  | c  | rps  | pairs/s | e2e p50 | e2e p99 |
  |----|------|---------|---------|---------|
  | 1  | 3.17 | 158.5   | 299 ms  | 476 ms  |
  | 4  | 7.72 | 386.1   | 500 ms  | 684 ms  |
  | 16 | 7.66 | 383.0   | 1974 ms | 2407 ms |
  | 64 | 7.66 | 383.0   | 3409 ms | 6472 ms |
- Peak throughput at c=4; c>=16 is pure queueing
- Artifact: `results/artifacts/qwen3-reranker-4b_eks_g6e-2xl_vllm_concurrency-sweep_20260514T163014Z.json`
- Validator: **PASS** (`standards/benchmark-commons/container/validate-artifact.py`)
- Substrate caveat embedded at `infrastructure.substrate_deviation` + `extensions.substrate_caveat`

### Deferred (out of scope for iteration 1)
- c=256 row (latency blow-up expected; needs batching strategy change)
- Pair-length axis sweep (512/1024/2048/4096)
- FP8/INT8 precision rows
- FiQA quality gate (harness not yet mirrored)
- O3 p99-per-pair SLO report: can be read off this artifact — at c=4, 13.7 ms/pair p99 (well under 50 ms target)

## Iteration 2 deploy (2026-05-14)

### Stage 6 — pair-length axis + extended concurrency ✓

Two validated artifacts added.

**Artifact A — pair-length sweep** (c=4 fixed, pair_length ∈ [512,1024,2048,4096]):

| pair_length | rps   | pairs/s | e2e p50 | e2e p99 |
|-------------|-------|---------|---------|---------|
| 512         | 13.99 | **700** | 270 ms  | 379 ms  |
| 1024        |  7.86 | 393     | 495 ms  | 673 ms  |
| 2048        |  4.09 | 204     | 953 ms  | 1275 ms |
| 4096        |  0.00 | 0       | fail    | fail    |

- **Peak pairs/s at L=512** (headline). pairs/s scales ~1/pair_length.
- L=4096 fails ALL 50 requests: template overhead pushes prompt to 4097 tokens vs `max_model_len=4096`. Captured as `metrics.failed=50` in artifact.
- Artifact: `results/artifacts/qwen3-reranker-4b_eks_g6e-2xl_vllm_pair-length-sweep_20260514T171508Z.json`
- Validator: **PASS**

**Artifact B — extended concurrency sweep** (pair_length=1024, levels [1,4,16,64,256]):

| c   | rps  | pairs/s | e2e p50 | e2e p99 | failed |
|-----|------|---------|---------|---------|--------|
| 1   | 3.33 | 167     | 292 ms  | 452 ms  | 0      |
| 4   | 7.78 | 389     | 497 ms  | 669 ms  | 0      |
| 16  | 7.60 | 380     | 1978 ms | 2465 ms | 0      |
| 64  | 7.31 | 366     | 3660 ms | 6772 ms | 0      |
| 256 | 7.78 | 389     | 3248 ms | 6355 ms | 0      |

- **Surprise: c=256 did NOT error**. Ties c=4 on pairs/s with lower p99 than c=64. Root cause: `n_steady=50` caps in-flight requests below the semaphore limit, so effective concurrency ≈ 50; vLLM's `max-num-seqs=64` absorbs the burst cleanly.
- Non-monotonic p99 at high c (c=256 p99 < c=64 p99): likely continuous-batching forms larger groups when admission is fully saturated.
- Artifact: `results/artifacts/qwen3-reranker-4b_eks_g6e-2xl_vllm_concurrency-sweep-extended_20260514T171607Z.json`
- Validator: **PASS**

### O1 matrix status
- concurrency axis [1,4,16,64,256] ✓
- pair-length axis [512,1024,2048,4096] ✓ (with 4096 failure documented)
- Spec's explicit O1 cells are now all covered on L40S substrate.
