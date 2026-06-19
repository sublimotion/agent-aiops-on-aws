# Kimi K2.6 × vLLM × LMCache — Compatibility Smoke Test

## Status: DRAFT (2026-05-13)

## Overview

Settle empirically whether **vLLM + LMCache + Kimi K2.6 (MLA single-group + NSA)** works on recent `dev` branches, before any HyperPod Inference Operator benchmark that would depend on this combination.

**Why this matters:**

The repo currently carries conflicting guidance:
- Old memory entry claimed "LMCache + NSA BLOCKED on v0.3.15, fix in PR #2629" — **incorrect** (PR #2629 is an SGLang-connector MLA fix, not an NSA fix).
- Corrected status (checked against LMCache `dev` on 2026-05-13):
  - **SGLang path**: LMCache `SGLangLayerwiseGPUConnector` lacks MLA support. PR #2629 OPEN with changes requested. Issue #3192 tracks the gap. Blocked.
  - **vLLM path**: PR #2951 (OPEN, targets `dev`) fixes MLA *multi-group* KV layout for GLM-5 (K_rope uint8 hd=132 + latent KV bf16 hd=576) and DeepSeek V3. **PR #2951's own description states K2.5/K2.6 use a single KV group** — the easy case. Likely already works on recent `dev`. Not validated in this repo.

The HyperPod Inference Operator's `kvCacheSpec` L1/L2 features only engage with LMCache-aware images. If vLLM+LMCache+K2.6 works, the operator's tiered-KV surface is usable for K2.6 agentic workloads; if it doesn't, K2.6 stays on SGLang HiCache and the operator's L2 adds nothing to this model. A ~1-hour smoke test resolves this ambiguity before anyone scopes a larger HyperPod+K2.6 benchmark.

**Scope:** This is explicitly a compatibility smoke, not a performance study. Pass/fail on correctness and basic KV reuse. If it passes, a follow-on spec can measure tiered-KV economics.

---

## Components

### 1. Compute

Reuse the existing K2.6 baseline node — no new capacity needed:

- **Platform**: EKS on EC2 (spot), `qn-sglang-eks-cluster` (us-west-2b)
- **GPU Node**: p6-b300.48xlarge (8× B300, NVSwitch, sm_103, AL2023 NVIDIA AMI)
- **Container runtime**: `nerdctl` (g7e convention; confirm on B300)
- **NVMe**: existing `/mnt/nvme` with K2.6 weights already staged from baseline

### 1a. GPU Pre-Flight

Existing baseline node has validated GPU health. Skip full Stage 4a; confirm only:

- [ ] No Xid errors in `dmesg` since last benchmark
- [ ] `nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader` returns all zeros
- [ ] K2.6 weights still present at `/mnt/nvme/models/kimi-k26-fp8/`

### 2. Model

- **Model**: `moonshotai/Kimi-K2.6` (1T MoE, 32B active, INT4 QAT, single-group MLA + NSA)
- **Path**: `/mnt/nvme/models/kimi-k26-fp8/` (reused from baseline — 594 GB on disk)
- **Context length**: 32768 for smoke (baseline is 131072; shorter context is sufficient and speeds pod restart)

### 3. Software

**Two images to build** (both OCI → local registry or ECR):

| Image | Base | Pin |
|---|---|---|
| `vllm-lmcache:k26-smoke` | `vllm/vllm-openai:dev` or the `voipmonitor/vllm:cu130-mtp-tuned-v3-20260423` image already used for K2.6 spec decode | git SHA from vLLM `main` on 2026-05-13 |
| LMCache install | `pip install git+https://github.com/LMCache/LMCache@<sha>` inside the image | git SHA from LMCache `dev` on 2026-05-13 |

**Critical version pinning**: both projects are on `dev` branches that move daily. Record SHAs in `lessons.md`. Do **not** use `:latest` or `:dev` floating tags — if the smoke fails, we need reproducibility for upstream issue reporting.

### 4. Networking

- Direct pod access via `kubectl port-forward svc/vllm-lmcache-smoke 8000:8000`. No ALB, no llm-d.

### 5. Storage

- Model weights: `/mnt/nvme/models/kimi-k26-fp8/` (hostPath)
- LMCache local storage: `/mnt/nvme/lmcache/` (hostPath) — backs L1 CPU cache and optional disk tier
- Results: `blueprints/kimi-k2.6-lmcache-smoke/results/`

---

## Test Protocol

### Phase 0 — Image build + basic load (15 min)

**Gate**: Pod reaches `Ready` and `/v1/models` lists `kimi-k2.6`.

```bash
# Minimal vLLM command, LMCache enabled via KVConnector
vllm serve /mnt/nvme/models/kimi-k26-fp8 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --kv-transfer-config '{
    "kv_connector": "LMCacheConnectorV1",
    "kv_role": "kv_both"
  }' \
  --host 0.0.0.0 --port 8000
```

LMCache config via env vars:
```
LMCACHE_CONFIG_FILE=/etc/lmcache/config.yaml
```

Minimal `config.yaml`:
```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 50   # GB — small on purpose
local_disk: null         # disable disk tier for smoke; re-enable in phase 2
remote_url: null
remote_serde: null
```

**Failure modes to flag explicitly:**
- ❌ `AttributeError: 'MLATokenToKVPool' object has no attribute 'k_buffer'` → hard block, matches old SGLang-side failure pattern
- ❌ Shape mismatch in LMCache layerwise save/restore → likely PR #2951 territory; capture full traceback
- ❌ Hang during `cache_engine.store()` → thread/deadlock, capture pyflame or py-spy
- ✅ Pod ready, `/v1/completions` returns valid text → proceed to Phase 1

### Phase 1 — KV reuse correctness (20 min)

**Gate**: Prefix cache hits measurable; output identical with and without LMCache.

10-prompt smoke set (keep small and reproducible):

```python
prompts = [
    # Shared 1K-token system prompt + varied user turn (exercises prefix reuse)
    {"system": SYSTEM_PROMPT_1K, "user": user_i} for user_i in USERS_10
]
```

Procedure:
1. Send 10 prompts sequentially at temperature=0, max_tokens=256
2. Resend the *same* 10 prompts (LMCache should serve from stored KV)
3. Check `lmcache:num_hits` / `lmcache:num_misses` Prometheus counters
4. Diff outputs pass-1 vs pass-2 (byte-identical at temp=0)

**Pass criteria**:
- [ ] Pass 2 `lmcache:num_hits` > 0 (KV actually restored)
- [ ] Pass 1 vs Pass 2 outputs byte-identical
- [ ] No crashes or CUDA illegal memory access across 20 total requests

### Phase 2 — Cross-restart KV persistence (15 min, optional)

**Gate**: Demonstrates L2 (disk) persistence — the feature that motivates using LMCache at all.

1. Re-run Phase 1 with `local_disk: /mnt/nvme/lmcache/` and `max_local_cpu_size: 5` (force disk spill)
2. Kill the pod via `kubectl delete pod`
3. Wait for replacement pod to become Ready
4. Re-send 10 prompts; check that some hits come from L2 (disk) rather than cold miss

**Pass criteria**:
- [ ] `lmcache:num_hits_from_disk` > 0 after restart
- [ ] Outputs still byte-identical to Phase 1 pass 1

### Phase 3 — Light concurrency smoke (10 min, optional)

Not a perf test — just confirm no race conditions under modest load.

- 16 concurrent clients × same 10 prompts × 2 rounds
- Check: no exceptions in vLLM logs, no CUDA errors, LMCache counters monotonic

---

## Metrics

### Primary (pass/fail gates)

| Metric | Gate |
|---|---|
| Pod reaches Ready with LMCache connector attached | Required |
| `lmcache:num_hits` > 0 on pass 2 | Required |
| Pass 1 vs Pass 2 outputs byte-identical at temp=0 | Required |
| 0 crashes across 20+ requests | Required |
| `lmcache:num_hits_from_disk` > 0 after pod restart (Phase 2) | Nice-to-have |

### Secondary (characterization, not gates)

- Store / restore latency per prompt (`lmcache:put_duration_seconds`, `lmcache:get_duration_seconds`)
- Chunk store size distribution (sanity-check on K2.6's compressed MLA KV)
- GPU memory delta with LMCache connector vs baseline (quantifies connector overhead)

---

## Success Criteria

This is a compatibility smoke, not a benchmark. Success is **binary**:

- **PASS**: All Phase 0 + Phase 1 gates green. Document LMCache SHA, vLLM SHA, config, and one-line status update for the memory file. Unlocks scoping of a real vLLM+LMCache+K2.6 benchmark on HyperPod.
- **PARTIAL**: Phase 0 passes but Phase 1 fails with reproducible error. File issue upstream, capture in `lessons.md`, flag as blocked pending upstream fix.
- **FAIL**: Phase 0 fails (hard incompatibility). Confirms LMCache + K2.6 is blocked on vLLM today; memory updated to reflect empirical result; `kimi-k2.6-hyperpod` style specs should plan around SGLang HiCache only.

## Termination Conditions

- **Hard stop**: 1.5 hours of GPU time (smoke should finish in ~1 hr; extra 0.5 hr for image rebuild if traceback points to fixable config).
- **Abort**: Two consecutive Phase 0 failures on different builds → file upstream issue, stop.

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| LMCache `dev` build breaks against vLLM image at the same date | Medium | Low | Pin both SHAs from the same day; fall back to the most recent tagged release of each |
| K2.6 single-group MLA hits an edge case PR #2951 didn't anticipate | Medium | Low | Full traceback → upstream issue; this is exactly what the smoke exists to discover |
| NSA sparsity interferes with LMCache chunked save | Low | Medium | Check chunk boundaries align with attention block boundaries; reduce `chunk_size` if not |
| B300 OOM with LMCache connector overhead | Low | Low | Phase 0 uses small `max_local_cpu_size: 50` GB and disables disk; leaves GPU untouched |

## Verification Criteria

### Stage 4a — GPU Health

- [ ] Reuse baseline health (see kimi-k2.6 blueprint); spot-check Xid + ECC only

### Stage 5 — Serving Stack

- [ ] Pod reaches `Ready` within 20 min of apply
- [ ] `curl localhost:8000/health` returns 200
- [ ] `/v1/models` lists `kimi-k2.6`
- [ ] vLLM startup logs show `LMCacheConnectorV1` attached successfully
- [ ] Container logs clean of CUDA illegal memory, shape mismatch, or LMCache tracebacks

### Stage 6 — Smoke

See Test Protocol above; this spec deliberately replaces the full benchmark stage with the 3-phase compatibility test.

### Stage 7 — Readiness Audit

- [ ] Outcome recorded (PASS / PARTIAL / FAIL) with exact vLLM + LMCache SHAs
- [ ] If PASS: unblock downstream specs — update `kimi-k2.6.md` KV cache table row for LMCache to "VALIDATED (vLLM, {SHA})"
- [ ] If FAIL: capture tracebacks, file upstream issue, update memory with empirical result
- [ ] `lessons.md` written with: image build steps, exact config, result, links to any upstream issues filed

---

## Non-Requirements

- **Not a performance benchmark** — no TTFT/throughput SLOs, no QPS sweep, no workload catalog runs
- **Not a HyperPod deployment** — runs on the existing vanilla EKS baseline; a successful smoke motivates a separate HyperPod spec later
- **Not SGLang + LMCache** — known blocked on PR #2629 / issue #3192; out of scope
- **Not multi-node** — single p6-b300 node
- **Not testing tool calling or agentic workloads** — deferred to follow-on perf benchmark if smoke passes

## Cost Estimate

| Item | Duration | Cost |
|---|---|---|
| p6-b300.48xlarge spot | ~1.5 hr | ~$24 |
| Image build (CodeBuild or local) | ~20 min | <$1 |
| **Total** | | **~$25** |

## Known Limitations

1. **`dev` branch churn**: Both vLLM and LMCache `dev` change daily. Pin SHAs; expect this smoke to go stale within ~2 weeks.
2. **Single-group MLA assumption**: PR #2951's description states K2.6 is single-group; if this assumption is wrong for the INT4 QAT variant specifically, the smoke will catch it but won't fix it.
3. **NSA + LMCache interaction untested upstream**: K2.6's NSA is a sparsity pattern applied post-KV-storage; should be transparent to LMCache, but no upstream test coverage confirms this.

## Relationship to Other Specs

| Spec | Relationship |
|---|---|
| `kimi-k2.6.md` (COMPLETE) | Shares the baseline infrastructure and model weights; this smoke adds the LMCache row to that spec's KV cache strategy table |
| `kimi-k2.6-speculative.md` (DRAFT) | Independent — that spec targets EAGLE3 + prefix caching, no LMCache dependency |
| `glm5-hyperpod.md` (DRAFT) | A PASS here removes the "kvCacheSpec untested with non-SGLang engines" limitation note; a FAIL confirms SGLang HiCache is the only viable path on HyperPod for MLA models |
| `glm5-lmcache.md` (historical) | Documented SGLang + LMCache failure on GLM-5 (different model, different connector); provides failure-mode reference patterns |

---

> **Note**: Operational artifacts (lessons, SHAs, tracebacks, upstream issue links)
> belong in the blueprint directory: `blueprints/kimi-k2.6-lmcache-smoke/lessons.md`.
