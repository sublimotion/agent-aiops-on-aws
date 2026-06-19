# Qwen3-235B Speculative Decode + Optimization Tier Benchmark

## Status: DRAFT (2026-05-14)

## Overview

Follow-up to `qwen3-235b-b300.md` (COMPLETE baseline — vLLM TP4 peak 11,820 tok/s @ c=512, vLLM TP2+DP4+EP peak 13,877 tok/s @ c=512). This spec mirrors `kimi-k2.6-speculative.md` methodology to evaluate speculative decode + HiCache + TP4+DP2 parallelism on Qwen3-235B, producing directly-comparable v1 envelope data across the two models.

**Why mirror Kimi's phase structure**:
- Same hardware (p6-b300.48xlarge, sm_103, NVSwitch NV18) — cross-model comparability
- Same observability stack (Prometheus + DCGM sidecars from Stage 4b) — complete TTFT/TPOT/DCGM data this time, not the Kimi-spec gap
- Same envelope format → vault-comparable via `results-vault/` dashboard

**Why SGLang-primary instead of vLLM**:
- First-party EAGLE3 draft `lmsys/Qwen3-235B-A22B-EAGLE3` only has SGLang launch recipe on its model card — vLLM path untested by the authors
- Avoids the Kimi Phase 2 vLLM custom-image blocker (L13)
- The existing vLLM baseline (TP4 / TP2+DP4+EP) stays as the no-spec reference curve; this session adds the SGLang+EAGLE3 optimization tier

**Baseline reference** (from `qwen3-235b-b300.md`):

| Engine | Config | Peak tok/s | Single-stream TTFT p50 | $/M output tokens |
|--------|--------|-----------|------------------------|--------------------|
| vLLM v0.19.1 | TP4, prefix cache | 11,820 @ c=512 | 42 ms (cold 168 ms) | $0.39 |
| vLLM v0.19.1 | TP2+DP4+EP | 13,877 @ c=512 | — (DP cold start 36× worse) | $0.33 est |

**Hypotheses this session tests**:

1. EAGLE3 with first-party `lmsys/Qwen3-235B-A22B-EAGLE3` draft gives 1.3–1.8× single-stream throughput (model card reports accept length 3.0–3.5 on GSM8K/MTBench — lower than Kimi's 5.0, so gains expected smaller)
2. `num_steps` sweep finds the optimal config — likely `num_steps=2` or `3` given accept-length ceiling
3. HiCache + EAGLE3 composes cleanly (same as Kimi Phase 4)
4. CUDA graphs dominate at low concurrency (confirming Kimi L19 generalizes beyond K2.6's MLA to GQA architectures)
5. TP4+DP2 with Qwen3-235B may not win as decisively as on Kimi — the model is smaller (235B vs 1T total) and fits TP4 comfortably, so TP8 isn't forced

**Non-goals**:
- No vLLM EAGLE3 comparison (no first-party draft path)
- No dynamic MLA/MHA routing (Qwen3 uses GQA, not MLA — this lever is inapplicable)
- No multi-node or disaggregation (same single-node scope as Kimi)

---

## Components

### 1. Compute

- **Instance**: p6-b300.48xlarge (8× B300 SXM6 AC, 275 GiB each, NVSwitch NV18)
- **Region / AZ**: **us-west-2b** (usw2-az2) — B300 spot confirmed at **$25.47/hr** (vs $25.65 in us-east-1c, same as Kimi session)
- **AMI**: AL2023 NVIDIA DLAMI (verify current id at deploy time — the Kimi session used `ami-027c3ae8019fc0d3a` in us-east-1; need us-west-2 equivalent)
- **NVMe**: 28 TB RAID0 at `/mnt/nvme`

### 1a. GPU & NCCL pre-flight

Standard Stage 4a — NCCL all_reduce must reach ~470 GB/s algo BW at 8 GB payload (matches Kimi measurement on identical HW).

### 1b. Observability (mandatory — Stage 4b)

Launch Prometheus + DCGM + node-exporter sidecars via `.claude/skills/benchmark-runner/scripts/bootstrap-observability.sh` BEFORE the serving stack starts. The Kimi-spec session lost all TTFT data to this — we do not repeat that mistake. Smoke test at `observability-smoke-test.sh` must pass before Stage 5.

### 2. Model

- **Target**: `Qwen/Qwen3-235B-A22B-Instruct-FP8` (235B total, 22B active, FP8 block-scaled, `max_position_embeddings=40960`)
- **Draft**: `lmsys/Qwen3-235B-A22B-EAGLE3` (~1B params, ~2 GB BF16)
- **Quantization**: FP8 (block_n=128; must satisfy `moe_intermediate_size=1536 / TP % 128 == 0` — TP4 gives 384 ✓, TP2 gives 768 ✓, TP8 gives 192 ✗)
- **Context length**: **40,960 tokens** (NOT 131,072 — FP8 variant lacks YaRN config per baseline lessons L#3)

### 3. Software matrix

| Track | Engine | Version | Config | Key flags |
|-------|--------|---------|--------|-----------|
| A | SGLang | v0.5.10-cu130 | EAGLE3 spec decode, defaults | `--speculative-algorithm=EAGLE3 --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-eagle-topk 1 --speculative-draft-model-path lmsys/Qwen3-235B-A22B-EAGLE3 --speculative-draft-attention-backend trtllm_mha` |
| B | SGLang | v0.5.10-cu130 | EAGLE3 sweep | Same + `num_steps ∈ {1,2,3,4}`, `num_draft ∈ {2,4,6,8}`, `topk=1` |
| C | SGLang | v0.5.10-cu130 | Phase 1b winner + HiCache | `--enable-hierarchical-cache --hicache-size 200` |
| D | SGLang | v0.5.10-cu130 | Phase 5a/b/c frontier variants | default-stack / `--disable-cuda-graph` / `--tp 4 --dp 2` |

### 4. Storage

- Target model `Qwen/Qwen3-235B-A22B-Instruct-FP8`: ~235 GB. Stage via direct HF download on the GPU node (3200 Gbps network; per Kimi L15, export `HF_TOKEN` explicitly + `HF_HUB_ENABLE_HF_TRANSFER=1`).
- Draft: ~2 GB, downloads in <1 min.
- Total staging: ~4–8 min at 40–60 GB/min (much faster than Kimi's 594 GB).

---

## Experiment Protocol

### Phase 0 — Roofline

**Goal**: Confirm hardware ceiling matches Kimi measurement (473 GB/s NCCL algo BW, 836 GB/s bus BW, DeepGEMM FP8 at MoE shapes).

Deliverables:
- [ ] NCCL all_reduce 8 GPU matches Kimi baseline within 5%
- [ ] DeepGEMM FP8 TFLOPS at Qwen3 MoE shapes (M=128, 512, 2048, 8192)
- [ ] Topology confirmed
- [ ] Arithmetic-intensity ceiling derived for Qwen3-235B (22B active × 0.5 B/param = 11 GB/token weight read, smaller than Kimi's 16 GB; implies higher BW ceiling)

### Phase 1 — SGLang EAGLE3 defaults

Run the draft's recommended config and capture the default baseline.

```bash
SGLANG_ENABLE_SPEC_V2=1 python3 -m sglang.launch_server \
  --model-path /mnt/nvme/models/qwen3-235b-fp8 \
  --tp 4 \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1 \
  --speculative-algorithm EAGLE3 \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 \
  --speculative-draft-model-path /mnt/nvme/models/qwen3-235b-eagle3 \
  --speculative-draft-attention-backend trtllm_mha \
  --trust-remote-code \
  --host 0.0.0.0 --port 30000
```

**Measurements**: concurrency sweep 1, 8, 32, 64, 128, 256, 512 using `bench-standard.py` (Prometheus-first).

### Phase 1b — Hyperparameter sweep

Mirror Kimi's pruned sweep (13 configs after `num_steps > num_draft_tokens` exclusion).

**Expected from model card**: accept length plateaus near 3.0–3.5, so `num_steps=2` likely matches `num_steps=3` in effective throughput while using less compute per decode step. `num_steps=4` unlikely to help (accept length won't reach 5).

Deliverable: `WINNER.env` with best `num_steps/num_draft/topk` for use in Phase 4/5.

### Phase 4 — Fullstack

Winner config + HiCache 200 GB/rank. Same protocol as Kimi Phase 4.

### Phase 5 — Frontier

- **5a**: default stack (CUDA graphs + overlap scheduler on, HiCache on). Reruns Phase 4 with cleaner isolation labels.
- **5b**: `--disable-cuda-graph` ablation. Validates L19 generalizes from MLA-MoE (Kimi) to GQA-MoE (Qwen3).
- **5c**: TP4+DP2 replicas. Expected win smaller than Kimi's +14% at c=256 because Qwen3-235B TP4 has more VRAM headroom than Kimi TP8.
- **5d**: FP4 kernel probe (same deferral — cutlass 3.x kernels not shipped in SGLang 0.5.10).

---

## Metrics

### Primary (must beat vLLM baseline to declare success)

| Metric | Baseline (vLLM TP4) | Target |
|--------|---------------------|--------|
| Single-stream tok/s @ c=1 | 103 tok/s | **≥150** (1.5× = EAGLE3 minimum) |
| Aggregate tok/s @ c=128 | ~5,000 (derived) | **≥6,000** |
| Aggregate tok/s @ c=256 | ~7,500 (derived) | **≥8,500** |
| $/M output tokens @ peak | $0.39 | **≤$0.35** |

### Secondary

- EAGLE3 acceptance rate (expect 0.7–0.8 vs Kimi's 1.0)
- Accept length (expect 3.0–3.5 per model card vs Kimi's 5.0)
- Crossover concurrency — where spec decode stops helping (likely lower than Kimi)

---

## Workloads

Mirror Kimi methodology. Phase 0/1/1b/4/5 all use `concurrency-sweep` workload card (2048 in / 512 out, levels 1/8/32/64/128/256/512). Optional W1–W6 rerun at winning config deferred to future session.

---

## Success criteria

1. **Roofline confirmed**: NCCL BW matches Kimi within 5%
2. **EAGLE3 validated**: acceptance rate and length match model card claims within 20%
3. **Single-stream ≥ 1.5×**: 150+ tok/s (from 103 baseline)
4. **Fullstack ≥ baseline at c≥128**: no regression from prefix-cache-only vLLM baseline
5. **No accuracy regression**: output quality on 20 coding prompts matches baseline
6. **CUDA graphs 5–7× at c=1** (validates L19 generalizes)
7. **Full envelope data** — TTFT/TPOT/ITL/E2E/DCGM all populated (no Kimi-style gaps)
8. **Vault-comparable artifacts**: 60+ new v1 envelopes land in `domains/gpu-serving/results-vault/`

## Termination conditions

- **Success**: ≥4 of 8 success criteria met
- **Hard stop**: 12 hours GPU time
- **Blocker**: SGLang EAGLE3 incompatible with Qwen3-235B FP8 path — fall back to vLLM ngram speculative as plan B
- **Budget ceiling**: $350

## Risk register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Draft `lmsys/Qwen3-235B-A22B-EAGLE3` not compatible with FP8 target | Low | High | LMSYS trained against the same base; FP8 quantization is target-only. If fails, try `RedHatAI/Qwen3-235B-A22B-speculator.eagle3` as fallback draft |
| FP8 block-size constraint forces TP4 only | Low | Medium | Baseline already validated; TP4 is the plan |
| EAGLE3 + TP4 deadlock at high concurrency | Medium | Medium | Same mitigation as Kimi: clear FlashInfer cubin cache before each config (L16), disable CUDA graphs if needed |
| Draft model download fails (HF auth / rate limit) | Low | Low | Same HF_TOKEN + hf_transfer pattern as Kimi L15 |
| us-west-2 spot reclaim | Medium | Medium | Checkpoint to S3 every 10 min (standard); resume from last checkpoint |
| Accept rate below model card claims | Medium | Low | Document and report; still publish artifacts — the comparison data has value even at lower acceptance |

## Verification criteria

### Stage 4a — GPU health
- [ ] All 8 GPUs ECC clean, no pending row remaps
- [ ] NCCL all_reduce passes at TP4 and TP8
- [ ] NVSwitch topology confirmed

### Stage 4b — Observability
- [ ] Prometheus + DCGM + node-exporter running
- [ ] Smoke test passes (engine histograms detected post-serving-start)
- [ ] S3 snapshot timer firing

### Stage 5 — Serving
- [ ] SGLang health `200` with EAGLE3 config
- [ ] Draft model loads without OOM
- [ ] Engine histograms visible in `curl /metrics`

### Stage 6 — Benchmarks
Follows Kimi Phase 1b/4/5 protocol. `bench-standard.py` emits v1 envelopes directly. Minimum:
- [ ] Concurrency sweep @ defaults (7 envelopes)
- [ ] 13 sweep configs × 5 concurrency levels = 65 envelopes (Phase 1b)
- [ ] Phase 4 fullstack sweep (7 envelopes)
- [ ] Phase 5a/b/c @ 6 concurrency levels each (18 envelopes)
- [ ] Total: 97 envelopes in `results/standard/`

### Stage 7 — Readiness audit
- [ ] All envelopes schema-validate
- [ ] Reconciliation field non-null; `reconciled=true` for >95% of runs
- [ ] Comparison report vs Kimi + Qwen3-235B baseline

---

## Estimated Cost

| Phase | Duration | Cost (spot $25.47/hr) |
|-------|----------|------------------------|
| Weight staging | ~15 min | $6.40 |
| Phase 0 roofline | ~20 min | $8.50 |
| Phase 1 defaults | ~45 min | $19 |
| Phase 1b sweep (13 configs) | ~2.5 hrs | $64 |
| Phase 4 fullstack | ~1 hr | $25 |
| Phase 5 frontier (5a/b/c/d) | ~2 hrs | $51 |
| Buffer + teardown | ~30 min | $13 |
| **Total** | **~7 hrs** | **~$187** |

Well under the $350 ceiling.

## Known limitations

1. **vLLM EAGLE3 untested** — no first-party draft recipe, no v1 envelope data for vLLM + spec decode on Qwen3-235B
2. **FP4 kernels unavailable** in SGLang 0.5.10 — Phase 5d remains profile-only
3. **Qwen3-235B-A22B-Instruct-2507 variant NOT used** — the `lmsys/Qwen3-235B-A22B-EAGLE3` draft was trained against the older `Qwen3-235B-A22B` base; if the 2507 target is needed, use `zhuyksir/EAGLE3-Qwen3-235B-A22B-Instruct-2507-FP8` (0.6B) or `nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507` (0.6B) instead

## Relationship to other specs

| Spec | Relationship |
|------|--------------|
| `qwen3-235b-b300.md` (COMPLETE) | Direct baseline — vLLM TP4 / TP2+DP4+EP reference curves |
| `kimi-k2.6-speculative.md` (COMPLETE) | Methodology template — same phase structure, same hardware, same observability mandate |
| `docs/inference-optimization-guide.md` | Results feed back into MTP section with Qwen3-specific accept-rate data |
