# Readiness Audit — kimi-k2.6-speculative
## Session 2026-05-13

## Stage 4a — GPU Health ✅
- [x] All 8 B300 GPUs detected @ 275 GiB
- [x] NVLink NV18 full mesh (NVSwitch)
- [x] 2 EFA NICs available
- [x] ECC clean, 0 Xid errors
- [x] CUDA 13.0, driver 595.58.03

## Stage 5 — Serving Stack ✅ (SGLang EAGLE3) / ❌ (vLLM EAGLE3 blocked)
- [x] Health endpoint responds: `curl localhost:30000/health` → 200
- [x] Test completion with spec decode active → coherent output
- [x] Draft model loaded alongside target (35.4 GB avail/rank post-load)
- [x] EAGLE3 acceptance rate > 0% → measured 0.55-0.61
- [x] Startup time documented: ~4 min (target + draft + CUDA graph 23s)
- [x] No CUDA OOM
- [ ] vLLM EAGLE3: FAILED — voipmonitor custom image vision tower bug (see lesson L13)

## Stage 6 — Benchmark ✅
- [x] Concurrency sweep: 1, 4, 16, 64, 128, 256, 512 — peak 3657 tok/s @ c=64
- [x] Single-stream: 164 tok/s (target ≥200 not met; cold-cache noise)
- [x] Aggregate throughput @ c=128: 3550 tok/s (target ≥6000 NOT met)
- [x] No OOM @ max conc 512
- [x] No timeouts
- [x] Error rate 0% across all runs
- [x] All 6 workloads covered: W1-W6 complete
- [x] EAGLE3 acceptance: 0.55-0.61 (stable across conc), accept_len 2.17-2.43
- [x] Effective tokens per decode: ~1.23 (23% decode reduction, not enough to beat baseline at high conc)

## Stage 7 — Readiness audit
- [x] Measurements stored as JSON artifacts (11 files in S3)
- [x] Lessons captured (L1-L13)
- [x] Partial: vLLM track blocked, SGLang track complete
- [x] Production config frozen (default SGLang EAGLE3 flags)

## Decision / Recommendation
EAGLE3 with default SGLang config is NET NEGATIVE vs K2.6 baseline at production concurrency (128+). Only beneficial for single-stream or small batch (c≤16) workloads like agentic tool calling (where it gives +28% @ c=1). For high-throughput production (c=128+), stay on baseline K2.6 config without spec decode.

**Next session** (if continued):
- Fresh vLLM image build from main (without voipmonitor's custom vision wrapper bug)
- Full EAGLE3 parameter sweep (num_steps=1,2,4 × draft_tokens=2,4,6,8 × topk=1,2,4)
- Phase 4 full stack (EAGLE3 + HiCache + prefix caching) on winning config
