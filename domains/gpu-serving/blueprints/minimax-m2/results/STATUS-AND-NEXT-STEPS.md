# MiniMax-M2 Serving — Engagement Status & Next Steps (paused 2026-06-29)

Internal engineering status. Frames what is **proven**, what is **blocked**, and the **exact next step**
to unblock. No customer references.

## TL;DR

We have a **defensible B200 benchmark** (59 points) and an **architecture decision framework**, but the
**full follow-on sweeps (distinct-prefix KV-tiering, DP/TP parallelism, H200 like-for-like) are blocked on a
single unresolved issue: vLLM cannot construct MiniMax-M2's tokenizer at startup** on the images tried. The
next step is cheap and off-GPU: prove a single pod boots with a **hyphen-free model directory** before any
further GPU spend.

## What is PROVEN (data in hand)

1. **B200 (4×, TP4, FP8, vLLM 0.19.1rc1) — 59-point Pareto** (`pareto-0.19-reference.jsonl`, visualized in `report.html`):
   - **Prefix caching is the dominant lever**: ~8× throughput vs cold (226 → 1,888 tok/s); TTFT stays flat (~4s to c128) where cold collapses (125s by c64).
   - **TP4+EP4 lifts the cached ceiling**: 2,158 tok/s @c128 (+40% over TP4) at lower TTFT.
   - **CPU KV offload ≈ GPU-only (±4%, noise)** on a single-shared-prefix workload — refutes a large-regression claim; offload neither helped nor hurt because one 90K prefix fits HBM (nothing to evict).
   - **100K/90%-reuse is KV-capacity-bound**: 90% hit confirmed; KV → 100% @c512 with queue depth 259 (the saturation knee).
2. **B200 FP8/SM100 is correct** with `--moe-backend triton` + `VLLM_USE_FLASHINFER_MOE_FP8=0` (avoids the FlashInfer float32-router-logits assertion, vLLM #33543). The mdc card's blanket "broken on B200" is refuted for FP8/SM100 (the broken combo is NVFP4/SM120).
3. **TP8 is invalid** for this model (FP8 block-128: MoE intermediate 1536/8=192). Valid TP ∈ {1,2,4}.
4. **Architecture decision framework** (`report.html` + `COMPARISON-CAVEATS.md`): cache-hit rate selects the lever — high-hit+fits-HBM → replicas; high-hit+working-set>HBM → tiered KV cache; low-hit+long-ctx → P/D disagg. **For a >90%-reuse workload, the answer is tiered KV cache, NOT disagg.** (Inferred; the measuring run is the blocked tiering sweep.)
5. **Model family evolution** (M2→M2.5→M2.7): same MoE arch frozen, capability scaled via agent-native RL; serving characterization transfers across versions.

## What is BLOCKED (and on exactly what)

**The follow-on sweeps cannot run until a single serving pod BOOTS MiniMax-M2 to /health 200.** Across many
attempts (minimax27 0.19.1rc1 image, stock v0.23.0, and a custom ECR image with sentencepiece+tiktoken baked
in) the pod dies at `create_engine_config` with:
- `Unrecognized configuration class transformers_modules.minimax_hyphen_m2...` (hyphen-mangled module name), and/or
- `You need sentencepiece or tiktoken installed` (even when they ARE installed/baked).

**Leading hypothesis (untested): the hyphenated model directory** `minimax-m2` → transformers builds a module
`minimax_hyphen_m2` that doesn't match the model's `auto_map` → tokenizer can't load. The unblock is a
one-line restage to a **hyphen-free dir** (`/mnt/nvme/models/minimax_m2`) + `--model` pointed there.

Blocked: P1 (distinct-prefix KV-tiering sweep), P2 (DP/TP parallelism sweep), H200 like-for-like.

## EXACT next step (cheap, off-GPU first)

1. On the **cheap g6e node** (~$1/hr, NOT a B200/H200): stage M2 to `/mnt/nvme/models/minimax_m2` (no hyphen) and boot ONE `vllm serve` pod.
2. Iterate there until `/health` 200 + a clean tool-call. If hyphen-free doesn't fix it, try `--tokenizer-mode slow` and/or a transformers-version bump in the baked image.
3. **Only after a pod boots clean**, scale a GPU node and launch the sweeps (runners are built & validated — see below).

## READY (built, validated, waiting on the unblock)

- `gen-serving-manifest.sh` (5 shapes × 3 KV arms, baked-image, TP-block-128-valid), `run-tiering-sweep.sh` (distinct-prefix N×arm×conc sweep), `run-pareto-sweep.sh` (DP/TP), `run-h200-knee.sh` (full H200 sweep), `run-p1p2.sh` / `run-all-phases.sh` orchestrators.
- Safety: context-pinning, PREFLIGHT interlock, **boot-smoke gate** (caught every bad fix before a sweep ran), **zero-runs guard**, trap-scaledown.
- ECR image `minimax-m2-vllm:v0.23.0-tok` (v0.23.0 + tokenizer deps baked).
- Distinct-prefix bench harness (`--num-prefixes`, `--access`), workload card `distinct-prefix-multitenant.yaml`, spec `minimax-m2-kv-tiering.md`.

## COST / OPERATIONAL lessons (the expensive ones — see lessons.md for detail)

- **~$180 idle-H200 leak**: a spot node joined AFTER the runner's 20-min wait timed out + aborted; the
  preflight-skip-scaledown safety meant the trap never fired → node idle 10h. **Fix**: node-wait > worst-case
  spot join (40+min); on abort, unconditional best-effort scaledown of the known NG; add an idle-NG watchdog;
  ALWAYS verify desired=0 after an abort.
- **Bake images off-GPU; validate model-load on a $1 node before spending $18-30/hr.** Most cycles were burned
  fixing a model-load issue on expensive GPU nodes that could have been isolated on g6e/CPU.
- **vLLM cache-hit is server-counter-only** (not per-request field); the obs pod is load-bearing for harvest.
- **B200 cold-start ~35-40min** (weight load 25.5min alone) — budget readiness probes ≥40min.
- **gen-manifest external DNS** needs `hostNetwork + dnsPolicy: Default` on this cluster's GPU nodes.

## Cost ledger (approximate, this engagement)
- The ~$180 idle-H200 leak is the single largest line. Plus several B200/H200 spot cold-starts (~$18-30/hr × ~0.5-1hr each) across the failed boot attempts. **Net lesson: the deterministic-image-then-cheap-smoke discipline (now documented) would have avoided most of it.**
