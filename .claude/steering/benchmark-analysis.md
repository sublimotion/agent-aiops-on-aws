# Benchmark Analysis Discipline

> Load this when **interpreting benchmark results, classifying bottlenecks, or comparing runs** — i.e. any time you're about to state a performance *conclusion* (benchmark-analyst agent, infra-deployer Stage 6, or ad-hoc analysis). Companion to `inference-first-principles.md`: that one predicts the regime *before* a run; this one governs what you may *assert after* one.
>
> Origin: 2026-06-16 Kimi-K2.6-NVFP4 session, where several confident conclusions (bottleneck class, cross-run comparison, "cache broken", "no MoE knob") were stated from inference/convenient-comparison and corrected only after user pushback. Root cause: stating reasoning with the confidence of measurement, and the user serving as the verification layer instead of the agent.

## The one rule

**Never state a performance conclusion the data in front of you doesn't directly support.** If you reasoned your way to it, it is a *hypothesis* until a measured metric confirms it. Label every finding:

- **[measured: <metric>]** — backed by a number you pulled this session. Cite the metric.
- **[inferred]** — reasoned from config/architecture/priors. **Must verify against live data BEFORE presenting to the user, not after.**

If you cannot verify an [inferred] claim before reporting, present it as an open question ("likely X — verifying"), never as a finding.

## Bottleneck classification — cite the metric, don't reason from the config

A bottleneck *class* (prefill-compute / decode-BW / KV-capacity / launch-bound / admission-capped) is a **measured** claim. Required evidence before asserting one:

| Claim | Required measured evidence |
|-------|---------------------------|
| KV-capacity-bound | engine KV-usage gauge near saturation (`token_usage`≈0.9+ / `gpu_cache_usage_perc`) AND queue growing |
| decode-compute-bound | throughput flat as concurrency rises AND a lever that adds *only* residency/admission (fp8-KV, max-running) does NOT move it |
| prefill-bound | **TTFT as a fraction of E2E latency** (pull `time_to_first_token` vs `e2e_request_latency`). High TTFT share ⇒ prefill matters. NEVER call something "decode-bound" without checking TTFT first. |
| launch/dispatch-bound | per-GEMM tuning loads but doesn't help; gain only from reducing kernel-launch count |

The 2026-06-16 error: declared "decode-compute-bound" from a flat throughput curve *before* pulling TTFT, which then showed 26% prefill. **Check TTFT share for ANY throughput-ceiling diagnosis.**

## Cross-run comparison — match before you compare

Before comparing this run to a prior benchmark, confirm ALL of these match, or state the mismatch loudly:

1. **Engine** — SGLang vs vLLM are different runs. Never compare our SGLang number to a prior vLLM number. (The 2026-06-16 error: quoted prior vLLM 10,437 against our SGLang 2,516 → looked like a 4× regression; engine-matched was 3,400, a 26% gap fully explained by workload.)
2. **Output length** — aggregate tok/s scales inversely with output tokens/req (short outputs cycle the batch faster). A 256-out run is not comparable to a 1024-out run.
3. **Input/context length** — longer context = more KV read per decode step.
4. **Quant + hardware + engine version.**

**The hardware-comparable metric is per-request tok/s at matched context+output (ideally single-stream)**, NOT peak aggregate. Aggregate conflates workload with hardware. When a clean comparison is impossible, say so — do not present a mismatched ratio as a regression or a win.

## Reasoning-efficiency claims — measure tokens-per-task × success-rate, not throughput

When a model release claims it is more *reasoning-efficient* ("fewer thinking tokens", "less overthinking", "X% lower reasoning-token usage"), the improvement is **invisible to our standard perf sweep** and must NOT be evaluated with it. Fixed-`--random-output-len` + `--ignore-eos` throughput rigs hold output length constant by design, which erases exactly the thing that changed: the model emitting *fewer tokens per solved task*. tok/s is the wrong numerator — the model isn't emitting tokens faster, it's emitting fewer per task. The improvement only surfaces as **tokens-to-completion AND success-rate on a fixed task set, EOS enabled, same harness for both models** (a SWE-bench / agent-trace eval, not a gpu-serving sweep). The cost unit is **$/solved-task, not $/1M-tokens.**

Two traps to guard against:
1. **Efficiency vs truncation confound.** "Fewer reasoning tokens" can be genuine efficiency OR truncated reasoning that drops hard-task accuracy — identical on a token-count metric, separable only with a paired success-rate metric. Always measure tokens-per-task *and* accuracy together, or you can't tell "smarter" from "lazier."
2. **Vendor cards report the efficiency in prose, disconnected from the accuracy table.** (2026-06-26, Kimi K2.7 Code: card claims "reducing thinking-token usage by ~30% vs K2.6" with no benchmark named, no "thinking token" definition, no task set, no tokens-per-task in the results table; thinking-mode forced on at temp=1.0 so it can't even be A/B'd off. The accuracy table and the efficiency claim live in separate, unjoined places.) Treat an unanchored efficiency number as an open question, and re-verify with a paired tokens-per-task × success run. Also watch for **asymmetric harnesses** in vendor comparisons (K2.7 ran its own Kimi Code CLI vs competitors in Codex/Claude Code "xhigh") — harness choice swings task results as much as the model (see agent-swarm: 38–88% spread for one model across harnesses).

## "X is broken" — distinguish the mechanism from the report

Before declaring a feature broken, separate **does it work** from **is it reported**. A null/zero in a client field is not proof the feature is off — check the engine's own server metric. (2026-06-16: "prefix caching broken" was a null `prompt_tokens_details` in vLLM's per-request response; the server counters showed 74% hit the whole time. SGLang reports it per-request, vLLM only in counters.)

## Capability claims — search the repo before asserting absence

"There's no knob for X" / "Y isn't supported" are **searchable** facts in this repo (lessons.md across blueprints, mdc cards, configs). Search before asserting absence — don't infer it from a config dump. (2026-06-16: claimed "no MoE tile-tuning knob for NVFP4" from inference; the exhaustive search both corrected the framing AND found the real answer — it's a documented no-op across 3 blueprints, for a better-evidenced reason.)

## Verify-before-crediting-a-lever — separate the explicit config from the effective config
<!-- validated: 2026-06-27 -->

**Before crediting a lever with a measured delta, verify what the effective configuration actually was, not what you set.** Model defaults, auto-resolution, and framework heuristics can silently enable features without an explicit flag, making a nominally "added" flag a no-op. Origin: GLM-5.2 glm5.2 L7 (2026-06-27).

**Symptom**: you measure a lever (e.g., "+fp8 KV") as T1 = +1.7% over T0, treat it as a weak lever, then discover T0 *already ran fp8 KV* because the model auto-resolves `quantization=fp8` → `kv_cache_dtype=fp8` on SGLang. The +1.7% was run-to-run noise, not evidence "fp8 KV doesn't help" — it's evidence the flag was redundant.

**Rule**: when a measured delta is suspiciously small (<3%, within the plateau threshold) for a lever expected to be structural (e.g., precision change), **check the engine logs or config dump for what actually ran** before promoting the number as a result. The explicit CLI flag is the *request*, not the truth. Grep the serving log for the auto-resolved dtype/backend/strategy. For precision specifically: check the KV pool allocation line for the effective torch dtype.

**When it applies**: any config that has framework heuristics or model-card defaults — quantization dtype (fp8/fp4), KV cache dtype, attention backend (FlashInfer vs Triton), MoE backend (DeepGEMM vs Triton), speculative decode auto-tuning. Does NOT apply to clean boolean toggles like `--enable-prefix-caching` (explicit off vs explicit on).

**Examples**:
- GLM-5.2 DSA on SGLang v0.5.13: `quantization=fp8` auto-resolves `kv_cache_dtype=fp8_e4m3` even if the flag is absent. Explicitly setting `--kv-cache-dtype fp8_e4m3` is a no-op.
- (Add more occurrences when they surface — this is the first measured instance of the pattern.)

**Why this didn't fire in B200 GLM-5.2**: the B200 STAGE6-REPORT (lesson L7 references it) shows T0 1708→T1 3004 (+76%) "fp8 KV", but T0 was BF16-everything (no `quantization=fp8`) so the auto-resolve didn't happen. B300 T0 was already FP8 weights (`quantization=fp8`), triggering the auto-KV-fp8 path. Same model, different T0 → different effective-config question.

## Benchmark client artifacts can fake server-side SLO failures
<!-- validated: 2026-06-27 -->

**Before diagnosing a concurrency ceiling as a server limit (KV exhaustion, admission cap, quality breach), confirm the failure against server-side metrics.** Client-side request failures (timeouts, connection errors, retries) can stem from harness bugs — connector limits, connection reuse across runs, ephemeral port exhaustion — and present identically to a real serving limit. Origin: GLM-5.2 glm5.2 L8 (2026-06-27).

**Symptom**: client driver reports 27–32% request failures at c384/c512 (looks like KV exhaustion or a serving crash). Server `num_retracted_reqs=0` throughout, engine logs show no errors, GPU util normal. The failure is CLIENT-side (harness config), not server-side (serving limit).

**Root causes observed**:
1. **aiohttp `TCPConnector(limit=conc*2)` with chained runs in one process** — closing/reused connections from the prior concurrency run poison the next. Symptom: c384 run immediately after c256 in the same process shows ~27% failures; fresh process + same c384 config = 0 failures.
2. **Ephemeral port exhaustion** — `limit=0` (unlimited) at c≥384 opens ~1000+ sockets; Linux default ephemeral range (~28K ports) can exhaust under rapid open/close. Symptom: `ConnectionResetError` / `OSError: Cannot assign requested address`.
3. **Timeout too short for high-latency c-points** — default client timeout (e.g., 60s) breached at high concurrency where TTFT climbs to 10–15s + decode 2048 tokens. Symptom: client-side timeout; server completed the request.

**Rules**:
1. **One fresh client process PER concurrency point** — never chain concurrency sweeps (`for c in [8,16,32,64,128,256,384]: run(c)`) in a single Python process. Launch a new process or container per c-point.
2. **Confirm failure class against server metrics** — check `num_retracted_reqs`, engine error logs, `nvidia-smi` for OOM/Xid before calling a client-side failure a serving limit. If server shows 0 errors and the failure is reproducible only in certain client configs, it's a harness bug.
3. **For c≥256, use `TCPConnector(limit=0)` (unlimited) AND a distributed driver** — split the concurrency across 2–3 client pods on separate nodes so each opens fewer sockets. The single-client "SLO error wall" at c384 is often harness-induced, not a true serving ceiling.

**Why this matters**: the L8 failure nearly cost a WRONG knee (c256 recorded as the ceiling when the true comfortable knee was c320, ceiling c384). Client-side harness bugs that fake server limits lead to under-sizing fleets and leaving real capacity on the table.

## SGLang metric names differ from Prometheus-scraped names — use underscore, not colon
<!-- stack: sglang>=0.5.13, prometheus=2.54 | validated: 2026-06-27 -->

SGLang emits Prometheus metrics with colons in the name (e.g., `sglang:time_to_first_token_seconds_bucket`). Prometheus **sanitizes `:` → `_` on ingestion** (colon is reserved for recording rules), so the stored metric is `sglang_time_to_first_token_seconds_bucket`. A PromQL query using the colon form is **invalid and silently returns empty** — no error, just zero datapoints.

**Rule**: when querying SGLang metrics from Prometheus, **always use underscore**, never colon. This applies to all `sglang:*` metrics, not just TTFT. The engine's `/metrics` endpoint shows the colon form; the Prometheus TSDB stores the underscore form. Origin: GLM-5.2 glm5.2 L5 (2026-06-27) — the bench-standard.py P0 TTFT gate was firing "TTFT p95 = null" because the query used `:` and returned empty.

**Other SGLang metric name differences** (beyond the colon swap):
- **TPOT**: engine emits no `time_per_output_token`; use `sglang_inter_token_latency_seconds_bucket` instead.
- **KV usage**: `sglang_token_usage` (not `_ratio`).
- **Cache hit**: `sglang_cache_hit_rate`.
- **Request success/error counters**: SGLang has NO native success/error counter. Derive from `sglang_http_responses_total{endpoint=~"/generate|/v1/chat/completions", status_code=~"2.."}` for success count.

**Fix location**: `.claude/skills/benchmark-runner/scripts/bench-standard.py` — the canonical bench driver. The P0 TTFT-null gate (which blocks any benchmark run if TTFT can't be measured) now uses the correct underscore names and was smoke-tested before the GLM-5.2 B300 sweep. Local or blueprint-runner-specific scripts that query SGLang metrics must apply the same corrections.

## End-of-session self-audit (before any customer-facing writeup)

Before producing a report/recommendation, re-read every stated conclusion and challenge each:
- Is this [measured] or [inferred]? Did I verify the inferred ones?
- Every cross-run comparison: engine + output-length + context matched?
- Any "broken"/"unsupported"/"no knob" claim: did I check the server metric / search the repo?
- Any bottleneck-class claim: did I cite the gauge, including TTFT share?
- Any reasoning-efficiency claim ("fewer thinking tokens"): measured as tokens-per-task × success-rate with EOS on, not a fixed-OSL throughput sweep? Vendor number treated as open until verified?
- Any result that *reverses* a prior finding (e.g. engine winner flipping): flagged as unexplained, not silently asserted?
- Any lever delta <5%: did I verify the effective config matched the explicit config (auto-resolve trap)?
- Any client-side failure at high concurrency: did I confirm against server metrics before diagnosing as a serving limit?

Run the `carryover-auditor` agent or an adversarial self-review over the conclusions. Unverified claims get downgraded to open questions, not shipped.
