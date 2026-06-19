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

## "X is broken" — distinguish the mechanism from the report

Before declaring a feature broken, separate **does it work** from **is it reported**. A null/zero in a client field is not proof the feature is off — check the engine's own server metric. (2026-06-16: "prefix caching broken" was a null `prompt_tokens_details` in vLLM's per-request response; the server counters showed 74% hit the whole time. SGLang reports it per-request, vLLM only in counters.)

## Capability claims — search the repo before asserting absence

"There's no knob for X" / "Y isn't supported" are **searchable** facts in this repo (lessons.md across blueprints, mdc cards, configs). Search before asserting absence — don't infer it from a config dump. (2026-06-16: claimed "no MoE tile-tuning knob for NVFP4" from inference; the exhaustive search both corrected the framing AND found the real answer — it's a documented no-op across 3 blueprints, for a better-evidenced reason.)

## End-of-session self-audit (before any customer-facing writeup)

Before producing a report/recommendation, re-read every stated conclusion and challenge each:
- Is this [measured] or [inferred]? Did I verify the inferred ones?
- Every cross-run comparison: engine + output-length + context matched?
- Any "broken"/"unsupported"/"no knob" claim: did I check the server metric / search the repo?
- Any bottleneck-class claim: did I cite the gauge, including TTFT share?
- Any result that *reverses* a prior finding (e.g. engine winner flipping): flagged as unexplained, not silently asserted?

Run the `carryover-auditor` agent or an adversarial self-review over the conclusions. Unverified claims get downgraded to open questions, not shipped.
