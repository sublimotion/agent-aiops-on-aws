# Comparison Caveats — read BEFORE quoting any number to the customer

> Guardrail against the "we're showing way more throughput than the customer had" trap.
> Our early sweep numbers look far higher than the customer's dashboards — but that delta
> is a STACK of confounds, not a clean win. Per `.claude/steering/benchmark-analysis.md`
> ("match before you compare"), every confound below must be stated or the headline misleads.

## The confounds (why our number ≠ their number)

| Axis | Customer (their dashboards) | Our sweep | Effect on the delta |
|------|------------------------------|-----------|---------------------|
| **GPU** | H200 (Hopper, ~4.8 TB/s HBM) | **B200 (Blackwell, ~8 TB/s HBM, ~2× FP8 FLOPs)** | A large chunk of "more throughput" is just a faster chip — NOT config or model |
| **GPU count / shape** | 4 GPUs | TP4 matched for the baseline; but DP shapes use all **8** | Only the TP4 configs are GPU-count-matched; DP shapes are NOT |
| **Engine** | vLLM v0.23.0 | vLLM **0.19.1rc1** (`minimax27` image) | Different scheduler/kernels; our build is OLDER |
| **KV config** | ran WITH CPU offload (their −38% RPM regression) | gpu-only baseline | We may be beating their *misconfigured* run, not their ceiling |
| **Workload** | real production traffic | synthetic, **byte-identical 90K shared prefix** | Our cache-hit may exceed their real reuse → inflates our aggregate throughput |
| **Metric** | RPM on their panels | aggregate tok/s at high concurrency | Aggregate conflates batching with hardware; not the comparable unit |

**Net**: our headline ≈ (B200 faster than H200) × (gpu-only beats their offload config) × (synthetic cache-hit possibly > real). Three multiplied effects — we cannot yet attribute how much is the chip vs the config.

## What we CAN claim honestly (transferable, hardware-robust)

These come from **on-identical-hardware A/Bs** the sweep is designed to produce — direction holds regardless of chip:

1. **gpu-only prefix caching vs their CPU-offload config, same TP4 on the same B200** — the apples-to-apples config A/B. If gpu-only wins, that's a real, actionable finding.
2. **Prefix caching flattens TTFT** — shape result (e.g. ~4s to c128 cached vs cold's collapse by c16). Robust to hardware.
3. **Which parallelism shape wins** (TP4 vs TP4+EP4 vs DP fan-outs) — relative deltas on the same node.
4. **Does CPU offload help UNDER reuse** (vs the customer's cold-run regression) and at what ITL cost — the core architecture question, measured on one chip.
5. **What bounds the knee** (KV-capacity / HBM-BW / compute / transfer) — a class claim from the telemetry.

## What we must NOT say

- ❌ "MiniMax-M2 gives you N× your current throughput" — conflates chip + config + cache-hit; the customer can't act on it and may be on H200 in prod.
- ❌ Any absolute tok/s presented without the "**B200, vLLM 0.19.1rc1, synthetic shared-prefix**" label.
- ❌ A B200 aggregate-throughput number compared to their H200 RPM as if it were like-for-like.

## The headline framing for the deliverable

- Lead with the **on-identical-hardware config deltas** (caching, parallelism, offload arms) — those are the transferable lessons.
- State the absolute throughput as **a B200 number**, with one caveat line: *"B200 is a materially faster chip than your H200; treat absolute throughput as a B200 figure and use the relative config deltas as the transferable findings."*
- **Decisive follow-on if their prod target is H200**: re-run the winning config on H200 for a true matched number. Until then, the B200 headline overstates what they'd see on their hardware.

## Status of matched-ness in THIS sweep
- TP4 configs: GPU-count-matched to the customer (4 GPUs), but NOT chip-matched (B200 vs H200) or engine-matched (0.19.1rc1 vs 0.23.0).
- DP configs (TP2+DP2, TP4+DP2, TP2+DP4): use 8 GPUs — NOT GPU-count-matched; these answer "best shape on the node we have," not "what the customer's 4-GPU box does."
