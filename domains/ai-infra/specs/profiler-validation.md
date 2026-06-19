# Spec 0 — Profiler Validation

## Status: PREREQUISITE

This is not a hypothesis-driven experiment. It is the **gating prerequisite** for every other spec in the domain. No other spec produces trustworthy numbers until Spec 0 passes.

## Purpose

Validate that `shared/profiler.py` produces a complete, consistent, additive stage attribution for vLLM cold start across at least two existing `gpu-serving` blueprints. Without this, every downstream spec is measuring different things with different boundaries on different clocks, and Spec D's stacking claim is unsupportable.

## Acceptance criteria

A profiler run is valid when *all* of the following hold across 5 consecutive runs of the same blueprint:

1. **All 14 canonical events are present** (T0 through T13). Missing event = invalid run.
2. **Every stage duration is non-negative.** Negative durations indicate clock skew or wrong source attribution.
3. **Stage durations sum to ≥95% of total wall-clock**, with the remainder accounted for in named gap buckets (T1→T2, T3→T4, T6→T7, T8→T9, T11→T12). Unaccounted-for time > 5% means a stage is mis-bounded.
4. **Run-to-run variance** on each stage is reported. Stages with σ/μ > 0.5 across 5 runs are flagged — they may need finer-grained instrumentation or a larger sample size in downstream specs.
5. **Two distinct fixtures pass.** Validating only on one blueprint hides blueprint-specific log format dependencies.

## Canonical event set (T0 through T13)

```
T0  pod_create               kubectl apply returns
T1  node_assigned             kubelet sets pod.spec.nodeName
T2  image_pull_start          kubelet event ImagePulling
T3  image_pull_complete       kubelet event ImagePulled
T4  container_created         containerd CRI CreateContainer returns
T5  container_started         containerd CRI StartContainer returns / PID 1 alive
T6  python_alive              first stdout line from PID 1
T7  weights_load_start        vLLM log: "Loading model weights"
T8  weights_loaded            vLLM log: "Model weights loaded in"
T9  jit_compile_start         vLLM log: torch.compile / DeepGEMM tuning starts
T10 jit_compile_done          vLLM log: capture done / Inductor cache populated
T11 cuda_graphs_done          vLLM log: "Capturing CUDA graphs" complete
T12 health_200                /health returns 200 first time
T13 first_token               first SSE chunk with non-empty content
```

## Stage attribution

| Stage | Boundary | Notes |
|---|---|---|
| node_provision | T0 → T1 | dominated by Karpenter or node prewarm |
| image_pull | T2 → T3 | the target of Spec A/E |
| container_start | T4 → T6 | runtime + Python interpreter init |
| model_load | T7 → T8 | the target of Spec B |
| jit_compile | T9 → T11 | the target of Spec C; envelope only |
| first_token_warmup | T12 → T13 | irreducible CUDA context + first decode |
| **gaps** | T1→T2, T3→T4, T6→T7, T8→T9, T11→T12 | **always reported separately, never hidden** |

## Out of scope for Spec 0

- Sub-stage attribution (per-Inductor pass, per-CUDA-graph capture). That's downstream spec instrumentation.
- eBPF profiling (Spec F adds it as an extension).
- nsys traces (Spec C adds them as an extension).
- Cross-pod profiling. One pod = one timeline. Multi-replica spec D measures each replica separately and aggregates.

## Validation procedure

1. Build slim images per `shared/images/README.md` (`./build.sh`). Spec 0 measures against slim, not stock upstream — measuring against bloat would inflate downstream wins. Build both `cu128` and `cu130` variants; cu130 is required for B300 (sm_103).
2. Pick two existing fixtures:
   - **Small fixture**: `qwen3-next` — runs on any GPU (g5/g6/g7e), low JIT cost. Validates the profiler's basic shape on cheap hardware.
   - **Large fixture**: `kimi-k2.6` (Kimi K2.6 FP8) — runs on **`p6-b300.48xlarge` spot in us-west-2b** via the `ai-infra-b300-spot` nodegroup. Exercises the high-JIT envelope (DeepGEMM autotune + torch.compile + CUDA graphs).
   Both fixture pod manifests live in `staging/manifests/`.
3. Run `shared/profiler.py` 5 times against each fixture, on a quiescent cluster. Driven by `staging/scripts/run-plan.sh small` (qwen3-next) and `run-plan.sh b300` (Kimi K2.6).
4. Run `shared/profiler_validate.py` against the 10 artifacts.
5. If validation fails, fix the profiler (regex drift, missing event source, etc.) and re-run. **Do not start any other spec until validation passes.**
6. Commit the 10 baseline artifacts to `domains/ai-infra/blueprints/profiler-validation/results/` as the canonical reference. Tag the slim image versions used in `metadata.image_tag`.
7. Also run one stock-upstream measurement per fixture (`vllm/vllm-openai:v0.21.0`) to record the bloat baseline for context in the lab's published findings. These are not part of the validation set — they're documentation only.

## Maintenance contract

The profiler is a **living artifact**. Every vLLM bump in any blueprint triggers a re-validation:

- CI runs `profiler_validate.py` against the latest blueprint's first cold start after each vLLM upgrade.
- If validation fails, regex updates land in `shared/log_patterns.yaml` keyed by vLLM version range.
- A failed validation blocks downstream spec progress until the regex set is fixed.

## Output

- 10 baseline artifacts in `results/`.
- `analysis.md` documenting per-stage means and variances on each fixture.
- A validated `shared/log_patterns.yaml` with the current vLLM version range covered.
- Confirmation that downstream specs can consume the canonical artifact format.

## References

- Profiler design discussion in main thread.
- `shared/profiler.py`, `shared/log_patterns.yaml`, `shared/profiler_validate.py`.
