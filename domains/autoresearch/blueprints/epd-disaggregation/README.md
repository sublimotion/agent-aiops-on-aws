# EPD Disaggregation Blueprint

Operational artifacts for the Encode–Prefill–Decode (EPD) disaggregation experiment. Spec lives at `domains/autoresearch/specs/epd-disaggregation.md`; agent instructions in `program.md`.

Two instances of one pattern:
- **VLM EPD** — vision encoder tier split from LLM prefill/decode; hand-off artifact = embeddings.
- **Wan 2.2 EPD** — denoise / VAE-decode / NVENC-encode tiers; hand-off artifact = latents.

The rule under test: *split where the hand-off artifact is small* (latents/embeddings, never pixels).

## Layout

```
configs/    # tier topology per config (stage → instance type → replica count)
results/    # experiments.jsonl, handoff_metrics/, cost_analysis.md
program.md  # agent loop + decision rules
```

## Baselines

| ID | Config | Source |
|----|--------|--------|
| B0 | Co-located (all stages on expensive GPU) | Synthesia synchronous (~82% util) |
| B1 | Intra-box async (dual-stream + NVENC) | Synthesia async (~99.9% util) |
| EPD | Disaggregated tiers | this experiment |

## What's intentionally NOT here yet

- `configs/*.yaml` — tier topologies. Added in Phase 1 once stage characterization confirms split points.
- Deployment scripts — will reuse `ray-serve-video` KubeRay infra + runtime_env pins.
- `results/*` — produced by experiment runs (empty until Phase 1).

## Prior art

`domains/gpu-serving/blueprints/ray-serve-video/` — disaggregated multimodal pipeline in all but name (Kafka, runtime_env isolation, in-memory hand-off). Reuse its infra and dependency pins.

## Related

- [Spec](../../specs/epd-disaggregation.md)
- [ray-serve-video](../../../gpu-serving/blueprints/ray-serve-video/) — prior art
- [kernel-optimization-agent](../kernel-optimization-agent/) — complementary (compute inside the kernel vs topology around the pipeline)
