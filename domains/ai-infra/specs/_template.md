# [Technique] — [Workload] Experiment

## Status: DRAFT | RUNNING | ANALYZING | COMPLETED | FALSIFIED | ARCHIVED

## Hypothesis

State the claim in one sentence, with a measurable threshold and a condition.

> Example: "Run:ai Model Streamer cuts cold-start time-to-first-token by ≥3× when loading from S3, and is neutral (±10%) when loading from NVMe."

A hypothesis without a threshold is a wish. A hypothesis without a condition is a slogan.

## Falsification criteria

What result would make us drop this technique? Be explicit. If you can't write this, the spec is not ready.

> Example: "Falsified if median cold-start improvement is <1.5× on S3 across all tested models, OR if any model shows >20% regression on NVMe."

## Why this matters

One paragraph: what production pain does this solve, what's the operational lever, what would a positive result enable.

## Stage-budget claim

Locate the technique in the cold-start pipeline (see README) and predict the stage-time delta. Use this table verbatim; it makes specs stackable.

| Stage | Baseline (sec) | Predicted with technique (sec) | Why |
|---|---|---|---|
| Node provision | | | |
| Image pull | | | |
| Container start | | | |
| Model load | | | |
| JIT / compile | | | |
| First token | | | |
| **Total** | | | |

Specify which **replica index** the prediction applies to: 1st (cold cluster), Nth (warm node pool), or all.

## Matrix

| Axis | Values |
|------|--------|
| Models | (e.g., qwen3-8b, glm-5-fp8, kimi-k2.6) — pick small/medium/large to test scaling |
| Hardware | (e.g., g7e.24xlarge, p5.48xlarge, p6-b300.48xlarge) |
| Storage backend | (e.g., S3, FSx Lustre, RAID0 NVMe) |
| Variants | baseline vs technique-on |

Total cells = product of axes. State which cells you'll actually run vs which are exploratory.

## Baseline

What is "off"? Specify exactly:
- Default vLLM `--load-format auto`
- Storage backend specified above
- Same hardware, same model, same model-len, same dtype

The baseline must run on the same infrastructure as the variant. No cross-cluster comparisons.

## Measurement

What is measured, how, by what tool. Reuse `shared/` harnesses where possible.

- **Primary metric**: (e.g., cold-start time from pod-create to first-token-streamed, in seconds)
- **Secondary metrics**: (e.g., steady-state TTFT P50/P99, throughput, GPU utilization during load)
- **Sample size**: how many runs per cell, how outliers are handled
- **Output format**: enriched JSON artifact per `standards/benchmark-commons/PROPOSAL.md`

## Fixtures

Which existing `gpu-serving` blueprints are used as the deployment substrate. Do not duplicate deployment logic here.

- `domains/gpu-serving/blueprints/<name>/` — used for model X
- `domains/gpu-serving/blueprints/<name>/` — used for model Y

## Rule the experiment would produce

If the hypothesis holds, what concrete steering rule lands in `.claude/steering/tech-stack.md`?

> Example: "For vLLM blueprints loading model weights from S3, default to `--load-format runai_streamer`. Skip when loading from RAID0 NVMe (negligible benefit). Requires `runai-model-streamer-s3` package in the container."

If you can't predict the rule shape, the experiment isn't well-scoped.

## Out of scope

- Adjacent techniques not under test (e.g., this experiment is *not* about LMCache, NIXL, etc.)
- Production rollout decisions (those follow from the rule, in a separate PR)

## Carryover audit (spec-design gate)

Before running, confirm no prior-blueprint lesson was forgotten:
- [ ] Ran the `carryover-auditor` agent on this spec (or equivalent self-check): scanned every `domains/**/lessons.md` whose stack (`model`/`engine`/`gpu_arch`/`hardware`/`failure_categories`) overlaps this experiment.
- [ ] Every applicable prior lesson — especially `outcome: failure`/`partial` — is reflected in the matrix, fixtures, or falsification criteria, OR noted as not applicable, citing its source (`<blueprint>/lessons.md` #N).

## Persistent caches via EBS snapshot (when applicable)

If the experiment produces a persistent artifact (compile cache, kernel cache, model weights, prefetch index, FUSE cache), consider the **EBS-snapshot pattern** instead of image-baking or hostPath:

```
EBS volume (per-AZ, snapshot-backed):
  /mnt/persistent/<artifact-type>/<config-hash>/
    <files>
```

Setup:
1. Provision EBS volume in target AZ (cheaper than FSx for persistent caches < 1 TB).
2. Bake artifact once: pod with `hostPath` mount writes to volume.
3. `aws ec2 create-snapshot` — incremental, 60s typical for sub-1 GB.
4. Tag snapshot with `(model, vllm_version, tp_size, config_hash)` for retrieval.
5. Production: PVC restored from snapshot, mounted at consumer's expected path.

Validated on Spec C-EBS for vLLM compile caches:
- Volume create from snapshot: ~9 s
- Attach + mount: ~10 s
- AOT cache HIT delivers identical compile speedup on fresh node as same-node hostPath
- Multi-variant: vLLM auto-keys by `<config_hash>`, multiple coexist on one volume
- See `domains/ai-infra/blueprints/spec-c-ebs-snapshot/results/ebs-cache-validation.json`

When NOT to use this pattern:
- Artifact > 50 GB and cold-cluster cold-start matters: EBS read bandwidth (~1-2 GB/s gp3) becomes bottleneck. Use instance NVMe + S3 sync instead.
- Single-AZ-only deployment: snapshot is per-region; you can copy across AZs but adds operational complexity.
- Artifact changes per-run (e.g., model weights for fine-tuning): defeats the snapshot lifecycle.

## Cost estimate

Rough $ estimate for the matrix. Cap before launching.

## References

- Upstream project: link
- Related specs in this domain or others
- Relevant memory entries (`memory/*.md`)
