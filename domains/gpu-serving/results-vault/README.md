# GPU Serving — Results Vault

Consolidated directory of every published v1 benchmark-commons envelope produced by `domains/gpu-serving/` blueprints. One flat namespace enables cross-blueprint comparison without walking per-session directories.

## Layout

```
results-vault/
├── <model>_<substrate>_<hw>_<engine-config>_<workload>_c<N>.json   ← symlink (single-point) OR
│                                                                     materialized file (split from rolled-up sweep)
├── index.json                                                       ← generated manifest
├── normalize-into-vault.py                                          ← consolidator (run first)
├── rebuild-index.py                                                 ← regenerator (run after normalize)
└── README.md                                                        ← this file
```

The vault contract is **one file = one operating point**. Conformant blueprint
artifacts are symlinked from `blueprints/<name>/results/`. Rolled-up artifacts
that pin a whole sweep into one file (`extensions.sweep_levels`,
`extensions.context_sweep`, `extensions.reranker.pair_length_sweep`) are
**materialized** into per-point files written into the vault — each carries
`extensions.normalized_from` pointing back to the rolled-up source. Primary
blueprint data is never modified.

## Filename convention

```
<model-slug>_<substrate>_<hw>_<engine-config-tag>_<workload-catalog-id>_c<concurrency>.json
```

Examples:
- `kimi-k2.6_ec2-spot_p6-b300_sglang-eagle3-s4d4k1-hicache200_concurrency-sweep_c128.json`
- `kimi-k2.6_ec2-spot_p6-b300_sglang-eagle3-s4d4-tp4dp2-hicache_concurrency-sweep_c256.json`
- `kimi-k2.6_ec2-spot_p6-b300_sglang-eagle3-s4d4-no-cudagraphs_concurrency-sweep_c1.json`  *(ablation)*

The filename is enough to identify the run without opening the file.

## index.json

Generated manifest with the most-queried fields hoisted to the top level (agg throughput, TTFT p50/p99, accept rate, cost). Regenerate when artifacts are added or changed:

```bash
python3 domains/gpu-serving/results-vault/rebuild-index.py
```

### Querying the vault

```bash
# All B300 runs
jq '.artifacts[] | select(.gpu_type=="B300") | {file, agg_tok_per_s}' index.json

# Best aggregate throughput at c=128
jq '.artifacts[] | select(.concurrency==128) | {engine_config_tag, agg_tok_per_s}' index.json \
  | jq -s 'sort_by(-.agg_tok_per_s) | .[0:5]'

# All configs with EAGLE3 num_steps=4 that passed SLO
jq '.artifacts[] | select(.speculative_num_steps==4 and .slo_overall_pass==true)' index.json
```

### Top-level fields in each index row

- Identity: `file`, `created_at`, `blueprint`, `phase`
- Model: `model_id`, `model_name`, `quantization`
- Engine: `engine_name`, `engine_config_tag`, `tensor_parallel`, `pipeline_parallel`, `data_parallel`, `speculative_algorithm`, `speculative_num_steps`
- Infra: `substrate`, `instance_type`, `gpu_type`, `gpu_count`
- Workload: `workload_catalog_id`, `concurrency`, `input_tokens_mean`, `output_tokens_mean`
- Perf: `agg_tok_per_s`, `ttft_p50_ms`, `ttft_p99_ms`, `tpot_p50_ms`, `tpot_p99_ms`, `e2e_p50_ms`, `e2e_p99_ms`, `error_rate`, `completed`
- Gates: `slo_overall_pass`
- Spec: `spec_accept_rate`, `spec_accept_length`
- Cost: `dollars_per_1m_output_tokens`

## When to add to the vault

New benchmark session produces artifacts in `domains/gpu-serving/blueprints/<name>/results/{standard,artifacts}/`. After the run:

```bash
# 1. Consolidate (symlink single-point + materialize rolled-up sweeps)
python3 domains/gpu-serving/results-vault/normalize-into-vault.py

# 2. Regenerate the manifest
python3 domains/gpu-serving/results-vault/rebuild-index.py
```

`normalize-into-vault.py` is idempotent and prunes stale materialized files
when the rolled-up source is removed. The `bench-standard.py` driver writes
single-point files directly; legacy emitters under
`<blueprint>/scripts/_*_sweep.py` and `emit_common_artifacts.py` still write
rolled-up shapes — these get split during normalization.

## What lives here vs in the blueprint

| Lives in blueprint `results/` | Lives in vault |
|---|---|
| Raw client output (`phase-*/c*.json`) | (not vaulted) |
| Session logs, `progress.md`, `lessons.md` | (not vaulted) |
| Standard v1 envelopes (`standard/*.json`) | **symlinked here** |
| Per-phase markdown reports | (not vaulted) |
| Prometheus snapshots (if kept locally) | (not vaulted, goes to S3) |

The vault is the comparison index. Per-blueprint `results/` is the forensic trail.

## Current contents

As of 2026-05-14, **116 artifacts across 3 models and 3 GPU types**:

| Blueprint | Model | GPU | Artifacts | Notes |
|---|---|---|---|---|
| `kimi-k2.6-speculative` | Kimi-K2.6 (1T MoE, FP8) | B300 | 95 | Phase 1b sweep + Phase 4/5. `ttft_ms: null` — no Prometheus at run time. |
| `nemotron-super` | Nemotron-3-Super (120B/12B, FP8, Mamba-MoE hybrid) | B200 | 14 | Agg vs disagg (Dynamo) comparison. Enriched from legacy custbench flat format. |
| `qwen3-embedding-8b-hyperpod` | Qwen3-Embedding-8B | A10G (g5.4xlarge) | 7 | Emitted v1 format natively. Burn-in + workload matrix. |

The Kimi K2.6-spec gap motivated the observability mandate in `.claude/skills/benchmark-runner/SKILL.md`. Future blueprints using `bench-standard.py` will populate TTFT correctly from Prometheus histograms.

### Enrichment notes

- **Nemotron**: `ttft_ms` populated from the flat custom format's `summary.ttft_ms`. TPOT is derived from `itl_ms` (inter-token latency, which includes the first token gap — so the standard `tpot_ms` percentiles are approximate). Per-request detail (`raw[]` array in the flat format) was NOT preserved in the envelope to keep file size manageable; the source files remain in `blueprints/nemotron-super/results/` for forensics. `output_tokens_total` is null because the flat format stored `total_output_chars` rather than token counts.

- **Qwen3-embedding**: Native v1 format, no enrichment needed. This is the reference pattern for future blueprints.

- **Kimi K2.6-spec**: `ttft_ms`, `tpot_ms.p90/p95/p99` all null — reconstruction impossible. See `blueprints/kimi-k2.6-speculative/scripts/enrich-to-standard.py` and `lessons.md` L14–L16 for context.
