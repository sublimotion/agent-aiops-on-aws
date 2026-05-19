# Example: Nemotron-3-Super on Dynamo (Disaggregated P/D)

Conforming input/output example for the enriched benchmark artifact standard.

## Files

| File | Role | Description |
|------|------|-------------|
| `benchmark.yaml` | **Input** (sidecar) | Deployment context authored once per blueprint. Feeds into the enrichment wrapper alongside AIPerf output. |
| `..._concurrency-sweep_c64.json` | **Output** (enriched artifact) | One artifact per concurrency level. This example shows the c=64 point from a concurrency sweep. |

## How it maps to the standard

### Input: `benchmark.yaml`

- **model** — HuggingFace ID, architecture type, quantization, context length
- **engine** — Container image, parallelism (TP=2), launch args, KV cache dtype
- **framework** — Dynamo disaggregated config (4P+4D, NIXL, round-robin router)
- **infrastructure** — p6-b200.48xlarge, EKS, NVSwitch
- **workloads** — References catalog IDs (`concurrency-sweep`, `coding-agent`) + one custom workload (`long-context-128k`)
- **slo** — Targets from the spec's success criteria

### Output: enriched artifact JSON

- **Envelope** — schema version, artifact ID, timestamp, source tool
- **Portable context** — model + engine + framework (everything needed to reproduce)
- **Descriptive metadata** — infrastructure (where it ran)
- **Core metrics** — TTFT, TPOT, ITL, E2E, throughput (from AIPerf)
- **SLO evaluation** — targets vs actuals, all passing
- **Extensions** — GPU telemetry, cache stats (with Mamba limitations noted), Dynamo-specific metrics (disagg speedup, NIXL latency, P:D ratio analysis), cost efficiency, per-request data pointer

## Naming convention

```
{model}_{substrate}_{instance}_{engine}-{framework}-{config}_{workload}_{level}.json
```

Example: `nemotron-super_eks_p6-b200_sglang-dynamo-disagg-4p4d_concurrency-sweep_c64.json`

## Usage

```bash
# Produce enriched artifact from AIPerf output + sidecar
python scripts/enrich-benchmark.py \
  --sidecar blueprints/nemotron-super/benchmark.yaml \
  --aiperf-output /tmp/aiperf_results_c64.json \
  --concurrency 64 \
  --output blueprints/nemotron-super/results/

# Validate against schema
python scripts/validate-artifact.py results/*.json
```
