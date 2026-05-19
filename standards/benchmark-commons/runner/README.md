# Benchmark Runner

Orchestrates benchmark execution on AWS infrastructure (EKS, HyperPod, bare metal) and produces common artifacts conforming to the benchmark artifact spec.

## Architecture

```
run-benchmark.sh
  ├── platforms/          # Infrastructure-specific job submission
  │   ├── eks.py          # kubectl apply benchmark-job → collect results
  │   ├── hyperpod.py     # Slurm/SSM job submission → collect results
  │   └── local.py        # SSH + direct execution (bare metal, spot)
  ├── adapters/           # Tool output → common artifact conversion
  │   ├── vllm.py         # vLLM bench_serving.py output → artifact
  │   ├── sglang.py       # SGLang bench_serving.py output → artifact
  │   ├── genai_perf.py   # NVIDIA GenAI-Perf output → artifact
  │   └── recon_perf.py   # recon-perf native export (passthrough)
  ├── publish.py          # Artifact → PR to target repos
  ├── compare.py          # Cross-artifact comparison + regression detection
  └── run-benchmark.sh    # Entry point
```

## Usage

### Run a benchmark

```bash
# EKS: Run coding-agent workload against a running vLLM endpoint
./run-benchmark.sh \
  --platform eks \
  --endpoint http://kimi-k26-svc:8000 \
  --workload coding-agent \
  --sidecar blueprints/kimi-k2.6/benchmark.yaml \
  --output blueprints/kimi-k2.6/results/

# HyperPod: Submit benchmark as Slurm job
./run-benchmark.sh \
  --platform hyperpod \
  --cluster hp-inference-01 \
  --endpoint http://10.0.1.100:8000 \
  --workload qps-sweep \
  --sidecar blueprints/glm5-hyperpod/benchmark.yaml \
  --output blueprints/glm5-hyperpod/results/

# Local (bare metal / spot instance)
./run-benchmark.sh \
  --platform local \
  --host 35.94.217.100 \
  --endpoint http://localhost:8000 \
  --workload chatbot-short \
  --sidecar benchmarks/benchmark.yaml \
  --output ./results/
```

### Compare results

```bash
# Compare two artifacts
./compare.py \
  results/kimi-k26_eks_b300_vllm_coding-agent_20260422.json \
  results/kimi-k26_eks_b300_vllm-eagle3_coding-agent_20260511.json

# Regression detection (new vs baseline)
./compare.py --regression \
  --baseline results/baseline.json \
  --candidate results/new-config.json \
  --threshold 5  # flag if >5% regression on any core metric
```

### Publish to community repos

```bash
# Publish a blueprint's results to ai-on-eks
./publish.py \
  --target ai-on-eks \
  --blueprint domains/gpu-serving/blueprints/kimi-k2.6/ \
  --repo ~/repos/ai-on-eks \
  --dry-run  # preview without creating PR

# Publish to hyperpod-recipes
./publish.py \
  --target hyperpod-recipes \
  --blueprint domains/gpu-serving/blueprints/glm5-hyperpod/ \
  --repo ~/repos/sagemaker-hyperpod-recipes
```

## Workflow

### Standard benchmark flow

```
1. Deploy model (via RALPH loop / infra-deployer)
2. Author benchmark.yaml sidecar (once per blueprint)
3. Run: ./run-benchmark.sh --platform eks --workload coding-agent ...
   a. Submits benchmark job to platform
   b. Collects raw tool output (vLLM/SGLang JSON)
   c. Runs adapter to produce common artifact
   d. Validates artifact against JSON Schema
   e. Enriches with GPU telemetry (if scraper running)
   f. Writes to blueprint results/ directory
4. Analyze: benchmark-analyst agent reads artifacts
5. Compare: ./compare.py across configs/engines
6. Publish: ./publish.py --target ai-on-eks
```

### Incremental composition benchmarking

For specs like `kimi-k2.6-speculative.md` that test additive optimizations:

```bash
# Step 1: Baseline (already exists)
# Step 2: + EAGLE3
./run-benchmark.sh --platform eks --workload coding-agent \
  --sidecar benchmark-eagle3.yaml --tag "eagle3-only" ...

# Step 3: + EAGLE3 + dynamic MLA
./run-benchmark.sh --platform eks --workload coding-agent \
  --sidecar benchmark-eagle3-mla.yaml --tag "eagle3+mla" ...

# Step 4: Full stack
./run-benchmark.sh --platform eks --workload coding-agent \
  --sidecar benchmark-fullstack.yaml --tag "fullstack" ...

# Compare all layers
./compare.py --series \
  results/*baseline*.json \
  results/*eagle3-only*.json \
  results/*eagle3+mla*.json \
  results/*fullstack*.json
```

## Sidecar Config (benchmark.yaml)

The sidecar provides context that the benchmark tool doesn't output. Author once per blueprint session:

```yaml
# See standards/benchmark-commons/examples/nemotron-super/benchmark.yaml
model:
  name: Kimi-K2.6
  id: moonshotai/Kimi-K2.6
  architecture: moe
  parameters_total: "1T"
  parameters_active: "32B"
  quantization: int4
  max_model_len: 131072

infrastructure:
  substrate: eks
  instance_type: p6-b300.48xlarge
  region: us-west-2
  gpu:
    name: B300
    arch: sm_103
    count: 8
    vram_gb: 268
    interconnect: nvswitch
  eks:
    cluster_version: "1.32"
    node_count: 1

engine:
  name: vllm
  version: "0.19.1"
  tensor_parallel: 8
  prefix_caching: true
  gpu_memory_utilization: 0.90
  speculative_decoding:
    method: eagle3
    draft_model: lightseekorg/kimi-k2.6-eagle3
    num_steps: 3
    num_draft_tokens: 4

slo:
  ttft_p99_ms: 500
  tpot_p99_ms: 15
  error_rate_max: 0.001
```

## Platform Details

### EKS

- Submits benchmark as a Kubernetes Job (`benchmark-job.yaml` template)
- Collects results from pod logs or shared PVC
- Supports in-cluster (same namespace) and cross-cluster (via LoadBalancer endpoint)
- GPU telemetry scraped from DCGM Prometheus endpoint if available

### HyperPod

- Submits via SSM RunCommand or Slurm sbatch
- Collects results from shared FSx volume
- Leverages HyperPod deep health checks for pre-benchmark GPU validation
- Tags artifacts with `deep_health_checks: true` and `auto_recovery: true` metadata

### Local (Bare Metal / Spot)

- SSH to instance, runs benchmark tool directly
- Results pulled via scp
- Used for: g7e spot instances, capacity block B300, direct debugging
- Lowest overhead, most flexible

## Adapters

Each adapter transforms raw tool output + sidecar config into a validated common artifact.

| Adapter | Input | Notes |
|---------|-------|-------|
| `vllm.py` | `benchmark_serving` JSON output | Maps `p50_ttft_ms` → `metrics.ttft_ms.p50`, etc. |
| `sglang.py` | `bench_serving` JSON output | Computes p50/p90 from `--output-details` arrays |
| `genai_perf.py` | GenAI-Perf Parquet/CSV | Maps NVIDIA metric names to common schema |
| `recon_perf.py` | recon-perf native JSON | Near-passthrough (closest to spec already) |

## Publication Targets

| Target | Repo | Path convention | PR template |
|--------|------|-----------------|-------------|
| ai-on-eks | `awslabs/ai-on-eks` | `blueprints/inference/{name}/benchmarks/` | Includes README, artifacts, sidecar |
| hyperpod-recipes | `aws/sagemaker-hyperpod-recipes` | `inference/{model}/benchmarks/` | Includes recipe, artifacts, sidecar |
| S3 (aggregation) | `s3://benchmark-commonss/` | Flat JSONL for Athena queries | Append-only |
