#!/usr/bin/env bash
set -euo pipefail

# Benchmark Runner — Entry Point
# Orchestrates benchmark execution on AWS platforms and produces common artifacts.
#
# Usage:
#   ./run-benchmark.sh --platform eks --endpoint http://svc:8000 --workload coding-agent \
#     --sidecar benchmark.yaml --output ./results/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKLOAD_DIR="$ARTIFACT_DIR/workloads"
SCHEMA_DIR="$ARTIFACT_DIR/container/schema"

# Defaults
PLATFORM="local"
ENDPOINT=""
WORKLOAD=""
SIDECAR=""
OUTPUT_DIR="./results"
TOOL="vllm"  # vllm, sglang, genai-perf, recon-perf
TAG=""
NUM_PROMPTS=""
DURATION=""
DRY_RUN=false
# CTO engagement extensions
POWER_SAMPLER="auto"          # auto | dcgm | nvidia-smi | off
DCGM_URL="http://localhost:9400/metrics"
QUALITY_EVALS=""              # comma-separated: mmlu,gsm8k,banking77
QUALITY_GATE="true"           # if true, fail the run when any eval fails
LOAD_FRACTION=""              # O11 sweep: 0.25 / 0.50 / 0.75 / 1.0
BURN_IN_HOURS=""              # O5 burn-in; when set, run stability capture
SENTINEL_PROMPTS=""           # O10 golden-prompt file path

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Required:
  --endpoint URL       Serving endpoint (e.g., http://localhost:8000)
  --workload ID        Workload catalog ID (e.g., coding-agent, chatbot-short, qps-sweep)
  --sidecar PATH       benchmark.yaml sidecar config

Options:
  --platform PLATFORM  eks | hyperpod | local (default: local)
  --tool TOOL          vllm | sglang | genai-perf | recon-perf (default: vllm)
  --output DIR         Output directory (default: ./results)
  --tag TAG            Optional tag for this run (e.g., "eagle3-only", "fullstack")
  --num-prompts N      Override workload num_prompts
  --duration N         Override workload duration_s
  --dry-run            Print commands without executing
  -h, --help           Show this help

CTO engagement extensions (for O3/O5/O9/O10/O11 cells):
  --power-sampler S    auto | dcgm | nvidia-smi | off (default: auto)
  --dcgm-url URL       DCGM exporter URL (default: http://localhost:9400/metrics)
  --quality-eval E     Comma-separated evals to gate on (mmlu,gsm8k,banking77,docvqa,librispeech)
  --no-quality-gate    Treat quality eval failures as warnings, not errors
  --load-fraction F    Record O11 load fraction (0.25 / 0.50 / 0.75 / 1.0)
  --burn-in-hours H    O5 burn-in mode: run load for H hours and capture drift
  --sentinel PATH      O10 golden-prompt JSON for silent-corruption canary

Workloads:
  chatbot-short        256 tok in, 128 out, 2 QPS
  chatbot-long         32K in, 512 out, 0.5 QPS
  batch-throughput     2048 in, 512 out, max rate
  rag-long-context     16K shared prefix, 256 out, 1 QPS
  coding-agent         4096 in, 2048 out, 4 QPS
  qps-sweep            2048 in, 512 out, sweep 0.5-16
  concurrency-sweep    2048 in, 256 out, sweep c=1-512

Examples:
  # Run coding-agent workload against local vLLM
  $0 --endpoint http://localhost:8000 --workload coding-agent --sidecar benchmark.yaml

  # Run on EKS with tag
  $0 --platform eks --endpoint http://kimi-svc:8000 --workload qps-sweep \\
     --sidecar benchmark.yaml --tag "baseline-v0.19.1"

  # Dry run to see what would execute
  $0 --endpoint http://localhost:8000 --workload chatbot-short \\
     --sidecar benchmark.yaml --dry-run
EOF
  exit 0
}

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --platform) PLATFORM="$2"; shift 2 ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --sidecar) SIDECAR="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --tool) TOOL="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --power-sampler) POWER_SAMPLER="$2"; shift 2 ;;
    --dcgm-url) DCGM_URL="$2"; shift 2 ;;
    --quality-eval) QUALITY_EVALS="$2"; shift 2 ;;
    --no-quality-gate) QUALITY_GATE="false"; shift ;;
    --load-fraction) LOAD_FRACTION="$2"; shift 2 ;;
    --burn-in-hours) BURN_IN_HOURS="$2"; shift 2 ;;
    --sentinel) SENTINEL_PROMPTS="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Validate required args
[[ -z "$ENDPOINT" ]] && echo "Error: --endpoint required" && exit 1
[[ -z "$WORKLOAD" ]] && echo "Error: --workload required" && exit 1
[[ -z "$SIDECAR" ]] && echo "Error: --sidecar required" && exit 1
[[ ! -f "$SIDECAR" ]] && echo "Error: sidecar not found: $SIDECAR" && exit 1

# Load workload config
WORKLOAD_FILE="$WORKLOAD_DIR/$WORKLOAD.yaml"
if [[ ! -f "$WORKLOAD_FILE" ]]; then
  echo "Error: workload not found: $WORKLOAD_FILE"
  echo "Available workloads:"
  ls "$WORKLOAD_DIR"/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | sed 's/^/  /'
  exit 1
fi

# Generate output filename
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
MODEL_SLUG=$(python3 -c "
import yaml
with open('$SIDECAR') as f:
    cfg = yaml.safe_load(f)
name = cfg.get('model', {}).get('name', 'unknown')
print(name.lower().replace(' ', '-').replace('/', '-')[:30])
")
ENGINE_SLUG=$(python3 -c "
import yaml
with open('$SIDECAR') as f:
    cfg = yaml.safe_load(f)
engine = cfg.get('engine', {}).get('name', 'unknown')
version = cfg.get('engine', {}).get('version', '')
print(f'{engine}')
")
INFRA_SLUG=$(python3 -c "
import yaml
with open('$SIDECAR') as f:
    cfg = yaml.safe_load(f)
substrate = cfg.get('infrastructure', {}).get('substrate', 'unknown')
instance = cfg.get('infrastructure', {}).get('instance_type', 'unknown').replace('.', '-')
print(f'{substrate}_{instance}')
")

if [[ -n "$TAG" ]]; then
  ARTIFACT_NAME="${MODEL_SLUG}_${INFRA_SLUG}_${ENGINE_SLUG}-${TAG}_${WORKLOAD}_${TIMESTAMP}"
else
  ARTIFACT_NAME="${MODEL_SLUG}_${INFRA_SLUG}_${ENGINE_SLUG}_${WORKLOAD}_${TIMESTAMP}"
fi

echo "=== Benchmark Runner ==="
echo "Platform:  $PLATFORM"
echo "Endpoint:  $ENDPOINT"
echo "Workload:  $WORKLOAD ($WORKLOAD_FILE)"
echo "Tool:      $TOOL"
echo "Sidecar:   $SIDECAR"
echo "Output:    $OUTPUT_DIR/$ARTIFACT_NAME.json"
echo "Tag:       ${TAG:-<none>}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
ENGAGEMENT_DIR="$OUTPUT_DIR/engagement_${ARTIFACT_NAME}"
mkdir -p "$ENGAGEMENT_DIR"

# ---------------------------------------------------------------------------
# Step 0a: Quality gate (O3) — runs BEFORE throughput capture so a failing
# precision config never produces a reportable throughput row.
# ---------------------------------------------------------------------------
QUALITY_PATHS=()
if [[ -n "$QUALITY_EVALS" && "$DRY_RUN" == "false" ]]; then
  echo "--- Step 0a: Quality gate (O3) ---"
  MODEL_NAME_FOR_EVAL=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$SIDECAR'))
print(cfg.get('model', {}).get('id', cfg.get('model', {}).get('name', 'unknown')))
")
  IFS=',' read -ra EVALS <<<"$QUALITY_EVALS"
  for eval_name in "${EVALS[@]}"; do
    BASELINE=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$SIDECAR'))
baselines = cfg.get('quality_baselines', {})
print(baselines.get('$eval_name', {}).get('bf16', ''))
")
    TOLERANCE=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$SIDECAR'))
baselines = cfg.get('quality_baselines', {})
print(baselines.get('$eval_name', {}).get('tolerance', 0.02))
")
    if [[ -z "$BASELINE" ]]; then
      echo "  [skip] $eval_name — no baseline in sidecar.quality_baselines.$eval_name.bf16"
      continue
    fi
    QPATH="$ENGAGEMENT_DIR/quality_${eval_name}.json"
    set +e
    python3 "$SCRIPT_DIR/../container/run-quality-eval.py" \
      --eval "$eval_name" \
      --endpoint "$ENDPOINT" \
      --model "$MODEL_NAME_FOR_EVAL" \
      --baseline-score "$BASELINE" \
      --tolerance "$TOLERANCE" \
      --output "$QPATH"
    rc=$?
    set -e
    QUALITY_PATHS+=("$(realpath --relative-to="$(dirname "$SIDECAR")" "$QPATH")")
    if [[ $rc -ne 0 && "$QUALITY_GATE" == "true" ]]; then
      echo "Quality gate FAILED for $eval_name. Aborting throughput run (O3 rule)."
      exit 2
    fi
  done
fi

# Step 1: Execute benchmark via platform
echo "--- Step 1: Execute benchmark ($PLATFORM) ---"
RAW_OUTPUT="$OUTPUT_DIR/.raw_${ARTIFACT_NAME}.json"

# ---------------------------------------------------------------------------
# Step 1a: Start background power + error sampler (O10 + O11).
# Duration heuristic: burn-in-hours wins, else workload.duration_s or 900s.
# ---------------------------------------------------------------------------
POWER_JSON=""
POWER_PID=""
if [[ "$POWER_SAMPLER" != "off" && "$DRY_RUN" == "false" ]]; then
  if [[ -n "$BURN_IN_HOURS" ]]; then
    POWER_DURATION=$(python3 -c "print(int(float('$BURN_IN_HOURS') * 3600))")
  else
    POWER_DURATION=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$WORKLOAD_FILE'))
dur = cfg.get('load', {}).get('duration_s')
if dur: print(dur)
else:   print(900)
")
  fi
  POWER_JSON="$ENGAGEMENT_DIR/power.json"
  SAMPLER_FLAG=""
  if [[ "$POWER_SAMPLER" == "nvidia-smi" ]]; then
    SAMPLER_FLAG="--nvidia-smi"
  elif [[ "$POWER_SAMPLER" == "auto" ]] && ! curl -sf -m 2 "$DCGM_URL" >/dev/null 2>&1; then
    echo "  DCGM exporter unreachable at $DCGM_URL — falling back to nvidia-smi"
    SAMPLER_FLAG="--nvidia-smi"
  fi
  echo "  Power sampler: ${POWER_DURATION}s → $POWER_JSON"
  python3 "$SCRIPT_DIR/../container/scrape-power.py" \
    --dcgm-url "$DCGM_URL" \
    --duration "$POWER_DURATION" \
    --output "$POWER_JSON" \
    $SAMPLER_FLAG &
  POWER_PID=$!
fi

case $PLATFORM in
  local)
    python3 "$SCRIPT_DIR/platforms/local.py" \
      --endpoint "$ENDPOINT" \
      --workload "$WORKLOAD_FILE" \
      --tool "$TOOL" \
      --output "$RAW_OUTPUT" \
      ${NUM_PROMPTS:+--num-prompts "$NUM_PROMPTS"} \
      ${DURATION:+--duration "$DURATION"} \
      ${DRY_RUN:+--dry-run}
    ;;
  eks)
    python3 "$SCRIPT_DIR/platforms/eks.py" \
      --endpoint "$ENDPOINT" \
      --workload "$WORKLOAD_FILE" \
      --tool "$TOOL" \
      --output "$RAW_OUTPUT" \
      ${NUM_PROMPTS:+--num-prompts "$NUM_PROMPTS"} \
      ${DURATION:+--duration "$DURATION"} \
      ${DRY_RUN:+--dry-run}
    ;;
  hyperpod)
    python3 "$SCRIPT_DIR/platforms/hyperpod.py" \
      --endpoint "$ENDPOINT" \
      --workload "$WORKLOAD_FILE" \
      --tool "$TOOL" \
      --output "$RAW_OUTPUT" \
      ${NUM_PROMPTS:+--num-prompts "$NUM_PROMPTS"} \
      ${DURATION:+--duration "$DURATION"} \
      ${DRY_RUN:+--dry-run}
    ;;
  *)
    echo "Error: unknown platform: $PLATFORM"
    exit 1
    ;;
esac

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run complete. No benchmark executed."
  exit 0
fi

# Wait for background power sampler to finish
if [[ -n "$POWER_PID" ]]; then
  echo "  Waiting for power sampler (PID $POWER_PID)..."
  wait "$POWER_PID" || echo "  [warn] power sampler exited non-zero"
fi

# ---------------------------------------------------------------------------
# Step 1b: Inject engagement artifact paths into a temp sidecar so adapters
# can merge them without the user hand-editing benchmark.yaml.
# ---------------------------------------------------------------------------
ENRICHED_SIDECAR="$ENGAGEMENT_DIR/sidecar.yaml"
python3 - <<PYEOF
import yaml, os
cfg = yaml.safe_load(open("$SIDECAR"))
artifacts = cfg.setdefault("artifacts", {})
power_json = "$POWER_JSON"
if power_json and os.path.exists(power_json):
    artifacts["power"] = os.path.relpath(power_json, os.path.dirname("$SIDECAR"))
quality_paths = """${QUALITY_PATHS[@]:-}""".split()
if quality_paths:
    artifacts["quality"] = quality_paths
if "$LOAD_FRACTION":
    artifacts["load_fraction"] = float("$LOAD_FRACTION")
yaml.safe_dump(cfg, open("$ENRICHED_SIDECAR", "w"))
PYEOF

# Step 2: Convert raw output to common artifact
echo ""
echo "--- Step 2: Convert to common artifact ($TOOL adapter) ---"
ARTIFACT_OUTPUT="$OUTPUT_DIR/$ARTIFACT_NAME.json"

python3 "$SCRIPT_DIR/adapters/${TOOL}.py" \
  --raw "$RAW_OUTPUT" \
  --sidecar "$ENRICHED_SIDECAR" \
  --workload "$WORKLOAD_FILE" \
  --output "$ARTIFACT_OUTPUT"

# Step 3: Validate against schema
echo ""
echo "--- Step 3: Validate artifact ---"
python3 "$ARTIFACT_DIR/container/validate-artifact.py" "$ARTIFACT_OUTPUT"

# Step 4: Summary
echo ""
echo "--- Results ---"
python3 -c "
import json
with open('$ARTIFACT_OUTPUT') as f:
    art = json.load(f)
m = art['metrics']
print(f\"  Completed:    {m['completed']} requests\")
print(f\"  Error rate:   {m['error_rate']*100:.1f}%\")
print(f\"  TTFT p50:     {m['ttft_ms']['p50']:.1f} ms\")
print(f\"  TTFT p99:     {m['ttft_ms']['p99']:.1f} ms\")
print(f\"  TPOT p50:     {m['tpot_ms']['p50']:.2f} ms\")
print(f\"  Output tok/s: {m['output_toks_per_s']:.1f}\")
if 'slo' in art and art['slo']:
    print(f\"  SLO pass:     {art['slo']['overall_pass']}\")
if 'quality' in art:
    print(f\"  Quality gate: {art['quality']['gate_passed']} ({len(art['quality']['evals'])} evals)\")
if 'power' in art and art['power'].get('tokens_per_joule') is not None:
    print(f\"  Tokens/joule: {art['power']['tokens_per_joule']:.2f}\")
    print(f\"  Avg power:    {art['power'].get('avg_fleet_power_watts', 0):.1f} W fleet\")
if 'hardware_errors' in art:
    e = art['hardware_errors']
    print(f\"  ECC SBE/DBE:  {e.get('ecc_sbe_delta', 0)} / {e.get('ecc_dbe_delta', 0)}\")
if 'stability' in art:
    s = art['stability']
    print(f\"  Drift:        {s.get('throughput_drift_pct', 0):.2f}% over {s.get('duration_hours', 0)}h\")
"

echo ""
echo "Artifact: $ARTIFACT_OUTPUT"
echo "Raw data: $RAW_OUTPUT"
