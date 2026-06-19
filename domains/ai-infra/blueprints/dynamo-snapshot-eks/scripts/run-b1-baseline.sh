#!/usr/bin/env bash
# B1 baseline — 5 cold-start measurements of Ministral-3B (no snapshot).
#
# Usage:  ./run-b1-baseline.sh [results-dir]
#
# Prereq:
#   - g7e nodegroup is up (iter 5b Stage B has launched)
#   - Manifests 00-50 are applied
#   - Deployment ministral-3b replicas=1 is healthy at least once
#
# Each measurement: scale to 0, wait Terminated, scale to 1, time pod-create
# until first POST /v1/completions returns a token. Records to JSON.
set -euo pipefail

NS=dynamo-snapshot
DEP=ministral-3b
SVC=ministral-3b
RUNS=${RUNS:-5}
RESULTS_DIR=${1:-results/e1/b1-baseline}
mkdir -p "$RESULTS_DIR"

ts() { date +%s.%N; }

run_once() {
  local idx=$1
  echo "=== B1 run $idx ==="
  kubectl -n "$NS" scale deploy/"$DEP" --replicas=0
  kubectl -n "$NS" wait --for=delete pod -l app="$DEP" --timeout=180s || true

  local t_start
  t_start=$(ts)
  kubectl -n "$NS" scale deploy/"$DEP" --replicas=1

  # Wait for the pod to be Ready (vLLM /v1/models 200), then issue a probe completion.
  kubectl -n "$NS" wait --for=condition=Ready pod -l app="$DEP" --timeout=600s
  local t_ready
  t_ready=$(ts)

  # Issue a tiny deterministic completion via the cluster service. We run from
  # a one-shot pod inside the cluster to avoid port-forward noise.
  local t_first_token
  t_first_token=$(kubectl -n "$NS" run --rm -i --restart=Never --image=curlimages/curl:8.10.1 \
    "b1-probe-$idx" -- sh -c "
      START=\$(date +%s.%N)
      curl -fsS -m 60 -X POST http://${SVC}:8000/v1/completions \
        -H 'Content-Type: application/json' \
        -d '{\"model\":\"ministral-3b\",\"prompt\":\"hello\",\"max_tokens\":1,\"temperature\":0,\"seed\":42,\"top_p\":1}' \
        > /dev/null
      END=\$(date +%s.%N)
      echo \$END
    " 2>/dev/null | tail -1)

  local cold=$(awk -v a="$t_start" -v b="$t_ready" 'BEGIN{printf "%.3f", b-a}')
  local first=$(awk -v a="$t_start" -v b="$t_first_token" 'BEGIN{printf "%.3f", b-a}')
  cat > "$RESULTS_DIR/b1-run-$idx.json" <<EOF
{
  "run": $idx,
  "t_start": $t_start,
  "t_ready": $t_ready,
  "t_first_token": $t_first_token,
  "seconds_to_ready": $cold,
  "seconds_to_first_token": $first
}
EOF
  echo "B1 run $idx: ready=${cold}s first_token=${first}s"
}

for i in $(seq 1 "$RUNS"); do
  run_once "$i"
done

# Summary
python3 - <<EOF
import json, glob, statistics
runs=[json.load(open(p)) for p in sorted(glob.glob("$RESULTS_DIR/b1-run-*.json"))]
ready=[r["seconds_to_ready"] for r in runs]
ftt=[r["seconds_to_first_token"] for r in runs]
out={
  "n": len(runs),
  "seconds_to_ready": {"p50": statistics.median(ready), "min": min(ready), "max": max(ready), "mean": statistics.mean(ready)},
  "seconds_to_first_token": {"p50": statistics.median(ftt), "min": min(ftt), "max": max(ftt), "mean": statistics.mean(ftt)},
  "raw": runs,
}
json.dump(out, open("$RESULTS_DIR/summary.json","w"), indent=2)
print(json.dumps(out, indent=2))
EOF
