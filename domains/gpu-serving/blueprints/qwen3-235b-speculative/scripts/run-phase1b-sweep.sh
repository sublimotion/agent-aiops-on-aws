#!/usr/bin/env bash
# Phase 1b sweep — cycle SGLang pod through 13 EAGLE3 configs.
# Expects kubectl context already set to qn-sglang.

set -euo pipefail
CTX="${KUBECTL_CONTEXT:-qn-sglang}"
KC="kubectl --context $CTX"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MANIFEST_TEMPLATE=$SCRIPT_DIR/../k8s/sglang-eagle3-phase1.yaml

cycle_pod() {
  local steps=$1 draft=$2 topk=$3
  local tag="s${steps}d${draft}k${topk}"
  local out_dir="/results/phase-1b/${tag}"

  # Check idempotency
  if $KC exec bench-runner -- test -f "$out_dir/done" 2>/dev/null; then
    echo "[skip] $tag"
    return 0
  fi

  echo "[cycle] $tag"
  $KC delete pod sglang-qwen3-235b-eagle3 --ignore-not-found --wait=true --timeout=120s 2>&1 | tail -2

  # Generate manifest with substituted params
  local tmp=$(mktemp)
  sed \
    -e "s/--speculative-num-steps 3/--speculative-num-steps $steps/" \
    -e "s/--speculative-num-draft-tokens 4/--speculative-num-draft-tokens $draft/" \
    -e "s/--speculative-eagle-topk 1/--speculative-eagle-topk $topk/" \
    "$MANIFEST_TEMPLATE" > "$tmp"

  $KC apply -f "$tmp" 2>&1 | tail -2
  rm "$tmp"

  # Wait for health
  echo "[wait] health check for $tag"
  for i in $(seq 1 60); do
    if $KC exec sglang-qwen3-235b-eagle3 -- curl -sf -m 5 http://localhost:30000/health 2>/dev/null; then
      echo "[ready] $tag after $((i*10))s"
      break
    fi
    sleep 10
  done

  # Wait 3s for first-token histogram bucket to flush
  sleep 3

  # Run sweep
  $KC exec bench-runner -- mkdir -p "$out_dir"
  for c in 1 8 32 64 128; do
    local total=$((c * 4))
    [ $c -ge 64 ] && total=$((c + c / 2))
    $KC exec bench-runner -- python3 /scripts/bench-one.py 30000 $c 512 256 $total "$out_dir/c${c}.json" sglang >/dev/null 2>&1 || echo "[fail] $tag c=$c"
  done
  $KC exec bench-runner -- touch "$out_dir/done"
  echo "[done] $tag"
}

# Pruned sweep: skip where num_steps > num_draft_tokens (infeasible)
for steps in 1 2 3 4; do
  for draft in 2 4 6 8; do
    if (( steps > draft )); then continue; fi
    cycle_pod $steps $draft 1 || echo "[skipped] s${steps}d${draft}k1"
  done
done

# Summary
$KC exec bench-runner -- python3 -c "
import json, glob, os
print(f'{\"tag\":<10} {\"c1/req\":>8} {\"c1/agg\":>8} {\"c8\":>7} {\"c32\":>7} {\"c64\":>7} {\"c128\":>7} {\"acc_len\":>8}')
for d in sorted(glob.glob('/results/phase-1b/*')):
    if not os.path.isdir(d): continue
    tag = os.path.basename(d)
    row = [tag]
    acc = None
    for c in [1,8,32,64,128]:
        f = f'{d}/c{c}.json'
        if os.path.exists(f):
            j = json.load(open(f))
            if c == 1:
                row.append(f'{j[\"per_req_tok_per_s\"]:.0f}')
                row.append(f'{j[\"agg_tok_per_s\"]:.0f}')
            else:
                row.append(f'{j[\"agg_tok_per_s\"]:.0f}')
            acc = j.get('spec_accept_length_mean')
        else:
            row.append('-')
            if c == 1: row.append('-')
    al = f'{acc:.2f}' if acc else '-'
    print(f'{row[0]:<10} {row[1]:>8} {row[2]:>8} {row[3]:>7} {row[4]:>7} {row[5]:>7} {row[6]:>7} {al:>8}')
"

echo "[phase-1b] complete"
