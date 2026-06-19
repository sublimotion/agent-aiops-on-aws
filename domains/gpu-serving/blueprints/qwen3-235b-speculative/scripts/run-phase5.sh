#!/usr/bin/env bash
# Phase 5 frontier — 5a default / 5b no-cuda-graph / 5c TP2+DP2
set -euo pipefail
CTX="${KUBECTL_CONTEXT:-qn-sglang}"
KC="kubectl --context $CTX"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_MANIFEST=$SCRIPT_DIR/../k8s/sglang-phase4-fullstack.yaml

cycle() {
  local variant=$1
  local extra_args=$2
  local tp=$3
  local dp=$4
  local tag="${variant}"
  local out_dir="/results/phase-5/${tag}"

  if $KC exec bench-runner -- test -f "$out_dir/done" 2>/dev/null; then
    echo "[skip] $tag"; return 0
  fi

  echo "[cycle] $tag  extra='$extra_args'  tp=$tp dp=$dp"
  $KC delete pod sglang-qwen3-235b-eagle3 --ignore-not-found --wait=true --timeout=120s 2>&1 | tail -2

  local tmp=$(mktemp)
  cp "$BASE_MANIFEST" "$tmp"
  # Inject extra args before `--enable-metrics`
  sed -i "s|--enable-metrics \\\\|$extra_args \\\\n            --enable-metrics \\\\|" "$tmp" || true
  # Adjust TP
  if [[ "$tp" != "4" ]]; then
    sed -i "s|--tp 4 |--tp $tp |" "$tmp"
  fi
  # Add DP flag if needed
  if [[ -n "$dp" && "$dp" != "1" ]]; then
    sed -i "s|--tp $tp |--tp $tp --dp $dp |" "$tmp"
  fi

  $KC apply -f "$tmp" 2>&1 | tail -2
  rm "$tmp"

  echo "[wait] health for $tag"
  local ready=0
  for i in $(seq 1 90); do
    if $KC exec sglang-qwen3-235b-eagle3 -- curl -sf -m 5 http://localhost:30000/health 2>/dev/null >/dev/null; then
      echo "[ready] $tag after $((i*10))s"; ready=1; break
    fi
    sleep 10
  done
  if [[ "$ready" != "1" ]]; then
    echo "[skip-fail] $tag — never became healthy"
    $KC logs sglang-qwen3-235b-eagle3 --tail=15 2>&1 | tail -15
    $KC exec bench-runner -- mkdir -p "$out_dir"
    $KC exec bench-runner -- touch "$out_dir/skipped"
    return 0
  fi
  sleep 3

  $KC exec bench-runner -- mkdir -p "$out_dir"
  for c in 1 8 32 64 128 256; do
    local total=$((c * 4))
    [ $c -ge 64 ] && total=$((c + c / 2))
    $KC exec bench-runner -- python3 /scripts/bench-one.py 30000 $c 512 256 $total "$out_dir/c${c}.json" sglang >/dev/null 2>&1 || echo "[fail] $tag c=$c"
  done
  $KC exec bench-runner -- touch "$out_dir/done"
  echo "[done] $tag"
}

# 5a: default stack (= Phase 4, rerun for labeling)
cycle "5a-default-stack" "" 4 ""
# 5b: --disable-cuda-graph (isolate graph benefit)
cycle "5b-no-cuda-graph" "--disable-cuda-graph" 4 ""
# 5c: TP2+DP2 (multi-tenant — smaller model, TP2 comfortable)
cycle "5c-tp2-dp2" "" 2 "2"

# Summary
$KC exec bench-runner -- python3 -c "
import json, glob, os
print(f'{\"variant\":<20} {\"c=1 per_req\":>13} {\"c=8\":>8} {\"c=32\":>7} {\"c=64\":>7} {\"c=128\":>7} {\"c=256\":>7} {\"acc_len\":>8}')
for d in sorted(glob.glob('/results/phase-5/*')):
    if not os.path.isdir(d) or not os.path.exists(f'{d}/done'): continue
    tag = os.path.basename(d)
    row = [tag]
    acc = None
    for c in [1,8,32,64,128,256]:
        f = f'{d}/c{c}.json'
        if not os.path.exists(f):
            row.append('-'); continue
        j = json.load(open(f))
        if c == 1: row.append(f'{j[\"per_req_tok_per_s\"]:.0f}')
        else: row.append(f'{j[\"agg_tok_per_s\"]:.0f}')
        acc = j.get('spec_accept_length_mean')
    al = f'{acc:.2f}' if acc else '-'
    print(f'{row[0]:<20} {row[1]:>13} {row[2]:>8} {row[3]:>7} {row[4]:>7} {row[5]:>7} {row[6]:>7} {al:>8}')
"
echo "[phase-5] complete"
