#!/usr/bin/env bash
# Stage 0a PRE-FLIGHT DRY RUN — runs OFF the GPU. Fail-closed: exit !=0 BLOCKS the B200 launch.
# Every failed B200 cycle this engagement was a bug this script would have caught for ~$0 before staging.
# Usage: bash preflight-dry.sh   (run from anywhere; paths are absolute-from-repo-root via REPO)
set -uo pipefail
REPO="/Users/phi/Documents/workbench/agent-aiops-on-aws"
M2="${REPO}/domains/gpu-serving/blueprints/minimax-m2"
KVT="${REPO}/domains/gpu-serving/blueprints/minimax-m2-kv-tiering"
GEN="${M2}/k8s/gen-serving-manifest.sh"
fail=0
ok(){   echo "  PASS: $*"; }
bad(){  echo "  FAIL: $*"; fail=1; }

echo "== [1] single manifest generator (no split-brain) =="
n=$(find "${REPO}/domains/gpu-serving" -name gen-serving-manifest.sh -type f | wc -l | tr -d ' ')
if [ "$n" = "1" ]; then ok "exactly one gen-serving-manifest.sh"
else
  # allow >1 ONLY if byte-identical (e.g. intentional copies kept in sync)
  if find "${REPO}/domains/gpu-serving" -name gen-serving-manifest.sh -type f -print0 | xargs -0 md5 -q 2>/dev/null | sort -u | wc -l | grep -q '^ *1$'; then
    ok "$n copies but byte-identical"
  else bad "$n DIVERGENT copies of gen-serving-manifest.sh — sync or symlink (split-brain bug)"; fi
fi

echo "== [2] orchestrators: bash -n + no hardcoded wrong NG =="
for s in "${KVT}/k8s/run-tiering-sweep.sh" "${M2}/k8s/run-pareto-sweep.sh" "${KVT}/k8s/run-offload-retest.sh"; do
  [ -f "$s" ] || { echo "  skip (absent): $(basename "$s")"; continue; }
  bash -n "$s" 2>/dev/null && ok "syntax $(basename "$s")" || bad "syntax error in $(basename "$s")"
  # NG name must be the live multi-AZ one (a hardcoded stale name = idle-leak / wrong-NG scaledown)
  if grep -qE 'NODEGROUP=.*ai-infra-use2-b200-spot"' "$s" && ! grep -qE 'spot-maz' "$s"; then
    bad "$(basename "$s") hardcodes the OLD single-AZ NG (ai-infra-use2-b200-spot) — must be -maz"
  fi
done

echo "== [3] manifest JSON validates with v0.23 keys (not v0.11) =="
for arm in gpu-only cpu-offload nvme-tiering; do
  bash "$GEN" tp4 "$arm" 2>/dev/null | grep -q "kv-transfer-config\|gpu-only" || { [ "$arm" = gpu-only ] || bad "gen-manifest produced nothing for $arm"; }
  if [ "$arm" != gpu-only ]; then
    line=$(bash "$GEN" tp4 "$arm" 2>/dev/null | grep "kv-transfer-config")
    echo "$line" | grep -q "num_cpu_blocks\|nvme_path\|num_nvme_blocks" && bad "$arm uses STALE v0.11 keys (num_cpu_blocks/nvme_path) — v0.23 wants cpu_bytes_to_use"
    echo "$line" | python3 -c "
import sys,re,json
m=re.search(r\"--kv-transfer-config '(\{.*?\})'\", sys.stdin.read())
assert m, 'no kv-transfer-config JSON'
d=json.loads(m.group(1)); ec=d['kv_connector_extra_config']
assert 'cpu_bytes_to_use' in ec, 'missing cpu_bytes_to_use (v0.23 required key)'
" 2>/dev/null && ok "$arm JSON valid + cpu_bytes_to_use present" || bad "$arm JSON invalid or missing cpu_bytes_to_use"
  fi
done
# tp4ep4 must generate (it's a SHAPE, not a kv_arm — boot_gate must call it as shape)
bash "$GEN" tp4ep4 gpu-only >/dev/null 2>&1 && ok "tp4ep4 shape generates" || bad "tp4ep4 gen-manifest failed (shape/arm bug?)"

echo "== [4] engine image is v0.23.0, not stale minimax27 =="
if grep -q "vllm/vllm-openai:v0.23.0" "$GEN" && ! grep -q "minimax27" "$GEN"; then ok "image v0.23.0"
else bad "gen-manifest image is not v0.23.0 (stale minimax27?)"; fi

echo "== [4b] served-model-name matches the bench client's --model (else every request 404s) =="
# v0.23 serves the model under --served-model-name; bench.py POSTs {"model":"MiniMax-M2"}. If the manifest
# omits --served-model-name, vLLM serves it under the --model PATH and returns http404 for "MiniMax-M2"
# (this zeroed 2 full sweeps: pods health-200 but every request 404).
bench_model=$(grep -oE '\-\-model MiniMax-M2' "${M2}/k8s/run-pareto-sweep.sh" | head -1)
if grep -q "served-model-name MiniMax-M2" "$GEN"; then ok "served-model-name MiniMax-M2 present (matches bench --model)"
else bad "gen-manifest has no '--served-model-name MiniMax-M2' — bench client's model='MiniMax-M2' will 404"; fi

echo "== [5] no external pgrep scaledown supervisor checked into the runners =="
if grep -rlE 'pgrep.*(run-p1p2|run-tiering|run-offload|run-pareto)' "${KVT}/k8s" "${M2}/k8s" 2>/dev/null | grep -q .; then
  bad "a runner contains a pgrep-based scaledown supervisor — remove it (false-fires, kills node mid-run)"
else ok "no external pgrep scaledown supervisor"; fi

echo ""
if [ "$fail" = 0 ]; then echo "PREFLIGHT DRY RUN: ALL PASS — safe to scale a B200."; exit 0
else echo "PREFLIGHT DRY RUN: FAILED — do NOT scale a B200 until the FAILs above are fixed."; exit 1; fi
