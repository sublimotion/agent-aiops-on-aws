#!/usr/bin/env bash
# AI-Infra Lab — Experiment run plan.
#
# Drives the experiment sequence in cost order, with the B300 spot window
# fenced to a single block of work. Designed to be killable and resumable:
# each phase's results land in domains/ai-infra/blueprints/<spec>/results/
# and the script skips already-completed phases.
#
# Usage:
#   ./run-plan.sh prepare        # build images, validate, no GPU
#   ./run-plan.sh small          # Spec 0 small + Spec E + Spec F (cheap)
#   ./run-plan.sh b300-up        # scale B300 nodegroup to 1, wait for capacity
#   ./run-plan.sh b300           # Specs 0-large, B, C, D (expensive — needs B300)
#   ./run-plan.sh b300-down      # scale B300 nodegroup to 0
#   ./run-plan.sh teardown       # full teardown (B300=0, namespace cleanup)
#
# Stop at any time with Ctrl-C; re-run a phase to resume.

set -euo pipefail

REGION="${REGION:-us-west-2}"
CLUSTER="${CLUSTER:-qn-sglang-eks-cluster}"
NODEGROUP="${NODEGROUP:-ai-infra-b300-spot}"
NAMESPACE="${NAMESPACE:-ai-infra}"
KIMI_BUCKET="${KIMI_BUCKET:-kimi-k2-bench-models-20260216163240701700000006}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${REGISTRY:-${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/ai-infra}"

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
RESULTS_DIR="${REPO_DIR}/domains/ai-infra/blueprints"
PROFILER="${REPO_DIR}/domains/ai-infra/shared/profiler.py"
VALIDATE="${REPO_DIR}/domains/ai-infra/shared/profiler_validate.py"

export REGION CLUSTER NAMESPACE KIMI_BUCKET REGISTRY

phase=$1

ensure_namespace() {
  kubectl get ns "$NAMESPACE" >/dev/null 2>&1 \
    || kubectl create namespace "$NAMESPACE"
}

scale_nodegroup() {
  local desired=$1
  echo "==> Setting $NODEGROUP desired=$desired"
  aws eks update-nodegroup-config \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NODEGROUP" \
    --scaling-config "minSize=0,maxSize=1,desiredSize=$desired" \
    --region "$REGION" >/dev/null
}

wait_for_b300() {
  echo "==> Waiting for B300 node to be Ready (max 30 min)..."
  local deadline=$(( $(date +%s) + 1800 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if kubectl get nodes -l ai-infra/role=b300-spot --no-headers 2>/dev/null \
        | grep -q ' Ready '; then
      echo "==> B300 ready"
      return 0
    fi
    sleep 20
  done
  echo "B300 did not become Ready in 30 min" >&2
  return 1
}

apply_fixture() {
  local fixture=$1     # qwen3-next | kimi-k2.6
  local variant=$2     # baseline | runai-streamer | mx-2nd-pod | ...
  local image_tag=$3
  local experiment=$4
  ensure_namespace
  IMAGE_TAG="$image_tag" VARIANT_TAG="$variant" EXPERIMENT="$experiment" \
    KIMI_BUCKET="$KIMI_BUCKET" \
    envsubst < "$REPO_DIR/domains/ai-infra/staging/manifests/${fixture}-fixture.yaml" \
    | kubectl apply -f -
}

run_profile() {
  local fixture=$1 variant=$2 image_tag=$3 experiment=$4 vllm_version=$5 model=$6
  local out_dir="$RESULTS_DIR/$experiment/results"
  mkdir -p "$out_dir"
  local out="$out_dir/${variant}-$(date +%Y%m%d-%H%M%S).json"

  apply_fixture "$fixture" "$variant" "$image_tag" "$experiment"

  python3 "$PROFILER" \
    --manifest <(IMAGE_TAG="$image_tag" VARIANT_TAG="$variant" EXPERIMENT="$experiment" \
                 KIMI_BUCKET="$KIMI_BUCKET" \
                 envsubst < "$REPO_DIR/domains/ai-infra/staging/manifests/${fixture}-fixture.yaml") \
    --namespace "$NAMESPACE" \
    --endpoint "http://${fixture}-fixture-${variant}.${NAMESPACE}.svc:8000" \
    --model "$model" \
    --experiment "$experiment" \
    --variant "fixture=${fixture}" \
    --variant "variant=${variant}" \
    --fixture "domains/gpu-serving/blueprints/${fixture}" \
    --vllm-version "$vllm_version" \
    --out "$out"

  echo "==> Wrote $out"
  kubectl delete -n "$NAMESPACE" pod "${fixture}-fixture-${variant}" --wait=false 2>/dev/null || true
  kubectl delete -n "$NAMESPACE" svc "${fixture}-fixture-${variant}" --wait=false 2>/dev/null || true
}

case "$phase" in
  prepare)
    echo "==> Phase: prepare"
    echo "==> Build slim images on the build host (separately)."
    echo "==> Then: cd domains/ai-infra/staging/terraform && terraform apply"
    echo "==> This populates ECR with vllm-slim and sglang-slim repos."
    echo "==> Validate the profiler offline before scaling B300."
    ensure_namespace
    ;;

  small)
    echo "==> Phase: small (Spec 0 small fixture + Spec E + Spec F)"
    # Spec 0 small fixture: 5 cold starts of qwen3-next on whatever GPU lands.
    for i in 1 2 3 4 5; do
      run_profile qwen3-next "baseline-$i" "$REGISTRY/vllm-slim:0.21.0-cu128" \
        spec-0-profiler-validation 0.21.0 "Qwen/Qwen3-Next-80B-A3B-Instruct"
    done

    # Spec E: FUSE tuning. Runs on the build host or any non-GPU node.
    echo "==> Spec E (FUSE tuning) — run on build host with shared/images/"
    echo "    See specs/fuse-tuning-for-snapshotters.md"

    # Spec F: access profiling. Cheap GPU instance.
    echo "==> Spec F (access profiling) — needs eBPF on a GPU node"
    echo "    See specs/cold-start-access-profiling.md"

    # Validate the small Spec 0 batch.
    python3 "$VALIDATE" "$RESULTS_DIR/spec-0-profiler-validation/results"/*.json
    ;;

  b300-up)
    echo "==> Phase: b300-up (scaling spot nodegroup to 1)"
    scale_nodegroup 1
    wait_for_b300
    ;;

  b300)
    echo "==> Phase: b300 (Spec 0 large, Spec B, Spec C, Spec D)"
    echo "==> WARNING: B300 is running at \$26+/hr. Cost: \$5+/min."
    local budget_start
    budget_start=$(date +%s)

    # Spec 0 large fixture: 5 cold starts of Kimi K2.6 on B300.
    for i in 1 2 3 4 5; do
      run_profile kimi-k2.6 "baseline-$i" "$REGISTRY/vllm-slim:0.21.0-cu130" \
        spec-0-profiler-validation 0.21.0 "kimi-k2.6"
    done
    python3 "$VALIDATE" "$RESULTS_DIR/spec-0-profiler-validation/results"/*.json

    # Spec B variants. Each is one cold start measurement on Kimi K2.6.
    for variant in baseline-loadformat init-s5cmd-leanimg runai-streamer mx-1st-pod mx-2nd-pod; do
      run_profile kimi-k2.6 "$variant" "$REGISTRY/vllm-slim:0.21.0-cu130" \
        spec-b-model-decoupling 0.21.0 "kimi-k2.6"
    done

    # Spec C variants.
    for variant in cache-empty cache-baked cache-pvc cache-warmup-promote; do
      run_profile kimi-k2.6 "$variant" "$REGISTRY/vllm-slim:0.21.0-cu130" \
        spec-c-compile-cache 0.21.0 "kimi-k2.6"
    done

    # Spec D: stacked. Combines winners from A/B/C. Driven by a separate
    # script once A/B/C have selected variants.
    echo "==> Spec D (stacked) is run after A/B/C analysis lands. Skipping for now."

    local budget_elapsed=$(( $(date +%s) - budget_start ))
    echo "==> b300 phase elapsed: $((budget_elapsed/60)) min"
    ;;

  b300-down)
    echo "==> Phase: b300-down (scaling B300 nodegroup to 0)"
    scale_nodegroup 0
    ;;

  teardown)
    echo "==> Phase: teardown"
    scale_nodegroup 0
    kubectl delete namespace "$NAMESPACE" --wait=false 2>/dev/null || true
    echo "==> Manual: terraform destroy in staging/terraform/ if you want to remove ECR + nodegroup"
    ;;

  *)
    echo "Unknown phase: $phase" >&2
    echo "Phases: prepare | small | b300-up | b300 | b300-down | teardown" >&2
    exit 2
    ;;
esac
