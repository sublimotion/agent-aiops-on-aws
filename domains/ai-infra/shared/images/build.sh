#!/usr/bin/env bash
# Build slim vLLM and SGLang images.
#
# Usage:
#   ./build.sh                  # build all variants
#   ./build.sh vllm cu128       # one engine + one CUDA variant
#   ./build.sh sglang cu130
#
# Requires docker buildx. Pushes are NOT automatic -- inspect locally first,
# tag for a registry separately.

set -euo pipefail

# Pinned versions. Bump these explicitly when upstream releases land.
VLLM_VERSION="${VLLM_VERSION:-0.21.0}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.12}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
PY_VERSION="${PY_VERSION:-3.12}"

REGISTRY="${REGISTRY:-ai-infra}"

declare -A CUDA_TAGS=(
  [cu128]="12.8.2"
  [cu130]="13.0.3"
)

ENGINES=("${1:-all}")
VARIANTS=("${2:-all}")

if [[ "${ENGINES[0]}" == "all" ]]; then
  ENGINES=(vllm sglang)
fi
if [[ "${VARIANTS[0]}" == "all" ]]; then
  VARIANTS=(cu128 cu130)
fi

build_one() {
  local engine="$1"
  local variant="$2"
  local cuda_tag="${CUDA_TAGS[$variant]:-}"
  if [[ -z "$cuda_tag" ]]; then
    echo "unknown variant $variant" >&2
    exit 2
  fi

  local version dockerfile tag
  case "$engine" in
    vllm)
      version="$VLLM_VERSION"
      dockerfile="Dockerfile.vllm-slim"
      tag="${REGISTRY}/vllm-slim:${VLLM_VERSION}-${variant}"
      ;;
    sglang)
      version="$SGLANG_VERSION"
      dockerfile="Dockerfile.sglang-slim"
      tag="${REGISTRY}/sglang-slim:${SGLANG_VERSION}-${variant}"
      ;;
    *)
      echo "unknown engine $engine" >&2
      exit 2
      ;;
  esac

  echo "==> Building ${tag} (CUDA=${cuda_tag}, py=${PY_VERSION}, version=${version})"

  local version_arg
  case "$engine" in
    vllm)   version_arg="VLLM_VERSION=${version}" ;;
    sglang) version_arg="SGLANG_VERSION=${version}" ;;
  esac

  docker buildx build \
    --build-arg "CUDA_TAG=${cuda_tag}" \
    --build-arg "PY_VERSION=${PY_VERSION}" \
    --build-arg "${version_arg}" \
    --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
    -t "${tag}" \
    -f "${dockerfile}" \
    --load \
    "$(dirname "$0")"
}

for engine in "${ENGINES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    build_one "$engine" "$variant"
  done
done

echo
echo "Build complete. Compare sizes with: ./compare-sizes.sh"
