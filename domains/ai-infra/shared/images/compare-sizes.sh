#!/usr/bin/env bash
# Compare upstream vLLM/SGLang image sizes against our slim builds.
#
# Pulls upstream defaults if not already cached, then prints a delta table.
# Reports both compressed (registry size, via docker manifest) and
# uncompressed (local disk size, via docker image inspect).

set -euo pipefail

VLLM_VERSION="${VLLM_VERSION:-0.21.0}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.12}"
REGISTRY="${REGISTRY:-ai-infra}"

UPSTREAM_VLLM="vllm/vllm-openai:v${VLLM_VERSION}"
UPSTREAM_VLLM_X86="vllm/vllm-openai:latest-x86_64-ubuntu2404"
UPSTREAM_SGLANG="lmsysorg/sglang:v${SGLANG_VERSION}-cu130"
UPSTREAM_SGLANG_RUNTIME="lmsysorg/sglang:v${SGLANG_VERSION}-cu130-runtime"

SLIM_VLLM_CU128="${REGISTRY}/vllm-slim:${VLLM_VERSION}-cu128"
SLIM_VLLM_CU130="${REGISTRY}/vllm-slim:${VLLM_VERSION}-cu130"
SLIM_SGLANG_CU128="${REGISTRY}/sglang-slim:${SGLANG_VERSION}-cu128"
SLIM_SGLANG_CU130="${REGISTRY}/sglang-slim:${SGLANG_VERSION}-cu130"

# Try to fetch compressed size from registry (requires docker login for some).
compressed_size() {
  local image="$1"
  local size_bytes
  size_bytes=$(docker manifest inspect --verbose "$image" 2>/dev/null \
    | python3 -c '
import json, sys
data = json.load(sys.stdin)
if isinstance(data, list):
    # Multi-arch index; pick amd64.
    for entry in data:
        plat = entry.get("Descriptor", {}).get("platform", {})
        if plat.get("architecture") == "amd64" and plat.get("os") == "linux":
            data = entry.get("SchemaV2Manifest", entry)
            break
manifest = data.get("SchemaV2Manifest", data)
total = sum(layer.get("size", 0) for layer in manifest.get("layers", []))
total += manifest.get("config", {}).get("size", 0)
print(total)
' 2>/dev/null) || size_bytes=""
  if [[ -z "$size_bytes" || "$size_bytes" == "0" ]]; then
    echo "n/a"
  else
    bc <<<"scale=2; $size_bytes / 1024 / 1024 / 1024"
    echo " GB"
  fi
}

uncompressed_size() {
  local image="$1"
  local size_bytes
  size_bytes=$(docker image inspect "$image" --format '{{.Size}}' 2>/dev/null) || size_bytes=""
  if [[ -z "$size_bytes" ]]; then
    echo "n/a (not pulled)"
  else
    awk -v s="$size_bytes" 'BEGIN { printf "%.2f GB\n", s / 1024 / 1024 / 1024 }'
  fi
}

print_row() {
  local label="$1"
  local image="$2"
  local comp uncomp
  comp=$(compressed_size "$image")
  uncomp=$(uncompressed_size "$image")
  printf "%-45s %-20s %-20s %s\n" "$label" "$comp" "$uncomp" "$image"
}

printf "%-45s %-20s %-20s %s\n" "Image" "Compressed" "Uncompressed" "Tag"
printf "%-45s %-20s %-20s %s\n" "$(printf '%.0s-' {1..45})" "$(printf '%.0s-' {1..20})" "$(printf '%.0s-' {1..20})" "----"

echo
echo "Upstream vLLM:"
print_row "vLLM upstream default"          "$UPSTREAM_VLLM"
print_row "vLLM upstream x86 ubuntu24.04"   "$UPSTREAM_VLLM_X86"

echo
echo "Slim vLLM:"
print_row "vLLM slim (cu128)"               "$SLIM_VLLM_CU128"
print_row "vLLM slim (cu130)"               "$SLIM_VLLM_CU130"

echo
echo "Upstream SGLang:"
print_row "SGLang upstream cu130 default"   "$UPSTREAM_SGLANG"
print_row "SGLang upstream cu130-runtime"   "$UPSTREAM_SGLANG_RUNTIME"

echo
echo "Slim SGLang:"
print_row "SGLang slim (cu128)"             "$SLIM_SGLANG_CU128"
print_row "SGLang slim (cu130)"             "$SLIM_SGLANG_CU130"

echo
echo "Note: Compressed sizes require registry access. n/a means image not"
echo "      published or registry inspection failed. Uncompressed = local"
echo "      disk size; n/a means not pulled."
