#!/usr/bin/env bash
# Pull latest repo state, build slim images, push to ECR.
#
# Run on the build host as ubuntu user. Reads REGISTRY + AWS_REGION from
# /etc/profile.d/ai-infra.sh (sourced via login shell).
#
# Usage:
#   ./pull-and-build.sh                    # build all engines + variants, push all
#   ./pull-and-build.sh vllm cu128         # one engine + one variant
#   PUSH=0 ./pull-and-build.sh             # build only, no push

set -euo pipefail

# shellcheck disable=SC1091
source /etc/profile.d/ai-infra.sh

REPO_DIR="${REPO_DIR:-/opt/build/agent-aiops-on-aws}"
PUSH="${PUSH:-1}"

cd "$REPO_DIR"
git fetch --all
git pull --ff-only

cd domains/ai-infra/shared/images

# Authenticate Docker against ECR. Token expires in 12h; safe to re-run.
ecr-login

echo "==> Building (registry=$REGISTRY)"
./build.sh "${1:-all}" "${2:-all}"

if [[ "$PUSH" != "1" ]]; then
    echo "PUSH=0 set; skipping push"
    exit 0
fi

# Push every tag we just built.
mapfile -t TAGS < <(docker images --format '{{.Repository}}:{{.Tag}}' \
    | grep '^ai-infra/' || true)

if [[ ${#TAGS[@]} -eq 0 ]]; then
    echo "no ai-infra/* images found locally; nothing to push"
    exit 0
fi

for local_tag in "${TAGS[@]}"; do
    remote_tag="${REGISTRY}/${local_tag#ai-infra/}"
    echo "==> Tagging and pushing ${local_tag} -> ${remote_tag}"
    docker tag "$local_tag" "$remote_tag"
    docker push "$remote_tag"
done

echo
echo "==> Sizes (compressed via ECR):"
for repo in vllm-slim sglang-slim; do
    aws ecr describe-images \
        --repository-name "ai-infra/${repo}" \
        --region "$AWS_REGION" \
        --query 'imageDetails[].[imageTags[0],imageSizeInBytes]' \
        --output table 2>/dev/null || true
done
