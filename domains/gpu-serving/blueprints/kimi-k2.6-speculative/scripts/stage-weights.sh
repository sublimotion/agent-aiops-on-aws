#!/usr/bin/env bash
# stage-weights.sh — Idempotent helper that downloads K2.6 target + EAGLE3 draft and uploads to S3.
#
# Intended to be RUN ON A us-east-1 HELPER EC2 (c6i.8xlarge recommended) with an IAM role that can
# write to the model bucket. This script is NOT run on the GPU node.
#
# Usage:
#   export HF_TOKEN=hf_xxx     # if target or draft is gated
#   bash stage-weights.sh

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
BUCKET="${MODEL_BUCKET:-kimi-k2-bench-models-20260216163240701700000006}"
TARGET_REPO="moonshotai/Kimi-K2.6"
TARGET_PREFIX="models/Kimi-K2.6"
DRAFT_REPO="lightseekorg/kimi-k2.6-eagle3"
DRAFT_PREFIX="models/kimi-k2.6-eagle3"
WORK="/mnt/stage"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }

install_deps() {
  log "Installing hf CLI + awscli"
  sudo dnf install -y python3.11 python3.11-pip || true
  pip3.11 install --user --upgrade 'huggingface_hub[cli]>=1.11' boto3
  export PATH="$HOME/.local/bin:$PATH"
  hf --version || { log "hf CLI missing"; exit 1; }
  aws --version
}

prep_workdir() {
  log "Preparing $WORK"
  sudo mkdir -p "$WORK"
  sudo chown -R "$(id -u):$(id -g)" "$WORK"
  df -h "$WORK"
}

have_in_s3() {
  local prefix="$1"
  local marker="$2"
  aws s3 ls "s3://$BUCKET/$prefix/$marker" --region "$REGION" >/dev/null 2>&1
}

stage_repo() {
  local repo="$1"
  local prefix="$2"
  local dir="$WORK/$(basename "$repo")"

  if have_in_s3 "$prefix" "config.json"; then
    log "$prefix already in S3 (config.json found) — skip"
    return 0
  fi

  log "Downloading $repo → $dir"
  mkdir -p "$dir"
  # Never pass --token on argv (visible in ps). hf CLI picks up HF_TOKEN from env or ~/.cache/huggingface/token.
  if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
  fi
  HF_TOKEN="${HF_TOKEN:-}" hf download "$repo" --local-dir "$dir" --max-workers 16

  log "Uploading $dir → s3://$BUCKET/$prefix/"
  aws s3 sync "$dir" "s3://$BUCKET/$prefix/" \
    --region "$REGION" \
    --only-show-errors \
    --storage-class STANDARD

  log "Cleaning local copy to reclaim disk"
  rm -rf "$dir"
}

main() {
  install_deps
  prep_workdir
  stage_repo "$TARGET_REPO" "$TARGET_PREFIX"
  stage_repo "$DRAFT_REPO" "$DRAFT_PREFIX"
  log "DONE — s3://$BUCKET/$TARGET_PREFIX/ and s3://$BUCKET/$DRAFT_PREFIX/"
  aws s3 ls "s3://$BUCKET/$TARGET_PREFIX/" --region "$REGION" --human-readable --summarize | tail -5
  aws s3 ls "s3://$BUCKET/$DRAFT_PREFIX/" --region "$REGION" --human-readable --summarize | tail -5
}

main "$@"
