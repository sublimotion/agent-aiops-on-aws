#!/bin/bash

set -ex

LOG_FILE="/var/log/provision/provisioning.log"
mkdir -p "/var/log/provision"
touch "$LOG_FILE"

logger() {
  echo "$@" | tee -a "$LOG_FILE"
}

logger "[start] on_create.sh"

if mount | grep -q /opt/sagemaker; then
  logger "Found secondary EBS volume. Setting containerd data root to /opt/sagemaker/containerd/data-root"
  mkdir -p /opt/sagemaker/containerd/data-root
  if [[ -f /etc/eks/containerd/containerd-config.toml ]]; then
    sed -i -e '/^[# ]*root\s*=/c\root = "/opt/sagemaker/containerd/data-root"' /etc/eks/containerd/containerd-config.toml
  else
    logger "Skipping containerd data root update; /etc/eks/containerd/containerd-config.toml not found"
  fi
fi

logger "no more steps to run"
logger "[stop] on_create.sh"
