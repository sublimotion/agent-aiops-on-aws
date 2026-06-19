#!/usr/bin/env bash
# Host-side bootstrap for Stage 0 dynamo-snapshot smoke test.
# Runs on the AL2023 NVIDIA GPU AMI as ec2-user (sudo).
# Idempotent.
set -euo pipefail

echo "=== nvidia-smi ==="
nvidia-smi || { echo "no nvidia-smi"; exit 1; }
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
echo "Driver major: ${DRIVER_MAJOR}"
if [ "${DRIVER_MAJOR}" -lt 580 ]; then
  echo "HARD GATE FAILED: driver major ${DRIVER_MAJOR} < 580 (Dynamo snapshot requires 580.xx+)"
  exit 2
fi

echo "=== Install build deps (CRIU + Docker) ==="
sudo dnf install -y --quiet \
  git make gcc gcc-c++ pkgconf-pkg-config \
  libbsd-devel libcap-devel libnl3-devel libnet-devel \
  protobuf-devel protobuf-c-devel protobuf-c-compiler protobuf-compiler \
  python3 python3-protobuf gnutls-devel libnftnl-devel libuuid-devel \
  iproute iptables procps-ng tar util-linux

echo "=== Build CRIU + cuda-checkpoint from source ==="
WORK=/opt/dynamo-snapshot
sudo mkdir -p "$WORK" && sudo chown ec2-user:ec2-user "$WORK"
cd "$WORK"

if [ ! -d criu ]; then
  git init criu
  cd criu
  git remote add origin https://github.com/checkpoint-restore/criu.git
  git fetch --depth 1 origin criu-dev
  git checkout FETCH_HEAD
  cd ..
fi
cd criu
make -j"$(nproc)"
sudo make install-criu install-lib install-cuda_plugin
criu --version
cd ..

if [ ! -d cuda-checkpoint ]; then
  git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git
fi
sudo install -m 0755 cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/sbin/cuda-checkpoint
/usr/local/sbin/cuda-checkpoint --version || /usr/local/sbin/cuda-checkpoint --help | head -5

echo "=== Sysctls for CRIU ==="
sudo sysctl -w kernel.yama.ptrace_scope=0
# ns_last_pid is per-namespace; ensure /proc/sys/kernel/ns_last_pid exists
ls -l /proc/sys/kernel/ns_last_pid || true

echo "=== Python env for vLLM ==="
if [ ! -d "$WORK/venv" ]; then
  python3 -m venv "$WORK/venv"
fi
source "$WORK/venv/bin/activate"
pip install --quiet --upgrade pip
# vLLM 0.20+ has the sleep()/wake_up() integration the spec requires.
pip install --quiet "vllm>=0.20" "transformers>=4.40" "huggingface_hub" "torch"

echo "=== Bootstrap complete ==="
