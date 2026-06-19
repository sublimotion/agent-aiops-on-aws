#!/usr/bin/env bash
# Phase 1 build bootstrap for the dynamo-snapshot bridging-cell retry.
# Builds CRIU from criu-dev with NVIDIA upstream PR #3021 (parallel-memfd) merged.
# Skips PR #3022 (AIO) per prior decision (conflicts with #3021 on criu/mem.c).
# Idempotent. Runs as ec2-user on AL2023 NVIDIA GPU AMI.
set -euo pipefail

echo "=== nvidia-smi ==="
nvidia-smi
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRIVER_MAJOR}" -lt 580 ]; then
  echo "HARD GATE FAILED: driver major ${DRIVER_MAJOR} < 580"
  exit 2
fi

echo "=== NVMe mount ==="
if ! mountpoint -q /mnt/nvme; then
  NVME_DEV=$(lsblk -dn -o NAME,SIZE,TYPE | awk '$3=="disk" && $1 ~ /nvme/ && $2 != "200G" && $2 != "16G" {print "/dev/"$1; exit}')
  if [ -z "${NVME_DEV:-}" ]; then
    NVME_DEV=$(lsblk -dn -o NAME,MOUNTPOINT,TYPE | awk '$3=="disk" && $1 ~ /nvme/ && $2=="" {print "/dev/"$1; exit}')
  fi
  echo "Formatting and mounting ${NVME_DEV} at /mnt/nvme"
  sudo mkfs.xfs -f "${NVME_DEV}"
  sudo mkdir -p /mnt/nvme
  sudo mount "${NVME_DEV}" /mnt/nvme
  sudo chown -R ec2-user:ec2-user /mnt/nvme
fi
df -h /mnt/nvme

echo "=== Install build deps ==="
sudo dnf install -y --quiet \
  git make gcc gcc-c++ pkgconf-pkg-config \
  libbsd-devel libcap-devel libnl3-devel libnet-devel \
  protobuf-devel protobuf-c-devel protobuf-c-compiler protobuf-compiler \
  python3 python3-protobuf gnutls-devel libnftnl-devel libuuid-devel \
  iproute iptables procps-ng tar util-linux libseccomp-devel jq

WORK=/opt/dynamo-snapshot
sudo mkdir -p "$WORK" && sudo chown ec2-user:ec2-user "$WORK"
cd "$WORK"

echo "=== Clone criu-dev and merge PR #3021 (parallel-memfd) ==="
if [ ! -d criu/.git ]; then
  rm -rf criu
  git init criu
  cd criu
  git remote add origin https://github.com/checkpoint-restore/criu.git
  git config user.email "deployer@example.com"
  git config user.name "deployer"
  # Fetch criu-dev and the PR #3021 branch
  git fetch --depth 50 origin criu-dev
  git checkout -b criu-dev FETCH_HEAD
  # PR #3021 is from dfeigin-nv:upstream/pr-1-parallel-memfd
  git remote add nv https://github.com/dfeigin-nv/criu.git
  git fetch --depth 50 nv upstream/pr-1-parallel-memfd
  # Merge it
  if ! git merge --no-edit FETCH_HEAD; then
    echo "MERGE FAILED — conflicts:"
    git status
    exit 3
  fi
  echo "PR #3021 merged. Head:"
  git log --oneline -5
  cd ..
fi

cd criu
echo "=== make ==="
make -j"$(nproc)" 2>&1 | tail -20
# Skip install-lib (python3.9 wheel issue on AL2023). Lessons: install-criu + install-cuda_plugin only.
sudo make install-criu install-cuda_plugin
criu --version
cd ..

echo "=== cuda-checkpoint ==="
if [ ! -d cuda-checkpoint ]; then
  git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git
fi
sudo install -m 0755 cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/sbin/cuda-checkpoint
/usr/local/sbin/cuda-checkpoint --version 2>&1 | head -3 || /usr/local/sbin/cuda-checkpoint --help 2>&1 | head -5

echo "=== Sysctls ==="
sudo sysctl -w kernel.yama.ptrace_scope=0

echo "=== Seccomp profile (block io_uring syscalls — mirror Dynamo agent) ==="
sudo mkdir -p /etc/seccomp
sudo tee /etc/seccomp/no-io-uring.json >/dev/null <<'EOF'
{
  "defaultAction": "SCMP_ACT_ALLOW",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": ["io_uring_setup", "io_uring_enter", "io_uring_register"],
      "action": "SCMP_ACT_ERRNO",
      "errnoRet": 38,
      "comment": "ENOSYS — force libuv/uvloop fallback to epoll for CRIU compatibility"
    }
  ]
}
EOF
echo "Seccomp profile written to /etc/seccomp/no-io-uring.json"

# Build a tiny C/seccomp helper that applies the profile to argv via prctl + libseccomp
echo "=== Build seccomp wrapper ==="
cat > "$WORK/seccomp-wrap.c" <<'EOF'
// Apply a no-io_uring seccomp filter, then exec argv[1..]
#include <seccomp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <errno.h>
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s prog [args...]\n", argv[0]); return 2; }
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0) { perror("prctl"); return 3; }
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_ALLOW);
    if (!ctx) { fprintf(stderr, "seccomp_init failed\n"); return 4; }
    // ENOSYS for io_uring syscalls
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(ENOSYS), SCMP_SYS(io_uring_setup), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(ENOSYS), SCMP_SYS(io_uring_enter), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ERRNO(ENOSYS), SCMP_SYS(io_uring_register), 0);
    if (seccomp_load(ctx) < 0) { fprintf(stderr, "seccomp_load failed\n"); return 5; }
    seccomp_release(ctx);
    execvp(argv[1], &argv[1]);
    perror("execvp");
    return 127;
}
EOF
gcc -O2 -o "$WORK/seccomp-wrap" "$WORK/seccomp-wrap.c" -lseccomp
sudo install -m 0755 "$WORK/seccomp-wrap" /usr/local/bin/seccomp-wrap
echo "seccomp-wrap installed: $(/usr/local/bin/seccomp-wrap /bin/true && echo OK)"

echo "=== Python venv on NVMe ==="
if [ ! -d /mnt/nvme/venv ]; then
  python3 -m venv /mnt/nvme/venv
fi
source /mnt/nvme/venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet "vllm>=0.20" "transformers>=4.40" "huggingface_hub" "torch"

echo "=== Pre-stage Qwen3-0.6B weights to NVMe ==="
mkdir -p /mnt/nvme/hf
HF_HOME=/mnt/nvme/hf python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='Qwen/Qwen3-0.6B', cache_dir='/mnt/nvme/hf')
print('Weights at:', p)
"

echo "=== Verify ==="
echo "criu: $(criu --version 2>&1 | head -1)"
echo "criu git: $(cd /opt/dynamo-snapshot/criu && git log --oneline -3)"
echo "cuda-checkpoint: $(/usr/local/sbin/cuda-checkpoint --version 2>&1 | head -1 || true)"
echo "seccomp-wrap: $(ls -l /usr/local/bin/seccomp-wrap)"
echo
echo "=== Bootstrap complete ==="
