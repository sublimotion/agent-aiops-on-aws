#!/usr/bin/env bash
# bootstrap-gpu-node.sh — Runs as EC2 userdata on p6-b300 spot joining qn-sglang-eks-cluster.
# Applies Kimi L14/L15/L16 lessons from day one.

set -eu  # dropped -x to avoid leaking secrets to cloud-init log
exec > >(tee -a /var/log/qwen3-bootstrap.log) 2>&1

REGION="us-west-2"
CLUSTER="qn-sglang-eks-cluster"

# ---- Install missing tools (Kimi L2) ----
dnf install -y mdadm xfsprogs jq awscli python3.11 python3.11-pip

# ---- NVMe RAID0 ----
nvmes=($(lsblk -dn -o NAME,SIZE,MODEL | awk '/Instance Storage/ {print "/dev/"$1}'))
if [[ ${#nvmes[@]} -ge 2 ]]; then
  mkdir -p /mnt/nvme
  if ! mountpoint -q /mnt/nvme; then
    mdadm --create --verbose /dev/md0 --level=0 --name=nvme-raid --raid-devices=${#nvmes[@]} "${nvmes[@]}"
    mkfs.xfs -f /dev/md0
    mount /dev/md0 /mnt/nvme
    echo "/dev/md0 /mnt/nvme xfs defaults,nofail 0 0" >> /etc/fstab
  fi
elif [[ ${#nvmes[@]} -eq 1 ]]; then
  mkdir -p /mnt/nvme
  mkfs.xfs -f "${nvmes[0]}"
  mount "${nvmes[0]}" /mnt/nvme
fi
if [[ ! -d /mnt/nvme || $(df /mnt/nvme --output=size 2>/dev/null | tail -1) -lt 1048576 ]]; then
  if [[ -d /opt/dlami/nvme ]]; then
    rm -rf /mnt/nvme && ln -s /opt/dlami/nvme /mnt/nvme
  fi
fi
mkdir -p /mnt/nvme/{models,hf-cache,results,logs,prom-data,prom-snapshots}
chown -R ec2-user:ec2-user /mnt/nvme

# ---- Pull HF token from SSM (L4 — never on argv) ----
HF_TOKEN_VAL=$(aws ssm get-parameter --name /qwen3/hf-token --with-decryption --region $REGION --query 'Parameter.Value' --output text 2>/dev/null || true)
if [[ -n "$HF_TOKEN_VAL" ]]; then
  mkdir -p /home/ec2-user/.cache/huggingface
  printf '%s' "$HF_TOKEN_VAL" > /home/ec2-user/.cache/huggingface/token
  chown -R ec2-user:ec2-user /home/ec2-user/.cache
  chmod 600 /home/ec2-user/.cache/huggingface/token
fi
unset HF_TOKEN_VAL

# ---- Install hf CLI for ec2-user ----
sudo -u ec2-user pip3.11 install --user --quiet 'huggingface_hub[cli,hf_transfer]' hf_transfer aiohttp pyyaml || true

# ---- Kick off weight staging in background (per L15 — explicit HF_TOKEN env + hf_transfer) ----
cat > /home/ec2-user/stage-weights.sh <<'STAGE'
#!/bin/bash
set -eu
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HOME=/mnt/nvme/hf-cache
export HF_TOKEN="$(cat /home/ec2-user/.cache/huggingface/token 2>/dev/null || true)"

mkdir -p /mnt/nvme/models

# Draft first — small, smoke-tests HF auth
(
  set -eu
  export PATH="$HOME/.local/bin:$PATH"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export HF_HOME=/mnt/nvme/hf-cache
  export HF_TOKEN="$(cat /home/ec2-user/.cache/huggingface/token 2>/dev/null || true)"
  hf download lmsys/Qwen3-235B-A22B-EAGLE3 --local-dir /mnt/nvme/models/qwen3-235b-eagle3 --max-workers 8
  touch /mnt/nvme/models/qwen3-235b-eagle3/.ready
  echo "[$(date -u +%FT%TZ)] DRAFT READY" >> /mnt/nvme/logs/stage.log
) > /mnt/nvme/logs/hf-draft.log 2>&1 &
DRAFT_PID=$!

(
  set -eu
  export PATH="$HOME/.local/bin:$PATH"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export HF_HOME=/mnt/nvme/hf-cache
  export HF_TOKEN="$(cat /home/ec2-user/.cache/huggingface/token 2>/dev/null || true)"
  hf download Qwen/Qwen3-235B-A22B-FP8 --local-dir /mnt/nvme/models/qwen3-235b-fp8 --max-workers 16 --exclude '*.md' '*.txt'
  touch /mnt/nvme/models/qwen3-235b-fp8/.ready
  echo "[$(date -u +%FT%TZ)] TARGET READY" >> /mnt/nvme/logs/stage.log
) > /mnt/nvme/logs/hf-target.log 2>&1 &
TARGET_PID=$!

echo "[$(date -u +%FT%TZ)] staging started draft=$DRAFT_PID target=$TARGET_PID" >> /mnt/nvme/logs/stage.log
wait $DRAFT_PID $TARGET_PID
touch /mnt/nvme/models/.all-ready
echo "[$(date -u +%FT%TZ)] ALL READY" >> /mnt/nvme/logs/stage.log
STAGE
chmod +x /home/ec2-user/stage-weights.sh
nohup sudo -u ec2-user bash /home/ec2-user/stage-weights.sh > /mnt/nvme/logs/stage-weights.out 2>&1 &

# ---- GPU sanity ----
nvidia-smi > /var/log/nvidia-smi.log 2>&1 || true
nvidia-smi topo -m > /var/log/nvidia-topo.log 2>&1 || true

touch /var/log/qwen3-bootstrap.done
