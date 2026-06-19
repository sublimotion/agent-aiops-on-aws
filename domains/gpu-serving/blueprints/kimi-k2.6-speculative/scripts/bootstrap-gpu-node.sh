#!/usr/bin/env bash
# bootstrap-gpu-node.sh — Runs as EC2 userdata on the p6-b300 spot node.
# Sets up: NVMe RAID0 at /mnt/nvme, docker, HF-cache, direct HF→NVMe weight pull, S3 mirror, ECR login.
# HF token pulled from SSM SecureString /kimi-k26/hf-token via instance role (never on argv — L4).

set -eux

exec > >(tee -a /var/log/kimi-bootstrap.log) 2>&1

REGION="us-east-1"
ACCOUNT="615299764834"
BUCKET="kimi-k2-bench-models-20260216163240701700000006"
RESULTS_BUCKET="kimi-k2-bench-results-20260216163240701700000007"

# ---------- Install missing tools (L2: AL2023 DLAMI is minimal) ----------
dnf install -y mdadm xfsprogs jq awscli python3.11 python3.11-pip

# ---------- NVMe RAID0 ----------
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

mkdir -p /mnt/nvme/{models,hf-cache,results,logs}
chown -R ec2-user:ec2-user /mnt/nvme

# ---------- Container runtime + ECR login ----------
systemctl enable --now docker || true
usermod -aG docker ec2-user || true
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

# ---------- Install hf CLI for ec2-user ----------
sudo -u ec2-user pip3.11 install --user --upgrade 'huggingface_hub[cli]>=1.11' boto3
# HF CLI will live at /home/ec2-user/.local/bin/hf

# ---------- Pull HF token from SSM and write to standard HF cache (never argv) ----------
HF_TOKEN_VAL=$(aws ssm get-parameter --name /kimi-k26/hf-token --with-decryption --region $REGION --query 'Parameter.Value' --output text)
mkdir -p /home/ec2-user/.cache/huggingface
printf '%s' "$HF_TOKEN_VAL" > /home/ec2-user/.cache/huggingface/token
chown -R ec2-user:ec2-user /home/ec2-user/.cache
chmod 600 /home/ec2-user/.cache/huggingface/token
unset HF_TOKEN_VAL

# ---------- Stage weights in background: parallel HF → NVMe, then S3 mirror ----------
# Direct HF pull uses the full 3200 Gbps node network. Draft (6 GB) + Target (~600 GB) in parallel.
# HF_HUB_ENABLE_HF_TRANSFER=1 routes through hf_transfer Rust client for max throughput.

cat > /home/ec2-user/stage-weights.sh <<'STAGE'
#!/usr/bin/env bash
set -eu
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/mnt/nvme/hf-cache

pip3.11 install --user --upgrade hf_transfer >/dev/null 2>&1 || true

REGION="us-east-1"
BUCKET="kimi-k2-bench-models-20260216163240701700000006"

# Draft first — tiny, serves as HF auth/network smoke test
(
  set -eu
  hf download lightseekorg/kimi-k2.6-eagle3 --local-dir /mnt/nvme/models/kimi-k26-eagle3 --max-workers 8
  aws s3 sync /mnt/nvme/models/kimi-k26-eagle3/ s3://${BUCKET}/models/kimi-k2.6-eagle3/ --region $REGION --only-show-errors
  touch /mnt/nvme/models/kimi-k26-eagle3/.s3-synced
) > /var/log/hf-draft.log 2>&1 &
DRAFT_PID=$!

# Target in parallel
(
  set -eu
  hf download moonshotai/Kimi-K2.6 --local-dir /mnt/nvme/models/kimi-k26-fp8 --max-workers 16
  aws s3 sync /mnt/nvme/models/kimi-k26-fp8/ s3://${BUCKET}/models/Kimi-K2.6/ --region $REGION --only-show-errors
  touch /mnt/nvme/models/kimi-k26-fp8/.s3-synced
) > /var/log/hf-target.log 2>&1 &
TARGET_PID=$!

wait $DRAFT_PID $TARGET_PID
touch /mnt/nvme/models/.ready
echo "[$(date -u +%FT%TZ)] All weights staged (HF + S3 mirror)" >> /var/log/stage-weights.log
STAGE

chmod +x /home/ec2-user/stage-weights.sh
chown ec2-user:ec2-user /home/ec2-user/stage-weights.sh
nohup sudo -u ec2-user bash /home/ec2-user/stage-weights.sh > /var/log/stage-weights.log 2>&1 &

# ---------- GPU sanity ----------
nvidia-smi > /var/log/nvidia-smi.log 2>&1 || true
nvidia-smi topo -m > /var/log/nvidia-topo.log 2>&1 || true

# ---------- Pre-pull images in parallel ----------
nohup bash -c "
  docker pull lmsysorg/sglang:v0.5.10-cu130 || docker pull lmsysorg/sglang:latest || true
" > /var/log/image-sglang.log 2>&1 &

nohup bash -c "
  docker pull vllm/vllm-openai:latest || true
" > /var/log/image-vllm.log 2>&1 &

nohup bash -c "
  docker pull nvcr.io/nvidia/pytorch:25.03-py3 || true
" > /var/log/image-pytorch.log 2>&1 &

# ---------- Pull blueprint scripts from S3 + watchers ----------
mkdir -p /opt/kimi/scripts
aws s3 sync s3://${RESULTS_BUCKET}/kimi-k2.6-speculative/scripts/ /opt/kimi/scripts/ --region $REGION --only-show-errors 2>/dev/null || true
chmod +x /opt/kimi/scripts/*.sh 2>/dev/null || true

if [[ -x /opt/kimi/scripts/sync-loop-gpu.sh ]]; then
  nohup /opt/kimi/scripts/sync-loop-gpu.sh > /var/log/sync-loop.log 2>&1 &
fi
if [[ -x /opt/kimi/scripts/spot-reclaim-watcher.sh ]]; then
  nohup /opt/kimi/scripts/spot-reclaim-watcher.sh > /var/log/spot-reclaim.log 2>&1 &
fi

touch /var/log/kimi-bootstrap.done
