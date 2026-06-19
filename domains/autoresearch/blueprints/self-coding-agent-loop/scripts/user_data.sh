#!/bin/bash
# p4de.24xlarge user-data bootstrap for self-coding-agent-loop experiment.
# Runs once at first boot. Idempotent: safe to re-run after reboot.
#
# Expected envs (set via tag or echoed at top by launch script):
#   EXPERIMENT_BUCKET=s3://agent-aiops-artifacts/self-coding-agent-loop
#   LAPTOP_SSH_TARGET=phi@<reverse-tunnel-or-tailscale-host>
#   LAPTOP_RUNS_DIR=/Users/phi/Documents/workbench/agent-aiops-on-aws/domains/autoresearch/blueprints/self-coding-agent-loop/runs

set -euxo pipefail
exec > >(tee /var/log/user-data.log) 2>&1

# ---- variables (populated by launch script via envsubst) ----
EXPERIMENT_BUCKET="${EXPERIMENT_BUCKET:-s3://agent-aiops-artifacts/self-coding-agent-loop}"
WORK_DIR=/mnt/nvme/self-coding-agent-loop
GEN0_S3="${EXPERIMENT_BUCKET}/gen0"

# ---- NVMe (Deep Learning AMI pre-configures all 8 NVMe drives as /opt/dlami/nvme LVM) ----
# We symlink /mnt/nvme → /opt/dlami/nvme to keep the rest of the script path-independent.
mkdir -p /opt/dlami/nvme
if [[ ! -e /mnt/nvme ]]; then
    ln -s /opt/dlami/nvme /mnt/nvme
fi
mkdir -p "$WORK_DIR"
chown -R ubuntu:ubuntu /opt/dlami/nvme

# ---- base packages ----
# AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
# CUDA + driver already installed. Default user is `ubuntu`, not `ec2-user`.
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git rsync tmux htop python3.11 python3.11-venv python3-pip tar jq awscli xfsprogs
# Default AMI user is `ubuntu` (Deep Learning Base AMI). We reference it directly below.

# ---- project checkout ----
cd "$WORK_DIR"
if [[ ! -d agent-aiops-on-aws ]]; then
    aws s3 cp "${EXPERIMENT_BUCKET}/repo-snapshot.tar.gz" /tmp/repo.tar.gz
    tar -xzf /tmp/repo.tar.gz -C "$WORK_DIR"
fi

# ---- python env ----
if [[ ! -d "$WORK_DIR/venv" ]]; then
    python3.11 -m venv "$WORK_DIR/venv"
    "$WORK_DIR/venv/bin/pip" install --upgrade pip wheel
    "$WORK_DIR/venv/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu121
    "$WORK_DIR/venv/bin/pip" install transformers peft accelerate datasets trl \
        bitsandbytes vllm==0.18.0 huggingface_hub pandas pyarrow scikit-learn
    # flash-attn requires torch installed first
    "$WORK_DIR/venv/bin/pip" install flash-attn --no-build-isolation
fi

# ---- Gen0 LoRA adapter ----
mkdir -p "$WORK_DIR/checkpoints/gen0"
aws s3 sync "$GEN0_S3" "$WORK_DIR/checkpoints/gen0/"

# ---- term handler (spot reclaim) ----
cp "$WORK_DIR/agent-aiops-on-aws/domains/autoresearch/blueprints/self-coding-agent-loop/scripts/term_handler.sh" /usr/local/bin/term_handler.sh
chmod +x /usr/local/bin/term_handler.sh

# Poll IMDS for spot interruption notice in background
cat > /etc/systemd/system/spot-watcher.service <<'EOF'
[Unit]
Description=Spot Termination Watcher
After=network.target

[Service]
ExecStart=/usr/local/bin/term_handler.sh watch
Restart=always
User=ubuntu
Environment=EXPERIMENT_BUCKET=__BUCKET__
Environment=WORK_DIR=/mnt/nvme/self-coding-agent-loop

[Install]
WantedBy=multi-user.target
EOF
sed -i "s|__BUCKET__|$EXPERIMENT_BUCKET|g" /etc/systemd/system/spot-watcher.service
systemctl daemon-reload
systemctl enable --now spot-watcher

# ---- S3 sync daemon (opportunistic, every 10 min, for all data not just reclaim) ----
cat > /etc/systemd/system/s3-sync.service <<EOF
[Unit]
Description=S3 periodic sync for self-coding-agent-loop
After=network.target

[Service]
ExecStart=/bin/bash -c "while true; do sleep 600; aws s3 sync $WORK_DIR/runs/ $EXPERIMENT_BUCKET/runs/ --exclude '*.tmp' 2>&1 | logger -t s3-sync; done"
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now s3-sync

touch "$WORK_DIR/.bootstrap-complete"
echo "bootstrap complete at $(date -u)"
