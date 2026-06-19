#!/usr/bin/env bash
# cloud-init bootstrap for the slim image build host.
#
# Reads from environment (set by terraform's user_data prelude):
#   TF_REGION, TF_REPO_URL, TF_REPO_REF, TF_AWS_ACCOUNT_ID
#
# Idempotent. Safe to re-run after instance reboot.

set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

: "${TF_REGION:?TF_REGION must be set}"
: "${TF_REPO_URL:?TF_REPO_URL must be set}"
: "${TF_REPO_REF:?TF_REPO_REF must be set}"
: "${TF_AWS_ACCOUNT_ID:?TF_AWS_ACCOUNT_ID must be set}"

# ---- packages ---------------------------------------------------------------

apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release \
    git jq unzip python3 python3-pip \
    build-essential

# Docker CE + buildx.
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y --no-install-recommends \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin

usermod -aG docker ubuntu
systemctl enable --now docker

# AWS CLI v2.
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip
unzip -q /tmp/awscli.zip -d /tmp/
/tmp/aws/install --update
rm -rf /tmp/awscli.zip /tmp/aws

# ---- repo + workspace -------------------------------------------------------

mkdir -p /opt/build
chown ubuntu:ubuntu /opt/build

sudo -u ubuntu \
    REPO_URL="$TF_REPO_URL" \
    REPO_REF="$TF_REPO_REF" \
    bash <<'EOF'
set -euxo pipefail
cd /opt/build
if [ ! -d agent-aiops-on-aws ]; then
    git clone "$REPO_URL" agent-aiops-on-aws
fi
cd agent-aiops-on-aws
git fetch --all --tags
git checkout "$REPO_REF"
EOF

# ---- ECR login helper -------------------------------------------------------

cat >/usr/local/bin/ecr-login <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin \
        "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
EOF
chmod +x /usr/local/bin/ecr-login

# ---- environment ------------------------------------------------------------

cat >/etc/profile.d/ai-infra.sh <<EOF
export AWS_REGION="$TF_REGION"
export AWS_DEFAULT_REGION="$TF_REGION"
export REGISTRY="$TF_AWS_ACCOUNT_ID.dkr.ecr.$TF_REGION.amazonaws.com/ai-infra"
export REPO_DIR="/opt/build/agent-aiops-on-aws"
EOF

# Buildx: dedicated builder with larger gc, on-disk layer cache.
sudo -u ubuntu bash <<'EOF'
set -euxo pipefail
mkdir -p /home/ubuntu/.docker
docker buildx create --name slim-builder --driver docker-container --use --bootstrap || true
EOF

echo "bootstrap complete"
