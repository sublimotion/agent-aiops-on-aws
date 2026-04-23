#!/bin/bash
# Setup script for SWE-bench evaluation EC2 instance
# Run on a fresh Ubuntu 22.04 m7i.4xlarge with 200GB gp3
#
# Usage: bash setup_ec2.sh

set -euo pipefail

echo "=== SWE-bench Eval Instance Setup ==="

# System packages
echo "[1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq docker.io python3-pip python3-venv git nodejs npm jq

# Docker setup
echo "[2/7] Configuring Docker..."
sudo usermod -aG docker $USER
sudo systemctl enable docker
sudo systemctl start docker

# Node.js 20+ (Claude Code requires it)
echo "[3/7] Installing Node.js 20..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y -qq nodejs

# Claude Code
echo "[4/7] Installing Claude Code..."
sudo npm install -g @anthropic-ai/claude-code

# Python environment
echo "[5/7] Setting up Python environment..."
python3 -m venv /home/$USER/swebench-env
source /home/$USER/swebench-env/bin/activate
pip install -q --upgrade pip
pip install -q swebench datasets boto3

# Verify installations
echo "[6/7] Verifying installations..."
docker --version
claude --version
python3 -c "import swebench; print(f'swebench {swebench.__version__}')"
python3 -c "import datasets; print(f'datasets {datasets.__version__}')"
python3 -c "import boto3; print(f'boto3 {boto3.__version__}')"

# Pre-warm SWE-bench Docker images (optional, takes ~30min)
echo "[7/7] Validating SWE-bench Docker eval..."
python3 -m swebench.harness.run_evaluation \
    --predictions_path gold \
    --max_workers 1 \
    --instance_ids sympy__sympy-20590 \
    --run_id validate-gold \
    --cache_level env

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Set ANTHROPIC_API_KEY: export ANTHROPIC_API_KEY=<your-key>"
echo "  2. Configure AWS credentials for Bedrock (verification tools)"
echo "  3. Copy scripts: scp -r scripts/ ec2-user@<ip>:~/vp-swebench/"
echo "  4. Activate env: source ~/swebench-env/bin/activate"
echo "  5. Run: python3 ~/vp-swebench/swebench_claude_code.py --output results/predictions_lite.jsonl"
echo "  6. Eval: python3 -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Lite --predictions_path results/predictions_lite.jsonl --max_workers 8 --run_id claude-code-verify-lite"
