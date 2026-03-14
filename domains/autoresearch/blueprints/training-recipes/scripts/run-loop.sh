#!/usr/bin/env bash
# Launch the autoresearch loop on a g7e instance.
#
# Usage:
#   ./run-loop.sh [ssh-host]
#
# Prerequisites:
#   - SSH access to g7e instance (default: from MEMORY.md)
#   - Claude Code CLI installed on the instance (or run locally with SSH tools)
#
# The script:
#   1. Clones autoresearch-colab to /mnt/nvme
#   2. Runs prepare.py to download data
#   3. Launches Claude Code with program.md as the prompt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLUEPRINT_DIR="$(dirname "$SCRIPT_DIR")"
SSH_HOST="${1:-ec2-user@35.94.217.100}"
SSH_KEY="${SSH_KEY:-~/.ssh/g7e-bench.pem}"
WORK_DIR="/mnt/nvme/autoresearch"

echo "=== Autoresearch: Training Recipes ==="
echo "Host: $SSH_HOST"
echo "Work dir: $WORK_DIR"
echo ""

# Step 1: Setup on remote
echo "--- Setting up remote environment ---"
ssh -i "$SSH_KEY" "$SSH_HOST" bash -s <<'REMOTE'
set -euo pipefail
cd /mnt/nvme

if [ ! -d autoresearch ]; then
    git clone https://github.com/aigorahub/autoresearch-colab.git autoresearch
    echo "Cloned autoresearch-colab"
else
    echo "autoresearch-colab already present"
fi

cd autoresearch

# Check if data is prepared
if [ ! -d data ]; then
    echo "Running prepare.py (downloads ~10GB FineWeb-Edu)..."
    python prepare.py
    echo "Data preparation complete"
else
    echo "Data already prepared"
fi

echo "Setup complete. Ready for autoresearch loop."
REMOTE

echo ""
echo "--- Remote setup complete ---"
echo ""
echo "To start the autoresearch loop, run Claude Code on the instance:"
echo ""
echo "  ssh -i $SSH_KEY $SSH_HOST"
echo "  cd $WORK_DIR"
echo "  claude --prompt \"$(cat "$BLUEPRINT_DIR/program.md" | head -5)...\""
echo ""
echo "Or copy program.md to the instance and use it as context:"
echo ""
echo "  scp -i $SSH_KEY $BLUEPRINT_DIR/program.md $SSH_HOST:$WORK_DIR/"
echo "  ssh -i $SSH_KEY $SSH_HOST 'cd $WORK_DIR && claude'"
