#!/usr/bin/env bash
# Setup training environment for SERA QLoRA fine-tuning
# Pulls PyTorch container and installs training dependencies
# Run once before sera-train.sh

set -euo pipefail

CONTAINER_NAME="sera-training"
IMAGE="nvcr.io/nvidia/pytorch:24.12-py3"

echo "=== Pulling PyTorch container image ==="
sudo nerdctl pull "$IMAGE"

echo ""
echo "=== Creating training container ==="
# Stop existing training container if any
sudo nerdctl rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Launch interactive training container with all GPUs and NVMe mounted
sudo nerdctl run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /mnt/nvme:/mnt/nvme \
    "$IMAGE" \
    sleep infinity

echo ""
echo "=== Installing training dependencies ==="
sudo nerdctl exec "$CONTAINER_NAME" pip install --no-cache-dir \
    "trl>=0.16.0" \
    "peft>=0.15.0" \
    "bitsandbytes>=0.45.0" \
    "deepspeed>=0.16.0" \
    "accelerate>=1.5.0" \
    "datasets>=3.3.0" \
    "flash-attn>=2.7.0" --no-build-isolation

echo ""
echo "=== Verifying installation ==="
sudo nerdctl exec "$CONTAINER_NAME" python3 -c "
import torch
import trl
import peft
import bitsandbytes
import deepspeed
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
print('trl:', trl.__version__)
print('peft:', peft.__version__)
print('bitsandbytes:', bitsandbytes.__version__)
print('deepspeed:', deepspeed.__version__)
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)')
"

echo ""
echo "=== Training container ready ==="
echo "To run training:"
echo "  sudo nerdctl exec $CONTAINER_NAME bash /mnt/nvme/sera-scripts/sera-train.sh"
echo ""
echo "To enter the container:"
echo "  sudo nerdctl exec -it $CONTAINER_NAME bash"
