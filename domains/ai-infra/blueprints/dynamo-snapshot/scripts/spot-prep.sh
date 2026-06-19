#!/usr/bin/env bash
# Re-prep a spot from the baked AMI: mount NVMe, restore venv + weights, plugin symlinks.
set -e

# Mount NVMe
if ! mountpoint -q /mnt/nvme; then
  sudo mkfs.xfs -f /dev/nvme1n1
  sudo mkdir -p /mnt/nvme
  sudo mount /dev/nvme1n1 /mnt/nvme
  sudo chown -R ec2-user:ec2-user /mnt/nvme
fi

# Plugin symlink (criu's default plugin dir is /usr/lib/criu/, not /usr/local/lib/criu/)
sudo mkdir -p /usr/lib/criu
sudo ln -sf /usr/local/lib/criu/cuda_plugin.so /usr/lib/criu/cuda_plugin.so

# cuda-checkpoint in default PATH
sudo ln -sf /usr/local/sbin/cuda-checkpoint /usr/local/bin/cuda-checkpoint
sudo ln -sf /usr/local/sbin/cuda-checkpoint /usr/sbin/cuda-checkpoint

# Re-stage venv + weights (NVMe is wiped per-spot)
if [ ! -x /mnt/nvme/venv/bin/python ]; then
  python3.12 -m venv /mnt/nvme/venv
  source /mnt/nvme/venv/bin/activate
  pip install --upgrade pip --quiet
  pip install --quiet "vllm==0.10.2" "transformers==4.55.4" "huggingface_hub" 2>&1 | tail -3
fi

mkdir -p /mnt/nvme/hf
if [ ! -d /mnt/nvme/hf/models--Qwen--Qwen3-0.6B/snapshots ] || [ -z "$(ls /mnt/nvme/hf/models--Qwen--Qwen3-0.6B/snapshots/ 2>/dev/null)" ]; then
  HF_HOME=/mnt/nvme/hf /mnt/nvme/venv/bin/python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='Qwen/Qwen3-0.6B', cache_dir='/mnt/nvme/hf')
print('weights:', p)
"
fi

echo "criu: $(criu --version 2>&1 | head -1)"
echo "vllm: $(/mnt/nvme/venv/bin/python -c 'import vllm; print(vllm.__version__)')"
echo "plugin: $(ls -la /usr/lib/criu/cuda_plugin.so)"
echo "ready"
