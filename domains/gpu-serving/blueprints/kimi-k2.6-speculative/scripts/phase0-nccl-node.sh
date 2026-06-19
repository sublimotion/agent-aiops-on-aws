#!/usr/bin/env bash
# phase0-nccl-node.sh — Runs on GPU node. Measures NCCL collectives and GPU topology.
set -eu

RESULTS=/mnt/nvme/results/phase-0-roofline
mkdir -p "$RESULTS"

nvidia-smi topo -m > "$RESULTS/topology.log" 2>&1 || true
nvidia-smi > "$RESULTS/nvidia-smi.log" 2>&1 || true

# NCCL all-reduce in pytorch container
sudo docker run --rm --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt/nvme:/mnt/nvme \
  nvcr.io/nvidia/pytorch:25.03-py3 \
  bash -c 'set -e; cd /tmp; \
    if [ ! -d nccl-tests ]; then git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git; fi; \
    cd nccl-tests; \
    make -j MPI=0 >/dev/null 2>&1; \
    echo "=== all_reduce_perf 8 GPU ==="; \
    ./build/all_reduce_perf -b 1M -e 8G -f 2 -g 8 2>&1 | tail -30; \
    echo ""; \
    echo "=== all_gather_perf 8 GPU ==="; \
    ./build/all_gather_perf -b 1M -e 2G -f 2 -g 8 2>&1 | tail -20' > "$RESULTS/nccl.log" 2>&1

echo "[phase0] NCCL + topology captured in $RESULTS"
