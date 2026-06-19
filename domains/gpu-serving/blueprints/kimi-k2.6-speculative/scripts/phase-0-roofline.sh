#!/usr/bin/env bash
# phase-0-roofline.sh — Roofline characterization.
# Runs on the GPU node. Expects weights at /mnt/nvme/models/kimi-k26-fp8/.
# Output: results/phase-0-roofline/*.json + nsys traces.

set -euo pipefail

RESULTS=/mnt/nvme/results/phase-0-roofline
mkdir -p "$RESULTS"

# 1. HBM sustained BW (DCGM memcpy kernel)
nvidia-smi dmon -s m -c 30 > "$RESULTS/hbm-bw.log" &
dmon_pid=$!

# 2. NCCL all_reduce — verifies NVSwitch (expect ~1.8 TB/s bisection on NVL5)
docker run --rm --gpus all --network host \
  nvcr.io/nvidia/pytorch:25.03-py3 \
  bash -c "cd /workspace && git clone https://github.com/NVIDIA/nccl-tests || true && cd nccl-tests && make && ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8" \
  > "$RESULTS/nccl-all-reduce.log" 2>&1

kill $dmon_pid 2>/dev/null || true

# 3. DeepGEMM FP8 microbenchmark (if deepgemm available in sglang image)
docker run --rm --gpus all --network host -v /mnt/nvme:/mnt/nvme \
  lmsysorg/sglang:v0.5.10-cu130 \
  python -c "
import torch
# Warm up FP8 GEMM at K2.6 MoE shapes (M=batch, N=hidden=8192, K=intermediate=2048)
# Measures peak DeepGEMM FP8 TFLOPS
try:
    from deep_gemm import gemm_fp8_fp8_bf16_nt as fp8_gemm
    import time
    for M in [128, 512, 2048, 8192]:
        a = torch.randn(M, 8192, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
        b = torch.randn(8192, 2048, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
        c = torch.zeros(M, 2048, dtype=torch.bfloat16, device='cuda')
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(100): fp8_gemm(a, b, c)
        torch.cuda.synchronize(); dt = (time.time()-t0)/100
        flops = 2*M*2048*8192 / dt / 1e12
        print(f'M={M}: {flops:.1f} TFLOPS')
except ImportError:
    print('deep_gemm not available in this image')
" > "$RESULTS/deepgemm-fp8.log" 2>&1

echo "Phase 0 microbenchmarks complete — see $RESULTS"
