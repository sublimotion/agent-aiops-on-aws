#!/bin/bash
# Phase 1 environment setup for kernel optimization agent on p5en.48xlarge
# Run: bash setup_environment.sh
set -euo pipefail

WORKSPACE="/opt/dlami/nvme/kernel-opt"
VENV="$WORKSPACE/venv"
RESULTS="$WORKSPACE/results"

echo "=== Kernel Optimization Agent: Environment Setup ==="
echo "Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo 'unknown')"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo ""

# 1. Activate venv
source "$VENV/bin/activate"
echo "[1/6] Python venv activated: $(python3 --version)"

# 2. Install core packages
echo "[2/6] Installing packages..."
pip install --quiet --upgrade pip
pip install --quiet \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    triton \
    numpy scipy pandas \
    vllm \
    huggingface_hub \
    flashinfer-python \
    pynvml \
    matplotlib

# Verify installs
python3 -c "
import torch, triton, vllm
print(f'  PyTorch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.device_count()} GPUs')
print(f'  Triton {triton.__version__}')
print(f'  vLLM {vllm.__version__}')
"

# 3. Build DeepGEMM
echo "[3/6] Building DeepGEMM..."
cd "$WORKSPACE/repos/DeepGEMM"
pip install --quiet -e . 2>/dev/null && echo "  DeepGEMM installed" || echo "  DeepGEMM: build failed (will debug later)"

# 4. Build FlashMoE
echo "[4/6] Building FlashMoE..."
cd "$WORKSPACE/repos/FlashMoE"
pip install --quiet -e . 2>/dev/null && echo "  FlashMoE installed" || echo "  FlashMoE: build failed (will debug later)"

# 5. Seed constraint database
echo "[5/6] Seeding constraint database..."
cd "$WORKSPACE"
python3 "$WORKSPACE/scripts/seed_constraints.py" --output "$RESULTS/constraints.jsonl"

# 6. Verify model download status
echo "[6/6] Model download status..."
MODEL_DIR="$WORKSPACE/models/kimi-k26-fp8"
if [ -f "$MODEL_DIR/config.json" ]; then
    SHARD_COUNT=$(ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l)
    TOTAL_SIZE=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
    echo "  K2.6 FP8: $SHARD_COUNT shards, $TOTAL_SIZE total"
    if [ "$SHARD_COUNT" -ge 90 ]; then
        echo "  ✓ Model download appears complete"
    else
        echo "  ⏳ Download in progress... (check: tail -f $WORKSPACE/download.log)"
    fi
else
    echo "  ✗ Model not found. Start download with:"
    echo "    hf download RedHatAI/Kimi-K2.6-FP8-BLOCK --local-dir $MODEL_DIR"
fi

echo ""
echo "=== Environment Ready ==="
echo "Workspace: $WORKSPACE"
echo "Next steps:"
echo "  1. Wait for model download: tail -f $WORKSPACE/download.log"
echo "  2. Run profiling: python3 $WORKSPACE/scripts/profile_baseline.py --mode throughput"
echo "  3. Run megakernel eval: python3 $WORKSPACE/scripts/benchmark_megakernels.py"
