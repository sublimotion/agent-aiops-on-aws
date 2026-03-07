#!/bin/bash
# setup-sglang-p5e.sh
# Setup script for SGLang HiCache on P5e instance
#
# Run this after SSH'ing into the P5e instance:
#   bash scripts/setup-sglang-p5e.sh
#
# Verifies hardware, pulls the SGLang image, and validates HiCache availability.

set -e

echo "============================================"
echo "SGLang HiCache Setup for P5e"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest-cu130}"
CTR="${CONTAINER_RUNTIME:-nerdctl}"

# ============================================
# Step 1: Verify we're on a P5e instance
# ============================================
log_info "Checking instance type..."
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
echo "Instance type: $INSTANCE_TYPE"

if [[ "$INSTANCE_TYPE" != p5e* ]]; then
    log_warn "Expected p5e instance, got: $INSTANCE_TYPE"
    log_warn "Continuing anyway..."
fi

# ============================================
# Step 2: Check NVIDIA driver and GPUs
# ============================================
log_info "Checking NVIDIA drivers..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log_info "Found $GPU_COUNT GPUs"

if [ "$GPU_COUNT" -ne 8 ]; then
    log_warn "Expected 8 GPUs (H200), found $GPU_COUNT"
fi

# Check CUDA driver version (need 580+ for cu130)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
log_info "NVIDIA driver: $DRIVER_VERSION"

if [ "$DRIVER_MAJOR" -lt 580 ]; then
    log_error "CUDA 13.0 requires driver 580+. Current: $DRIVER_VERSION"
    exit 1
fi

# ============================================
# Step 3: Check EFA interfaces
# ============================================
log_info "Checking EFA interfaces..."
if command -v fi_info &>/dev/null; then
    EFA_COUNT=$(fi_info -p efa 2>/dev/null | grep -c "provider: efa" || echo "0")
    log_info "EFA providers found: $EFA_COUNT"
else
    log_warn "fi_info not available — EFA check skipped"
fi

if ls /dev/infiniband/uverbs* 2>/dev/null; then
    log_info "InfiniBand/EFA devices present"
else
    log_warn "No /dev/infiniband devices found — RDMA/Mooncake may not work"
fi

# ============================================
# Step 4: Pull SGLang image
# ============================================
log_info "Pulling SGLang image: $SGLANG_IMAGE"
${CTR} pull "$SGLANG_IMAGE"

# ============================================
# Step 5: Validate SGLang inside container
# ============================================
log_info "Validating SGLang installation..."

${CTR} run --rm "$SGLANG_IMAGE" \
    python3 -c "
import sglang
print(f'SGLang version: {sglang.__version__}')

# Check HiCache availability
from sglang.srt.server_args import ServerArgs
import inspect
sig = inspect.signature(ServerArgs.__init__)
if 'enable_hierarchical_cache' in sig.parameters:
    print('HiCache: AVAILABLE')
else:
    print('HiCache: NOT AVAILABLE — upgrade SGLang')
    exit(1)
"

if [ $? -eq 0 ]; then
    log_info "SGLang validation passed"
else
    log_error "SGLang validation failed"
    exit 1
fi

# ============================================
# Step 6: Setup directories
# ============================================
log_info "Setting up directories..."
sudo mkdir -p /mnt/fsx
sudo mkdir -p /mnt/nvme
sudo mkdir -p /mnt/nvme/kv-cache
sudo mkdir -p /workspace/results

# ============================================
# Step 7: Build Mooncake image (optional)
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/docker/Dockerfile.sglang-hicache"

if [ -f "$DOCKERFILE" ]; then
    log_info "Building sglang-mooncake:cu130 image (includes Mooncake for Phase 3)..."
    ${CTR} build -f "$DOCKERFILE" -t sglang-mooncake:cu130 "${SCRIPT_DIR}/docker/"
    log_info "Image sglang-mooncake:cu130 built"
else
    log_warn "Dockerfile.sglang-hicache not found at $DOCKERFILE — skipping Mooncake image build"
fi

# ============================================
# Done
# ============================================
echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  Phase 1 (HiCache baseline):    bash configs/sglang-hicache.sh"
echo "  Phase 2 (HiCache + NVMe L3):   bash configs/sglang-hicache-nvme.sh"
echo "  Phase 3 (HiCache + Mooncake):  bash configs/sglang-mooncake.sh"
echo ""
echo "Run benchmarks:  python3 scripts/run-benchmarks.py --config sglang-hicache"
echo ""
