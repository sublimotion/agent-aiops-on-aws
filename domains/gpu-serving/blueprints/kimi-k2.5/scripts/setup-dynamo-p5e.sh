#!/bin/bash
# setup-dynamo-p5e.sh
# Setup script for NVIDIA Dynamo KVBM on P5e instance
#
# Run this after SSH'ing into the P5e instance:
#   curl -sSL <url>/setup-dynamo-p5e.sh | bash
#
# Or copy and run manually

set -e

echo "============================================"
echo "NVIDIA Dynamo KVBM Setup for P5e"
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

# ============================================
# Step 3: Check GDS support
# ============================================
log_info "Checking GDS support..."
if ls /dev/nvidia-fs* 2>/dev/null; then
    log_info "GDS device found"
else
    log_warn "GDS device not found - may need to load nvidia-fs module"
    # Try loading the module
    sudo modprobe nvidia-fs 2>/dev/null || log_warn "Could not load nvidia-fs module"
fi

# Check for gdscheck
if which gdscheck >/dev/null 2>&1; then
    log_info "gdscheck available at: $(which gdscheck)"
elif [ -f /usr/local/cuda/gds/tools/gdscheck ]; then
    log_info "gdscheck available at: /usr/local/cuda/gds/tools/gdscheck"
else
    log_warn "gdscheck not found - GDS tools may not be installed"
fi

# ============================================
# Step 4: Install Python packages
# ============================================
log_info "Installing Python packages..."

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install ai-dynamo with vLLM backend
log_info "Installing ai-dynamo[vllm]..."
pip install "ai-dynamo[vllm]>=0.9.0"

# Install NIXL with CUDA 12 support
log_info "Installing nixl[cu12]..."
pip install "nixl[cu12]>=0.9.0"

# Install additional dependencies
log_info "Installing additional dependencies..."
pip install \
    "transformers>=5.0.0" \
    etcd3 \
    nats-py \
    pyzmq \
    msgpack \
    pandas \
    matplotlib \
    tqdm

# ============================================
# Step 5: Verify installation
# ============================================
log_info "Verifying installation..."

echo ""
echo "Package versions:"
pip show ai-dynamo | grep -E "^(Name|Version):" || log_error "ai-dynamo not installed"
pip show ai-dynamo-vllm | grep -E "^(Name|Version):" || log_warn "ai-dynamo-vllm not found (may be bundled)"
pip show nixl | grep -E "^(Name|Version):" || log_error "nixl not installed"

# Test import
python3 -c "import dynamo; print(f'Dynamo import OK: {dynamo.__version__}')" 2>/dev/null || \
    python3 -c "import ai_dynamo; print('ai_dynamo import OK')" 2>/dev/null || \
    log_warn "Could not import dynamo module"

# ============================================
# Step 6: Setup directories
# ============================================
log_info "Setting up directories..."
sudo mkdir -p /mnt/fsx
sudo mkdir -p /mnt/nvme
sudo mkdir -p /workspace/results
sudo mkdir -p /workspace/models

# ============================================
# Step 7: Download Kimi K2.5 tokenizer
# ============================================
log_info "Pre-downloading Kimi K2.5 tokenizer..."
python3 -c "
from transformers import AutoTokenizer
print('Downloading Kimi-K2.5 tokenizer...')
try:
    AutoTokenizer.from_pretrained('moonshotai/Kimi-K2.5', trust_remote_code=True)
    print('Tokenizer downloaded successfully')
except Exception as e:
    print(f'Tokenizer download failed: {e}')
    print('Will use FSx copy instead')
" || log_warn "Tokenizer download failed - will use FSx copy"

# ============================================
# Step 8: Create helper scripts
# ============================================
log_info "Creating helper scripts..."

# GDS validation script
cat > /usr/local/bin/validate-gds.sh << 'SCRIPT'
#!/bin/bash
echo "=== GDS Validation ==="
echo ""
echo "1. NVIDIA driver:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1

echo ""
echo "2. GDS device:"
ls -la /dev/nvidia-fs* 2>/dev/null || echo "Not found"

echo ""
echo "3. cuFile:"
gdscheck -V 2>/dev/null || /usr/local/cuda/gds/tools/gdscheck -V 2>/dev/null || echo "Not found"

echo ""
echo "4. FSx mount:"
if mountpoint -q /mnt/fsx 2>/dev/null; then
    df -h /mnt/fsx
    echo ""
    echo "5. GDS on FSx:"
    gdscheck -p /mnt/fsx 2>/dev/null || /usr/local/cuda/gds/tools/gdscheck -p /mnt/fsx 2>/dev/null
else
    echo "FSx not mounted"
fi
SCRIPT
sudo chmod +x /usr/local/bin/validate-gds.sh

# FSx mount script
cat > /usr/local/bin/mount-fsx.sh << 'SCRIPT'
#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: mount-fsx.sh <fsx-dns-name> [mount-name]"
    echo "Example: mount-fsx.sh fs-xxx.fsx.us-east-2.amazonaws.com fsx12345"
    exit 1
fi

FSX_DNS=$1
MOUNT_NAME=${2:-$(echo $FSX_DNS | cut -d'.' -f1 | sed 's/fs-//')}

echo "Mounting FSx: $FSX_DNS / $MOUNT_NAME"
sudo mkdir -p /mnt/fsx
sudo mount -t lustre -o noatime,flock,lazystatfs \
    ${FSX_DNS}@tcp:/${MOUNT_NAME} /mnt/fsx

echo "Verifying mount..."
df -h /mnt/fsx
SCRIPT
sudo chmod +x /usr/local/bin/mount-fsx.sh

# ============================================
# Step 9: Set environment variables
# ============================================
log_info "Setting environment variables..."

cat >> ~/.bashrc << 'ENV'

# Dynamo KVBM environment
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DYNAMO_EVENT_PLANE=zmq
export CUFILE_ENV_PATH_JSON=/etc/cufile.json
export NIXL_LOG_LEVEL=INFO
export VLLM_ATTENTION_BACKEND=FLASHINFER
ENV

source ~/.bashrc

# ============================================
# Done
# ============================================
echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Mount FSx: mount-fsx.sh <fsx-dns-name> <mount-name>"
echo "  2. Validate GDS: validate-gds.sh"
echo "  3. Copy model to FSx: rsync -av /path/to/Kimi-K2.5 /mnt/fsx/models/"
echo "  4. Run Dynamo: dynamo-run --config /etc/dynamo/config.yaml"
echo ""
