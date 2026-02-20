#!/bin/bash
# setup-lmcache-p5e.sh
# Setup script for LMCache on P5e instance for comparison benchmarks
#
# Run after Dynamo benchmarks complete:
#   bash setup-lmcache-p5e.sh

set -e

echo "============================================"
echo "LMCache Setup for P5e (Comparison)"
echo "============================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ============================================
# Step 1: Install LMCache
# ============================================
log_info "Installing LMCache..."

# LMCache requires specific vLLM version
pip3.11 install --upgrade pip setuptools wheel

# Install LMCache with vLLM integration
pip3.11 install "lmcache[vllm]>=0.5.0" 2>/dev/null || \
    pip3.11 install lmcache vllm lmcache-vllm 2>/dev/null || \
    log_warn "Standard install failed, trying alternative..."

# Alternative: install from GitHub if PyPI fails
if ! python3.11 -c "import lmcache" 2>/dev/null; then
    log_info "Trying GitHub install..."
    pip3.11 install git+https://github.com/LMCache/LMCache.git
fi

# ============================================
# Step 2: Verify installation
# ============================================
log_info "Verifying LMCache installation..."

python3.11 -c "
import lmcache
print(f'LMCache version: {lmcache.__version__}')
" || log_warn "LMCache import failed"

# ============================================
# Step 3: Create LMCache config for FSx
# ============================================
log_info "Creating LMCache configuration..."

cat > /tmp/lmcache-config.yaml << 'EOF'
# LMCache configuration for FSx comparison
storage:
  # FSx Lustre backend
  type: disk
  path: /mnt/fsx-dynamo/lmcache
  max_size_gb: 50000  # 50 TiB

cache:
  # Enable prefix caching
  enable_prefix_caching: true
  # Block size in tokens
  block_size: 64
  # Eviction policy
  eviction_policy: lru

vllm:
  # Match Dynamo settings
  tensor_parallel_size: 8
  max_model_len: 32768
  gpu_memory_utilization: 0.85
  enable_prefix_caching: true

model:
  name: moonshotai/Kimi-K2.5
  path: /mnt/fsx-dynamo/models/Kimi-K2.5
  trust_remote_code: true
EOF

# ============================================
# Step 4: Create LMCache launch script
# ============================================
log_info "Creating LMCache launch script..."

cat > /usr/local/bin/start-lmcache.sh << 'SCRIPT'
#!/bin/bash
# Start LMCache with vLLM for Kimi K2.5

export HF_HUB_OFFLINE=1
export VLLM_ATTENTION_BACKEND=FLASHINFER

# Create LMCache storage directory
mkdir -p /mnt/fsx-dynamo/lmcache

# Start vLLM with LMCache enabled
nohup python3.11 -m lmcache.vllm.entrypoints.openai.api_server \
    --model /mnt/fsx-dynamo/models/Kimi-K2.5 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --lmcache-config-file /tmp/lmcache-config.yaml \
    --port 30081 \
    > /tmp/lmcache-server.log 2>&1 &

echo $! > /tmp/lmcache.pid
echo "LMCache server started with PID $(cat /tmp/lmcache.pid)"
echo "Logs: /tmp/lmcache-server.log"
SCRIPT
chmod +x /usr/local/bin/start-lmcache.sh

# ============================================
# Done
# ============================================
echo ""
echo "============================================"
echo "LMCache setup complete!"
echo "============================================"
echo ""
echo "To start LMCache server:"
echo "  start-lmcache.sh"
echo ""
echo "To run comparison benchmarks:"
echo "  python3.11 run_comparison_benchmarks.py"
echo ""
