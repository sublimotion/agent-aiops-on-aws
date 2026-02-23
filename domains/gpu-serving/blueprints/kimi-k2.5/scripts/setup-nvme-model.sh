#!/bin/bash
# setup-nvme-model.sh
# Setup NVMe RAID array and copy model from FSx for faster loading
#
# P5e.48xlarge has 8x 3.84TB NVMe drives = ~30TB total
# RAID0 provides ~25 GB/s sequential read (vs ~1-3 GB/s from FSx)
# Model loading from NVMe: ~20-30 minutes vs ~4 hours from FSx
#
# Usage:
#   sudo bash setup-nvme-model.sh [--fsx-dns <dns>] [--fsx-mount <mount-name>] [--model-name <name>]
#
# Examples:
#   sudo bash setup-nvme-model.sh
#   sudo bash setup-nvme-model.sh --fsx-dns fs-054d9e83babdb7e99.fsx.us-east-2.amazonaws.com --fsx-mount 62q5dbev
#   sudo bash setup-nvme-model.sh --model-name Kimi-K2.5

set -e

# Default configuration
FSX_DNS="${FSX_DNS:-}"
FSX_MOUNT_NAME="${FSX_MOUNT_NAME:-}"
FSX_MOUNT_POINT="/mnt/fsx-dynamo"
NVME_MOUNT_POINT="/mnt/nvme"
MODEL_NAME="${MODEL_NAME:-Kimi-K2.5}"
MODEL_SOURCE_PATH="${FSX_MOUNT_POINT}/models/${MODEL_NAME}"
MODEL_DEST_PATH="${NVME_MOUNT_POINT}/models/${MODEL_NAME}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fsx-dns)
            FSX_DNS="$2"
            shift 2
            ;;
        --fsx-mount)
            FSX_MOUNT_NAME="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            MODEL_SOURCE_PATH="${FSX_MOUNT_POINT}/models/${MODEL_NAME}"
            MODEL_DEST_PATH="${NVME_MOUNT_POINT}/models/${MODEL_NAME}"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--fsx-dns <dns>] [--fsx-mount <mount-name>] [--model-name <name>]"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "============================================"
echo "  NVMe RAID Setup & Model Copy Script"
echo "  For P5e.48xlarge (8x 3.84TB NVMe)"
echo "============================================"
echo ""

# ============================================
# Step 1: Check if running on P5e
# ============================================
log_step "1/7: Checking instance type..."

INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
log_info "Instance type: $INSTANCE_TYPE"

if [[ "$INSTANCE_TYPE" != p5e* && "$INSTANCE_TYPE" != p5.* ]]; then
    log_warn "Expected P5/P5e instance, got: $INSTANCE_TYPE"
    log_warn "NVMe configuration may differ. Continuing anyway..."
fi

# ============================================
# Step 2: Discover NVMe devices
# ============================================
log_step "2/7: Discovering NVMe devices..."

# List all NVMe devices (exclude root volume which is usually nvme0)
NVME_DEVICES=$(lsblk -d -o NAME,SIZE,TYPE | grep nvme | grep -v nvme0 | awk '{print "/dev/"$1}' | sort)
NVME_COUNT=$(echo "$NVME_DEVICES" | wc -l)

if [ -z "$NVME_DEVICES" ] || [ "$NVME_COUNT" -lt 1 ]; then
    log_error "No NVMe devices found (excluding root volume)"
    log_info "Available block devices:"
    lsblk -d
    exit 1
fi

log_info "Found $NVME_COUNT NVMe devices:"
echo "$NVME_DEVICES" | while read dev; do
    SIZE=$(lsblk -d -o SIZE "$dev" | tail -1)
    echo "  $dev ($SIZE)"
done

# ============================================
# Step 3: Check if RAID already exists
# ============================================
log_step "3/7: Checking existing RAID configuration..."

if [ -e /dev/md0 ]; then
    if mountpoint -q "$NVME_MOUNT_POINT" 2>/dev/null; then
        log_info "RAID array /dev/md0 already exists and mounted at $NVME_MOUNT_POINT"
        df -h "$NVME_MOUNT_POINT"
        SKIP_RAID=true
    else
        log_warn "RAID array /dev/md0 exists but not mounted"
        log_info "Attempting to assemble and mount..."
        mdadm --assemble /dev/md0 2>/dev/null || true
        SKIP_RAID=false
    fi
else
    SKIP_RAID=false
fi

# ============================================
# Step 4: Create RAID0 array
# ============================================
if [ "$SKIP_RAID" != "true" ]; then
    log_step "4/7: Creating RAID0 array..."

    # Stop any existing array
    mdadm --stop /dev/md0 2>/dev/null || true

    # Zero superblocks on all devices
    log_info "Zeroing superblocks on NVMe devices..."
    echo "$NVME_DEVICES" | while read dev; do
        mdadm --zero-superblock "$dev" 2>/dev/null || true
    done

    # Create RAID0 array
    DEVICE_LIST=$(echo "$NVME_DEVICES" | tr '\n' ' ')
    log_info "Creating RAID0 with devices: $DEVICE_LIST"

    mdadm --create /dev/md0 \
        --level=0 \
        --raid-devices=$NVME_COUNT \
        $DEVICE_LIST \
        --force

    # Wait for array to sync
    sleep 2

    # Format with XFS (best for large files)
    log_info "Formatting /dev/md0 with XFS..."
    mkfs.xfs -f /dev/md0

    # Create mount point and mount
    mkdir -p "$NVME_MOUNT_POINT"
    mount /dev/md0 "$NVME_MOUNT_POINT"

    log_info "RAID0 array created and mounted:"
    df -h "$NVME_MOUNT_POINT"
else
    log_info "Skipping RAID creation (already exists)"
fi

# ============================================
# Step 5: Mount FSx if needed
# ============================================
log_step "5/7: Checking FSx mount..."

if mountpoint -q "$FSX_MOUNT_POINT" 2>/dev/null; then
    log_info "FSx already mounted at $FSX_MOUNT_POINT"
    df -h "$FSX_MOUNT_POINT"
else
    if [ -z "$FSX_DNS" ] || [ -z "$FSX_MOUNT_NAME" ]; then
        log_warn "FSx not mounted and no DNS/mount-name provided"
        log_info "Attempting to auto-detect FSx from Terraform outputs..."

        # Try to get FSx info from AWS CLI
        FSX_INFO=$(aws fsx describe-file-systems --query 'FileSystems[?FileSystemType==`LUSTRE`].{DNS:DNSName,Mount:LustreMountName}' --output json 2>/dev/null | jq -r '.[0] | "\(.DNS) \(.Mount)"' 2>/dev/null || echo "")

        if [ -n "$FSX_INFO" ] && [ "$FSX_INFO" != "null null" ]; then
            FSX_DNS=$(echo "$FSX_INFO" | awk '{print $1}')
            FSX_MOUNT_NAME=$(echo "$FSX_INFO" | awk '{print $2}')
            log_info "Auto-detected FSx: DNS=$FSX_DNS, Mount=$FSX_MOUNT_NAME"
        else
            log_error "Could not auto-detect FSx. Please provide --fsx-dns and --fsx-mount"
            log_info "Example: $0 --fsx-dns fs-xxx.fsx.us-east-2.amazonaws.com --fsx-mount abcd1234"
            exit 1
        fi
    fi

    log_info "Mounting FSx: ${FSX_DNS}@tcp:/${FSX_MOUNT_NAME} -> $FSX_MOUNT_POINT"
    mkdir -p "$FSX_MOUNT_POINT"
    mount -t lustre -o noatime,flock,lazystatfs \
        "${FSX_DNS}@tcp:/${FSX_MOUNT_NAME}" "$FSX_MOUNT_POINT"

    log_info "FSx mounted successfully:"
    df -h "$FSX_MOUNT_POINT"
fi

# ============================================
# Step 6: Copy model to NVMe
# ============================================
log_step "6/7: Copying model to NVMe..."

MODEL_SOURCE_PATH="${FSX_MOUNT_POINT}/models/${MODEL_NAME}"
MODEL_DEST_PATH="${NVME_MOUNT_POINT}/models/${MODEL_NAME}"

if [ ! -d "$MODEL_SOURCE_PATH" ]; then
    log_error "Model not found at: $MODEL_SOURCE_PATH"
    log_info "Available models on FSx:"
    ls -la "${FSX_MOUNT_POINT}/models/" 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Check source size
SOURCE_SIZE=$(du -sh "$MODEL_SOURCE_PATH" 2>/dev/null | awk '{print $1}')
log_info "Source model: $MODEL_SOURCE_PATH ($SOURCE_SIZE)"

# Check if already copied
if [ -d "$MODEL_DEST_PATH" ]; then
    DEST_SIZE=$(du -sh "$MODEL_DEST_PATH" 2>/dev/null | awk '{print $1}')
    log_info "Model already exists on NVMe: $MODEL_DEST_PATH ($DEST_SIZE)"

    # Compare sizes
    SOURCE_BYTES=$(du -sb "$MODEL_SOURCE_PATH" | awk '{print $1}')
    DEST_BYTES=$(du -sb "$MODEL_DEST_PATH" | awk '{print $1}')

    if [ "$SOURCE_BYTES" -eq "$DEST_BYTES" ]; then
        log_info "Model copy verified (sizes match)"
        SKIP_COPY=true
    else
        log_warn "Model sizes differ (source: $SOURCE_SIZE, dest: $DEST_SIZE)"
        log_info "Re-copying model..."
        rm -rf "$MODEL_DEST_PATH"
        SKIP_COPY=false
    fi
else
    SKIP_COPY=false
fi

if [ "$SKIP_COPY" != "true" ]; then
    mkdir -p "${NVME_MOUNT_POINT}/models"

    log_info "Copying model from FSx to NVMe..."
    log_info "This will take approximately 5-15 minutes for a 500GB model"

    # Use rsync with progress
    START_TIME=$(date +%s)

    rsync -av --progress --info=progress2 \
        "$MODEL_SOURCE_PATH/" "$MODEL_DEST_PATH/"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    log_info "Model copy completed in ${DURATION} seconds"

    # Verify copy
    DEST_SIZE=$(du -sh "$MODEL_DEST_PATH" 2>/dev/null | awk '{print $1}')
    log_info "Copied model size: $DEST_SIZE"
fi

# ============================================
# Step 7: Create symlink and update config
# ============================================
log_step "7/7: Creating configuration..."

# Create a convenient symlink
ln -sfn "$MODEL_DEST_PATH" /workspace/model 2>/dev/null || true

# Display final status
echo ""
echo "============================================"
echo "  NVMe Setup Complete!"
echo "============================================"
echo ""
echo "NVMe RAID Array:"
df -h "$NVME_MOUNT_POINT"
echo ""
echo "Model Location:"
echo "  FSx:  $MODEL_SOURCE_PATH"
echo "  NVMe: $MODEL_DEST_PATH"
echo ""
echo "Model Files:"
ls -la "$MODEL_DEST_PATH" | head -10
echo ""
echo "To use NVMe model with Dynamo/vLLM:"
echo "  --model $MODEL_DEST_PATH"
echo ""
echo "Expected loading time improvement:"
echo "  FSx:  ~4 hours (64 shards × 4 min/shard)"
echo "  NVMe: ~30 minutes (64 shards × 30s/shard)"
echo ""

# Create helper script for starting vLLM with NVMe model
# CRITICAL: Uses --enforce-eager and NCCL_TIMEOUT for MoE model loading
cat > /usr/local/bin/start-vllm-nvme.sh << 'SCRIPT'
#!/bin/bash
# Start vLLM with Kimi K2.5 from NVMe
# Optimized for MoE model loading (see docs/MOE_LOADING_BEST_PRACTICES.md)

MODEL_PATH="/mnt/nvme/models/Kimi-K2.5"

# CRITICAL environment variables for MoE loading
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_TIMEOUT=1800  # 30 min timeout for large MoE models
export VLLM_ATTENTION_BACKEND=FLASHINFER

echo "============================================"
echo "  Starting vLLM with Kimi K2.5 (MoE)"
echo "============================================"
echo ""
echo "Model: $MODEL_PATH"
echo ""
echo "Key MoE optimizations:"
echo "  --enforce-eager           Skip torch.compile (prevents hangs)"
echo "  --gpu-memory-utilization 0.85  Leave headroom for expert routing"
echo "  --swap-space 32           Handle MoE memory variability"
echo "  NCCL_TIMEOUT=1800         Extended timeout for large models"
echo ""

# Kill any existing vLLM processes
pkill -9 -f "vllm.*Kimi" 2>/dev/null || true
sleep 2

screen -dmS vllm bash -c "python3.11 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --enforce-eager \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --swap-space 32 \
    --disable-log-requests \
    --port 30080 \
    2>&1 | tee /tmp/vllm-nvme.log"

echo "vLLM started in screen session 'vllm'"
echo ""
echo "Monitor loading progress:"
echo "  screen -r vllm"
echo "  tail -f /tmp/vllm-nvme.log | grep -E 'Loading|shards|Progress'"
echo "  watch -n 5 'nvidia-smi --query-gpu=memory.used --format=csv'"
echo ""
echo "Expected loading time: 25-35 minutes (64 shards from NVMe)"
echo "Health check: curl http://localhost:30080/health"
SCRIPT
chmod +x /usr/local/bin/start-vllm-nvme.sh

cat > /usr/local/bin/start-dynamo-nvme.sh << 'SCRIPT'
#!/bin/bash
# Start Dynamo with Kimi K2.5 from NVMe
# Optimized for MoE model loading (see docs/MOE_LOADING_BEST_PRACTICES.md)

MODEL_PATH="/mnt/nvme/models/Kimi-K2.5"

# CRITICAL environment variables for MoE loading
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_TIMEOUT=1800  # 30 min timeout for large MoE models
export VLLM_ATTENTION_BACKEND=FLASHINFER

echo "============================================"
echo "  Starting Dynamo with Kimi K2.5 (MoE)"
echo "============================================"
echo ""
echo "Model: $MODEL_PATH"
echo ""
echo "Key MoE optimizations:"
echo "  --enforce-eager           Skip torch.compile (prevents hangs)"
echo "  --gpu-memory-utilization 0.85  Leave headroom for expert routing"
echo "  NCCL_TIMEOUT=1800         Extended timeout for large models"
echo ""

# Kill any existing Dynamo processes
pkill -9 -f "dynamo.*Kimi" 2>/dev/null || true
pkill -9 -f "python.*dynamo" 2>/dev/null || true
sleep 2

screen -dmS dynamo bash -c "python3.11 -m dynamo.vllm \
    --model $MODEL_PATH \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --enforce-eager \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --store-kv mem \
    --request-plane tcp \
    --event-plane zmq \
    --port 30080 \
    2>&1 | tee /tmp/dynamo-nvme.log"

echo "Dynamo started in screen session 'dynamo'"
echo ""
echo "Monitor loading progress:"
echo "  screen -r dynamo"
echo "  tail -f /tmp/dynamo-nvme.log | grep -E 'Loading|shards|Progress'"
echo "  watch -n 5 'nvidia-smi --query-gpu=memory.used --format=csv'"
echo ""
echo "Expected loading time: 25-35 minutes (64 shards from NVMe)"
echo "Health check: curl http://localhost:30080/health"
SCRIPT
chmod +x /usr/local/bin/start-dynamo-nvme.sh

log_info "Helper scripts created:"
echo "  start-vllm-nvme.sh   - Start vLLM with NVMe model"
echo "  start-dynamo-nvme.sh - Start Dynamo with NVMe model"
echo ""
