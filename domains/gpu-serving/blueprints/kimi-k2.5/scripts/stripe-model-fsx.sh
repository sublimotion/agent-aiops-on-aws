#!/bin/bash
# stripe-model-fsx.sh
# Re-stripe model files across all FSx Lustre OSTs for maximum throughput
#
# Problem: Default file creation on FSx Lustre uses single OST (~240 MB/s limit)
# Solution: Stripe files across ALL OSTs to achieve aggregate throughput (5-10+ GB/s)
#
# With 98 TiB PERSISTENT_2 at 1000 MB/s/TiB:
# - Single OST: ~240 MB/s (what we observed)
# - All OSTs striped: 5-10 GB/s (with GDS: up to 150 GB/s)
#
# Usage:
#   sudo bash stripe-model-fsx.sh [--model-name <name>] [--fsx-mount <path>]
#
# Examples:
#   sudo bash stripe-model-fsx.sh
#   sudo bash stripe-model-fsx.sh --model-name Kimi-K2.5 --fsx-mount /mnt/fsx-dynamo

set -e

# Default configuration
FSX_MOUNT="${FSX_MOUNT:-/mnt/fsx-dynamo}"
MODEL_NAME="${MODEL_NAME:-Kimi-K2.5}"
STRIPE_COUNT="${STRIPE_COUNT:--1}"  # -1 means all OSTs

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
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --fsx-mount)
            FSX_MOUNT="$2"
            shift 2
            ;;
        --stripe-count)
            STRIPE_COUNT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--model-name <name>] [--fsx-mount <path>] [--stripe-count <count>]"
            echo ""
            echo "Options:"
            echo "  --model-name    Model directory name (default: Kimi-K2.5)"
            echo "  --fsx-mount     FSx mount point (default: /mnt/fsx-dynamo)"
            echo "  --stripe-count  Number of OSTs to stripe across (-1 = all, default: -1)"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

MODEL_SOURCE="${FSX_MOUNT}/models/${MODEL_NAME}"
MODEL_STRIPED="${FSX_MOUNT}/models/${MODEL_NAME}-striped"

echo ""
echo "============================================"
echo "  FSx Lustre File Striping for ${MODEL_NAME}"
echo "============================================"
echo ""

# ============================================
# Step 1: Verify FSx mount and Lustre tools
# ============================================
log_step "1/6: Verifying FSx mount..."

if ! mountpoint -q "$FSX_MOUNT" 2>/dev/null; then
    log_error "FSx not mounted at $FSX_MOUNT"
    log_info "Mount FSx first with: mount -t lustre <fsx-dns>@tcp:/<mount-name> $FSX_MOUNT"
    exit 1
fi

# Check if this is actually Lustre
if ! df -T "$FSX_MOUNT" | grep -q lustre; then
    log_error "$FSX_MOUNT is not a Lustre filesystem"
    exit 1
fi

log_info "FSx mounted at $FSX_MOUNT"
df -h "$FSX_MOUNT"

# ============================================
# Step 2: Check lfs tool availability
# ============================================
log_step "2/6: Checking Lustre tools..."

if ! which lfs >/dev/null 2>&1; then
    log_error "lfs command not found. Install lustre-client package."
    exit 1
fi

log_info "lfs tool available: $(which lfs)"

# ============================================
# Step 3: Analyze current OST configuration
# ============================================
log_step "3/6: Analyzing OST configuration..."

echo ""
echo "Lustre filesystem info:"
lfs df -h "$FSX_MOUNT"

OST_COUNT=$(lfs df "$FSX_MOUNT" | grep -c "OST:" || echo "0")
log_info "Number of OSTs: $OST_COUNT"

# ============================================
# Step 4: Check current model striping
# ============================================
log_step "4/6: Checking current model striping..."

if [ ! -d "$MODEL_SOURCE" ]; then
    log_error "Model not found at: $MODEL_SOURCE"
    log_info "Available models:"
    ls -la "${FSX_MOUNT}/models/" 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo ""
echo "Current striping for model files:"
SAMPLE_FILES=$(ls "$MODEL_SOURCE"/*.safetensors 2>/dev/null | head -3)
if [ -n "$SAMPLE_FILES" ]; then
    for f in $SAMPLE_FILES; do
        echo "  $(basename $f):"
        lfs getstripe "$f" 2>/dev/null | grep -E "stripe_count|stripe_size|obdidx" | head -5
    done
else
    log_warn "No safetensors files found, checking other files..."
    ls "$MODEL_SOURCE" | head -5 | while read f; do
        echo "  $f:"
        lfs getstripe "$MODEL_SOURCE/$f" 2>/dev/null | grep -E "stripe_count|stripe_size" | head -2
    done
fi

# ============================================
# Step 5: Create striped destination directory
# ============================================
log_step "5/6: Creating striped destination directory..."

if [ -d "$MODEL_STRIPED" ]; then
    EXISTING_SIZE=$(du -sh "$MODEL_STRIPED" 2>/dev/null | awk '{print $1}')
    log_warn "Striped directory already exists: $MODEL_STRIPED ($EXISTING_SIZE)"

    read -p "Delete and re-stripe? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing existing striped directory..."
        rm -rf "$MODEL_STRIPED"
    else
        log_info "Keeping existing directory. Exiting."
        exit 0
    fi
fi

mkdir -p "$MODEL_STRIPED"

# Set striping on the new directory BEFORE copying files
# -c -1 means stripe across ALL available OSTs
# -S 1M sets stripe size to 1MB for better large file performance
log_info "Setting stripe pattern on destination directory..."
log_info "  Stripe count: $STRIPE_COUNT (all OSTs)"
log_info "  Stripe size: 1MB"

lfs setstripe -c $STRIPE_COUNT -S 1M "$MODEL_STRIPED"

echo ""
echo "New directory striping:"
lfs getstripe -d "$MODEL_STRIPED"

# ============================================
# Step 6: Copy model with progress
# ============================================
log_step "6/6: Copying model files with optimal striping..."

SOURCE_SIZE=$(du -sh "$MODEL_SOURCE" 2>/dev/null | awk '{print $1}')
log_info "Source model size: $SOURCE_SIZE"
log_info "This will take approximately 5-15 minutes depending on FSx throughput..."

START_TIME=$(date +%s)

# Use rsync to copy with progress
rsync -av --progress --info=progress2 \
    "$MODEL_SOURCE/" "$MODEL_STRIPED/"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Calculate throughput
SOURCE_BYTES=$(du -sb "$MODEL_SOURCE" | awk '{print $1}')
THROUGHPUT_MBPS=$((SOURCE_BYTES / DURATION / 1024 / 1024))

log_info "Copy completed in ${DURATION} seconds (~${THROUGHPUT_MBPS} MB/s)"

# ============================================
# Verify striping on copied files
# ============================================
echo ""
echo "Verifying striping on copied files:"
SAMPLE_STRIPED=$(ls "$MODEL_STRIPED"/*.safetensors 2>/dev/null | head -2)
if [ -n "$SAMPLE_STRIPED" ]; then
    for f in $SAMPLE_STRIPED; do
        echo "  $(basename $f):"
        lfs getstripe "$f" 2>/dev/null | grep -E "stripe_count|stripe_size|obdidx" | head -8
    done
fi

# ============================================
# Summary
# ============================================
echo ""
echo "============================================"
echo "  Striping Complete!"
echo "============================================"
echo ""
echo "Original model (single OST, ~240 MB/s):"
echo "  $MODEL_SOURCE"
echo ""
echo "Striped model (all OSTs, ~5-10+ GB/s):"
echo "  $MODEL_STRIPED"
echo ""
echo "Copy duration: ${DURATION} seconds"
echo "Throughput: ~${THROUGHPUT_MBPS} MB/s"
echo ""
echo "To use striped model:"
echo "  --model $MODEL_STRIPED"
echo ""
echo "Expected performance improvement:"
echo "  Before (1 OST):  ~240 MB/s  → ~4 hours to load 540GB"
echo "  After (all OSTs): ~5-10 GB/s → ~1-2 min to load 540GB"
echo "  With GDS+EFA:    ~50-150 GB/s → seconds to load"
echo ""
echo "Recommendation: For fastest loading, still use NVMe RAID0"
echo "  See: setup-nvme-model.sh"
echo ""
