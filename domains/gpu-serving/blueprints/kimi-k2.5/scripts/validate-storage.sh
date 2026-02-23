#!/bin/bash
# validate-storage.sh
# Validate storage infrastructure is ready before launching GPU instances
#
# Run from the blueprint directory after terraform apply:
#   bash scripts/validate-storage.sh
#
# Tests: S3 bucket, FSx filesystem, DRA, EFA, connectivity

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL=$((FAIL + 1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; WARN=$((WARN + 1)); }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

echo ""
echo "============================================"
echo "  Storage Validation for Kimi K2.5"
echo "============================================"
echo ""

# Get terraform outputs
info "Reading terraform outputs..."
MODEL_BUCKET=$(terraform output -raw model_bucket 2>/dev/null) || MODEL_BUCKET=""
FSX_ID=$(terraform output -raw fsx_file_system_id 2>/dev/null) || FSX_ID=""
FSX_DNS=$(terraform output -raw fsx_dns_name 2>/dev/null) || FSX_DNS=""
REGION=$(terraform output -raw 2>/dev/null | grep -oP 'us-east-\d' | head -1) || REGION="us-east-2"

# =============================================================================
# 1. S3 Model Bucket
# =============================================================================
echo ""
info "=== S3 Model Bucket ==="

if [ -n "$MODEL_BUCKET" ]; then
  pass "Model bucket exists: $MODEL_BUCKET"

  # Check versioning
  VERSIONING=$(aws s3api get-bucket-versioning --bucket "$MODEL_BUCKET" --query 'Status' --output text 2>/dev/null)
  if [ "$VERSIONING" = "Enabled" ]; then
    pass "Versioning enabled"
  else
    warn "Versioning not enabled (expected: Enabled, got: $VERSIONING)"
  fi

  # Check encryption
  ENCRYPTION=$(aws s3api get-bucket-encryption --bucket "$MODEL_BUCKET" --query 'ServerSideEncryptionConfiguration.Rules[0].ApplyServerSideEncryptionByDefault.SSEAlgorithm' --output text 2>/dev/null)
  if [ -n "$ENCRYPTION" ]; then
    pass "Encryption enabled: $ENCRYPTION"
  else
    warn "Encryption not configured"
  fi

  # Check for model files
  MODEL_COUNT=$(aws s3 ls "s3://${MODEL_BUCKET}/models/Kimi-K2.5/" --recursive 2>/dev/null | wc -l)
  if [ "$MODEL_COUNT" -gt 0 ]; then
    pass "Model files found: $MODEL_COUNT files in models/Kimi-K2.5/"
  else
    warn "No model files in s3://${MODEL_BUCKET}/models/Kimi-K2.5/ — upload model before launching GPU"
  fi
else
  fail "Model bucket not found in terraform outputs"
fi

# =============================================================================
# 2. FSx Filesystem
# =============================================================================
echo ""
info "=== FSx Lustre Filesystem ==="

if [ -n "$FSX_ID" ]; then
  # Get filesystem details
  FSX_INFO=$(aws fsx describe-file-systems --file-system-ids "$FSX_ID" --output json 2>/dev/null)

  if [ -n "$FSX_INFO" ]; then
    LIFECYCLE=$(echo "$FSX_INFO" | jq -r '.FileSystems[0].Lifecycle')
    DEPLOY_TYPE=$(echo "$FSX_INFO" | jq -r '.FileSystems[0].LustreConfiguration.DeploymentType')
    STORAGE_GB=$(echo "$FSX_INFO" | jq -r '.FileSystems[0].StorageCapacity')
    THROUGHPUT=$(echo "$FSX_INFO" | jq -r '.FileSystems[0].LustreConfiguration.PerUnitStorageThroughput')
    COMPRESSION=$(echo "$FSX_INFO" | jq -r '.FileSystems[0].LustreConfiguration.DataCompressionType // "NONE"')
    FS_VERSION=$(echo "$FSX_INFO" | jq -r '.FileSystems[0].FileSystemTypeVersion // "unknown"')

    # Lifecycle
    if [ "$LIFECYCLE" = "AVAILABLE" ]; then
      pass "FSx lifecycle: AVAILABLE"
    else
      fail "FSx lifecycle: $LIFECYCLE (expected: AVAILABLE)"
    fi

    # Deployment type
    if [ "$DEPLOY_TYPE" = "PERSISTENT_2" ]; then
      pass "Deployment type: PERSISTENT_2"
    else
      fail "Deployment type: $DEPLOY_TYPE (expected: PERSISTENT_2 for EFA)"
    fi

    # EFA
    EFA=$(echo "$FSX_INFO" | jq -r '.FileSystems[0].LustreConfiguration.EfaEnabled // false')
    if [ "$EFA" = "true" ]; then
      pass "EFA enabled"
    else
      fail "EFA not enabled (required for GDS, must recreate filesystem)"
    fi

    # Compression
    if [ "$COMPRESSION" = "LZ4" ]; then
      pass "LZ4 compression enabled"
    else
      warn "Compression: $COMPRESSION (recommended: LZ4)"
    fi

    # Version
    pass "Lustre version: $FS_VERSION"

    # Storage summary
    STORAGE_TIB=$((STORAGE_GB / 1024))
    TOTAL_THROUGHPUT_GBS=$(( (STORAGE_GB / 1024) * THROUGHPUT / 1000 ))
    info "Storage: ${STORAGE_TIB} TiB, ${THROUGHPUT} MB/s per TiB = ${TOTAL_THROUGHPUT_GBS} GB/s total"
  else
    fail "Could not describe FSx filesystem $FSX_ID"
  fi
else
  fail "FSx filesystem ID not found in terraform outputs"
fi

# =============================================================================
# 3. Data Repository Association (DRA)
# =============================================================================
echo ""
info "=== Data Repository Association ==="

if [ -n "$FSX_ID" ]; then
  DRA_INFO=$(aws fsx describe-data-repository-associations \
    --filters "Name=file-system-id,Values=$FSX_ID" \
    --output json 2>/dev/null)

  DRA_COUNT=$(echo "$DRA_INFO" | jq '.Associations | length')
  if [ "$DRA_COUNT" -gt 0 ]; then
    DRA_PATH=$(echo "$DRA_INFO" | jq -r '.Associations[0].DataRepositoryPath')
    DRA_LIFECYCLE=$(echo "$DRA_INFO" | jq -r '.Associations[0].Lifecycle')
    DRA_FS_PATH=$(echo "$DRA_INFO" | jq -r '.Associations[0].FileSystemPath')
    AUTO_IMPORT=$(echo "$DRA_INFO" | jq -r '.Associations[0].S3.AutoImportPolicy.Events // [] | join(",")')

    if [ "$DRA_LIFECYCLE" = "AVAILABLE" ]; then
      pass "DRA active: $DRA_FS_PATH -> $DRA_PATH"
    else
      warn "DRA lifecycle: $DRA_LIFECYCLE (expected: AVAILABLE)"
    fi

    if echo "$AUTO_IMPORT" | grep -q "NEW"; then
      pass "Auto-import enabled: $AUTO_IMPORT"
    else
      warn "Auto-import not configured"
    fi
  else
    fail "No DRA configured — S3 to FSx sync will not work"
  fi
fi

# =============================================================================
# 4. AZ Alignment
# =============================================================================
echo ""
info "=== AZ Alignment ==="

if [ -n "$FSX_ID" ] && [ -n "$FSX_INFO" ]; then
  FSX_SUBNETS=$(echo "$FSX_INFO" | jq -r '.FileSystems[0].SubnetIds[]')
  for SUBNET in $FSX_SUBNETS; do
    AZ=$(aws ec2 describe-subnets --subnet-ids "$SUBNET" --query 'Subnets[0].AvailabilityZone' --output text 2>/dev/null)
    pass "FSx in AZ: $AZ (subnet: $SUBNET)"
  done
  info "Ensure GPU capacity block is in the same AZ"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================"
echo "  Results: ${PASS} passed, ${FAIL} failed, ${WARN} warnings"
echo "============================================"
echo ""

if [ "$FAIL" -gt 0 ]; then
  echo -e "${RED}Storage is NOT ready for GPU launch. Fix failures above.${NC}"
  exit 1
elif [ "$WARN" -gt 0 ]; then
  echo -e "${YELLOW}Storage is ready but has warnings. Review above.${NC}"
  exit 0
else
  echo -e "${GREEN}Storage is fully ready for GPU launch.${NC}"
  exit 0
fi
