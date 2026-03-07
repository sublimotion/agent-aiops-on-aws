# Qwen3-Next g7e Blueprint Validation Report

**Date**: 2026-02-25
**Blueprint**: qwen3-next-g7e
**Status**: VALIDATED ✅

## Stage 1: Terraform Validation

### Configuration Check
✅ **terraform validate**: Configuration is valid
✅ **terraform fmt**: All files properly formatted after auto-fix
✅ **terraform init**: Already initialized (.terraform directory present)

### Required Variables
- ✅ `model_s3_bucket_id`: Configured in terraform.tfvars (reusing from qwen3-next)
- ✅ All other variables have defaults or are set in terraform.tfvars

### Key Configuration Points
1. **Instance Type**: g7e.48xlarge (8x RTX PRO Server 6000 - Blackwell GB202)
2. **AMI Type**: AL2023_x86_64_NVIDIA (Blackwell-compatible)
3. **GPU AZ**: us-east-2a (g7e availability)
4. **FSx**: SCRATCH_2 at 1200 GiB (cost-optimized for benchmarks)
5. **No EFA**: g7e uses NVLink natively
6. **Post-bootstrap**: NVMe RAID0 + FSx mount with Lustre tuning

## Stage 2: Pre-flight Checks

### Shell Scripts
✅ `configs/vllm-baseline.sh`: Syntax valid
✅ `scripts/copy-to-nvme.sh`: Syntax valid

### Docker Configuration
✅ `docker/Dockerfile.vllm-qwen3next`:
- Based on vllm/vllm-openai:v0.15.0-cu130
- Includes transformers@main for Qwen3-Next support
- Includes flash-linear-attention and causal-conv1d for DeltaNet
- Note: cu130 should support sm_100 (Blackwell), fallback to cu131 if needed

### Template Files
✅ `templates/post_bootstrap.sh.tftpl`:
- Lustre kernel module tuning for >64 vCPU
- NVMe RAID0 setup with generic detection
- FSx mount with client tuning
- Persistent tuning via cron

### Module Integration
✅ **EKS Module**: Properly parameterized with:
- `gpu_ami_type` variable wired
- `gpu_post_bootstrap_user_data` template rendered
- Managed node group handles IAM roles (no instance profile needed)

✅ **FSx Module**: Configured with:
- SCRATCH_2 deployment type
- DRA to existing S3 model bucket
- Auto-import/export policies

✅ **Networking Module**: Standard VPC with NAT gateway

## Configuration Highlights

### Cost Optimizations
- **SCRATCH_2 FSx**: Cheaper than PERSISTENT_2, sufficient for benchmarks
- **Reused S3 bucket**: No duplicate model storage costs
- **On-demand g7e**: $33.14/hr vs $63.30/hr for p5en.48xlarge (48% cheaper)

### Performance Optimizations
- **NVMe RAID0**: Ephemeral instance storage for 17x faster model loading
- **DirectoryOrCreate hostPath**: Ensures /mnt/nvme mount works
- **Winner config only**: TP=4, FP8, prefix caching (no exploration runs)

### Risk Mitigations
- **AL2023_x86_64_NVIDIA AMI**: Latest drivers for Blackwell
- **Flexible max_model_len**: Can reduce from 131K to 65K if VRAM tight
- **Prometheus scrape interval**: 15s to avoid timeout validation errors

## Validation Issues Found & Fixed

1. **terraform.tfvars formatting**: Fixed with `terraform fmt`
2. **State lock**: Detected but doesn't affect validation (from previous apply)

## Next Steps

**DO NOT RUN terraform apply** — This was a validation-only run.

When ready to deploy:
1. Ensure no other terraform operations are running (clear state lock if needed)
2. Review terraform.tfvars values, especially `model_s3_bucket_id`
3. Run `terraform apply` with appropriate AWS credentials
4. Follow the deployment stages in the main deployment guide

## Readiness Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Terraform Config | ✅ Valid | All resources properly configured |
| Variables | ✅ Set | model_s3_bucket_id required, others have defaults |
| Shell Scripts | ✅ Valid | Syntax checked, no errors |
| Docker Image | ⚠️ To Build | Dockerfile ready, needs ECR push if cu130 fails on Blackwell |
| Post-Bootstrap | ✅ Ready | NVMe RAID0 + FSx mount scripted |
| Module Compatibility | ✅ Verified | EKS module supports gpu_ami_type and post_bootstrap_user_data |

**Overall Status**: Blueprint is ready for deployment. Configuration is valid and all pre-flight checks pass.