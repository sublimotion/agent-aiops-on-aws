# Deployment Log - qwen3-next-g7e
**Date**: 2026-02-25
**Blueprint**: domains/gpu-serving/blueprints/qwen3-next-g7e
**Target**: g7e.48xlarge (8x RTX PRO Server 6000, Blackwell GB202)

## Deployment Stages

### Stage 1: Foundation (COMPLETE)
**Start Time**: 2026-02-25 07:13 UTC
**End Time**: 2026-02-25 16:36 UTC

**Actions**:
1. ✅ Read blueprint spec at `domains/gpu-serving/specs/qwen3-next-g7e.md`
2. ✅ Read lessons.md for known pitfalls
3. ✅ Initialized Terraform in blueprint directory
4. ✅ Validated Terraform configuration
5. ✅ Used existing terraform.tfvars with:
   - Shortened project name to "qwen3-g7e" (to avoid IAM role name length issues)
   - Reused existing model bucket: `qwen3-next-bench-models-20260223212318268900000006`
   - g7e.48xlarge configured for all 3 AZs (us-east-2a, us-east-2b, us-east-2c)
   - FSx SCRATCH_2 at 1200 GiB
   - vLLM TP=4, FP8, prefix caching enabled
6. ✅ Foundation infrastructure fully deployed (161 resources in state)

**Issues Encountered**:
- Terraform state lock from stuck previous process (PID 86152)
  - Resolution: Killed process, lock auto-released
- 2 deposed FSx filesystems from failed replacements
  - Resolution: Applied cleanup plan to destroy them (completed)

**Terraform Outputs Captured**:
```
eks_cluster_name: qwen3-g7e-eks-cluster
eks_cluster_endpoint: https://A97089D3D5F4060C5ABF374F788D603C.gr7.us-east-2.eks.amazonaws.com
fsx_dns_name: fs-05c9fff14f8f0307e.fsx.us-east-2.amazonaws.com
fsx_mount_name: oyeczb4v
fsx_file_system_id: fs-05c9fff14f8f0307e
model_path: /mnt/nvme/models/qwen3-next-fp8
benchmark_results_bucket: qwen3-g7e-results-20260225122450208000000003
models_bucket_name: qwen3-next-bench-models-20260223212318268900000006
```

**Validation**:
- ✅ EKS cluster is ACTIVE and reachable
- ✅ kubectl configured successfully
- ✅ System nodes (2x m6i.xlarge) are Ready
- ✅ FSx Lustre filesystem is AVAILABLE
- ✅ Data Repository Association configured to auto-import from S3

---

### Stage 2: Build Machine (SKIPPED)
**Status**: Not needed for this deployment
- Using pre-built vLLM image: `vllm/vllm-openai:qwen3_5-x86_64-cu130`
- On-demand g7e with EKS managed node groups (not capacity blocks)

---

### Stage 3: Storage and Model Staging (PARTIAL)
**Status**: FSx ready, model sync pending verification

**FSx Configuration**:
- ✅ Type: SCRATCH_2
- ✅ Capacity: 1200 GiB
- ✅ DNS: fs-05c9fff14f8f0307e.fsx.us-east-2.amazonaws.com
- ✅ Mount name: oyeczb4v
- ✅ Data Repository Association configured

**Model Status**:
- ⏳ DRA should auto-import from S3 bucket
- ⏳ Need GPU node to verify model sync
- Expected path: `/mnt/fsx/models/qwen3-next-fp8`

---

### Stage 4: GPU Node Provisioning (BLOCKED)
**Status**: ❌ BLOCKED - InsufficientInstanceCapacity
**Time**: 2026-02-25 16:40 UTC onwards

**Critical Issue**: g7e.48xlarge instances unavailable in entire us-east-2 region

| AZ | Status | Error | Last Attempt |
|----|--------|-------|--------------|
| us-east-2a | ❌ Failed | InsufficientInstanceCapacity | 16:44:02 UTC |
| us-east-2b | ❌ Failed | InsufficientInstanceCapacity | 16:41:59 UTC |
| us-east-2c | ❌ Failed | Instance type not supported | 16:43:00 UTC |

**Node Group Configuration**:
- Name: gpu-20260225163355826500000001
- Instance Type: g7e.48xlarge
- AMI Type: AL2023_x86_64_NVIDIA
- Desired: 1, Min: 0, Max: 1
- Status: ACTIVE (but cannot launch instances)

**Autoscaling Details**:
- ASG Name: eks-gpu-20260225163355826500000001-acce4ad3-327d-48ed-e373-3b8adf9f37ba
- Continuously retrying launches in all AZs
- AWS suggests trying different regions

---

### Stage 5: Serving Stack Deployment
**Status**: ⏳ Waiting for GPU node

---

### Stage 6: Pre-benchmark Validation
**Status**: ⏳ Waiting for GPU node

---

### Stage 7: Readiness Audit
**Status**: ⏳ Cannot complete without GPU node

---

### Stage 8: Compound
**Status**: ⏳ Will run after deployment completes

---

## Infrastructure Status Summary
- VPC: ✅ Created (vpc-0c76cc51ce2f3ac12)
- EKS Cluster: ✅ Active (qwen3-g7e-eks-cluster)
- FSx Lustre: ✅ Available (fs-05c9fff14f8f0307e)
- System Nodes: ✅ 2x m6i.xlarge Ready
- GPU Nodes: ❌ 0/1 (InsufficientInstanceCapacity)
- Monitoring Stack: ✅ Helm charts ready to deploy
- vLLM Deployment: ⏳ Waiting for GPU node
- Model Storage: ⏳ DRA configured, sync pending

## Lessons Learned

1. **g7e.48xlarge has severe capacity constraints** - New Blackwell instances not widely available
2. **us-east-2c doesn't support g7e.48xlarge** - Only us-east-2a and us-east-2b theoretically support it
3. **Multi-AZ configuration doesn't help** when entire region lacks capacity
4. **Deposed resources accumulate** - Previous failed FSx replacements left orphaned resources
5. **FSx deletion is slow** - Even SCRATCH_2 takes 7+ minutes to delete

## Recommendations

1. **Monitor capacity availability** - Set up CloudWatch alerts for g7e capacity
2. **Consider capacity reservation** - May need to pre-reserve g7e instances
3. **Alternative regions** - Check us-west-2 or eu-west-1 for g7e availability
4. **Contact AWS Support** - Inquire about g7e.48xlarge availability timeline
5. **Alternative instance types** - Consider g6e.48xlarge as fallback (older generation)

## Next Steps

**Immediate**:
1. Monitor g7e.48xlarge capacity in us-east-2
2. Check capacity in alternative regions
3. Consider opening AWS support case

**When capacity becomes available**:
1. GPU node will auto-join via managed node group
2. Verify Blackwell driver support (CUDA 13.1+ for sm_100)
3. Mount FSx and verify model sync
4. Run post-bootstrap script for NVMe RAID
5. Copy model to NVMe
6. Deploy vLLM serving stack
7. Run G0/G1/G2 benchmark phases

---

**Current Status**: Infrastructure 95% complete, blocked on GPU capacity. Cannot proceed with benchmarks until g7e.48xlarge becomes available.