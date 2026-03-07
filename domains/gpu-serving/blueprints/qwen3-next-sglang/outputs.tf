# Qwen3-Next SGLang Blueprint Outputs

output "vpc_id" {
  description = "VPC ID"
  value       = module.networking.vpc_id
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = module.eks.configure_kubectl
}

# FSx outputs
output "fsx_file_system_id" {
  description = "FSx Lustre filesystem ID"
  value       = var.enable_fsx_lustre ? module.fsx_lustre[0].file_system_id : null
}

output "fsx_dns_name" {
  description = "FSx Lustre DNS name for mounting"
  value       = var.enable_fsx_lustre ? module.fsx_lustre[0].dns_name : null
}

output "fsx_mount_name" {
  description = "FSx Lustre mount name"
  value       = var.enable_fsx_lustre ? module.fsx_lustre[0].mount_name : null
}

# S3 model bucket (existing, referenced)
output "models_bucket_name" {
  description = "S3 bucket for model storage"
  value       = local.models_bucket_id
}

# Benchmark results
output "benchmark_results_bucket" {
  description = "S3 bucket for benchmark results"
  value       = aws_s3_bucket.benchmark_results.id
}

output "quick_start" {
  description = "Quick start commands for Qwen3-Next on SGLang"
  value       = <<-EOT
    # 1. Configure kubectl
    ${module.eks.configure_kubectl}

    # 2. Check GPU node
    kubectl get nodes -l nvidia.com/gpu.present=true

    # 3. SSH to GPU node and copy model to NVMe
    # FSx auto-syncs from S3; copy to NVMe for fastest loading:
    #   cp -r /mnt/fsx/models/qwen3-next-fp8 /mnt/nvme/models/

    # 4. Run SGLang baseline (Phase S0/S1)
    bash configs/sglang-baseline.sh

    # 5. Test inference
    curl http://localhost:30000/v1/models

    # 6. Run HiCache L2 (Phase S3)
    nerdctl rm -f sglang-qwen3-baseline
    bash configs/sglang-hicache-l2.sh

    # 7. Run HiCache L2+L3 NVMe (Phase S3c)
    nerdctl rm -f sglang-qwen3-hicache-l2
    bash configs/sglang-hicache-nvme.sh
  EOT
}

output "instance_config" {
  description = "Instance configuration summary"
  value = {
    instance_type  = var.gpu_instance_types[0]
    gpu_count      = var.sglang_gpu_count
    tp_size        = var.sglang_tp_size
    context_length = var.sglang_context_length
    sglang_image   = var.sglang_image
  }
}
