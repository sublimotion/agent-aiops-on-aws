# Qwen3-Next g7e Blueprint Outputs

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

output "fsx_mount_command" {
  description = "Command to mount FSx on a compute node"
  value       = var.enable_fsx_lustre ? "sudo mount -t lustre -o noatime,flock ${module.fsx_lustre[0].dns_name}@tcp:/${module.fsx_lustre[0].mount_name} /mnt/fsx" : null
}

# S3 model bucket (existing, referenced)
output "models_bucket_name" {
  description = "S3 bucket for model storage (existing, from qwen3-next)"
  value       = data.aws_s3_bucket.models.id
}

# Serving engine outputs
output "serving_endpoint" {
  description = "Serving engine service endpoint URL"
  value       = var.enable_vllm ? "http://vllm-qwen3-next.ml-inference.svc.cluster.local:8000" : null
}

output "model_path" {
  description = "Model path being served"
  value       = var.enable_vllm ? var.vllm_model_id : null
}

# Benchmark results
output "benchmark_results_bucket" {
  description = "S3 bucket for benchmark results"
  value       = aws_s3_bucket.benchmark_results.id
}

# Quick reference
output "quick_start" {
  description = "Quick start commands for Qwen3-Next on g7e"
  value       = <<-EOT
    # 1. Configure kubectl
    ${module.eks.configure_kubectl}

    # 2. Check nodes (g7e should auto-join via EKS managed node group)
    kubectl get nodes

    # 3. Verify FSx DRA sync (model should auto-import from existing S3 bucket)
    kubectl exec -it -n ml-inference deploy/vllm-qwen3-next -- ls /mnt/fsx/models/

    # 4. Copy model to NVMe for fastest serving
    ./scripts/copy-to-nvme.sh

    # 5. Check serving pod status
    kubectl get pods -n ml-inference

    # 6. Port forward for local testing
    kubectl port-forward -n ml-inference svc/vllm-qwen3-next 8000:8000

    # 7. Test inference
    curl http://localhost:8000/v1/models

    # 8. Run benchmarks
    ./scripts/run-benchmarks.sh g0
  EOT
}
