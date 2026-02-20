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

output "sagemaker_domain_url" {
  description = "SageMaker Studio URL"
  value       = var.enable_sagemaker ? module.sagemaker[0].domain_url : null
}

output "vllm_endpoint" {
  description = "vLLM service endpoint"
  value       = var.enable_vllm ? module.vllm[0].service_endpoint : null
}

output "vllm_load_balancer" {
  description = "vLLM load balancer hostname"
  value       = var.enable_vllm ? module.vllm[0].load_balancer_hostname : null
}

output "vllm_model" {
  description = "Model being served"
  value       = var.enable_vllm ? module.vllm[0].model_id : null
}

# FSx Lustre outputs (when enabled)
output "fsx_file_system_id" {
  description = "FSx Lustre filesystem ID"
  value       = var.enable_fsx_lustre ? module.fsx_lustre[0].file_system_id : null
}

output "fsx_dns_name" {
  description = "FSx Lustre DNS name"
  value       = var.enable_fsx_lustre ? module.fsx_lustre[0].dns_name : null
}

output "fsx_mount_command" {
  description = "FSx Lustre mount command"
  value       = var.enable_fsx_lustre ? module.fsx_lustre[0].mount_command : null
}

# KV Cache config
output "kv_cache_config" {
  description = "Active KV cache configuration"
  value       = var.kv_cache_config
}
