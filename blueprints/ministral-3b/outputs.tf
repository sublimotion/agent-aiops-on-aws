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
