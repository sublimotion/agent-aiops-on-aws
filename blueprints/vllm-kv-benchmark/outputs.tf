# Outputs for vLLM KV Cache Benchmark Blueprint

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_version" {
  description = "EKS cluster version"
  value       = var.eks_cluster_version
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.networking.vpc_id
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.aws_region}"
}

# FSx Lustre outputs
output "fsx_file_system_id" {
  description = "FSx Lustre file system ID"
  value       = var.enable_fsx_lustre ? aws_fsx_lustre_file_system.kv_cache[0].id : null
}

output "fsx_dns_name" {
  description = "FSx Lustre DNS name"
  value       = var.enable_fsx_lustre ? aws_fsx_lustre_file_system.kv_cache[0].dns_name : null
}

output "fsx_mount_name" {
  description = "FSx Lustre mount name"
  value       = var.enable_fsx_lustre ? aws_fsx_lustre_file_system.kv_cache[0].mount_name : null
}

# vLLM outputs
output "vllm_endpoint_internal" {
  description = "Internal vLLM endpoint (ClusterIP)"
  value       = var.enable_vllm ? "http://vllm-benchmark.ml-inference.svc.cluster.local:8000" : null
}

output "vllm_node_port" {
  description = "vLLM NodePort for local access"
  value       = var.enable_vllm ? var.vllm_node_port : null
}

output "port_forward_command" {
  description = "Command to port-forward vLLM service"
  value       = var.enable_vllm ? "kubectl port-forward -n ml-inference svc/vllm-benchmark 30080:8000" : null
}

# Monitoring outputs
output "prometheus_node_port" {
  description = "Prometheus NodePort"
  value       = var.enable_monitoring ? 30090 : null
}

output "prometheus_port_forward" {
  description = "Command to port-forward Prometheus"
  value       = var.enable_monitoring ? "kubectl port-forward -n monitoring svc/prometheus-server 9090:80" : null
}

# Benchmark configuration summary
output "benchmark_config" {
  description = "Summary of benchmark configuration"
  value = {
    model            = var.vllm_model_id
    gpu_type         = "g7e.xlarge (L40S 48GB)"
    max_model_len    = var.vllm_max_model_len
    prefix_caching   = var.vllm_enable_prefix_caching
    cpu_offload_gb   = var.vllm_cpu_offload_gb
    swap_space_gb    = var.vllm_swap_space_gb
    fsx_enabled      = var.enable_fsx_lustre
    fsx_capacity_gib = var.enable_fsx_lustre ? var.fsx_storage_capacity : 0
    monitoring       = var.enable_monitoring
    scrape_interval  = var.prometheus_scrape_interval
    endpoint         = "localhost:${var.vllm_node_port}"
  }
}
