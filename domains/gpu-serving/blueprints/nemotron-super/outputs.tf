# Nemotron-3-Super Blueprint Outputs

output "cluster_name" {
  description = "EKS cluster name"
  value       = var.eks_cluster_name
}

output "serving_backend" {
  description = "Active serving backend"
  value       = var.serving_backend
}

output "serving_endpoint" {
  description = "ClusterIP service endpoint"
  value       = "http://nemotron-super.${local.namespace}.svc.cluster.local:${local.serving_port}"
}

output "serving_port" {
  description = "Serving port"
  value       = local.serving_port
}

output "node_port" {
  description = "NodePort for external access"
  value       = var.node_port
}

output "num_workers" {
  description = "Number of worker replicas"
  value       = var.num_workers
}

output "tp_size" {
  description = "Tensor parallelism per worker"
  value       = var.tp_size
}

output "total_gpus" {
  description = "Total GPUs allocated"
  value       = var.num_workers * var.tp_size
}

output "benchmark_bucket" {
  description = "S3 bucket for benchmark results"
  value       = aws_s3_bucket.benchmark_results.id
}

output "quick_start" {
  description = "Quick start commands"
  value       = <<-EOT
    # Port-forward to the service
    kubectl --context ${var.eks_cluster_name} -n ${local.namespace} port-forward svc/nemotron-super ${local.serving_port}:${local.serving_port}

    # Test health
    curl http://localhost:${local.serving_port}/health

    # Chat completion
    curl http://localhost:${local.serving_port}/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "nemotron-3-super", "messages": [{"role": "user", "content": "Hello"}], "temperature": 1.0, "top_p": 0.95}'
  EOT
}
