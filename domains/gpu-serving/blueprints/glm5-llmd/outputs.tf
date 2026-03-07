# GLM-5 llm-d Blueprint Outputs

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

output "fsx_file_system_id" {
  description = "FSx Lustre filesystem ID"
  value       = var.enable_fsx_lustre ? module.fsx_lustre[0].file_system_id : null
}

output "fsx_dns_name" {
  description = "FSx Lustre DNS name"
  value       = var.enable_fsx_lustre ? module.fsx_lustre[0].dns_name : null
}

output "models_bucket_name" {
  description = "S3 bucket for model storage"
  value       = aws_s3_bucket.models.id
}

output "benchmark_results_bucket" {
  description = "S3 bucket for benchmark results"
  value       = aws_s3_bucket.benchmark_results.id
}

output "serving_endpoint" {
  description = "vLLM headless service endpoint"
  value       = var.enable_serving ? "http://vllm-glm5.ml-inference.svc.cluster.local:8000" : null
}

output "gpu_node_instance_profile_arn" {
  description = "GPU node IAM instance profile ARN"
  value       = aws_iam_instance_profile.gpu_node.arn
}

output "gpu_user_data_base64" {
  description = "Base64-encoded user data for capacity block GPU instances"
  value       = local.gpu_user_data
  sensitive   = true
}

output "quick_start" {
  description = "Quick start commands for GLM-5 llm-d"
  value       = <<-EOT
    # 1. Stage model weights to S3
    ./scripts/stage-model.sh ${aws_s3_bucket.models.id}

    # 2. Configure kubectl
    ${module.eks.configure_kubectl}

    # 3. Install Gateway API CRDs + llm-d EPP
    ./scripts/install-gateway.sh

    # 4. Apply manifests
    kubectl apply -f manifests/redis.yaml
    kubectl apply -f manifests/lmcache-config-gds.yaml
    kubectl apply -f manifests/glm5-deployment.yaml
    kubectl apply -f manifests/glm5-service.yaml
    kubectl apply -f manifests/glm5-inferencepool.yaml
    kubectl apply -f manifests/glm5-httproute.yaml

    # 5. After GPU nodes join, copy model to NVMe on each
    ./scripts/copy-to-nvme.sh

    # 6. Validate routing
    ./scripts/validate-routing.sh

    # 7. Test via gateway
    curl http://<gateway-ip>/v1/models
  EOT
}
