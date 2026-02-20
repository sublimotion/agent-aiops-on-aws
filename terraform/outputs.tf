# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of public subnet IDs"
  value       = module.vpc.public_subnets
}

# EKS Outputs
output "eks_cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "Endpoint for EKS cluster API server"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_arn" {
  description = "ARN of the EKS cluster"
  value       = module.eks.cluster_arn
}

output "eks_cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data for the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "eks_oidc_provider_arn" {
  description = "ARN of the OIDC Provider for EKS"
  value       = module.eks.oidc_provider_arn
}

# kubectl configuration command
output "configure_kubectl" {
  description = "Command to configure kubectl for the EKS cluster"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

# SageMaker Outputs
output "sagemaker_domain_id" {
  description = "ID of the SageMaker Studio domain"
  value       = var.enable_sagemaker_studio ? aws_sagemaker_domain.studio[0].id : null
}

output "sagemaker_domain_url" {
  description = "URL for SageMaker Studio domain"
  value       = var.enable_sagemaker_studio ? aws_sagemaker_domain.studio[0].url : null
}

output "sagemaker_user_profile_name" {
  description = "Name of the default SageMaker user profile"
  value       = var.enable_sagemaker_studio ? aws_sagemaker_user_profile.default[0].user_profile_name : null
}

output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

# vLLM Outputs
output "vllm_service_endpoint" {
  description = "Internal service endpoint for vLLM"
  value       = var.enable_vllm_deployment ? "http://vllm-ministral.ml-inference.svc.cluster.local:8000" : null
}

output "vllm_lb_hostname" {
  description = "Load balancer hostname for vLLM (internal NLB)"
  value       = var.enable_vllm_deployment ? kubernetes_service.vllm_lb[0].status[0].load_balancer[0].ingress[0].hostname : null
}

output "vllm_model_id" {
  description = "Model ID being served by vLLM"
  value       = var.vllm_model_id
}

output "vllm_model_source" {
  description = "Model loading source (S3 or HuggingFace Hub)"
  value       = var.vllm_use_s3_model ? "S3 (Run:ai Model Streamer)" : "HuggingFace Hub"
}

output "kv_cache_config" {
  description = "Active KV cache offloading configuration"
  value       = var.kv_cache_config
}

# FSx Lustre Outputs
output "fsx_file_system_id" {
  description = "FSx Lustre filesystem ID"
  value       = var.enable_fsx_lustre ? aws_fsx_lustre_file_system.kv_cache[0].id : null
}

output "fsx_dns_name" {
  description = "FSx Lustre DNS name for mounting"
  value       = var.enable_fsx_lustre ? aws_fsx_lustre_file_system.kv_cache[0].dns_name : null
}

output "fsx_mount_name" {
  description = "FSx Lustre mount name"
  value       = var.enable_fsx_lustre ? aws_fsx_lustre_file_system.kv_cache[0].mount_name : null
}

output "fsx_mount_command" {
  description = "FSx Lustre mount command for reference"
  value       = var.enable_fsx_lustre ? "mount -t lustre ${aws_fsx_lustre_file_system.kv_cache[0].dns_name}@tcp:/${aws_fsx_lustre_file_system.kv_cache[0].mount_name} /mnt/fsx" : null
}

# KV Cache Benchmark Instructions
output "benchmark_instructions" {
  description = "Instructions for running KV cache benchmarks"
  value = (
    var.enable_fsx_lustre
    ? <<-EOT
    KV Cache Benchmark Quick Start:

    1. Current config: ${var.kv_cache_config}
       Model: ${var.vllm_model_id}

    2. Switch KV cache configs:
       terraform apply -var="kv_cache_config=cpu-light"
       terraform apply -var="kv_cache_config=cpu-aggressive"
       terraform apply -var="kv_cache_config=fsx-swap"
       terraform apply -var="kv_cache_config=hybrid"

    3. Switch to Nemotron-3-8B for benchmarks:
       terraform apply -var="vllm_model_id=nvidia/Nemotron-3-8B-Chat-SFT" -var="kv_cache_config=fsx-swap"

    4. Monitor PCIe bandwidth during tests:
       kubectl exec -n ml-inference deploy/vllm-ministral -- nvidia-smi dmon -s pucvmet -d 1

    5. Check vLLM logs:
       kubectl logs -n ml-inference deploy/vllm-ministral -f
    EOT
    : "Enable FSx Lustre to run KV cache benchmarks: terraform apply -var=\"enable_fsx_lustre=true\""
  )
}

# S3 Model Bucket Outputs
output "model_bucket_name" {
  description = "Name of the S3 bucket for model storage"
  value       = aws_s3_bucket.models.id
}

output "model_bucket_arn" {
  description = "ARN of the S3 bucket for model storage"
  value       = aws_s3_bucket.models.arn
}

output "model_s3_path" {
  description = "Full S3 path for model storage"
  value       = var.vllm_use_s3_model ? "s3://${aws_s3_bucket.models.id}/${var.vllm_s3_model_path}" : "N/A (using HuggingFace Hub)"
}

# Connection Instructions
output "sagemaker_eks_connection_instructions" {
  description = "Instructions for connecting to EKS from SageMaker Studio"
  value       = <<-EOT
    To connect to EKS from SageMaker Studio VSCode:

    1. Open SageMaker Studio and launch Code Editor (VSCode)
    2. Open a terminal in VSCode
    3. Run: aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}
    4. Verify: kubectl get nodes
    5. Test vLLM: curl http://vllm-ministral.ml-inference.svc.cluster.local:8000/v1/models

    The SageMaker execution role has been granted EKS cluster admin access.
  EOT
}

output "model_upload_instructions" {
  description = "Instructions for uploading model to S3"
  value       = <<-EOT
    To upload a model to S3 for vLLM with Run:ai Model Streamer:

    1. Download the model from HuggingFace:
       huggingface-cli download ${var.vllm_model_id} --local-dir ./model

    2. Upload to S3:
       aws s3 cp ./model s3://${aws_s3_bucket.models.id}/${var.vllm_s3_model_path}/ --recursive

    3. Verify upload:
       aws s3 ls s3://${aws_s3_bucket.models.id}/${var.vllm_s3_model_path}/

    4. Restart vLLM deployment to load the model:
       kubectl rollout restart deployment/vllm-ministral -n ml-inference

    The model will be streamed from S3 directly to GPU memory using Run:ai Model Streamer.
  EOT
}
