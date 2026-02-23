# Kimi K2.5 Blueprint Outputs

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

output "fsx_mount_command" {
  description = "Command to mount FSx on a compute node"
  value       = var.enable_fsx_lustre ? "sudo mount -t lustre -o noatime,flock ${module.fsx_lustre[0].dns_name}@tcp:/${module.fsx_lustre[0].mount_name} /mnt/fsx" : null
}

# S3 model bucket
output "model_bucket" {
  description = "S3 bucket for model storage (source of truth)"
  value       = aws_s3_bucket.models.id
}

output "model_upload_command" {
  description = "Command to upload Kimi K2.5 model to S3"
  value       = "aws s3 sync /path/to/Kimi-K2.5/ s3://${aws_s3_bucket.models.id}/models/Kimi-K2.5/ --region ${var.aws_region}"
}

# vLLM outputs
output "vllm_endpoint" {
  description = "vLLM service endpoint URL"
  value       = var.enable_vllm ? "http://vllm-kimi-k2.ml-inference.svc.cluster.local:8000" : null
}

output "vllm_model" {
  description = "Model being served"
  value       = var.enable_vllm ? var.vllm_model_id : null
}

# GPU node outputs
output "gpu_node_instance_profile_arn" {
  description = "GPU node IAM instance profile ARN"
  value       = aws_iam_instance_profile.gpu_node.arn
}

output "capacity_block_launch_command" {
  description = "AWS CLI command to launch a p5e.48xlarge via capacity block"
  value       = <<-EOT
    aws ec2 run-instances \
      --instance-type p5e.48xlarge \
      --capacity-reservation-specification 'CapacityReservationTarget={CapacityReservationId=${var.capacity_reservation_id}}' \
      --iam-instance-profile Arn=${aws_iam_instance_profile.gpu_node.arn} \
      --subnet-id ${local.gpu_subnet_ids[0]} \
      --security-group-ids ${module.eks.node_security_group_id} ${aws_security_group.gpu_nodeport.id}${var.enable_fsx_lustre ? " ${module.fsx_lustre[0].security_group_id}" : ""} \
      --user-data '${local.gpu_user_data}' \
      --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=${var.gpu_volume_size},VolumeType=gp3}' \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=${var.project_name}-gpu-node},{Key=kubernetes.io/cluster/${module.eks.cluster_name},Value=owned}]' \
      --region ${var.aws_region}
  EOT
}

# Benchmark results
output "benchmark_results_bucket" {
  description = "S3 bucket for benchmark results"
  value       = aws_s3_bucket.benchmark_results.id
}

# Quick reference
output "quick_start" {
  description = "Quick start commands for Kimi K2.5"
  value       = <<-EOT
    # 1. Upload model to S3 (source of truth)
    aws s3 sync /path/to/Kimi-K2.5/ s3://${aws_s3_bucket.models.id}/models/Kimi-K2.5/

    # 2. Configure kubectl
    ${module.eks.configure_kubectl}

    # 3. Check vLLM pod status
    kubectl get pods -n ml-inference

    # 4. Port forward for local testing
    kubectl port-forward -n ml-inference svc/vllm-kimi-k2 8000:8000

    # 5. Test inference
    curl http://localhost:8000/v1/models

    # 6. Test chat completion
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "moonshotai/Kimi-K2.5", "messages": [{"role": "user", "content": "Hello"}]}'
  EOT
}
