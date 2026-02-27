# Qwen3-Next Customer Benchmark Outputs

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
  description = "FSx Lustre DNS name for mounting"
  value       = var.enable_fsx_lustre ? module.fsx_lustre[0].dns_name : null
}

output "fsx_mount_name" {
  description = "FSx Lustre mount name"
  value       = var.enable_fsx_lustre ? module.fsx_lustre[0].mount_name : null
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
  description = "Serving engine service endpoint URL"
  value       = var.enable_vllm ? "http://${local.app_label}.ml-inference.svc.cluster.local:8000" : null
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

output "capacity_block_launch_command" {
  description = "AWS CLI command to launch a p5en.48xlarge via capacity block"
  value       = <<-EOT
    aws ec2 run-instances \
      --instance-type p5en.48xlarge \
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

output "quick_start" {
  description = "Quick start commands for customer benchmark"
  value       = <<-EOT
    # 1. Build customer vLLM image
    docker build -f docker/Dockerfile.vllm-customer -t qwen3-next-custbench:latest docker/

    # 2. Stage model weights to S3 (auto-syncs to FSx)
    aws s3 sync /path/to/qwen3-next/ s3://${aws_s3_bucket.models.id}/qwen3-next/ --region ${var.aws_region}

    # 3. Configure kubectl
    ${module.eks.configure_kubectl}

    # 4. After GPU node joins, copy model to NVMe
    # ssh to GPU node, then: cp -r /mnt/fsx/models/qwen3-next /mnt/nvme/models/qwen3-next-fp8

    # 5. Run benchmarks
    # T1 (customer config): ./scripts/run-benchmarks.sh t1
    # T2 (optimized):       ./scripts/run-benchmarks.sh t2
    # T3 (no MTP):          ./scripts/run-benchmarks.sh t3
    # T4 (load scaling):    ./scripts/run-benchmarks.sh t4
    # All:                  ./scripts/run-benchmarks.sh all
  EOT
}
