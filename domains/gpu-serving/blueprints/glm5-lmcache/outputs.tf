# GLM-5 LMCache Blueprint Outputs

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
  description = "SGLang service endpoint URL"
  value       = var.enable_serving ? "http://sglang-glm5.ml-inference.svc.cluster.local:30000" : null
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
  description = "AWS CLI command to launch p5e.48xlarge via capacity block"
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

output "quick_start" {
  description = "Quick start commands for GLM-5 LMCache"
  value       = <<-EOT
    # 1. Stage model weights to S3
    ./scripts/stage-model.sh ${aws_s3_bucket.models.id}

    # 2. Configure kubectl
    ${module.eks.configure_kubectl}

    # 3. After GPU node joins, copy model to NVMe
    ./scripts/copy-to-nvme.sh

    # 4. Check serving pod status
    kubectl get pods -n ml-inference

    # 5. Port forward for testing
    kubectl port-forward -n ml-inference svc/sglang-glm5 30000:30000

    # 6. Test inference
    curl http://localhost:30000/v1/models

    # 7. Swap LMCache config (e.g., to GDS)
    kubectl create configmap lmcache-config -n ml-inference --from-file=config.yaml=manifests/lmcache-config-gds.yaml -o yaml --dry-run=client | kubectl apply -f -
    kubectl rollout restart deployment/sglang-glm5 -n ml-inference
  EOT
}
