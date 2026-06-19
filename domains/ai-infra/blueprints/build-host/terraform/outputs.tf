output "instance_id" {
  value = aws_instance.build_host.id
}

output "public_ip" {
  value = aws_instance.build_host.public_ip
}

output "ssh_command" {
  value = "ssh -i ~/.ssh/${var.ssh_key_name}.pem ubuntu@${aws_instance.build_host.public_ip}"
}

output "ssm_command" {
  value = "aws ssm start-session --target ${aws_instance.build_host.id} --region ${var.region}"
}

output "vllm_ecr_repo" {
  value = data.aws_ecr_repository.vllm_slim.repository_url
}

output "sglang_ecr_repo" {
  value = data.aws_ecr_repository.sglang_slim.repository_url
}

output "registry_uri" {
  description = "Use as REGISTRY env var when invoking build.sh on the host"
  value       = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com/ai-infra"
}
