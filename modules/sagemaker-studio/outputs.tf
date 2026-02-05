output "domain_id" {
  description = "SageMaker Studio domain ID"
  value       = aws_sagemaker_domain.studio.id
}

output "domain_url" {
  description = "SageMaker Studio domain URL"
  value       = aws_sagemaker_domain.studio.url
}

output "domain_arn" {
  description = "SageMaker Studio domain ARN"
  value       = aws_sagemaker_domain.studio.arn
}

output "execution_role_arn" {
  description = "SageMaker execution role ARN"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "security_group_id" {
  description = "SageMaker Studio security group ID"
  value       = aws_security_group.sagemaker_studio.id
}

output "user_profile_name" {
  description = "SageMaker user profile name"
  value       = aws_sagemaker_user_profile.default.user_profile_name
}

output "kms_key_arn" {
  description = "KMS key ARN for SageMaker encryption"
  value       = aws_kms_key.sagemaker.arn
}
