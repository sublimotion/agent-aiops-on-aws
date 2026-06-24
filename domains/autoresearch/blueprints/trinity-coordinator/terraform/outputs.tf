output "run_role_arn" {
  description = "IRSA run-role ARN to annotate the Job's ServiceAccount with (eks.amazonaws.com/role-arn)"
  value       = aws_iam_role.run.arn
}

output "artifact_bucket" {
  description = "Bucket holding trinity-coordinator/ artifacts"
  value       = local.bucket_name
}

output "artifact_s3_uri_base" {
  description = "Base S3 URI for run artifacts. The Job appends <run-id>/ (set TRINITY_S3_URI)."
  value       = "s3://${local.bucket_name}/${local.prefix}/"
}

output "worker_inference_profile_arns" {
  description = "Bedrock inference-profile ARNs the run-role may invoke"
  value       = local.inference_profile_arns
}
