output "ecr_repository_url" {
  description = "Push the runtime image here (tag = profile version, e.g. :v1)"
  value       = aws_ecr_repository.runtime.repository_url
}

output "artifact_bucket" {
  description = "AGENT_RUNNER_ARTIFACT_BUCKET"
  value       = aws_s3_bucket.artifacts.id
}

output "state_table" {
  description = "AGENT_RUNNER_STATE_TABLE"
  value       = aws_dynamodb_table.runs.name
}

output "run_role_arn" {
  description = "Annotate per-run ServiceAccounts with this role (eks.amazonaws.com/role-arn)"
  value       = aws_iam_role.run.arn
}

output "registry" {
  description = "AGENT_RUNNER_REGISTRY (account ECR endpoint)"
  value       = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${data.aws_region.current.region}.amazonaws.com"
}

output "cli_env" {
  description = "Export block for the agent-runner CLI"
  value       = <<-EOT
    export AGENT_RUNNER_REGISTRY=${data.aws_caller_identity.current.account_id}.dkr.ecr.${data.aws_region.current.region}.amazonaws.com
    export AGENT_RUNNER_ARTIFACT_BUCKET=${aws_s3_bucket.artifacts.id}
    export AGENT_RUNNER_STATE_TABLE=${aws_dynamodb_table.runs.name}
    export AGENT_RUNNER_REGION=${data.aws_region.current.region}
  EOT
}
