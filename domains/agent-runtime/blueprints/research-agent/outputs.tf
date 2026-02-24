output "mcp_endpoint_url" {
  description = "AgentCore Gateway MCP endpoint — paste into Claude Desktop mcp.json"
  value       = module.gateway.mcp_endpoint_url
  sensitive   = true
}

output "service_endpoint" {
  description = "Internal ALB endpoint for the research agent (http://...)"
  value       = module.websocket_proxy.service_endpoint
}

output "ecr_repository_url" {
  description = "ECR repository URL for docker push"
  value       = aws_ecr_repository.agent.repository_url
}

output "s3_output_bucket" {
  description = "S3 bucket name for research output PDFs"
  value       = aws_s3_bucket.output.bucket
}

output "cognito_user_pool_id" {
  description = "Cognito user pool ID"
  value       = module.auth.user_pool_id
}

output "cognito_app_client_id" {
  description = "Cognito app client ID"
  value       = module.auth.app_client_id
}

output "bedrock_agent_id" {
  description = "Bedrock agent ID"
  value       = module.runtime.agent_id
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.websocket_proxy.cluster_name
}

output "claude_desktop_mcp_config" {
  description = "Snippet to paste into Claude Desktop ~/.claude/mcp.json"
  value = jsonencode({
    mcpServers = {
      "research-agent" = {
        url = module.gateway.mcp_endpoint_url
        headers = {
          Authorization = "Bearer <cognito_id_token>"
        }
      }
    }
  })
  sensitive = true
}