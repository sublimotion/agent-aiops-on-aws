output "agent_id" {
  description = "Bedrock AgentCore Runtime agent ID"
  value       = aws_bedrockagent_agent.this.agent_id
}

output "agent_arn" {
  description = "Bedrock AgentCore Runtime agent ARN"
  value       = aws_bedrockagent_agent.this.agent_arn
}

output "agent_alias_id" {
  description = "Agent alias ID"
  value       = aws_bedrockagent_agent_alias.this.agent_alias_id
}

output "agent_alias_arn" {
  description = "Agent alias ARN"
  value       = aws_bedrockagent_agent_alias.this.agent_alias_arn
}

output "execution_role_arn" {
  description = "IAM role ARN used by the agent"
  value       = aws_iam_role.agent.arn
}
