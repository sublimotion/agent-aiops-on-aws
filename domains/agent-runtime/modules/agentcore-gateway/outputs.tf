output "mcp_endpoint_url" {
  description = "AgentCore Gateway MCP endpoint URL — paste into Claude Desktop mcp.json"
  value       = local.mcp_endpoint_url
}

output "gateway_id" {
  description = "AgentCore Gateway ID"
  value       = local.gateway_id
}
