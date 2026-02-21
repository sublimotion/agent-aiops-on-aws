output "service_endpoint" {
  description = "WebSocket proxy service endpoint (wss://...)"
  value       = "" # populated after first RALPH loop
}

output "cluster_name" {
  description = "ECS cluster name"
  value       = "" # populated after first RALPH loop
}

output "service_name" {
  description = "ECS service name"
  value       = "" # populated after first RALPH loop
}
