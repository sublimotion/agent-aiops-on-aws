output "service_endpoint" {
  description = "ALB DNS name for the agent service (http://...)"
  value       = "http://${aws_lb.this.dns_name}"
}

output "alb_dns_name" {
  description = "Raw ALB DNS name"
  value       = aws_lb.this.dns_name
}

output "cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.this.name
}

output "service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.agent.name
}

output "task_role_arn" {
  description = "ECS task IAM role ARN"
  value       = aws_iam_role.task.arn
}
