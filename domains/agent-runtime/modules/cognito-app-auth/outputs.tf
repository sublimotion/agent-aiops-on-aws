output "user_pool_id" {
  description = "Cognito user pool ID"
  value       = aws_cognito_user_pool.this.id
}

output "user_pool_arn" {
  description = "Cognito user pool ARN"
  value       = aws_cognito_user_pool.this.arn
}

output "app_client_id" {
  description = "Cognito app client ID"
  value       = aws_cognito_user_pool_client.this.id
}
