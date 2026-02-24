output "table_name" {
  description = "DynamoDB session state table name"
  value       = aws_dynamodb_table.sessions.name
}

output "table_arn" {
  description = "DynamoDB session state table ARN"
  value       = aws_dynamodb_table.sessions.arn
}
