output "file_system_id" {
  description = "FSx Lustre filesystem ID"
  value       = aws_fsx_lustre_file_system.main.id
}

output "dns_name" {
  description = "DNS name for the FSx filesystem"
  value       = aws_fsx_lustre_file_system.main.dns_name
}

output "mount_name" {
  description = "Mount name for the FSx filesystem"
  value       = aws_fsx_lustre_file_system.main.mount_name
}

output "security_group_id" {
  description = "Security group ID for FSx"
  value       = aws_security_group.fsx.id
}

output "arn" {
  description = "ARN of the FSx filesystem"
  value       = aws_fsx_lustre_file_system.main.arn
}

output "iam_policy_arn" {
  description = "ARN of IAM policy for FSx access (use with IRSA)"
  value       = aws_iam_policy.fsx_access.arn
}
