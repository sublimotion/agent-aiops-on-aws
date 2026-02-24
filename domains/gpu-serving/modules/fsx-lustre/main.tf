# FSx Lustre Module - High-performance filesystem for model storage and KV cache

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Security group for FSx Lustre
resource "aws_security_group" "fsx" {
  name_prefix = "${var.project_name}-fsx-"
  description = "Security group for FSx Lustre filesystem"
  vpc_id      = var.vpc_id

  # Lustre client traffic (port 988) — from EKS nodes
  ingress {
    description     = "Lustre MGS/MGC traffic"
    from_port       = 988
    to_port         = 988
    protocol        = "tcp"
    security_groups = var.allowed_security_group_ids
  }

  # Lustre inter-router traffic (port range 1018-1023) — from EKS nodes
  ingress {
    description     = "Lustre inter-node communication"
    from_port       = 1018
    to_port         = 1023
    protocol        = "tcp"
    security_groups = var.allowed_security_group_ids
  }

  # Lustre requires self-referencing rules for internal LNET communication
  ingress {
    description = "Lustre self - MGS/MGC traffic"
    from_port   = 988
    to_port     = 988
    protocol    = "tcp"
    self        = true
  }

  ingress {
    description = "Lustre self - inter-node communication"
    from_port   = 1018
    to_port     = 1023
    protocol    = "tcp"
    self        = true
  }

  # EFA requires all internal traffic within the security group
  dynamic "ingress" {
    for_each = var.efa_enabled ? [1] : []
    content {
      description = "EFA - all internal traffic"
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      self        = true
    }
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # EFA requires all internal egress within the security group
  dynamic "egress" {
    for_each = var.efa_enabled ? [1] : []
    content {
      description = "EFA - all internal egress"
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      self        = true
    }
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-fsx-sg"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# FSx Lustre filesystem
resource "aws_fsx_lustre_file_system" "main" {
  storage_capacity            = var.storage_capacity_gib
  subnet_ids                  = [var.subnet_id]
  security_group_ids          = [aws_security_group.fsx.id]
  deployment_type             = var.deployment_type
  per_unit_storage_throughput = var.deployment_type == "PERSISTENT_1" || var.deployment_type == "PERSISTENT_2" ? var.per_unit_storage_throughput : null
  data_compression_type       = var.data_compression_type
  file_system_type_version    = var.file_system_type_version

  # EFA + GDS support (PERSISTENT_2 only, must be set at creation)
  efa_enabled = var.efa_enabled

  # Metadata configuration (required when efa_enabled = true)
  dynamic "metadata_configuration" {
    for_each = var.efa_enabled ? [1] : []
    content {
      mode = "AUTOMATIC"
    }
  }

  log_configuration {
    level       = var.log_level
    destination = var.log_destination
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-fsx"
  })
}

# IAM policy for FSx access (used by IRSA for pods that need FSx API access)
resource "aws_iam_policy" "fsx_access" {
  name_prefix = "${var.project_name}-fsx-access-"
  description = "Allow FSx Lustre filesystem access for ${var.project_name}"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "fsx:DescribeFileSystems",
          "fsx:DescribeDataRepositoryAssociations",
          "fsx:DescribeDataRepositoryTasks"
        ]
        Resource = [
          aws_fsx_lustre_file_system.main.arn,
          "arn:aws:fsx:${data.aws_region.current.id}:${data.aws_caller_identity.current.account_id}:association/${aws_fsx_lustre_file_system.main.id}/*",
          "arn:aws:fsx:${data.aws_region.current.id}:${data.aws_caller_identity.current.account_id}:task/*"
        ]
      }
    ]
  })

  tags = var.tags
}
