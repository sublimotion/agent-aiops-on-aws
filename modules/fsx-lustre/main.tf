# FSx Lustre Module - High-performance filesystem for KV cache offloading

data "aws_caller_identity" "current" {}

# Security group for FSx Lustre
resource "aws_security_group" "fsx" {
  name_prefix = "${var.project_name}-fsx-"
  description = "Security group for FSx Lustre filesystem"
  vpc_id      = var.vpc_id

  # Lustre client traffic (port 988)
  ingress {
    description     = "Lustre client traffic from EKS nodes"
    from_port       = 988
    to_port         = 988
    protocol        = "tcp"
    security_groups = var.allowed_security_group_ids
  }

  # Lustre client traffic (port range 1021-1023)
  ingress {
    description     = "Lustre client traffic from EKS nodes"
    from_port       = 1021
    to_port         = 1023
    protocol        = "tcp"
    security_groups = var.allowed_security_group_ids
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
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
  storage_capacity            = var.storage_capacity
  subnet_ids                  = [var.subnet_id]
  security_group_ids          = [aws_security_group.fsx.id]
  deployment_type             = var.deployment_type
  per_unit_storage_throughput = var.deployment_type == "PERSISTENT_1" || var.deployment_type == "PERSISTENT_2" ? var.per_unit_storage_throughput : null

  log_configuration {
    level       = var.log_level
    destination = var.log_destination
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-fsx"
  })
}
