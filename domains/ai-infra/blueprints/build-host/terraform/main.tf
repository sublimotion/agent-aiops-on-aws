###############################################################################
# Slim image build host
#
# Spot c7i.4xlarge in an existing EKS cluster's VPC. Pulls the repo, runs
# domains/ai-infra/shared/images/build.sh, pushes to private ECR.
#
# Why this exists: builds eat tens of GB of disk and 30-60 min CPU. Local
# laptops shouldn't pay that cost. Tear down (`terraform destroy`) when the
# build queue empties.
###############################################################################

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

###############################################################################
# Reuse existing VPC via the EKS cluster lookup pattern.
###############################################################################

data "aws_eks_cluster" "this" {
  name = var.eks_cluster_name
}

# Public subnets so we can SSH directly without a bastion. The instance has
# no production traffic; this is a developer build box.
data "aws_subnets" "public" {
  filter {
    name   = "vpc-id"
    values = [data.aws_eks_cluster.this.vpc_config[0].vpc_id]
  }
  filter {
    name   = "tag:Name"
    values = ["*public*"]
  }
}

data "aws_caller_identity" "current" {}

###############################################################################
# ECR repos are managed by staging/terraform/main.tf — looked up here.
###############################################################################

data "aws_ecr_repository" "vllm_slim" {
  name = "ai-infra/vllm-slim"
}

data "aws_ecr_repository" "sglang_slim" {
  name = "ai-infra/sglang-slim"
}

###############################################################################
# IAM: instance role with ECR push, SSM (so we can session-manager in if SSH
# is firewalled), and CloudWatch Logs.
###############################################################################

resource "aws_iam_role" "build_host" {
  name = "${var.project_name}-build-host"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = { Project = var.project_name }
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.build_host.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "ecr_push" {
  name = "ecr-push"
  role = aws_iam_role.build_host.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage",
      ]
      Resource = "*"
    }]
  })
}

resource "aws_iam_instance_profile" "build_host" {
  name = "${var.project_name}-build-host"
  role = aws_iam_role.build_host.name
}

###############################################################################
# Security group: SSH from a single CIDR (your office/home), egress all.
###############################################################################

resource "aws_security_group" "build_host" {
  name_prefix = "${var.project_name}-build-host-"
  vpc_id      = data.aws_eks_cluster.this.vpc_config[0].vpc_id
  description = "Slim image build host"

  ingress {
    description = "SSH from operator"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_ingress_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-build-host" }
}

###############################################################################
# AMI lookup (Ubuntu 22.04 amd64).
###############################################################################

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

###############################################################################
# Spot c7i.4xlarge instance.
###############################################################################

resource "aws_instance" "build_host" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnets.public.ids[0]
  vpc_security_group_ids      = [aws_security_group.build_host.id]
  iam_instance_profile        = aws_iam_instance_profile.build_host.name
  key_name                    = var.ssh_key_name
  associate_public_ip_address = true

  # On-demand: build is short (~30-60 min) and AZ-a c7i spot has been bouncing.
  # Cost is ~$0.72/hr; total build session ~$1.50. Negligible vs spot churn cost.

  root_block_device {
    volume_size           = var.disk_size_gb
    volume_type           = "gp3"
    iops                  = 3000
    throughput            = 250
    delete_on_termination = true
    encrypted             = true
  }

  # Prepend terraform-supplied env, then run the script verbatim.
  # Avoids templatefile() collisions with the script's bash ${VAR:-default} patterns.
  user_data = <<-EOT
    #!/usr/bin/env bash
    export TF_REGION="${var.region}"
    export TF_REPO_URL="${var.repo_url}"
    export TF_REPO_REF="${var.repo_ref}"
    export TF_AWS_ACCOUNT_ID="${data.aws_caller_identity.current.account_id}"
    ${file("${path.module}/../scripts/bootstrap.sh")}
  EOT

  tags = {
    Name    = "${var.project_name}-build-host"
    Project = var.project_name
    Role    = "image-build"
  }
}
