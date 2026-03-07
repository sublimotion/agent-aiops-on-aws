# ---------------------------------------------------------------------------
# Temporary Build Instance for Container Image
# ---------------------------------------------------------------------------

data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-arm64"]
  }

  filter {
    name   = "architecture"
    values = ["arm64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_iam_role" "build_instance" {
  name = "${var.name}-build-instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "build_instance_ssm" {
  role       = aws_iam_role.build_instance.id
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "build_instance_ecr" {
  role = aws_iam_role.build_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "build_instance" {
  name = "${var.name}-build-instance"
  role = aws_iam_role.build_instance.name
}

resource "aws_security_group" "build_instance" {
  name        = "${var.name}-build-instance"
  description = "Security group for build instance"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.name}-build-instance" }
}

resource "aws_instance" "build" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = "t4g.medium"
  subnet_id              = aws_subnet.private[0].id
  iam_instance_profile   = aws_iam_instance_profile.build_instance.name
  vpc_security_group_ids = [aws_security_group.build_instance.id]

  user_data = <<-EOF
    #!/bin/bash
    # Install Docker and tools
    yum update -y
    yum install -y docker git
    systemctl start docker
    systemctl enable docker
    usermod -a -G docker ec2-user

    # Install docker buildx for multi-platform builds
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -L "https://github.com/docker/buildx/releases/latest/download/buildx-linux-arm64" -o /usr/local/lib/docker/cli-plugins/docker-buildx
    chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx

    # Pre-authenticate to ECR
    aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com
  EOF

  tags = {
    Name = "${var.name}-build-instance"
  }
}

output "build_instance_id" {
  value       = aws_instance.build.id
  description = "Instance ID for SSM session: aws ssm start-session --target <id>"
}