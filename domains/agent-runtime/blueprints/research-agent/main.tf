terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.50"
    }
    null = {
      source  = "hashicorp/null"
      version = ">= 3.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  default_tags { tags = var.tags }
}

data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# Networking — private VPC with 2 AZs, all egress via VPC endpoints
# ---------------------------------------------------------------------------

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = "${var.name}-vpc" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 1)
  availability_zone = var.availability_zones[count.index]
  tags              = { Name = "${var.name}-private-${count.index + 1}" }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.name}-rt-private" }
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# Security group shared by all Interface VPC endpoints
resource "aws_security_group" "endpoints" {
  name        = "${var.name}-vpc-endpoints"
  description = "Allow HTTPS from within the VPC"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.name}-vpc-endpoints" }
}

locals {
  interface_endpoints = [
    "bedrock-runtime",
    "bedrock-agent-runtime",
    "ecr.api",
    "ecr.dkr",
    "secretsmanager",
    "elasticfilesystem",
    "logs", # required: Fargate validates awslogs log group at task start
  ]
}

resource "aws_vpc_endpoint" "interface" {
  for_each            = toset(local.interface_endpoints)
  vpc_id              = aws_vpc.main.id
  service_name        = "com.amazonaws.${var.aws_region}.${each.key}"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.endpoints.id]
  private_dns_enabled = true
  tags                = { Name = "${var.name}-vpce-${replace(each.key, ".", "-")}" }
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]
  tags              = { Name = "${var.name}-vpce-s3" }
}

# ---------------------------------------------------------------------------
# ECR — container image repository
# ---------------------------------------------------------------------------

resource "aws_ecr_repository" "agent" {
  name                 = var.name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "agent" {
  repository = aws_ecr_repository.agent.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# EFS — inter-agent file coordination (/app/files)
# ---------------------------------------------------------------------------

resource "aws_efs_file_system" "files" {
  encrypted        = true
  performance_mode = "generalPurpose"
  throughput_mode  = "elastic"
  tags             = { Name = "${var.name}-files" }
}

resource "aws_security_group" "efs" {
  name        = "${var.name}-efs"
  description = "Allow NFS from ECS tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.name}-efs" }
}

resource "aws_efs_mount_target" "files" {
  count           = 2
  file_system_id  = aws_efs_file_system.files.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}

# ---------------------------------------------------------------------------
# S3 — research output bucket
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "output" {
  bucket_prefix = "${var.name}-output-"
}

resource "aws_s3_bucket_versioning" "output" {
  bucket = aws_s3_bucket.output.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_lifecycle_configuration" "output" {
  bucket = aws_s3_bucket.output.id
  rule {
    id     = "expire-old-reports"
    status = "Enabled"
    expiration { days = 30 }
  }
}

resource "aws_s3_bucket_public_access_block" "output" {
  bucket                  = aws_s3_bucket.output.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# Secrets Manager — search API keys
# ---------------------------------------------------------------------------

resource "aws_secretsmanager_secret" "tavily_api_key" {
  name        = "${var.name}/tavily-api-key"
  description = "Tavily Search API key (primary web search backend)"
}

resource "aws_secretsmanager_secret_version" "tavily_api_key" {
  secret_id     = aws_secretsmanager_secret.tavily_api_key.id
  secret_string = var.tavily_api_key
}

resource "aws_secretsmanager_secret" "brave_api_key" {
  name        = "${var.name}/brave-api-key"
  description = "Brave Search API key (fallback web search backend)"
}

resource "aws_secretsmanager_secret_version" "brave_api_key" {
  secret_id     = aws_secretsmanager_secret.brave_api_key.id
  secret_string = var.brave_api_key
}

# ---------------------------------------------------------------------------
# Module: AgentCore Runtime (Bedrock agent + IAM)
# ---------------------------------------------------------------------------

module "runtime" {
  source = "../../modules/agentcore-runtime"

  name                     = var.name
  foundation_model_id      = var.foundation_model_id
  idle_session_ttl_seconds = 1800
  efs_file_system_arn      = aws_efs_file_system.files.arn
  s3_output_bucket_arn     = aws_s3_bucket.output.arn
  search_secrets_arns      = [aws_secretsmanager_secret.tavily_api_key.arn, aws_secretsmanager_secret.brave_api_key.arn]
  tags                     = var.tags
}

# ---------------------------------------------------------------------------
# Module: Agent Memory (DynamoDB session table)
# ---------------------------------------------------------------------------

module "memory" {
  source = "../../modules/agent-memory"

  name         = var.name
  billing_mode = "PAY_PER_REQUEST"
  tags         = var.tags
}

# ---------------------------------------------------------------------------
# Module: Cognito Auth
# ---------------------------------------------------------------------------

module "auth" {
  source = "../../modules/cognito-app-auth"

  name                        = var.name
  access_token_validity_hours = 1
  id_token_validity_hours     = 1
  tags                        = var.tags
}

# ---------------------------------------------------------------------------
# Module: WebSocket Proxy / ECS Service
# ---------------------------------------------------------------------------

module "websocket_proxy" {
  source = "../../modules/websocket-proxy"

  name                  = var.name
  vpc_id                = aws_vpc.main.id
  subnet_ids            = aws_subnet.private[*].id
  image_uri             = "${aws_ecr_repository.agent.repository_url}:${var.container_image_tag}"
  agent_id              = module.runtime.agent_id
  agent_alias_id        = module.runtime.agent_alias_id
  cognito_user_pool_id  = module.auth.user_pool_id
  cognito_app_client_id = module.auth.app_client_id
  efs_file_system_id        = aws_efs_file_system.files.id
  s3_output_bucket_arn      = aws_s3_bucket.output.arn
  s3_output_bucket_name     = aws_s3_bucket.output.bucket
  search_api_secret_arn     = aws_secretsmanager_secret.brave_api_key.arn
  tavily_api_key_secret_arn = aws_secretsmanager_secret.tavily_api_key.arn
  session_table_arn     = module.memory.table_arn
  session_table_name    = module.memory.table_name
  cpu                   = var.ecs_cpu
  memory                = var.ecs_memory
  tags                  = var.tags

  depends_on = [aws_efs_mount_target.files]
}

# ---------------------------------------------------------------------------
# Module: AgentCore Gateway (MCP endpoint)
# ---------------------------------------------------------------------------

module "gateway" {
  source = "../../modules/agentcore-gateway"

  name                  = var.name
  agent_id              = module.runtime.agent_id
  agent_alias_id        = module.runtime.agent_alias_id
  tool_name             = "research"
  tool_description      = "Run a multi-agent research pipeline and return a PDF report URL."
  cognito_user_pool_arn = module.auth.user_pool_arn
  region                = var.aws_region
  tags                  = var.tags
}
