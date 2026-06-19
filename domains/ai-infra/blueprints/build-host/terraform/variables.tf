variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Project tag prefix"
  type        = string
  default     = "ai-infra"
}

variable "eks_cluster_name" {
  description = "Existing EKS cluster whose VPC and subnets we reuse"
  type        = string
  default     = "qn-sglang-eks-cluster"
}

variable "instance_type" {
  description = "Build host instance type. CPU-bound; no GPU needed."
  type        = string
  default     = "c7i.4xlarge"
}

variable "disk_size_gb" {
  description = "Root volume size. Each slim variant ~25 GB uncompressed plus Docker layer cache."
  type        = number
  default     = 200
}

variable "ssh_key_name" {
  description = "Name of an existing EC2 key pair for SSH"
  type        = string
}

variable "ssh_ingress_cidr" {
  description = "CIDR allowed to SSH (your office/home IP /32 recommended)"
  type        = string
}

variable "repo_url" {
  description = "Git URL of the agent-aiops-on-aws repo, used by bootstrap to clone"
  type        = string
  default     = "https://github.com/anthropics/agent-aiops-on-aws.git"
}

variable "repo_ref" {
  description = "Git ref to check out (branch, tag, or sha)"
  type        = string
  default     = "main"
}
