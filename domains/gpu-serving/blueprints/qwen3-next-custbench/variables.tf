variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "benchmark"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "qwen3-custbench"
}

# Networking
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.1.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-2a", "us-east-2b", "us-east-2c"]
}

variable "gpu_availability_zones" {
  description = "Availability zones for GPU nodes (p5en capacity block availability)"
  type        = list(string)
  default     = ["us-east-2c"]
}

variable "capacity_reservation_id" {
  description = "Capacity reservation ID for p5en.48xlarge capacity block"
  type        = string
  default     = ""
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for public registry access"
  type        = bool
  default     = true
}

# EKS
variable "eks_cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.32"
}

variable "gpu_instance_types" {
  description = "GPU instance type - p5en.48xlarge with 8x H200"
  type        = list(string)
  default     = ["p5en.48xlarge"]
}

variable "gpu_desired_size" {
  description = "Desired GPU nodes (0 = launch via capacity block CLI)"
  type        = number
  default     = 0
}

variable "gpu_min_size" {
  type    = number
  default = 0
}

variable "gpu_max_size" {
  type    = number
  default = 1
}

variable "gpu_volume_size" {
  description = "Root volume size for GPU nodes (GB)"
  type        = number
  default     = 500
}

variable "enable_efa" {
  description = "Enable Elastic Fabric Adapter for NVLink TP communication"
  type        = bool
  default     = true
}

# FSx Lustre
variable "enable_fsx_lustre" {
  description = "Enable FSx Lustre for model storage"
  type        = bool
  default     = true
}

variable "fsx_storage_capacity" {
  description = "FSx Lustre storage capacity in GiB"
  type        = number
  default     = 4800
}

variable "fsx_deployment_type" {
  description = "FSx Lustre deployment type"
  type        = string
  default     = "PERSISTENT_2"
}

variable "fsx_per_unit_throughput" {
  description = "FSx throughput per TiB in MB/s"
  type        = number
  default     = 1000
}

# Serving Engine
variable "enable_vllm" {
  description = "Enable serving engine deployment"
  type        = bool
  default     = true
}

variable "vllm_model_id" {
  description = "Local model path"
  type        = string
  default     = "/mnt/nvme/models/qwen3-next-fp8"
}

variable "vllm_image" {
  description = "vLLM container image (customer's nightly build)"
  type        = string
  default     = "qwen3-next-custbench:latest"
}

variable "nvidia_device_plugin_image" {
  description = "NVIDIA device plugin image"
  type        = string
  default     = "nvcr.io/nvidia/k8s-device-plugin:v0.14.5"
}

variable "vllm_max_model_len" {
  description = "Maximum model context length"
  type        = number
  default     = 32768
}

variable "vllm_gpu_count" {
  description = "Number of GPUs (4 for TP=4)"
  type        = number
  default     = 4
}

variable "vllm_cpu_request" {
  description = "CPU request for serving pod"
  type        = string
  default     = "32"
}

# Monitoring
variable "enable_monitoring" {
  description = "Enable Prometheus monitoring stack"
  type        = bool
  default     = true
}

variable "prometheus_scrape_interval" {
  description = "Prometheus scrape interval"
  type        = string
  default     = "1s"
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  default     = "admin"
  sensitive   = true
}

variable "ecr_registry" {
  description = "Private ECR registry URI"
  type        = string
  default     = ""
}

variable "vllm_node_port" {
  description = "NodePort for serving engine service"
  type        = number
  default     = 30080
}
