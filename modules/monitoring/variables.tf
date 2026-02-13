variable "namespace" {
  description = "Kubernetes namespace for monitoring stack"
  type        = string
  default     = "monitoring"
}

variable "prometheus_version" {
  description = "Prometheus Helm chart version"
  type        = string
  default     = "25.8.0"
}

variable "grafana_version" {
  description = "Grafana Helm chart version"
  type        = string
  default     = "7.0.19"
}

variable "scrape_interval" {
  description = "Prometheus scrape interval"
  type        = string
  default     = "1s"
}

variable "evaluation_interval" {
  description = "Prometheus rule evaluation interval"
  type        = string
  default     = "1s"
}

variable "enable_persistence" {
  description = "Enable persistent storage for Prometheus and Grafana"
  type        = bool
  default     = true
}

variable "prometheus_storage_size" {
  description = "Prometheus storage size"
  type        = string
  default     = "20Gi"
}

variable "grafana_storage_size" {
  description = "Grafana storage size"
  type        = string
  default     = "5Gi"
}

variable "storage_class" {
  description = "Storage class for persistent volumes"
  type        = string
  default     = "gp3"
}

variable "enable_alertmanager" {
  description = "Enable Alertmanager"
  type        = bool
  default     = false
}

variable "vllm_targets" {
  description = "List of vLLM service endpoints for Prometheus to scrape"
  type        = list(string)
  default     = ["vllm.ml-inference.svc.cluster.local:8000"]
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  default     = "admin"
  sensitive   = true
}

variable "grafana_service_type" {
  description = "Grafana service type (ClusterIP, NodePort, LoadBalancer)"
  type        = string
  default     = "NodePort"
}

variable "grafana_node_port" {
  description = "NodePort for Grafana service"
  type        = number
  default     = 30300
}
