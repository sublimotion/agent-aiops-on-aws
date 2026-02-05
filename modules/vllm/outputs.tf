output "namespace" {
  description = "Kubernetes namespace"
  value       = kubernetes_namespace.ml.metadata[0].name
}

output "deployment_name" {
  description = "Deployment name"
  value       = kubernetes_deployment.vllm.metadata[0].name
}

output "service_endpoint" {
  description = "Internal service endpoint"
  value       = "http://${kubernetes_service.vllm.metadata[0].name}.${kubernetes_namespace.ml.metadata[0].name}.svc.cluster.local:8000"
}

output "load_balancer_hostname" {
  description = "Load balancer hostname (if enabled)"
  value       = var.enable_load_balancer ? kubernetes_service.vllm_lb[0].status[0].load_balancer[0].ingress[0].hostname : null
}

output "model_id" {
  description = "Model ID being served"
  value       = var.model_id
}
