output "prometheus_endpoint" {
  description = "Prometheus server endpoint"
  value       = "http://prometheus-server.${var.namespace}.svc.cluster.local:80"
}

output "grafana_endpoint" {
  description = "Grafana endpoint"
  value       = var.grafana_service_type == "NodePort" ? "http://<node-ip>:${var.grafana_node_port}" : "http://grafana.${var.namespace}.svc.cluster.local:80"
}

output "namespace" {
  description = "Monitoring namespace"
  value       = kubernetes_namespace.monitoring.metadata[0].name
}
