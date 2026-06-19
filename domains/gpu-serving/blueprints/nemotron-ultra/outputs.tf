# Nemotron-3-Ultra Blueprint Outputs

output "cluster_name" {
  description = "EKS cluster name (reused)"
  value       = var.eks_cluster_name
}

output "namespace" {
  description = "Kubernetes namespace"
  value       = var.namespace
}

output "serving_endpoint" {
  description = "ClusterIP service endpoint"
  value       = "http://nemotron-ultra.${var.namespace}.svc.cluster.local:8000"
}

output "node_port" {
  description = "NodePort for external access"
  value       = var.node_port
}

output "model_id" {
  description = "HuggingFace model id under test"
  value       = var.model_id
}

output "served_model_name" {
  description = "Model name registered in the OpenAI API"
  value       = var.served_model_name
}

output "tp_size" {
  description = "Tensor parallelism degree (smoke = TP4 single replica)"
  value       = var.tp_size
}

output "gpu_node_group" {
  description = "Existing managed node group to scale 0->1 for the GPU node"
  value       = var.gpu_node_group_name
}

output "quick_start" {
  description = "Quick start commands"
  value       = <<-EOT
    # 1. Scale the B300 spot node group up (us-west-2b / usw2-az2)
    aws eks update-nodegroup-config --cluster-name ${var.eks_cluster_name} \
      --nodegroup-name ${var.gpu_node_group_name} \
      --scaling-config minSize=0,maxSize=1,desiredSize=1 --region ${var.aws_region}

    # 2. Stage weights (HF -> S3 -> NVMe) — see scripts/stage-model.sh
    # 3. terraform apply  (deploys the vLLM serving pod)

    # 4. Port-forward and smoke test
    kubectl -n ${var.namespace} port-forward svc/nemotron-ultra 8000:8000
    curl http://localhost:8000/health

    # 5. Scale node group back to 0 when done (cost discipline — ~$27/hr B300 spot)
    aws eks update-nodegroup-config --cluster-name ${var.eks_cluster_name} \
      --nodegroup-name ${var.gpu_node_group_name} \
      --scaling-config minSize=0,maxSize=1,desiredSize=0 --region ${var.aws_region}
  EOT
}
