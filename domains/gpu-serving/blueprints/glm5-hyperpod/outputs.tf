# GLM-5 HyperPod Blueprint Outputs

# =============================================================================
# Cluster Access
# =============================================================================

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = var.eks_cluster_name
}

output "hyperpod_cluster_name" {
  description = "SageMaker HyperPod cluster name"
  value       = var.hyperpod_cluster_name
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --name ${var.eks_cluster_name} --region ${var.aws_region}"
}

# =============================================================================
# S3 Buckets
# =============================================================================

output "model_weights_bucket" {
  description = "S3 bucket for GLM-5 model weights"
  value       = aws_s3_bucket.model_weights.id
}

output "benchmark_results_bucket" {
  description = "S3 bucket for benchmark results"
  value       = aws_s3_bucket.benchmark_results.id
}

# =============================================================================
# ECR Repositories
# =============================================================================

output "sglang_ecr_uri" {
  description = "ECR URI for custom SGLang container"
  value       = aws_ecr_repository.sglang_glm5.repository_url
}

output "bench_runner_ecr_uri" {
  description = "ECR URI for benchmark runner container"
  value       = aws_ecr_repository.bench_runner.repository_url
}

# =============================================================================
# SSM Access
# =============================================================================

output "ssm_access_command" {
  description = "SSM session command template (fill in cluster-id and instance-id)"
  value       = <<-EOT
    # List cluster nodes
    aws sagemaker list-cluster-nodes --cluster-name ${var.hyperpod_cluster_name} --region ${var.aws_region}

    # Start SSM session (replace <cluster-id> and <instance-id>)
    aws ssm start-session \
        --target sagemaker-cluster:<cluster-id>_glm5-p5e-<instance-id> \
        --region ${var.aws_region}
  EOT
}

# =============================================================================
# Invoke Endpoint
# =============================================================================

output "invoke_endpoint_example" {
  description = "Python code to invoke the GLM-5 endpoint via SageMaker Runtime"
  value       = <<-EOT
    import boto3, json
    runtime = boto3.client('sagemaker-runtime', region_name='${var.aws_region}')
    response = runtime.invoke_endpoint(
        EndpointName='glm5-fp8',
        ContentType='application/json',
        Body=json.dumps({
            "model": "${var.sglang_served_model_name}",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 512,
            "temperature": 0.7
        })
    )
    print(json.loads(response['Body'].read()))
  EOT
}

# =============================================================================
# Quick Start
# =============================================================================

output "quick_start" {
  description = "Quick start steps for GLM-5 on HyperPod"
  value       = <<-EOT
    # 1. Stage model weights to S3
    ./scripts/stage-model.sh ${aws_s3_bucket.model_weights.id}

    # 2. Build and push SGLang container to ECR
    ./scripts/copy-images-to-ecr.sh ${var.aws_region}

    # 3. Configure kubectl
    aws eks update-kubeconfig --name ${var.eks_cluster_name} --region ${var.aws_region}

    # 4. Run pre-flight GPU validation
    ./scripts/pre-flight.sh

    # 5. Apply InferenceEndpointConfig CRD (H200 track)
    ./scripts/apply-crd.sh manifests/glm5-inference-endpoint-h200.yaml ${aws_ecr_repository.sglang_glm5.repository_url}:${var.sglang_image_tag}

    # 6. Wait for model to load (monitor via AMP/AMG)
    kubectl get inferenceendpointconfig -w

    # 7. Test inference via SageMaker endpoint
    python3 -c "$(cat <<'PYEOF'
    import boto3, json
    runtime = boto3.client('sagemaker-runtime', region_name='${var.aws_region}')
    resp = runtime.invoke_endpoint(
        EndpointName='glm5-fp8',
        ContentType='application/json',
        Body=json.dumps({"model": "${var.sglang_served_model_name}", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64})
    )
    print(json.loads(resp['Body'].read()))
    PYEOF
    )"

    # 8. Check benchmark results bucket
    aws s3 ls s3://${aws_s3_bucket.benchmark_results.id}/
  EOT
}
