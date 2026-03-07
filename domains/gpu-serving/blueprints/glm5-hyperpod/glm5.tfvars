aws_region            = "us-west-2"
resource_name_prefix  = "glm5"
kubernetes_version    = "1.32"
eks_cluster_name      = "glm5-hyperpod-eks"
hyperpod_cluster_name = "glm5-cluster"

instance_groups = [
  {
    name                      = "glm5-p5e"
    instance_type             = "ml.p5e.48xlarge"
    instance_count            = 1
    ebs_volume_size_in_gb     = 500
    threads_per_core          = 2
    enable_stress_check       = true
    enable_connectivity_check = true
    lifecycle_script          = "on_create.sh"
    availability_zone_id      = "usw2-az2"
    training_plan_arn         = "arn:aws:sagemaker:us-west-2:ACCOUNT:training-plan/glm5-bench"
  }
]

fsx_storage_capacity = 4800
fsx_throughput       = 500
