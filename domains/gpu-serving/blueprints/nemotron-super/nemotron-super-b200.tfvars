# Nemotron-3-Super on p6-b200.48xlarge (8x B200 183GB)
# Reuses existing glm5-lmcache-b200 EKS cluster

aws_region   = "us-east-2"
project_name = "nemotron-super"
environment  = "dev"

# Existing cluster
eks_cluster_name     = "glm5-lmcache-b200-eks-cluster"
namespace            = "ml-inference"
service_account_name = "default"

# Model
model_path     = "/mnt/nvme/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
model_fsx_path = "models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"

# Serving — aggregated TP2 x 4 workers (P0 baseline)
serving_backend        = "vllm"
tp_size                = 2
num_workers            = 4
gpu_memory_utilization = 0.90
max_num_seqs           = 256

# Resources per worker
worker_cpu    = 32
worker_memory = "128Gi"

# Storage
fsx_pvc_name   = "fsx-ip-pvc"
fsx_ip         = "10.0.19.231"
fsx_mount_name = "gsyc5b4v"

# Networking
node_port = 30088
