# Nemotron-3-Ultra-550B-A55B-NVFP4 on p6-b300.48xlarge (8x B300 275GB), us-west-2b
# Reuses existing qn-sglang-eks-cluster + ai-infra-b300-spot managed node group.
# Smoke config: vLLM v0.22.0-cu130, TP4 single replica (GPUs 0-3).

aws_region   = "us-west-2"
project_name = "nemotron-ultra"
environment  = "dev"

# Existing cluster + node group (reused)
eks_cluster_name        = "qn-sglang-eks-cluster"
namespace               = "ai-infra"
service_account_name    = "default"
gpu_node_group_name     = "ai-infra-b300-spot"
gpu_node_taint_key      = "ai-infra/b300"
gpu_node_label_selector = { "ai-infra/role" = "b300-spot" }

# Model
model_id          = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
model_path        = "/mnt/nvme/models/nemotron-3-ultra-nvfp4"
served_model_name = "nvidia/nemotron-3-ultra"
s3_model_bucket   = "qn-sglang-models-20260303161715850900000007"
s3_model_prefix   = "nemotron-3-ultra-nvfp4"

# Serving — vLLM TP4 single replica (NVIDIA's documented unit)
serving_image          = "vllm/vllm-openai:v0.22.0-cu130"
tp_size                = 4
max_model_len          = 262144
gpu_memory_utilization = 0.90
max_num_seqs           = 16
max_num_batched_tokens = 32768

# Stage 0c WARN: MTP + prefix caching may conflict with mamba 'align' mode.
# First attempt keeps the verbatim card config (true). Flip to false on startup failure.
enable_prefix_caching = true

# Networking
node_port = 30090
