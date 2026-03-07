# GLM-5 llm-d — Example Overrides
# Uses vllm/vllm-openai:glm5 with tool calling (glm47) + reasoning (glm45)
project_name           = "glm5-llmd"
aws_region             = "us-east-2"
gpu_availability_zones = ["us-east-2b"]
gpu_instance_types     = ["p6-b200.48xlarge"]

# Multi-replica
replicas     = 2
gpu_max_size = 4

# vLLM GLM-5 image (official, includes DeepGEMM + tool parser)
vllm_image = "vllm/vllm-openai:glm5"

# LMCache enabled with Redis L2
# WARNING: LMCache NSA/MLA compatibility on vLLM needs validation
# Set enable_lmcache = false for first deploy, test before enabling
enable_lmcache    = false
lmcache_local_cpu = true
lmcache_redis_url = "redis://redis-glm5.ml-inference.svc.cluster.local:6379"

# FSx
fsx_storage_capacity    = 4800
fsx_per_unit_throughput = 500
