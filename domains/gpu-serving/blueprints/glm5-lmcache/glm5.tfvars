# GLM-5 LMCache — Example Overrides
project_name            = "glm5-lmcache"
aws_region              = "us-east-2"
gpu_availability_zones  = ["us-east-2c"]
capacity_reservation_id = ""

# LMCache off by default (baseline first)
enable_lmcache = false

# FSx
fsx_storage_capacity    = 4800
fsx_per_unit_throughput = 500
