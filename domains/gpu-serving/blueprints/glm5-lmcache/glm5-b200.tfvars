# GLM-5 LMCache on B200 — Capacity Block cr-0827eef18c1c46bcd
project_name            = "glm5-lmcache-b200"
aws_region              = "us-east-2"
gpu_availability_zones  = ["us-east-2b"]
capacity_reservation_id = "cr-0827eef18c1c46bcd"

# B200 instance
gpu_instance_types = ["p6-b200.48xlarge"]

# LMCache off by default (baseline first)
enable_lmcache = false

# B200 has 1,536 GB HBM — can use higher mem fraction
mem_fraction_static = 0.90

# FSx in same AZ
fsx_storage_capacity    = 4800
fsx_per_unit_throughput = 500
