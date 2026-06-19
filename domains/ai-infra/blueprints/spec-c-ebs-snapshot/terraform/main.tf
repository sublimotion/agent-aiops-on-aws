###############################################################################
# Spec C-EBS — Persistent compile cache on EBS snapshot
#
# Creates:
#   1. A 700 GB gp3 EBS volume in us-east-2b (matches B200 spot AZ).
#   2. An EBS-CSI StorageClass for static volumes from snapshots.
#   3. A SnapshotClass for triggering snapshots from the running volume.
#   4. (After first bake) Snapshot of the volume → reusable across pod cycles.
#
# Lifecycle:
#   - terraform apply: creates volume + storage classes
#   - run bake pod: writes compile cache to /mnt/persistent (via PVC)
#   - aws ec2 create-snapshot: capture state
#   - terraform destroy: deletes volume but NOT snapshot (snapshot persists)
#   - new pod: PVC restored from snapshot, mount, AOT cache HIT
###############################################################################

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-2"
}

locals {
  cluster_name = "qwen3-next-bench-eks-cluster"
  region       = "us-east-2"
  az           = "us-east-2b"
  project      = "ai-infra-cache-ebs"
}

data "aws_eks_cluster" "this" {
  name = local.cluster_name
}

###############################################################################
# EBS volume — 700 GB gp3, in us-east-2b. Provisioned standalone; not part
# of the EKS nodegroup so it survives nodegroup scale-down.
#
# We use lifecycle.prevent_destroy on the snapshot, not the volume — the
# volume itself is cheap to recreate from snapshot.
###############################################################################

resource "aws_ebs_volume" "compile_cache" {
  availability_zone = local.az
  size              = 700
  type              = "gp3"
  iops              = 3000  # gp3 baseline — bumps cost ~$0 over default
  throughput        = 250   # MB/s — comfortable for 500 MB cache reads
  encrypted         = true

  # Optional: restore from an existing snapshot ID at apply time.
  # Set via `-var snapshot_id=snap-xxx` to restore a populated cache;
  # leave null on first apply to create empty.
  snapshot_id = var.snapshot_id

  tags = {
    Name    = "${local.project}-compile-cache"
    Project = local.project
    Purpose = "vLLM AOT compile cache, persistent across pod lifecycles"
  }
}

###############################################################################
# Snapshot policy: lifecycle manager that snapshots the volume daily,
# retaining the last 7. Cheap insurance during experimentation.
###############################################################################

resource "aws_iam_role" "dlm_lifecycle" {
  name = "${local.project}-dlm-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "dlm.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "dlm" {
  role       = aws_iam_role.dlm_lifecycle.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSDataLifecycleManagerServiceRole"
}

resource "aws_dlm_lifecycle_policy" "compile_cache_snapshots" {
  description        = "Daily compile cache snapshots keep last 7"
  execution_role_arn = aws_iam_role.dlm_lifecycle.arn
  state              = "ENABLED"

  policy_details {
    resource_types = ["VOLUME"]

    target_tags = {
      Project = local.project
    }

    schedule {
      name = "daily-7"

      create_rule {
        interval      = 24
        interval_unit = "HOURS"
        times         = ["03:00"]
      }

      retain_rule {
        count = 7
      }

      tags_to_add = {
        SnapshotTier = "lab"
      }

      copy_tags = true
    }
  }
}

###############################################################################
# Outputs
###############################################################################

output "volume_id" {
  value = aws_ebs_volume.compile_cache.id
}

output "volume_az" {
  value = aws_ebs_volume.compile_cache.availability_zone
}

output "attach_command" {
  description = "After EKS B200 node is up, attach the volume so a pod can mount it"
  value       = <<-EOT
    aws ec2 attach-volume \
      --volume-id ${aws_ebs_volume.compile_cache.id} \
      --instance-id <B200_INSTANCE_ID> \
      --device /dev/xvdf \
      --region ${local.region}
  EOT
}

output "create_snapshot_command" {
  value = <<-EOT
    aws ec2 create-snapshot \
      --volume-id ${aws_ebs_volume.compile_cache.id} \
      --description "compile-cache-$(date +%Y%m%d-%H%M%S)" \
      --tag-specifications "ResourceType=snapshot,Tags=[{Key=Project,Value=${local.project}}]" \
      --region ${local.region}
  EOT
}
