variable "snapshot_id" {
  description = <<-EOT
    Optional EBS snapshot ID to restore from on volume creation. Leave null
    on first apply to create empty volume that the bake pod populates.
    On subsequent applies after a snapshot exists, set this to the snapshot
    ID so terraform-managed volume comes up pre-populated.
  EOT
  type    = string
  default = null
}
