#!/usr/bin/env bash
# Set up RAID0 across the local instance-store NVMe SSDs of a p6-b200.48xlarge
# and mount at /mnt/nvme. Idempotent: no-op if /mnt/nvme is already a RAID mount.
#
# Runs INSIDE the host mount namespace (via nsenter from a privileged pod).
# Device layout (probed 2026-06-11): nvme0n1 = EBS root (has partitions);
# nvme1n1..nvme8n1 = 8x local NVMe (no partitions) -> RAID0 -> /mnt/nvme.
set -euo pipefail

MNT=/mnt/nvme
MD=/dev/md0

# Already mounted as a real (non-root) fs? Done.
if mountpoint -q "$MNT" && [ "$(findmnt -no SOURCE "$MNT")" != "/dev/nvme0n1p1" ]; then
  echo "[setup-nvme] $MNT already a dedicated mount ($(findmnt -no SOURCE "$MNT")); skipping."
  df -h "$MNT"
  exit 0
fi

# Collect local NVMe data disks = every nvmeXn1 that has NO partitions and is NOT the root disk.
DISKS=()
for d in /dev/nvme[1-9]n1 /dev/nvme1[0-9]n1; do
  [ -b "$d" ] || continue
  # skip if it carries the root filesystem
  if lsblk -no MOUNTPOINT "$d" 2>/dev/null | grep -q '/$'; then continue; fi
  DISKS+=("$d")
done

echo "[setup-nvme] RAID0 candidate disks: ${DISKS[*]}"
if [ "${#DISKS[@]}" -lt 1 ]; then
  echo "[setup-nvme] ERROR: no local NVMe data disks found" >&2
  exit 1
fi

# Tear down any stale array on these (fresh spot node should be clean).
mdadm --stop "$MD" 2>/dev/null || true

yes | mdadm --create "$MD" --level=0 --raid-devices="${#DISKS[@]}" "${DISKS[@]}" --force
mkfs.ext4 -F -m 0 "$MD"
mkdir -p "$MNT"
mount "$MD" "$MNT"
mkdir -p "$MNT/models"
chmod 0777 "$MNT" "$MNT/models"

echo "[setup-nvme] RAID0 mounted:"
df -h "$MNT"
cat /proc/mdstat
