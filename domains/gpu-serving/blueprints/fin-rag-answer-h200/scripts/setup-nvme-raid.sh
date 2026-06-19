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

echo "[setup-nvme] local NVMe data disks: ${DISKS[*]}"
if [ "${#DISKS[@]}" -lt 1 ]; then
  echo "[setup-nvme] ERROR: no local NVMe data disks found" >&2
  exit 1
fi

mkdir -p "$MNT"
if [ "${#DISKS[@]}" -eq 1 ]; then
  # Single local NVMe (e.g. g7e.12xlarge has one 3800GB disk) -> plain ext4, no RAID.
  DEV="${DISKS[0]}"
  echo "[setup-nvme] single disk $DEV -> ext4 (no RAID)"
  mkfs.ext4 -F -m 0 "$DEV"
  mount "$DEV" "$MNT"
else
  # Multiple local NVMe (e.g. g7e.48xlarge, p5e.48xlarge) -> RAID0 stripe.
  # Idempotent + non-interactive: stop any prior array, wipe stale superblocks so
  # --create never prompts, and use --run (not a `yes |` pipe, which trips pipefail
  # via SIGPIPE on a fresh-but-rerun node and leaves the array unmounted).
  mdadm --stop "$MD" 2>/dev/null || true
  for d in "${DISKS[@]}"; do mdadm --zero-superblock "$d" 2>/dev/null || true; done
  mdadm --create "$MD" --level=0 --raid-devices="${#DISKS[@]}" "${DISKS[@]}" --force --run
  mkfs.ext4 -F -m 0 "$MD"
  mount "$MD" "$MNT"
fi
mkdir -p "$MNT/models"
chmod 0777 "$MNT" "$MNT/models"

echo "[setup-nvme] mounted:"
df -h "$MNT"
cat /proc/mdstat 2>/dev/null || true
