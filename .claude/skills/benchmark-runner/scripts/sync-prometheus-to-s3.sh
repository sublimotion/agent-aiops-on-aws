#!/usr/bin/env bash
# sync-prometheus-to-s3.sh — Take a Prometheus TSDB snapshot and push to S3.
# Called every 10 min by systemd timer installed by bootstrap-observability.sh.
#
# Env:
#   S3_BUCKET   — target bucket (without s3:// prefix)
#   BLUEPRINT   — blueprint name (e.g. kimi-k2.6-speculative)
#   SESSION_ID  — session id set at bootstrap time (UTC timestamp)
#
# Idempotent — only uploads new snapshot dirs. Runs as root via systemd.

set -euo pipefail

S3_BUCKET="${S3_BUCKET:?S3_BUCKET required}"
BLUEPRINT="${BLUEPRINT:-unknown}"
SESSION_ID="${SESSION_ID:-unknown}"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

# 1. Trigger snapshot (writes to /mnt/nvme/prom-snapshots/<auto-name>)
SNAP_RESP=$(curl -sf -XPOST http://localhost:9090/api/v1/admin/tsdb/snapshot || true)
if [[ -z "$SNAP_RESP" ]]; then
  log "WARN: snapshot API unreachable — is Prometheus running?"
  exit 0
fi
SNAP_NAME=$(echo "$SNAP_RESP" | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['name'])" 2>/dev/null || true)
if [[ -z "$SNAP_NAME" ]]; then
  log "WARN: snapshot name not returned, response: $SNAP_RESP"
  exit 0
fi

SNAP_PATH="/mnt/nvme/prom-snapshots/$SNAP_NAME"
if [[ ! -d "$SNAP_PATH" ]]; then
  log "WARN: snapshot path $SNAP_PATH does not exist"
  exit 0
fi

# 2. Sync to S3 (deduped on content by aws s3 sync)
DEST="s3://$S3_BUCKET/prometheus/$BLUEPRINT/$SESSION_ID/$SNAP_NAME/"
log "Syncing $SNAP_PATH → $DEST"
aws s3 sync "$SNAP_PATH" "$DEST" --only-show-errors

# 3. Sync live WAL too (so forensics work even if last snapshot is stale)
aws s3 sync /mnt/nvme/prom-data/wal \
  "s3://$S3_BUCKET/prometheus/$BLUEPRINT/$SESSION_ID/wal-latest/" \
  --only-show-errors --delete

# 4. Delete snapshots older than 24h locally to free NVMe
find /mnt/nvme/prom-snapshots/ -mindepth 1 -maxdepth 1 -type d -mmin +1440 -exec rm -rf {} \; 2>/dev/null || true

log "Sync complete"
