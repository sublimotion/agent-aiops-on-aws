#!/usr/bin/env bash
# bootstrap-observability.sh — Install Prometheus + DCGM + node-exporter on the benchmark GPU node.
#
# MUST run AFTER the GPU node is up but BEFORE the serving stack starts. The bench driver
# relies on metrics being populated before any request is issued.
#
# Usage on the GPU node:
#   bash bootstrap-observability.sh [<s3-results-bucket>] [<blueprint-name>]

set -euo pipefail

S3_BUCKET="${1:-}"
BLUEPRINT="${2:-unknown}"
SESSION_ID="$(date -u +%Y%m%dT%H%M%SZ)"
NODE_NAME="$(hostname)"

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE="$SKILL_DIR/templates/observability-stack.docker-compose.yml"
PROM_CFG_SRC="$SKILL_DIR/templates/prometheus-bench.yaml"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

log "Bootstrapping observability on node=$NODE_NAME blueprint=$BLUEPRINT session=$SESSION_ID"

# ---------- Prepare directories ----------
sudo mkdir -p /etc/prometheus-bench /mnt/nvme/prom-data /mnt/nvme/prom-snapshots
sudo chown -R 65534:65534 /mnt/nvme/prom-data /mnt/nvme/prom-snapshots

# ---------- Render scrape config with session labels ----------
sudo bash -c "NODE_NAME='$NODE_NAME' BLUEPRINT='$BLUEPRINT' SESSION_ID='$SESSION_ID' \
  envsubst < '$PROM_CFG_SRC' > /etc/prometheus-bench/prometheus.yml"
log "Wrote /etc/prometheus-bench/prometheus.yml"

# ---------- Pull images ahead of docker compose up ----------
sudo docker pull prom/prometheus:v2.54.1 >/dev/null &
sudo docker pull nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04 >/dev/null &
sudo docker pull prom/node-exporter:v1.8.2 >/dev/null &
wait

# ---------- Launch stack ----------
sudo docker compose -f "$COMPOSE" up -d
log "Observability stack started"

# ---------- Install S3 sync systemd timer ----------
if [[ -n "$S3_BUCKET" ]]; then
  SYNC_SCRIPT="$SKILL_DIR/scripts/sync-prometheus-to-s3.sh"
  sudo cp "$SYNC_SCRIPT" /usr/local/bin/sync-prom-to-s3.sh
  sudo chmod +x /usr/local/bin/sync-prom-to-s3.sh

  cat <<UNIT | sudo tee /etc/systemd/system/prom-sync.service >/dev/null
[Unit]
Description=Snapshot Prometheus and sync to S3
After=docker.service
[Service]
Type=oneshot
Environment=S3_BUCKET=$S3_BUCKET
Environment=BLUEPRINT=$BLUEPRINT
Environment=SESSION_ID=$SESSION_ID
ExecStart=/usr/local/bin/sync-prom-to-s3.sh
UNIT

  cat <<TIMER | sudo tee /etc/systemd/system/prom-sync.timer >/dev/null
[Unit]
Description=Run Prometheus S3 sync every 10 minutes
[Timer]
OnBootSec=5min
OnUnitActiveSec=10min
Unit=prom-sync.service
[Install]
WantedBy=timers.target
TIMER

  sudo systemctl daemon-reload
  sudo systemctl enable --now prom-sync.timer
  log "Enabled prom-sync.timer → s3://$S3_BUCKET/prometheus/$BLUEPRINT/$SESSION_ID/"
else
  log "WARN: no S3 bucket provided, Prometheus data will NOT be synced (data lost on spot reclaim)"
fi

log "Bootstrap complete. Run: bash $SKILL_DIR/scripts/observability-smoke-test.sh"
