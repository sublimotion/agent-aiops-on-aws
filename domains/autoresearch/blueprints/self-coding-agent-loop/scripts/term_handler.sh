#!/bin/bash
# Spot termination handler for self-coding-agent-loop p4de.
#
# Invocation:
#   term_handler.sh watch   — long-running; polls IMDS and fires on interruption.
#   term_handler.sh flush   — called by watch OR manually; snaps everything to S3.
#
# Strategy when AWS signals interruption (2-min warning):
#   1. Write "reclaim-in-progress" marker
#   2. Send SIGUSR1 to any training processes (they should checkpoint on receipt)
#   3. Wait up to 20s for graceful checkpoint
#   4. Blocking `aws s3 sync` of runs/ — S3 in same region (us-east-1) is fast
#   5. Optional rsync to laptop if SSH target reachable (best-effort, non-blocking)
#
# S3 is the authoritative backup (same-region, reliable bandwidth).
# Laptop rsync is secondary (best-effort, requires SSH tunnel or tailscale).

set -u
WORK_DIR="${WORK_DIR:-/mnt/nvme/self-coding-agent-loop}"
BUCKET="${EXPERIMENT_BUCKET:-s3://agent-aiops-artifacts/self-coding-agent-loop}"
IMDS_TOKEN_TTL=300

imds_token() {
    curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: $IMDS_TOKEN_TTL"
}

check_interruption() {
    local tok=$1
    curl -sS -o /dev/null -w "%{http_code}" \
        "http://169.254.169.254/latest/meta-data/spot/instance-action" \
        -H "X-aws-ec2-metadata-token: $tok"
}

flush() {
    echo "[$(date -u +%H:%M:%S)] term_handler flush: reclaim-in-progress" | tee -a "$WORK_DIR/runs/reclaim.log"
    touch "$WORK_DIR/runs/.reclaim-in-progress"

    # Signal any training processes to checkpoint (convention: they trap SIGUSR1)
    pkill -SIGUSR1 -f 'train_sft.py' 2>/dev/null || true
    pkill -SIGUSR1 -f 'train_grpo.py' 2>/dev/null || true
    pkill -SIGUSR1 -f 'train_lora.py' 2>/dev/null || true
    sleep 20  # let them flush

    # Blocking S3 sync — this is the authoritative backup
    aws s3 sync "$WORK_DIR/runs/" "$BUCKET/runs/" --only-show-errors 2>&1 | tee -a "$WORK_DIR/runs/reclaim.log"
    aws s3 sync "$WORK_DIR/checkpoints/" "$BUCKET/checkpoints/" --only-show-errors 2>&1 | tee -a "$WORK_DIR/runs/reclaim.log"

    # Best-effort laptop rsync
    if [[ -n "${LAPTOP_SSH_TARGET:-}" ]] && [[ -n "${LAPTOP_RUNS_DIR:-}" ]]; then
        timeout 60 rsync -az --partial --append-verify "$WORK_DIR/runs/" \
            "$LAPTOP_SSH_TARGET:$LAPTOP_RUNS_DIR/" 2>&1 | tee -a "$WORK_DIR/runs/reclaim.log" || true
    fi

    echo "[$(date -u +%H:%M:%S)] flush complete" | tee -a "$WORK_DIR/runs/reclaim.log"
}

watch() {
    # IMDS token refresh loop
    while true; do
        tok=$(imds_token || true)
        [[ -z "$tok" ]] && sleep 5 && continue
        code=$(check_interruption "$tok" || echo "000")
        if [[ "$code" == "200" ]]; then
            echo "[$(date -u +%H:%M:%S)] spot interruption signal detected" | tee -a "$WORK_DIR/runs/reclaim.log"
            flush
            # Don't exit — stay alive to handle repeat polls gracefully
            sleep 120
        fi
        sleep 5
    done
}

case "${1:-watch}" in
    watch) watch ;;
    flush) flush ;;
    *) echo "usage: $0 {watch|flush}"; exit 1 ;;
esac
