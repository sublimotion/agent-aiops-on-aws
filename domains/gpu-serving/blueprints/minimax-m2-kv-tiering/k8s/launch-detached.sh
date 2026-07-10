#!/usr/bin/env bash
# Detached launcher for the unattended Pareto sweep.
# Detaches via nohup (survives the operator's shell/SSH closing — macOS has no setsid).
# The 420-min HARD CAP is enforced TWICE (defense in depth):
#   1. `timeout` here delivers SIGTERM at the cap -> runner's TERM trap -> EXIT trap -> scaledown.
#   2. The runner itself starts an internal watchdog (WALL_CAP_MIN) that kills its own pgid if
#      `timeout` is ever unavailable. Either path guarantees the nodegroup scales to 0.
#
#   ./launch-detached.sh           # full grid, detached
#   VALIDATE_ONLY=1 ./launch-detached.sh   # one-config plumbing check, detached
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$(cd "${HERE}/../results" && pwd)"
LOG="${RESULTS}/sweep.log"
CAP_MIN=420

# ── Context pin (2026-06-27 wrong-cluster incident) ───────────────────────────
# A parallel us-west-2 sweep keeps stealing kubectl current-context (flips to
# qn-sglang-usw2). NEVER rely on current-context. Pin to the context that maps to
# the intended cluster and EXPORT it so the runner (and any kubectl added here)
# always targets the right cluster. Use `kubectl --context "$CTX"` for any kubectl
# call added to this launcher — do NOT call bare `kubectl`.
CTX="${CTX:-qwen3-next-bench-eks-cluster}"
export EXPECT_CONTEXT="$CTX"   # run-tiering-sweep.sh reads this and pins every kubectl call to it
KCTX=(kubectl --context "$CTX")

# Prefer GNU timeout (coreutils). Fall back to gtimeout, else run bare (the runner's own
# internal watchdog still enforces the cap).
TIMEOUT_BIN=""
if command -v timeout >/dev/null 2>&1; then TIMEOUT_BIN="timeout";
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_BIN="gtimeout"; fi

echo "[$(date -u +%FT%TZ)] launching detached sweep (cap=${CAP_MIN}m, timeout_bin='${TIMEOUT_BIN:-none/internal}'). log=${LOG}" | tee -a "$LOG"

if [ -n "$TIMEOUT_BIN" ]; then
  nohup "$TIMEOUT_BIN" --preserve-status --signal=TERM --kill-after=5m "${CAP_MIN}m" \
    bash "${HERE}/run-tiering-sweep.sh" >> "$LOG" 2>&1 &
else
  # No external timeout: the runner's internal WALL_CAP watchdog is the sole cap enforcer.
  WALL_CAP_MIN="${CAP_MIN}" nohup bash "${HERE}/run-tiering-sweep.sh" >> "$LOG" 2>&1 &
fi
PID=$!
echo "$PID" > "${RESULTS}/sweep.pid"
disown "$PID" 2>/dev/null || true
echo "detached. pid=${PID} (written to ${RESULTS}/sweep.pid)"
echo "watch:   tail -f ${LOG}"
echo "status:  cat ${RESULTS}/STATUS"
