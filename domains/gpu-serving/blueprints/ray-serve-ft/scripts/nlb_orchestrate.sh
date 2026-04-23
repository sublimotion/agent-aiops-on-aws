#!/usr/bin/env bash
set -euo pipefail

# NLB Fault Injection Orchestrator
# Runs traffic from a dedicated client pod (in-cluster, NLB ClusterIP) and injects
# faults from local machine. The client pod runs on a system node and is NOT affected
# by head/worker fault injection.
#
# Usage: ./scripts/nlb_orchestrate.sh T5   (or T1, T3, all)

NAMESPACE="ray-ft"
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"
CLIENT_POD="traffic-client"
HEAD_POD=$(kubectl get pods -n $NAMESPACE -l ray-node=head -o jsonpath='{.items[0].metadata.name}')
NLB_IP=$(kubectl get svc -n $NAMESPACE yolo-ft-nlb -o jsonpath='{.spec.clusterIP}')
NLB_URL="http://${NLB_IP}:80/"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Client pod: $CLIENT_POD"
echo "Head pod: $HEAD_POD"
echo "NLB ClusterIP: $NLB_IP"
echo "NLB URL: $NLB_URL"
echo "Results dir: $RESULTS_DIR"
echo ""

# Copy traffic script to client pod
kubectl cp "$(dirname "$0")/nlb_traffic.py" "$NAMESPACE/$CLIENT_POD:/tmp/nlb_traffic.py"

run_test() {
    local TEST_NAME=$1
    local DURATION=$2
    local INJECT_DELAY=$3
    local FAULT_CMD=$4
    local FAULT_DESC=$5

    echo "============================================================"
    echo "$TEST_NAME: $FAULT_DESC"
    echo "============================================================"
    echo "  Duration: ${DURATION}s, Fault at: ${INJECT_DELAY}s"
    echo ""

    # Start traffic in background on head pod
    local LOG="/tmp/nlb_${TEST_NAME}.log"
    kubectl exec -n $NAMESPACE $CLIENT_POD -- \
        python3 /tmp/nlb_traffic.py --url "$NLB_URL" --rps 50 --duration "$DURATION" \
        > "$RESULTS_DIR/nlb_${TEST_NAME}_${TIMESTAMP}.log" 2>&1 &
    local TRAFFIC_PID=$!
    echo "  Traffic PID: $TRAFFIC_PID"

    # Wait for warmup
    echo "  Warming up ${INJECT_DELAY}s..."
    sleep "$INJECT_DELAY"

    # Inject fault
    echo "  INJECTING FAULT: $FAULT_DESC"
    eval "$FAULT_CMD"

    # Wait for traffic to finish
    echo "  Waiting for traffic to complete..."
    wait $TRAFFIC_PID 2>/dev/null || true

    # Extract summary from log
    local LOG_FILE="$RESULTS_DIR/nlb_${TEST_NAME}_${TIMESTAMP}.log"
    echo ""
    echo "  --- Raw output ---"
    cat "$LOG_FILE"
    echo ""

    # Extract SUMMARY JSON line
    local SUMMARY_LINE=$(command grep '^SUMMARY ' "$LOG_FILE" | tail -1 | sed 's/^SUMMARY //')
    if [ -n "$SUMMARY_LINE" ]; then
        echo "$SUMMARY_LINE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
d['test'] = '$TEST_NAME'
d['fault'] = '$FAULT_DESC'
d['via'] = 'NLB ClusterIP (worker-only targets, in-cluster)'
print(json.dumps(d, indent=2))
" > "$RESULTS_DIR/nlb_${TEST_NAME}_summary.json"
        echo "  Summary: $RESULTS_DIR/nlb_${TEST_NAME}_summary.json"
        cat "$RESULTS_DIR/nlb_${TEST_NAME}_summary.json"
    fi
    echo ""
}

cleanup_t2() {
    # Uncordon nodes after T2
    local NODES=$(kubectl get nodes --selector=role=gpu -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')
    for node in $NODES; do
        kubectl uncordon "$node" 2>/dev/null || true
    done
}

# Get worker info
WORKER_PODS=$(kubectl get pods -n $NAMESPACE -l ray-node=worker -o jsonpath='{.items[0].metadata.name}')
WORKER_NODE=$(kubectl get pods -n $NAMESPACE -l ray-node=worker -o jsonpath='{.items[0].spec.nodeName}')

TEST="${1:-all}"

case "$TEST" in
    T5|t5)
        run_test "T5" 240 60 \
            "kubectl exec -n $NAMESPACE $HEAD_POD -c ray-head -- pkill -f ProxyActor || true" \
            "Kill head HTTP proxy"
        ;;
    T3|t3)
        run_test "T3" 300 60 \
            "kubectl delete pod -n $NAMESPACE $HEAD_POD --force --grace-period=0" \
            "Kill head pod (GCS FT ON)"
        ;;
    T1|t1)
        run_test "T1" 240 60 \
            "kubectl exec -n $NAMESPACE $WORKER_PODS -c ray-worker -- pkill -f 'ray::SERVE_REPLICA' || true" \
            "Kill YOLO replica in $WORKER_PODS"
        ;;
    T2|t2)
        run_test "T2" 300 60 \
            "kubectl drain $WORKER_NODE --ignore-daemonsets --delete-emptydir-data --force --grace-period=30 --timeout=120s || true" \
            "Drain worker node $WORKER_NODE"
        cleanup_t2
        ;;
    all)
        echo "Running all tests: T5, T3, T1, T2"
        echo ""

        # T5 first (non-destructive)
        run_test "T5" 240 60 \
            "kubectl exec -n $NAMESPACE $HEAD_POD -c ray-head -- pkill -f ProxyActor || true" \
            "Kill head HTTP proxy"

        echo "Waiting 60s for recovery..."
        sleep 60

        # T3 (kills head — need to re-resolve head pod after)
        run_test "T3" 300 60 \
            "kubectl delete pod -n $NAMESPACE $HEAD_POD --force --grace-period=0" \
            "Kill head pod (GCS FT ON)"

        echo "Waiting 180s for head recovery + YOLO redeploy..."
        sleep 180

        # Refresh head pod name after recreation
        echo "Waiting for new head pod..."
        for i in $(seq 1 30); do
            HEAD_POD=$(kubectl get pods -n $NAMESPACE -l ray-node=head -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
            if [ -n "$HEAD_POD" ]; then
                PHASE=$(kubectl get pod -n $NAMESPACE $HEAD_POD -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
                if [ "$PHASE" = "Running" ]; then echo "New head pod: $HEAD_POD ($PHASE)"; break; fi
            fi
            echo "  waiting..."
            sleep 15
        done

        # Wait for serve to be healthy
        echo "Waiting for YOLO to be healthy..."
        for i in $(seq 1 30); do
            STATUS=$(kubectl exec -n $NAMESPACE $HEAD_POD -c ray-head -- python3 -c "
import ray; from ray import serve; ray.init(address='auto')
s = serve.status()
for a in s.applications.values(): print(a.status.name)
" 2>&1 | tail -1)
            echo "  $STATUS"
            if [ "$STATUS" = "RUNNING" ]; then break; fi
            sleep 15
        done

        # T1
        WORKER_PODS=$(kubectl get pods -n $NAMESPACE -l ray-node=worker -o jsonpath='{.items[0].metadata.name}')
        run_test "T1" 240 60 \
            "kubectl exec -n $NAMESPACE $WORKER_PODS -c ray-worker -- pkill -f 'ray::SERVE_REPLICA' || true" \
            "Kill YOLO replica in $WORKER_PODS"

        echo "Waiting 180s for replica recovery..."
        sleep 180

        # T2
        WORKER_NODE=$(kubectl get pods -n $NAMESPACE -l ray-node=worker -o jsonpath='{.items[0].spec.nodeName}')
        run_test "T2" 300 60 \
            "kubectl drain $WORKER_NODE --ignore-daemonsets --delete-emptydir-data --force --grace-period=30 --timeout=120s || true" \
            "Drain worker node $WORKER_NODE"
        cleanup_t2

        # Combine results
        echo ""
        echo "============================================================"
        echo "COMBINED RESULTS"
        echo "============================================================"
        python3 -c "
import json, glob, os
results = []
for f in sorted(glob.glob('$RESULTS_DIR/nlb_T*_summary.json')):
    with open(f) as fh:
        results.append(json.load(fh))
with open('$RESULTS_DIR/nlb_ft_all_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print(f'{\"Test\":<6} {\"ErrRate\":>8} {\"ErrWin\":>8} {\"P50ms\":>8} {\"P99ms\":>8}')
print('-' * 45)
for r in results:
    print(f'{r[\"test\"]:<6} {r[\"error_rate_pct\"]:>7.1f}% {r[\"error_window_s\"]:>7.1f}s {r[\"latency_p50_ms\"]:>7.0f} {r[\"latency_p99_ms\"]:>7.0f}')
"
        ;;
    *)
        echo "Usage: $0 {T1|T2|T3|T5|all}"
        exit 1
        ;;
esac

echo ""
echo "Done."
