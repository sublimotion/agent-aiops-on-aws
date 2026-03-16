#!/usr/bin/env bash
# Start 4x vLLM replicas serving Devstral Small 2 FP8 on ports 8001-8004
# with a round-robin load balancer on port 9000.
#
# Usage:
#   bash setup_vllm.sh          # Start serving
#   bash setup_vllm.sh stop     # Stop all replicas

set -euo pipefail

MODEL="/mnt/nvme/models/devstral-small-2-fp8"
PORTS=(8001 8002 8003 8004)
LB_PORT=9000

if [[ "${1:-}" == "stop" ]]; then
    echo "Stopping vLLM replicas..."
    for port in "${PORTS[@]}"; do
        pkill -f "vllm.*--port ${port}" 2>/dev/null || true
    done
    pkill -f "lb-proxy.py" 2>/dev/null || true
    echo "Done."
    exit 0
fi

echo "Starting 4x vLLM replicas for Devstral Small 2 FP8..."

for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}
    gpu=$i
    log="/mnt/nvme/vllm-${port}.log"

    echo "  GPU $gpu → port $port (log: $log)"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$port" \
        --served-model-name devstral-small-2-fp8 \
        --tool-call-parser mistral \
        --enable-auto-tool-choice \
        --max-model-len 32768 \
        --dtype auto \
        --gpu-memory-utilization 0.90 \
        > "$log" 2>&1 &
done

echo "Waiting for replicas to start..."
for port in "${PORTS[@]}"; do
    for attempt in $(seq 1 60); do
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "  Port $port: ready"
            break
        fi
        sleep 5
    done
done

echo "Starting load balancer on port $LB_PORT..."
nohup python3 /mnt/nvme/sera-scripts/lb-proxy.py \
    --listen-port $LB_PORT \
    --backends "${PORTS[@]/#/localhost:}" \
    > /mnt/nvme/lb-proxy.log 2>&1 &

echo "Done. Endpoint: http://localhost:${LB_PORT}/v1"
echo "Test: curl http://localhost:${LB_PORT}/v1/models"
