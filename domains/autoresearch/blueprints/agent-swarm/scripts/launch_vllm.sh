#!/usr/bin/env bash
# Launch vLLM for a specific model in the agent-swarm experiment.
# Usage: ./launch_vllm.sh <model_id> [port]
#
# Models:
#   devstral-small-2   — Devstral Small 2 24B FP8 (TP1, GPU 0)
#   qwen25-coder-32b   — Qwen 2.5 Coder 32B Instruct (TP1, GPU 0)
#   swesmith-qwen25-32b — SWE-smith Qwen 2.5 Coder 32B (TP1, GPU 0)
#   qwen35-397b        — Qwen3.5-397B-A17B FP8 (TP4, GPU 0-3)

set -euo pipefail

MODEL_ID="${1:?Usage: $0 <model_id> [port]}"
PORT="${2:-8000}"

case "$MODEL_ID" in
    devstral-small-2)
        WEIGHTS="/mnt/nvme/models/devstral-small-2-fp8"
        TP=1
        CONTEXT=65536
        EXTRA_ARGS="--tool-call-parser mistral --enable-auto-tool-choice --enable-prefix-caching"
        ;;
    qwen25-coder-32b)
        WEIGHTS="/mnt/nvme/models/qwen25-coder-32b"
        TP=1
        CONTEXT=65536
        EXTRA_ARGS="--tool-call-parser hermes --enable-auto-tool-choice --enable-prefix-caching"
        ;;
    swesmith-qwen25-32b)
        WEIGHTS="/mnt/nvme/models/swe-agent-lm-32b"
        TP=1
        CONTEXT=65536
        EXTRA_ARGS="--tool-call-parser hermes --enable-auto-tool-choice --enable-prefix-caching"
        ;;
    qwen35-397b)
        WEIGHTS="/mnt/nvme/models/qwen3-next-fp8"
        TP=4
        CONTEXT=65536
        EXTRA_ARGS="--tool-call-parser hermes --enable-auto-tool-choice --enable-prefix-caching"
        ;;
    *)
        echo "Unknown model: $MODEL_ID"
        echo "Choose from: devstral-small-2, qwen25-coder-32b, swesmith-qwen25-32b, qwen35-397b"
        exit 1
        ;;
esac

# Kill any existing vLLM process
echo "Stopping existing vLLM..."
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

echo "Launching vLLM for $MODEL_ID (TP=$TP, context=$CONTEXT, port=$PORT)..."
echo "Weights: $WEIGHTS"

# shellcheck disable=SC2086
python3 -m vllm.entrypoints.openai.api_server \
    --model "$WEIGHTS" \
    --served-model-name "$MODEL_ID" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$CONTEXT" \
    --port "$PORT" \
    --dtype auto \
    --trust-remote-code \
    $EXTRA_ARGS \
    2>&1 | tee "/mnt/nvme/vllm_${MODEL_ID}.log"
