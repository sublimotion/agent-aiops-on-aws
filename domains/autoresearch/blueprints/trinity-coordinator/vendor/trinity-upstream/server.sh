#!/bin/bash

# Start vLLM servers in the background
for i in {0..3}; do
    gpu_device=\$i
    
    case \$i in
        0) port=8325; model="Qwen/Qwen3-32B" ;;
        1) port=8326; model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" ;;
        2) port=8327; model="google/gemma-3-27b-it" ;;
        3) port=8328; model="Qwen/Qwen3-32B" ;;
    esac
    
    CUDA_VISIBLE_DEVICES=\${gpu_device} vllm serve "\${model}" \
        --max-model-len=16384 \
        --gpu_memory_utilization=0.95 \
        --port \${port} &
done

echo "All servers started, keeping job running..."

wait

echo "Job finished at \$(date)"
EOF