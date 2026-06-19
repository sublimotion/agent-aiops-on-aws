#!/usr/bin/env bash
# A/B/C arm launcher for PrisKV shared-prefix-cache experiment (g7e.12xl, 2× TP1).
# Run ON the node. Each arm tears down the previous one first.
#
# Arm A: 2 vanilla vLLM replicas, engine-local APC, round-robin front (no prefix awareness)
# Arm B: 2 vanilla vLLM replicas, engine-local APC, PREFIX-AWARE routing
# Arm C: 2 vLLM replicas, engine-local APC OFF, shared PrisKV L2 cache, round-robin front
#
# Stock engine = vllm/vllm-openai:v0.10.2 (sm_120-capable). PrisKV engine = vllm-aibrix:v0.10.2.
set -euo pipefail

MODEL=/mnt/nvme/models/Qwen3-32B-FP8
SERVED=Qwen3-32B-FP8
MAXLEN=24000
IMG_STOCK=vllm/vllm-openai:v0.10.2
IMG_PRISKV=vllm-aibrix:v0.10.2
IMG_PRISKV_SERVER=priskv:local

teardown() {
  docker rm -f vllm-r0 vllm-r1 priskv-server priskv-redis router 2>/dev/null || true
  sleep 2
}

# --- replica launcher (stock) : $1=name $2=gpu $3=hostport ---
launch_stock_replica() {
  docker run -d --name "$1" --gpus "\"device=$2\"" --network host \
    -v "$MODEL":"$MODEL":ro --entrypoint python3 "$IMG_STOCK" \
    -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port "$3" \
    --model "$MODEL" --served-model-name "$SERVED" --tensor-parallel-size 1 \
    --max-model-len "$MAXLEN" --gpu-memory-utilization 0.90 \
    --enable-prefix-caching --disable-log-requests
}

arm_A() {  # local cache + round-robin
  teardown
  launch_stock_replica vllm-r0 0 18001
  launch_stock_replica vllm-r1 1 18002
  # round-robin nginx on :8080 over the two replicas
  docker run -d --name router --network host nginx:1.27-alpine sh -c '
    cat > /etc/nginx/nginx.conf <<EOF
events {}
http {
  upstream b { server 127.0.0.1:18001; server 127.0.0.1:18002; }
  server { listen 8080; location / { proxy_pass http://b; proxy_buffering off; } }
}
EOF
    nginx -g "daemon off;"'
  echo "Arm A up on :8080 (round-robin, local APC)"
}

arm_B() {  # local cache + PREFIX-AWARE routing
  teardown
  launch_stock_replica vllm-r0 0 18001
  launch_stock_replica vllm-r1 1 18002
  # prefix-aware: hash the shared-prefix region of the body to a stable upstream.
  # nginx split_clients on a request header the bench sets ($http_x_prefix_key),
  # so same-prefix requests pin to the same replica (the routing-to-state competitor to PrisKV).
  docker run -d --name router --network host nginx:1.27-alpine sh -c '
    cat > /etc/nginx/nginx.conf <<EOF
events {}
http {
  upstream r0 { server 127.0.0.1:18001; }
  upstream r1 { server 127.0.0.1:18002; }
  split_clients "\$http_x_prefix_key" \$pool { 50% r0; 50% r1; }
  server { listen 8080; location / { proxy_pass http://\$pool; proxy_buffering off; } }
}
EOF
    nginx -g "daemon off;"'
  echo "Arm B up on :8080 (prefix-aware via X-Prefix-Key header)"
}

arm_C() {  # shared PrisKV L2, local L1 off, round-robin
  teardown
  # 1) redis (cluster-meta store) + 2) priskv-server
  docker run -d --name priskv-redis --network host redis:7.4.2 \
    redis-server --port 16379 --requirepass kvcache-redis --appendonly no
  sleep 3
  META='{"version":1,"nodes":[{"name":"node0","addr":"127.0.0.1","port":9000,"slots":[{"start":0,"end":4095}]}]}'
  echo "$META" | docker run -i --rm --network host redis:7.4.2 \
    redis-cli -h 127.0.0.1 -p 16379 -x SET priskv_cluster_metadata
  docker run -i --rm --network host redis:7.4.2 \
    redis-cli -h 127.0.0.1 -p 16379 CONFIG SET requirepass kvcache-redis || true
  # priskv-server: UCX transport (no RDMA HW on single node), shared-mem+tcp tls
  docker run -d --name priskv-server --network host --ipc host \
    -e PRISKV_TRANSPORT=UCX -e UCX_TLS=sm,self,tcp \
    --entrypoint /workspace/priskv-server "$IMG_PRISKV_SERVER" \
    -a 127.0.0.1 -p 9000 -v 1048576 -b 524288 -k 2097152 -K 256 -t 16 \
    --acl any -L stdout -l notice
  sleep 5
  # 3) two PrisKV-enabled replicas: L1 OFF, L2=PRISKV shared, APC off (connector owns reuse)
  for pair in "vllm-r0 0 18001" "vllm-r1 1 18002"; do
    set -- $pair
    docker run -d --name "$1" --gpus "\"device=$2\"" --network host --ipc host \
      -v "$MODEL":"$MODEL":ro \
      -e AIBRIX_KV_CACHE_OL_BLOCK_SIZE=64 \
      -e AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED=0 \
      -e AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND=PRISKV \
      -e AIBRIX_KV_CACHE_OL_PRISKV_REMOTE_ADDR=127.0.0.1 \
      -e AIBRIX_KV_CACHE_OL_PRISKV_REMOTE_PORT=16379 \
      -e AIBRIX_KV_CACHE_OL_PRISKV_PASSWORD=kvcache-redis \
      -e PRISKV_CLUSTER_META="$META" \
      --entrypoint python3 "$IMG_PRISKV" \
      -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port "$3" \
      --model "$MODEL" --served-model-name "$SERVED" --tensor-parallel-size 1 \
      --max-model-len "$MAXLEN" --gpu-memory-utilization 0.90 \
      --no-enable-prefix-caching \
      --kv-transfer-config '{"kv_connector":"AIBrixOffloadingConnectorV1Type3","kv_role":"kv_both"}' \
      --disable-log-requests
  done
  docker run -d --name router --network host nginx:1.27-alpine sh -c '
    cat > /etc/nginx/nginx.conf <<EOF
events {}
http {
  upstream b { server 127.0.0.1:18001; server 127.0.0.1:18002; }
  server { listen 8080; location / { proxy_pass http://b; proxy_buffering off; } }
}
EOF
    nginx -g "daemon off;"'
  echo "Arm C up on :8080 (round-robin, shared PrisKV L2, L1 off)"
}

case "${1:-}" in
  A) arm_A ;; B) arm_B ;; C) arm_C ;; down) teardown ;;
  *) echo "usage: $0 {A|B|C|down}"; exit 1 ;;
esac
