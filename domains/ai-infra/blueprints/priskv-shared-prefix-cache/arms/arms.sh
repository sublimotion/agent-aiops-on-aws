#!/usr/bin/env bash
set -uo pipefail
MODEL=/mnt/nvme/models/Qwen3-32B-FP8
SERVED=Qwen3-32B-FP8
MAXLEN=24000
IMG_STOCK=vllm/vllm-openai:v0.10.2
IMG_PRISKV=vllm-priskv:final2
IMG_SERVER=priskv:local

teardown() {
  docker rm -f vllm-test vllm-r0 vllm-r1 priskv-server priskv-redis router 2>/dev/null
  sleep 2
}

wait_health() {  # $1=port $2=label
  for i in $(seq 1 40); do
    h=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$1/health" 2>/dev/null)
    [ "$h" = "200" ] && { echo "$2 healthy"; return 0; }
    docker ps -q -f name=vllm-r >/dev/null || true
    sleep 10
  done
  echo "$2 FAILED to become healthy"; return 1
}

stock_replica() {  # $1=name $2=gpu $3=port
  docker run -d --name "$1" --gpus "\"device=$2\"" --network host \
    -v "$MODEL":"$MODEL":ro --entrypoint python3 "$IMG_STOCK" \
    -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port "$3" \
    --model "$MODEL" --served-model-name "$SERVED" --tensor-parallel-size 1 \
    --max-model-len "$MAXLEN" --gpu-memory-utilization 0.90 \
    --enable-prefix-caching --disable-log-requests >/dev/null
}

start_priskv_infra() {
  docker run -d --name priskv-redis --network host redis:7.4.2 \
    redis-server --port 16379 --requirepass kvcache-redis --appendonly no >/dev/null
  sleep 3
  docker cp /mnt/nvme/priskv/exp/meta.json priskv-redis:/tmp/meta.json
  docker exec priskv-redis sh -c "tr -d '\n' < /tmp/meta.json | redis-cli -p 16379 -a kvcache-redis -x SET priskv_cluster_metadata" 2>/dev/null
  docker run -d --name priskv-server --network host --ipc host --shm-size 40g \
    -e PRISKV_TRANSPORT=ucx -e UCX_TLS=tcp -e PRISKV_USE_SHM=n \
    --entrypoint /workspace/priskv-server "$IMG_SERVER" \
    -a 127.0.0.1 -p 9000 -v 65536 -b 524288 -k 2097152 -K 256 -t 16 --acl any -L stdout -l notice >/dev/null
  sleep 5
}

priskv_replica() {  # $1=name $2=gpu $3=port $4=scport
  docker run -d --name "$1" --gpus "\"device=$2\"" --network host --ipc host \
    -v "$MODEL":"$MODEL":ro \
    -e VLLM_AIBRIX_SIDE_CHANNEL_PORT="$4" \
    -e AIBRIX_KV_CACHE_OL_BLOCK_SIZE=64 -e AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED=0 \
    -e AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND=PRIS \
    -e AIBRIX_KV_CACHE_OL_PRIS_REMOTE_ADDR=127.0.0.1 -e AIBRIX_KV_CACHE_OL_PRIS_REMOTE_PORT=16379 \
    -e AIBRIX_KV_CACHE_OL_PRIS_PASSWORD=kvcache-redis \
    -e PRISKV_TRANSPORT=ucx -e UCX_TLS=tcp -e PRISKV_USE_SHM=n \
    --entrypoint python3 "$IMG_PRISKV" \
    -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port "$3" \
    --model "$MODEL" --served-model-name "$SERVED" --tensor-parallel-size 1 \
    --max-model-len "$MAXLEN" --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"AIBrixOffloadingConnectorV1Type3","kv_role":"kv_both"}' \
    --disable-log-requests >/dev/null
}

rr_router() {  # round-robin over r0/r1
  docker run -d --name router --network host nginx:1.27-alpine sh -c '
cat > /etc/nginx/nginx.conf <<EOF
events {}
http {
  upstream b { server 127.0.0.1:18001; server 127.0.0.1:18002; }
  server { listen 8080; location / { proxy_pass http://b; proxy_buffering off; proxy_read_timeout 300s; } }
}
EOF
nginx -g "daemon off;"' >/dev/null
}

prefix_router() {  # prefix-aware: pin by X-Prefix-Key header
  docker run -d --name router --network host nginx:1.27-alpine sh -c '
cat > /etc/nginx/nginx.conf <<EOF
events {}
http {
  upstream r0 { server 127.0.0.1:18001; }
  upstream r1 { server 127.0.0.1:18002; }
  split_clients "\$http_x_prefix_key" \$pool { 50% r0; * r1; }
  server { listen 8080; location / { proxy_pass http://\$pool; proxy_buffering off; proxy_read_timeout 300s; } }
}
EOF
nginx -g "daemon off;"' >/dev/null
}

case "${1:-}" in
  A) teardown; stock_replica vllm-r0 0 18001; stock_replica vllm-r1 1 18002
     wait_health 18001 r0; wait_health 18002 r1; rr_router; echo "ARM A UP" ;;
  B) teardown; stock_replica vllm-r0 0 18001; stock_replica vllm-r1 1 18002
     wait_health 18001 r0; wait_health 18002 r1; prefix_router; echo "ARM B UP" ;;
  C) teardown; start_priskv_infra; priskv_replica vllm-r0 0 18001 6667; priskv_replica vllm-r1 1 18002 6677
     wait_health 18001 r0; wait_health 18002 r1; rr_router; echo "ARM C UP" ;;
  down) teardown; echo "DOWN" ;;
  *) echo "usage: $0 {A|B|C|down}"; exit 1 ;;
esac
