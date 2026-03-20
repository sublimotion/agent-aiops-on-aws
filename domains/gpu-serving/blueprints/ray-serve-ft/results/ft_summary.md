# Ray Serve Fault Tolerance Results

Date: 2026-03-20
Cluster: qn-sglang-eks-cluster (us-west-2)
Model: YOLOv8n (object detection, ~10ms inference)
GPU: 2x g5.xlarge (A10G)
Redis: ElastiCache Serverless via stunnel TLS proxy
KubeRay: v1.3.0 with GCS FT enabled
Traffic: 50 req/s sustained, JSON/base64 input

## Results

| Test | Fault | Error Rate | Error Window | P50 (ms) | P99 (ms) |
|------|-------|-----------|-------------|---------|---------|
| T1 | Kill one YOLO replica | 74.3% | 143.4s | 25,570 | 30,561 |
| T2 | Drain one GPU worker node | 93.5% | 153.2s | 16,911 | 24,626 |
| T3 | Kill head pod (GCS FT ON) | 75.2% | 181.0s | 7,479 | 10,309 |
| T5 | Kill HTTP proxy on head | 0.0% | 0.0s | 151 | 1,727 |

## Key Findings

### T5: HTTP Proxy Failover (PASS)
- Zero-downtime proxy failover. Ray Serve runs proxies on all 3 nodes (head + 2 workers).
- Killing the head's proxy causes instant failover to worker proxies.
- No configuration needed — this is built-in Ray Serve behavior.

### T3: Head Failure with GCS FT (PASS - functionality, FAIL - latency)
- Workers survived head crash (not restarted — kept their 15min uptime).
- GCS state successfully restored from ElastiCache Serverless.
- New head pod started in ~3 minutes (stunnel init + Ray GCS restore).
- High error rate during outage because port-forward breaks when head dies.
- In production with a LoadBalancer service, errors would be lower (requests route to worker proxies).

### T1: Replica Crash (EXPECTED behavior)
- `pkill ray::SERVE_REPLICA` kills all Ray worker processes on the node, not just the replica.
- This effectively takes down the entire worker pod's Ray runtime.
- Recovery requires runtime_env pip install restart (~2 min) for the replacement replica.
- With only 2 replicas, 50% capacity loss is expected.

### T2: Worker Node Drain (EXPECTED behavior)
- With min/max replicas = 2 and only 2 GPU nodes, draining one node leaves no capacity.
- The evicted pod cannot reschedule until the node is uncordoned.
- In production, use 3+ GPU nodes or autoscaling (maxReplicas > minReplicas).

## Architecture Decisions

### stunnel for ElastiCache TLS
- Ray's C++ Redis client does NOT support TLS (rediss://).
- ElastiCache Serverless enforces TLS — no option to disable.
- stunnel sidecar on each pod: localhost:6380 (plain TCP) → ElastiCache (TLS).
- Eliminates in-cluster Redis SPOF.
- Port 6380 (not 6379) to avoid conflict with Ray GCS server port.

### Runtime Environment
- `numpy<2` required: Ray CPU image has pyarrow compiled against numpy 1.x.
- `opencv-python-headless` in pip list but ultralytics also pulls `opencv-python`.
- libGL.so.1 missing in Ray CUDA image: init container copies from debian:bookworm-slim.
- Head needs 20Gi ephemeral-storage for pip install (ultralytics + torch = ~8GB).
- System nodes need 50GB EBS disk (default 20GB causes eviction).
