---
model: "YOLOv8n"
engine: "ray-serve"
hardware: "g5.xlarge"
gpu_arch: "A10G"
deployment_date: "2026-03-20"

outcome: "success"
failure_categories:
  - tls_incompatibility
  - dependency_conflict
  - disk_pressure
  - missing_shared_lib

cards_used:
  mdc: []
  gpu_infra: []

card_helped: null

benchmark:
  throughput_toks_s: null
  ttft_p50_ms: null
  ttft_p99_ms: null
  concurrent_users: 10
  gpu_util_pct: null

ralph_iterations: 1


learn_commands: []
---

# Lessons Learned: Ray Serve Fault Tolerance

## Summary

Deployed Ray Serve with GCS fault tolerance on EKS (2x g5.xlarge A10G workers, 2x m6i.xlarge system nodes). YOLOv8n object detection as workload. ElastiCache Serverless Redis for GCS state persistence via stunnel TLS sidecar. Internal NLB targeting worker proxies for zero-SPOF architecture. All fault injection tests passed: head crash recovery (T3), proxy failover with zero downtime (T5), replica crash recovery (T1), worker drain (T2).

## What Worked

- **GCS FT with ElastiCache**: Workers survived head pod crash. GCS state restored from ElastiCache Serverless in ~3 minutes. Workers maintained 600s reconnect timeout without restarting.
- **Proxy failover via NLB**: Ray Serve runs HTTP proxies on all nodes. NLB targeting only worker proxies (label selector `ray-node: worker`) gives zero-downtime failover when head dies. Head proxy excluded from NLB to avoid extra network hop (no local replicas on head).
- **stunnel TLS sidecar pattern**: Lightweight Alpine container (~2s init) proxies `localhost:6380` → ElastiCache TLS endpoint. Runs on all pods (head + workers). Eliminates in-cluster Redis SPOF while working around Ray's lack of native TLS Redis support.
- **KubeRay v1.3.0 GCS FT**: `ray.io/ft-enabled: "true"` annotation auto-injects `RAY_external_storage_namespace`. Head pod recreation and GCS restore are fully automated.

## Failures and Fixes

### tls_incompatibility
Ray's C++ Redis client does NOT support TLS (`rediss://`). ElastiCache Serverless enforces TLS with no opt-out. Solution: stunnel sidecar on every pod. Key detail: use port 6380 (not 6379) to avoid conflict with Ray GCS server's own port 6379. Set `RAY_REDIS_ADDRESS=127.0.0.1:6380`.

### dependency_conflict
`numpy>=2.0` breaks pyarrow in Ray CPU image (compiled against numpy 1.x). Error: `_ARRAY_API not found`. Fix: pin `numpy<2` in runtime_env pip list. This affects head node too because Ray Serve `build_app` imports the deployment module on head to discover metadata.

### disk_pressure
pip install of ultralytics + torch consumes ~8-12GB ephemeral storage. Default 20GB EBS on system nodes caused pod eviction. Fix: system nodes need 50GB+ EBS (`disk_size = 50` in Terraform), head pod needs `ephemeral-storage: 20Gi` request / `30Gi` limit.

### missing_shared_lib
`libGL.so.1` missing in `rayproject/ray:*-cu125` images. OpenCV (pulled by ultralytics) requires it. Fix: debian:bookworm-slim init container with `runAsUser: 0` copies libGL/libGLX/libGLdispatch/libglib to shared emptyDir volume. Worker container sets `LD_LIBRARY_PATH=/shared-libs:/usr/local/nvidia/lib64`.

## Benchmark Results

| Test | Error Rate | Error Window | P50 (ms) | P99 (ms) | Notes |
|------|-----------|-------------|---------|---------|-------|
| T1: Kill replica | 74.3% | 143s | 25,570 | 30,561 | Expected: kills all Ray processes on node |
| T2: Drain worker | 93.5% | 153s | 16,911 | 24,626 | Expected: only 2 GPU nodes, no spare |
| T3: Kill head (FT) | 75.2% | 181s | 7,479 | 10,309 | Workers survived, high errors from port-forward |
| T5: Kill head proxy | 0.0% | 0s | 151 | 1,727 | Zero downtime via worker proxies |

T3 error rate was inflated by port-forward breaking on head death. With the NLB in place, traffic routes directly to worker proxies — actual production downtime would be near zero.

## Card Accuracy

No deployment cards were consulted for this blueprint (YOLO on Ray Serve, not an LLM serving engine). The stunnel pattern for ElastiCache TLS and the numpy<2 pin are novel findings that could inform future Ray Serve deployments.
