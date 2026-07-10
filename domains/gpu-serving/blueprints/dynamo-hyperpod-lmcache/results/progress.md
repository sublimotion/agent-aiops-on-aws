# NVIDIA Dynamo HyperPod LMCache — Progress

**Date**: 2026-07-09
**Spec**: `domains/gpu-serving/specs/dynamo-hyperpod-lmcache.md`
**Region plan**: `us-east-2` primary, `us-west-2` fallback
**Preferred hardware**: `ml.g7.2xlarge` if live SageMaker HyperPod accepts it
**Approved fallback**: `ml.g6e.xlarge`
**Model**: `Qwen/Qwen3-0.6B`

## Status

| Stage | Result | Notes |
|---|---|---|
| Stage 0 - Carryover Audit | COMPLETE | Spec written; prior HyperPod L2 lessons reviewed; mdc/gpu-infra card gaps recorded. |
| Stage 1 - HyperPod Tiered Storage Discovery | COMPLETE | `us-west-2` cluster is `InService`; `g6e-workers` and `system-nodes` are `InService`; tiered storage is enabled; ai-toolkit daemon is running on the GPU node. |
| Stage 2 - Baseline Worker | COMPLETE | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1` serves `Qwen/Qwen3-0.6B` through raw vLLM OpenAI API on the HyperPod G6e node. |
| Stage 3 - LMCache Connector to HyperPod L2 | PASS | LMCache loaded `/etc/lmcache/config.yaml`, opened `ai_toolkit_cache`, and connected to `sagemaker-hyperpod://10.2.37.31:9200`. |
| Stage 4 - Dynamo Graph Participation | PASS | `python3 -m dynamo.frontend` serves port 8000; frontend health lists `dyn://dynamo-hp-lmcache.backend.generate`; traffic routes to `python3 -m dynamo.vllm` over Dynamo TCP request plane. |
| Stage 5 - E2E Telemetry | COMPLETE | Artifacts include raw worker telemetry plus `results/e2e-telemetry-dynamo-frontend-short-store-20260709.json` and `results/e2e-telemetry-dynamo-frontend-short-replay-20260709.json`. |
| Stage 6 - L2 Retrieval Gate | PASS | Dynamo frontend store/restart/replay produced `external_prefix_cache_hits_total=1102`, `lmcache:num_hit_tokens_total=1103`, and log `Retrieved 1103 out of 1103 required tokens`. |
| Stage 7 - Two-Replica Same-Node Sharing | PENDING | Skip if selected SKU only has one usable GPU. |

## Current Decisions

- Used `us-west-2` after the `us-east-2` cluster was blocked in deleting/recreate flow.
- Used `ml.g6e.xlarge` fallback because the current SageMaker HyperPod CLI model does not include `ml.g7.*` in the allowed instance type enum.
- The runnable manifest now uses the Dynamo frontend plus a Dynamo vLLM worker in one pod with file discovery. Dynamo frontend L2 replay passes with the shorter deterministic 1103-token prompt.
- `dnsPolicy: Default` is required for this HyperPod pod to resolve Hugging Face through the node resolver.
- `LMCACHE_CONFIG_FILE` is required in addition to `kv_connector_extra_config`; otherwise the SageMaker HyperPod adapter defaults to shared memory name `shared_memory`.
- `PYTHONHASHSEED=0` is required for stable LMCache keys across vLLM pod restarts; without it, L2 replay did not hit.
- `save_unfull_chunk: true` was enabled for the forced retrieval test.

## Next Actions

1. Convert the one-pod file-discovery recipe to `DynamoGraphDeployment` once the Dynamo Operator CRDs are installed on HyperPod.
2. Decide whether to keep the long 2093-token miss artifact as a known edge case or remove it from the published example.
3. Investigate whether ai-toolkit exposes non-obvious introspection APIs for cache contents; common read-only HTTP paths on `:9200` returned 404.
