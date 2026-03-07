# GLM-5 llm-d — Multi-Replica Prefix-Cache-Aware Routing Spec

## Status: SINGLE-REPLICA VALIDATED / EXT-PROC PARTIAL (2026-03-07)

## Overview

Multi-replica GLM-5-FP8 on EKS with llm-d inference scheduler for prefix-cache-aware routing. Uses **vLLM** (not SGLang) because llm-d's KV cache tracking (KVEvents) and EPP scoring require vLLM-specific integrations.

Pivoted from HyperPod (`glm5-hyperpod.md`) because Training Plans are unavailable (account not allowlisted). Uses vanilla EKS with capacity blocks instead.

**Model**: GLM-5 by Zhipu AI — 744B MoE (256 routed + 1 shared expert, top-8, ~40B active), MLA + DSA attention. FP8 variant ~733 GB on disk.
**Model ID**: `zai-org/GLM-5-FP8`
**vLLM recipe**: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html

## Architecture

```
                    ┌─────────────┐
                    │   Envoy     │
                    │  Gateway    │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │  llm-d EPP  │ ← PrecisePrefixCacheScorer
                    │  (ext-proc) │ ← LoadAwareScorer
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐ ┌───┴───┐ ┌─────┴─────┐
        │ vLLM + LC │ │ vLLM  │ │ vLLM + LC │
        │ Replica 0 │ │ Rep 1 │ │ Replica N │
        └─────┬─────┘ └───┬───┘ └─────┬─────┘
              │            │            │
        ┌─────┴────────────┴────────────┴─────┐
        │           Redis (L2 cache)          │
        │        FSx Lustre (L2 disk)         │
        └─────────────────────────────────────┘
```

## Key Decisions

1. **vLLM with official GLM-5 image** — Use `vllm/vllm-openai:glm5` (pre-built with DeepGEMM, GLM-5 model architecture, and tool-call parser). llm-d requires vLLM's KVEvents protocol for cache-aware routing; SGLang is incompatible with llm-d's EPP.
2. **Tool calling via `--tool-call-parser glm47`** — vLLM's official GLM-5 recipe provides structured OpenAI-compatible tool calls. Combined with `--enable-auto-tool-choice` for coding agent workloads.
3. **Reasoning mode via `--reasoning-parser glm45`** — Thinking mode enabled by default. Disable per-request with `"chat_template_kwargs": {"enable_thinking": false}`.
4. **MTP speculative decoding** — `--speculative-config.method mtp --speculative-config.num_speculative_tokens 1` for faster generation.
5. **LMCache via KVConnector** — vLLM's `LMCacheConnectorV1` provides L1 (CPU) + L2 (FSx/Redis) tiered offloading. Note: LMCache NSA/MLA compatibility with vLLM's GLM-5 implementation needs validation (SGLang's implementation uses fused `kv_buffer` which breaks LMCache — vLLM may differ). See `glm5-lmcache/lessons.md` #10.
6. **llm-d EPP for routing** — PrecisePrefixCacheScorer routes requests to pods with cached prefixes. LoadAwareScorer balances load.
7. **Redis for cross-replica L2** — enables KV cache sharing across vLLM replicas.
8. **Gateway API InferencePool CRD** — standard K8s-native inference abstraction.

## Serving Config

| Parameter | Value |
|-----------|-------|
| Engine | vLLM (`vllm/vllm-openai:glm5`) |
| TP | 8 |
| Max Model Length | 131072 |
| GPU Memory Utilization | 0.85 |
| Swap Space | 32 |
| Prefix Caching | Enabled |
| Tool Call Parser | glm47 |
| Reasoning Parser | glm45 |
| Speculative Decoding | MTP (1 speculative token) |
| KV Connector | LMCacheConnectorV1 |
| Port | 8000 |

## Reference Benchmark (vLLM docs, H200x8)

| Workload | Output Throughput | Total Throughput | Mean TTFT | Mean ITL |
|----------|:-----------------:|:----------------:|:---------:|:--------:|
| 8K input / 1K output, 32 prompts | 459 tok/s | 4,041 tok/s | 13.5s | 54.5ms |

Note: B200x8 (this blueprint's target) should outperform H200x8. SGLang HiCache on B200x8 achieved 2,602 tok/s output at 128 concurrent (see `glm5-lmcache` results).

## SGLang Alternative (Tool Calling)

SGLang patched images with GLM-5 tool calling support (PR #19925):
- `lmsysorg/sglang:glm5-blackwell-patched` (B200)
- `lmsysorg/sglang:glm5-hopper-patched` (H200)

These parse GLM-5's native XML tool-call format into OpenAI-compatible `tool_calls`. If llm-d adds SGLang backend support, SGLang + HiCache would be preferred (2.86x throughput improvement proven in glm5-lmcache Phase 1).

## Risk: LMCache + NSA/MLA Compatibility

LMCache v0.3.15 is incompatible with SGLang's NSA/MLA attention (`kv_buffer` vs `k_buffer`/`v_buffer`). vLLM's GLM-5 implementation may use a different KV cache layout. **Must validate LMCache compatibility on vLLM before deploying with `--kv-transfer-config`**. Fallback: run without LMCache (prefix caching only + llm-d routing).

## Infrastructure

- **Instances**: 2+ p6-b200.48xlarge (8x B200 183GB, NVSwitch NVL5+) or p5e.48xlarge (8x H200 141GB)
- **AMI**: amazon-eks-node-al2023-x86_64-nvidia (B200 requires AL2023 for Fabric Manager)
- **Storage**: FSx Lustre PERSISTENT_2 4800 GiB @ 500 MB/s/TiB
- **Cache**: Redis on system node (75 GB, allkeys-lfu)
- **Routing**: llm-d EPP via Envoy Gateway
- **Region**: us-east-2

## Blueprint

`domains/gpu-serving/blueprints/glm5-llmd/`
