# SGLang on HyperPod ai-toolkit L2 — findings

Cross-framework completeness test: can SGLang use the same HyperPod ai-toolkit managed L2 tiered-storage
daemon (`sagemaker-hyperpod://<node-ip>:9200`) that the vLLM-based `dynamo-hyperpod-lmcache` and
`llmd-hyperpod-lmcache` blueprints proved? Reuses the same us-west-2 cluster.

## Verdict (2026-07-10): PASS — 3rd of 3 frameworks proven

SGLang stores to and reads from the HyperPod ai-toolkit managed L2 daemon via LMCache IP mode.
store→restart→replay on a fresh (cold-L0/L1) pod returned **cached_tokens=768** with
`LMCache retrieve started: lookup=768 ... retrieve 768 new tokens` — the KV can only have come from
the managed L2 segment. Store side: `Stored 934/934 tokens, 7.9 GB/s`. Artifact:
`results/e2e-telemetry-sglang-l2-proof-20260710.json`.

## How it works (the recipe)

- **No custom image build.** Stock `lmsysorg/sglang:latest` (Python 3.12, CUDA 13) + runtime
  `pip install lmcache` → lmcache **0.5.1**, whose default PyPI wheel is the **CUDA-13** build, so
  `import lmcache.c_ops` loads cleanly. The image ships the SageMaker adapter AND the sglang integration.
- **LMCache, not HiCache.** ai-toolkit L2 is reached through `--enable-lmcache`
  (LMCache = "alternative to HiCache"), NOT via `--hicache-storage-backend` (mooncake/3fs/nixl — a
  separate stack that can't speak the ai_toolkit_cache protocol).
- **IP mode required.** `lmc_radix_cache.py` line 139 hardcodes `self._mode = LMCacheMode.MP`. MP dials a
  standalone `lmcache server` over ZMQ and reads ONLY `mp_host`/`mp_port` — it does NOT forward
  `remote_url`. **IP mode** (`LMCacheLayerwiseConnector(config_file=...)`) forwards the FULL config, so
  `remote_url: sagemaker-hyperpod://` reaches the same adapter the vLLM path uses. Select it with a
  one-line `sed` patch: `LMCacheMode.MP` → `LMCacheMode.IP`.
- **`use_layerwise: true` required.** SGLang IP `store_layer` calls `assert_layerwise_gpu_connector`;
  without `use_layerwise: true` in the lmcache config, `CreateGPUConnector` builds a non-layerwise
  connector and every store fails with `AssertionError` (`store_kv failed`). Setting it builds
  `SGLangLayerwiseGPUConnector` and stores succeed.
- Carries ALL the vLLM-path gotchas: `type: File` shm mount of `/dev/shm/ai_toolkit_cache` (NOT the
  whole dir), `PYTHONHASHSEED=0`, `save_unfull_chunk: true`, `sagemaker_hyperpod_shared_memory_name:
  ai_toolkit_cache` (here it MUST come from the lmcache config file — SGLang has no
  `--kv-transfer-config` flag), uid-1000, daemon-recreate recovery, `dnsPolicy: Default`.

## Correcting the earlier (wrong) blocker

An earlier pass concluded "blocked on CUDA 12 vs 13". That was wrong: it came from trying to GRAFT the
dynamo image's CUDA-12-compiled lmcache tree into a CUDA-13 SGLang image. Installing lmcache fresh
(`pip install lmcache`) pulls the CUDA-13 wheel that matches the image — no build, no mismatch. The
original `pip install lmcache` "no wheel" failure was a Python 3.14 slim pod, not a CUDA issue.

## Deploy artifacts

- `manifests/sglang-lmcache.yaml` — namespace + ConfigMap (config template) + Service + Deployment.
  The container command does pip-install → MP→IP sed patch → render config with `$NODE_IP` → launch.
- `configs/lmcache-config.yaml` — the lmcache config (IP mode, remote_url, use_layerwise, shm name).
- `results/e2e-telemetry-sglang-l2-proof-20260710.json` — the PASS proof.

## Status

- [x] Adapter + sglang integration present (lmcache 0.5.1 via pip in stock SGLang image)
- [x] c_ops loads (CUDA-13 wheel matches stock SGLang CUDA 13)
- [x] SGLang serves Qwen3-0.6B on the g6e node
- [x] LMCache IP mode connects to ai-toolkit L2 (`Shared memory opened: ai_toolkit_cache`)
- [x] store works (`use_layerwise: true` fix) — `Stored 934/934 tokens`
- [x] store → restart → replay external L2 hit — `cached_tokens=768` on fresh pod. **PASS**

## Publish note (3-framework repo)

All three engines now proven on HyperPod ai-toolkit managed L2, same cluster, same Qwen3-0.6B:
| Framework | Blueprint | Path to L2 | Proof |
|---|---|---|---|
| NVIDIA Dynamo (vLLM) | dynamo-hyperpod-lmcache | vLLM LMCacheConnectorV1 | external hits 1102 |
| llm-d (vLLM) | llmd-hyperpod-lmcache | vLLM LMCacheConnectorV1 | external hits 742, 99.9% |
| SGLang | sglang-hyperpod-lmcache | LMCache IP mode (use_layerwise) | cached_tokens 768 |

Shared cross-cutting rules already in `.claude/steering/tech-stack.md` (type:File shm mount, cross-node
networking, etc.). SGLang adds two engine-specific requirements: MP→IP patch + `use_layerwise: true`.
