---
model: "Qwen/Qwen3-0.6B"
engine: "sglang"
hardware: "ml.g6e.xlarge"
gpu_arch: "sm_89"
deployment_date: "2026-07-10"

outcome: "success"
failure_categories:
  - "lmcache_sglang_use_layerwise_required"
  - "lmcache_sglang_mp_vs_ip_mode"
  - "lmcache_cuda_wheel_mismatch_on_graft"

cards_used:
  mdc: []
  gpu_infra: []

card_helped: null

benchmark:
  throughput_toks_s: null
  ttft_p50_ms: null
  ttft_p99_ms: null
  concurrent_users: null
  gpu_util_pct: null

ralph_iterations: null


learn_commands:
  - 'mdc learn "Qwen/Qwen3-0.6B" sglang "SGLang -> HyperPod ai-toolkit L2 via LMCache IP mode (NOT HiCache). Stock lmsysorg/sglang (Py3.12/CUDA13) + pip install lmcache (0.5.1, CUDA-13 wheel is PyPI default, no build). Patch lmc_radix_cache.py self._mode MP->IP (MP only forwards mp_host/mp_port, not remote_url). REQUIRE use_layerwise:true in lmcache config or IP store_layer fails assert_layerwise_gpu_connector -> store_kv AssertionError. shm name only settable via config extra_config (no --kv-transfer-config flag). type:File shm mount, PYTHONHASHSEED=0, save_unfull_chunk. Proof: store 934/934 -> restart -> replay cached_tokens=768."'
  - 'gpu-infra learn -c platform "SGLang on HyperPod ai-toolkit L2: reuses the same SageMakerHyperPodConnectorAdapter + type:File /dev/shm/ai_toolkit_cache mount as the vLLM engines. SGLang-specific: needs LMCache IP mode (MP->IP source patch) + use_layerwise:true; MP mode dials a standalone lmcache server and never forwards remote_url."'
---

# SGLang HyperPod LMCache — Lessons

## Outcome

**PASS (2026-07-10)** — 3rd of 3 frameworks proven on HyperPod ai-toolkit managed L2 (with Dynamo/vLLM
and llm-d/vLLM). store→restart→replay on a cold fresh pod: `cached_tokens=768`,
`LMCache retrieve started: lookup=768, retrieve 768 new tokens`; store `Stored 934/934 tokens, 7.9 GB/s`.

## Carried Forward (shared with the vLLM recipes — do not rediscover)

- **`type: File` shm mount** of `/dev/shm/ai_toolkit_cache`, NOT the whole `/dev/shm`. Directory mount →
  terminating client `shm_unlink` destroys the daemon segment → all clients poisoned. See
  `.claude/steering/tech-stack.md` §SageMaker HyperPod.
- **Daemon-recreate recovery**: scale clients to 0 → delete the ai-toolkit pod → scale back, if the
  segment is missing (the `type:File` mount fails to schedule without it).
- `PYTHONHASHSEED=0` + `save_unfull_chunk: true` for deterministic cross-restart hits.
- shm name mismatch: default `shared_memory` vs ai-toolkit's `ai_toolkit_cache`.
- `dnsPolicy: Default` for HF pulls on the HyperPod node.

## SGLang-specific lessons (NEW)

### 1. Reach ai-toolkit L2 via LMCache, not HiCache
`--enable-lmcache` (the "alternative to HiCache") is the path. `--hicache-storage-backend`
(mooncake/hf3fs/nixl/aibrix/file) is a separate stack that cannot speak the `sagemaker-hyperpod://` /
`ai_toolkit_cache` protocol.

### 2. MP → IP mode is a required one-line source patch
`python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py` line 139 hardcodes
`self._mode = LMCacheMode.MP`. MP dials a standalone `lmcache server` over ZMQ and reads ONLY
`mp_host`/`mp_port` — it never forwards `remote_url`, so it can't reach ai-toolkit directly. IP mode
(`LMCacheLayerwiseConnector(config_file=cli_lmc_cfg)`, line 154) forwards the FULL config. Patch:
`sed -i 's/self._mode = LMCacheMode.MP/self._mode = LMCacheMode.IP/' <file>`. (Line 138 comment literally
says "set `self._mode = LMCacheMode.IP` here".)

### 3. `use_layerwise: true` is mandatory (the subtle one)
SGLang IP `store_layer` calls `assert_layerwise_gpu_connector(self.gpu_connector)`
(`lmcache/v1/cache_engine.py`). With the default `use_layerwise: false`, `CreateGPUConnector` builds a
non-layerwise connector and EVERY store fails `AssertionError: LMCache store_kv failed` — the connector
connects and serving works, so it looks healthy while nothing persists. `use_layerwise: true` builds
`SGLangLayerwiseGPUConnector` and stores succeed (`store_kv completed: stored N tokens`).

### 4. No custom image needed — pip-install lmcache fresh (CUDA story corrected)
Stock `lmsysorg/sglang:latest` = Python 3.12 / **CUDA 13** / no lmcache. `pip install lmcache` →
**0.5.1**, whose default PyPI wheel is the **CUDA-13** build, so `import lmcache.c_ops` loads cleanly.
An earlier pass wrongly concluded "blocked on CUDA 12 vs 13" — that came from GRAFTING the dynamo image's
CUDA-12-compiled lmcache tree into the CUDA-13 SGLang image (`ImportError: libcudart.so.12`). Installing
fresh pulls the matching wheel. (`pip install lmcache` has no wheel for Python 3.13/3.14 — stock SGLang's
3.12 is required.)

### 5. shm name only settable via the lmcache config file
SGLang has NO `--kv-transfer-config` flag (unlike vLLM), so
`extra_config.sagemaker_hyperpod_shared_memory_name: ai_toolkit_cache` MUST be in the
`--lmcache-config-file` yaml, or the adapter defaults to `shared_memory` and never connects.
