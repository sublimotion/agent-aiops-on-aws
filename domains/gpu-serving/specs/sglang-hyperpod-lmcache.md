# SGLang on SageMaker HyperPod — Managed L2 via LMCache IP Mode Recipe Spec

## Status: VALIDATED — PASS (2026-07-10)

Third and final framework in the "HyperPod ai-toolkit managed L2 across all engines" set, alongside
`dynamo-hyperpod-lmcache` (NVIDIA Dynamo/vLLM) and `llmd-hyperpod-lmcache` (llm-d/vLLM). Proves an
SGLang server — not the HyperPod Inference Operator — uses the ai-toolkit managed L2 tiered-storage
daemon (`sagemaker-hyperpod://<node-ip>:9200`) through LMCache's **IP (in-process) mode**.

Deployed on the reused cluster (us-west-2, EKS `qn-sglang-eks-cluster`, K8s 1.32, `ml.g6e.xlarge`).
store→restart→replay on a cold fresh pod returned **cached_tokens=768** (`LMCache retrieve started:
lookup=768, retrieve 768 new tokens`); store side `Stored 934/934 tokens, 7.9 GB/s`. Blueprint:
`domains/gpu-serving/blueprints/sglang-hyperpod-lmcache/` (see `results/findings.md` + telemetry).

## Overview

SGLang reaches the ai-toolkit L2 daemon through **LMCache**, the documented "alternative to HiCache" —
NOT through SGLang HiCache. HiCache's storage backends (`--hicache-storage-backend {file,mooncake,hf3fs,
nixl,aibrix}`) are a separate stack that does not speak the `sagemaker-hyperpod://` / `ai_toolkit_cache`
shared-memory protocol. The `SageMakerHyperPodConnectorAdapter` lives in LMCache core and is
engine-agnostic — the same adapter the vLLM `LMCacheConnectorV1` path uses.

```bash
# Runtime bootstrap in a stock SGLang image (no custom build):
pip install lmcache                                    # 0.5.1, CUDA-13 PyPI wheel matches stock SGLang
sed -i 's/self._mode = LMCacheMode.MP/self._mode = LMCacheMode.IP/' <lmc_radix_cache.py>
python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
  --enable-lmcache --lmcache-config-file /tmp/lmcache.yaml   # config has remote_url + use_layerwise
```

## Integration Thesis

- **The connector is engine-agnostic.** lmcache 0.5.1 ships `lmcache.integration.sglang` AND
  `lmcache.integration.vllm` plus the `SageMakerHyperPodConnectorAdapter`. Any engine whose LMCache
  integration forwards `remote_url` reaches the same ai-toolkit L2 segment.
- **SGLang has two LMCache transport modes** (`mem_cache/storage/lmcache/lmc_radix_cache.py`):
  - **MP (default, line 139):** SGLang dials a *standalone* `lmcache server` daemon over ZMQ and reads
    ONLY `mp_host`/`mp_port` — it does **not** forward `remote_url`. Reaching ai-toolkit this way would
    require the standalone daemon itself to hold the `sagemaker-hyperpod://` connection.
  - **IP (in-process):** `LMCacheLayerwiseConnector(config_file=<full yaml>)` forwards the ENTIRE config,
    so `remote_url: sagemaker-hyperpod://<node-ip>:9200` reaches the adapter — the direct analogue of the
    vLLM data path. **This spec uses IP mode.**
- Selecting IP mode is a one-line source patch (line 138 comment literally says "set `self._mode =
  LMCacheMode.IP` here"). This is the one unavoidable code touch; everything else is config.

## Components

### 1. Compute

Identical to `llmd-hyperpod-lmcache`. Reuse the existing HyperPod cluster with tiered storage enabled
and the ai-toolkit daemon running on the GPU node. `ml.g6e.xlarge` (1 GPU, sm_89) validated; the
store→restart→replay proof is single-replica so 1 GPU is enough. Namespace `sglang-hp-lmcache`.

### 2. Model

`Qwen/Qwen3-0.6B` — same as the other two blueprints for a direct cross-engine comparison. Non-gated,
standard attention, TP=1, `--context-length 8192`. A connector/managed-component recipe, not a benchmark.

### 3. Serving Stack

| Component | Requirement |
|---|---|
| Image | **Stock `lmsysorg/sglang:latest`** (Python 3.12, CUDA 13). No custom build. |
| LMCache | `pip install lmcache` at container start → **0.5.1**. The default PyPI wheel is the **CUDA-13** build, so `import lmcache.c_ops` loads against the stock image's CUDA 13. (Do NOT graft a CUDA-12 lmcache tree from another image — that was the false "CUDA mismatch" blocker.) `pip install lmcache` has no wheel for Python 3.13/3.14 — the stock SGLang 3.12 is required. |
| Mode patch | `sed -i 's/self._mode = LMCacheMode.MP/self._mode = LMCacheMode.IP/'` on the resolved `lmc_radix_cache.py`. Verify the line changed before launch. |
| Server flags | `--enable-lmcache --lmcache-config-file /tmp/lmcache.yaml` (SGLang has NO `--kv-transfer-config` — all LMCache config, including the shm name, comes from this file). |
| Launch | `python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 --mem-fraction-static 0.70 --context-length 8192 --enable-lmcache --lmcache-config-file /tmp/lmcache.yaml` |

LMCache config file (rendered at start with the node IP; SGLang has no `$(NODE_IP)` env substitution in
the yaml, so `sed` the placeholder from `$NODE_IP`):

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 4
save_unfull_chunk: true          # deterministic cross-restart L2 hit
use_layerwise: true              # REQUIRED — see below
remote_url: "sagemaker-hyperpod://<NODE_IP>:9200"
remote_serde: "naive"
extra_config:
  sagemaker_hyperpod_shared_memory_name: ai_toolkit_cache
```

**`use_layerwise: true` is mandatory for SGLang IP mode.** SGLang's IP `store_layer` path calls
`assert_layerwise_gpu_connector(self.gpu_connector)` (`lmcache/v1/cache_engine.py`). Without
`use_layerwise: true`, `CreateGPUConnector` builds a non-layerwise connector and EVERY store fails with
`AssertionError: LMCache store_kv failed` — the connector connects and serving works, but nothing
persists to L2. With it, LMCache builds `SGLangLayerwiseGPUConnector` and stores succeed.

Carry ALL the vLLM-path gotchas (see `llmd-hyperpod-lmcache` §3): `type: File` shm mount of
`/dev/shm/ai_toolkit_cache` (NOT the whole `/dev/shm`), `PYTHONHASHSEED=0`, uid-1000 (the segment is
`0600 uid 1000`; stock SGLang runs as root so it can read it — verify), daemon-recreate recovery
sequence, `dnsPolicy: Default` for HF pulls.

### 4/5. FSx L3 / Observability

Optional, as in the sibling specs. SGLang exposes `/health`, `/generate`, `/v1/chat/completions`;
LMCache store/retrieve evidence is in the server logs (`Stored N/N tokens`, `LMCache retrieve started:
lookup=N`). The e2e probe writes a telemetry artifact under `results/`.

## Validation Stages

- **Stage 0 — Carryover audit.** Read `llmd-hyperpod-lmcache` + `dynamo-hyperpod-lmcache` lessons and
  this blueprint's `results/findings.md`. Carry forward the shm `type:File` mount, daemon-recreate
  sequence, PYTHONHASHSEED, save_unfull_chunk. Note the two SGLang-only requirements: MP→IP patch and
  `use_layerwise: true`.
- **Stage 1 — HyperPod discovery.** Same as siblings: tiered storage enabled, ai-toolkit daemon on the
  GPU node, `/dev/shm/ai_toolkit_cache` present. The `type:File` mount fails to schedule if the segment
  is missing (a prior Directory-mounted or terminating client may have destroyed it). Recovery sequence:
  (1) `kubectl scale deploy/<any-other-client> --replicas=0` to detach all clients;
  (2) `kubectl delete pod -n aws-hyperpod -l app=ai-toolkit` — the daemon's `setup` init container
  recreates the 1 GiB segment; wait for Ready;
  (3) deploy SGLang. Verify the segment survives the SGLang pod's own restart (it will, via `type:File`).
- **Stage 2 — SGLang baseline (L0).** Deploy stock SGLang + `pip install lmcache` + MP→IP patch, serve
  Qwen3-0.6B, `/health` 200, `/generate` returns. Gate: serving works before trusting L2.
- **Stage 3 — LMCache IP mode → HyperPod L2.** Confirm startup logs show `Creating SageMaker HyperPod
  connector`, `Shared memory opened: ai_toolkit_cache (1024.00 MB)`, `Connection initialized`. Send a
  long prefix; confirm `Stored N/N tokens` (NOT `store_kv failed` — if it fails, `use_layerwise` is
  missing). Gate: a real store lands.
- **Stage 4 — Store → restart → replay L2 hit proof (core deliverable).** Store a ≥512-token
  deterministic prefix via `/generate`; `kubectl rollout restart` the SGLang pod (wipes L0/L1; the
  `type:File` mount keeps the daemon segment alive); replay the identical prefix on the fresh pod.
  PASS = non-zero `cached_tokens` + `LMCache retrieve started: lookup=N` on the cold pod. Write the
  telemetry artifact.
- **Stage 5 — (optional) two-replica same-node sharing.** Skip on a single-GPU SKU with reason.

## Success Criteria

| Criteria | Stage | Type |
|---|---|---|
| Stock SGLang + `pip install lmcache` gives a working CUDA-matched lmcache (`c_ops` loads) | 2 | Critical |
| MP→IP patch applied; IP connector built | 2 | Critical |
| `use_layerwise: true` set; stores succeed (no `store_kv failed` assertion) | 3 | Critical |
| LMCache IP mode opens `ai_toolkit_cache` and connects to `sagemaker-hyperpod://<ip>:9200` | 3 | Critical |
| store → restart → replay proves a cold-pod L2 hit (`cached_tokens>0`) | 4 | Critical |
| Telemetry artifact saved | 4 | Critical |
| Two-replica sharing works or skipped w/ reason | 5 | Important |

## Known Risks and Blockers

| Risk | Severity | Detail | Mitigation |
|---|---|---|---|
| Missing `use_layerwise: true` | **CRITICAL** | SGLang IP `store_layer` asserts a layerwise GPU connector; without it every store fails `AssertionError` while serving still "works". | Set `use_layerwise: true` in the lmcache config. |
| Wrong mode (MP not patched to IP) | High | MP mode reads only mp_host/mp_port, never forwards `remote_url` → never touches ai-toolkit L2. | `sed` line 139 MP→IP; verify the changed line before launch. |
| Grafting a mismatched-CUDA lmcache | High | Copying another image's compiled lmcache (`c_ops.so`) fails `libcudart.so.X` if CUDA versions differ. | Don't graft — `pip install lmcache` fresh so `c_ops` matches the image's CUDA. |
| Wrong Python | Medium | `pip install lmcache` has no wheel for Python 3.13/3.14. | Use stock SGLang (Python 3.12). |
| Whole `/dev/shm` (Directory) mount | **CRITICAL** | Same as vLLM path: a terminating client `shm_unlink`s the daemon segment → all clients poisoned. | `type: File` mount of `/dev/shm/ai_toolkit_cache`; daemon-recreate recovery. |
| shm name only settable via config | Medium | SGLang has no `--kv-transfer-config`; the default shm name (`shared_memory`) won't match ai-toolkit. | Set `extra_config.sagemaker_hyperpod_shared_memory_name: ai_toolkit_cache` in the config file. |

## Non-Requirements

- Production throughput/latency benchmarking (recipe + smoke proof).
- HiCache / Mooncake / 3FS / NIXL backends (separate stack; not the ai-toolkit L2 path).
- P/D disaggregation, wide-EP, multi-node.
- Replacing the HyperPod Inference Operator's managed serving path.

## References

- Sibling recipes: `domains/gpu-serving/specs/llmd-hyperpod-lmcache.md`, `.../dynamo-hyperpod-lmcache.md`
- Blueprint: `domains/gpu-serving/blueprints/sglang-hyperpod-lmcache/` (manifest, config, findings, telemetry)
- SGLang LMCache integration: `python/sglang/srt/mem_cache/storage/lmcache/` (upstream)
- LMCache install (CUDA 13 default): `https://docs.lmcache.ai/getting_started/installation.html`
- Shared steering: `.claude/steering/tech-stack.md` §SageMaker HyperPod (shm `type:File` mount, cross-node networking, llm-d restructure)

> **Note**: Operational artifacts belong in `domains/gpu-serving/blueprints/sglang-hyperpod-lmcache/`.
