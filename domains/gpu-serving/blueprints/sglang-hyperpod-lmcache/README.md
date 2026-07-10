# SGLang HyperPod LMCache Recipe

Serve **SGLang** on SageMaker HyperPod and connect it to the ai-toolkit managed L2 tiered-storage daemon
(`sagemaker-hyperpod://<node-ip>:9200`) via LMCache **IP mode**. Third of three cross-engine HyperPod L2
recipes, alongside [`dynamo-hyperpod-lmcache`](../dynamo-hyperpod-lmcache/) and
[`llmd-hyperpod-lmcache`](../llmd-hyperpod-lmcache/).

Spec: [`domains/gpu-serving/specs/sglang-hyperpod-lmcache.md`](../../specs/sglang-hyperpod-lmcache.md)

## Result Snapshot — PASS (2026-07-10)

- Region: `us-west-2` (reused the shared HyperPod cluster)
- Hardware: `ml.g6e.xlarge` (1 GPU, sm_89)
- Model: `Qwen/Qwen3-0.6B`
- Image: **stock `lmsysorg/sglang:latest`** (Python 3.12, CUDA 13) + runtime `pip install lmcache` (0.5.1) — no custom build
- Namespace: `sglang-hp-lmcache`
- L2 daemon: `sagemaker-hyperpod://10.2.37.31:9200` (shm `ai_toolkit_cache`)
- **Proof**: store `Stored 934/934 tokens, 7.9 GB/s` → restart → replay on cold fresh pod → `cached_tokens=768`, `LMCache retrieve started: lookup=768`. Artifact: `results/e2e-telemetry-sglang-l2-proof-20260710.json`.

## What This Example Proves

- An SGLang server — not the HyperPod Inference Operator — writes to and reads from the HyperPod
  managed L2 daemon via LMCache.
- SGLang reaches L2 through **LMCache IP mode** (`--enable-lmcache`), NOT HiCache/Mooncake.
- The L2 hit survives an SGLang pod restart (L0/L1 wiped), so it can only come from the managed daemon.

## Deploy

```bash
kubectl apply -f manifests/sglang-lmcache.yaml
kubectl rollout status deploy/sglang-lmcache -n sglang-hp-lmcache --timeout=6m
```

The container command (see the manifest) does it all at start: `pip install lmcache` → `sed` patch
`LMCacheMode.MP → IP` in `lmc_radix_cache.py` → render the lmcache config with the node IP → launch
`sglang.launch_server --enable-lmcache --lmcache-config-file`.

> **Prereq**: the ai-toolkit segment `/dev/shm/ai_toolkit_cache` must exist before the `type:File` mount
> can schedule. If missing: scale clients to 0 → `kubectl delete pod -n aws-hyperpod -l app=ai-toolkit`
> (recreates the 1 GiB segment) → deploy.

## Validate + L2 proof

```bash
kubectl port-forward -n sglang-hp-lmcache svc/sglang-lmcache 8100:8000 &
# store a long deterministic prefix
curl -s http://127.0.0.1:8100/generate -H 'Content-Type: application/json' \
  -d '{"text":"<long stable prefix> Answer in one word:","sampling_params":{"temperature":0,"max_new_tokens":8}}'
# -> server log: "Stored N/N tokens"

kubectl rollout restart deploy/sglang-lmcache -n sglang-hp-lmcache
kubectl rollout status  deploy/sglang-lmcache -n sglang-hp-lmcache --timeout=6m
# replay the identical prefix on the fresh pod
curl -s http://127.0.0.1:8100/generate -H 'Content-Type: application/json' -d '<same body>'
# -> meta_info.cached_tokens > 0 ; server log: "LMCache retrieve started: lookup=N"
```

PASS requires non-zero `cached_tokens` (and the `LMCache retrieve started` log) on the post-restart pod —
the only surviving KV source is the ai-toolkit daemon.

## Two SGLang-specific requirements (not in the vLLM recipes)

1. **MP → IP mode patch.** `lmc_radix_cache.py` hardcodes `self._mode = LMCacheMode.MP` (line 139). MP
   dials a standalone `lmcache server` and reads only `mp_host`/`mp_port` — it never forwards
   `remote_url`. IP mode (`LMCacheLayerwiseConnector(config_file=...)`) forwards the full config so
   `remote_url: sagemaker-hyperpod://` reaches the adapter. The manifest `sed`s this at start.
2. **`use_layerwise: true`** in the lmcache config. SGLang IP `store_layer` asserts a layerwise GPU
   connector; without it every store fails `AssertionError: LMCache store_kv failed` while serving still
   works. With it, LMCache builds `SGLangLayerwiseGPUConnector`.

## Key Artifacts

- `manifests/sglang-lmcache.yaml` — namespace + config-template ConfigMap + Service + Deployment (self-bootstrapping).
- `configs/lmcache-config.yaml` — the lmcache config (IP mode, remote_url, `use_layerwise`, shm name).
- `results/findings.md` — full recipe rationale + the corrected CUDA "blocker" story.
- `results/e2e-telemetry-sglang-l2-proof-20260710.json` — the PASS proof.

## Notes (carried from the vLLM recipes)

- `type: File` shm mount of `/dev/shm/ai_toolkit_cache` (NOT the whole dir) — a Directory mount lets a
  terminating client `shm_unlink` the daemon segment and poison all clients.
- `PYTHONHASHSEED=0` + `save_unfull_chunk: true` for deterministic cross-restart hits.
- shm name MUST be set via the lmcache config `extra_config` (SGLang has no `--kv-transfer-config`).
- `dnsPolicy: Default` if HF pulls fail on the HyperPod node.
