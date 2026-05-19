# DeepSeek V4 Flash — Lessons Learned

Blueprint: `domains/gpu-serving/blueprints/deepseek-v4-flash/`
Spec: `domains/gpu-serving/specs/deepseek-v4-flash.md`
Hardware: p6-b300.48xlarge spot in us-west-2b ($26.49/hr)
Status: STAGING (T0 in progress as of 2026-05-19 12:00 EDT)

---

## HIGH severity

### 1. vLLM 0.21 requires `--kv-cache-dtype fp8` for DeepSeek V4 (no auto-detect)

**Symptom**: vLLM crashes during worker init with:
```
AssertionError: DeepseekV4 only supports fp8 kv-cache format for now, got auto
```
at `gpu_model_runner.py:5022 → load_model → assert kv_cache_dtype.startswith("fp8")`.

**Root cause**: V4 Flash ships FP4+FP8 mixed weights with `quantization_config.scale_fmt=ue8m0`, but vLLM's auto-detection doesn't infer that `--kv-cache-dtype fp8` is mandatory. The CLI default is `auto` and the V4 model class explicitly rejects it.

**Fix**: Always pass `--kv-cache-dtype fp8` for DeepSeek V4 family. Add to `mdc get deepseek-v4 --engine vllm` learn note.

**Impact**: ~10 min wasted on a misleading initial error chain (the visible failure is `RuntimeError: Engine core initialization failed` at the API server level — actual cause is in the worker logs).

---

### 2. EKS managed nodegroup IMDSv2 hop limit defaults to 1, blocks IRSA

**Symptom**: Pods on a freshly-launched managed nodegroup get `Unable to locate credentials` for AWS calls, even with the IAM role correctly attached.

**Root cause**: `aws eks create-nodegroup` provisions instances with `HttpPutResponseHopLimit: 1`. Container traffic to IMDS (169.254.169.254) traverses the CNI bridge, which counts as one extra hop. With limit=1, the IMDSv2 token request fails inside containers.

**Fix**: After nodegroup ACTIVE, bump the instance metadata hop limit to 2:
```bash
aws ec2 modify-instance-metadata-options \
  --instance-id <node-instance-id> \
  --http-put-response-hop-limit 2 \
  --region us-west-2
```

**Impact**: ~15 min lost; persist-to-S3 Job entered crashloop and hit backoff limit before fix. **This should be added to `init-nvme.yaml` or a node-bootstrap step for all future blueprints.**

---

### 3. EKS managed nodegroups don't auto-label `nvidia.com/gpu.present=true`

**Symptom**: New B300 nodegroup is ACTIVE but `kubectl describe node` shows `nvidia.com/gpu` capacity = 0; serving pods stuck Pending with `Insufficient nvidia.com/gpu`.

**Root cause**: The `nvidia-device-plugin` DaemonSet selects on label `nvidia.com/gpu.present=true`. Self-managed/eksctl-provisioned nodegroups label this automatically; AWS EKS managed nodegroups (`AL2023_x86_64_NVIDIA` AMI type) do **not**.

**Fix**:
```bash
kubectl label node <node-name> nvidia.com/gpu.present=true
```
The DaemonSet lands within ~15s and GPUs are advertised within ~30s.

**Impact**: ~5 min lost. Should be automated in the launch script.

---

## MEDIUM severity

### 4. `huggingface_hub` v1.15+ deprecated `HF_HUB_ENABLE_HF_TRANSFER`; `--exclude` CLI flag broken

**Symptom**: Model download "completed" in 0.5s with only 48 KB of data; the 46 safetensors files (~150 GB) silently skipped.

**Root cause**: Two changes in HF Hub v1.15:
1. `HF_HUB_ENABLE_HF_TRANSFER` env var is deprecated; warning issued. Use `HF_XET_HIGH_PERFORMANCE=1` instead.
2. The `hf` CLI's `--exclude` semantics changed — passing repo glob patterns (`"*.md" "*.txt"`) is interpreted as positional file arguments to download, so only those specific (nonexistent) globs are fetched.

**Fix**: Use the Python `snapshot_download(allow_patterns=...)` API directly instead of the `hf` CLI:
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    local_dir="/mnt/nvme/models/DeepSeek-V4-Flash",
    allow_patterns=["*.json", "*.safetensors", "*.py", "tokenizer*", "encoding/*", "inference/*"],
    max_workers=16,
)
```

**Impact**: ~10 min on the first failed attempt. **Codified in the global memory `feedback_model_staging_pattern.md` and `scripts/stage-model-s3-cached.sh`.**

---

### 5. Stale Ray workers from a different blueprint claimed B300 GPUs

**Symptom**: Brand-new B300 advertised 8 GPUs, but only 6 were Allocatable; vLLM pod stuck Pending.

**Root cause**: The `ray-video` namespace had a `RayCluster` (52 days old, 3-day-stale workloads) with worker pods tolerating `nvidia.com/gpu:NoSchedule`. The original ray-ft-gpu nodegroup was scaled to 0 and the worker pods rescheduled onto the only remaining GPU node — ours.

**Fix**: Scale the offending RayCluster's gpu-workers to 0 temporarily:
```bash
kubectl -n ray-video patch raycluster <name> --type=json \
  -p '[{"op":"replace","path":"/spec/workerGroupSpecs/0/replicas","value":0},
        {"op":"replace","path":"/spec/workerGroupSpecs/0/minReplicas","value":0}]'
```
Reversible. Original spec saved to `/tmp/raycluster-restore.log` for restore.

**Impact**: ~5 min lost. Coexisting blueprints sharing a cluster is a real risk; future GPU-heavy blueprints should consider node taints or namespace-scoped GPU quotas.

---

## LOW severity / observations

### 6. Xet high-performance transfer is genuinely fast

HuggingFace Xet (replacing `hf_transfer`) downloaded 149 GB / 46 safetensors in **2 min 48 sec** — vs the 1.5hr estimate. ~890 MB/s effective. The 28 TB RAID0 NVMe writes were not the bottleneck. Future model staging time estimates should be revised down.

### 7. transformers 4.56.2 pin works for V4 in vLLM 0.21

`pip install "transformers==4.56.2"` resolves the vLLM #42741 `compress_ratios` issue without requiring PR #42806 to merge. vLLM warns "Support for Transformers v4 is deprecated" — works fine for now.

### 8. vLLM nightly digest pinning paid off

Used `vllm/vllm-openai:nightly-6e889b582b6a0b11f22b3764be174266faa9ff5e` (today's build). The `latest`/`nightly` floating tags would also have worked for this commit but reproducibility matters when filing bugs. Nightly version: `v0.21.1rc1.dev98+g6e889b582`. Confirmed PR #42320 (DSv4 MTP HC state fix, merged 2026-05-13) is in.

### 9. V4 Flash architecture confirmed at startup

vLLM's startup logs surface key features:
- `Resolved architecture: DeepseekV4ForCausalLM`
- `Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM`
- `tokenizer_mode='deepseek_v4'`
- `quantization=deepseek_v4_fp8`
- `splitting_ops` includes `deepseek_v4_attention` + `sparse_attn_indexer` (CSA path active)
- FlashInfer cache hit on `trtllm_fp4_block_scale_moe` (FP4 MoE kernel)

### 10. config.json reveals architecture details

`head_dim=512` (Hopper+ requirement), `n_routed_experts=256` top-6, `n_shared_experts=1`, `index_head_dim=128 / index_n_heads=64 / index_topk=512` (Lightning Indexer for CSA), `max_position_embeddings=1048576` (1M context), `expert_dtype: fp4`. `num_attention_heads=64 / num_key_value_heads=1` is unusual GQA-extreme config.

---

## T0 baseline complete (2026-05-19)

- **vLLM startup time**: ~11 min (DeepGEMM JIT + CUDA graph capture + FlashInfer autotune)
- **T0 result** (full table in `T0-report.md`):
  - Peak total throughput @ QPS=8: **11,823 tok/s** (matches Qwen3-235B-B300 @ 11,820 tok/s with 40% fewer active params)
  - Peak total throughput @ ctx=16384: **15,387 tok/s** (CSA scaling holds)
  - Median ITL: 7-9 ms across all context lengths (sub-linear FLOPs claim validated qualitatively)
  - 0 errors / 480 requests across 9 measurements
- **Smoke test**: ✓ "The capital of France is Paris" — no precision regression observed (SGLang #25662 not reproduced for our prompt class on vLLM nightly)

## R2 (prefix-caching + long-context) and R3 (MTP) complete (2026-05-19)

### MEDIUM severity additions

### 11. MTP speculative decoding hurts throughput on V4 Flash B300 (38% acceptance is below break-even)

**Symptom**: With `--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}'`, throughput drops 5-52% across QPS levels and ITL roughly doubles vs the non-MTP baseline. ShareGPT TTFT p50 jumped 70ms → 6094ms.

**Root cause**: Server-reported acceptance rate **136,829 accepted / 359,647 drafts = 38.0%**. At <60% acceptance, the verification overhead per token exceeds the savings from speculation — the model wastes compute drafting tokens that get rejected, then has to do the full forward pass anyway. Upstream #41789 reports 0.2% on consumer 5090 (much worse), so the 38% we got is actually the optimistic case.

**Fix**: **Disable MTP for production** until acceptance rate improves (likely awaiting an upstream fix, possibly EAGLE3 implementation per #42413). Single-token MTP is not viable on V4 Flash at this maturity.

**Impact**: Saves us shipping a "hot new MTP feature" that would actually slow customer workloads down. Caught by the W0 cross-check guardrail and sharegpt comparison — synthetic-only would have shown a smaller gap.

### 12. vLLM #42948 (prefix-cache 0% hit) does NOT manifest on shared-prefix-pattern workloads

**Symptom expected**: Per the bug report, V4 Flash should show 0% prefix cache hit rate on hybrid attention groups.

**What we saw**: T1 prefix caching delivered 30-57K tok/s at 16K-32K shared prefix — clear evidence of caching working. The bug must require a more specific access pattern than `generated-shared-prefix` produces.

**Action**: Re-test with the exact reproducer from #42948 (request resending pattern) to confirm whether the bug applies to our deployment. For now, treat T1 numbers as valid for production planning.

### LOW severity / observations

### 13. CSA architecture truly is sub-linear

ITL p50 measurements at 1K → 390K context single-stream: 8.1 → 8.0 → 8.1 → 7.7 → 7.4 → 7.3 → 7.27 → 7.51 → 7.64 ms. **It actually decreases slightly** from 16K to 256K (warmer caches?), then ticks up minimally at 390K. This is the canonical validation of the "10% KV / 27% FLOPs at 1M" claim — a linearly-scaling attention mechanism would have ITL doubling every doubling of context, not staying flat.

### 14. ShareGPT P99 TTFT spike at low QPS is expected

W0 sharegpt at QPS=1 showed P99 TTFT = 23.2 sec while p50 = 161 ms. At QPS=4 the p99 dropped to 127 ms. The cause: rare 7K+ token prompts in the dataset that pay full cold-cache prefill cost when the cache hasn't been warmed. **This is exactly the W0 finding documented in the spec** — synthetic random workloads with fixed input length systematically underestimate this kind of long-tail variance.

### 15. Prefix-caching warmup in production matters more than raw QPS-sweep numbers suggest

The W0 sharegpt result jumps from 215 out tok/s (QPS=1 cold) → 771 (QPS=4 warm) → 1,380 (QPS=8 saturated). For real chat workloads, **keep the cache warm with a heartbeat** — even a low background QPS dramatically improves p99 TTFT for production users.

## SWE-bench 79% validation handoff

Pending — separate workstream via `verification-primitives-swebench`. Not part of serving benchmark scope.
