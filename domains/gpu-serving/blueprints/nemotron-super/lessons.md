# Nemotron-3-Super Lessons Learned

## Pre-Deployment

### #1 — Mamba Hybrid KV Cache Incompatibilities
vLLM disaggregated mode is NOT supported for Nemotron-3-Super due to hybrid KV cache (Mamba recurrent state + attention KV). Dynamo PR #7216 confirms this. Only SGLang and TRT-LLM support disaggregated prefill/decode under Dynamo for this model.

### #2 — KV Block Reuse Disabled for Mamba
Standard KV block reuse (prefix caching) is incompatible with Mamba recurrent state. TRT-LLM requires `block_reuse: false`. KV-overlap scoring provides no benefit — round-robin routing used for disaggregated mode.

### #3 — NGC Dynamo Image Tags
Dynamo runtime images on NGC use `0.9.1` (no `v` prefix). Example: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.1`. Tag `v0.9.1` returns MANIFEST_UNKNOWN.

### #4 — Custom Reasoning Parser (vLLM only)
vLLM requires `super_v3_reasoning_parser.py` from the HF model repo, passed via `--reasoning-parser-plugin`. SGLang and TRT-LLM use built-in `nano_v3` parser. The plugin file is bundled with the model weights on FSx/NVMe.

### #5 — Model Weights Staged on FSx
Model successfully downloaded to FSx at `/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` (119 GB, 26 safetensors shards). Used `huggingface_hub.snapshot_download()` Python API — `huggingface-cli` had PATH/API issues in container.

### #6 — Attention Backend Requirement
vLLM requires `--attention-backend TRITON_ATTN` for hybrid Mamba+Attention architecture. FlashAttention/FlashInfer may not support select attention layers.

### #7 — Temperature Requirement
Model requires `temperature=1.0, top_p=0.95` for all tasks. Do NOT use greedy decoding (temperature=0).

### #8 — Reusing GLM-5 B200 Cluster
Blueprint connects to existing `glm5-lmcache-b200-eks-cluster` via Terraform data sources. FSx PVC is `glm5-fsx-pvc` (not `fsx-lustre-pvc`). Same VPC, EKS, FSx, and GPU node infrastructure.

### #9 — TRT-LLM Runtime Images 401
NGC `trtllm-runtime` images return 401 Unauthorized even with NGC auth. May require specific NGC API key or org access. Skipping TRT-LLM for initial deployment.

### #10 — AL2023 NVIDIA AMI Lacks Lustre Client Userspace
The `AL2023_x86_64_NVIDIA` AMI does NOT include `lustre-client` userspace tools (mount.lustre, lctl). The kernel module (Lustre 2.15.6) is loaded but FSx CSI driver fails with `Can't parse NID` or `client profile could not be read`. Fix: install `lustre-client` via privileged init container chroot: `chroot /host dnf install -y lustre-client`. Original on-demand node group used `AL2_x86_64_GPU` which includes Lustre client.

### #11 — FSx Security Group for Spot Nodes
EKS spot node groups created without a launch template get the cluster SG only (`eks-cluster-sg-*`), not the custom GPU node SG. FSx inbound rules must explicitly allow the cluster SG on ports 988, 1018-1023 (Lustre). Without this, mount hangs or returns EINVAL.

### #12 — FSx DNS Resolves to Single MGS IP
FSx Lustre has 3 ENIs but DNS resolves to only one (the MGS). When creating IP-based PVs, use the DNS-resolved IP, not arbitrary ENI IPs. Wrong IP causes `client profile could not be read from MGS`.

### #13 — Dynamo vLLM Runtime Cold Start on B200
Model load: 31s (26 safetensors from NVMe). Total startup: ~3 min (weights 31s + model init 14s + torch.compile 43s + CUDA graphs 9s). Much faster than SGLang GLM-5 (15 min DeepGEMM JIT). KV cache: 100 GiB (8.7M tokens).

### #15 — Initial Benchmark: vLLM TP=2x1 on B200
Single worker (TP=2, 2x B200). Conc=1: TTFT 311ms, ITL 6.6ms, 80 tok/s. Conc=16: TTFT 1,359ms (p50), 469 tok/s, 5.9x throughput scaling. TTFT p90 blows up to 12.7s at conc=16 — classic head-of-line blocking. This validates the need for disaggregated prefill/decode. KV cache: 100 GiB (8.7M tokens). Model only uses 57.6 GiB VRAM with TP=2.

### #16 — High-Concurrency Benchmark: vLLM TP=2x4 on B200 (8 GPUs)
Four workers (TP=2 each, 8x B200 total). Concurrency sweep results (4096 in, 1024 out):
- Conc=1: TTFT 304ms, ITL 6.6ms, 76.5 tok/s (single-stream baseline)
- Conc=16: TTFT 504ms (p50) / 11.9s (p90), ITL 12.3ms, 513 tok/s
- Conc=64: TTFT 974ms (p50) / 9.4s (p90), ITL 20.3ms, 1,081 tok/s ← sweet spot
- Conc=128: TTFT 6.2s (p50) / 21.0s (p90), ITL 28.0ms, 1,228 tok/s
- Conc=256: TTFT 21.2s (p50) / 44.5s (p90), ITL 44.7ms, 1,449 tok/s ← peak throughput
Sweet spot is conc=64 (sub-1s TTFT p50, 1K+ tok/s). TTFT p90 blows up from conc=16 onwards — head-of-line blocking from aggregated prefill/decode sharing GPUs. Strong case for disaggregated mode (Dynamo + SGLang).

### #17 — SGLang Disagg NIXL Transfer Fails for Mamba Hybrid
SGLang `--disaggregation-mode prefill/decode` with `--disaggregation-transfer-backend nixl` crashes on NIXL KV transfer for Nemotron-3-Super. The decode worker connects to the prefill's bootstrap server (port 8998) but "Lost connection with prefill instance" — prefill crashes during NIXL KVSender operation. Root cause: Mamba-2 recurrent state + attention KV hybrid cache is not handled by NIXL's standard KV transfer protocol. The `dynamo-0.9.1` SGLang image also has a separate `ModelOptFp8LinearMethod.create_weights()` bug with `ReplicatedLinear` (6 vs 5 args), requiring the `v0.5.9` image instead. Disaggregated serving for Mamba hybrids may need custom transfer logic or Dynamo-native support.

### #18 — SGLang PD Router Architecture
SGLang disaggregation requires a **router** that sends the same request to BOTH prefill and decode workers. The prefill worker runs a `CommonKVBootstrapServer` on port 8998. Client request must include `bootstrap_host` (prefill pod IP), `bootstrap_port` (8998), and `bootstrap_room` (unique ID). The official router is `sglang_router.launch_router --pd-disaggregation` (Rust, needs compilation). The decode worker creates a `KVReceiver` that connects to the bootstrap server to coordinate NIXL transfer.

### #19 — SGLang Aggregated Benchmark: vLLM Wins on Nemotron-Super
SGLang v0.5.9 aggregated (TP=2 x 4 workers, 8x B200) vs vLLM (Dynamo 0.9.1 image):
- Conc=1: SGLang 54.1 tok/s vs vLLM 76.5 tok/s (vLLM +41%)
- Conc=64: SGLang 924 tok/s, TTFT 4.1s vs vLLM 1,081 tok/s, TTFT 974ms (vLLM +17% throughput, 4.3x lower TTFT)
- Conc=256: SGLang 1,238 tok/s vs vLLM 1,449 tok/s (vLLM +17%)
- ITL is close: SGLang 36ms vs vLLM 45ms at conc=256 (SGLang slightly better at high conc)
vLLM's TRITON_ATTN backend handles Mamba-2 hybrid attention more efficiently for prefill. Both suffer identical head-of-line blocking at high concurrency (TTFT p90 > 40s at conc=256).

### #20 — SGLang PR References for Nemotron-Super Bugs
- **ModelOptFp8 bug**: Fixed in SGLang PR #18447 (merged 2026-02-16). The `dynamo-0.9.1` NGC image predates this fix — use `v0.5.9` or later.
- **NIXL Mamba disagg crash**: Open issues #19158, #18414, #19045. PR #19254 (hybrid linear PP+PD support) in progress but not merged. Issue #17447 tracks disagg refactor, #12867 is roadmap. No ETA for Mamba hybrid KV transfer support.

### #21 — Dynamo 1.0.0 NIXL LIBFABRIC Disagg on EFA (Dynamo PR #7369)
Dynamo PR #7369 demonstrates disaggregated prefill/decode on g7e using vLLM's `NixlConnector` with LIBFABRIC backend over EFA. Two modes:

**With EFA (g7e.12xlarge+)**: NIXL uses LIBFABRIC's EFA provider for kernel-bypass (OS-bypass) networking between nodes. KV transfer path: GPU VRAM → cudaMemcpy → CPU buffer → EFA SRD (kernel-bypass, not true RDMA) → CPU buffer → cudaMemcpy → GPU VRAM. EFA uses AWS SRD protocol — it bypasses the kernel networking stack but still requires CPU bounce on both sides. True GPUDirect RDMA (NIC↔GPU DMA without CPU) requires InfiniBand + `nvidia-peermem` (p5/p5e only). Requires `vpc.amazonaws.com/efa: "1"` resource request, `IPC_LOCK` capability, and `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.0-efa-amd64` image with EFA installer baked in. Config:
```
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'
```

**Without EFA (TCP fallback)**: LIBFABRIC falls back to TCP provider. Must add `"kv_buffer_device":"cpu"` (default is `cuda` which requires RDMA). Config:
```
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cpu","kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'
```

**Key insight**: g7e supports EFA on ALL sizes (12xl: 1 EFA, 24xl: 2 EFA, 48xl: 4 EFA). EFA is a *network* interface for inter-node RDMA — independent of GPU interconnect (PCIe vs NVLink). We incorrectly assumed g7e couldn't do disagg because it's PCIe-only, but PCIe = no NVLink *within* a node, while EFA = RDMA *between* nodes. These are orthogonal.

**Still blocked for Nemotron-Super**: This Dynamo path uses vLLM's `NixlConnector` which requires standard transformer KV cache. Nemotron-Super's Mamba-2 hybrid recurrent state is incompatible with all KV transfer connectors (HMA disabled → can't unify hybrid KV cache specs). The EFA/LIBFABRIC discovery doesn't change the Mamba blocker — it's a KV format issue, not a transport issue.

### #22 — Spec-Decode NOT Deployable on vLLM 0.18.1 TP2 (fin-rag P1)
Both speculative-decode paths are blocked at TP2 for Nemotron-3-Super on vLLM 0.18.1:
- **MTP**: blocked at TP2 (not deployable on this stack).
- **n-gram**: graph-captured n-gram (`--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'`) crashes during Mamba-2 CUDA-graph capture at TP2. The only way to get it to start was to add `--enforce-eager`, which disables CUDA-graph capture entirely.

**Verdict: spec-decode is NOT deployable for Nemotron-3-Super on vLLM 0.18.1 at TP2.** This is the customer answer and is consistent with the prefill-dominated fin-rag workload, where spec-decode was always the secondary lever.

**Decision (operator):** did NOT pursue the enforce-eager acceptance measurement; eager latency is non-representative and the crashed path (Mamba2 graph capture) is required in production, so an eager acceptance number is not actionable. Spec-decode verdict stands as not-deployable on 0.18.1 TP2. Spec-decode axis CLOSED.

Serving reverted to the graph-captured FP8 winner: agg-tp2-x4, `--max-num-batched-tokens 16384`, `--kv-cache-dtype fp8`, `--attention-backend TRITON_ATTN`, NO spec-decode, NO enforce-eager. Deployment `fin-rag-vllm-fp8` in namespace `ml-inference`.

### #14 — Spot B200 Available in us-east-2
p6-b200.48xlarge spot instances available at ~$17.90-18.90/hr (us-east-2b/2c). Created via EKS managed node group with `--capacity-type SPOT`. Must use `--ami-type AL2023_x86_64_NVIDIA` (default `AL2023_x86_64_STANDARD` has no GPU drivers).
