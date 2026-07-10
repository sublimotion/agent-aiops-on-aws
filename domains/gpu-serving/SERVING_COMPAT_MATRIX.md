# GPU Serving Stack Compatibility Matrix

**Last Updated**: 2026-04-22

Quick-reference compatibility matrix for vLLM and SGLang serving engines across NVIDIA Hopper (H100/H200) and Blackwell (B200, RTX PRO 6000) GPU architectures. Use this to decide what works on what hardware with which engine before deployment.

---

## Instance Quick Reference

| Instance | GPU | Count | VRAM | Interconnect | Compute Capability | NCCL Status |
|---|---|---|---|---|---|---|
| p5e.48xlarge | H200 SXM | 8 | 141GB HBM3e | NVSwitch 900GB/s | sm_90 | **Stable** |
| p6-b200.48xlarge | B200 SXM | 8 | 192GB HBM3e | NVSwitch 1.8TB/s | sm_100 | **Stable** |
| g7e.24xlarge | RTX PRO 6000 Blackwell | 4 | 96GB GDDR7 | PCIe Gen5 | sm_120 | **BROKEN** (NCCL ≤2.25.1)* |

\* NCCL 2.25.1 has shared memory bug on Blackwell PCIe-only topology. Fixed in NCCL 2.26.2 (NGC 25.03+). vLLM inference unaffected (uses custom allreduce).

---

## Attention Backends

> This section lists the *kernel implementations* per GPU. For the *architectural families* they serve (dense/GQA, MLA, sparse, linear/gated-delta, Mamba-2 hybrid) and the serving consequence of each — prefix-cacheability, HiCache/disagg compatibility, decode bottleneck, which models use which — see `docs/inference-optimization-guide.md` §15 (Attention Family Taxonomy).

### vLLM

| Backend | H100/H200 (sm_90) | B200 (sm_100) | RTX PRO 6000 (sm_120) | Notes |
|---|:---:|:---:|:---:|---|
| **FlashAttention** | FA3 (default) | FA4 (default) | FA4 (default) | FA2 fallback on older architectures |
| **FlashInfer** | Supported | Supported | Supported | Second priority on Blackwell, first on Hopper |
| **FlashMLA** | All variants | All variants | Field: **None** | FLASHMLA, FLASHMLA_SPARSE, FLASH_ATTN_MLA, FLASHINFER_MLA, CUTLASS_MLA, TRITON_MLA. Per docs: sm_120 has kernels, field: none work |
| **Triton Attention** | Supported | Supported | Supported | Fallback backend |
| **Flex Attention** | Supported | Supported | Supported | Lowest priority |

**Backend Priority Order**:
- **Hopper (sm_8x-9x)**: FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX_ATTENTION
- **Blackwell (sm_10x)**: FLASHINFER → FLASH_ATTN → TRITON_ATTN → FLEX_ATTENTION

### SGLang

| Backend | H100/H200 (sm_90) | B200 (sm_100) | RTX PRO 6000 (sm_120) | Notes |
|---|:---:|:---:|:---:|---|
| **MHA (Standard)** | FA3 (default, CUDA 12.3+ req) | TRTLLM MHA (default) | TRTLLM MHA | Blackwell: only triton, trtllm_mha, or fa4 allowed |
| **MLA (DeepSeek/Kimi)** | FA3 (default) | TRTLLM MLA (default) | Field: None* | Blackwell default supports FP8 KV cache |
| **FlashAttention-4** | Supported (caveat) | Supported | Supported | **FA4 on Hopper**: decode degrades -49% at 16K due to missing SplitKV |
| **FlashMLA** | Supported | Supported | Field: **None*** | Dedicated MLA backend, FP8/FP4 KV cache, page_size=64 constraint |

\* Field-validated: RTX PRO 6000 (sm_120) has no working FlashMLA or TRTLLM MLA support. BF16-only KV cache.

---

## Quantization

### Compute Capability Requirements

| Quantization | Min Compute | H100/H200 | B200 | RTX PRO 6000 | Notes |
|---|---|:---:|:---:|:---:|---|
| **FP8 (W8A8)** | ≥ 8.9 | ✅ | ✅ | ✅ | Native support on Hopper and Blackwell |
| **INT4 (GPTQ/AWQ)** | > 8.0 | ✅ | ✅ | Field: ⚠️ | Ampere+. CUDA-only kernels |
| **INT4 Marlin** | > 8.0 | ✅ | Field: ❓ | Field: **❌** | RTX PRO 6000: Marlin MoE broken |
| **FP4 (modelopt)** | ≥ 10.0 | ❌ | ✅ | ✅ | Blackwell-native (SGLang only) |

### Field-Validated Issues

| Issue | Affected Hardware | Status | Mitigation |
|---|---|---|---|
| **Marlin MoE on RTX PRO 6000** | g7e.24xlarge (sm_120) | BROKEN | Use FP8 or BF16 |
| **Compressed-tensors INT4 Marlin on H200** | p5e.48xlarge (sm_90) | ⚠️ Works, PTX mismatch possible | Match CUDA version (vLLM wheel vs driver) |
| **GPTQ-Int4 with qwen3_5_moe** | All | Garbage output (vLLM 0.18) | Use FP8 or FP16 |
| **Marlin PTX issue** (#38619) | H200 | vLLM CUDA 12.9 wheel + driver 12.8 | Upgrade driver to 12.9+ |

**Engine Support**:
- **vLLM**: FP8, INT4 (GPTQ/AWQ/Marlin), BF16
- **SGLang**: FP8, INT4 (GPTQ/AWQ/Marlin), FP4 (Blackwell-only via modelopt_fp4), BF16

---

## KV Cache

| Feature | Engine | H100/H200 (sm_90) | B200 (sm_100) | RTX PRO 6000 (sm_120) | Notes |
|---|---|:---:|:---:|:---:|---|
| **BF16 KV cache** | Both | ✅ | ✅ | ✅ | Universal baseline |
| **FP8 KV cache** | Both | ✅ | ✅ (TRTLLM MLA) | Field: **❌** | RTX PRO 6000: BF16 only |
| **FP4 KV cache** | SGLang | ❌ | ✅ (FA4/Triton/Flex) | ✅ | Blackwell-native |
| **Prefix caching** | Both | ✅ | ✅ | ✅ | Automatic (vLLM) / RadixAttention (SGLang) |
| **HiCache** | SGLang | ✅ | ✅ | ✅ | Hierarchical NVMe offload. Field: +71% throughput on GLM-5 B200 MLA |
| **LMCache** | vLLM | ✅ (non-MLA) | ✅ (non-MLA) | ✅ (non-MLA) | **BLOCKED for all MLA models** (bugs #2881, #2947, #2636) |

**Field Validation**:
- **HiCache on GLM-5 (B200)**: 71% throughput gain at 64 concurrent requests (2.86x peak vs baseline). Native `NSATokenToKVPoolHost` handles fused `kv_buffer`. Requires `hicache-size` > device KV pool (~82 GB/rank for GLM-5).
- **LMCache + MLA**: Shape mismatch bugs prevent use on DeepSeek, Kimi K2, GLM-5 architectures. Use SGLang HiCache instead.
- **RTX PRO 6000**: Only BF16 KV cache supported. No FP8/FP4 KV cache in field testing.

---

## MLA (Multi-head Latent Attention)

Critical for DeepSeek-V2, DeepSeek-V3, Kimi K2 family, GLM-5 architectures.

| Capability | H200 (sm_90) | B200 (sm_100) | RTX PRO 6000 (sm_120) |
|---|:---:|:---:|:---:|
| **FlashMLA (decode)** | Full support | Sparse only* | Field: **None** |
| **DeepGEMM MoE** | ✅ | ✅ (CUDA 12.9+) | Field: **None** |
| **FP8 KV cache** | ✅ | ✅ (post fix) | Field: **❌** (BF16 only) |
| **INT4 Marlin** | ✅ | Untested | Field: **❌** |
| **Official support** | Reference HW | Not documented | Not documented |

\* B200 FlashMLA sparse mode requires specific SGLang configuration.

**Deployment Recommendation**:
- **H200 (p5e)**: Mature MLA support, all features work, reference hardware for MLA models
- **B200 (p6)**: Requires CUDA 12.9+, DeepGEMM JIT compilation (~15 min first start), sparse FlashMLA
- **RTX PRO 6000 (g7e)**: NOT recommended for MLA models. Use BF16 baseline only, no accelerated MLA kernels

From field testing: `kimi-k2-thinking` spec (2026-03-15).

---

## Speculative Decoding

### vLLM

| Method | H100/H200 | B200 | RTX PRO 6000 | Notes |
|---|:---:|:---:|:---:|---|
| **EAGLE** | ✅ | ✅ | ✅ | |
| **MTP** (Multi-Token Prediction) | ✅ | ✅ | ✅ | Requires native MTP heads (DeepSeek, Kimi K2) |
| **Draft model** | ✅ | ✅ | ✅ | |
| **N-gram** | ✅ | ✅ | ✅ | |
| **Speculators** | ✅ | ✅ | ✅ | |

**Field Validation**: vLLM GLM-5 on B200 with `--speculative-config.method mtp --speculative-config.num_speculative_tokens 1` (see `glm5-llmd` lessons).

### SGLang

| Method | H100/H200 | B200 | RTX PRO 6000 | Notes |
|---|:---:|:---:|:---:|---|
| **EAGLE-2** | ✅ | ✅ | ✅ | |
| **EAGLE-3** | ✅ | ✅ | ✅ | Best throughput |
| **MTP** | ✅ | ✅ | ✅ | |
| **Standalone draft** | ✅ | ✅ | ✅ | |
| **N-gram** | ✅ | ✅ | ✅ | CUDA-only, disables DP attention |

No documented Blackwell-specific limitations for speculative decoding on either engine.

---

## P/D Disaggregation

| Feature | vLLM | SGLang | H100/H200 | B200 | RTX PRO 6000 | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **NIXL** | ✅ | ✅ | ✅ | ✅ | ✅ | Dense, MLA, MoE architectures |
| **UCX backends** | ✅ | ✅ | ✅ | ✅ | ✅ | cuda_ipc, RDMA, libfabric |
| **cuda_ipc on NVSwitch** | ⚠️ | ⚠️ | ⚠️ | ⚠️ | N/A | Disabled by default (NIXL #1097). Safe for TP=1 |
| **EFA (g7e)** | ✅ | ✅ | N/A | N/A | ✅ | Kernel-bypass SRD, NOT true RDMA. NIXL LIBFABRIC works |

**cuda_ipc on NVSwitch**: Disabled by default to avoid NCCL contention. TP=1 is the sweet spot for single-node P/D disagg: no NCCL collectives → cuda_ipc safe → full NVLink BW for KV transfer. TP>1 on NVSwitch: use chunked prefill or RDMA loopback instead.

**EFA on g7e**: All instance sizes support EFA (12xl: 1, 24xl: 2, 48xl: 4 interfaces). EFA uses kernel-bypass SRD protocol, NOT true RDMA. Still requires CPU bounce (GPU→cudaMemcpy→CPU→EFA→CPU→cudaMemcpy→GPU). Enables NIXL LIBFABRIC disagg P/D between nodes. True GPUDirect RDMA (NIC↔GPU DMA) only on p5/p5e with InfiniBand + nvidia-peermem.

See [pd_disagg_single_node.md](pd_disagg_single_node.md) for full analysis.

---

## Tool Calling & Reasoning Parsers

| Parser | Engine | Status | Known Issues |
|---|---|---|---|
| **kimi_k2** | vLLM | ⚠️ Works | Streaming truncation/leakage (#38579) |
| **kimi_k2** | SGLang | ⚠️ Works | Regex boundary bug (#22173) |
| **hermes** | vLLM | ✅ | Works for Qwen3.5 |
| **hermes** | SGLang | ✅ | |
| **qwen3_xml** | vLLM | ✅ | Qwen3.5 MoE, use with `--reasoning-parser qwen3` |
| **mistral** | vLLM | ✅ | Requires `--enable-auto-tool-choice` for Devstral |
| **glm47** | vLLM | ✅ | GLM-5, use with `--reasoning-parser glm45 --enable-auto-tool-choice` |

**Field Notes**:
- **Qwen 2.5 Coder 32B**: Outputs bare JSON `{"name":"...","arguments":{...}}` in content, NOT `tool_calls` object. vLLM hermes parser returns empty `tool_calls`. Only SERA harness works (regex fallback).
- **Qwen3.5 MoE**: Proper `tool_calls` with `finish_reason: tool_calls`. Hermes parser works, OpenCode works.
- **Mistral chat template strict ordering**: `tool` role must follow `assistant` with `tool_calls`. Inserting `user` messages between causes `ValueError: Unexpected role 'tool' after role 'user'`.

---

## Other Features

| Feature | vLLM | SGLang | Notes |
|---|---|---|---|
| **Prefix Caching** | ✅ Automatic | ✅ RadixAttention | |
| **Chunked Prefill** | ✅ | ✅ | |
| **LoRA** | ✅ | ✅ | |
| **Pipeline Parallelism** | ✅ | ✅ | |
| **Expert Parallelism** | ✅ | ✅ | For MoE models |
| **Structured Output** | ✅ | ✅ | |
| **CUDA Graphs** | ✅ | ✅ | SGLang: can break with certain configs |
| **torch.compile** | ✅ | ⚠️ Limited | vLLM has broader support |
| **Sleep Mode** | ✅ | ❌ | vLLM-only feature |
| **HiCache** (NVMe offload) | ❌ | ✅ | SGLang-only, field: +71% throughput on GLM-5 |
| **Anthropic API** (/v1/messages) | ✅ | ❌ | vLLM has native support |

---

## Hardware-Specific Gotchas

### B200 (p6-b200.48xlarge)

| Issue | Mitigation |
|---|---|
| **AMI requirement** | Must use AL2023 NVIDIA AMI (ami-02bb9f913067dadb1). AL2 kernel lacks `ib_umad` module for Fabric Manager on NVL5+ |
| **Capacity block termination** | Takes ~10 min before slot frees up. Launch with `--instance-market-options '{"MarketType":"capacity-block"}'` |
| **EKS bootstrap** | AL2023 uses `nodeadm` (MIME multipart with `application/node.eks.aws`), not `/etc/eks/bootstrap.sh` |
| **Driver** | 580.126.09, CUDA 13.0 |
| **DeepGEMM JIT** | First start requires ~15-16 min compilation for MoE models (GLM-5, Kimi K2) |
| **SGLang GLM-5** | Use `lmsysorg/sglang:glm5-blackwell` image. Add `--host 0.0.0.0` to serve externally |

### RTX PRO 6000 (g7e.24xlarge)

| Issue | Mitigation |
|---|---|
| **NCCL broken** | NCCL ≤2.25.1 fails on Blackwell PCIe topology. Use NGC 25.03+ (NCCL 2.26.2) |
| **Container runtime** | `nerdctl` (not docker). Use `sudo nerdctl` |
| **Networking** | Containers need `--network host` (no CNI plugin on bare metal) |
| **MLA models** | NOT recommended. No FlashMLA, no DeepGEMM, no FP8 KV cache. Use H200 or B200 |
| **Marlin MoE** | Broken. Use FP8 or BF16 quantization |
| **NVMe mount** | `/mnt/nvme` |

### H200 (p5e.48xlarge)

| Issue | Mitigation |
|---|---|
| **Topology** | NVSwitch, mature NCCL support, no known issues |
| **Reference hardware** | Best choice for MLA models (DeepSeek, Kimi K2, GLM-5) |
| **Capacity** | More widely available than B200 |

---

## Deployment Decision Tree

```
┌─ What GPU architecture do you have?
│
├─ H100/H200 (sm_90, p5e)
│  ├─ MLA model (DeepSeek/Kimi/GLM)? → vLLM or SGLang, all features work ✅
│  ├─ Need KV offload? → vLLM + native prefix caching OR SGLang + HiCache
│  └─ Standard transformer? → vLLM or SGLang, choose based on features
│
├─ B200 (sm_100, p6)
│  ├─ MLA model? → SGLang preferred (TRTLLM MLA + FP8 KV cache)
│  ├─ MoE model? → Expect 15-16 min DeepGEMM JIT compilation on first start
│  ├─ Need FP4? → SGLang only (modelopt_fp4)
│  └─ Standard transformer? → vLLM or SGLang, FA4 default
│
└─ RTX PRO 6000 (sm_120, g7e)
   ├─ MLA model? → **NOT RECOMMENDED**. Use H200 or B200 instead
   ├─ MoE model? → vLLM with FP8 or BF16 (NOT Marlin)
   ├─ Multi-GPU? → Upgrade to NCCL 2.26.2+ (NGC 25.03)
   └─ Standard transformer? → vLLM or SGLang, BF16 or FP8 quantization
```

---

## Data Sources

- **Field-validated**: Lessons from blueprints in `domains/gpu-serving/blueprints/`
  - `kimi-k2-thinking`: MLA + B200/H200/g7e comparison
  - `glm5-lmcache`: LMCache bugs with MLA models
  - `glm5-llmd`: B200 + vLLM + llm-d, DeepGEMM JIT
  - `devstral-sera`: NCCL bug on g7e, RTX PRO 6000 limitations
  - `ray-serve-ft`: EKS + Ray Serve fault tolerance
- **Per docs**: vLLM and SGLang GitHub repositories, official documentation
- **Memory**: `.claude/projects/-Users-phi-Documents-workbench-agent-aiops-on-aws/memory/MEMORY.md`

---

## Maintenance

This matrix should be updated when:
1. New GPU architectures are validated (e.g., B300, H300)
2. New serving engine versions introduce breaking changes
3. Field testing reveals new compatibility issues
4. Upstream fixes resolve known blockers (e.g., LMCache MLA support)

Last major updates:
- 2026-04-22: Initial version based on H100/H200/B200/g7e field testing
