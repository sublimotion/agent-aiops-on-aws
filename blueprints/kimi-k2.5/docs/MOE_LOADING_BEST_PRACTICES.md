# MoE Model Loading Best Practices: Kimi K2.5 on vLLM

**Date**: February 2026
**Model**: moonshotai/Kimi-K2.5
**Hardware**: p5e.48xlarge (8x H200, 1.1TB HBM)

---

## Executive Summary

Loading Kimi K2.5 (1T parameters, 64 safetensor shards, ~540GB) requires specific optimizations due to its MoE architecture. Our loading failures at 20-25% were likely caused by:

1. **Sequential shard loading** bottlenecking on I/O
2. **Missing `--enforce-eager` flag** causing torch.compile hangs
3. **NCCL timeout** (default 30min may be insufficient)
4. **Capacity block expiration** during long loading

---

## 1. Kimi K2.5 Model Specifications

| Attribute | Value |
|-----------|-------|
| **Architecture** | Mixture-of-Experts (MoE) |
| **Total Parameters** | 1 Trillion |
| **Activated Parameters** | 32B per forward pass |
| **Layers** | 61 (1 dense + 60 MoE) |
| **Number of Experts** | 384 |
| **Selected Experts/Token** | 8 |
| **Context Length** | 256K tokens |
| **Attention** | Multi-head Latent Attention (MLA) |
| **Quantization** | Native INT4 (CompressedTensorsWNA16MarlinMoE) |
| **Model Size** | ~540 GB (64 safetensor shards) |
| **HuggingFace ID** | moonshotai/Kimi-K2.5 |

---

## 2. Why MoE Loading Is Different

### 2.1 Sparse Architecture Challenges
- **384 experts** distributed across 64 safetensor files
- Each shard must be loaded and distributed to correct GPU
- Expert weights are sparse - only 8/384 activate per token
- Irregular memory access patterns during expert routing

### 2.2 Memory Patterns
```
Dense Model:  [GPU 0] <- [Weights] <- [Disk]
              Sequential, predictable

MoE Model:    [GPU 0] <- [Expert 0,8,16...] <- [Shard 1]
              [GPU 1] <- [Expert 1,9,17...] <- [Shard 2]
              ...
              Complex routing, parallel loading beneficial
```

### 2.3 Attention: MLA vs Standard MHA
Kimi K2.5 uses **Multi-head Latent Attention (MLA)** which requires:
- `FLASH_ATTN_MLA` attention backend (auto-selected)
- Special KV cache handling (compressed latent states)
- vLLM 0.14.0+ for proper support

### 2.4 Shard-to-GPU Mapping in vLLM

vLLM does not map safetensor shards 1:1 to GPUs. Instead, it uses a two-phase process:

1. **Index parsing**: vLLM reads `model.safetensors.index.json` to build a map of tensor name → shard file
2. **Weight sharding**: For each tensor, vLLM's `ShardedStateLoader` determines which TP (tensor parallel) rank owns which slice based on the tensor's role:
   - **Attention weights**: split along head dimension across TP ranks
   - **MLP/Expert weights**: split along hidden dimension across TP ranks
   - **Expert routing**: replicated on all ranks (small, needed everywhere)

```
model.safetensors.index.json
  ├── "model.layers.0.mlp.experts.0.w1.weight" → shard-00003
  ├── "model.layers.0.mlp.experts.1.w1.weight" → shard-00003
  └── ...

vLLM reads shard-00003, extracts expert 0 w1, splits across 8 GPUs:
  GPU 0 gets columns [0:hidden/8]
  GPU 1 gets columns [hidden/8:2*hidden/8]
  ...
```

Key implications:
- A single shard file may contain weights destined for **all 8 GPUs**
- Every GPU must wait for a shard to load before extracting its slice
- You cannot control or override this mapping — it is determined by the model's tensor layout

### 2.5 Avoiding Loading Hotspots

With 64 shards and 8 GPUs, hotspots arise from two sources:

**I/O hotspots** (storage bottleneck):
- Default FSx striping places each shard file on a single OST
- Multiple shards on the same OST create contention
- Fix: `lfs setstripe -c -1` spreads each file across all OSTs (see section 5.2)
- NVMe RAID0 eliminates this by design — striping is automatic

**Memory hotspots** (uneven GPU allocation):
- MoE expert weights are not uniformly distributed across shards
- Some shards contain more expert parameters than others
- vLLM 0.15+ uses `ThreadPoolExecutor` for parallel shard reads, reducing sequential bottlenecks

**Mitigation strategies**:
| Strategy | Effect |
|----------|--------|
| FSx striping (`lfs setstripe -c -1`) | Eliminates per-OST I/O bottleneck |
| NVMe RAID0 local copy | Removes network entirely |
| vLLM parallel loading (default in 0.15+) | Reads multiple shards concurrently |
| Run:ai Model Streamer | Streams per-layer, inherently parallel (see section 4.5) |
| CloudWatch OST monitoring | Detects uneven throughput distribution |

---

## 3. Root Cause Analysis: Our Loading Failures

### 3.1 Observed Behavior
- Loading starts successfully
- Progress reaches 20-25% (14-16/64 shards)
- Process silently terminates
- No OOM errors in dmesg
- No error messages in logs

### 3.2 Likely Causes

| Cause | Evidence | Probability |
|-------|----------|-------------|
| **Capacity block expiration** | Instance terminated at exact reservation end | **High** |
| **NCCL timeout** | No error messages, distributed setup | Medium |
| **torch.compile hang** | Long pauses between progress updates | Medium |
| **Sequential I/O bottleneck** | ~4 min/shard from FSx | Low |

### 3.3 What We Got Right
- GDS validation passed (2+ GB/s)
- FSx mounted correctly
- Model downloaded completely (540 GB)
- vLLM 0.15.1 supports Kimi K2.5

### 3.4 What We Missed
- **`--enforce-eager`**: Skips torch.compile, avoids hangs
- **NCCL_TIMEOUT**: Should be 1800+ seconds for large MoE
- **NVMe loading**: Would be 8x faster than FSx
- **Capacity block duration**: Needed 6+ hours, had less

---

## 4. Optimized Loading Configuration

### 4.1 Environment Variables
```bash
# Disable HuggingFace online checks
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Extend NCCL timeout for large model loading
export NCCL_TIMEOUT=1800  # 30 minutes (default is ~30 min)

# Set attention backend
export VLLM_ATTENTION_BACKEND=FLASHINFER

# Optional: Debug logging
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=INFO
```

### 4.2 Critical vLLM Flags for MoE

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/nvme/models/Kimi-K2.5 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --enforce-eager \                    # CRITICAL: Skip torch.compile
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \      # Leave headroom for MoE routing
    --max-model-len 32768 \
    --swap-space 32 \                    # GB of swap for KV cache
    --disable-log-requests \
    --port 30080
```

### 4.3 Kimi K2.5 Specific Flags (Optional)
```bash
    --mm-encoder-tp-mode data \          # For vision encoder
    --tool-call-parser kimi_k2 \         # For tool calling
    --reasoning-parser kimi_k2 \         # For reasoning output
```

### 4.4 Flag Explanations

| Flag | Purpose | Why Important for MoE |
|------|---------|----------------------|
| `--enforce-eager` | Disable torch.compile | Avoids 10-15 min compilation hang |
| `--tensor-parallel-size 8` | Shard across GPUs | MoE requires TP = GPU count |
| `--gpu-memory-utilization 0.85` | Memory allocation | Leave room for expert routing |
| `--swap-space 32` | KV cache overflow | MoE has variable memory usage |
| `--trust-remote-code` | Custom model code | Required for Kimi architecture |
| `--enable-prefix-caching` | Cache common prefixes | Critical for KV cache benchmarks |

### 4.5 Run:ai Model Streamer for MoE Loading

The Run:ai Model Streamer (`--load-format runai_streamer`) streams model weights directly from S3 to GPU memory, bypassing local disk entirely. This is already configured in our Terraform (`terraform/vllm.tf`) for smaller models.

**How it works with MoE**:
```
Traditional:  S3 → FSx/NVMe → CPU RAM → GPU HBM  (3 copies)
Streamer:     S3 → CPU buffer → GPU HBM            (1 intermediate copy)
```

**Potential benefits for Kimi K2.5**:
| Aspect | Traditional (NVMe) | Run:ai Streamer |
|--------|-------------------|-----------------|
| Pre-staging required | Yes (10-15 min copy) | No |
| Storage needed | 540GB local | None (streams from S3) |
| Parallelism | File-level (ThreadPoolExecutor) | Layer-level (concurrent streams) |
| Resumability | Restart from scratch | Can resume from last layer |

**Compatibility concerns**:
- Kimi K2.5 requires `--trust-remote-code` for custom architecture — verify streamer supports custom weight loaders
- MoE expert routing tables must load fully before inference can begin — streaming may not help with this bottleneck
- `RUNAI_STREAMER_MEMORY_LIMIT` (default 4Gi) may need tuning for 64-shard models — increase to 8-16Gi for larger buffer pool

**Recommended test**:
```bash
# Compare loading times
# 1. NVMe baseline (current approach)
time python -m vllm.entrypoints.openai.api_server \
    --model /mnt/nvme/models/Kimi-K2.5 \
    --tensor-parallel-size 8 --enforce-eager ...

# 2. Run:ai streamer from S3
export RUNAI_STREAMER_MEMORY_LIMIT=16Gi
time python -m vllm.entrypoints.openai.api_server \
    --model s3://bucket/models/Kimi-K2.5 \
    --load-format runai_streamer \
    --tensor-parallel-size 8 --enforce-eager ...
```

**Verdict**: Worth testing. The streamer eliminates the pre-staging step entirely, which could save 10-15 minutes and remove NVMe RAID setup complexity. However, S3 bandwidth (~100 Gbps per instance) may be slower than NVMe RAID0 (~25 GB/s = 200 Gbps) for the initial bulk load.

### 4.6 Ahead-of-Time Compilation and Cache Persistence

The `--enforce-eager` flag skips `torch.compile` entirely, but compilation provides runtime performance benefits (kernel fusion, memory optimization). To get both fast loading and optimized inference:

**Option 1: Persist the compilation cache**
```bash
# Set cache directory (persists compiled kernels across restarts)
export TORCHINDUCTOR_CACHE_DIR=/mnt/nvme/torch_cache
export TORCH_COMPILE_CACHE_SIZE=16  # GB

# First run: slow (compiles all kernels, 10-15 min for MoE)
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/nvme/models/Kimi-K2.5 \
    --tensor-parallel-size 8 ...
    # Note: do NOT use --enforce-eager

# Subsequent runs: fast (reuses cached compilations)
# Same command — torch.compile finds cached kernels
```

**Option 2: Warm-up then switch (Kubernetes)**
```yaml
# Mount a PV for torch cache persistence across pod restarts
volumes:
  - name: torch-cache
    persistentVolumeClaim:
      claimName: torch-compile-cache
volumeMounts:
  - name: torch-cache
    mountPath: /root/.cache/torch_inductor
```

**Option 3: Two-phase startup**
1. Start with `--enforce-eager` for fast initial availability
2. Trigger background compilation via vLLM's `--compilation-config` (vLLM 0.15+)
3. Compiled kernels take effect for subsequent requests

**MoE-specific considerations**:
- Expert routing kernels are the main compilation target — 384 experts means many unique kernel variants
- Compilation cache size can reach 2-4 GB for large MoE models
- `TORCHINDUCTOR_CACHE_DIR` should be on NVMe, not FSx, for fast cache reads
- If using capacity blocks, the compilation cache must be on FSx or S3 to survive instance termination

---

## 5. Storage Optimization

### 5.1 Loading Time Comparison

| Storage | Bandwidth | 540GB Load Time | Notes |
|---------|-----------|-----------------|-------|
| FSx Lustre (1 OST) | ~240 MB/s | ~4 hours | **Our observed time** - files not striped! |
| FSx Lustre (all OSTs) | ~5-10 GB/s | ~1-2 min | Requires `lfs setstripe -c -1` |
| FSx Lustre (GDS+EFA+striped) | ~50-150 GB/s | ~5-10 sec | Maximum theoretical |
| Local NVMe RAID0 | ~25 GB/s | ~20-30 min | **Recommended for reliability** |
| Host RAM (tmpfs) | ~200 GB/s | ~3 min | If RAM available |

### 5.2 Why Our FSx Was Slow: The Striping Problem

**Root Cause**: Files downloaded to FSx Lustre used default single-OST striping.

```
SLOW (what we had):
  [File] → [1 OST] → ~240 MB/s
           ↓
        Bottleneck!

FAST (with proper striping):
  [File] → [OST 0] ──┐
         → [OST 1] ──┼→ ~5-10 GB/s aggregate
         → [OST 2] ──┤
         → ...      ──┘
```

**Our FSx Configuration**:
- 98 TiB PERSISTENT_2 with 1000 MB/s/TiB = ~98 GB/s aggregate
- ~21 OSTs (4.8 TiB each)
- Per-OST throughput: ~5 Gbps (~625 MB/s max, ~240 MB/s observed)

**Solution**: Stripe files across all OSTs before loading:
```bash
# Create striped directory
mkdir -p /mnt/fsx/models/Kimi-K2.5-striped
lfs setstripe -c -1 -S 1M /mnt/fsx/models/Kimi-K2.5-striped/

# Copy with striping (one-time operation)
rsync -av /mnt/fsx/models/Kimi-K2.5/ /mnt/fsx/models/Kimi-K2.5-striped/

# Use striped copy for loading
--model /mnt/fsx/models/Kimi-K2.5-striped
```

See `scripts/stripe-model-fsx.sh` for automated striping.

### 5.3 NVMe RAID0 Setup (P5e)
```bash
# P5e has 8x 3.84TB NVMe = ~30TB total
mdadm --create /dev/md0 --level=0 --raid-devices=8 \
    /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 \
    /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1 /dev/nvme8n1
mkfs.xfs -f /dev/md0
mount /dev/md0 /mnt/nvme

# Copy model (one-time, ~5-15 min)
rsync -av --progress /mnt/fsx/models/Kimi-K2.5/ /mnt/nvme/models/Kimi-K2.5/
```

See `scripts/setup-nvme-model.sh` for automated NVMe setup.

### 5.4 Why NVMe > FSx for Loading
1. **Parallel shard reads**: vLLM 0.15.1+ uses ThreadPoolExecutor
2. **No network latency**: Direct PCIe access
3. **GDS on NVMe**: GPU can DMA directly from NVMe
4. **Consistent throughput**: No contention from other FSx users

### 5.5 FSx Client-Side Tuning for Large Model Loading

Per [AWS HyperPod FSx best practices](https://awslabs.github.io/ai-on-sagemaker-hyperpod/docs/Tips/Common/Fsx%20for%20Lustre%20best%20practices), the default Lustre client settings are not tuned for large I/O on high-memory, high-vCPU instances like P5e.

**Apply immediately after mount** (these do not persist across reboots):
```bash
# Increase dirty page buffer — allows more write buffering before flush
# Helps when downloading model to FSx or writing KV cache
sudo lctl set_param osc.*.max_dirty_mb=64

# P5e has 2 TiB RAM — tune LRU for large memory systems (>64 GiB)
sudo lctl set_param ldlm.namespaces.*.lru_max_age=600000
sudo lctl set_param ldlm.namespaces.*.lru_size=$((100 * $(nproc)))  # 100 * 192 = 19200 on P5e

# Increase in-flight I/O operations for parallel shard reads
sudo lctl set_param osc.*.max_rpcs_in_flight=32
sudo lctl set_param mdc.*.max_rpcs_in_flight=64
```

**Kernel module tuning** (requires reboot — for P5e with >64 vCPUs):
```bash
# /etc/modprobe.d/lustre.conf
options ptlrpc ptl_send_rpc=256
options ksocklnd nscheds=8
```

**Persist tuning across reboots** via cron (since `lctl set_param` is volatile):
```bash
# /etc/cron.d/lustre-tuning
@reboot root sleep 30 && lctl set_param osc.*.max_dirty_mb=64 && \
  lctl set_param ldlm.namespaces.*.lru_max_age=600000 && \
  lctl set_param ldlm.namespaces.*.lru_size=19200 && \
  lctl set_param osc.*.max_rpcs_in_flight=32
```

**Monitor OST throughput distribution** via CloudWatch:
- Check `DataReadBytes` and `DataWriteBytes` per OST
- Uneven distribution indicates striping issues
- Target: all OSTs within 20% of average throughput

**Impact on model loading**:
| Setting | Before Tuning | After Tuning | Effect |
|---------|--------------|--------------|--------|
| `max_dirty_mb` | 32 (default) | 64 | 2x write buffer |
| `lru_size` | ~100 (default) | 19200 | Fewer lock evictions on large systems |
| `max_rpcs_in_flight` | 8 (default) | 32 | 4x concurrent I/O operations |

### 5.6 I/O Patterns: Model Loading vs KV Cache Offloading

The I/O bottlenecks from model loading and KV cache offloading are related but have different characteristics:

| Characteristic | Model Loading | KV Cache Offloading |
|----------------|--------------|---------------------|
| **Frequency** | One-time at startup | Continuous during inference |
| **Direction** | Read-only (disk → GPU) | Read/write (bidirectional) |
| **I/O size** | Large sequential (8-10 GB per shard) | Small random blocks (KV cache chunks) |
| **Total volume** | ~540 GB (fixed) | Varies with context length and concurrency |
| **Latency sensitivity** | Tolerant (startup cost) | Critical (affects TTFT and ITL) |
| **Bandwidth need** | Bulk throughput | Low-latency random access |

**Shared problems**:
- FSx striping matters for both — unstriped files on a single OST bottleneck regardless of access pattern
- PCIe saturation affects both — our benchmarks showed ~6x throughput degradation with CPU-based KV cache offloading due to PCIe bandwidth contention between model weights and KV transfers
- NVMe is preferred for both, but for different reasons: loading needs bulk bandwidth, KV cache needs low latency

**KV cache-specific I/O concerns**:
- LMCache uses chunked transfers (configurable via `lmcache_chunk_size`, default 256 tokens) that are better suited to FSx's striped layout than native vLLM's `--swap-space`
- Native vLLM swap writes KV blocks sequentially which can create write amplification on Lustre
- For FSx-backed KV cache, the same client tuning from section 5.5 applies — increase `max_rpcs_in_flight` for concurrent cache block transfers

**Recommendation**: Use NVMe for model loading, FSx (with LMCache) for KV cache sharing across pods. This separates the two I/O patterns onto different storage backends.

---

## 6. Expected Loading Timeline

### 6.1 With Optimizations (NVMe + enforce-eager)
| Phase | Duration | Progress Indicator |
|-------|----------|-------------------|
| Config loading | 30s | "Loading model config" |
| Tokenizer init | 1-2 min | "Loading tokenizer" |
| Weight loading | 20-30 min | "Loading checkpoint shards: X/64" |
| KV cache alloc | 1 min | "CUDA memory allocated" |
| **Total** | **25-35 min** | "Application startup complete" |

### 6.2 Without Optimizations (FSx + torch.compile)
| Phase | Duration | Notes |
|-------|----------|-------|
| Config loading | 30s | |
| Tokenizer init | 1-2 min | |
| Weight loading | 3-4 hours | I/O bottleneck |
| torch.compile | 10-15 min | Can hang indefinitely |
| **Total** | **4+ hours** | Prone to timeout/failure |

---

## 7. Monitoring and Troubleshooting

### 7.1 Health Check Commands
```bash
# GPU memory during loading (should increase gradually)
watch -n 5 'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv'

# Host memory (should stay below 90%)
watch -n 5 'free -h'

# Process status (should show python with high CPU)
ps aux | grep python | grep -v grep

# Shard loading progress
tail -f /tmp/vllm.log | grep -E "Loading|Progress|shards"

# NCCL status (if distributed)
tail -f /tmp/vllm.log | grep -i nccl
```

### 7.2 Failure Diagnosis

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Stops at specific shard | Corrupted safetensor | Re-download model |
| Hangs after all shards | torch.compile | Add `--enforce-eager` |
| OOM during loading | Too many experts on one GPU | Check TP=8 |
| NCCL timeout | Large model + slow storage | Set NCCL_TIMEOUT=1800 |
| Silent termination | Process killed (OOM/timeout) | Check `dmesg`, increase swap |
| Memory slowly fills | Normal MoE loading | Wait, monitor progress |

### 7.3 Verify Successful Loading
```bash
# Check if server is ready
curl -s http://localhost:30080/health | jq .

# Check model info
curl -s http://localhost:30080/v1/models | jq .

# Test inference
curl http://localhost:30080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"moonshotai/Kimi-K2.5","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'
```

---

## 8. Production Launch Script

### 8.1 Complete Script
```bash
#!/bin/bash
# start-kimi-k2.5.sh - Production launch for Kimi K2.5 on P5e

set -e

MODEL_PATH="${MODEL_PATH:-/mnt/nvme/models/Kimi-K2.5}"
PORT="${PORT:-30080}"
LOG_FILE="${LOG_FILE:-/tmp/vllm-kimi.log}"

# Environment
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_TIMEOUT=1800
export VLLM_ATTENTION_BACKEND=FLASHINFER

# Kill existing
pkill -9 -f "vllm.*Kimi" 2>/dev/null || true
sleep 2

echo "Starting Kimi K2.5 on port $PORT..."
echo "Model: $MODEL_PATH"
echo "Log: $LOG_FILE"

# Launch in screen
screen -dmS vllm bash -c "python3.11 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --enforce-eager \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --swap-space 32 \
    --disable-log-requests \
    --port $PORT \
    2>&1 | tee $LOG_FILE"

echo "Launched in screen session 'vllm'"
echo ""
echo "Monitor: screen -r vllm"
echo "Logs:    tail -f $LOG_FILE"
echo "Health:  curl http://localhost:$PORT/health"
```

### 8.2 Systemd Service (Alternative)
```ini
# /etc/systemd/system/vllm-kimi.service
[Unit]
Description=vLLM Kimi K2.5 Server
After=network.target

[Service]
Type=simple
User=root
Environment="HF_HUB_OFFLINE=1"
Environment="NCCL_TIMEOUT=1800"
Environment="VLLM_ATTENTION_BACKEND=FLASHINFER"
ExecStart=/usr/bin/python3.11 -m vllm.entrypoints.openai.api_server \
    --model /mnt/nvme/models/Kimi-K2.5 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --enforce-eager \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --port 30080
Restart=on-failure
RestartSec=30
TimeoutStartSec=3600

[Install]
WantedBy=multi-user.target
```

---

## 9. Capacity Block Planning

### 9.1 Time Requirements
| Activity | Duration |
|----------|----------|
| Instance launch + EKS join | 5-10 min |
| FSx mount | 1-2 min |
| NVMe RAID setup | 2-3 min |
| Model copy FSx → NVMe | 10-15 min |
| Model loading (NVMe) | 25-35 min |
| Benchmark suite | 2-4 hours |
| **Minimum recommended** | **6 hours** |
| **Safe margin** | **8 hours** |

### 9.2 Capacity Block Strategy
1. Reserve **8-hour block** for comfortable margin
2. Run NVMe setup immediately after launch
3. Start model copy while setting up environment
4. Launch vLLM with `--enforce-eager`
5. Monitor first 30 min for loading completion
6. Only then start benchmarks

---

## 10. Summary: Key Takeaways

### Must-Have Configurations
1. **`--enforce-eager`** - Prevents torch.compile hangs
2. **`NCCL_TIMEOUT=1800`** - 30 min timeout for large models
3. **`--tensor-parallel-size 8`** - Match GPU count exactly
4. **`--gpu-memory-utilization 0.85`** - Leave MoE routing headroom
5. **`--trust-remote-code`** - Required for custom architecture

### Recommended Optimizations
1. **NVMe RAID0** - 8x faster than unstriped FSx for loading
2. **FSx striping** - If using FSx, stripe files with `lfs setstripe -c -1`
3. **Pre-stage model** - Copy to NVMe before starting vLLM
4. **Screen session** - Persists through SSH disconnects
5. **8-hour capacity block** - Safe margin for all activities

### Monitoring Checklist
- [ ] GPU memory increasing during load
- [ ] Host memory < 90%
- [ ] Shard progress in logs
- [ ] No NCCL errors
- [ ] Health endpoint responds after loading

---

## 11. Multi-Node MoE Inference

Kimi K2.5 fits on a single p5e.48xlarge (8x H200, 1.1 TB HBM), but multi-node becomes necessary for:
- Longer context lengths (>128K) requiring more KV cache memory
- Higher throughput via data parallelism across nodes
- Even larger MoE models that exceed single-node HBM

### 11.1 Parallelism Strategies

| Strategy | How It Works | When to Use |
|----------|-------------|-------------|
| **Tensor Parallelism (TP)** | Split weight matrices across GPUs within a node | Always — use TP=8 per node (NVLink is fast) |
| **Pipeline Parallelism (PP)** | Split layers across nodes | When model layers exceed single-node memory |
| **Expert Parallelism (EP)** | Distribute experts across nodes | MoE-native — reduces per-node expert count |
| **Data Parallelism (DP)** | Replicate model, split requests across replicas | When throughput > single-node capacity |

**Recommended for Kimi K2.5 on 2x p5e**:
```
Node 0: TP=8, layers 0-30  (PP rank 0)
Node 1: TP=8, layers 31-60 (PP rank 1)
```

Or with expert parallelism:
```
Node 0: TP=8, all layers, experts 0-191   (EP rank 0)
Node 1: TP=8, all layers, experts 192-383 (EP rank 1)
```

### 11.2 vLLM Multi-Node Setup

vLLM uses Ray for multi-node distributed inference:
```bash
# Node 0 (head)
ray start --head --port=6379

# Node 1 (worker)
ray start --address=<node0-ip>:6379

# Launch vLLM on head node
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/fsx/models/Kimi-K2.5-striped \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --enforce-eager \
    --trust-remote-code \
    --port 30080
```

### 11.3 Network Requirements (EFA/RDMA)

Inter-node communication for MoE is latency-sensitive because expert routing decisions require all-to-all communication every layer.

```bash
# Required environment variables for EFA on P5e
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_PROTO=Simple
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600          # 60 min — longer for multi-node
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
```

**Bandwidth requirements**:
| Communication | Pattern | Volume per Token | P5e EFA Bandwidth |
|---------------|---------|-----------------|-------------------|
| TP all-reduce | Intra-node (NVLink) | ~hidden_dim bytes | 900 GB/s (NVLink) |
| PP send/recv | Inter-node (EFA) | ~batch * hidden_dim | 3.2 Tbps (EFA v2) |
| EP all-to-all | Inter-node (EFA) | ~batch * expert_dim * 8 | 3.2 Tbps (EFA v2) |

### 11.4 Storage Implications

| Storage | Single Node | Multi-Node |
|---------|------------|------------|
| **NVMe** | Preferred for loading | Each node needs its own copy (10-15 min per node) |
| **FSx Lustre** | Slower but simpler | Shared across nodes — single copy, mount on all |
| **S3 + Run:ai Streamer** | Eliminates local copy | Each node streams independently — no coordination needed |

For multi-node, FSx or S3 streaming becomes more attractive since NVMe requires per-node model copies. With properly striped FSx (section 5.2) and client tuning (section 5.5), FSx aggregate bandwidth can serve multiple nodes simultaneously.

---

## 12. GB200 / Grace Blackwell Considerations

The optimizations in this document are largely specific to the H100/H200 + PCIe/EFA architecture. GB200 (Grace Blackwell) fundamentally changes the constraints.

### 12.1 GB200 NVL72 Architecture

| Attribute | P5e (H200) | GB200 NVL72 |
|-----------|-----------|-------------|
| **GPUs per rack** | 8 | 72 |
| **HBM per GPU** | 141 GB (HBM3e) | 192 GB (HBM3e) |
| **Total HBM** | 1.1 TB | 13.8 TB |
| **GPU-GPU interconnect** | NVLink 4 (900 GB/s) | NVLink 5 (1.8 TB/s) |
| **Inter-node** | EFA (3.2 Tbps) | NVSwitch (all-to-all within rack) |
| **CPU** | Intel Xeon (x86) | Grace (ARM), 480 GB LPDDR5X |
| **Local storage** | 8x 3.84 TB NVMe | Varies by config |

### 12.2 What Changes for MoE Loading

**Problems that go away**:
- **Storage I/O bottleneck**: 13.8 TB HBM means Kimi K2.5 (540 GB) fits in ~3 GPUs' memory. Model can be loaded once and broadcast via NVLink to all GPUs at 1.8 TB/s — no need for parallel shard reads from disk
- **NCCL timeout**: NVLink 5 within the rack is orders of magnitude faster than EFA. All-to-all for expert routing happens in microseconds, not milliseconds
- **Expert routing overhead**: NVSwitch provides full-bisection bandwidth between all 72 GPUs — no hotspot possible within the rack
- **KV cache offloading**: 13.8 TB HBM pool likely eliminates the need for disk/CPU KV cache offloading entirely

**Problems that remain**:
- **torch.compile hangs**: Still applies — `--enforce-eager` is hardware-independent
- **`--trust-remote-code`**: Still required for custom model architectures
- **Initial download**: 540 GB must still be fetched from S3/HuggingFace to the rack

**New considerations**:
- **Grace CPU memory tier**: 480 GB LPDDR5X per Grace CPU provides an intermediate memory tier (faster than NVMe, cheaper than HBM) — useful for KV cache overflow on extremely long contexts
- **TP scaling**: With 72 GPUs, TP=72 is possible but adds overhead. Better to use TP=8 + EP=9 (72 GPUs / 8 per TP group = 9 expert parallel groups)
- **Power and cost**: GB200 NVL72 is a full-rack solution — significantly higher cost than a single P5e for a model that fits on one node

### 12.3 Recommended Configuration for GB200

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/models/Kimi-K2.5 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 \       # Likely unnecessary — model fits
    --trust-remote-code \
    --enforce-eager \
    --gpu-memory-utilization 0.95 \    # More headroom available
    --max-model-len 131072 \           # Can push to 128K+ with 13.8 TB HBM
    --port 30080
```

### 12.4 When to Use GB200 vs P5e for MoE

| Scenario | Recommended | Rationale |
|----------|------------|-----------|
| Single Kimi K2.5 instance | P5e | Model fits, GB200 is overkill |
| Kimi K2.5 with 256K context | GB200 | KV cache at 256K exceeds P5e HBM |
| Multiple large MoE models | GB200 | 13.8 TB HBM hosts several models |
| Expert parallelism research | GB200 | NVSwitch enables efficient EP |
| Cost-sensitive benchmarking | P5e | Lower cost per hour |

**AWS availability**: As of February 2026, GB200-based instances (expected P6 family) are not yet generally available. Current best option is P5e (H200). The optimizations in this document (NVMe staging, FSx striping, enforce-eager) are specific to P5e and will be less relevant on GB200.
