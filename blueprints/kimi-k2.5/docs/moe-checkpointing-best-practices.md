# MoE Model Checkpointing Best Practices

**Date**: February 2026
**Scope**: Distributed checkpointing for Mixture-of-Experts models (100B+ parameters)
**Sources**: NVIDIA Megatron-Core, PyTorch DCP/TorchTitan, DeepSpeed, CoreWeave, Lambda Labs

---

## Executive Summary

MoE models present unique checkpointing challenges compared to dense models: significantly larger total parameter counts (e.g., Kimi K2.5 has 1T total but only 32B active), expert-parallel sharding across GPUs, and higher training instability requiring more frequent saves. This document covers best practices from NVIDIA, PyTorch, DeepSpeed, and neo-cloud providers for efficient, reliable MoE checkpointing.

---

## 1. Why MoE Checkpointing Is Different

### 1.1 Scale Amplification

| Model | Total Params | Active Params | Checkpoint Size (BF16) | Optimizer State (Adam, FP32) |
|-------|-------------|---------------|----------------------|----------------------------|
| Llama 3.1 70B (dense) | 70B | 70B | ~140 GB | ~840 GB |
| Mixtral 8x7B (MoE) | 47B | 13B | ~94 GB | ~564 GB |
| DeepSeek V3 (MoE) | 671B | 37B | ~1.3 TB | ~8 TB |
| Kimi K2.5 (MoE) | 1T | 32B | ~2 TB | ~12 TB |

The optimizer state is typically 3x the model size (Adam stores fp32 master weights + first/second moments). For a 1T MoE model, that is ~12 TB of optimizer state alone.

### 1.2 Expert Parallelism Adds a Sharding Dimension

Dense models use Tensor Parallelism (TP), Pipeline Parallelism (PP), and Data Parallelism (DP). MoE adds **Expert Parallelism (EP)** -- experts are distributed across EP ranks, creating a 4D parallelism space:

```
Dense:  (PP, TP, DP)          -- 3 dimensions
MoE:    (PP, TP, EP, DP)      -- 4 dimensions
```

Each rank holds a subset of experts. The checkpoint system must track which experts live on which rank and reconstruct them correctly on load -- especially when the parallelism configuration changes between save and load.

### 1.3 Training Instability Requires More Frequent Saves

MoE models are inherently less stable than dense models due to:
- **Router load imbalance** causing expert collapse (all tokens routed to few experts)
- **Sensitivity to router precision** (must use fp32 for gating, can use bf16 for experts)
- **Larger loss spikes** from routing instabilities

This means checkpointing frequency should be **higher** than for dense models, making checkpoint I/O efficiency even more critical.

---

## 2. Framework-Specific Approaches

### 2.1 NVIDIA Megatron-Core

Megatron-Core provides the most mature MoE checkpoint implementation via its distributed checkpointing system.

**Key Data Structure: `ShardedTensor`**

Each tensor is annotated with its global shape, local shard coordinates, and a `replica_id` 3-tuple `(PP_rank, TP_rank, DP_rank)`:

```python
# Megatron-Core expert checkpoint mapping (simplified)
# Each expert's local weights are mapped to their global position
local_expert_offset = ep_rank * num_local_experts

# FC1 weights: sharded along dim 1 (output features)
# FC2 weights: sharded along dim 0 (input features)
ShardedTensor(
    key=f"layers.{layer}.experts.weight",
    global_shape=(num_global_experts, hidden_dim, expert_dim),
    local_offset=(local_expert_offset, tp_offset, 0),
    replica_id=(pp_rank, tp_rank, dp_rank),
    allreduce=False  # No gradient sync across EP ranks
)
```

**Two Expert Implementations with Checkpoint Interchangeability:**

| Implementation | Storage | Performance | Checkpoint Format |
|----------------|---------|-------------|-------------------|
| **GroupedMLP** (fused) | Single tensor per layer | Faster (fused kernels) | Sliced from fused tensor |
| **SequentialMLP** (individual) | Separate tensor per expert | Simpler | Individual state dicts |

Both formats are interconvertible via prefix replacement -- you can save with GroupedMLP and load with SequentialMLP or vice versa.

**Recommended Configuration (NeMo):**
```yaml
# Checkpoint format
dist_ckpt_format: 'torch_dist'            # Recommended over 'zarr'
dist_ckpt_load_on_device: True
dist_ckpt_parallel_save: True              # Parallel writes across ranks
dist_ckpt_parallel_load: True              # Parallel reads across ranks
dist_ckpt_torch_dist_multiproc: 2          # Writer threads per rank
dist_ckpt_assume_constant_structure: False  # Set True after first 2 saves

# Reshardability
ckpt_optim_fully_reshardable: True         # Essential for changing EP/TP/PP
dist_ckpt_parallel_dist_opt: True          # Parallel optimizer save
```

**Async Checkpointing:**

Megatron-Core implements two async strategies:

| Strategy | How It Works | When to Use |
|----------|-------------|-------------|
| **TemporalAsyncCaller** | Spawns new process per checkpoint | Infrequent saves |
| **PersistentAsyncCaller** | Persistent worker process with queues | Production training (recommended) |

Execution flow:
1. **Preload**: Stage tensors from GPU to CPU (`preload_fn()`)
2. **Async write**: Background process writes to storage
3. **Finalize**: Synchronous callbacks (metadata, etc.)
4. **Barrier**: `all_reduce` ensures all ranks complete

Performance: ~1 second per checkpoint with distributed fused optimizer (Megatron), ~3 seconds with Apex distributed Adam.

### 2.2 PyTorch Distributed Checkpoint (DCP) / TorchTitan

PyTorch's native approach uses **DTensor** -- tensors annotated with placement information on a **DeviceMesh**.

**TorchTitan's 5D Mesh for MoE:**

```python
# Dense parameters mesh
("pp", "dp_replicate", "fsdp", "tp")

# Sparse (expert) parameters mesh
("pp", "dp_replicate", "efsdp", "ep", "etp")
```

Where:
- `efsdp` = expert-FSDP (FSDP sharding within EP groups)
- `ep` = expert parallel dimension
- `etp` = expert tensor parallel (TP within each expert)
- `efsdp = fsdp * tp // (etp * ep)` -- automatically computed

**Expert Weight Distribution as DTensors:**

```python
# EP only: experts sharded along first axis
expert_weight: DTensor with placement [Shard(0)]  # split by expert index

# TP only: weights sharded within each expert
w1: DTensor with placement [Shard(1)]  # column-wise
w2: DTensor with placement [Shard(2)]  # row-wise

# EP + TP: combined sharding
w1: DTensor with placement [Shard(0), Shard(1)]  # expert + column
```

**FSDP2 Integration:**

TorchTitan applies FSDP2 with **separate meshes** for dense and expert parameters:

```python
# Dense parameters use standard FSDP mesh
dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])

# Expert parameters use EP-aware FSDP mesh
edp_mesh = parallel_dims.get_mesh(["dp_replicate", "efsdp"])

apply_fsdp(
    model,
    dp_mesh,
    ep_degree=parallel_dims.ep,
    edp_mesh=edp_mesh,  # Expert-specific sharding mesh
    ...
)
```

**Async Checkpointing (TorchTitan):**

Three modes available via `torchtitan/components/checkpoint.py`:

| Mode | Mechanism | Performance |
|------|-----------|-------------|
| `DISABLED` | Synchronous `dcp.save()` | Blocks training |
| `ASYNC` | Background thread via `dcp.async_save()` | Good |
| `ASYNC_WITH_PINNED_MEM` | Pinned memory + CUDA streams | Best |

The `ASYNC_WITH_PINNED_MEM` mode uses `DefaultStager` with two-phase futures:
```python
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

stager = DefaultStager(StagingOptions(
    use_pinned_memory=True,     # Page-locked host memory for fast GPU->CPU
    use_shared_memory=True,     # Multi-process access to staged data
    use_async_staging=True,     # Background thread via ThreadPoolExecutor
    use_non_blocking_copy=True  # CUDA stream-based async copies
))

# Returns two separate futures:
response = dcp.async_save(state_dict, stager=stager)
response.staging_completion  # When GPU memory is freed (training can resume)
response.upload_completion   # When data reaches storage (checkpoint durable)
```

**Other Checkpoint Features:**
- DCP resharding: load checkpoint saved with N GPUs on M GPUs
- Seed checkpoints: create from single CPU, load on any GPU count
- HuggingFace format conversion scripts included
- Plan caching: `DefaultSavePlanner(enable_plan_caching=True)` avoids redundant coordinator communication on repeated saves

### 2.3 DeepSpeed

DeepSpeed provides MoE support through ZeRO Stages 1 and 2 (ZeRO-3 is NOT supported with MoE).

**Expert Parallelism in DeepSpeed:**

```python
# DeepSpeed creates separate communication groups for experts
expert_parallel_group       # For expert weight all-to-all
expert_data_parallel_group  # For gradient averaging within EP groups

# Partition count adjusted for MoE groups
partition_count = dist.get_world_size(group=expert_dp_process_group)
```

**Universal Checkpointing (key feature for MoE):**

DeepSpeed's Universal Checkpoint format decouples checkpoint structure from parallelism configuration:

```bash
# Step 1: Save ZeRO checkpoint normally
trainer.save_checkpoint(checkpoint_dir)

# Step 2: Convert to universal format
python ds_to_universal.py --input_folder ckpt/ --output_folder universal_ckpt/

# Step 3: Resume with different parallelism
deepspeed --num_gpus 16 train.py --universal-checkpoint --load universal_ckpt/
```

Configuration:
```json
{
    "checkpoint": {
        "load_universal": true,
        "use_node_local_storage": true
    }
}
```

**Weight Recovery Utilities:**
```python
# Extract fp32 weights from ZeRO checkpoint
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
```

---

## 3. Storage Architecture for MoE Checkpoints

### 3.1 Storage Tier Recommendations

| Tier | Use Case | Technology | Throughput |
|------|----------|-----------|------------|
| **Hot** (training) | Frequent intermediate checkpoints | Local NVMe RAID0, parallel FS | 10-25 GB/s |
| **Warm** (durable) | Important checkpoints during training | Shared filesystem (FSx, VAST) | 5-10 GB/s |
| **Cold** (archive) | Final models, long-term retention | Object storage (S3) | Variable |

### 3.2 Provider-Specific Storage

**AWS (our setup):**
- **NVMe RAID0**: 8x 3.84TB on P5e = ~25 GB/s read. Best for checkpoint staging.
- **FSx for Lustre**: Stripe with `lfs setstripe -c -1` for parallel shard writes. See MoE Loading best practices doc for client tuning.
- **S3**: Final checkpoint archive. Use Run:ai Streamer for direct GPU streaming.

**CoreWeave:**
- **AI Object Storage** (S3-compatible): Primary checkpoint store. Up to 2 GB/s per GPU.
  - LOTA (Local Object Transport Accelerator) caches reads at ~7 GB/s per GPU
  - Tiered pricing: Hot (0-7 days), Warm (7-30 days), Cold (30+ days)
- **Distributed File Storage** (VAST-backed): ~11 GiB/s single-node, ~8 GiB/s per node at scale
  - Automatic snapshots every 6 hours (72-hour retention) -- free checkpoint safety net
  - Async bulk deletion via `.vast_trash` for cycling large MoE checkpoints
- **Tensorizer**: Open-source fast serialization, ~5 GB/s wire-speed loading
  ```python
  from tensorizer import TensorSerializer, TensorDeserializer
  # Save directly to S3
  serializer = TensorSerializer("s3://bucket/model.tensors")
  serializer.write_module(model)
  # Load with lazy tensor support (useful for selective expert loading)
  deserializer = TensorDeserializer("s3://bucket/model.tensors", device="cuda:0")
  deserializer.load_into_module(model)
  ```

**Lambda Labs:**
- **Shared Filesystem**: NFS-based, $0.20/GB/month, mounted across all cluster nodes
- **S3 Adapter**: S3 API access to Lambda Filesystem without provisioning VMs
  - Supports multipart uploads for large checkpoint files
  - Use for exporting checkpoints post-training
- **Local NVMe**: 24 TB per node on H100 1-Click Clusters
- **NFS caching**: `cachefilesd` for repeated model access across nodes

### 3.3 Checkpoint I/O Pattern

MoE checkpointing has a distinct I/O profile:

```
Model (read-heavy, write-once):     GPU HBM → CPU RAM → Storage
Optimizer (write-heavy, periodic):  GPU HBM → CPU RAM → Storage  (3x model size!)
Router (tiny, critical):            GPU HBM → CPU RAM → Storage  (must be fp32)
```

**Separation strategy**: Use fast local storage (NVMe) for the synchronous preload (GPU→CPU), then async write to durable storage (shared FS or object storage). This is what Megatron-Core's PersistentAsyncCaller implements.

---

## 4. Checkpoint Format Comparison

| Format | Strengths | Weaknesses | Best For |
|--------|-----------|------------|----------|
| **PyTorch DCP** (`torch.distributed.checkpoint`) | Native sharding, resharding support, async save | PyTorch 2.x+ only | Active training |
| **Megatron Distributed** (`torch_dist`) | Mature EP support, async, parallel save | Megatron-specific | Megatron-Core training |
| **DeepSpeed Universal** | Cross-parallelism portability | Requires conversion step | Changing cluster configs |
| **Safetensors** | Memory-mapped, zero-copy, HF ecosystem | No sharding awareness | Final model distribution |
| **CoreWeave Tensorizer** | Wire-speed S3 streaming, lazy loading | Tensors only (no optimizer) | Fast model deployment |

**Recommendation**: Use **PyTorch DCP or Megatron Distributed** during training, **DeepSpeed Universal** when changing parallelism, **Safetensors** for final distribution.

---

## 5. Resharding: Changing Parallelism Between Checkpoints

A critical MoE capability is loading a checkpoint saved with one parallelism configuration into a different one (e.g., trained with EP=4, resume with EP=8).

### 5.1 How Resharding Works

Because each expert's `sharded_state_dict` records its global offset within the full `(num_experts, hidden_dim, ...)` tensor, the loading process computes new local slices:

```
Saved:  EP=4, 96 experts/rank    →  global tensor (384, hidden, expert_dim)
Loaded: EP=8, 48 experts/rank    →  each rank reads its 48 experts from global tensor
```

### 5.2 Supported Resharding Dimensions

| Framework | TP | PP | DP | EP |
|-----------|----|----|----|----|
| Megatron-Core | Yes | Yes | Yes | Yes |
| PyTorch DCP | Yes | Yes | Yes | Yes (via DTensor) |
| DeepSpeed Universal | Yes | Yes | Yes | Yes (via conversion) |

### 5.3 Resharding Configuration

**Megatron-Core:**
```yaml
ckpt_optim_fully_reshardable: True
dist_ckpt_parallel_dist_opt: True
```

**DeepSpeed:**
```json
{"checkpoint": {"load_universal": true}}
```

**PyTorch DCP**: Automatic via DTensor placement metadata.

---

## 6. Checkpoint Frequency and Monitoring

### 6.1 Recommended Frequencies

| Training Phase | Dense Model | MoE Model | Rationale |
|----------------|------------|-----------|-----------|
| Early (first 5-10% tokens) | Every 1000 steps | Every 200-500 steps | Highest instability risk |
| Stable training | Every 2000 steps | Every 500-1000 steps | Expert collapse risk persists |
| Fine-tuning | Every 500 steps | Every 200 steps | MoE needs higher LR, more volatile |

### 6.2 Defensive Checkpoint Triggers

Monitor these signals and save an extra checkpoint when they spike:

| Signal | Threshold | Action |
|--------|-----------|--------|
| **Auxiliary loss** spike | >2x running average | Immediate checkpoint |
| **Token drop rate** | >5-10% | Checkpoint + investigate |
| **Expert utilization** imbalance | Most-busy expert >60% more tokens than least-busy | Checkpoint + adjust aux loss |
| **Router logit magnitude** | Sudden increase | Checkpoint (z-loss may be insufficient) |
| **Loss spike** | >3 std deviations | Checkpoint before and after |

### 6.3 Checkpoint Retention Strategy

```
Step N:      Save full checkpoint (model + optimizer + scheduler)
Step N+500:  Save lightweight checkpoint (model only, last_save_model_only=True)
Step N+1000: Save full checkpoint, delete checkpoint from N-2000
...
Keep: last 3 full checkpoints + last 5 lightweight checkpoints
```

---

## 7. Reducing Checkpoint Size

### 7.1 Optimizer State Sharding

The optimizer is the largest component (~3x model size). Strategies:

| Strategy | Savings | Framework |
|----------|---------|-----------|
| **ZeRO Stage 1** | Optimizer partitioned across DP ranks | DeepSpeed |
| **ZeRO Stage 2** | + gradient partitioning | DeepSpeed |
| **FSDP2** | Full sharding across DP ranks | PyTorch |
| **Distributed optimizer** | Each rank saves its partition | Megatron-Core |

With 8 DP ranks, each rank saves 1/8th of the optimizer state.

### 7.2 Precision Reduction for Intermediate Checkpoints

```python
# Save model weights in bf16 for intermediate checkpoints
# Keep fp32 optimizer state only in full checkpoints
config.export_dtype = "bfloat16"
config.last_save_model_only = True  # Skip optimizer for lightweight saves
```

### 7.3 Avoiding Checkpoint Bloat

DeepSpeed's ZeRO can cause size inflation due to tensor flattening interacting with PyTorch's storage management. Use:
```python
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
```

---

## 8. Recovery and Resumption

### 8.1 MoE-Specific Recovery Considerations

1. **Router state integrity**: Always verify gating network weights on load. Corrupted router weights break all expert routing.

2. **Auxiliary loss warmup on resume**: After loading a checkpoint, warm up the aux loss coefficient over 100-200 steps to avoid sudden load imbalance corrections.

3. **Expert group reconfiguration**: When changing EP degree, use universal checkpoints (DeepSpeed) or reshardable format (Megatron-Core). Manual expert tensor splitting is error-prone.

4. **Multi-node recovery without shared filesystem**:
   ```json
   {"checkpoint": {"use_node_local_storage": true}}
   ```

### 8.2 Recovery Checklist

- [ ] Verify all shard files are present and non-corrupted
- [ ] Check router weights are fp32 (not accidentally saved in bf16)
- [ ] Confirm expert count matches between checkpoint and model config
- [ ] If changing EP/TP/PP, use reshardable checkpoint format
- [ ] Warm up aux loss coefficient for first 100-200 steps after resume
- [ ] Monitor expert utilization immediately after resume

---

## 9. Production Checkpoint Pipeline

### 9.1 Two-Tier Architecture

```
Training Loop
    │
    ├─ Every 500 steps: Async checkpoint to NVMe (fast, local)
    │   └─ PersistentAsyncCaller (Megatron) or DCP async (PyTorch)
    │       └─ GPU → CPU staging (preload_fn)
    │           └─ Background write to /mnt/nvme/checkpoints/
    │
    ├─ Every 2000 steps: Promote to durable storage (background)
    │   └─ rsync or S3 upload from NVMe to FSx/S3
    │       └─ Runs as separate process, does not block training
    │
    └─ End of training: Convert to safetensors for distribution
        └─ scripts/checkpoint_conversion/convert_to_hf.py (TorchTitan)
        └─ zero_to_fp32.py (DeepSpeed)
```

### 9.2 Example: Megatron-Core Async Checkpoint with NVMe Staging

```bash
# Environment
export NCCL_TIMEOUT=3600

# Training launch with async checkpointing
python -m megatron.core.train \
    --num-experts 384 \
    --expert-model-parallel-size 8 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --save /mnt/nvme/checkpoints \
    --save-interval 500 \
    --dist-ckpt-format torch_dist \
    --dist-ckpt-parallel-save \
    --async-save \
    --ckpt-fully-parallel-save \
    --ckpt-optim-fully-reshardable
```

### 9.3 Example: DeepSpeed MoE Checkpoint Config

```json
{
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true
    },
    "checkpoint": {
        "load_universal": false,
        "use_node_local_storage": false,
        "tag_validation": "warn"
    },
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto"
}
```

---

## 10. Training Stability Best Practices (Reducing Checkpoint Needs)

Better stability means fewer defensive checkpoints. From Switch Transformer, ST-MoE, and HuggingFace research:

```python
moe_stability_config = {
    # Router precision -- MUST be fp32 to avoid routing errors
    "router_precision": "float32",

    # Expert computation can use reduced precision
    "expert_precision": "bfloat16",

    # Z-loss: penalizes large router logits, reduces roundoff errors
    "moe_z_loss_coeff": 0.01,

    # Auxiliary load-balancing loss: ensures uniform routing
    "moe_aux_loss_coeff": 0.01,

    # Expert capacity factor: 1.0-1.25 recommended
    # Formula: (tokens_per_batch / num_experts) * capacity_factor
    "moe_expert_capacity_factor": 1.25,

    # Higher dropout in expert layers (MoE overfits more easily)
    "expert_dropout": 0.2,

    # Token dispatch strategy
    "moe_token_dispatcher_type": "alltoall",  # or "allgather"
}
```

---

## 11. Comparison: AWS vs Neo-Cloud Providers

| Capability | AWS (P5e + FSx) | CoreWeave | Lambda Labs |
|------------|-----------------|-----------|-------------|
| **Fast local storage** | NVMe RAID0 (~25 GB/s) | Local NVMe | 24 TB NVMe per node |
| **Shared filesystem** | FSx for Lustre (stripe-tunable) | VAST Distributed FS (~11 GiB/s) | NFS (shared filesystem) |
| **Object storage** | S3 | AI Object Storage + LOTA | Filesystem S3 Adapter |
| **Checkpoint acceleration** | Run:ai Streamer, GDS | Tensorizer (~5 GB/s) | cachefilesd for NFS |
| **Auto-snapshots** | FSx backups (manual) | Every 6h, 72h retention (free) | Manual |
| **Cross-cluster access** | S3 (global) | Object storage only | S3 Adapter |
| **GPU interconnect** | EFA v2 (3.2 Tbps) | InfiniBand 400 Gb/s | InfiniBand 400 Gb/s |

---

## 12. Key Constraints and Gotchas

1. **DeepSpeed MoE does NOT support ZeRO Stage 3** -- use Stage 1 or 2 only
2. **Contiguous gradients must be `True`** for MoE with ZeRO Stage 2
3. **Router weights MUST be fp32** -- bf16 causes routing errors and expert collapse
4. **`--enforce-eager` for loading** -- torch.compile can hang during MoE weight loading (see MoE Loading best practices)
5. **NCCL timeout**: Set to 3600s+ for multi-node MoE checkpointing
6. **Checkpoint size estimation**: Total params x 2 bytes (BF16 weights) + Total params x 12 bytes (Adam optimizer) = ~14 bytes per parameter
7. **FSx striping**: Without `lfs setstripe -c -1`, checkpoint writes bottleneck on a single OST
8. **`clone_tensors_for_torch_save()`**: Required in DeepSpeed to avoid checkpoint size inflation from ZeRO's tensor flattening

---

## 13. Summary: Decision Matrix

| Decision | Recommendation |
|----------|---------------|
| **Checkpoint format** | PyTorch DCP or Megatron `torch_dist` for training; safetensors for distribution |
| **Async checkpointing** | Always -- use PersistentAsyncCaller (Megatron) or DCP async (PyTorch) |
| **Checkpoint frequency** | 500-1000 steps (2x more frequent than equivalent dense model) |
| **Storage for checkpoints** | NVMe for async staging; shared FS or S3 for durable storage |
| **Reshardability** | Always enable -- you will change parallelism at some point |
| **Optimizer checkpointing** | Shard across DP ranks (ZeRO-1/2 or FSDP2) |
| **Router precision** | fp32 always, even when model uses bf16 |
| **Recovery strategy** | Keep last 3 full checkpoints + 5 lightweight checkpoints |

---

## References

- [NVIDIA Megatron-Core Distributed Checkpointing](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/dist_checkpointing)
- [PyTorch Distributed Checkpoint (DCP)](https://pytorch.org/docs/stable/distributed.checkpoint.html)
- [TorchTitan Expert Parallel](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/expert_parallel.py)
- [DeepSpeed MoE Tutorial](https://www.deepspeed.ai/tutorials/mixture-of-experts/)
- [DeepSpeed Universal Checkpointing](https://arxiv.org/abs/2406.18820)
- [CoreWeave AI Object Storage](https://docs.coreweave.com/)
- [CoreWeave Tensorizer](https://github.com/coreweave/tensorizer)
- [Lambda Labs 1-Click Clusters](https://lambda.ai/blog/introducing-lambda-1-click-clusters-a-new-way-to-train-large-ai-models)
- [HuggingFace MoE Guide](https://huggingface.co/blog/moe)
- [Switch Transformer Paper](https://arxiv.org/abs/2101.03961)
