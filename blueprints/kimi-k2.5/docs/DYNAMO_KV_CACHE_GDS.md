# NVIDIA Dynamo KV Cache Manager with GDS

## Overview

Dynamo's KV Block Manager (KVBM) implements tiered KV cache offloading with GPU Direct Storage (GDS) support via NIXL. On a single node like p5e.48xlarge, this enables direct GPU-to-NVMe transfers for KV cache blocks without CPU involvement.

## Storage Tiers

KVBM organizes storage in four tiers:

| Tier | Pool | Storage | Latency |
|------|------|---------|---------|
| G1 | Device Pool | GPU HBM (H200 143GB) | ~ns |
| G2 | Host Pool | CPU pinned DRAM | ~us |
| G3 | Disk Pool | Local NVMe via NIXL+GDS | ~10us |
| G4 | Remote | Cloud/networked storage | ~ms |

For our p5e.48xlarge:
- **G1**: 1144GB total HBM across 8x H200 (~72GB used by model, ~1072GB available)
- **G2**: Up to 2TB host DRAM
- **G3**: Local NVMe at `/mnt/nvme` (3.8TB, ~microsecond GDS latency)
- **G4**: FSx Lustre at `/mnt/fsx` (overflow/persistence)

## GDS Support

### How It Works

Dynamo uses NIXL's **GDS_MT** (multi-threaded GDS) backend for disk tier transfers:

- **Device → Disk (offload)**: NIXL Write via GDS when available, POSIX fallback otherwise
- **Disk → Device (onboard)**: NIXL Read via GDS, bypassing CPU entirely
- **`bypass_cpu_mem` flag**: Enables direct G1↔G3 transfers, skipping host memory

GDS capability is **auto-detected at runtime**. Dynamo performs a real 4096-byte test transfer between device and disk to verify GDS works. The result is cached in a static `GDS_SUPPORTED` variable.

### Fallback Behavior

If GDS is not available, transfers use a 2-hop path:
1. Device → Host (CUDA memcpy)
2. Host → Disk (POSIX I/O with `O_DIRECT`)

### NIXL GDS Backends

NIXL provides two GDS plugins:

1. **`cuda_gds`**: Basic GDS backend using cuFile APIs
2. **`gds_mt`**: Multi-threaded variant using TaskFlow (used by Dynamo)

Both support batched transfers (default 128 requests/batch, 16 batch pool). Max request size defaults to 16MB with automatic chunking for larger transfers.

## Configuration

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `DYN_KVBM_DISK_CACHE_GB` | `500` | Disk tier size (local NVMe) |
| `DYN_KVBM_DISK_CACHE_DIR` | `/mnt/nvme/kv-cache` | Disk cache directory |
| `DYN_KVBM_CPU_CACHE_GB` | `128` | Host DRAM tier size |
| `DYN_KVBM_NIXL_BACKEND_GDS_MT` | `true` | Enable multi-threaded GDS |
| `CUFILE_ENV_PATH_JSON` | `/etc/cufile.json` | cuFile driver config |
| `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER` | `false` | Keep frequency filter (default: blocks need >= 2 accesses before disk offload) |

### Disk Tier Path Selection

**FSx Lustre is the preferred G3 target** despite higher latency than local NVMe:

- `/mnt/fsx/kv-cache/dynamo` — FSx Lustre with GDS, persistent and shareable across instances
- `/mnt/nvme/kv-cache` — local NVMe, lower latency but ephemeral and instance-bound

FSx advantages over local NVMe for KV cache:
- **Persistent**: KV cache survives instance replacement/scaling events
- **Shareable**: Multiple p5e instances can mount the same FSx volume and share warm cache
- **Pre-warmed inference**: New instances immediately benefit from previously cached KV blocks
- **Multi-node ready**: Natural path to distributed KV cache when scaling out

Local NVMe can be used as an additional caching layer (G2.5) between DRAM and FSx if needed.

### cuFile Configuration (`/etc/cufile.json`)

```json
{
  "properties": {
    "max_direct_io_size_kb": 16384,
    "max_device_cache_size_kb": 2097152,
    "max_device_pinned_mem_size_kb": 1048576,
    "posix_pool_slab_size_kb": 4096,
    "posix_pool_slab_count": 128,
    "rdma_dev_addr_list": ["mlx5_0"],
    "allow_compat_mode": true
  }
}
```

## Offload Policy

KVBM applies frequency-based filtering before offloading to disk:
- Blocks must be accessed >= 2 times before eligible for disk offload
- Protects SSD lifespan and avoids offloading one-shot KV blocks
- Can be disabled with `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true` for benchmarking

## Async Write-Back Architecture

This is the critical architectural difference between Dynamo KVBM and LMCache, and the primary reason KVBM avoids the head-of-line blocking problem observed in our LMCache benchmarks.

### The LMCache Problem (Synchronous Write-Back)

LMCache's `LMCacheConnectorV1` with `kv_role: kv_both` writes KV blocks to FSx **synchronously within vLLM's single-threaded scheduling loop**. Each request must complete its FSx write before the scheduler can admit the next request. Under moderate concurrency (25 sessions at 60% KV utilization), this caused:

- Only 5 of 25 sessions running concurrently (vs all 25 for baseline)
- 13x worse foreground TTFT (5,330ms vs 403ms)
- 42x worse p99 TTFT (23,444ms vs 560ms)

The root cause is that LMCache's `put()` call executes data movement on the same thread as the vLLM scheduler, creating a bottleneck regardless of I/O backend speed (GDS at 9 GB/s or POSIX at 1-3 GB/s).

### How Dynamo KVBM Solves This

KVBM uses a **Leader-Worker architecture** that completely separates scheduling decisions from data movement:

**Leader (Scheduler-side)**: Runs in the vLLM scheduler process. Only does hash matching and metadata serialization — **no data copies**. The `build_connector_metadata()` call produces a serialized byte blob sent to workers via ZMQ.

**Worker (GPU-side)**: Receives metadata and enqueues transfers to the async transfer system. The `save_kv_layer` method collects offloading operations and enqueues them — it does not perform synchronous copies:

```rust
fn save_kv_layer(&mut self, _layer_name: String) -> anyhow::Result<()> {
    self.layers_complete += 1;
    if self.layers_complete == self.kv_cache_layers.len() {
        let offloading_operations = std::mem::take(&mut self.offloading_operations);
        event_sync_blocking(self.layer_events[self.layers_complete - 1]);
        // Enqueue to async scheduler — NOT synchronous copies
        for operation in &offloading_operations {
            self.connector.enqueue_request(operation.clone());
        }
    }
    Ok(())
}
```

### Async Transfer Pipeline

Each transfer direction runs on its own dedicated tokio async task with its own CUDA stream:

```rust
let device_offload_transfer_ctx = Arc::new(
    TransferContext::new(
        config.nixl_agent.clone(),
        cuda_ctx.new_stream()?,  // separate CUDA stream — does not block inference
        config.async_rt_handle.clone(),
        Some(pool_config.clone()),
    )
);
```

Transfer workers communicate via unbounded mpsc channels:

```
Scheduler loop                    Worker threads (async)
     │                                  │
     ├─ build_connector_metadata()      │
     │   (hash lookup + serialize)      │
     │                                  │
     ├─── ZMQ ──────────────────────►  enqueue_request()
     │                                  │
     │   (scheduler continues           ├─► device_offload_tx  ──► GPU→CPU task
     │    immediately)                  ├─► host_offload_tx    ──► CPU→NVMe task
     │                                  └─► disk_onboard_tx    ──► NVMe→GPU task
```

The `LocalTransferManager` runs up to `MAX_CONCURRENT_TRANSFERS = 4` simultaneously using `FuturesUnordered`, with 16-block batching:

```rust
let mut pending_transfers: FuturesUnordered<TransferFuture<...>> = FuturesUnordered::new();
loop {
    tokio::select! {
        Some(future) = futures_rx.recv() => {
            while pending_transfers.len() >= max_concurrent_transfers {
                if let Some(pending_transfer) = pending_transfers.next().await {
                    completion_manager.handle_complete(pending_transfer).await?;
                }
            }
            pending_transfers.push(future);
        }
        Some(pending_transfer) = pending_transfers.next() => {
            completion_manager.handle_complete(pending_transfer).await?;
        }
    }
}
```

### Architectural Comparison

| Aspect | LMCache (LMCacheConnectorV1) | Dynamo KVBM |
|--------|------------------------------|-------------|
| **Write-back model** | Synchronous in scheduler loop | Async via dedicated Rust worker threads |
| **Data movement thread** | Same thread as vLLM scheduler | Separate tokio tasks per transfer direction |
| **CUDA stream** | Shares with inference | Separate CUDA streams per transfer type |
| **Blocking behavior** | `put()` blocks scheduler until copy completes | `enqueue_request()` returns immediately |
| **Concurrency** | Sequential — one request at a time | Up to 4 concurrent transfers, 16-block batches |
| **Scheduler impact** | Data movement on scheduler thread | Scheduler only builds metadata (hash + serialize) |
| **Implementation** | Python (with some C++ for CUDA) | Rust (with Python bindings via PyO3) |

### Caveats

1. **`event_sync_blocking` on worker thread**: There is one synchronous wait point — the worker calls `event_sync_blocking()` on the last layer's CUDA event before enqueuing offloads. This ensures the forward pass data is written before offload begins, but this wait is on the **worker thread, not the scheduler thread**. The code has a TODO to move this to a CUDA stream wait.

2. **vLLM integration maturity**: KVBM with vLLM is supported but some code paths have TODOs and the connector API is still evolving.

3. **G4 remote tier**: Treated as opaque blob storage. Sophisticated hot/cold promotion between tiers is left to external storage providers via the event plane.

### Expected Impact on Our Workload

Based on our LMCache benchmarks (see `results/BENCHMARK_REPORT.md`), the async architecture should eliminate the moderate-pressure regression:

| Scenario | LMCache (sync) | Dynamo KVBM (async, expected) |
|----------|----------------|-------------------------------|
| 25 sessions × 24K tokens | 5 running, 20 waiting | 25 running, 0 waiting |
| Foreground TTFT | 5,330ms (13x worse) | ~400ms (near baseline) |
| Foreground p99 | 23,444ms | ~560ms (near baseline) |
| Throughput overhead | 45% slower | Minimal (async I/O overlaps with compute) |

The key prediction is that KVBM's scheduler loop stays fast (metadata only, ~microseconds) while data movement overlaps with inference on separate CUDA streams and CPU threads.

## Comparison with LMCache

| Feature | LMCache (GDS) | Dynamo KVBM |
|---------|---------------|-------------|
| GDS support | Via cuFile Python bindings | Via NIXL GDS_MT (native C++) |
| Tier count | 1 (GPU → FSx) | 4 (GPU → DRAM → NVMe → Remote) |
| CPU bypass | Yes (when cuFile available) | Yes (bypass_cpu_mem flag) |
| Auto-detection | Falls back to POSIX | Runtime capability test |
| Offload policy | Prefix-based | Frequency-based (>= 2 accesses) |
| Multi-node | Via NATS/etcd coordination | Via NIXL RDMA/UCX |
| Single-node optimized | No (designed for distributed) | Yes (tiered local offload) |
| Write-back model | Synchronous (blocks scheduler) | Asynchronous (dedicated worker threads) |
| Head-of-line blocking | Yes — 13x TTFT penalty at 60% KV | No — scheduler only does metadata |

## References

- [Dynamo KVBM Design Doc](https://github.com/ai-dynamo/dynamo/blob/main/docs/pages/design-docs/kvbm-design.md)
- [NIXL GDS Plugin](https://github.com/ai-dynamo/nixl/tree/main/src/plugins/gds_mt)
- [Dynamo KV Cache Manager Guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/kv_cache_manager.md)
