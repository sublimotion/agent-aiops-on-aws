# Storage, Interconnect & I/O Best Practices for ML Training

**Date**: February 2026
**Scope**: Storage types, interconnect technologies, RDMA/GDS techniques, and framework I/O backends for checkpoint and model loading across hyperscalers and neo-clouds
**Sources**: NVIDIA GDS/Magnum IO, PyTorch DCP, Megatron-Core, DeepSpeed, Run:ai, CoreWeave Tensorizer, KvikIO, cloud provider documentation

---

## Executive Summary

ML training checkpoint I/O is bottlenecked by storage, not compute or networking. Modern GPU clusters deliver 900-1,800 GB/s intra-node (NVLink) and 400+ GB/s inter-node (EFA/InfiniBand), but storage write throughput ranges from 5-150 GB/s depending on configuration. This document catalogs every storage type, interconnect technology, and I/O technique available across hyperscalers and neo-clouds, with specific configuration guidance for maximizing checkpoint throughput.

**Key finding**: No major ML framework currently uses GPUDirect Storage (GDS) for checkpoint I/O in production. Despite 2-8x bandwidth improvements, all frameworks use CPU-staged writes. KvikIO is the only tool providing native GDS bindings that could bridge this gap.

---

## 1. GPUDirect Storage (GDS)

### 1.1 Architecture

GDS enables direct DMA transfers between GPU memory and storage devices, bypassing the CPU bounce buffer entirely:

```
Traditional:  Storage --> CPU DRAM (bounce buffer) --> GPU VRAM  (two copies)
GDS:          Storage --> GPU VRAM                               (single DMA)
```

Components:
- **User-space library**: `libcufile.so` (cuFile API)
- **Kernel driver**: `nvidia-fs.ko` (not required for NVMe as of CUDA 12.8)
- **Storage drivers**: Modified to support GPU address translation via PCI BAR mappings

### 1.2 Performance

| Metric | Traditional Path | GDS Path | Improvement |
|--------|-----------------|----------|-------------|
| NVMe local (per PCIe tree) | 12.0-12.5 GB/s | 13.3 GB/s | ~1.1x |
| DGX-2 aggregate | ~100 GB/s | 215 GB/s | ~2x |
| cuDF CSV reader | baseline | -- | 8.8x |
| End-to-end latency (80 GB) | baseline | -- | 3.8x lower |
| CPU utilization | baseline | -- | 3x reduction |

### 1.3 Supported Filesystems

GDS supports: **Lustre** (FSx for Lustre, DDN EXAScaler), **WekaFS**, **IBM Spectrum Scale**, **NVMe/NVMe-oF**, **BeeGFS**, **NFSoRDMA**, **VAST**, **ScaleFlux CSD**.

Local filesystems (ext4/XFS) do NOT support GDS -- they fall back to POSIX compatibility mode.

### 1.4 Configuration (`/etc/cufile.json`)

```json
{
  "max_direct_io_size_kb": 16384,
  "max_device_cache_size_kb": 131072,
  "max_device_pinned_mem_size_kb": 33554432,
  "max_io_threads": 4,
  "max_request_parallelism": 4,
  "allow_compat_mode": false,
  "use_poll_mode": false,
  "rdma_dev_addr_list": ["10.0.0.1"],
  "rdma_load_balancing_policy": "RoundRobin",
  "rdma_dynamic_routing": true
}
```

**Alignment requirement**: All I/O benefits from **4KB alignment** across four dimensions: file offset, GPU buffer pointer, transfer size, and buffer offset. Unaligned operations trigger internal bounce buffers, negating the zero-copy advantage.

### 1.5 cuFile API Patterns

**Synchronous** (simplest):
```c
cuFileRead(fh, gpu_buffer, size, file_offset, buf_offset);
cuFileWrite(fh, gpu_buffer, size, file_offset, buf_offset);
```

**Async stream-ordered** (fire-and-forget into CUDA streams):
```c
cuFileReadAsync(fh, bufPtr, &size, &file_offset, &buf_offset, &bytes_read, stream);
cuFileWriteAsync(fh, bufPtr, &size, &file_offset, &buf_offset, &bytes_written, stream);
```

**Batch I/O** (best for parallel shard reads/writes):
```c
cuFileBatchIOSetUp(&batch, max_nr);
cuFileBatchIOSubmit(batch, nr, iocb_params, flags);
cuFileBatchIOGetStatus(batch, min_nr, &nr, events, &timeout);
cuFileBatchIODestroy(batch);
```

Batch I/O amortizes overhead across many operations -- recommended for applications handling multiple non-contiguous file offsets with smaller transfers (<64KB).

---

## 2. GPUDirect RDMA

### 2.1 Architecture

GPUDirect RDMA enables network adapters to read/write GPU memory directly via PCI BAR mappings, eliminating the CPU from the data path:

```
Without GPUDirect RDMA:  GPU VRAM -> PCIe -> CPU DRAM -> PCIe -> NIC -> Network  (two PCIe hops)
With GPUDirect RDMA:     GPU VRAM -> PCIe -> NIC -> Network                      (one PCIe hop)
```

**Hardware requirements**: Mellanox ConnectX-5/6/7, `nvidia-peermem` kernel module (since CUDA 11.4), IOMMU disabled or passthrough mode.

### 2.2 PCI Topology Impact

| PCI Topology | Performance |
|--------------|-------------|
| Same PCIe switch | Optimal |
| Single CPU/IOH | Acceptable but degraded |
| Cross-QPI/HT paths | Extremely limited or unreliable |

BAR pinning is expensive (milliseconds), so lazy unpinning strategies are used -- memory stays pinned across multiple transfers.

### 2.3 Provider Support

| Provider | GPUDirect RDMA | Transport |
|----------|---------------|-----------|
| AWS (P5/P5e/P5en) | Yes | EFA + SRD |
| Azure (ND H100 v5) | Yes | InfiniBand NDR |
| GCP (A3 Mega/Ultra) | Yes (via TCPX/O) | Custom TCP offload |
| CoreWeave | Yes | InfiniBand NDR |
| Lambda Labs | Yes | InfiniBand NDR |

---

## 3. Interconnect Technologies

### 3.1 Intra-Node: NVLink

| Generation | Platform | Bandwidth per GPU | Aggregate (8-GPU) | Switch |
|------------|----------|------------------|--------------------|--------|
| NVLink 4 | H100/H200 | 900 GB/s | 7.2 TB/s | 3rd gen NVSwitch (4 per node) |
| NVLink 5 | B200/GB200 | 1,800 GB/s | 14.4 TB/s | 4th gen NVSwitch (2 per node) |
| NVLink 5 | NVL72 (72 GPU) | 1,800 GB/s | 130 TB/s | 9 switch trays, 144 ports each |

**NVL72 changes the checkpoint equation**: With 72 GPUs in a single NVLink domain at 130 TB/s, model state consolidation for even a 1T-parameter model completes in ~15ms. The bottleneck shifts entirely to storage write speed. Inter-node network traffic for checkpointing is eliminated for models that fit within a single NVL72 domain.

**NVSwitch SHARP** (Scalable Hierarchical Aggregation and Reduction Protocol): Performs in-network all-reduce inside the switch fabric, accelerating state consolidation without consuming GPU compute cycles.

### 3.2 Intra-Node: PCIe Gen5

| Specification | Value |
|---------------|-------|
| Per-lane bandwidth | 3.94 GB/s (32 GT/s, 128/130b encoding) |
| x16 slot (GPU connection) | 31.5 GB/s per direction |
| x4 slot (NVMe drive) | ~7.88 GB/s per direction |

**PCIe is the hidden checkpoint bottleneck**: GPU PCIe bandwidth is shared between checkpoint I/O and ongoing compute. For a 140 GB checkpoint (70B model, FP16), the NVMe write phase takes ~2.5s with 8 striped drives, but PCIe contention with compute can extend this.

**Mitigation**: Async checkpointing (GPU-to-CPU copy during computation), NVLink-based staging (copy to a "staging GPU" via NVLink, write from there), or GDS (bypasses CPU entirely).

### 3.3 Inter-Node Comparison

| Technology | Bandwidth (8-GPU node) | Latency | GPUDirect RDMA | In-Network Reduction | Used By |
|-----------|----------------------|---------|----------------|---------------------|---------|
| InfiniBand NDR | 3,200 Gbps (8x400) | ~0.6 us | Yes | Yes (SHARP) | Azure, CoreWeave, Lambda |
| InfiniBand XDR | 6,400 Gbps (8x800) | ~0.6 us | Yes | Yes (SHARP) | GB200 NVL72 clusters |
| AWS EFA v2/v3 | 3,200 Gbps (32 NICs) | ~2-5 us | Yes | No | AWS P5/P5e/P5en |
| GCP GPUDirect-TCPX | 1,800 Gbps | ~3-5 us | Yes (custom) | No | GCP A3 Mega |
| GCP GPUDirect-TCPXO | 3,600 Gbps | ~3-5 us | Yes (custom) | No | GCP A3 Ultra |
| RoCE v2 | Up to 400 Gbps | ~1-2 us | Yes | No | On-prem Ethernet |
| Spectrum-X | Up to 400 Gbps | ~1-2 us | Yes (enhanced) | No | NVIDIA Ethernet |

**EFA vs InfiniBand for checkpoint I/O**: InfiniBand's 0.6 us latency advantage matters for barrier synchronization (small messages), but for bulk checkpoint data transfers (GB-scale), bandwidth is equivalent. EFA's SRD multipath routing provides more consistent tail latency under congestion. InfiniBand SHARP accelerates all-gather operations used for state consolidation but does not help the actual storage write (which is point-to-point).

### 3.4 Per-Provider Instance Specifications

#### AWS

| Instance | GPUs | Intra-node | EFA BW | NVMe | Notes |
|----------|------|-----------|--------|------|-------|
| P4d.24xlarge | 8x A100 40GB | NVSwitch (600 GB/s) | 400 Gbps | 8 TB | EFA v1 |
| P5.48xlarge | 8x H100 80GB | NVLink 4 (900 GB/s) | 3,200 Gbps | 30.4 TB | EFA v2, 32 NICs |
| P5e.48xlarge | 8x H100 80GB | NVLink 4 (900 GB/s) | 3,200 Gbps | 30.4 TB | EFA v2, 32 NICs |
| P5en.48xlarge | 8x H200 141GB | NVLink 4 (900 GB/s) | 3,200 Gbps | 30.4 TB | EFA v3, 35% lower latency |
| Trn2 UltraServer | 64x Trainium2 | NeuronLink | 12,800 Gbps | 32 TB | 64 NICs |

#### GCP

| Instance | GPUs | Intra-node | Network BW | Local SSD |
|----------|------|-----------|------------|-----------|
| A3 Mega | 8x H100 80GB | NVLink 4 (900 GB/s) | 1,800 Gbps | 6 TB |
| A3 Ultra | 8x H200 141GB | NVLink 4 (900 GB/s) | 3,600 Gbps | 12 TB |

#### Azure

| Instance | GPUs | Intra-node | IB BW | NVMe |
|----------|------|-----------|-------|------|
| ND H100 v5 | 8x H100 80GB | NVLink 4 (900 GB/s) | 3,200 Gbps (8x NDR 400) | 28 TB |
| ND MI300X v5 | 8x MI300X | -- | 3,200 Gbps | 28 TB |

Azure provides dedicated per-GPU InfiniBand ports (one ConnectX-7 per GPU), meaning checkpoint I/O from each GPU does not contend with other GPUs' network traffic.

---

## 4. Storage Types Across Providers

### 4.1 Master Comparison Table

| Provider | Storage | Protocol | Max Throughput | Latency | POSIX | Multi-Node | GDS | Best Checkpoint Use |
|----------|---------|----------|---------------|---------|-------|------------|-----|---------------------|
| **AWS** | FSx Lustre (Persistent 2) | Lustre | 150 GB/s (EFA+GDS) | Sub-ms | Yes | RWX | Yes | Large-scale distributed training |
| **AWS** | FSx Lustre (Scratch) | Lustre | 200 MB/s per TiB | Sub-ms | Yes | RWX | Yes | Ephemeral training jobs |
| **AWS** | EFS (Elastic) | NFS v4.1 | 60 GiB/s read / 5 GiB/s write | 250 us / 2.7 ms | Yes | RWX | No | Smaller checkpoints, simplicity |
| **AWS** | S3 | HTTP/REST | ~100 Gbps per instance | 10-100 ms | No | RWX | No | Cold storage, async upload |
| **AWS** | Instance NVMe (P5/P5e) | NVMe | ~14 GB/s (8 drives striped: 50+ GB/s) | <100 us | Yes | RWO | Yes | Local staging layer |
| **AWS** | EBS io2 Block Express | NVMe | 4 GB/s, 256K IOPS | <1 ms | Yes | RWO | No | Single-node persistent |
| **AWS** | FSx NetApp ONTAP | NFS/SMB/iSCSI | Multi-GB/s | Low ms | Yes | RWX | No | Enterprise multi-protocol |
| **AWS** | FSx OpenZFS | NFS | 1M+ IOPS | ~200 us | Yes | RWX | No | Instant snapshots, dev/test |
| **GCP** | Filestore Zonal | NFS | 26 GB/s read / 8.8 GB/s write (100 TiB) | Low ms | Yes | RWX | No | Medium-scale shared |
| **GCP** | Parallelstore | DAOS | 115 GiB/s read / 50 GiB/s write (100 TiB) | 0.3 ms | Yes | RWX | No | HPC/ML checkpoint I/O |
| **GCP** | Cloud Storage (GCS) | HTTP/REST | Auto-scales | 10-100 ms | No | RWX | No | Async archival |
| **GCP** | Local SSD (A3) | NVMe | 660 MiB/s per disk | <100 us | Yes | RWO | No | Local staging |
| **Azure** | Managed Lustre | Lustre | HPC-grade (scales with capacity) | Sub-ms | Yes | RWX | No* | Large-scale distributed |
| **Azure** | NetApp Files (Ultra) | NFS/SMB | 128 MiB/s per TiB | Sub-ms | Yes | RWX | No | High-perf shared FS |
| **Azure** | Azure Files Premium | NFS 4.1 | 10.3 GiB/s max | Low ms | Yes | RWX | No | Cloud-native shared |
| **Azure** | Blob Storage | HTTP/REST | Multi-GB/s | 10-100 ms | No | RWX | No | Async archival |
| **Azure** | NVMe Local (ND H100) | NVMe | ~1 TB local | <100 us | Yes | RWO | RDMA** | Local staging |
| **CoreWeave** | VAST (Distributed FS) | NFS | Multi-GB/s (all-flash) | Low ms | Yes | RWX | Possible*** | Shared training checkpoints |
| **CoreWeave** | AI Object Storage + LOTA | S3-compat | Multi-GB/s | Low ms (cached) | No | RWX | No | LOTA-accelerated reads |
| **Lambda** | Persistent Filesystem | Proprietary | Not published | Not published | Yes | RWX | No | Persistent between runs (up to 8 EB) |
| **Lambda** | S3 Adapter | S3-compat | Limited by tooling | Higher | No | RWX | No | External offload |

\* Azure ND VMs support GPUDirect RDMA via InfiniBand, but Managed Lustre GDS support is not documented.
\** GPUDirect RDMA supported via InfiniBand across nodes.
\*** VAST supports GDS on-premises; CoreWeave deployment details not confirmed.

### 4.2 AWS FSx for Lustre: Deep Dive

FSx Lustre with EFA + GDS is the highest-throughput cloud storage option for checkpoint I/O.

**Throughput by configuration**:

| Client Configuration | Max Throughput Per Client |
|---------------------|--------------------------|
| Standard (ENA) | 100 Gbps (12.5 GB/s) |
| EFA | 700 Gbps (87.5 GB/s) |
| EFA + GPUDirect Storage | **1200 Gbps (150 GB/s)** |

#### EFA Provisioning Constraints

EFA support **must be enabled when creating the FSx file system**. It cannot be added to an existing filesystem. Plan accordingly before provisioning infrastructure.

**Requirements for EFA-enabled FSx**:

| Requirement | Description |
|-------------|-------------|
| **Deployment Type** | Must be `PERSISTENT_2` |
| **Same AZ** | FSx and compute instances must be in the same Availability Zone (cross-AZ EFA not supported) |
| **Supported OS** | Amazon Linux 2023, RHEL 9.5+, or Ubuntu 22.04+ with kernel 6.8+ |
| **Instance Type** | Must support EFA (P5, P5e, P5en, Trn2, etc.) |
| **Minimum Storage** | Depends on throughput tier (see table below) |

**Minimum storage capacity for EFA**:

| Per Unit Storage Throughput | Minimum Storage Capacity (EFA) |
|-----------------------------|-------------------------------|
| 125 MB/s/TiB | 38.4 TB |
| 250 MB/s/TiB | 19.2 TB |
| 500 MB/s/TiB | 9.6 TB |
| 1000 MB/s/TiB | 4.8 TB |

**CLI provisioning with EFA**:
```bash
aws fsx create-file-system \
  --file-system-type LUSTRE \
  --storage-capacity 4800 \
  --subnet-ids $SUBNET_ID \
  --security-group-ids $SG_ID \
  --lustre-configuration '{
    "DeploymentType": "PERSISTENT_2",
    "PerUnitStorageThroughput": 1000,
    "DataCompressionType": "LZ4",
    "EfaEnabled": true
  }'
```

**Kubernetes StorageClass with EFA** (for EKS/HyperPod):
```yaml
parameters:
  deploymentType: PERSISTENT_2
  perUnitStorageThroughput: "250"
  dataCompressionType: "LZ4"
  efaEnabled: "true"
```

**Terraform** (for this repo's blueprints):
```hcl
resource "aws_fsx_lustre_file_system" "checkpoint" {
  storage_capacity    = 4800  # Must meet minimum for chosen throughput
  subnet_ids          = [aws_subnet.private.id]
  deployment_type     = "PERSISTENT_2"
  storage_type        = "SSD"
  per_unit_storage_throughput = 1000

  efa_enabled = true  # Cannot be changed after creation

  metadata_configuration {
    iops = 12000
    mode = "USER_PROVISIONED"
  }
}
```

#### Progressive File Layout (PFL)

Default since August 2023 -- automatically stripes larger files across more OSTs:
```bash
# Default PFL on FSx:
# Files <= 100 MiB: stripe count 1
# Files <= 10 GiB:  stripe count 8
# Files <= 100 GiB: stripe count 16
# Files > 100 GiB:  stripe count 32 (all OSTs)
lfs setstripe -E 100M -c 1 -E 10G -c 8 -E 100G -c 16 -E -1 -c 32 /mountname
```

#### LZ4 Data Compression

LZ4 compression can be enabled post-creation and increases effective throughput by reducing data written to disk. Optimized for speed over ratio -- minimal impact on file system performance:

```bash
# Enable via CLI (can be done on existing filesystem)
aws fsx update-file-system --file-system-id fs-xxx \
  --lustre-configuration DataCompressionType=LZ4
```

#### Client Tuning (per AWS HyperPod Best Practices)

Source: [AWS HyperPod FSx Best Practices](https://awslabs.github.io/ai-on-sagemaker-hyperpod/docs/Tips/Common/Fsx%20for%20Lustre%20best%20practices)

These settings do NOT persist across reboots. Use a boot cron job to reapply.

**All instances**:
```bash
sudo lctl set_param osc.*.max_dirty_mb=64
```

**Instances with >64 GiB memory** (P5e has 2 TiB):
```bash
sudo lctl set_param ldlm.namespaces.*.lru_max_age=600000
sudo lctl set_param ldlm.namespaces.*.lru_size=$((100 * $(nproc)))  # 19200 on P5e
```

**Instances with >64 vCPUs** (P5e has 192 vCPUs) -- kernel module tuning, requires reboot:
```bash
echo "options ptlrpc ptlrpcd_per_cpt_max=32" >> /etc/modprobe.d/modprobe.conf
echo "options ksocklnd credits=2560" >> /etc/modprobe.d/modprobe.conf
sudo reboot
```

**After mount** (post-reboot, for >64 vCPU instances):
```bash
sudo lctl set_param osc.*OST*.max_rpcs_in_flight=32
sudo lctl set_param mdc.*.max_rpcs_in_flight=64
sudo lctl set_param mdc.*.max_mod_rpcs_in_flight=50
```

**Persist tuning via cron** (since `lctl set_param` is volatile):
```bash
# /etc/cron.d/lustre-tuning
@reboot root sleep 30 && \
  lctl set_param osc.*.max_dirty_mb=64 && \
  lctl set_param ldlm.namespaces.*.lru_max_age=600000 && \
  lctl set_param ldlm.namespaces.*.lru_size=19200 && \
  lctl set_param osc.*OST*.max_rpcs_in_flight=32 && \
  lctl set_param mdc.*.max_rpcs_in_flight=64 && \
  lctl set_param mdc.*.max_mod_rpcs_in_flight=50
```

**Impact summary**:

| Setting | Default | Tuned | Effect |
|---------|---------|-------|--------|
| `max_dirty_mb` | 32 | 64 | 2x write buffer before flush |
| `lru_size` | ~100 | 19200 (on P5e) | Fewer lock evictions on large systems |
| `lru_max_age` | (short) | 600000 ms | Locks retained longer, fewer re-acquisitions |
| `max_rpcs_in_flight` (OSC) | 8 | 32 | 4x concurrent I/O to storage targets |
| `max_rpcs_in_flight` (MDC) | 8 | 64 | 8x concurrent metadata operations |
| `max_mod_rpcs_in_flight` | 12 | 50 | 4x concurrent modification RPCs |
| `ptlrpcd_per_cpt_max` | (low) | 32 | More RPC daemon threads per CPT |
| `ksocklnd credits` | (low) | 2560 | Larger socket credit window |

#### Metadata Performance

Limit each directory to **fewer than 100,000 files** on Persistent 2 systems. Large directories increase lock acquisition time for metadata operations.

#### OST Workload Balancing

Monitor via CloudWatch: check `DataReadBytes` and `DataWriteBytes` per OST. If any single OST exceeds ~240 MBps (the throughput capacity of a single 1.2 TiB disk), the workload is unbalanced.

**Fix with striping**:
```bash
# Stripe high-throughput files across all OSTs
sudo lfs setstripe -c -1 -S 4M /fsx/checkpoints/
```

**For S3-imported data**, set `ImportedFileChunkSize` at filesystem creation to distribute imported files evenly. Example: for a 7 TiB filesystem (6 OSTs) with 2.4 GiB files, set chunk size to `2.4 GiB / 6 = 400 MiB`.

#### Checkpoint-Specific Optimization

```bash
# Stripe across all OSTs for large checkpoints
sudo lfs setstripe -c -1 -S 4M /fsx/checkpoints/

# Load GDS kernel module (for GPU-direct I/O)
sudo modprobe nvidia_fs
```

### 4.3 GCP Parallelstore (DAOS-based)

GCP's answer to FSx Lustre, built on Intel DAOS with end-to-end userspace I/O:

| Metric | Per TiB | At 100 TiB |
|--------|---------|------------|
| Read throughput | 1.15 GiB/s | 115 GiB/s |
| Write throughput | 0.5 GiB/s | 50 GiB/s |
| Read IOPS | 30K | 3M |
| 4K read latency | 0.3 ms | 0.3 ms |

**Capacity**: 12 TiB to 100 TiB. Backed by local SSD with 2+1 erasure coding. MTTDL of 2-16 months (not zone-replicated).

---

## 5. Parallel Filesystems

### 5.1 Lustre

The dominant parallel filesystem for ML training. Used by AWS FSx, Azure Managed Lustre, and DDN EXAScaler.

**Striping strategies for checkpoints**:

```bash
# Strategy 1: Maximum parallelism for large checkpoints (>1GB per shard)
lfs setstripe -c -1 -S 4M /mnt/lustre/checkpoints/

# Strategy 2: PFL for mixed sizes (small metadata + large tensors)
lfs setstripe -E 1M -c 1 -E 1G -c 4 -E -1 -c -1 /mnt/lustre/checkpoints/

# Strategy 3: For FSDP sharded checkpoints (many equal-size files)
# Single-stripe per file, Lustre round-robins across OSTs automatically
lfs setstripe -c 1 /mnt/lustre/checkpoints/rank_shards/
```

**Key principle**: When N ranks each write one file, use stripe count 1 per file (avoids lock contention). When one rank writes a giant file, use stripe count -1 (fans out across all OSTs).

**Client tuning**:
```bash
lctl set_param llite.*.max_read_ahead_mb=256      # Read-ahead for loading
lctl set_param osc.*.max_dirty_mb=512              # Buffered writes
lctl set_param osc.*.max_pages_per_rpc=4096        # Large sequential I/O
lctl set_param llite.*.max_cached_mb=2048           # Client-side caching
```

### 5.2 WekaFS

Software-defined parallel filesystem that runs directly on GPU server NVMe and spare CPU cores.

**Performance per GPU server**: >30 GB/s read, >12 GB/s write, >1M IOPS, microsecond latency.

**Real-world ML results** (Cohere deployment): Checkpoint operations 10x faster, GPU utilization 90% (3x improvement over ~30% industry average).

**Key features**: Erasure coding tolerating 4 simultaneous failures, tiered NVMe + object storage, KV cache offloading, container-native.

### 5.3 VAST Data

All-flash disaggregated shared-everything architecture. Used by CoreWeave as primary storage.

Supports NFS, S3, and SMB simultaneously. Uses NVMe-oF/RDMA for backend transport. On-premises deployments support GPUDirect.

### 5.4 BeeGFS

Lightweight parallel filesystem with **BeeOND** (BeeGFS On Demand) -- creates ephemeral parallel FS from local NVMe on compute nodes for burst checkpoint I/O.

```bash
beegfs-ctl --setpattern --chunksize=4m --numtargets=16 /mnt/beegfs/checkpoints/
```

### 5.5 IBM Spectrum Scale (GPFS)

Distributed metadata (no single MDS bottleneck), token-based locking, native RDMA. Up to 2 TB/s aggregate. Used by TOP500 supercomputers. IBM Cloud only as managed service.

---

## 6. NVMe over Fabrics (NVMe-oF)

### 6.1 Transport Variants

| Transport | Added Latency | Throughput | Requirements |
|-----------|--------------|------------|--------------|
| NVMe/RDMA (IB or RoCEv2) | 5-15 us | Line-rate (100-400 Gbps) | RDMA NICs, lossless Ethernet or IB |
| NVMe/TCP | 30-80 us | Network-limited | Standard Ethernet |
| NVMe/FC | 20-50 us | 32/64G FC | FC SAN infrastructure |

**NVMe-oF preserves the NVMe command set** across the fabric. With RDMA transport, submission queue entries use RDMA Send, data transfer uses RDMA Read/Write (zero-copy), and the entire path bypasses the kernel TCP/IP stack.

### 6.2 Checkpoint Write Time Comparison (70B model, ~140 GB)

| Storage | Bandwidth | Write Time | Notes |
|---------|-----------|------------|-------|
| 8x local NVMe (RAID0) | 50+ GB/s | ~3s | Ephemeral |
| FSx Lustre + EFA + GDS | 150 GB/s | <1s | Persistent, shared |
| FSx Lustre + EFA (no GDS) | 87.5 GB/s | ~1.6s | Persistent, shared |
| NVMe/RDMA (100G single) | 12 GB/s | ~12s | Single link |
| Lustre (tuned, no EFA) | 5-10 GB/s | 14-28s | Typical HPC |
| NFS v4.1 | 2-4 GB/s | 35-70s | Not suitable for large ckpts |
| S3 (multipart) | 1-5 GB/s | 28-140s | Archival only |

---

## 7. Framework Storage Backends

### 7.1 PyTorch Distributed Checkpoint (DCP)

**Storage abstraction**: `StorageWriter` / `StorageReader` interfaces.

| Backend | Description |
|---------|-------------|
| `FileSystemWriter/Reader` | POSIX filesystem. Multi-threaded, configurable buffer, `os.fsync` for durability. Supports SafeTensors format. |
| `FsspecWriter/Reader` | Cloud-agnostic via `fsspec`. Auto-detects from URI: `s3://`, `gs://`, `abfs://`. |
| `HuggingFaceStorageWriter/Reader` | SafeTensors with distributed support, optional post-save consolidation. |

```python
from torch.distributed.checkpoint import save, load
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint._fsspec_filesystem import FsspecWriter

# Local filesystem
save(state_dict, storage_writer=FileSystemWriter("/fsx/ckpt/step_1000", thread_count=4))

# S3 via fsspec
save(state_dict, storage_writer=FsspecWriter("s3://bucket/ckpt/step_1000"))
```

**GDS support**: None. All DCP transfers route through CPU memory. A custom `StorageWriter` using `torch.cuda.gds.GdsFile` or KvikIO would be needed.

**Async**: Strong support via `async_save()` with `DefaultStager` using pinned memory for GPU-to-CPU staging.

### 7.2 Megatron-Core

**Storage abstraction**: Pluggable strategy pattern with `SaveShardedStrategy` / `LoadShardedStrategy`.

| Strategy | Description |
|----------|-------------|
| `TorchDistSaveShardedStrategy` | Wraps PyTorch DCP, translates MCore ShardedTensors |
| `FileSystemWriterAsync` | 3-stage async: GPU-to-CPU preload, staging, multi-process writes |
| `FullyParallelSaveStrategyWrapper` | Distributes I/O across all ranks (eliminates coordinator bottleneck) |

**GDS support**: None in open-source code. `nvidia-cufile-cu12` is explicitly excluded from Docker builds. The `MultiStorageClient` (MSC) abstraction suggests proprietary backends may exist.

**Async**: Excellent. `async_save()` returns `AsyncRequest`, non-blocking GPU-to-CPU copies, multi-process parallel writes with GC disabled during writes to prevent CUDA errors.

### 7.3 DeepSpeed

**Storage abstraction**: `CheckpointEngine` base class.

| Engine | Description |
|--------|-------------|
| `TorchCheckpointEngine` | Default. `torch.save()` / `torch.load()`. Synchronous. |
| `NebulaCheckpointEngine` | Azure-specific. Three-tier storage (fast/persistent). Async commit. |
| `DecoupledCheckpointEngine` | Fully async via subprocess + queue dispatch. 5-minute timeout. |
| `FastCheckpointEngine` | Factory-created concurrent writers. |
| `DataStatesCheckpointEngine` | External library with deferred persistence. |

**GDS support**: None. Uses Linux AIO (`libaio`) for NVMe offload. Pinned CPU memory staging.

### 7.4 Run:ai Model Streamer

C++ core with Python bindings. Streams SafeTensors from local/NFS/S3 with concurrent read threads:

```bash
vllm serve <model> --load-format runai_streamer \
  --model-loader-extra-config '{"concurrency": 8, "memory_limit": 5368709120}'
```

**Environment variables**:
- `RUNAI_STREAMER_CONCURRENCY`: Parallel loading threads
- `RUNAI_STREAMER_MEMORY_LIMIT`: Memory constraint (bytes)
- `RUNAI_STREAMER_S3_ENDPOINT`: S3 endpoint (auto-derived from `AWS_ENDPOINT_URL`)

Streams through CPU memory buffer, not GPU-direct. Minimizes GPU idle time through concurrent pipeline.

### 7.5 CoreWeave Tensorizer

Metadata-first file format enabling lazy tensor loading and direct-to-device deserialization:

```python
from tensorizer import TensorDeserializer

with TensorDeserializer("s3://bucket/model.tensors", lazy_load=True, device="cuda:0") as d:
    expert_weights = d["model.experts.3.weight"]  # On-demand loading
```

| Backend | Throughput |
|---------|-----------|
| Local filesystem | 2.25-4.625 GiB/s |
| HTTP/HTTPS | 0.875-1.125 GiB/s |
| Amazon S3 | Configurable credentials |
| Redis | Preliminary support |

vLLM integration: `--load-format tensorizer --model-loader-extra-config '{"tensorizer_uri": "s3://..."}'`

### 7.6 NVIDIA KvikIO

**The only tool providing native GPUDirect Storage bindings for ML workloads.**

```python
import kvikio
import cupy as cp

# GPU-direct file I/O -- bypasses CPU entirely
with kvikio.CuFile("/path/to/checkpoint", "r") as f:
    gpu_array = cp.empty(shape, dtype)
    f.read(gpu_array)

# Async with futures
with kvikio.CuFile("/path/to/data", "r") as f:
    future1 = f.pread(gpu_array[:50])
    future2 = f.pread(gpu_array[50:], file_offset=gpu_array[:50].nbytes)
    future1.get()
    future2.get()
```

**Zarr + GDS integration**:
```python
import kvikio.zarr
store = kvikio.zarr.GDSStore("/path/to/zarr_array")
z = zarr.open(store)
gpu_data = z[:]  # NVMe -> GPU via GDS, no CPU bounce buffer
```

Supports local NVMe (primary), S3, WebHDFS, HTTP. Falls back gracefully when GDS is unavailable.

### 7.7 Framework GDS Support Summary

| Framework | GDS Support | Checkpoint I/O Path | Notes |
|-----------|------------|---------------------|-------|
| PyTorch (core) | `torch.cuda.gds.GdsFile` (prototype) | Direct GPU-Storage | Low-level API only; not in DCP |
| PyTorch DCP | None | CPU-staged FileSystemWriter | Custom StorageWriter needed |
| Megatron-LM | None (explicitly excluded) | Standard filesystem | MSC may have proprietary GDS |
| DeepSpeed | None | Linux AIO (libaio) | Pinned CPU memory |
| KvikIO | **Full native GDS** | Direct GPU-NVMe DMA | Only true GDS solution |
| NCCL | GPUDirect RDMA for collectives | GPU-to-GPU via IB/RoCE | One-sided ops (2.29+) |

---

## 8. Advanced I/O Techniques

### 8.1 Direct I/O (O_DIRECT)

Bypasses Linux page cache. Prevents cache pollution from large checkpoints and provides predictable write latency. Requires 4KB-aligned buffers.

GDS is effectively O_DIRECT from GPU memory to storage, bypassing both page cache and CPU memory.

### 8.2 Memory-Mapped I/O (mmap)

Safetensors uses mmap for model loading:
- **CPU loading**: 76.6x faster than `torch.load()` (4ms vs 307ms for GPT-2)
- **GPU loading**: 2.1x faster (165ms vs 354ms)

For sharded checkpoints, mmap enables lazy loading -- only pages actually accessed are read from storage.

### 8.3 io_uring

Linux async I/O with submission/completion ring buffers. Polling mode achieves 1.7M IOPS vs Linux AIO's 608K IOPS (2.8x). Not yet used by PyTorch/Megatron checkpoint paths but could benefit custom implementations.

### 8.4 Checkpoint Compression

| Algorithm | Ratio | Compress Speed | Decompress Speed |
|-----------|-------|---------------|-----------------|
| zstd -1 (default) | 2.9x | 510 MB/s | 1,550 MB/s |
| zstd --fast=1 | 2.4x | 545 MB/s | 1,850 MB/s |
| lz4 (default) | 2.1x | 675 MB/s | 3,850 MB/s |

**Rule of thumb**: Compress for slow storage (S3, NFS, HDD). Skip for fast storage (local NVMe, FSx+GDS) where compression speed becomes the bottleneck.

### 8.5 Delta/Incremental Checkpointing

- **Tensor-level delta**: Compare state dicts, save only changed tensors
- **Optimizer state skipping**: Save optimizer states less frequently (reconstructible)
- **LoRA-aware**: Only save adapter weights (70B model: 140 GB full vs ~200 MB LoRA adapters = 700x reduction)

### 8.6 Copy-on-Write Snapshots (ZFS)

Instant filesystem snapshots (nanoseconds, regardless of checkpoint size). Space-efficient (only changed blocks consume storage). Built-in compression and checksumming. Useful on local NVMe for point-in-time checkpoint versioning before async upload.

---

## 9. NCCL Transport Configuration

### 9.1 GPUDirect RDMA Control

```bash
NCCL_NET_GDR_LEVEL=SYS      # LOC|PIX|PXB|PHB|SYS -- controls GPU-to-NIC distance
NCCL_NET_GDR_READ=1          # Enable GDR for reads (default on NVLink platforms)
```

### 9.2 InfiniBand Tuning

```bash
NCCL_IB_HCA=mlx5_0           # Filter Host Channel Adapters
NCCL_IB_TIMEOUT=20            # IB timeout: 4.096 us x 2^timeout
NCCL_IB_RETRY_CNT=7           # Retry count
NCCL_IB_ADAPTIVE_ROUTING=1    # Adaptive routing (default on IB)
NCCL_IB_PCI_RELAXED_ORDERING=2 # Auto-detect relaxed ordering
```

### 9.3 Buffer and Channel Tuning

```bash
NCCL_BUFFSIZE=4194304         # Buffer between GPU pairs (4 MiB default)
NCCL_MAX_NCHANNELS=32         # Communication channels
NCCL_P2P_NET_CHUNKSIZE=131072 # Message chunk size (128 KB default)
NCCL_CROSS_NIC=2              # 0=avoid, 1=allow, 2=auto cross-rail
```

### 9.4 Algorithm and Protocol

```bash
NCCL_ALGO=Ring,Tree,NVLS      # Ring, Tree, CollnetChain, CollnetDirect, NVLS, NVLSTree, PAT
NCCL_PROTO=Simple              # LL (low-latency), LL128, Simple
```

### 9.5 One-Sided Operations (NCCL 2.29+)

Directly relevant to async checkpoint transfer:
```c
ncclPutSignal()    // Write data to remote GPU without matching receive
ncclWaitSignal()   // Sync on target rank
ncclSignal()       // Notification without data
```

Memory registered via `ncclCommWindowRegister()`. Enables async remote checkpoint writes where a saving rank pushes data without the receiver actively participating.

---

## 10. Storage Tiering Architecture

### 10.1 The AZ Problem

FSx for Lustre is **AZ-specific** -- it must be in the same Availability Zone as compute instances, and EFA does not work cross-AZ. This creates a fundamental architectural constraint:

- GPU capacity blocks may only be available in specific AZs
- FSx must be pre-provisioned in the same AZ (and EFA must be enabled at creation time)
- If you lose access to that AZ (capacity block expiration, spot interruption), the FSx data is stranded
- S3 is region-wide and survives AZ changes

**S3 must be the durable source of truth.** FSx and NVMe are performance caches.

### 10.2 Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Tier 0: GPU Memory + NVMe (Ephemeral, Fastest)            │
│  - Active model weights in GPU HBM                          │
│  - NVMe RAID0 for staging (~50 GB/s, 30 TB on P5e)         │
│  - Lost on instance termination                             │
├─────────────────────────────────────────────────────────────┤
│  Tier 1: FSx for Lustre (AZ-Scoped, Fast)                  │
│  - Shared across nodes in same AZ                           │
│  - EFA+GDS: 150 GB/s per client                             │
│  - Persistent within AZ, but AZ-locked                      │
│  - Linked to S3 via Data Repository Association             │
├─────────────────────────────────────────────────────────────┤
│  Tier 2: Amazon S3 (Region-Wide, Durable)                   │
│  - Source of truth for models and checkpoints               │
│  - ~100 Gbps per instance (12.5 GB/s)                       │
│  - Survives AZ changes, capacity block expiration           │
│  - Run:ai Streamer can load directly to GPU                 │
└─────────────────────────────────────────────────────────────┘
```

### 10.3 FSx-to-S3 Data Repository Association (DRA)

DRA links an FSx filesystem path to an S3 prefix, enabling automatic bi-directional sync:

```bash
aws fsx create-data-repository-association \
  --file-system-id ${FSX_ID} \
  --file-system-path "/models" \
  --data-repository-path s3://${S3_BUCKET}/models \
  --s3 'AutoImportPolicy={Events=[NEW,CHANGED,DELETED]},AutoExportPolicy={Events=[NEW,CHANGED,DELETED]}' \
  --batch-import-meta-data-on-create \
  --region ${AWS_REGION}
```

**Key behaviors**:
- **AutoImport**: New/changed/deleted objects in S3 are reflected in FSx automatically
- **AutoExport**: New/changed/deleted files on FSx are pushed to S3 automatically
- **Lazy loading**: Files imported from S3 appear in FSx metadata immediately, but data is fetched on first access (or can be pre-loaded with `hsm_restore`)
- **Batch import**: `--batch-import-meta-data-on-create` imports all existing S3 metadata at association time

**Pre-loading model weights** (avoid first-access latency):
```bash
# After DRA creation, force-load all model files from S3 into FSx
find /fsx/models/moonshotai/Kimi-K2.5 -type f -exec lfs hsm_restore {} \;

# Monitor import progress
lfs hsm_action /fsx/models/moonshotai/Kimi-K2.5/*
```

**Checkpoint archival**: With AutoExport enabled, checkpoints written to FSx are automatically replicated to S3. No separate upload step needed.

### 10.4 Model Loading Strategies

Different strategies suit different operational constraints:

#### Strategy A: FSx Pre-Staged (Fastest Loading, AZ-Locked)

```
S3 ──DRA──> FSx for Lustre ──EFA+GDS──> GPU
                                          │
                              NVMe copy ──┘ (optional, for even faster reload)
```

- **Best for**: Production inference with stable AZ, multi-node clusters sharing model
- **Loading speed**: 150 GB/s (EFA+GDS) or 87.5 GB/s (EFA only)
- **Pre-staging**: DRA + `hsm_restore` or initial model download
- **Constraint**: Model must exist on FSx before serving starts; AZ-locked

#### Strategy B: NVMe Local Copy (Fastest Single-Node, Ephemeral)

```
S3 ──s5cmd──> NVMe RAID0 ──GDS──> GPU
```

- **Best for**: Single-node inference, capacity blocks, benchmarking
- **Loading speed**: ~50 GB/s from NVMe (after S3 copy completes)
- **Pre-staging**: `s5cmd cp s3://bucket/model/ /nvme/model/` (~5-10 min for 540 GB)
- **Constraint**: Data lost on instance termination; per-node copy required

#### Strategy C: S3 Direct via Run:ai Streamer (Zero Pre-Staging)

```
S3 ──Run:ai Streamer──> CPU buffer ──> GPU
```

```bash
vllm serve moonshotai/Kimi-K2.5 \
  --load-format runai_streamer \
  --model-loader-extra-config '{"concurrency": 16, "memory_limit": 17179869184}' \
  --tensor-parallel-size 8
```

- **Best for**: Elastic/ephemeral clusters, capacity blocks, multi-AZ failover
- **Loading speed**: ~12.5 GB/s (S3 per-instance bandwidth, ~100 Gbps)
- **Pre-staging**: None -- streams directly from S3
- **Constraint**: Slower initial load (~45s for 540 GB model); not GPU-direct (CPU bounce)
- **Advantage**: AZ-independent, no FSx needed, no local copy, works immediately after instance launch

#### Strategy D: S3 Direct via Tensorizer (Zero Pre-Staging, Lazy Load)

```
S3 ──Tensorizer──> CPU/GPU (lazy, per-tensor)
```

```bash
vllm serve moonshotai/Kimi-K2.5 \
  --load-format tensorizer \
  --model-loader-extra-config '{"tensorizer_uri": "s3://bucket/kimi-k2.5.tensors"}'
```

- **Best for**: Inference where first-token latency is acceptable, models pre-serialized in Tensorizer format
- **Loading speed**: 2-5 GiB/s from S3
- **Pre-staging**: Must serialize model to Tensorizer format first (one-time cost)
- **Constraint**: Requires format conversion; slower than Run:ai for safetensors

#### Strategy E: Hybrid FSx + S3 Fallback (Production Recommended)

```
Primary:   FSx (EFA+GDS) ──> GPU        [fast path, AZ-locked]
Fallback:  S3 (Run:ai Streamer) ──> GPU  [slow path, any AZ]
```

- **Best for**: Production with AZ resilience
- **Implementation**: Check if FSx mount exists; if yes, load from FSx. If not (new AZ, FSx unavailable), fall back to Run:ai Streamer from S3.
- **DRA ensures S3 always has latest**: Checkpoints on FSx auto-export to S3

### 10.5 Strategy Decision Matrix

| Constraint | Recommended Strategy | Why |
|-----------|---------------------|-----|
| Stable AZ, multi-node | A (FSx Pre-Staged) | Shared storage, 150 GB/s, GDS |
| Capacity blocks, single AZ | A + B (FSx + NVMe) | FSx for sharing, NVMe for fastest per-node |
| Capacity blocks, AZ may change | C (Run:ai from S3) | No pre-staging, AZ-independent |
| Ephemeral/spot instances | C (Run:ai from S3) | Zero setup, instant start |
| Production with failover | E (Hybrid FSx + S3) | Fast primary, resilient fallback |
| Cost-optimized (no FSx) | C (Run:ai from S3) | Eliminates FSx cost entirely |
| Multi-region DR | S3 cross-region replication + C | Models in multiple regions |

### 10.6 Loading Speed Comparison (Kimi K2.5, ~540 GB)

| Strategy | Bandwidth | Load Time | Pre-Stage Time | Total Time to First Token |
|----------|-----------|-----------|----------------|--------------------------|
| FSx + EFA + GDS | 150 GB/s | ~4s | DRA + hsm_restore (~5-10 min once) | ~4s (after pre-stage) |
| FSx + EFA (no GDS) | 87.5 GB/s | ~6s | Same as above | ~6s (after pre-stage) |
| NVMe RAID0 | 50 GB/s | ~11s | s5cmd from S3 (~5-10 min) | ~11s (after pre-stage) |
| S3 via Run:ai Streamer | 12.5 GB/s | ~45s | None | ~45s |
| S3 via Tensorizer | 2-5 GiB/s | ~2-4 min | Serialization (one-time) | ~2-4 min |
| FSx without EFA (ENA) | 12.5 GB/s | ~45s | DRA + hsm_restore | ~45s (after pre-stage) |

---

## 11. Recommended Checkpoint I/O Architecture

### 11.1 Bandwidth Hierarchy

```
NVLink 5 (NVL72):     130,000 GB/s  (intra-domain consolidation)
NVLink 4 (8-GPU):       7,200 GB/s  (intra-node)
GCP A3 Ultra:              450 GB/s  (inter-node, Ethernet)
AWS P5/Azure ND H100:      400 GB/s  (inter-node, EFA/IB)
FSx Lustre (EFA+GDS):     150 GB/s  (parallel filesystem)
NVMe (8-drive RAID0):      56 GB/s  (local staging)
PCIe Gen5 (per GPU):       31.5 GB/s (GPU <-> CPU)
S3 per instance:           12.5 GB/s (object storage)
NVMe (single drive):        7 GB/s  (sequential write)
```

### 11.2 By Checkpoint Size

| Size | Use Case | Recommended Approach |
|------|----------|---------------------|
| < 1 GB | LoRA adapters | Local NVMe + async S3 |
| 1-10 GB | 7B models | FSx Lustre scratch, single-stripe per shard |
| 10-100 GB | 70B models | FSx Lustre persistent, PFL, EFA |
| 100+ GB | 405B+ / MoE models | FSx Lustre + EFA + GDS, all-OST striping, async DCP |

### 11.3 Optimal Stack (AWS P5e)

```
GPU Memory
    |
    | GPUDirect Storage (nvidia_fs)
    |
NVMe over PCIe / EFA fabric
    |
    | Lustre client (kernel module)
    |
FSx for Lustre (striped across OSTs)
    |
    | DRA (auto-export) -> S3 (durable archive)
```

Configuration:
1. EFA enabled at FSx creation time (`EfaEnabled: true`)
2. DRA linking FSx to S3 with AutoImport + AutoExport
3. GDS kernel module loaded (`modprobe nvidia_fs`)
4. Checkpoint directory striped: `lfs setstripe -c -1 -S 4M`
5. Client tuned per HyperPod best practices (section 4.2)
6. PyTorch DCP `FileSystemWriter` for sharded saves
7. Async checkpointing to overlap with training

---

## 12. Key Takeaways

1. **S3 is the source of truth**: FSx and NVMe are AZ-scoped performance caches. S3 survives AZ changes, capacity block expiration, and instance termination. Use DRA to keep FSx and S3 in sync.

2. **Intra-node is never the bottleneck**: NVLink bandwidth (900+ GB/s) dwarfs any storage or network path. State dict consolidation within a node is negligible cost.

3. **Inter-node bandwidth is now sufficient**: At 3,200+ Gbps across all major providers, the network is rarely the checkpoint bottleneck for per-node sharded writes.

4. **Storage is the true checkpoint bottleneck**: Write throughput (10-150 GB/s) is 3-40x lower than available network bandwidth.

5. **PCIe contention is the hidden cost**: Checkpoint writes through PCIe compete with ongoing GPU compute. Async checkpointing with CPU staging or GDS is essential.

6. **GDS is available but unused**: Despite 2-8x bandwidth improvements, no major ML framework uses GDS for checkpointing. KvikIO + a custom DCP `StorageWriter` is the path forward.

7. **FSx Lustre + EFA + GDS is the gold standard**: 150 GB/s per client, POSIX, multi-node RWX, GPUDirect. But EFA must be enabled at creation and requires PERSISTENT_2 with minimum storage thresholds.

8. **Run:ai Streamer eliminates pre-staging**: Direct S3-to-GPU streaming at ~12.5 GB/s removes FSx dependency entirely. Best for elastic/ephemeral workloads where AZ flexibility matters more than loading speed.

9. **Compress for slow storage only**: zstd at 510 MB/s becomes the bottleneck on fast storage (NVMe, FSx+GDS). Only compress when writing to S3/NFS/HDD.

10. **Network topology affects simultaneous checkpoints**: Rail-optimized networks benefit from aligning checkpoint targets to rails. Fat-tree networks provide full bisection bandwidth regardless.
