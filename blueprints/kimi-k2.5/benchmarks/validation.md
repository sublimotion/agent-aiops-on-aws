# GDS Validation on p5e.48xlarge with FSx Lustre

**Date**: February 17, 2026
**Instance**: p5e.48xlarge (8x NVIDIA H200 143GB HBM)
**Storage**: FSx Lustre 4.8 TiB PERSISTENT_2 (1000 MB/s/TiB) + NVMe RAID0 28TB
**OS**: Amazon Linux 2023 (kernel 6.1.161)
**GDS**: nvidia-gds 12.9.1, nvidia_fs 2.25.7, libcufile 2.12

---

## 1. Verify GDS Installation

```bash
# Check GDS packages
rpm -qa | grep -i gds

# Expected:
#   gds-tools-12-9-1.14.1.1-1.x86_64
#   nvidia-gds-12-9-12.9.1-1.x86_64
#   nvidia-gds-12.9.1-1.x86_64
```

## 2. Verify nvidia_fs Kernel Module

```bash
# Check module is loaded
lsmod | grep nvidia_fs

# Expected:
#   nvidia_fs  262144  0
#   nvidia     14323712  171 nvidia_uvm,nvidia_fs,nvidia_modeset

# Check module details
modinfo nvidia_fs | head -5

# Expected:
#   filename:    /lib/modules/.../extra/nvidia-fs.ko
#   description: NVIDIA GPUDirect Storage
#   version:     2.25.7
```

## 3. Platform Validation (gdscheck -p)

```bash
export PATH=$PATH:/usr/local/cuda-12.9/gds/tools
gdscheck -p
```

### What to check in output

**Driver Configuration** — Lustre support must show `Supported`:
```
DDN EXAScaler      : Supported
```

**GPU INFO** — All GPUs must show `supports GDS` and `IOMMU State: Disabled`:
```
GPU index 0 NVIDIA H200 bar:1 bar size (MiB):262144 supports GDS, IOMMU State: Disabled
GPU index 1 NVIDIA H200 bar:1 bar size (MiB):262144 supports GDS, IOMMU State: Disabled
...
GPU index 7 NVIDIA H200 bar:1 bar size (MiB):262144 supports GDS, IOMMU State: Disabled
```

**Platform verification** — Must say `succeeded`:
```
Platform verification succeeded
```

### Expected limitations on p5e

- `Mellanox PeerDirect: Disabled` — EFA does not expose PeerDirect, so RDMA bypass is unavailable. GDS still works via the nvidia_fs kernel module path.
- `Userspace RDMA: Unsupported` — Same reason. Standard GDS DMA path is used instead.
- `NVMe P2PDMA: Unsupported` — P2P DMA between NVMe and GPU not supported on this platform. GDS uses the kernel bounce buffer path for NVMe.

## 4. nvidia_fs Driver Statistics

```bash
cat /proc/driver/nvidia-fs/stats
```

Check for:
- `GDS Version` and `NVFS Driver` version strings present
- `err=0` on Reads and Writes (no IO errors)
- `Bar1-map: ok=N err=0` (BAR1 GPU memory mapping works)

## 5. GDS I/O Benchmarks (gdsio)

### 5a. NVMe RAID0 — GPU-Direct Read

Tests the fastest possible GDS path (local PCIe storage to GPU).

```bash
export PATH=$PATH:/usr/local/cuda-12.9/gds/tools

# Create test file
mkdir -p /mnt/nvme/gds-test
dd if=/dev/urandom of=/mnt/nvme/gds-test/testfile bs=1M count=128 2>/dev/null

# Run: GPU 0, 4 threads, 128MB file, READ, GPU-Direct, 10 seconds
gdsio -f /mnt/nvme/gds-test/testfile -d 0 -w 4 -s 128M -x 0 -I 1 -T 10
```

**Expected**: `XferType: GPUD`, throughput ~6-8 GiB/s, latency ~400-650 us.

### 5b. FSx Lustre — GPU-Direct Read

Tests GDS over network storage (FSx Lustre).

```bash
# Create test file on FSx
mkdir -p /mnt/fsx/gds-test
dd if=/dev/urandom of=/mnt/fsx/gds-test/testfile bs=1M count=128 2>/dev/null

# Run: GPU 0, 4 threads, 128MB file, READ, GPU-Direct, 10 seconds
gdsio -f /mnt/fsx/gds-test/testfile -d 0 -w 4 -s 128M -x 0 -I 1 -T 10
```

**Expected**: `XferType: GPUD`, throughput ~0.5-0.6 GiB/s per stream, latency ~6,000-7,000 us.

### 5c. FSx Lustre — GPU-Direct Read (Compat Mode)

Tests GDS compat mode fallback path.

```bash
# -I 2 = compat mode (cuFile with POSIX fallback)
gdsio -f /mnt/fsx/gds-test/testfile -d 0 -w 4 -s 128M -x 0 -I 2 -T 10
```

**Expected**: Similar throughput to direct mode on FSx (~0.5-0.6 GiB/s). Both modes are network-bound.

### 5d. Cleanup

```bash
rm -rf /mnt/nvme/gds-test /mnt/fsx/gds-test
```

### gdsio Flags Reference

| Flag | Meaning |
|------|---------|
| `-f` | File path |
| `-d` | GPU device index (0-7) |
| `-w` | Number of IO threads |
| `-s` | Dataset size |
| `-x` | IO type: 0=read, 1=write |
| `-I` | Transfer type: 0=auto, 1=GDS GPU-Direct, 2=compat/POSIX |
| `-T` | Duration in seconds |

## 6. cuFile Configuration

The default config is at `/etc/cufile.json`. Key settings for Lustre:

```bash
# View Lustre-specific settings
python3 -c "
import json
with open('/etc/cufile.json') as f:
    # Strip comments (cufile.json has C-style comments)
    lines = [l for l in f if not l.strip().startswith('//')]
    data = json.loads(''.join(lines))
print('Lustre posix_gds_min_kb:', data['fs']['lustre']['posix_gds_min_kb'])
print('allow_compat_mode:', data['properties']['allow_compat_mode'])
print('max_direct_io_size_kb:', data['properties']['max_direct_io_size_kb'])
print('parallel_io:', data['execution']['parallel_io'])
"
```

Applications can override with: `export CUFILE_ENV_PATH_JSON=/path/to/custom/cufile.json`

---

## Validation Results (February 17, 2026)

### Platform Check

| Check | Result |
|-------|--------|
| GDS release | 1.14.1.1 |
| nvidia_fs driver | 2.25.7 |
| libcufile | 2.12 |
| Lustre client | 2.15.6 |
| DDN EXAScaler support | Supported |
| IOMMU | Disabled |
| All 8 GPUs GDS-capable | Yes (256 GB BAR1 each) |
| Platform verification | Succeeded |
| PeerDirect (RDMA bypass) | Disabled (EFA limitation) |

### I/O Benchmarks

| Test | Transfer Mode | Throughput | Avg Latency |
|------|--------------|------------|-------------|
| NVMe RAID0 read | GPU-Direct (GPUD) | **8.25 GiB/s** | 473 us |
| FSx Lustre read (direct) | GPU-Direct (GPUD) | **0.58 GiB/s** | 6,759 us |
| FSx Lustre read (compat) | GPU-Direct (GPUD) | **0.58 GiB/s** | 6,759 us |
| FSx Lustre write (CPU) | CPU-only (CPUONLY) | **0.58 GiB/s** | 6,763 us |

### Interpretation

- **GDS is functional on both NVMe and FSx Lustre.** All reads show `XferType: GPUD`, confirming data transfers directly to GPU memory without CPU bounce.
- **NVMe is 14x faster than FSx** — expected. NVMe is local PCIe (~25 GB/s aggregate for RAID0), while FSx is bounded by network throughput (~4.7 GB/s provisioned for 4.8 TiB at 1000 MB/s/TiB).
- **Single-stream FSx at 0.58 GiB/s** is consistent with provisioned throughput. With 8 GPUs running concurrent KV cache operations, aggregate FSx throughput will scale toward the provisioned limit.
- **PeerDirect disabled** means GDS uses the nvidia_fs kernel module DMA path rather than RDMA bypass. This is normal for EFA-based instances and does not prevent GDS from functioning.

---

## 7. Model Loading Benchmark (Kimi K2.5 — 595 GB)

### Method

Full sequential read of all 64 safetensors shards (554.3 GB on disk) using Python `read()` with 8 MB chunks. Page cache dropped before each run (`echo 3 > /proc/sys/vm/drop_caches`). Throughput extrapolated to the full 595 GB model weight.

```bash
# Drop caches and run benchmark
sync; echo 3 > /proc/sys/vm/drop_caches
python3 bench_io.py nvme   # or fsx
```

### Results

| Storage | Files | Data Read | Time | Throughput | Full Model Load (595 GB) |
|---------|-------|-----------|------|------------|--------------------------|
| **NVMe RAID0** (8x 3.5TB local) | 64 | 554.30 GB | 57.35s | **9.67 GB/s** | **61.6s (1.0 min)** |
| **FSx Lustre** (4.8 TiB, 1000 MB/s/TiB) | 64 | 554.30 GB | 982.77s | **0.56 GB/s** | **1054.9s (17.6 min)** |

### Interpretation

- **NVMe is 17x faster than FSx for model loading.** This is the dominant factor in cold-start latency.
- **NVMe at 9.67 GB/s** approaches the theoretical RAID0 aggregate of ~25 GB/s. Single-threaded Python I/O doesn't fully saturate the array. Multi-threaded or `O_DIRECT` reads would push higher.
- **FSx at 0.56 GB/s** matches the single-client Lustre throughput observed in gdsio tests. The provisioned 4.7 GB/s is aggregate across all clients — a single sequential reader cannot saturate it.
- **Recommendation**: Stage model weights on NVMe for serving (1 min cold start). Use FSx as the durable source with DRA for S3 tiering. A `cp` from FSx→NVMe takes ~8 min for the full model.

---

## 8. CUDA 13.0 Driver Compatibility (vLLM Container Images)

**Date**: February 17, 2026
**Driver**: NVIDIA 580.126.09 (CUDA 13.0)
**AMI**: Amazon Linux 2023 (AL2023_x86_64_NVIDIA), kernel 6.1.161
**Container toolkit**: nvidia-container-toolkit 1.18.2

### Problem

P5e instances launched with the latest AL2023 NVIDIA AMI ship with driver **580.126.09** which reports **CUDA 13.0**. This is a major version bump from CUDA 12.x. Standard vLLM container images fail with:

```
RuntimeError: Error 803: system has unsupported display driver / cuda driver combination
```

### Images Tested

| Image | PyTorch | CUDA Compiled | Result |
|-------|---------|---------------|--------|
| `vllm/vllm-openai:v0.15.1` | 2.9.1+cu129 | 12.9 | **Error 803** — CUDA 12.x runtime incompatible with CUDA 13.0 driver |
| `vllm/vllm-openai:v0.15.1-cu130` | 2.9.1+cu130 | 13.0 | **Error 803** — PyTorch 2.9.1 has driver compat bug with 580.x |
| `vllm/vllm-openai:cu130-nightly` | 2.10.0+cu130 | 13.0 | **Working** — PyTorch 2.10.0 fixes driver 580 compatibility |

### Root Cause

1. **CUDA 12.x → 13.0 is a breaking change** for container forward compatibility. Unlike minor version bumps (12.4→12.9), the 12→13 major boundary is not forward-compatible.
2. **PyTorch 2.9.1+cu130** has a bug where `cudaGetDeviceCount()` fails on driver 580 even though the CUDA versions match. This was fixed in PyTorch 2.10.0.

### Resolution

Use `vllm/vllm-openai:cu130-nightly` (vLLM v0.16.0rc2, PyTorch 2.10.0+cu130). Model loaded successfully from NVMe in **~4.5 minutes** with TP=8 on 8x H200.

### Benchmark Image Compatibility Matrix

All benchmark images rebuilt for CUDA 13.0 driver compatibility and pushed to ECR:

| Config | ECR Image | Base | PyTorch | CUDA | Status |
|--------|-----------|------|---------|------|--------|
| Baseline | `vllm-openai:cu130-nightly` | `vllm/vllm-openai:cu130-nightly` | 2.10.0+cu130 | 13.0 | **Working** |
| A (LMCache) | `vllm-openai:cu130-nightly` | same as baseline | 2.10.0+cu130 | 13.0 | **Working** |
| B (Mooncake) | `vllm-mooncake:cu130` | `vllm/vllm-openai:cu130-nightly` | 2.10.0+cu130 | 13.0 | **Rebuilt** |
| C (Dynamo) | `dynamo-kvbm:cu130` | `nvcr.io/nvidia/pytorch:26.01-py3` | 2.10.0+cu13.1 | 13.1 | **Rebuilt** |

**Dynamo base image selection notes:**
- `25.01-py3`: CUDA 12.8 — incompatible with driver 580/CUDA 13.0
- `25.03-py3`: NCCL hang issue (see [dynamo#1065](https://github.com/ai-dynamo/dynamo/pull/1065))
- `25.05-py3`: CUDA 12.9 + `pytest==8.1.1` constraint conflicts with `ai-dynamo>=8.3.4` requirement
- `26.01-py3`: PyTorch 2.10.0+cu13.1 — all dependencies resolve cleanly

**Old images** (`vllm-mooncake:v0.15.1`, `dynamo-kvbm:v0.9.0`) remain in ECR but are **incompatible** with the p5e CUDA 13.0 driver.

### Lesson

Always verify CUDA driver/container compatibility when launching new GPU instances. Pin to a known-working image tag (e.g., `cu130-nightly-d00df624f`) for reproducible benchmarks rather than floating `nightly` tags.
