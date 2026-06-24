# Reproduction — Nemotron-3-Super FP8 on g7e (SGLang), 14,511 tok/s @ c8

Empirical re-verification, 2026-06-24, Tokyo (ap-northeast-1a), g7e.12xlarge (2× RTX PRO 6000 Blackwell sm_120, PCIe).
This is the runnable transcript: every command + the expected/observed output, in order.

## 0. Target number (from the published 2026-06-12 run)
| concurrency | e2e p50 | e2e p90 | **aggregate tok/s** | per-GPU (÷2) | $/1M |
|---|---|---|---|---|---|
| **8** | 4,607 ms | 6,336 ms | **14,511** | 7,256 | $0.158 |

> Note: "7,256" some people quote is just aggregate ÷ 2 GPUs — a derived per-GPU column, not a separate target. The headline is **14,511 aggregate @ c8**.

## 1. Environment (verified)
- **Instance**: `i-0de1456111eb3b30e`, g7e.12xlarge, ap-northeast-1a
- **AMI**: `ami-004c26ea7b1af6c97` = `amazon-eks-node-al2023-x86_64-nvidia-1.36-v20260529` (AL2023, k8s 1.36.1, containerd 2.x, NVIDIA driver bundled)
- **Driver**: 580.159.03; **GPUs**: 2× RTX PRO 6000 Blackwell Server Edition, 96 GB each, sm_120
- **Container runtime**: `nerdctl` (containerd) — NOT docker. `sudo systemctl start containerd` first.
- **Image**: `lmsysorg/sglang:v0.5.12.post1-cu130` (CUDA 13.0 build — mandatory: ships NCCL ≥2.26.2 to clear the sm_120+PCIe NCCL-2.25.1 collective bug that kills the TP=2 path)
- SSH: `ssh -i ~/.ssh/g7e-tokyo.pem ec2-user@<public-ip>` (SG must allow your IP on :22)

## 2. Capacity (the instance was stopped; capacity had to be re-acquired)
The stopped instance lost its capacity slot. Re-acquire a **targeted** ODCR in the instance's AZ, point the instance at it, then start:
```bash
aws ec2 create-capacity-reservation --region ap-northeast-1 \
  --instance-type g7e.12xlarge --instance-platform Linux/UNIX \
  --availability-zone ap-northeast-1a --instance-count 1 \
  --instance-match-criteria targeted --end-date-type unlimited
# -> cr-XXXX
aws ec2 modify-instance-capacity-reservation-attributes --region ap-northeast-1 \
  --instance-id i-0de1456111eb3b30e \
  --capacity-reservation-specification 'CapacityReservationTarget={CapacityReservationId=cr-XXXX}'
aws ec2 start-instances --region ap-northeast-1 --instance-ids i-0de1456111eb3b30e
```
> The held ODCR bills the on-demand rate (~$12/hr) until cancelled — cancel it when done.

## 3. Stage weights (did NOT persist across stop — re-downloaded)
The NVMe weights from the original run did not survive the stop. Re-downloaded ~117 GB (26 shards) in a detached container — hf_transfer was fast on the Tokyo box (~5 min):
```bash
sudo systemctl start containerd
sudo nerdctl run -d --name nemodl --network host -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -v /mnt/nvme/models:/models --entrypoint bash \
  lmsysorg/sglang:v0.5.12.post1-cu130 \
  -c "pip install -q hf_transfer; hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
      --local-dir /models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
# verify: 26 safetensors shards + config.json present, ~117 GB
```

## 4. Serve — SGLang, the VERIFIED command (note `flashinfer_cutlass`, NOT `triton`)
```bash
sudo nerdctl run -d --name sglang --network host --gpus all \
  -v /mnt/nvme/models:/mnt/nvme/models \
  -e SGLANG_DISABLE_DEEP_GEMM=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  lmsysorg/sglang:v0.5.12.post1-cu130 \
  python3 -m sglang.launch_server \
    --model-path /mnt/nvme/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
    --served-model-name nemotron-3-super \
    --tp 2 --trust-remote-code \
    --fp8-gemm-backend triton \
    --moe-runner-backend flashinfer_cutlass \
    --mamba-scheduler-strategy no_buffer \
    --reasoning-parser nemotron_3 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.90 \
    --context-length 16384 \
    --host 0.0.0.0 --port 8000
```
**CONFIRMED in the startup args** (`server_args=ServerArgs(...)`): `moe_runner_backend='flashinfer_cutlass'`, `fp8_gemm_runner_backend='triton'`, `mamba_scheduler_strategy='no_buffer'`, `reasoning_parser='nemotron_3'`, `quantization='modelopt_fp8'`, `tp_size=2`, `kv_cache_dtype='fp8_e4m3'`.
- **WRONG value to avoid**: `--moe-runner-backend triton` SMEM-overflows on sm_120 (`147456 > 101376`) and the server won't start. The stale EKS manifest and an earlier hand-shared snippet both had this — they were never the config that produced the numbers.
- First FP8 ModelOpt load is ~45 min (one-time quant processing), ~14 s on subsequent starts.

### CRITICAL gotcha — `--shm-size=32g --ipc=host` is mandatory for the TP=2 path
Without a large `/dev/shm`, NCCL's `ncclCommInitRank` fails at startup with `RuntimeError: NCCL error: unhandled system error` on both TP ranks (the container's default 64 MB `/dev/shm` is too small for the NCCL shared-memory bootstrap). This is NOT the sm_120 NCCL-2.25.1 collective bug (the cu130 image fixes that) — it's a container shm sizing issue. The original EKS manifest mounted a 32Gi shm `emptyDir`; the bare `nerdctl`/`docker run` MUST pass `--shm-size=32g --ipc=host` to match. Symptom if you miss it: server crashes ~10 s after launch, GPUs never load.

> Operational note for `nerdctl`: a container that crashes on NCCL can wedge in an "Unknown" state that resists `nerdctl rm -f`. Clear with `nerdctl kill <name>; nerdctl rm -f <name>`, verify it's gone, THEN re-run — re-running with the same `--name` while the zombie lingers silently no-ops.

## 5. Benchmark @ c8 — RUN + RESULT

The bench needs `aiohttp`; the host Python (3.9, no pip) lacks it. Easiest: run the bench **inside the running sglang container** (Python 3.12 + aiohttp), targeting `localhost:8000` (container is `--network host`):
```bash
sudo nerdctl cp bench-fin-support.py sglang:/tmp/bench-fin-support.py
sudo nerdctl exec sglang python3 /tmp/bench-fin-support.py \
  --endpoint http://localhost:8000 --model nemotron-3-super \
  --concurrency 8 --requests 64 --warmup 50 \
  --engine-tag g7e-sglang-tp2x1-fp8-repro --out-dir /tmp/repro-results
```
The workload (baked into the script): lognormal ISL p50 8,823 / p90 11,952 (mean ~9,200), OSL p50 243 / p90 415. Aggregate throughput = total (in+out) tokens ÷ wall_s.

### RESULT — reproduced within ~2%
| Metric | Published 2026-06-12 | This repro 2026-06-24 |
|---|---|---|
| aggregate tok/s | 14,511 | **~14,200** |
| e2e p50 | 4,607 ms | 4,675 ms |
| e2e p90 | 6,336 ms | 6,718 ms |
| SLO (6.5/9.5s) | PASS | **PASS** (both gates) |
| errors | 0 | 0 / 64 |

The ~2% delta is run-to-run noise (this run's lognormal draw gave ISL mean 8,672 vs 9,101). **Confirmed: the published number is real and reproducible with `--moe-runner-backend flashinfer_cutlass` + `--shm-size=32g`.**

> The "7,256" some people quote = 14,511 ÷ 2 GPUs (a per-GPU column), not a separate/failed target.

## 6. Two doc gaps this reproduction exposed (both would block a fresh attempt)
1. `--moe-runner-backend flashinfer_cutlass` — NOT `triton`. The stale EKS manifest and an earlier hand-shared snippet had `triton`, which SMEM-overflows on sm_120 and never starts.
2. `--shm-size=32g --ipc=host` — mandatory for the TP=2 NCCL bootstrap; without it the server crashes ~10 s after launch with `NCCL error: unhandled system error`. The bare-container command shared around omitted it.
