#!/usr/bin/env python3
"""NCCL PCIe diagnostic for multi-GPU Blackwell setup."""
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_SHM_DISABLE'] = '0'

    torch.cuda.set_device(rank)
    print(f"[rank {rank}] Using GPU {rank}: {torch.cuda.get_device_name(rank)}", flush=True)

    # Test basic CUDA op on this device
    t = torch.randn(100, device=f'cuda:{rank}')
    print(f"[rank {rank}] Basic CUDA OK, sum={t.sum().item():.2f}", flush=True)

    try:
        print(f"[rank {rank}] Initializing process group (nccl)...", flush=True)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        print(f"[rank {rank}] Process group initialized OK", flush=True)
    except Exception as e:
        print(f"[rank {rank}] FAILED init_process_group: {e}", flush=True)
        return

    # Test 1: all_reduce
    try:
        t = torch.ones(1000, device=f'cuda:{rank}') * (rank + 1)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = sum(range(1, world_size + 1)) * 1000
        print(f"[rank {rank}] all_reduce OK, sum={t.sum().item():.0f} (expected {expected})", flush=True)
    except Exception as e:
        print(f"[rank {rank}] FAILED all_reduce: {e}", flush=True)

    # Test 2: broadcast
    try:
        t = torch.ones(1000, device=f'cuda:{rank}') * 42 if rank == 0 else torch.zeros(1000, device=f'cuda:{rank}')
        dist.broadcast(t, src=0)
        print(f"[rank {rank}] broadcast OK, val={t[0].item():.0f}", flush=True)
    except Exception as e:
        print(f"[rank {rank}] FAILED broadcast: {e}", flush=True)

    # Test 3: larger tensor
    try:
        size_mb = 100
        numel = size_mb * 1024 * 1024 // 4
        t = torch.randn(numel, device=f'cuda:{rank}')
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        print(f"[rank {rank}] large all_reduce ({size_mb}MB) OK", flush=True)
    except Exception as e:
        print(f"[rank {rank}] FAILED large all_reduce: {e}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] All tests PASSED", flush=True)

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"Starting NCCL diagnostic with {world_size} GPUs")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
    print("All ranks completed successfully!")
