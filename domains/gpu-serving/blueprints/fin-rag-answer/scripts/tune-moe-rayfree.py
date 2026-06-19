#!/usr/bin/env python3
"""Ray-free single-GPU MoE tile tuner for Nemotron-3-Super FP8 on B200.

The stock vLLM benchmark_moe.py imports `ray` at module top-level purely for
multi-GPU orchestration; the GPU nodes have no pip egress and ray isn't baked
into the image. We stub ray in sys.modules, then reuse the script's own
benchmark_config / get_configs_compute_bound / save_configs against ONE GPU.

Tunes the grouped FP8 expert GEMM (E=512, N=1344 @ TP=2, top-22) over the
decode-step batch sizes seen per replica at the c192 SLO knee (~48 seqs/replica).
Writes the tuned config into vLLM's fused_moe/configs/ so the running model
auto-loads it on next start.

Usage (inside the vLLM container, one visible GPU):
  CUDA_VISIBLE_DEVICES=0 python3 tune-moe-rayfree.py \
      --model /fsx/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
      --tp-size 2 --dtype fp8_w8a8 \
      --batches 16,24,32,48,64,96,128 --save-dir /tmp/tuned-moe
"""
import argparse, sys, types, json, os, time

# --- stub ray BEFORE importing the benchmark module ---
ray = types.ModuleType("ray")
def _remote(*a, **k):
    def deco(x): return x
    return deco
ray.remote = _remote
ray.init = lambda *a, **k: None
ray.get = lambda x: x
ray.get_gpu_ids = lambda: [0]
ray.available_resources = lambda: {"GPU": 1}
sys.modules["ray"] = ray
ray_exp = types.ModuleType("ray.experimental")
ray_tqdm = types.ModuleType("ray.experimental.tqdm_ray")
from tqdm import tqdm as _tqdm
ray_tqdm.tqdm = _tqdm
ray_exp.tqdm_ray = ray_tqdm
sys.modules["ray.experimental"] = ray_exp
sys.modules["ray.experimental.tqdm_ray"] = ray_tqdm

# Now import the stock tuner's internals
sys.path.insert(0, "/vllm-workspace/benchmarks/kernels")
import benchmark_moe as bm
import torch
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--dtype", default="fp8_w8a8")
    ap.add_argument("--batches", default="16,24,32,48,64,96,128")
    ap.add_argument("--save-dir", default="/tmp/tuned-moe")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--prune", action="store_true",
                    help="restrict to small-batch decode-regime tiles (DCGM showed "
                         "the c192 knee is launch-bound, not FLOP-bound)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA device visible"
    torch.cuda.set_device(0)
    torch.set_default_device("cuda")  # benchmark_config allocates w/o device kwarg

    cfg = get_config(model=args.model, trust_remote_code=True)
    E, topk, intermediate_size, hidden_size = bm.get_model_params(cfg)
    bm.ensure_divisibility(intermediate_size, args.tp_size, "intermediate_size")
    shard_intermediate_size = 2 * intermediate_size // args.tp_size
    dtype = cfg.dtype
    use_fp8 = args.dtype == "fp8_w8a8"
    use_int8 = args.dtype == "int8_w8a16"
    block_quant_shape = bm.get_weight_block_size_safety(cfg)

    print(f"[tune] E={E} topk={topk} inter={intermediate_size} hidden={hidden_size} "
          f"shard_inter={shard_intermediate_size} dtype={args.dtype} "
          f"block_quant_shape={block_quant_shape} device={torch.cuda.get_device_name(0)}")

    is_fp16 = not (use_fp8 or use_int8)
    search_space = bm.get_configs_compute_bound(is_fp16, block_quant_shape)
    if args.prune:
        # At the c192 knee DCGM showed launch/scheduling-bound, small per-step
        # token counts: tokens/expert is tiny, so large BLOCK_M wastes a tile.
        # Keep small-M, modest-N/K tiles + the stage/warp counts that hide
        # launch latency. This cuts the search ~6x for a tractable single-GPU run.
        from itertools import product
        ranges = dict(BLOCK_SIZE_M=[16, 32, 64], BLOCK_SIZE_N=[64, 128, 256],
                      BLOCK_SIZE_K=[128, 256], GROUP_SIZE_M=[1, 16, 32],
                      num_warps=[4, 8], num_stages=[3, 4, 5])
        keys, vals = zip(*ranges.items())
        search_space = [dict(zip(keys, c)) for c in product(*vals)]
    print(f"[tune] search space = {len(search_space)} configs/batch "
          f"({'pruned' if args.prune else 'full'})")

    batches = [int(x) for x in args.batches.split(",")]
    best = {}
    for nt in batches:
        t0 = time.time()
        best_cfg, best_t = None, float("inf")
        for idx, c in enumerate(_tqdm(search_space, desc=f"bs={nt}")):
            try:
                kt = bm.benchmark_config(
                    c, nt, E, shard_intermediate_size, hidden_size, topk,
                    dtype, use_fp8, use_int8, False,
                    num_iters=20, block_quant_shape=block_quant_shape,
                    use_deep_gemm=False)
            except Exception:
                continue
            if kt < best_t:
                best_t, best_cfg = kt, c
            if idx and idx % 50 == 0:
                bm.clear_triton_cache()
        bm.clear_triton_cache()
        assert best_cfg is not None, f"no valid config for bs={nt}"
        best[nt] = bm.sort_config(best_cfg)
        print(f"[tune] bs={nt} best={best[nt]} {best_t*1e6:.1f}us "
              f"({time.time()-t0:.0f}s)")

    bm.save_configs(best, E, shard_intermediate_size, hidden_size, topk,
                    dtype, use_fp8, use_int8, False, block_quant_shape,
                    args.save_dir)
    print("[tune] DONE")


if __name__ == "__main__":
    main()
