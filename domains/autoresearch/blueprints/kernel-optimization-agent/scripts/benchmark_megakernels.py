#!/usr/bin/env python3
"""Phase 1: Cherry-pick evaluation of existing megakernels on K2.6.

Evaluates each megakernel INDIVIDUALLY against K2.6 baseline:
1. DeepGEMM Mega MoE
2. FlashMoE persistent kernel
3. Alpha-MoE (if available as standalone)

Cherry-pick discipline: each is tested in isolation to avoid
bundled evaluation masking individual regressions.

Usage:
    python benchmark_megakernels.py --kernel deepgemm
    python benchmark_megakernels.py --kernel flashmoe
    python benchmark_megakernels.py --kernel all
"""

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path

import torch
import numpy as np

RESULTS_DIR = "/opt/dlami/nvme/kernel-opt/results"
REPOS_DIR = "/opt/dlami/nvme/kernel-opt/repos"

# K2.6 MoE dimensions
MOE_CONFIG = {
    "hidden_size": 7168,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 384,
    "num_experts_per_tok": 8,
    "n_group": 1,
    "n_shared_experts": 1,
    "dtype": torch.bfloat16,  # weights are FP8 but activations BF16
}


def create_synthetic_moe_inputs(batch_size=128, seq_len=1):
    """Create synthetic inputs matching K2.6 MoE layer shape."""
    hidden = MOE_CONFIG["hidden_size"]
    n_experts = MOE_CONFIG["n_routed_experts"]
    top_k = MOE_CONFIG["num_experts_per_tok"]

    # Hidden states: [batch_size * seq_len, hidden_size]
    tokens = batch_size * seq_len
    hidden_states = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")

    # Router logits: [tokens, n_experts]
    router_logits = torch.randn(tokens, n_experts, dtype=torch.float32, device="cuda")

    # Top-k selection
    topk_weights, topk_ids = torch.topk(router_logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1).to(torch.bfloat16)

    # Expert weights (simplified — in practice these are FP8 quantized)
    # gate_proj: [n_experts, moe_intermediate_size, hidden_size]
    # up_proj:   [n_experts, moe_intermediate_size, hidden_size]
    # down_proj: [n_experts, hidden_size, moe_intermediate_size]
    gate_proj = torch.randn(n_experts, MOE_CONFIG["moe_intermediate_size"], hidden,
                           dtype=torch.bfloat16, device="cuda")
    up_proj = torch.randn(n_experts, MOE_CONFIG["moe_intermediate_size"], hidden,
                         dtype=torch.bfloat16, device="cuda")
    down_proj = torch.randn(n_experts, hidden, MOE_CONFIG["moe_intermediate_size"],
                           dtype=torch.bfloat16, device="cuda")

    return {
        "hidden_states": hidden_states,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "gate_proj": gate_proj,
        "up_proj": up_proj,
        "down_proj": down_proj,
    }


def pytorch_reference_moe(inputs):
    """PyTorch reference MoE implementation for correctness checking."""
    hidden_states = inputs["hidden_states"]
    topk_weights = inputs["topk_weights"]
    topk_ids = inputs["topk_ids"]
    gate_proj = inputs["gate_proj"]
    up_proj = inputs["up_proj"]
    down_proj = inputs["down_proj"]

    tokens, hidden = hidden_states.shape
    top_k = topk_ids.shape[1]
    output = torch.zeros_like(hidden_states)

    for i in range(tokens):
        for j in range(top_k):
            expert_id = topk_ids[i, j].item()
            weight = topk_weights[i, j]
            # SwiGLU: gate * silu(up)
            gate_out = hidden_states[i] @ gate_proj[expert_id].T
            up_out = hidden_states[i] @ up_proj[expert_id].T
            intermediate = torch.nn.functional.silu(gate_out) * up_out
            expert_out = intermediate @ down_proj[expert_id].T
            output[i] += weight * expert_out

    return output


def benchmark_kernel(kernel_fn, inputs, warmup=10, iters=100, name="kernel"):
    """Benchmark a kernel function with warmup and timing."""
    # Warmup
    for _ in range(warmup):
        kernel_fn(inputs)
    torch.cuda.synchronize()

    # Timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        kernel_fn(inputs)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    return {
        "name": name,
        "median_ms": round(np.median(times), 3),
        "mean_ms": round(np.mean(times), 3),
        "std_ms": round(np.std(times), 3),
        "min_ms": round(np.min(times), 3),
        "max_ms": round(np.max(times), 3),
        "p99_ms": round(np.percentile(times, 99), 3),
        "iters": iters,
    }


def check_correctness(kernel_fn, inputs, reference_output, rtol=1e-2, name="kernel"):
    """Check kernel correctness against reference."""
    output = kernel_fn(inputs)
    if output is None:
        return {"name": name, "correct": False, "error": "kernel returned None"}

    max_diff = (output - reference_output).abs().max().item()
    mean_diff = (output - reference_output).abs().mean().item()
    ref_scale = reference_output.abs().mean().item()
    relative_error = max_diff / (ref_scale + 1e-8)

    passed = relative_error < rtol * 5  # 5x tolerance threshold

    return {
        "name": name,
        "correct": passed,
        "max_abs_diff": round(max_diff, 6),
        "mean_abs_diff": round(mean_diff, 6),
        "relative_error": round(relative_error, 6),
        "rtol_threshold": rtol,
    }


def try_deepgemm_moe(inputs):
    """Attempt to run DeepGEMM Mega MoE on K2.6 inputs."""
    sys.path.insert(0, f"{REPOS_DIR}/DeepGEMM")
    try:
        import deep_gemm
        print("  DeepGEMM imported successfully")
        # TODO: Adapt DeepGEMM API for 384-expert flat routing
        # DeepGEMM expects grouped experts — need to verify if flat works
        print("  WARNING: DeepGEMM default is grouped routing, testing flat compatibility...")
        return None  # placeholder until API is adapted
    except ImportError as e:
        print(f"  DeepGEMM import failed: {e}")
        print("  Attempting build...")
        subprocess.run(["pip", "install", "-e", f"{REPOS_DIR}/DeepGEMM"],
                      capture_output=True)
        try:
            import deep_gemm
            return None  # TODO: implement
        except ImportError as e2:
            return {"error": str(e2), "status": "build_failed"}


def try_flashmoe(inputs):
    """Attempt to run FlashMoE persistent kernel on K2.6 inputs."""
    sys.path.insert(0, f"{REPOS_DIR}/FlashMoE")
    try:
        # FlashMoE API
        from flashmoe import moe_forward
        print("  FlashMoE imported successfully")
        # TODO: Adapt FlashMoE API for K2.6 dimensions
        return None  # placeholder
    except ImportError as e:
        print(f"  FlashMoE import failed: {e}")
        return {"error": str(e), "status": "import_failed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["deepgemm", "flashmoe", "all"], default="all")
    parser.add_argument("--batch-sizes", default="1,32,128,512",
                       help="Comma-separated batch sizes to test")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    results = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": {k: str(v) for k, v in MOE_CONFIG.items()}}

    for bs in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch size: {bs} tokens")
        print(f"{'='*60}")

        inputs = create_synthetic_moe_inputs(batch_size=bs)
        print(f"  Hidden states: {inputs['hidden_states'].shape}")
        print(f"  TopK IDs: {inputs['topk_ids'].shape}")
        print(f"  Expert weights: {inputs['gate_proj'].shape}")

        # Reference (always run)
        print("\n  [Reference] PyTorch MoE...")
        if bs <= 32:  # Reference is O(n*k*experts) — only feasible for small batches
            ref_output = pytorch_reference_moe(inputs)
            ref_perf = benchmark_kernel(lambda x: pytorch_reference_moe(x), inputs,
                                       warmup=2, iters=5, name="pytorch_reference")
            print(f"    {ref_perf['median_ms']} ms (median)")
        else:
            print("    Skipped (too slow for large batch)")
            ref_output = None

        # DeepGEMM
        if args.kernel in ("deepgemm", "all"):
            print("\n  [Cherry-pick] DeepGEMM Mega MoE...")
            dg_result = try_deepgemm_moe(inputs)
            results.setdefault("deepgemm", {})[str(bs)] = dg_result or {"status": "not_adapted"}

        # FlashMoE
        if args.kernel in ("flashmoe", "all"):
            print("\n  [Cherry-pick] FlashMoE persistent kernel...")
            fm_result = try_flashmoe(inputs)
            results.setdefault("flashmoe", {})[str(bs)] = fm_result or {"status": "not_adapted"}

    # Save results
    output_file = f"{RESULTS_DIR}/megakernel_eval.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== Results saved to {output_file} ===")


if __name__ == "__main__":
    main()
