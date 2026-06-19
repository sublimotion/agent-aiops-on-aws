#!/usr/bin/env python3
"""Seed the constraint database with K2.6 architectural + H200 hardware facts.

Creates the initial constraints.jsonl that the kernel optimization agent
injects into every generation prompt. These are "hard" constraints — always
active, never demoted.

Usage:
    python seed_constraints.py --output /opt/dlami/nvme/kernel-opt/results/constraints.jsonl
"""

import json
import argparse
from datetime import datetime

SEED_CONSTRAINTS = [
    # === K2.6 Architecture Constraints (Hard) ===
    {
        "id": "arch-001",
        "type": "hard",
        "category": "architecture",
        "region": "all",
        "rule": "K2.6 has 384 routed experts with n_group=1 (flat routing, no grouped top-k). Top-8 experts selected from full pool of 384.",
        "evidence": "config.json: n_routed_experts=384, num_experts_per_tok=8, n_group=1",
        "implication": "Dispatch tile sizes must handle 384-way load balancing. Group-based optimizations (DeepSeek V3 uses n_group=8) do NOT apply.",
    },
    {
        "id": "arch-002",
        "type": "hard",
        "category": "architecture",
        "region": "moe_dispatch",
        "rule": "K2.6 MoE intermediate size is 2048 per expert (vs DeepSeek V3's 2048). Total MoE capacity: 384 * 2048 = 786,432 parameters per layer.",
        "evidence": "config.json: moe_intermediate_size=2048",
        "implication": "Expert weight matrices are small (2048x7168 and 7168x2048). Memory bandwidth dominates for single-token decoding.",
    },
    {
        "id": "arch-003",
        "type": "hard",
        "category": "architecture",
        "region": "mla_decode",
        "rule": "K2.6 uses MLA (Multi-head Latent Attention) with kv_lora_rank=512, v_head_dim=128, q_lora_rank=1536, qk_nope_head_dim=128, qk_rope_head_dim=64.",
        "evidence": "config.json: identical MLA dims to DeepSeek V3",
        "implication": "FlashInfer MLA kernels optimized for DS V3 should work directly. KV cache is compressed (512-dim latent vs 64*128*2=16384 full).",
    },
    {
        "id": "arch-004",
        "type": "hard",
        "category": "architecture",
        "region": "mla_decode",
        "rule": "K2.6 has 64 attention heads (vs DeepSeek V3's 128). num_key_value_heads=64.",
        "evidence": "config.json: num_attention_heads=64, num_key_value_heads=64",
        "implication": "Half the head count means different optimal parallelism. Kernels may have different occupancy characteristics.",
    },
    {
        "id": "arch-005",
        "type": "hard",
        "category": "architecture",
        "region": "all",
        "rule": "K2.6 has 61 transformer layers with MoE in every layer (moe_layer_freq=1, first_k_dense_replace=1). Only layer 0 is dense.",
        "evidence": "config.json: num_hidden_layers=61, moe_layer_freq=1, first_k_dense_replace=1",
        "implication": "MoE dispatch is on the critical path for 60/61 layers. Even small per-layer improvement compounds 60x.",
    },
    {
        "id": "arch-006",
        "type": "hard",
        "category": "architecture",
        "region": "moe_dispatch",
        "rule": "K2.6 has 1 shared expert that always runs alongside the top-8 routed experts (n_shared_experts=1).",
        "evidence": "config.json: n_shared_experts=1",
        "implication": "Fused shared-expert kernels (vLLM #39280) apply. Shared expert can overlap with routed computation.",
    },
    # === H200 Hardware Constraints (Hard) ===
    {
        "id": "hw-001",
        "type": "hard",
        "category": "hardware",
        "region": "all",
        "rule": "H200 SM90 (Hopper): 132 SMs, 228KB shared memory per SM (configurable L1/shared split), 80GB HBM3e per GPU at 3.35 TB/s.",
        "evidence": "nvidia-smi + H200 datasheet. p5en has 141GB variant.",
        "implication": "Actually 141GB HBM3e on p5en. Peak BW 3.35 TB/s. Shared memory budget: 227KB usable.",
    },
    {
        "id": "hw-002",
        "type": "hard",
        "category": "hardware",
        "region": "all",
        "rule": "H200 peak FP8 tensor core throughput: 3,958 TFLOPS (with sparsity) / 1,979 TFLOPS (dense).",
        "evidence": "H200 datasheet",
        "implication": "Arithmetic intensity threshold for compute-bound: >590 ops/byte (dense FP8). Most MoE dispatch is memory-bound.",
    },
    {
        "id": "hw-003",
        "type": "hard",
        "category": "hardware",
        "region": "all",
        "rule": "p5en.48xlarge: 8x H200 connected via NVLink 4 / NVSwitch. 900 GB/s bidirectional NVLink bandwidth per GPU.",
        "evidence": "AWS instance specs + nvidia-smi topo -m",
        "implication": "Tensor parallelism across 8 GPUs is fast (900 GB/s NVLink >> PCIe). Allreduce not a bottleneck for TP8.",
    },
    {
        "id": "hw-004",
        "type": "hard",
        "category": "hardware",
        "region": "all",
        "rule": "Hopper TMA (Tensor Memory Accelerator) available. Enables async bulk memory copies without SM involvement.",
        "evidence": "SM90 architecture",
        "implication": "Triton kernels can use tl.async_copy / tl.tensor_descriptor for HBM→shared prefetch. Critical for memory-bound MoE dispatch.",
    },
    # === B300 Reference Constraints (for Phase 3 transfer) ===
    {
        "id": "hw-b300-001",
        "type": "soft",
        "category": "hardware",
        "region": "all",
        "rule": "B300 SM103 (Blackwell): 227KB shared memory, TMA, TCGEN5 (5th gen tensor cores), NVLink 5, 2.4 TB/s HBM3e (lower than H200).",
        "evidence": "B300 datasheet + prior benchmarks",
        "implication": "Phase 3 autotuning needed. Lower BW means MoE dispatch even more memory-bound. Tile sizes may differ.",
    },
    # === FP8 Correctness Constraints ===
    {
        "id": "fp8-001",
        "type": "hard",
        "category": "correctness",
        "region": "all",
        "rule": "FP8 block quantization uses 128x128 blocks. Dequantization must respect block boundaries.",
        "evidence": "quantization_config.weights.block_structure=[128,128]",
        "implication": "Custom kernels touching weights must handle FP8 dequant at block granularity. Input activations use group_size=128.",
    },
    {
        "id": "fp8-002",
        "type": "hard",
        "category": "correctness",
        "region": "all",
        "rule": "FP8 correctness tolerance: rtol=1e-2 with 100-input statistical verification. Reject if max deviation exceeds 5x tolerance on any single input.",
        "evidence": "Spec definition",
        "implication": "Aggressive FP8 optimizations must be verified against BF16 reference. Edge cases (large magnitudes, near-zero) need explicit testing.",
    },
    # === Known Dead Ends (from upstream) ===
    {
        "id": "dead-001",
        "type": "soft",
        "category": "performance",
        "region": "moe_dispatch",
        "rule": "Group-based expert routing optimizations (batching by group) are INAPPLICABLE to K2.6 because n_group=1.",
        "evidence": "DeepSeek V3 uses n_group=8, K2.6 uses n_group=1",
        "implication": "Skip any approach that assumes grouped expert selection. The full 384-expert pool is flat.",
    },
    {
        "id": "dead-002",
        "type": "soft",
        "category": "performance",
        "region": "moe_dispatch",
        "rule": "DeepGEMM Mega MoE default configs are tuned for 256 experts with 8-group routing. Will need re-tuning for 384 flat.",
        "evidence": "DeepGEMM source: default expert_count=256, n_group=8",
        "implication": "Cannot use DeepGEMM out-of-box for K2.6. Must modify dispatch logic and re-autotune tile parameters.",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/opt/dlami/nvme/kernel-opt/results/constraints.jsonl")
    args = parser.parse_args()

    timestamp = datetime.now().isoformat()

    with open(args.output, "w") as f:
        for constraint in SEED_CONSTRAINTS:
            constraint["timestamp"] = timestamp
            constraint["source"] = "seed"
            constraint["severity"] = "critical" if constraint["type"] == "hard" else "high"
            f.write(json.dumps(constraint) + "\n")

    print(f"Seeded {len(SEED_CONSTRAINTS)} constraints to {args.output}")
    print(f"  Hard: {sum(1 for c in SEED_CONSTRAINTS if c['type'] == 'hard')}")
    print(f"  Soft: {sum(1 for c in SEED_CONSTRAINTS if c['type'] == 'soft')}")
    print(f"  Categories: {set(c['category'] for c in SEED_CONSTRAINTS)}")
    print(f"  Regions: {set(c['region'] for c in SEED_CONSTRAINTS)}")


if __name__ == "__main__":
    main()
