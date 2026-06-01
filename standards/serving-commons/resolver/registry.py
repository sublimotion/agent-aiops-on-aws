#!/usr/bin/env python3
"""registry.py — the deterministic rule table for serving configs.

THE CONTRACT. Every hard-won serving rule that can be checked from a declared
config lives here as a pure function. The compiler runs them all and fails
CLOSED on any FAIL. This is the serving-side analog of benchmark-commons'
DATASET_HANDLERS / LOAD_EXPANDERS: a lookup table that an LLM cannot
reinterpret.

A check is `(ServingConfig) -> Optional[Finding]`. Return None when the rule does
not apply to this config (e.g. an FP8-MoE rule on a dense bf16 model). Return a
Finding with:
  - verdict: "fail" (violates a hard rule — config must not deploy as-is),
             "warn"  (likely-wrong but deploy may proceed; operator must read),
             "info"  (an applicable note worth surfacing).
  - reason: the verbatim consequence, copied from the steering rule so the
            message is identical to what an operator would read in tech-stack.md.
  - source: where the rule is documented, so a human can audit it.
  - fix:    the concrete remediation, when there is a deterministic one.

Each Finding's `reason` text is lifted verbatim from
`.claude/steering/tech-stack.md` — do NOT paraphrase. If the steering rule
changes, update the reason here and the citation date. The conformance test
asserts every check has a non-empty source.

To add a rule: write a `_chk_*` function, append it to CHECKS, and add a
passing+failing fixture in tests/. See CONTRIBUTING.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from model import ServingConfig


@dataclass(frozen=True)
class Finding:
    rule: str                 # short stable id, e.g. "fp8-moe-tp-divisibility"
    verdict: str              # "fail" | "warn" | "info"
    reason: str               # verbatim consequence
    source: str               # citation
    fix: Optional[str] = None # deterministic remediation, if any


Check = Callable[[ServingConfig], Optional[Finding]]

TECH_STACK = ".claude/steering/tech-stack.md"


# ---------------------------------------------------------------------------
# Hard rules — a FAIL blocks deployment. These are deterministic arithmetic or
# compatibility facts, not judgment calls.
# ---------------------------------------------------------------------------

def _chk_fp8_moe_tp_divisibility(cfg: ServingConfig) -> Optional[Finding]:
    """FP8 MoE: moe_intermediate_size / TP must be divisible by block_n=128."""
    m, e = cfg.model, cfg.engine
    if not (m.is_moe and m.is_fp8):
        return None
    if m.moe_intermediate_size is None:
        return Finding(
            rule="fp8-moe-tp-divisibility",
            verdict="warn",
            reason=("FP8 MoE model but moe_intermediate_size is unknown — cannot "
                    "verify TP divisibility. vLLM raises ValueError: output_size "
                    "not divisible by block_n at model load if it fails."),
            source=f"{TECH_STACK} §'FP8 MoE models require TP divisibility check against block_n=128'",
            fix=("Add model.moe_intermediate_size to the sidecar (from the "
                 "model's config.json) so this check can run before capacity reservation."))
    tp = e.tensor_parallel
    if m.moe_intermediate_size % (tp * 128) == 0:
        return None
    # Suggest the largest divisor-of-config TP that satisfies the rule.
    ok = [t for t in (1, 2, 4, 8) if t <= max(tp, cfg.hardware.gpu_count)
          and m.moe_intermediate_size % (t * 128) == 0]
    return Finding(
        rule="fp8-moe-tp-divisibility",
        verdict="fail",
        reason=(f"For fine-grained FP8 quantized MoE models (block_size=128), "
                f"moe_intermediate_size / tensor_parallel_size must be divisible "
                f"by 128. moe_intermediate_size={m.moe_intermediate_size}, TP={tp} "
                f"-> {m.moe_intermediate_size // tp if tp else 0} which is not "
                f"divisible by 128. vLLM raises ValueError: output_size not "
                f"divisible by block_n at model load time."),
        source=f"{TECH_STACK} §'FP8 MoE models require TP divisibility check against block_n=128' (qwen3-235b-b300 lessons L1)",
        fix=(f"Use TP in {ok} which satisfies moe_intermediate_size % (TP*128) == 0."
             if ok else "No TP in {1,2,4,8} satisfies the rule for this model; "
             "check moe_intermediate_size or use a non-block-quantized variant."))


def _chk_max_model_len_vs_position(cfg: ServingConfig) -> Optional[Finding]:
    """--max-model-len above max_position_embeddings is refused by vLLM."""
    m = cfg.model
    if m.max_model_len is None or m.max_position_embeddings is None:
        return None
    if m.max_model_len <= m.max_position_embeddings:
        return None
    return Finding(
        rule="max-model-len-vs-position",
        verdict="fail",
        reason=(f"max_model_len={m.max_model_len} exceeds the model's "
                f"max_position_embeddings={m.max_position_embeddings}. vLLM refuses "
                f"--max-model-len above max_position_embeddings unless "
                f"VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 is set. Model cards often cite "
                f"YaRN-extended lengths that differ from the actual config.json."),
        source=f"{TECH_STACK} §'Always verify max_position_embeddings from downloaded config.json' (qwen3-235b-b300 lessons L2)",
        fix=(f"Set max_model_len <= {m.max_position_embeddings}, or set "
             f"VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 deliberately if YaRN scaling is intended."))


def _chk_b200_requires_al2023(cfg: ServingConfig) -> Optional[Finding]:
    """p6-b200/b300 NVSwitch needs AL2023 (kernel has ib_umad for Fabric Manager)."""
    h = cfg.hardware
    if not h.is_b200_class:
        return None
    if (h.ami_family or "").lower() == "al2":
        return Finding(
            rule="b200-requires-al2023",
            verdict="fail",
            reason=("Amazon Linux 2 (kernel 5.10) does not compile the ib_umad "
                    "kernel module required by NVIDIA Fabric Manager on NVL5+ "
                    "systems. Without Fabric Manager, CUDA returns error 802 "
                    "(cudaErrorSystemNotReady) — nvidia-smi shows GPUs on the host "
                    "but containers cannot access them."),
            source=f"{TECH_STACK} §'B200 NVL5+ requires AL2023 AMI'",
            fix="Use an Amazon Linux 2023 NVIDIA AMI (e.g. amazon-eks-node-al2023-x86_64-nvidia-1.32, kernel 6.1).")
    return None


def _chk_hicache_size_vs_device_pool(cfg: ServingConfig) -> Optional[Finding]:
    """SGLang HiCache asserts host_memory > device_memory at init."""
    e = cfg.engine
    if not e.hierarchical_cache:
        return None
    if e.hicache_size_gb is None:
        return Finding(
            rule="hicache-size-vs-device-pool",
            verdict="warn",
            reason=("HiCache is enabled but --hicache-size is unset. SGLang HiCache "
                    "asserts host_memory > device_memory during initialization. The "
                    "default --hicache-ratio 2.0 can exceed available system RAM and "
                    "OOM on memory-constrained instances."),
            source=f"{TECH_STACK} §'HiCache --hicache-size must exceed device KV pool size to pass initialization assertion'",
            fix="Set engine.hicache_size_gb to at least the device KV pool size + margin (e.g. 100 for an ~82 GB/rank pool).")
    return None


# ---------------------------------------------------------------------------
# Warnings — likely-wrong, operator must read. Deploy may proceed.
# ---------------------------------------------------------------------------

def _chk_mtp_on_pcie(cfg: ServingConfig) -> Optional[Finding]:
    """MTP spec-decode degrades throughput on PCIe-interconnected GPUs."""
    e, h = cfg.engine, cfg.hardware
    sd = e.speculative_decode
    if sd is None or not h.is_pcie:
        return None
    if (sd.algorithm or "").lower() not in ("mtp", "qwen3_next_mtp", "eagle3", "eagle", "ngram"):
        return None
    return Finding(
        rule="specdec-on-pcie",
        verdict="warn",
        reason=("Speculative decoding adds inter-GPU communication overhead for "
                "speculative head computation and verification. On PCIe-"
                "interconnected GPUs this overhead exceeds the benefit, causing "
                "throughput degradation of 2-41% across QPS levels. Spec-decode is "
                "designed for NVLink-interconnected GPUs (10-20x higher bandwidth)."),
        source=f"{TECH_STACK} §'MTP speculative decoding degrades throughput on PCIe-interconnected GPUs'",
        fix="Default to baseline (no spec-decode) on PCIe platforms; benchmark on target hardware before enabling.")


def _chk_specdec_acceptance_gate(cfg: ServingConfig) -> Optional[Finding]:
    """<60% measured draft acceptance is a net throughput LOSS."""
    sd = cfg.engine.speculative_decode
    if sd is None:
        return None
    if sd.measured_acceptance is None:
        return Finding(
            rule="specdec-acceptance-gate",
            verdict="info",
            reason=("Speculative decoding is profitable only above ~60% draft "
                    "acceptance; <60% is net negative (verification cost dominates, "
                    "ITL roughly doubles). Acceptance has not been measured for this "
                    "config yet."),
            source=f"{TECH_STACK} §'Speculative decoding under ~60% draft acceptance rate is a net throughput LOSS'",
            fix=("Before promoting to production, run a real-distribution workload "
                 "(sharegpt/production-mix, NOT synthetic random) and read "
                 "vllm:spec_decode_num_accepted_tokens_total / "
                 "vllm:spec_decode_num_drafts_total. Disable if <0.60."))
    if sd.measured_acceptance < 0.60:
        return Finding(
            rule="specdec-acceptance-gate",
            verdict="fail",
            reason=(f"Measured draft acceptance {sd.measured_acceptance:.0%} is below "
                    f"the ~60% break-even. <60% acceptance is a net throughput LOSS "
                    f"— verification pass cost dominates and ITL roughly doubles."),
            source=f"{TECH_STACK} §'Speculative decoding under ~60% draft acceptance rate is a net throughput LOSS'",
            fix="Disable speculative decoding for this model/workload.")
    return None


def _chk_mamba_mtp_prefix_cache(cfg: ServingConfig) -> Optional[Finding]:
    """Mamba-hybrid + MTP conflicts with prefix caching (mamba 'align' mode)."""
    m, e = cfg.model, cfg.engine
    sd = e.speculative_decode
    if not (m.is_mamba_hybrid and sd is not None):
        return None
    if e.prefix_caching is not False:  # True or unset (default on)
        return Finding(
            rule="mamba-mtp-prefix-cache",
            verdict="warn",
            reason=("MTP speculative decoding conflicts with mamba 'align' mode in "
                    "vLLM, requiring --no-enable-prefix-caching to work at all "
                    "(which further degrades performance). Hybrid attention+mamba "
                    "architectures trigger mamba cache mode."),
            source=f"{TECH_STACK} §'Mamba hybrid architectures have different caching and speculative decoding constraints'",
            fix="Set engine.prefix_caching: false when running MTP on a mamba-hybrid model, or drop spec-decode.")
    return None


def _chk_hybrid_hicache_cuda_graph(cfg: ServingConfig) -> Optional[Finding]:
    """Hybrid attention + HiCache requires CUDA graph disabled."""
    m, e = cfg.model, cfg.engine
    if not (m.is_mamba_hybrid and e.hierarchical_cache):
        return None
    if e.cuda_graph is not False:
        return Finding(
            rule="hybrid-hicache-cuda-graph",
            verdict="warn",
            reason=("CUDA graph compilation conflicts with HiCache's dynamic memory "
                    "management for hybrid attention models (e.g. DeltaNet+GQA). "
                    "Use --disable-cuda-graph."),
            source=f"{TECH_STACK} §'Hybrid attention + HiCache requires CUDA graph disabled'",
            fix="Set engine.cuda_graph: false for hybrid-architecture + HiCache deployments.")
    return None


def _chk_lmcache_mla_incompat(cfg: ServingConfig) -> Optional[Finding]:
    """LMCache crashes on MLA/NSA fused-KV models — use SGLang HiCache instead."""
    m, e = cfg.model, cfg.engine
    uses_lmcache = bool(e.extra_args.get("enable-lmcache")) or \
        "lmcache" in (e.name or "").lower()
    if not (m.is_mla and uses_lmcache):
        return None
    return Finding(
        rule="lmcache-mla-incompat",
        verdict="fail",
        reason=("LMCache's adapter expects separate k_buffer/v_buffer attributes. "
                "MLA/NSA models (e.g. GLM-5 glm_moe_dsa, DeepSeek V3) use a fused "
                "kv_buffer and crash with AttributeError: object has no attribute "
                "'k_buffer'. LMCache PR #2629 (MLA layerwise) is not merged."),
        source=f"{TECH_STACK} §'LMCache v0.3.15 incompatible with SGLang NSA/MLA attention'",
        fix="Use SGLang built-in HiCache (--enable-hierarchical-cache) for MLA models; it understands the fused kv_buffer.")


def _chk_blackwell_cuda_tag(cfg: ServingConfig) -> Optional[Finding]:
    """Blackwell sm_120 needs cu130, not cu131 (tag may not exist)."""
    h, e = cfg.hardware, cfg.engine
    img = (e.container_image or "")
    if h.gpu_arch != "sm_120":
        return None
    if "cu131" in img:
        return Finding(
            rule="blackwell-cuda-tag",
            verdict="warn",
            reason=("For Blackwell GPUs (sm_120) use CUDA 13.0 (cu130), not CUDA "
                    "13.1 (cu131). Not all registries publish cu131 tags; pulling a "
                    "non-existent tag wastes deployment time."),
            source=f"{TECH_STACK} §'Verify CUDA image tags before deployment — cu131 vs cu130 for Blackwell'",
            fix="Use a cu130 image tag (e.g. lmsysorg/sglang:v0.5.9-cu130).")
    return None


# Order: hard fails first (so the first FAIL surfaced is a real blocker), then
# warnings, then info. The compiler runs ALL of them regardless of order.
CHECKS: list[Check] = [
    _chk_fp8_moe_tp_divisibility,
    _chk_max_model_len_vs_position,
    _chk_b200_requires_al2023,
    _chk_lmcache_mla_incompat,
    _chk_hicache_size_vs_device_pool,
    _chk_mtp_on_pcie,
    _chk_mamba_mtp_prefix_cache,
    _chk_hybrid_hicache_cuda_graph,
    _chk_blackwell_cuda_tag,
    _chk_specdec_acceptance_gate,
]
