#!/usr/bin/env python3
"""model.py — the declarative ServingConfig parsed from a blueprint sidecar.

A ServingConfig is the *what* of a deployment: which model, on which engine with
which optimization flags, on which hardware. It is parsed from the `model:`,
`engine:`, and `infrastructure:` blocks of a benchmark.yaml sidecar (the same
sidecar the benchmark runner consumes) plus optional model-card facts that the
sidecar does not carry (e.g. `moe_intermediate_size`, which lives in the model's
config.json / mdc card).

This module is PURE parsing — no I/O, no validation. Validation lives in
`registry.py` (the rule table) and `compiler.py` (the pure resolver). Keeping the
parse step dumb means the rules operate on a stable, typed shape rather than
re-deriving fields from raw dict access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def _as_int(v) -> Optional[int]:
    """Coerce a scalar to int, tolerating None and numeric strings."""
    if v is None:
        return None
    if isinstance(v, bool):  # bool is a subclass of int — reject silently-wrong values
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    id: str
    architecture: str                 # e.g. "moe", "hybrid-mamba-moe", "dense"
    quantization: str                 # e.g. "fp8", "fp4", "int4", "bf16", "none"
    max_model_len: Optional[int] = None
    # Card facts the sidecar usually lacks — required for some divisibility rules.
    moe_intermediate_size: Optional[int] = None
    # FP8 quant granularity: "block" (fine-grained, block_n=128 → TP-divisibility
    # constraint applies), "per-tensor"/"per-channel" (no block sharding → no
    # constraint), or None (unknown → treat conservatively as block-wise).
    fp8_granularity: Optional[str] = None
    max_position_embeddings: Optional[int] = None
    supports_mtp: Optional[bool] = None         # native multi-token-prediction heads
    is_mla: bool = False                        # MLA / NSA fused-KV attention
    is_mamba_hybrid: bool = False               # attention + mamba/deltanet hybrid

    @property
    def is_moe(self) -> bool:
        return "moe" in (self.architecture or "").lower()

    @property
    def is_fp8(self) -> bool:
        return (self.quantization or "").lower() in ("fp8", "fp8_e4m3", "fp8_e5m2")

    @property
    def is_fp8_block_quantized(self) -> bool:
        """The block_n=128 TP-divisibility constraint only applies to fine-grained
        (block-wise) FP8. Per-tensor / per-channel FP8 has no block sharding, so
        moe_intermediate_size need not be divisible by TP*128. Unknown → treat as
        block-wise (conservative, fail-closed)."""
        if not self.is_fp8:
            return False
        g = (self.fp8_granularity or "").lower()
        if g in ("per-tensor", "per_tensor", "per-channel", "per_channel", "tensor", "channel"):
            return False
        return True  # "block" or unknown


@dataclass(frozen=True)
class SpecDecode:
    algorithm: str                    # "EAGLE3", "MTP", "ngram", ...
    draft_model: Optional[str] = None
    num_steps: Optional[int] = None
    num_draft_tokens: Optional[int] = None
    # Measured acceptance, if a prior benchmark filled it in. None = not yet run.
    measured_acceptance: Optional[float] = None


@dataclass(frozen=True)
class EngineSpec:
    name: str                         # "vllm" | "sglang" | "dynamo" | ...
    version: Optional[str] = None
    container_image: Optional[str] = None
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: Optional[int] = None
    expert_parallel: Optional[int] = None
    kv_cache_dtype: str = "auto"
    attention_backend: str = "auto"
    prefix_caching: Optional[bool] = None
    hierarchical_cache: Optional[bool] = None   # SGLang HiCache enabled
    hicache_size_gb: Optional[int] = None
    cuda_graph: Optional[bool] = None           # None = engine default (usually on)
    speculative_decode: Optional[SpecDecode] = None
    extra_args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class HardwareSpec:
    substrate: str                    # "eks" | "hyperpod" | "ec2-spot" | "local"
    instance_type: str
    region: Optional[str] = None
    gpu_name: Optional[str] = None    # "B200", "B300", "H200", "RTX PRO 6000", ...
    gpu_arch: Optional[str] = None    # "sm_100", "sm_103", "sm_120", "sm_90", ...
    gpu_count: int = 1
    vram_gb: Optional[int] = None
    interconnect: Optional[str] = None  # "nvswitch" | "nvlink" | "pcie"
    ami_family: Optional[str] = None    # "al2023" | "al2" | None (unknown)

    @property
    def is_blackwell(self) -> bool:
        return (self.gpu_arch or "") in ("sm_100", "sm_100f", "sm_103", "sm_120")

    @property
    def is_pcie(self) -> bool:
        return (self.interconnect or "").lower() == "pcie"

    @property
    def is_b200_class(self) -> bool:
        # NVL5+ Fabric Manager constraint applies to p6-b200 / p6-b300 NVSwitch.
        it = (self.instance_type or "").lower()
        return it.startswith("p6-b200") or it.startswith("p6-b300")


@dataclass(frozen=True)
class ServingConfig:
    model: ModelSpec
    engine: EngineSpec
    hardware: HardwareSpec
    source: Optional[str] = None      # provenance: path to the sidecar, for messages


def _parse_spec_decode(raw) -> Optional[SpecDecode]:
    if not raw:
        return None
    return SpecDecode(
        algorithm=str(raw.get("algorithm") or raw.get("method") or "unknown"),
        draft_model=raw.get("draft_model"),
        num_steps=_as_int(raw.get("num_steps")),
        num_draft_tokens=_as_int(raw.get("num_draft_tokens")),
        measured_acceptance=raw.get("measured_acceptance"),
    )


def _arch_flags(architecture: str, model_block: dict) -> tuple[bool, bool]:
    """Derive (is_mla, is_mamba_hybrid) from architecture string + explicit hints."""
    a = (architecture or "").lower()
    is_mla = bool(model_block.get("is_mla")) or any(
        k in a for k in ("mla", "nsa", "dsa", "latent"))
    is_mamba = bool(model_block.get("is_mamba_hybrid")) or any(
        k in a for k in ("mamba", "deltanet", "hybrid"))
    return is_mla, is_mamba


def from_sidecar(sidecar: dict, *, card: Optional[dict] = None,
                 source: Optional[str] = None) -> ServingConfig:
    """Build a ServingConfig from a parsed sidecar dict.

    `card` is an optional model-deployment-card dict supplying facts the sidecar
    lacks (moe_intermediate_size, supports_mtp, max_position_embeddings). Sidecar
    values win when both are present — the sidecar is the deployment's own truth.
    """
    card = card or {}
    m = sidecar.get("model", {}) or {}
    e = sidecar.get("engine", {}) or {}
    infra = sidecar.get("infrastructure", {}) or {}
    gpu = infra.get("gpu", {}) or {}

    arch = m.get("architecture", "") or ""
    is_mla, is_mamba = _arch_flags(arch, m)

    model = ModelSpec(
        name=m.get("name", "unknown"),
        id=m.get("id", m.get("name", "unknown")),
        architecture=arch,
        quantization=str(m.get("quantization", "none")),
        max_model_len=_as_int(m.get("max_model_len")),
        moe_intermediate_size=_as_int(
            m.get("moe_intermediate_size", card.get("moe_intermediate_size"))),
        fp8_granularity=m.get("fp8_granularity", card.get("fp8_granularity")),
        max_position_embeddings=_as_int(
            m.get("max_position_embeddings", card.get("max_position_embeddings"))),
        supports_mtp=m.get("supports_mtp", card.get("supports_mtp")),
        is_mla=is_mla,
        is_mamba_hybrid=is_mamba,
    )

    engine = EngineSpec(
        name=str(e.get("name", "unknown")),
        version=e.get("version") or e.get("container_image"),
        container_image=e.get("container_image"),
        tensor_parallel=_as_int(e.get("tensor_parallel")) or 1,
        pipeline_parallel=_as_int(e.get("pipeline_parallel")) or 1,
        data_parallel=_as_int(e.get("data_parallel")),
        expert_parallel=_as_int(e.get("expert_parallel")),
        kv_cache_dtype=str(e.get("kv_cache_dtype", "auto")),
        attention_backend=str(e.get("attention_backend", "auto")),
        prefix_caching=e.get("prefix_caching"),
        hierarchical_cache=e.get("hierarchical_cache"),
        hicache_size_gb=_as_int(e.get("hicache_size_gb")),
        cuda_graph=e.get("cuda_graph"),
        speculative_decode=_parse_spec_decode(e.get("speculative_decode")),
        extra_args=dict(e.get("extra_args", {}) or {}),
    )

    hardware = HardwareSpec(
        substrate=str(infra.get("substrate", "unknown")),
        instance_type=str(infra.get("instance_type", "unknown")),
        region=infra.get("region"),
        gpu_name=gpu.get("name"),
        gpu_arch=gpu.get("arch"),
        gpu_count=_as_int(gpu.get("count")) or 1,
        vram_gb=_as_int(gpu.get("vram_gb")),
        interconnect=gpu.get("interconnect"),
        ami_family=infra.get("ami_family"),
    )

    return ServingConfig(model=model, engine=engine, hardware=hardware, source=source)
