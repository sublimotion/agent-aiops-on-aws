#!/usr/bin/env python3
"""compiler.py — pure card->ExecutionPlan compiler.

`compile_card(card, sidecar, tool)` is a PURE function: same inputs always
produce the same plan, no network, no filesystem, no side effects. That is what
makes the runner deterministic and the conformance test possible.

Contract:
  - Resolves dataset.type + load.type via registry.py.
  - Applies sidecar `override` / `workload_overrides` deterministically.
  - Expands sweeps (qps, concurrency, context-length, quant modes) into a flat
    list of concrete steps, each with its own argv and filename label.
  - Propagates the goodput SLO (ttft/tpot) into vendor argv.
  - VENDOR plans carry ready-to-exec argv. ORCHESTRATED plans carry an executor
    name + reason (resolved in orchestrators.py).
  - FAILS CLOSED: an unmapped dataset/load type raises UnsupportedWorkload.
    Never a silent default.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from registry import (
    DATASET_HANDLERS, LOAD_EXPANDERS, UnsupportedWorkload,
    DatasetResolution, LoadStep, _OrchestratedLoad,
)


@dataclass
class PlanStep:
    """One fully-resolved execution point ready to run or dry-run."""
    label: str                       # filename suffix component
    argv: list[str]                  # complete vendor benchmark argv (no endpoint/model)
    concurrency: Optional[int] = None
    request_rate: Optional[float] = None
    num_prompts: Optional[int] = None
    duration_s: Optional[int] = None
    context_len: Optional[int] = None
    mode: Optional[str] = None       # quant pareto: offline|server


@dataclass
class ExecutionPlan:
    catalog_id: str
    tool: str
    kind: str                        # "vendor" | "orchestrated"
    modality: str = "text"
    steps: list[PlanStep] = field(default_factory=list)        # vendor only
    orchestrator: Optional[str] = None                         # orchestrated only
    reason: Optional[str] = None
    dataset_summary: dict = field(default_factory=dict)
    slo_targets: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Sidecar override merge — deterministic deep-merge of card under sidecar.
# --------------------------------------------------------------------------

def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _find_card_override(sidecar: dict, catalog_id: str) -> dict:
    """Locate the per-card override block in the sidecar's `workloads:` list.
    Returns {} when absent. Supports both `override:` and `workload_overrides:`.
    """
    for entry in (sidecar.get("workloads") or []):
        if entry.get("catalog_id") == catalog_id:
            return entry.get("override") or entry.get("workload_overrides") or {}
    # Top-level workload_overrides (used by burn-in / cohost / trace-replay cards)
    return sidecar.get("workload_overrides", {})


# --------------------------------------------------------------------------
# Goodput — turn card.slo into a vendor --goodput flag (ttft:X tpot:Y in ms).
# --------------------------------------------------------------------------

def _goodput_argv(slo: dict) -> list[str]:
    parts = []
    # vendor --goodput accepts ttft / tpot / e2el keys in ms
    if slo.get("ttft_p99_ms") is not None:
        parts.append(f"ttft:{int(slo['ttft_p99_ms'])}")
    if slo.get("tpot_p99_ms") is not None:
        parts.append(f"tpot:{int(slo['tpot_p99_ms'])}")
    if slo.get("e2e_p99_ms") is not None:
        parts.append(f"e2el:{int(slo['e2e_p99_ms'])}")
    return ["--goodput", *parts] if parts else []


# --------------------------------------------------------------------------
# The compiler.
# --------------------------------------------------------------------------

def compile_card(card: dict, sidecar: dict, tool: str) -> ExecutionPlan:
    catalog_id = card.get("catalog_id", "unknown")
    modality = card.get("modality", "text")

    # 1. Apply sidecar overrides deterministically (card values are the base).
    override = _find_card_override(sidecar, catalog_id)
    merged = _deep_merge(card, override)

    dataset = merged.get("dataset", {}) or {}
    load = merged.get("load", {}) or {}
    slo = merged.get("slo", {}) or {}
    dtype = dataset.get("type")

    # 2. Inject sidecar-supplied trace path into dataset (trace-replay/real).
    trace_cfg = override.get("trace") or (sidecar.get("workload_overrides", {}) or {}).get("trace")
    if trace_cfg and dtype in ("trace-replay", "real"):
        dataset = {**dataset, **trace_cfg}

    if dtype not in DATASET_HANDLERS:
        raise UnsupportedWorkload(
            f"[{catalog_id}] dataset.type {dtype!r} has no registered handler. "
            f"Add one to registry.DATASET_HANDLERS or fix the card — the runner "
            f"will not guess.")

    # 3. Resolve dataset.
    ds: DatasetResolution = DATASET_HANDLERS[dtype](dataset, tool, modality)

    plan = ExecutionPlan(catalog_id=catalog_id, tool=tool, kind=ds.kind,
                         modality=modality, dataset_summary=ds.summary,
                         slo_targets=slo)

    # 4a. Dataset itself demands orchestration -> done (no vendor steps).
    if ds.kind == "orchestrated":
        plan.orchestrator = ds.orchestrator
        plan.reason = ds.reason
        return plan

    # 4b. Resolve load. quantization-pareto uses `modes:` instead of `type:`.
    if "modes" in load and "type" not in load:
        plan.steps = _expand_modes(load, ds, slo, tool)
        return plan

    ltype = load.get("type")
    if ltype not in LOAD_EXPANDERS:
        raise UnsupportedWorkload(
            f"[{catalog_id}] load.type {ltype!r} has no registered expander. "
            f"Add one to registry.LOAD_EXPANDERS or fix the card.")

    # An orchestrated load.type overrides a vendor dataset (e.g. burn-in mixes,
    # cold-start probe). Surface as orchestrated.
    try:
        load_steps: list[LoadStep] = LOAD_EXPANDERS[ltype](load)
    except _OrchestratedLoad as ol:
        plan.kind = "orchestrated"
        plan.orchestrator = ol.executor
        plan.reason = ol.reason
        return plan

    # 5. context-length / clip-length outer loop (concurrency-sweep, video).
    context_lens = _context_axis(dataset)

    goodput = _goodput_argv(slo)
    steps: list[PlanStep] = []
    for ctx in context_lens:
        ds_argv = _ds_argv_for_context(ds, dataset, tool, ctx)
        for ls in load_steps:
            label = ls.label if ctx is None else f"ctx{ctx}_{ls.label}"
            steps.append(PlanStep(
                label=label,
                argv=[*ds_argv, *ls.argv, *goodput],
                concurrency=ls.concurrency, request_rate=ls.request_rate,
                num_prompts=ls.num_prompts, duration_s=ls.duration_s,
                context_len=ctx))
    plan.steps = steps
    return plan


def _context_axis(dataset: dict) -> list[Optional[int]]:
    """Outer sweep axis for cards that vary context length per run."""
    cl = dataset.get("context_lengths")
    if isinstance(cl, list) and len(cl) > 1:
        return list(cl)
    # rag-1m uses context_tiers (list of dicts with prefix_tokens)
    tiers = dataset.get("context_tiers")
    if isinstance(tiers, list) and tiers:
        return [t.get("prefix_tokens") for t in tiers]
    return [None]


def _ds_argv_for_context(ds: DatasetResolution, dataset: dict, tool: str,
                         ctx: Optional[int]) -> list[str]:
    """Re-stamp the swept dimension when an outer context loop is active.

    For random datasets the axis is total input length; for shared-prefix
    datasets (rag-1m-context's context_tiers) the axis is the PREFIX length —
    that is the whole point of the tier sweep, so we must restamp the right flag
    or every tier would be identical.
    """
    if ctx is None:
        return list(ds.argv)
    argv = list(ds.argv)
    # Restamp whichever length flag the resolved dataset actually uses.
    for flag in ("--random-input-len", "--prefix-repetition-prefix-len",
                 "--gsp-system-prompt-len"):
        if flag in argv:
            argv[argv.index(flag) + 1] = str(int(ctx))
            break
    return argv


def _expand_modes(load: dict, ds: DatasetResolution, slo: dict, tool: str) -> list[PlanStep]:
    """quantization-pareto: offline (max-rate) + server (qps + latency SLO)."""
    goodput = _goodput_argv(slo)
    steps = []
    for m in load["modes"]:
        mid = m.get("id", "mode")
        if m.get("type") == "max-rate":
            argv = [*ds.argv, "--request-rate", "inf", "--num-prompts", "500", *goodput]
            steps.append(PlanStep(label=mid, argv=argv, request_rate=None,
                                  num_prompts=500, duration_s=m.get("duration_s"),
                                  mode=mid))
        elif m.get("type") == "qps-constrained":
            rate = m.get("target_qps", 4)
            dur = m.get("duration_s", 900)
            np = int(rate * dur)
            argv = [*ds.argv, "--request-rate", str(rate), "--num-prompts", str(np), *goodput]
            steps.append(PlanStep(label=mid, argv=argv, request_rate=rate,
                                  num_prompts=np, duration_s=dur, mode=mid))
        else:
            raise UnsupportedWorkload(
                f"quantization-pareto mode {m.get('type')!r} not supported")
    return steps
