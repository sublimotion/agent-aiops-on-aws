#!/usr/bin/env python3
"""orchestrators.py — bespoke executors for cards the vendor bench tools can't run.

Each function here is referenced by name from registry.py and dispatched by the
platform driver when a card compiles to an ORCHESTRATED plan. Signature:

    def <executor>(plan, endpoint: str, model: str, output: Path, sidecar: dict) -> None

These executors require live GPU infrastructure and bespoke request logic, so
they are implemented incrementally. Until an executor is implemented it raises
NotImplementedError with the registered reason — a deterministic, honest outcome
(NOT a silent fallback to a wrong vendor run). The card→executor *mapping* is
already deterministic and asserted by tests/test_card_conformance.py.

Implementation order is driven by demand; see runner/CONTRIBUTING.md for the
recipe. When you implement one, also add a behavioural test under tests/.
"""
from __future__ import annotations

from pathlib import Path


def _not_yet(name: str, plan) -> None:
    raise NotImplementedError(
        f"orchestrated executor '{name}' is registered but not yet implemented.\n"
        f"Card: {plan.catalog_id}\n"
        f"Reason it needs orchestration: {plan.reason}\n"
        f"Implement it in runner/orchestrators.py per runner/CONTRIBUTING.md, or "
        f"run a vendor-executable card instead. The runner will NOT substitute a "
        f"vendor run — that would produce a misleading result.")


def agentic_session_runner(plan, endpoint, model, output: Path, sidecar: dict) -> None:
    """coding-agent: stateful multi-turn loop, shared system prompt, inter-turn
    tool-execution delays. Must hold conversation KV across turns and measure
    cold vs warm TTFT per turn."""
    _not_yet("agentic_session_runner", plan)


def multiturn_session_runner(plan, endpoint, model, output: Path, sidecar: dict) -> None:
    """multi-turn-chat: per-session growing history, ttft_by_turn_index."""
    _not_yet("multiturn_session_runner", plan)


def cohost_runner(plan, endpoint, model, output: Path, sidecar: dict) -> None:
    """cohost-isolation / mig-partitioning: N concurrent tenants, rotating
    noisy-neighbour role, per-tenant isolation score."""
    _not_yet("cohost_runner", plan)


def burn_in_runner(plan, endpoint, model, output: Path, sidecar: dict) -> None:
    """burn-in: sustained soak sliced into windows; each window emits a raw JSON
    consolidated by container/analyze-burn-in.py into a drift curve."""
    _not_yet("burn_in_runner", plan)


def cold_start_probe(plan, endpoint, model, output: Path, sidecar: dict) -> None:
    """cold-start: phase-timed load→first-token (weights, compile, CUDA graph
    capture) + first-100 latency curve."""
    _not_yet("cold_start_probe", plan)


def power_efficiency_runner(plan, endpoint, model, output: Path, sidecar: dict) -> None:
    """power-efficiency: tokens/joule at 25/50/75/100% of measured ceiling, plus
    ECC/SDC sentinel canary. Needs the ceiling first."""
    _not_yet("power_efficiency_runner", plan)


def audio_runner(plan, endpoint, model, output: Path, sidecar: dict) -> None:
    """transcription-sweep: clip uploads to /v1/audio/transcriptions, RTFx as the
    primary metric instead of tok/s."""
    _not_yet("audio_runner", plan)


def video_runner(plan, endpoint, model, output: Path, sidecar: dict) -> None:
    """video-summary: multimodal video_url payloads, per-frame encoder prefill,
    clip-length × concurrency grid."""
    _not_yet("video_runner", plan)


def fin_replay_runner(plan, endpoint, model, output: Path, sidecar: dict) -> None:
    """fin-support: replay a REAL prompt corpus (seed) + profile-matched
    synthetic augmentation when num_prompts exceeds the corpus.

    Per-request construction (why vendor bench can't do this):
      - Load the seed corpus path from sidecar workload_overrides.trace.path.
      - For requests within corpus_size: replay distinct real prompts.
      - For requests beyond corpus_size: synthesise a NEW prompt that keeps the
        ~3,050-token shared system header VERBATIM (so the prefix cache sees the
        real cacheable region) but generates a UNIQUE body — recombined
        guideline blocks + synthetic retrieved passages + a varied user query —
        sized to the measured ISL distribution (p50 8823 / p90 11952).
      - NEVER resend an identical prompt: identical replays warm the cache
        artificially and produce a fake ~100% hit rate.

    Cache-realism guard (the whole reason this is orchestrated):
      - Tag each request real|synthetic; emit real_vs_synthetic_split.
      - Compare measured prefix_hit_rate against the corpus reuse ceiling
        (~0.30 of blocks). If it materially exceeds the ceiling, FAIL the run
        as cache-inflated rather than reporting a misleading SLO pass.
      - Emit isl_ks_distance(real, synthetic); flag if synthetic ISL drifts
        from the real distribution.
    """
    _not_yet("fin_replay_runner", plan)
