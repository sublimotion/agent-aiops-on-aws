#!/usr/bin/env python3
"""registry.py — the deterministic card->command contract.

This module is the single place that knows how to turn a workload card's
declared `(dataset.type, load.type)` into concrete benchmark execution. It
exists to remove all interpretation from the runner: there is exactly ONE
registered resolution per pair found in workloads/*.yaml, and anything not
registered raises `UnsupportedWorkload`. Nothing here ever falls back to a
degenerate default (the old failure mode that silently produced misleading
results — see SKILL.md, the Kimi K2.6 TTFT-loss incident).

Two registries:
  DATASET_HANDLERS  keyed by dataset.type  -> emits vendor dataset argv,
                                              OR declares an orchestrated executor.
  LOAD_EXPANDERS    keyed by load.type     -> expands into N concrete load steps.

A card is VENDOR-executable only if BOTH its dataset.type and load.type are
vendor-mappable; otherwise it is ORCHESTRATED and routed to a named executor
in orchestrators.py. Orchestrated executors that need live GPU infra to be
written correctly raise NotImplementedError at run time with the reason —
the *mapping* is still deterministic and asserted by the conformance test.

Vendor flag references:
  vLLM   `vllm bench serve` / `vllm.entrypoints.openai.bench_serving`
  SGLang `python3 -m sglang.bench_serving`
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# --------------------------------------------------------------------------
# Errors — both fail CLOSED. The whole point of the module.
# --------------------------------------------------------------------------

class UnsupportedWorkload(Exception):
    """Raised when a (dataset.type, load.type, tool) triple has no registered
    resolution. This is a configuration error, not a runtime fallback."""


# --------------------------------------------------------------------------
# Data structures (plain dataclasses — JSON/dry-run friendly, no side effects)
# --------------------------------------------------------------------------

@dataclass
class DatasetResolution:
    """Result of resolving dataset.type for a given tool.

    Exactly one of (argv, orchestrator) is set:
      - argv:         vendor dataset flags, e.g. ['--dataset-name','random',...]
      - orchestrator: name of a bespoke executor in orchestrators.py
    """
    kind: str  # "vendor" | "orchestrated"
    argv: list[str] = field(default_factory=list)
    orchestrator: Optional[str] = None
    reason: Optional[str] = None          # why orchestrated (for humans + envelope)
    # dataset summary that lands in the v1 envelope workload.dataset block
    summary: dict = field(default_factory=dict)


@dataclass
class LoadStep:
    """One concrete execution point produced by expanding a load.type."""
    label: str                  # filename suffix, e.g. "c64", "qps2.0", "offline"
    argv: list[str]             # load flags for this step
    concurrency: Optional[int] = None
    request_rate: Optional[float] = None   # None => unbounded (max rate)
    num_prompts: Optional[int] = None
    duration_s: Optional[int] = None
    extra: dict = field(default_factory=dict)


# Handler signatures
DatasetHandler = Callable[[dict, str, str], DatasetResolution]  # (dataset, tool, modality)
LoadExpander = Callable[[dict], list[LoadStep]]


# ==========================================================================
# DATASET HANDLERS
# ==========================================================================

def _mean(v, default=None):
    """Cards express token counts as either an int or {mean, std_dev}."""
    if isinstance(v, dict):
        return v.get("mean", default)
    return v if v is not None else default


def _ds_synthetic(dataset: dict, tool: str, modality: str) -> DatasetResolution:
    """Fixed-length random prompts. The bread-and-butter synthetic workload.

    `synthetic` is also used by video-summary (modality=video) — that is NOT a
    text random workload; route it to the video orchestrator instead of emitting
    text flags that the server would reject.
    """
    if modality == "video":
        return DatasetResolution(
            kind="orchestrated", orchestrator="video_runner",
            reason="video modality: multimodal video_url payloads + per-frame "
                   "encoder prefill; vendor text bench cannot build these requests",
            summary={"type": "synthetic", "modality": "video"})
    if modality == "audio":
        return DatasetResolution(
            kind="orchestrated", orchestrator="audio_runner",
            reason="audio modality: clip uploads to /v1/audio/transcriptions",
            summary={"type": "synthetic", "modality": "audio"})

    # concurrency-sweep cards carry context_lengths instead of input_tokens.
    if "context_lengths" in dataset:
        ctxs = dataset["context_lengths"]
        in_len = ctxs[0] if isinstance(ctxs, list) and ctxs else 2048
    else:
        in_len = _mean(dataset.get("input_tokens"), 2048)
    out_len = _mean(dataset.get("output_tokens"), 256)

    if tool == "vllm":
        argv = ["--dataset-name", "random",
                "--random-input-len", str(int(in_len)),
                "--random-output-len", str(int(out_len))]
    elif tool == "sglang":
        argv = ["--dataset-name", "random",
                "--random-input-len", str(int(in_len)),
                "--random-output-len", str(int(out_len))]
    else:
        raise UnsupportedWorkload(f"synthetic dataset: unknown tool {tool!r}")

    return DatasetResolution(kind="vendor", argv=argv,
                             summary={"type": "synthetic",
                                      "input_tokens": {"mean": int(in_len)},
                                      "output_tokens": {"mean": int(out_len)}})


def _ds_shared_prefix(dataset: dict, tool: str, modality: str) -> DatasetResolution:
    """Shared-prefix workloads: large cacheable prefix + unique suffix.

    Covers card dataset types generated-shared-prefix, synthetic-shared-prefix.
    Maps to vLLM `prefix_repetition` and SGLang `generated-shared-prefix` (gsp).
    This is the workload the old runner silently dropped — it emitted NO dataset
    flags, so prefix caching was never actually exercised.
    """
    prefix = (dataset.get("shared_prefix_tokens")
              or dataset.get("shared_system_prompt_tokens") or 4096)
    suffix = _mean(dataset.get("unique_suffix_tokens")
                   or dataset.get("per_user_query_tokens"), 256)
    out_len = _mean(dataset.get("output_tokens"), 256)
    groups = dataset.get("prefix_groups") or dataset.get("num_prefixes") or 4

    if tool == "vllm":
        argv = ["--dataset-name", "prefix_repetition",
                "--prefix-repetition-prefix-len", str(int(prefix)),
                "--prefix-repetition-suffix-len", str(int(suffix)),
                "--prefix-repetition-num-prefixes", str(int(groups)),
                "--prefix-repetition-output-len", str(int(out_len))]
    elif tool == "sglang":
        # SGLang gsp dataset. Flag names per sglang.bench_serving --dataset-name
        # generated-shared-prefix. VERIFY against the pinned SGLang version on
        # first real run (dry-run prints the argv).
        argv = ["--dataset-name", "generated-shared-prefix",
                "--gsp-num-groups", str(int(groups)),
                "--gsp-system-prompt-len", str(int(prefix)),
                "--gsp-question-len", str(int(suffix)),
                "--gsp-output-len", str(int(out_len))]
    else:
        raise UnsupportedWorkload(f"shared-prefix dataset: unknown tool {tool!r}")

    return DatasetResolution(kind="vendor", argv=argv,
                             summary={"type": "generated-shared-prefix",
                                      "shared_prefix_tokens": int(prefix),
                                      "unique_suffix_tokens": {"mean": int(suffix)},
                                      "prefix_groups": int(groups),
                                      "output_tokens": {"mean": int(out_len)}})


def _ds_rag(dataset: dict, tool: str, modality: str) -> DatasetResolution:
    """synthetic-rag: per-query retrieved context, NO shared prefix
    (prefix_reuse: false). Each request is a distinct long-ish input, so this
    maps to a random dataset sized at system_prompt + retrieved_context + question.
    """
    sys_p = dataset.get("system_prompt_tokens", 0) or 0
    ctx = _mean(dataset.get("retrieved_context_tokens"), 5000)
    q = _mean(dataset.get("user_question_tokens"), 128)
    in_len = int(sys_p + ctx + q)
    out_len = _mean(dataset.get("output_tokens"), 384)

    argv = ["--dataset-name", "random",
            "--random-input-len", str(in_len),
            "--random-output-len", str(int(out_len))]
    return DatasetResolution(kind="vendor", argv=argv,
                             summary={"type": "synthetic-rag",
                                      "input_tokens": {"mean": in_len},
                                      "output_tokens": {"mean": int(out_len)},
                                      "prefix_reuse": False})


def _ds_real_trace(dataset: dict, tool: str, modality: str) -> DatasetResolution:
    """real / trace-replay: replay recorded traffic (ShareGPT, LMSYS, custom).

    The sidecar normally supplies the trace path via workload_overrides; the
    compiler injects `dataset_path` into the dataset dict before calling us.
    Falls back to the card's declared `fallback` source (sharegpt) so the card
    is runnable out of the box, but a missing path for a `custom` source is an
    error, not a silent default.
    """
    src = dataset.get("source") or (dataset.get("fallback") or {}).get("source", "sharegpt")
    path = dataset.get("dataset_path") or dataset.get("path")

    if src == "sharegpt":
        if tool == "vllm":
            argv = ["--dataset-name", "sharegpt"]
            if path:
                argv += ["--dataset-path", path]
        elif tool == "sglang":
            argv = ["--dataset-name", "sharegpt"]
            if path:
                argv += ["--dataset-path", path]
        else:
            raise UnsupportedWorkload(f"sharegpt: unknown tool {tool!r}")
    elif src in ("lmsys-chat-1m", "custom", "hf"):
        if not path:
            raise UnsupportedWorkload(
                f"trace-replay source {src!r} requires a dataset path; supply "
                f"workload_overrides.trace.path in the sidecar")
        if tool != "vllm":
            raise UnsupportedWorkload(
                f"trace-replay source {src!r} only wired for vllm hf dataset")
        argv = ["--dataset-name", "hf", "--dataset-path", path]
    else:
        raise UnsupportedWorkload(f"trace-replay: unknown source {src!r}")

    return DatasetResolution(kind="vendor", argv=argv,
                             summary={"type": "trace-replay", "source": src,
                                      "dataset_path": path})


# ---- Orchestrated dataset types (need a bespoke executor) ----

def _orchestrated(name: str, reason: str, dtype: str) -> DatasetHandler:
    def handler(dataset: dict, tool: str, modality: str) -> DatasetResolution:
        return DatasetResolution(kind="orchestrated", orchestrator=name,
                                 reason=reason, summary={"type": dtype})
    return handler


DATASET_HANDLERS: dict[str, DatasetHandler] = {
    "synthetic":               _ds_synthetic,
    "generated-shared-prefix": _ds_shared_prefix,
    "synthetic-shared-prefix": _ds_shared_prefix,
    "synthetic-rag":           _ds_rag,
    "real":                    _ds_real_trace,
    "trace-replay":            _ds_real_trace,
    "synthetic-agentic": _orchestrated(
        "agentic_session_runner",
        "stateful multi-turn agent loop with inter-turn tool-execution delays "
        "and a shared cacheable system prompt; vendor bench tools issue only "
        "independent single-shot requests",
        "synthetic-agentic"),
    "synthetic-multiturn": _orchestrated(
        "multiturn_session_runner",
        "conversation history grows within a session (per-session prefix reuse); "
        "vendor bench tools cannot accumulate KV across turns",
        "synthetic-multiturn"),
    "ensemble-from-sidecar": _orchestrated(
        "cohost_runner",
        "multiple co-tenant models driven concurrently with a rotating "
        "noisy-neighbour role; requires N load generators, not one",
        "ensemble-from-sidecar"),
    "mix-from-sidecar": _orchestrated(
        "burn_in_runner",
        "long sustained soak sliced into fixed windows with drift analysis; "
        "traffic is a weighted mix of other cards",
        "mix-from-sidecar"),
    "single-probe": _orchestrated(
        "cold_start_probe",
        "phase-timed cold-start measurement (weights load, compile, CUDA graph "
        "capture, first-100 latency curve); not a steady-state load test",
        "single-probe"),
    "audio-clips": _orchestrated(
        "audio_runner",
        "audio transcription: clip uploads to /v1/audio/transcriptions, RTFx "
        "as the primary metric instead of tok/s",
        "audio-clips"),
    "real-seeded-synthetic": _orchestrated(
        "fin_replay_runner",
        "replays a real prompt corpus as the seed, then augments with "
        "profile-matched synthetic requests when volume exceeds the corpus: "
        "the shared system header is kept VERBATIM (cacheable) while each "
        "synthetic body is unique and ISL-matched. Vendor bench tools can only "
        "loop a dataset, producing fake ~100%% prefix-cache hits that invalidate "
        "the result — this needs bespoke per-request construction + a "
        "cache-realism guard",
        "real-seeded-synthetic"),
}


# ==========================================================================
# LOAD EXPANDERS
# ==========================================================================

def _load_argv(tool: str, *, request_rate=None, max_concurrency=None,
               num_prompts=None) -> list[str]:
    """Common load flags shared by both vendor tools."""
    argv: list[str] = []
    if request_rate is None:
        argv += ["--request-rate", "inf"]
    else:
        argv += ["--request-rate", str(request_rate)]
    if max_concurrency is not None:
        argv += ["--max-concurrency", str(int(max_concurrency))]
    if num_prompts is not None:
        argv += ["--num-prompts", str(int(num_prompts))]
    return argv


def _lx_constant(load: dict) -> list[LoadStep]:
    """Single steady-state step. request_rate OR concurrent_sessions."""
    rate = load.get("request_rate")
    conc = load.get("concurrent_sessions") or load.get("max_concurrency")
    np = (load.get("num_prompts") or load.get("num_sessions")
          or load.get("num_prompts_per_level") or 100)
    label = f"qps{rate}" if rate is not None else (f"c{conc}" if conc else "constant")
    return [LoadStep(label=label,
                     argv=_load_argv("_", request_rate=rate, max_concurrency=conc,
                                     num_prompts=np),
                     concurrency=conc, request_rate=rate, num_prompts=np,
                     duration_s=load.get("duration_s"))]


def _lx_open_loop(load: dict) -> list[LoadStep]:
    """Max-rate throughput (request_rate null => inf)."""
    np = load.get("num_prompts", 500)
    return [LoadStep(label="maxrate",
                     argv=_load_argv("_", request_rate=None, num_prompts=np),
                     request_rate=None, num_prompts=np)]


def _lx_poisson(load: dict) -> list[LoadStep]:
    rate = load.get("request_rate", 4.0)
    np = load.get("num_prompts", 500)
    burst = load.get("burstiness")
    argv = _load_argv("_", request_rate=rate, num_prompts=np)
    if burst is not None:
        argv += ["--burstiness", str(burst)]
    return [LoadStep(label=f"qps{rate}", argv=argv, request_rate=rate, num_prompts=np)]


def _lx_qps_constrained(load: dict) -> list[LoadStep]:
    rate = load.get("target_qps", 2)
    dur = load.get("duration_s", 900)
    # num_prompts derived from rate*duration so the step is deterministic.
    np = load.get("num_prompts") or int(rate * dur)
    return [LoadStep(label=f"qps{rate}",
                     argv=_load_argv("_", request_rate=rate, num_prompts=np),
                     request_rate=rate, num_prompts=np, duration_s=dur)]


def _lx_sweep(load: dict) -> list[LoadStep]:
    """QPS sweep -> one step per rate."""
    rates = load.get("rates")
    if not rates:
        raise UnsupportedWorkload("load.type sweep requires a `rates` list")
    np = load.get("num_prompts_per_rate", 100)
    return [LoadStep(label=f"qps{r}",
                     argv=_load_argv("_", request_rate=r, num_prompts=np),
                     request_rate=r, num_prompts=np)
            for r in rates]


def _lx_concurrency_sweep(load: dict) -> list[LoadStep]:
    """Concurrency sweep -> one step per level. (Outer context_lengths loop is
    handled by the compiler, which calls this once per context.)"""
    levels = load.get("levels")
    if not levels:
        raise UnsupportedWorkload("load.type concurrency-sweep requires `levels`")
    np = load.get("num_prompts_per_level", 50)
    steps = []
    for c in levels:
        argv = _load_argv("_", request_rate=None, max_concurrency=c, num_prompts=np)
        steps.append(LoadStep(label=f"c{c}", argv=argv, concurrency=c,
                              request_rate=None, num_prompts=np,
                              duration_s=load.get("per_level_duration_s")))
    return steps


# ---- Orchestrated load types ----

class _OrchestratedLoad(Exception):
    """Sentinel: load.type itself demands orchestration regardless of dataset."""
    def __init__(self, executor: str, reason: str):
        self.executor = executor
        self.reason = reason


def _lx_orchestrated(executor: str, reason: str) -> LoadExpander:
    def expander(load: dict) -> list[LoadStep]:
        raise _OrchestratedLoad(executor, reason)
    return expander


LOAD_EXPANDERS: dict[str, LoadExpander] = {
    "constant":         _lx_constant,
    "open-loop":        _lx_open_loop,
    "poisson":          _lx_poisson,
    "qps-constrained":  _lx_qps_constrained,
    "sweep":            _lx_sweep,
    "concurrency-sweep": _lx_concurrency_sweep,
    "sustained": _lx_orchestrated(
        "burn_in_runner",
        "sustained soak at a fraction of measured ceiling, sliced into windows"),
    "ceiling-fraction-sweep": _lx_orchestrated(
        "power_efficiency_runner",
        "requires a previously-measured throughput ceiling to derive each load "
        "fraction; sidecar must supply the ceiling or it is measured first"),
    "ensemble-steady": _lx_orchestrated(
        "cohost_runner",
        "steady multi-tenant load with a rotating noisy-neighbour role"),
    "first-100": _lx_orchestrated(
        "cold_start_probe",
        "cold-start-inclusive first-100-inference latency curve"),
}
