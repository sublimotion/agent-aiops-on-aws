"""
Trinity worker pool — the 7-worker all-Bedrock fleet that replaces upstream's
mixed OpenAI/Anthropic/Gemini-direct + locally-vLLM-served open models.

Single source of truth for:
  POOL              — ordered list[WorkerConfig] (ord 0..6), role-class spread preserved
  LLM_NAMES         — the friendly names Trinity's CMA-ES head indexes (es_log order)
  by_friendly_name  — friendly llm_name -> WorkerConfig
  AGENT_CONFIGS     — dict consumed by fugu.utils.set_worker_agent_configs()

The CMA-ES router head outputs L (=len(POOL)) worker logits + 3 role logits. The
worker ordinal MUST stay stable across train/eval and match the head's output dim,
so POOL order is the contract — do not reorder without retraining.

Verified live against the us-east-1 catalog 2026-06-24 (see lessons.md Gate 0.0):
  - Nova Premier is provider-marked Legacy / access-denied via Converse on-demand
    -> ord 2 uses us.amazon.nova-pro-v1:0 (Amazon, distinct-provider slot preserved).
  - DeepSeek-R1 on-demand rejects the bare id -> requires the us. inference profile.
  - Anthropic frontier (Opus 4.8 / Sonnet 4.6) require the us. cross-region profile.
  - qwen.qwen3-32b-v1:0 serves BOTH ord 5 (reasoning) and ord 6 (direct): reasoning
    mode is selected via additionalModelRequestFields={"reasoning_effort": "..."},
    which emits a reasoningContent block; default mode emits text only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Regions used for throttle-driven round-robin (see bedrock_clients.py). us-east-1
# first for best model availability; the others add TPM headroom under the
# 512-candidate/iter CMA-ES burst.
REGIONS = ["us-east-1", "us-west-2", "us-east-2"]


@dataclass(frozen=True)
class WorkerConfig:
    ord: int
    friendly_name: str          # the name CMA-ES indexes (matches es_log llm_names order)
    role_class: str             # closed-frontier | open-mid | open-reasoning | open-direct
    model_id: str               # Bedrock modelId (inference-profile-prefixed where required)
    transport: str = "converse"  # "converse" (SigV4/IRSA) | "openai_compat" (bearer token, GPT-5.5 only)
    reasoning: bool = False     # enable thinking/reasoning emission
    # Per-worker concurrency cap (per-region). Frontier models have tighter TPM, so
    # start them lower; raise per the throttle-rate telemetry (spec §throttle handling).
    concurrency: int = 10
    api_quirks: tuple = ()      # e.g. ("no-temperature",) for some Anthropic frontier ids
    # Home region — _query_converse tries this FIRST, then rotates REGIONS for TPM
    # headroom. Critical: qwen/deepseek open flagships are us-west-2-only; without a
    # home-region hint every call wastes attempt-0 failing in us-east-1 (lessons #57).
    region: str = "us-east-1"
    notes: str = ""


# ---------------------------------------------------------------------------
# POOL v2 — DIFFERENTIATED-8 (2026-06-26). ord order IS the head's worker-logit
# order; this is a FRESH pool (head re-widened 7→8, no warm-start from the old
# 7-ckpt). Composition chosen from the differentiation probe (results/
# diff_probe_v2.json, n=30 LiveCodeBench): keep the high-capability open
# flagships + a frontier anchor + a cheap tier; drop redundant/weak (sonnet≈haiku,
# nova 0.43, deepseek-r1 priciest+866s/run). Pass@1 / $-per-prob from the probe.
# ---------------------------------------------------------------------------
POOL: list[WorkerConfig] = [
    WorkerConfig(
        0, "deepseek-v3", "open-flagship",
        "deepseek.v3-v1:0", region="us-west-2", concurrency=8,
        notes="Probe BEST: pass 0.87 @ $0.0022/prob. Open flagship, us-west-2 only. "
              "Returns prose+```python (not <answer> tags) — extraction must isolate the fence.",
    ),
    WorkerConfig(
        1, "qwen3-235b", "open-flagship",
        "qwen.qwen3-235b-a22b-2507-v1:0", region="us-west-2", concurrency=10,
        notes="Probe 0.77 @ $0.0009. Open flagship, us-west-2 only.",
    ),
    WorkerConfig(
        2, "qwen3-coder-480b", "open-flagship",
        "qwen.qwen3-coder-480b-a35b-v1:0", region="us-west-2", concurrency=8,
        notes="Probe 0.73 @ $0.0009. Coding-specialized 480B MoE, open, us-west-2 only.",
    ),
    WorkerConfig(
        3, "gpt-oss-120b", "open-flagship",
        "openai.gpt-oss-120b-1:0", region="us-east-1", concurrency=10,
        notes="Probe 0.73 @ $0.00065 — cheapest of the strong tier. OpenAI open-weight.",
    ),
    WorkerConfig(
        4, "claude-opus-4-8", "closed-frontier",
        "us.anthropic.claude-opus-4-8", region="us-east-1", concurrency=6,
        api_quirks=("no-temperature",),
        notes="Frontier anchor / escalation target. Probe 0.73 @ $0.0106 (10-50× the "
              "open flagships). Rejects `temperature` → no-temperature quirk.",
    ),
    WorkerConfig(
        5, "claude-haiku-4-5", "closed-cheap-frontier",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0", region="us-east-1", concurrency=10,
        notes="Probe 0.67 @ $0.0041. Anthropic cheap tier — a closed-model mid option.",
    ),
    WorkerConfig(
        6, "qwen3-32b-direct", "open-cheap",
        "qwen.qwen3-32b-v1:0", region="us-east-1", reasoning=False, concurrency=10,
        notes="Probe 0.67 @ $0.0002 — CHEAPEST in the pool. Direct (no-thinking) mode.",
    ),
    WorkerConfig(
        7, "gemma-3-27b", "open-cheap",
        "google.gemma-3-27b-it", region="us-east-1", concurrency=10,
        notes="Probe 0.47 @ $0.0010. Weak cheap floor — gives the router a low-cost "
              "low-capability option to trade against.",
    ),
]

# Friendly names in head/index order — feed straight to the Trinity Task llm_names.
LLM_NAMES: list[str] = [w.friendly_name for w in POOL]

by_friendly_name: dict[str, WorkerConfig] = {w.friendly_name: w for w in POOL}
by_ord: dict[int, WorkerConfig] = {w.ord: w for w in POOL}


def reasoning_effort_for(w: WorkerConfig) -> Optional[str]:
    """Reasoning knob per worker, or None.

    Live-verified 2026-06-24:
    - **Qwen3-32B** accepts OpenAI-style `reasoning_effort` (high|medium|low|
      minimal|none); the flag is what distinguishes ord-5 (reasoning) from
      ord-6 (direct). NOT the Anthropic reasoning_config schema.
    - **DeepSeek-R1 REJECTS `reasoning_effort`** (ValidationException) — it
      reasons by default with no flag, emitting both a reasoningContent and a
      text block. So we must NOT send the flag to it.
    Only emit the flag for models that accept it (currently: Qwen3 family).
    """
    if not w.reasoning:
        return None
    if "qwen" in w.model_id.lower():
        return "high"
    return None  # DeepSeek-R1 et al. reason natively; no flag (rejects it)


# AGENT_CONFIGS: fugu.utils._resolve_agent_complete_info reads this registry. We
# keep server/port absent so fugu routes through the (monkeypatched) Bedrock
# dispatch rather than the vLLM HTTP path. model_name carries the Bedrock id;
# payload carries the per-worker reasoning + transport knobs our dispatch reads.
AGENT_CONFIGS: dict[str, dict] = {
    w.friendly_name: {
        "model_name": w.friendly_name,   # keep friendly; dispatch maps -> model_id
        "payload": {
            "_bedrock_model_id": w.model_id,
            "_transport": w.transport,
            "_reasoning_effort": reasoning_effort_for(w),
            "_ord": w.ord,
        },
    }
    for w in POOL
}


if __name__ == "__main__":
    print(f"Trinity Bedrock pool: {len(POOL)} workers")
    for w in POOL:
        re = reasoning_effort_for(w)
        print(f"  ord {w.ord}: {w.friendly_name:24s} [{w.role_class:15s}] "
              f"{w.model_id:40s} transport={w.transport} reasoning_effort={re}")
    print(f"\nLLM_NAMES (head order): {LLM_NAMES}")
