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
    notes: str = ""


# ---------------------------------------------------------------------------
# POOL — ord order IS the head's worker-logit order. Mirrors es_log.json
# llm_names (gpt-5, claude-sonnet, gemini-pro, deepseek-r1, gemma, qwen-reason,
# qwen-direct) one-for-one on role class, substituting verified Bedrock ids.
# ---------------------------------------------------------------------------
POOL: list[WorkerConfig] = [
    WorkerConfig(
        0, "claude-opus-4-8", "closed-frontier",
        "us.anthropic.claude-opus-4-8",
        concurrency=6,
        api_quirks=("no-temperature",),  # Opus 4.8 deprecates `temperature` (verified live 2026-06-24)
        notes="Frontier anchor (replaces upstream GPT-5). IAM-invokable via Converse. "
              "Rejects `temperature` (reasoning model) → api_quirks no-temperature. "
              "GPT-5.5 is an optional ord-0 swap (transport=openai_compat) gated on the "
              "operator bearer token — see bedrock_clients.py GPT-5.5 path.",
    ),
    WorkerConfig(
        1, "claude-sonnet-4-6", "closed-frontier",
        "us.anthropic.claude-sonnet-4-6",
        concurrency=8,
        notes="Replaces upstream Claude-4-Sonnet.",
    ),
    WorkerConfig(
        2, "nova-pro", "closed-frontier",
        "us.amazon.nova-pro-v1:0",
        concurrency=10,
        notes="Distinct-provider frontier slot (replaces upstream Gemini-2.5-pro). "
              "Nova Premier was provider-marked Legacy/access-denied on-demand "
              "2026-06-24, so Nova Pro is the live Amazon substitute.",
    ),
    WorkerConfig(
        3, "gemma-3-27b", "open-mid",
        "google.gemma-3-27b-it",
        concurrency=10,
        notes="Mid open generalist (matches upstream Gemma-3-27B).",
    ),
    WorkerConfig(
        4, "deepseek-r1", "open-reasoning",
        "us.deepseek.r1-v1:0",
        reasoning=True, concurrency=8,
        notes="Reasoning open model (replaces upstream DeepSeek-R1-Distill-Qwen-32B). "
              "Bare id rejects on-demand; us. inference profile required.",
    ),
    WorkerConfig(
        5, "qwen3-32b-reasoning", "open-reasoning",
        "qwen.qwen3-32b-v1:0",
        reasoning=True, concurrency=10,
        notes="Qwen3-32B reasoning mode (additionalModelRequestFields.reasoning_effort).",
    ),
    WorkerConfig(
        6, "qwen3-32b-direct", "open-direct",
        "qwen.qwen3-32b-v1:0",
        reasoning=False, concurrency=10,
        notes="Same weights as ord 5, direct (no-thinking) mode. Distinct pool slot.",
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
