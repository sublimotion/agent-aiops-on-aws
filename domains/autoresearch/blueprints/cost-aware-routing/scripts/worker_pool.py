"""
Worker pool — single source of truth for the 9-worker Bedrock fleet.

Exposes:
  POOL                — list of WorkerConfig (ord 0..8, ranked by ascending $/query)
  compute_cost(in, out, ord) -> float ($/call)
  cost_normalized(cost) -> float in [0, 1]
  build_metadata_prompt() -> str (rich-metadata router prompt header)
  invoke_worker(client, ord, prompt, **kw) -> dict (handles Opus-4.7 quirk)

Pricing source: Anthropic published rates (Opus/Sonnet/Haiku) + AWS Bedrock
us-west-2 published rates for Bedrock-only models. Extrapolations noted in
plan-addendum-2026-05-27.md §1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

REGION = "us-west-2"


@dataclass(frozen=True)
class WorkerConfig:
    ord: int
    name: str
    model_id: str
    in_per_1M: float
    out_per_1M: float
    qualitative_strength: str
    api_quirks: tuple = ()  # e.g., ("no-temperature",)


# Costs at 200 input + 800 output tokens (realistic GRPO rollout shape).
POOL: list[WorkerConfig] = [
    WorkerConfig(0, "gemma-3-27b-it",
                 "google.gemma-3-27b-it",
                 0.23, 0.38,
                 "Mid-tier generalist; fastest cheap option for short factual or simple reasoning."),
    WorkerConfig(1, "gpt-oss-120b",
                 "openai.gpt-oss-120b-1:0",
                 0.15, 0.60,
                 "Cheap and fast; good at short arithmetic and pattern lookup; weak at multi-step reasoning."),
    WorkerConfig(2, "qwen3-32b",
                 "qwen.qwen3-32b-v1:0",
                 0.15, 0.62,
                 "Mid-tier with strong tool calling; good at structured outputs and short reasoning."),
    WorkerConfig(3, "qwen3-coder-480b",
                 "qwen.qwen3-coder-480b-a35b-v1:0",
                 0.50, 1.20,
                 "Code specialist, frontier-class; best for HumanEval / LCB / patch generation."),
    WorkerConfig(4, "mistral-large-3",
                 "mistral.mistral-large-3-675b-instruct",
                 0.50, 1.50,
                 "Strong generalist, multilingual; reliable on MMLU-style factual breadth."),
    WorkerConfig(5, "deepseek-v3.2",
                 "deepseek.v3.2",
                 0.62, 1.85,
                 "Strong reasoning at moderate cost; good MATH/AIME mid-tier choice."),
    WorkerConfig(6, "haiku-4-5",
                 "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                 1.00, 5.00,
                 "Fast Anthropic frontier; excellent at quick lookup and instructions; weaker at hard math."),
    WorkerConfig(7, "sonnet-4-6",
                 "us.anthropic.claude-sonnet-4-6",
                 3.00, 15.00,
                 "Strong frontier, balanced; reliable on hard reasoning and long context."),
    WorkerConfig(8, "opus-4-7",
                 "us.anthropic.claude-opus-4-7",
                 5.00, 25.00,
                 "Top-tier reasoning; best for multi-step proofs and frontier-hard problems; expensive.",
                 ("no-temperature",)),
]

# Per-rollout cost at the standard rollout shape (200 in / 800 out).
# Used for cost_normalized's min/max anchors.
ROLLOUT_IN_TOKENS = 200
ROLLOUT_OUT_TOKENS = 800


def per_call_cost_usd(in_tok: int, out_tok: int, ord_: int) -> float:
    """Actual $/call given measured token counts (preferred during training)."""
    w = POOL[ord_]
    return in_tok * w.in_per_1M / 1e6 + out_tok * w.out_per_1M / 1e6


def reference_cost_usd(ord_: int) -> float:
    """Cost at the standard rollout shape — used for normalization anchors."""
    return per_call_cost_usd(ROLLOUT_IN_TOKENS, ROLLOUT_OUT_TOKENS, ord_)


_MIN_REF = min(reference_cost_usd(i) for i in range(len(POOL)))
_MAX_REF = max(reference_cost_usd(i) for i in range(len(POOL)))


def cost_normalized(cost_usd: float) -> float:
    """Map an actual $/call to [0, 1] using the reference (200/800) anchors.

    Clamps below 0 and above 1 (an unusually long Opus call could exceed
    the reference max; clamp prevents `-α · cost > -α`).
    """
    z = (cost_usd - _MIN_REF) / (_MAX_REF - _MIN_REF)
    return max(0.0, min(1.0, z))


def build_metadata_prompt() -> str:
    """Header injected at the top of every router rollout prompt.

    Lists each worker with ord, name, $/1M in/out (at 200/800: $/query), and
    a one-line qualitative strength. Designed to break ordinal-lock by giving
    the router enough metadata to pattern-match capability → ord.
    """
    lines = [
        "You are a routing controller. Pick exactly ONE worker (by ord) to answer the user's question.",
        "Each worker has a distinct cost-quality profile. Pick the cheapest worker likely to answer correctly.",
        "",
        "Available workers:",
    ]
    for w in POOL:
        ref_q = reference_cost_usd(w.ord)
        lines.append(
            f"  ord_{w.ord}: {w.name}  (${w.in_per_1M:.2f}/${w.out_per_1M:.2f} per 1M tok in/out → "
            f"${ref_q:.5f}/query at typical CoT length)\n"
            f"           {w.qualitative_strength}"
        )
    lines.append("")
    lines.append("Respond with EXACTLY one line: PICK ord_N (where N is 0..8). Optionally add one short sentence of justification.")
    return "\n".join(lines)


def invoke_worker(
    client,
    ord_: int,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    system: Optional[str] = None,
) -> dict:
    """Invoke a Bedrock worker via converse(); handles per-model quirks.

    Returns dict with: text, input_tokens, output_tokens, elapsed_s, error.
    """
    w = POOL[ord_]
    cfg: dict = {"maxTokens": max_tokens}
    if "no-temperature" not in w.api_quirks:
        cfg["temperature"] = temperature

    kwargs = {
        "modelId": w.model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": cfg,
    }
    if system:
        kwargs["system"] = [{"text": system}]

    t0 = time.time()
    try:
        resp = client.converse(**kwargs)
        dur = time.time() - t0
        out = resp["output"]["message"]["content"]
        text = out[0].get("text", "") if out else ""
        usage = resp.get("usage", {})
        return {
            "ord": ord_,
            "name": w.name,
            "text": text,
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "elapsed_s": round(dur, 3),
            "error": None,
        }
    except Exception as e:
        return {
            "ord": ord_,
            "name": w.name,
            "text": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_s": round(time.time() - t0, 3),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


if __name__ == "__main__":
    print(build_metadata_prompt())
    print()
    print(f"Reference cost range: ${_MIN_REF:.5f} (ord_0) → ${_MAX_REF:.5f} (ord_8)")
    print(f"Spread: {_MAX_REF / _MIN_REF:.1f}×")
