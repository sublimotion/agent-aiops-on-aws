"""
Verified Bedrock pricing — replaces fugu/cost.py's hardcoded GPT/DeepSeek/
Anthropic/Gemini/OpenSource tiers with per-worker $/1M-token rates keyed on the
Trinity Bedrock pool.

Cost feeds the CMA-ES cost_bonus_weight reward term (spec OQ3 — upstream ran
cost_bonus_weight=0.0; the cost-aware sweep is our extension). Even at 0.0 we
track spend so run_trinity_agent.py can enforce the hard cost cap.

Pricing policy (mirrors cost-aware-routing): snapshot published Bedrock rates at
run-start. The Pricing API omits Anthropic 4.5+ and several newest ids (the
carryover lesson), so the table below is the authoritative fallback, version-
stamped here and overridable from a run-start snapshot JSON via load_snapshot().
Rates are USD per 1M tokens, us-east-1, on-demand, 2026-06-24.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Dict, Optional

from worker_pool_bedrock import POOL, by_friendly_name


@dataclass(frozen=True)
class Price:
    in_per_1m: float
    out_per_1m: float


# Authoritative fallback table (USD / 1M tokens). Keyed by friendly_name so it
# tracks the pool ordinals, not raw model ids (ord 5/6 share an id but bill the
# same per-token rate; they differ only in reasoning-token volume).
PRICES: Dict[str, Price] = {
    "claude-opus-4-8":       Price(5.00, 25.00),   # Anthropic frontier
    "claude-sonnet-4-6":     Price(3.00, 15.00),
    "nova-pro":              Price(0.80, 3.20),    # Amazon Nova Pro
    "gemma-3-27b":           Price(0.23, 0.38),    # matches cost-aware-routing snapshot
    "deepseek-r1":           Price(1.35, 5.40),    # DeepSeek-R1 (reasoning; high output volume)
    "qwen3-32b-reasoning":   Price(0.15, 0.62),
    "qwen3-32b-direct":      Price(0.15, 0.62),
    # GPT-5.5 (optional ord-0 swap) bills to the bearer-token account, tracked
    # separately. Placeholder until the operator confirms the grant's rate.
    "gpt-5.5":               Price(1.25, 10.00),
}

# Approx chars-per-token for the lightweight estimator used when Bedrock usage
# metadata is unavailable. ~4 chars/token is the standard rough proxy.
_CHARS_PER_TOKEN = 4.0

_VERSION = "2026-06-24"


def load_snapshot(path: str) -> None:
    """Overlay a run-start price snapshot (JSON: {friendly_name: [in,out]})."""
    with open(path) as f:
        snap = json.load(f)
    for name, (i, o) in snap.items():
        PRICES[name] = Price(float(i), float(o))


# ---------------------------------------------------------------------------
# Cost tracking, drop-in for fugu.cost (track_cost / get_cost_summary / reset).
# fugu.utils calls fugu.cost.track_cost(model, messages, response); we rebind it.
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_TOTALS: Dict[str, dict] = {}


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _messages_text(messages) -> str:
    parts = []
    for m in messages:
        c = m.get("content", "")
        parts.append(c if isinstance(c, str) else json.dumps(c))
    return "\n".join(parts)


def track_cost(model: str, messages, response: str) -> dict:
    price = PRICES.get(model)
    if price is None:
        w = by_friendly_name.get(model)
        price = PRICES.get(w.friendly_name) if w else None
    in_tok = _estimate_tokens(_messages_text(messages))
    out_tok = _estimate_tokens(response)
    if price is not None:
        cost = in_tok * price.in_per_1m / 1e6 + out_tok * price.out_per_1m / 1e6
    else:
        cost = 0.0
    with _LOCK:
        d = _TOTALS.setdefault(model, {"queries": 0, "in": 0, "out": 0, "cost": 0.0})
        d["queries"] += 1
        d["in"] += in_tok
        d["out"] += out_tok
        d["cost"] += cost
        total = sum(v["cost"] for v in _TOTALS.values())
    return {"model": model, "input_tokens": in_tok, "output_tokens": out_tok,
            "cost": cost, "total_cost": total}


def track_cost_from_usage(model: str, input_tokens: int, output_tokens: int) -> dict:
    """Preferred path when Bedrock returns real usage metadata."""
    price = PRICES.get(model) or Price(0.0, 0.0)
    cost = input_tokens * price.in_per_1m / 1e6 + output_tokens * price.out_per_1m / 1e6
    with _LOCK:
        d = _TOTALS.setdefault(model, {"queries": 0, "in": 0, "out": 0, "cost": 0.0})
        d["queries"] += 1
        d["in"] += input_tokens
        d["out"] += output_tokens
        d["cost"] += cost
        total = sum(v["cost"] for v in _TOTALS.values())
    return {"model": model, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "cost": cost, "total_cost": total}


def get_cost_summary() -> dict:
    with _LOCK:
        models = [{
            "model": m, "queries": d["queries"],
            "total_input_tokens": d["in"], "total_output_tokens": d["out"],
            "total_tokens": d["in"] + d["out"], "total_cost": d["cost"],
        } for m, d in _TOTALS.items()]
        total = sum(d["cost"] for d in _TOTALS.values())
    return {"models": models, "total_cost": total, "pricing_version": _VERSION}


def reset_costs() -> None:
    with _LOCK:
        _TOTALS.clear()


def total_spend() -> float:
    with _LOCK:
        return sum(d["cost"] for d in _TOTALS.values())


def install() -> None:
    """Rebind fugu.cost's tracking functions to the Bedrock-priced versions."""
    import fugu.cost as C
    import fugu.utils as U
    C.track_cost = track_cost
    C.get_cost_summary = get_cost_summary
    C.reset_costs = reset_costs
    # fugu.utils imported these by name at module load; rebind those refs too.
    U.track_cost = track_cost
    U.get_cost_summary = get_cost_summary


if __name__ == "__main__":
    print(f"Bedrock pricing snapshot {_VERSION} (USD / 1M tokens):")
    for w in POOL:
        p = PRICES[w.friendly_name]
        print(f"  ord {w.ord} {w.friendly_name:24s} in={p.in_per_1m:6.2f} out={p.out_per_1m:6.2f}")
