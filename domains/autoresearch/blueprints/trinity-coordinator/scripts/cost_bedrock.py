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
import os
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
    "claude-haiku-4-5":      Price(1.00, 5.00),    # Anthropic cheap tier (differentiated-8)
    "nova-pro":              Price(0.80, 3.20),    # Amazon Nova Pro
    "gemma-3-27b":           Price(0.23, 0.38),    # matches cost-aware-routing snapshot
    "deepseek-r1":           Price(1.35, 5.40),    # DeepSeek-R1 (reasoning; high output volume)
    "deepseek-v3":           Price(0.58, 1.68),    # DeepSeek-V3 open flagship (differentiated-8)
    "qwen3-32b-reasoning":   Price(0.15, 0.62),
    "qwen3-32b-direct":      Price(0.15, 0.62),
    "qwen3-235b":            Price(0.22, 0.88),    # Qwen3-235B open flagship (differentiated-8)
    "qwen3-coder-480b":      Price(0.45, 1.80),    # Qwen3-Coder-480B MoE (differentiated-8)
    "gpt-oss-120b":          Price(0.15, 0.60),    # OpenAI open-weight flagship (differentiated-8)
    # GPT-5.5 (optional ord-0 swap) bills to the bearer-token account, tracked
    # separately. Placeholder until the operator confirms the grant's rate.
    "gpt-5.5":               Price(1.25, 10.00),
}

# Approx chars-per-token for the lightweight estimator used when Bedrock usage
# metadata is unavailable. ~4 chars/token is the standard rough proxy.
_CHARS_PER_TOKEN = 4.0

# Per-model MEASURED average tokens/call, learned live from real Converse usage
# (populated by record_usage below). Used to price an episode from its routing
# trace with REAL token rates rather than a hardcoded guess. Falls back to a
# neutral default until enough calls are observed.
_TOKEN_OBS: Dict[str, dict] = {}   # model -> {"calls", "in", "out"}
_OBS_LOCK = threading.Lock()


def record_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    """Provider-agnostic token counter. Called at the single Converse seam with
    Bedrock's normalized usage (inputTokens/outputTokens — identical schema across
    Anthropic/Amazon/Qwen/DeepSeek/Gemma). Pricing is applied SEPARATELY (map
    tokens→$ via PRICES), so raw counts stay reconcilable if prices change.
    Flushes to the per-PID telemetry sink so the MAIN process (which prices
    episodes) sees tokens measured inside spawned workers (lessons #20)."""
    with _OBS_LOCK:
        d = _TOKEN_OBS.setdefault(model, {"calls": 0, "in": 0, "out": 0})
        d["calls"] += 1
        d["in"] += int(input_tokens or 0)
        d["out"] += int(output_tokens or 0)
    _flush_token_sink()


def _flush_token_sink() -> None:
    dpath = _telemetry_dir()
    if not dpath:
        return
    with _OBS_LOCK:
        payload = {m: dict(v) for m, v in _TOKEN_OBS.items()}
    tmp = os.path.join(dpath, f"tokens_{os.getpid()}.json.tmp")
    dst = os.path.join(dpath, f"tokens_{os.getpid()}.json")
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, dst)
    except Exception:
        pass


def _aggregate_token_obs() -> Dict[str, dict]:
    """Main-process view: sum per-PID token files + this interpreter's own."""
    with _OBS_LOCK:
        merged = {m: dict(v) for m, v in _TOKEN_OBS.items()}
    dpath = _telemetry_dir()
    if dpath:
        for fn in os.listdir(dpath):
            if fn.startswith("tokens_") and fn.endswith(".json") and fn != f"tokens_{os.getpid()}.json":
                try:
                    per = json.load(open(os.path.join(dpath, fn)))
                except Exception:
                    continue
                for m, v in per.items():
                    a = merged.setdefault(m, {"calls": 0, "in": 0, "out": 0})
                    a["calls"] += v.get("calls", 0); a["in"] += v.get("in", 0); a["out"] += v.get("out", 0)
    return merged


def _avg_tokens(model: str) -> tuple[float, float]:
    obs = _aggregate_token_obs()
    d = obs.get(model)
    if d and d["calls"]:
        return d["in"] / d["calls"], d["out"] / d["calls"]
    return 1100.0, 500.0   # neutral fallback before observations accumulate


def episode_cost_from_agent_ids(agent_ids, llm_names) -> float:
    """Price one episode from its routing trace (agent_ids across turns) using
    REAL measured per-model avg tokens × the verified price table. Tokens are
    measured (record_usage); only the $/Mtok mapping is from PRICES — so this is
    exact-up-to-token-averages and fully reconcilable, not a hardcoded proxy."""
    if not agent_ids:
        return 0.0
    total = 0.0
    for aid in agent_ids:
        if aid is None or aid < 0 or aid >= len(llm_names):
            continue
        name = llm_names[aid]
        price = PRICES.get(name) or Price(0.0, 0.0)
        in_tok, out_tok = _avg_tokens(name)
        total += in_tok * price.in_per_1m / 1e6 + out_tok * price.out_per_1m / 1e6
    return total

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
    _flush_cost_sink()
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
    _flush_cost_sink()
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


# ---------------------------------------------------------------------------
# Spawn-isolation bridge: _TOTALS lives per-interpreter, so the main process
# can't see spend accumulated inside spawned Pool workers (the full-run cost cap
# would be blind — lessons #20). Each worker flushes its cumulative _TOTALS to
# <dir>/cost_<pid>.json after every tracked call; the main process sums all PID
# files. Per-PID files are overwritten with monotonic cumulative totals, so
# last-write-wins is exact and needs no cross-process lock. Dir is passed via
# env (CAR_TRINITY_TELEMETRY_DIR), the same channel sitecustomize already uses.
# ---------------------------------------------------------------------------
def _telemetry_dir() -> Optional[str]:
    d = os.environ.get("CAR_TRINITY_TELEMETRY_DIR")
    if d:
        os.makedirs(d, exist_ok=True)
    return d


def _flush_cost_sink() -> None:
    d = _telemetry_dir()
    if not d:
        return
    with _LOCK:
        payload = {m: dict(v) for m, v in _TOTALS.items()}
    tmp = os.path.join(d, f"cost_{os.getpid()}.json.tmp")
    dst = os.path.join(d, f"cost_{os.getpid()}.json")
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, dst)   # atomic
    except Exception:
        pass


def aggregate_spend() -> float:
    """Main-process view of TOTAL spend across all worker PIDs + this process."""
    d = _telemetry_dir()
    total = total_spend()   # this interpreter's own (usually 0 in main)
    if not d:
        return total
    for fn in os.listdir(d):
        if fn.startswith("cost_") and fn.endswith(".json"):
            try:
                with open(os.path.join(d, fn)) as f:
                    per = json.load(f)
                total += sum(v.get("cost", 0.0) for v in per.values())
            except Exception:
                continue
    return total


def aggregate_cost_summary() -> dict:
    """Merge per-PID cost files into one summary (for final exfil)."""
    d = _telemetry_dir()
    merged: Dict[str, dict] = {}
    # seed with this process
    with _LOCK:
        for m, v in _TOTALS.items():
            merged[m] = dict(v)
    if d:
        for fn in os.listdir(d):
            if fn.startswith("cost_") and fn.endswith(".json"):
                try:
                    with open(os.path.join(d, fn)) as f:
                        per = json.load(f)
                except Exception:
                    continue
                for m, v in per.items():
                    # per-PID files are cumulative; take the MAX per PID is wrong
                    # across PIDs — instead sum across distinct PID files (each PID
                    # is a disjoint set of calls).
                    agg = merged.setdefault(m, {"queries": 0, "in": 0, "out": 0, "cost": 0.0})
                    # avoid double-counting this process's own file: skip if it's us
                    if fn == f"cost_{os.getpid()}.json":
                        continue
                    agg["queries"] += v.get("queries", 0)
                    agg["in"] += v.get("in", 0)
                    agg["out"] += v.get("out", 0)
                    agg["cost"] += v.get("cost", 0.0)
    models = [{
        "model": m, "queries": dd["queries"],
        "total_input_tokens": dd["in"], "total_output_tokens": dd["out"],
        "total_tokens": dd["in"] + dd["out"], "total_cost": dd["cost"],
    } for m, dd in merged.items()]
    return {"models": models, "total_cost": sum(d["cost"] for d in merged.values()),
            "pricing_version": _VERSION}


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
