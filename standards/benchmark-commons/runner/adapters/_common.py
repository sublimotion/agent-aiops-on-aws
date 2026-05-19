"""
Shared adapter helpers for the CTO engagement blobs.

Every adapter now merges in (when the corresponding file is present):
  - quality   — O3 quality-gate results (produced by run-quality-eval.py)
  - power     — O11 power efficiency (produced by scrape-power.py)
  - hardware_errors — O10 ECC/SDC counters (from scrape-power.py + dmesg grep)
  - stability — O5 72-hour drift (produced by the drift analyzer)
  - cold_start — O9 breakdown (produced by the serving-stack startup probe)

Each blob is optional. If the file is missing the adapter just skips it.
The sidecar points at them via the `artifacts:` section:

  artifacts:
    quality: ./quality/mmlu.json
    power: ./power/run.json
    stability: ./stability/drift.json
    cold_start: ./startup.json

All paths are resolved relative to the sidecar file.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_optional(sidecar_path: Path, rel_path: str | None) -> dict | None:
    if not rel_path:
        return None
    p = (sidecar_path.parent / rel_path).resolve()
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def merge_engagement_blobs(artifact: dict, sidecar: dict, sidecar_path: Path) -> None:
    """Merge optional engagement artifacts (quality/power/etc) into `artifact` in place."""
    artifacts_cfg = sidecar.get("artifacts", {}) or {}

    # O3 quality gate — can be a single file or a list (multi-eval gate)
    quality_ref = artifacts_cfg.get("quality")
    evals: list[dict] = []
    if isinstance(quality_ref, str):
        q = load_optional(sidecar_path, quality_ref)
        if q:
            evals.append(q)
    elif isinstance(quality_ref, list):
        for p in quality_ref:
            q = load_optional(sidecar_path, p)
            if q:
                evals.append(q)
    if evals:
        gate_passed = all(e.get("passed", False) for e in evals)
        artifact["quality"] = {"gate_passed": gate_passed, "evals": evals}

    # O11 power + O10 hardware errors (same source file)
    power_data = load_optional(sidecar_path, artifacts_cfg.get("power"))
    if power_data and "summary" in power_data:
        s = power_data["summary"]
        power_block: dict = {
            "source": power_data.get("source", "dcgm"),
            "duration_s": power_data.get("duration_s"),
        }
        if "avg_fleet_power_watts" in s:
            power_block["avg_fleet_power_watts"] = s["avg_fleet_power_watts"]
        if "total_energy_joules" in s:
            power_block["total_energy_joules"] = s["total_energy_joules"]
            toks = artifact.get("metrics", {}).get("total_output_tokens", 0)
            if toks and s["total_energy_joules"] > 0:
                power_block["tokens_per_joule"] = toks / s["total_energy_joules"]
        if "load_fraction" in artifacts_cfg:
            power_block["load_fraction"] = artifacts_cfg["load_fraction"]
        for key in ("power_watts", "gpu_util_pct", "temp_c"):
            if key in s:
                power_block[key] = {k: v for k, v in s[key].items()
                                    if k in ("mean", "p50", "p95", "p99", "max", "min", "per_gpu_mean")}
        artifact["power"] = power_block

        # O10 ECC / interconnect error deltas live in the same power summary
        err_block: dict = {}
        for key, schema_key in (
            ("ecc_sbe_total_delta", "ecc_sbe_delta"),
            ("ecc_dbe_total_delta", "ecc_dbe_delta"),
        ):
            if key in s:
                err_block[schema_key] = int(s[key])
        for key in ("nvlink_crc_errors", "pcie_replay_count"):
            if key in s and isinstance(s[key], dict) and "max" in s[key] and "min" in s[key]:
                err_block[f"{key}_delta"] = int(s[key]["max"] - s[key]["min"])
        if artifacts_cfg.get("sentinel_divergences") is not None:
            err_block["sentinel_divergences"] = int(artifacts_cfg["sentinel_divergences"])
        if err_block:
            artifact["hardware_errors"] = err_block

    # O5 stability (burn-in drift)
    stab = load_optional(sidecar_path, artifacts_cfg.get("stability"))
    if stab:
        artifact["stability"] = stab

    # O9 cold start
    cs = load_optional(sidecar_path, artifacts_cfg.get("cold_start"))
    if cs:
        artifact["cold_start"] = cs


def compute_iac(metrics: dict, sidecar: dict) -> dict | None:
    """Extracted IAC computation so vLLM + SGLang share the same math."""
    cost_config = sidecar.get("cost")
    if not cost_config or metrics.get("output_toks_per_s", 0) <= 0:
        return None

    spot_price = cost_config.get("spot_price_per_hr", 0)
    utilization = cost_config.get("utilization_target", 0.7)
    effective_toks_per_hr = metrics["output_toks_per_s"] * 3600 * utilization
    cost_per_m_tokens = (spot_price / effective_toks_per_hr) * 1_000_000 if effective_toks_per_hr > 0 else 0

    iac = {
        "cost_per_m_output_tokens": round(cost_per_m_tokens, 4),
        "instance_type": cost_config.get("instance_type", "unknown"),
        "spot_price_per_hr": spot_price,
        "utilization_target": utilization,
    }

    intel = sidecar.get("intelligence", {})
    if intel and intel.get("pass_rate") and intel.get("tokens_per_task", {}).get("output"):
        tokens_per_task_out = intel["tokens_per_task"]["output"]
        tokens_per_task_in = intel["tokens_per_task"].get("input", 0)
        cached_ratio = intel["tokens_per_task"].get("cached_input_ratio", 0)
        pass_rate = intel["pass_rate"]
        human_cost = intel.get("human_intervention_cost", 25.0)

        cost_per_task = (tokens_per_task_out / 1_000_000) * cost_per_m_tokens
        if tokens_per_task_in > 0:
            input_cost_per_m = cost_per_m_tokens * 0.3
            uncached_input = tokens_per_task_in * (1 - cached_ratio)
            cost_per_task += (uncached_input / 1_000_000) * input_cost_per_m

        iac["cost_per_task"] = round(cost_per_task, 6)
        iac["cost_per_success"] = round(cost_per_task / pass_rate, 6)
        iac["true_cost_with_human_fallback"] = round(
            cost_per_task + (1 - pass_rate) * human_cost, 2
        )
        iac["pass_rate"] = pass_rate
        iac["eval_source"] = intel.get("eval_source", "unknown")

    return iac
