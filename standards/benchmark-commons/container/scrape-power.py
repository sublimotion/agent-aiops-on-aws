#!/usr/bin/env python3
"""
Samples GPU power and utilization during a benchmark run.

Runs as a background process alongside the load generator, sampling at 1 Hz
(DCGM convention). Produces a JSON summary with the fields needed for O11
(power efficiency tokens/joule) and O10 (ECC error counters).

Preferred source: NVIDIA DCGM Prometheus exporter (DCGM_FI_DEV_POWER_USAGE).
Fallback: nvidia-smi query loop.

Usage:
  scrape-power.py --dcgm-url http://localhost:9400/metrics --duration 900 --output power.json
  scrape-power.py --nvidia-smi --duration 900 --output power.json  # fallback
"""

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests


DCGM_PATTERNS = {
    "power_watts": r"DCGM_FI_DEV_POWER_USAGE",
    "gpu_util_pct": r"DCGM_FI_DEV_GPU_UTIL",
    "mem_util_pct": r"DCGM_FI_DEV_MEM_COPY_UTIL",
    "sm_clock_mhz": r"DCGM_FI_DEV_SM_CLOCK",
    "mem_clock_mhz": r"DCGM_FI_DEV_MEM_CLOCK",
    "temp_c": r"DCGM_FI_DEV_GPU_TEMP",
    "ecc_sbe_total": r"DCGM_FI_DEV_ECC_SBE_AGG_TOTAL",
    "ecc_dbe_total": r"DCGM_FI_DEV_ECC_DBE_AGG_TOTAL",
    "nvlink_crc_errors": r"DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL",
    "pcie_replay_count": r"DCGM_FI_DEV_PCIE_REPLAY_COUNTER",
}


def scrape_dcgm(url: str) -> dict:
    """One sample from the DCGM exporter; returns {metric_key: [values per GPU]}."""
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        return {"error": str(e)}

    out = {k: [] for k in DCGM_PATTERNS}
    for line in resp.text.splitlines():
        if line.startswith("#"):
            continue
        for key, pattern in DCGM_PATTERNS.items():
            if line.startswith(pattern):
                try:
                    out[key].append(float(line.rsplit(" ", 1)[1]))
                except (ValueError, IndexError):
                    pass
                break
    return out


def scrape_nvidia_smi() -> dict:
    """Fallback sampler using nvidia-smi --query-gpu."""
    if not shutil.which("nvidia-smi"):
        return {"error": "nvidia-smi not on PATH"}
    fields = [
        "power.draw",
        "utilization.gpu",
        "utilization.memory",
        "clocks.current.sm",
        "clocks.current.memory",
        "temperature.gpu",
        "ecc.errors.corrected.aggregate.total",
        "ecc.errors.uncorrected.aggregate.total",
    ]
    query = ",".join(fields)
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except Exception as e:
        return {"error": str(e)}

    keys = [
        "power_watts", "gpu_util_pct", "mem_util_pct",
        "sm_clock_mhz", "mem_clock_mhz", "temp_c",
        "ecc_sbe_total", "ecc_dbe_total",
    ]
    out = {k: [] for k in keys}
    for row in proc.stdout.strip().splitlines():
        vals = [v.strip() for v in row.split(",")]
        for k, v in zip(keys, vals):
            try:
                out[k].append(float(v))
            except ValueError:
                pass
    return out


def summarize(samples: list[dict], num_gpus: int, duration_s: float) -> dict:
    """Reduce a time series of per-GPU samples to the fields O11 + O10 need."""
    # Per-metric, per-GPU series
    series: dict[str, list[list[float]]] = {}
    for s in samples:
        for key, values in s.items():
            if key == "error" or not values:
                continue
            series.setdefault(key, [[] for _ in range(num_gpus)])
            for i, v in enumerate(values[:num_gpus]):
                series[key][i].append(v)

    summary = {}
    for key, per_gpu in series.items():
        flat = [v for gpu_series in per_gpu for v in gpu_series]
        if not flat:
            continue
        summary[key] = {
            "mean": statistics.mean(flat),
            "p50": statistics.median(flat),
            "p95": _pct(flat, 0.95),
            "p99": _pct(flat, 0.99),
            "max": max(flat),
            "min": min(flat),
            "per_gpu_mean": [statistics.mean(g) if g else None for g in per_gpu],
        }

    # O11 derived fields: total energy and average fleet power
    if "power_watts" in summary:
        sum_per_sample = []
        for s in samples:
            pw = s.get("power_watts", [])
            if pw:
                sum_per_sample.append(sum(pw))
        if sum_per_sample:
            avg_fleet_power_w = statistics.mean(sum_per_sample)
            summary["total_energy_joules"] = avg_fleet_power_w * duration_s
            summary["avg_fleet_power_watts"] = avg_fleet_power_w

    # O10 derived fields: ECC delta over the run (first → last sample)
    for ecc_key in ("ecc_sbe_total", "ecc_dbe_total"):
        if ecc_key in series:
            first = [g[0] for g in series[ecc_key] if g]
            last = [g[-1] for g in series[ecc_key] if g]
            if first and last:
                summary[f"{ecc_key}_delta"] = sum(last) - sum(first)

    return summary


def _pct(values: list[float], q: float) -> float:
    vs = sorted(values)
    idx = min(len(vs) - 1, int(q * len(vs)))
    return vs[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcgm-url", default="http://localhost:9400/metrics")
    parser.add_argument("--nvidia-smi", action="store_true", help="Fallback: use nvidia-smi")
    parser.add_argument("--duration", type=int, required=True, help="Sampling duration (s)")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval (s)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sampler = (lambda: scrape_nvidia_smi()) if args.nvidia_smi else (lambda: scrape_dcgm(args.dcgm_url))

    print(f"Power sampling: {args.duration}s @ {args.interval}s interval "
          f"({'nvidia-smi' if args.nvidia_smi else 'DCGM'})", file=sys.stderr)

    samples = []
    start = time.time()
    num_gpus = 0
    while time.time() - start < args.duration:
        s = sampler()
        if "error" not in s:
            samples.append(s)
            if num_gpus == 0 and s.get("power_watts"):
                num_gpus = len(s["power_watts"])
        time.sleep(args.interval)

    duration_actual = time.time() - start
    summary = summarize(samples, num_gpus, duration_actual)

    output = {
        "source": "nvidia-smi" if args.nvidia_smi else "dcgm",
        "duration_s": duration_actual,
        "interval_s": args.interval,
        "num_samples": len(samples),
        "num_gpus": num_gpus,
        "summary": summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {args.output} ({len(samples)} samples, {num_gpus} GPUs)", file=sys.stderr)


if __name__ == "__main__":
    main()
