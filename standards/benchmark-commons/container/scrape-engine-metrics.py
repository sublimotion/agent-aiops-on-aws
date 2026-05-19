#!/usr/bin/env python3
"""
Scrapes engine-internal Prometheus metrics during a benchmark run.
Runs as a background process alongside AIPerf, sampling at a fixed interval.

Produces a JSON summary (mean/max/min per metric) for the extensions block
of the enriched artifact.

Usage:
  scrape-engine-metrics --url http://localhost:8000/metrics --duration 120 --output metrics.json

Supported engines:
  - vLLM: vllm:gpu_cache_usage_perc, vllm:num_requests_running, vllm:cache_hit_rate
  - SGLang: sglang_* equivalent metrics
  - Dynamo: dynamo_kvbm_*, dynamo_nixl_* metrics
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

# Metrics we care about — engine-agnostic names mapped to possible Prometheus metric names
METRIC_PATTERNS = {
    "kv_utilization_pct": [
        r"vllm:gpu_cache_usage_perc",
        r"sglang_gpu_cache_usage",
        r"dynamo_kv_cache_usage",
    ],
    "prefix_hit_rate": [
        r"vllm:cache_hit_rate",
        r"sglang_prefix_cache_hit_rate",
        r"dynamo_prefix_hit_rate",
    ],
    "running_requests": [
        r"vllm:num_requests_running",
        r"sglang_num_running_requests",
        r"dynamo_running_requests",
    ],
    "waiting_requests": [
        r"vllm:num_requests_waiting",
        r"sglang_num_waiting_requests",
    ],
    "gpu_cache_blocks_used": [
        r"vllm:num_gpu_blocks_used",
    ],
    "gpu_cache_blocks_total": [
        r"vllm:num_gpu_blocks",
    ],
    "eviction_count": [
        r"vllm:cache_evictions_total",
        r"sglang_cache_evictions_total",
    ],
    "spec_decode_acceptance": [
        r"vllm:spec_decode_draft_acceptance_rate",
        r"sglang_spec_decode_acceptance_rate",
    ],
}


def parse_prometheus_text(text: str) -> dict:
    """Parse Prometheus text exposition format into metric_name -> value."""
    metrics = {}
    for line in text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        # metric_name{labels} value [timestamp]
        # or metric_name value [timestamp]
        match = re.match(r"^([^\s{]+)(?:\{[^}]*\})?\s+([\d.eE+-]+)", line)
        if match:
            name = match.group(1)
            try:
                value = float(match.group(2))
                metrics[name] = value
            except ValueError:
                pass
    return metrics


def match_metric(raw_metrics: dict, patterns: list) -> float | None:
    """Find the first matching metric from a list of patterns."""
    for pattern in patterns:
        for name, value in raw_metrics.items():
            if re.match(pattern, name):
                return value
    return None


def scrape_once(url: str) -> dict:
    """Scrape metrics endpoint once and return normalized values."""
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return {}
        raw = parse_prometheus_text(resp.text)
    except (requests.RequestException, ConnectionError):
        return {}

    result = {}
    for canonical_name, patterns in METRIC_PATTERNS.items():
        value = match_metric(raw, patterns)
        if value is not None:
            result[canonical_name] = value
    return result


def summarize(samples: list[dict]) -> dict:
    """Compute mean/max/min for each metric across samples."""
    if not samples:
        return {}

    # Collect per-metric series
    series = defaultdict(list)
    for sample in samples:
        for key, value in sample.items():
            series[key].append(value)

    summary = {}
    for key, values in series.items():
        if values:
            summary[key] = {
                "mean": round(sum(values) / len(values), 4),
                "max": round(max(values), 4),
                "min": round(min(values), 4),
                "samples": len(values),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Scrape engine metrics during benchmark")
    parser.add_argument("--url", default="http://localhost:8000/metrics", help="Prometheus metrics endpoint")
    parser.add_argument("--duration", type=int, required=True, help="Scrape duration in seconds")
    parser.add_argument("--interval", type=int, default=5, help="Scrape interval in seconds")
    parser.add_argument("--output", required=True, help="Output JSON file")

    args = parser.parse_args()

    print(f"Scraping {args.url} every {args.interval}s for {args.duration}s", flush=True)

    samples = []
    start = time.time()
    while time.time() - start < args.duration:
        sample = scrape_once(args.url)
        if sample:
            samples.append(sample)
        time.sleep(args.interval)

    summary = summarize(samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Collected {len(samples)} samples, {len(summary)} metrics → {output_path}", flush=True)


if __name__ == "__main__":
    main()
