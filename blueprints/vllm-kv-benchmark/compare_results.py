#!/usr/bin/env python3
"""Compare baseline vs LMCache benchmark results."""

import json
from pathlib import Path

def load_results(directory: str, pattern: str):
    """Load most recent result matching pattern."""
    dir_path = Path(directory)
    files = sorted(dir_path.glob(pattern), reverse=True)
    if files:
        with open(files[0]) as f:
            return json.load(f)
    return None

def format_metrics(data):
    """Format metrics for display."""
    if not data:
        return "  No data"
    m = data["metrics"]
    return f"  TTFT p50: {m['ttft_ms']['p50']:.0f}ms, E2E p50: {m['e2e_ms']['p50']:.0f}ms, Throughput: {m['throughput']['tokens_per_second']:.1f} tok/s"

def main():
    baseline_dir = "results/kimi-k2.5-p5e"
    lmcache_dir = "results/kimi-k2.5-p5e-lmcache"

    workloads = [
        ("reasoning_math", ["low", "medium", "high"]),
        ("code_generation", ["low", "medium", "high"]),
        ("multi_turn_qa", ["low", "medium", "high"]),
        ("long_context_rag", ["low", "medium", "high"]),
        ("agentic_tool_use", ["low", "medium", "high"]),
    ]

    print("=" * 60)
    print("Kimi K2.5 on p5e.48xlarge: Baseline vs LMCache")
    print("=" * 60)
    print()

    for workload, qps_levels in workloads:
        print(f"### {workload.upper()}")
        print()
        for qps in qps_levels:
            pattern = f"kimi_k2.5_{workload}_{qps}_*.json"

            baseline = load_results(baseline_dir, pattern)
            lmcache = load_results(lmcache_dir, pattern)

            qps_val = {"low": 0.5, "medium": 2.0, "high": 5.0}[qps]
            print(f"QPS: {qps} ({qps_val})")
            print(f"Baseline:{format_metrics(baseline)}")
            print(f"LMCache: {format_metrics(lmcache)}")

            # Calculate improvement
            if baseline and lmcache:
                b_ttft = baseline["metrics"]["ttft_ms"]["p50"]
                l_ttft = lmcache["metrics"]["ttft_ms"]["p50"]
                ttft_change = ((l_ttft - b_ttft) / b_ttft) * 100

                b_tput = baseline["metrics"]["throughput"]["tokens_per_second"]
                l_tput = lmcache["metrics"]["throughput"]["tokens_per_second"]
                tput_change = ((l_tput - b_tput) / b_tput) * 100

                print(f"  Change:  TTFT: {ttft_change:+.1f}%, Throughput: {tput_change:+.1f}%")
            print()
        print()

if __name__ == "__main__":
    main()
