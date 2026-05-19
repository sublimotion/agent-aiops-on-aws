#!/usr/bin/env python3
"""
Local platform — runs benchmark tool directly against an endpoint.
Used for bare metal, spot instances, or localhost testing.
"""

import argparse
import json
import subprocess
import sys
import yaml
from pathlib import Path


def build_vllm_command(endpoint: str, workload: dict, num_prompts: int | None, duration: int | None) -> list[str]:
    """Build vLLM benchmark_serving.py command from workload spec."""
    dataset = workload.get("dataset", {})
    load = workload.get("load", {})
    api = workload.get("api", {})

    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.bench_serving",
        "--base-url", endpoint,
        "--model", "default",  # overridden by sidecar
    ]

    # Dataset params
    if dataset.get("type") == "synthetic":
        input_tokens = dataset.get("input_tokens", {})
        output_tokens = dataset.get("output_tokens", {})
        if isinstance(input_tokens, dict):
            cmd.extend(["--sonnet-input-len", str(input_tokens.get("mean", 256))])
        else:
            cmd.extend(["--sonnet-input-len", str(input_tokens)])
        if isinstance(output_tokens, dict):
            cmd.extend(["--sonnet-output-len", str(output_tokens.get("mean", 128))])
        else:
            cmd.extend(["--sonnet-output-len", str(output_tokens)])
        cmd.extend(["--dataset-name", "sonnet"])
    elif dataset.get("type") == "sharegpt":
        cmd.extend(["--dataset-name", "sharegpt"])

    # Load params
    np = num_prompts or load.get("num_prompts", 100)
    cmd.extend(["--num-prompts", str(np)])

    if load.get("type") == "constant" and load.get("request_rate"):
        cmd.extend(["--request-rate", str(load["request_rate"])])

    # API params
    if api.get("endpoint", "").endswith("chat/completions"):
        cmd.extend(["--endpoint", "/v1/chat/completions"])

    # Output
    cmd.extend(["--save-result"])

    return cmd


def build_sglang_command(endpoint: str, workload: dict, num_prompts: int | None, duration: int | None) -> list[str]:
    """Build SGLang bench_serving.py command from workload spec."""
    dataset = workload.get("dataset", {})
    load = workload.get("load", {})

    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--base-url", endpoint,
    ]

    # Dataset
    if dataset.get("type") == "synthetic":
        input_tokens = dataset.get("input_tokens", {})
        output_tokens = dataset.get("output_tokens", {})
        mean_in = input_tokens.get("mean", 256) if isinstance(input_tokens, dict) else input_tokens
        mean_out = output_tokens.get("mean", 128) if isinstance(output_tokens, dict) else output_tokens
        cmd.extend(["--random-input-len", str(mean_in)])
        cmd.extend(["--random-output-len", str(mean_out)])
        cmd.extend(["--dataset-name", "random"])

    # Load
    np = num_prompts or load.get("num_prompts", 100)
    cmd.extend(["--num-prompts", str(np)])

    if load.get("request_rate"):
        cmd.extend(["--request-rate", str(load["request_rate"])])

    cmd.extend(["--output-file", "/tmp/sglang_bench_output.json"])

    return cmd


TOOL_BUILDERS = {
    "vllm": build_vllm_command,
    "sglang": build_sglang_command,
}


def main():
    parser = argparse.ArgumentParser(description="Local platform benchmark execution")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--tool", required=True, choices=list(TOOL_BUILDERS.keys()))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--duration", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load workload
    with open(args.workload) as f:
        workload = yaml.safe_load(f)

    # Build command
    builder = TOOL_BUILDERS[args.tool]
    cmd = builder(args.endpoint, workload, args.num_prompts, args.duration)

    if args.dry_run:
        print(f"[DRY RUN] Would execute:")
        print(f"  {' '.join(cmd)}")
        return

    print(f"Executing: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Benchmark failed (exit {result.returncode}):")
        print(result.stderr[-2000:] if result.stderr else "No stderr")
        sys.exit(1)

    # Find output file (tool-specific)
    if args.tool == "vllm":
        # vLLM saves to current dir with timestamp
        import glob
        outputs = sorted(glob.glob("*.json"), key=lambda f: Path(f).stat().st_mtime, reverse=True)
        if outputs:
            Path(outputs[0]).rename(args.output)
    elif args.tool == "sglang":
        Path("/tmp/sglang_bench_output.json").rename(args.output)

    print(f"Raw output saved to: {args.output}")


if __name__ == "__main__":
    main()
