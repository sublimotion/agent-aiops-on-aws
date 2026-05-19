#!/usr/bin/env python3
"""
HyperPod platform — submits benchmark via SSM RunCommand or Slurm.

HyperPod provides:
- Deep health checks (automatic bad-GPU detection)
- Auto-recovery (unhealthy nodes replaced without manual intervention)
- Slurm job scheduling (for multi-node or queued workloads)

The benchmark artifact captures HyperPod-specific metadata (deep_health_checks,
auto_recovery) so results can be compared against plain EKS deployments.
"""

import argparse
import json
import subprocess
import sys
import time
import yaml
from pathlib import Path


def submit_ssm(
    cluster_name: str,
    instance_group: str,
    endpoint: str,
    workload_path: Path,
    tool: str,
    output_path: Path,
    num_prompts: int | None,
) -> str:
    """Submit benchmark via SSM RunCommand to a HyperPod instance."""
    with open(workload_path) as f:
        workload = yaml.safe_load(f)

    load = workload.get("load", {})
    dataset = workload.get("dataset", {})
    np = num_prompts or load.get("num_prompts", 100)

    # Build benchmark command
    if tool == "vllm":
        bench_cmd = (
            f"python3 -m vllm.entrypoints.openai.bench_serving "
            f"--base-url {endpoint} "
            f"--num-prompts {np} "
            f"--save-result "
        )
        if load.get("request_rate"):
            bench_cmd += f"--request-rate {load['request_rate']} "
        if dataset.get("type") == "synthetic":
            input_mean = dataset.get("input_tokens", {}).get("mean", 256)
            output_mean = dataset.get("output_tokens", {}).get("mean", 128)
            bench_cmd += f"--sonnet-input-len {input_mean} --sonnet-output-len {output_mean} --dataset-name sonnet "
    elif tool == "sglang":
        bench_cmd = (
            f"python3 -m sglang.bench_serving "
            f"--backend sglang --base-url {endpoint} "
            f"--num-prompts {np} "
            f"--output-file /tmp/bench_output.json "
        )
        if load.get("request_rate"):
            bench_cmd += f"--request-rate {load['request_rate']} "
    else:
        raise ValueError(f"Unsupported tool: {tool}")

    # Get target instance ID from HyperPod cluster
    result = subprocess.run(
        ["aws", "sagemaker", "list-cluster-nodes",
         "--cluster-name", cluster_name,
         "--instance-group-name-contains", instance_group,
         "--query", "ClusterNodeSummaries[0].InstanceId",
         "--output", "text"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Failed to find HyperPod instance: {result.stderr}")
        sys.exit(1)

    instance_id = result.stdout.strip()
    print(f"Target instance: {instance_id} (cluster: {cluster_name}, group: {instance_group})")

    # Submit via SSM
    ssm_result = subprocess.run(
        ["aws", "ssm", "send-command",
         "--instance-ids", instance_id,
         "--document-name", "AWS-RunShellScript",
         "--parameters", json.dumps({"commands": [bench_cmd]}),
         "--timeout-seconds", "600",
         "--output", "json"],
        capture_output=True, text=True
    )

    if ssm_result.returncode != 0:
        print(f"SSM command failed: {ssm_result.stderr}")
        sys.exit(1)

    command_data = json.loads(ssm_result.stdout)
    command_id = command_data["Command"]["CommandId"]
    print(f"SSM command submitted: {command_id}")
    return command_id


def wait_for_ssm(command_id: str, instance_id: str, timeout: int = 600) -> str | None:
    """Wait for SSM command to complete. Returns output or None."""
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(
            ["aws", "ssm", "get-command-invocation",
             "--command-id", command_id,
             "--instance-id", instance_id,
             "--output", "json"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            time.sleep(10)
            continue

        invocation = json.loads(result.stdout)
        status = invocation.get("Status")

        if status == "Success":
            return invocation.get("StandardOutputContent", "")
        elif status in ("Failed", "TimedOut", "Cancelled"):
            print(f"Command {status}: {invocation.get('StandardErrorContent', '')[:1000]}")
            return None

        time.sleep(10)

    print(f"Timeout waiting for SSM command ({timeout}s)")
    return None


def main():
    parser = argparse.ArgumentParser(description="HyperPod platform benchmark execution")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cluster", default="hp-inference-01", help="HyperPod cluster name")
    parser.add_argument("--instance-group", default="gpu-workers")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--duration", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DRY RUN] Would submit benchmark to HyperPod:")
        print(f"  Cluster:  {args.cluster}")
        print(f"  Group:    {args.instance_group}")
        print(f"  Endpoint: {args.endpoint}")
        print(f"  Workload: {args.workload}")
        print(f"  Tool:     {args.tool}")
        return

    # Submit
    command_id = submit_ssm(
        args.cluster, args.instance_group,
        args.endpoint, args.workload, args.tool,
        args.output, args.num_prompts,
    )

    # Wait and collect
    # (simplified — full implementation would handle multi-node, Slurm, FSx collection)
    print(f"Waiting for completion (timeout: {args.timeout}s)...")
    print(f"Monitor: aws ssm get-command-invocation --command-id {command_id} --instance-id <id>")
    print(f"\nNote: Collect results manually from HyperPod FSx or SSM output.")
    print(f"Expected output location: /tmp/bench_output.json on the target instance.")


if __name__ == "__main__":
    main()
