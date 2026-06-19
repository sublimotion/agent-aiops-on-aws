#!/usr/bin/env python3
"""HyperPod platform — executes a COMPILED plan via SSM RunCommand / Slurm.

This platform decides WHERE the benchmark runs (an SSM target inside a HyperPod
cluster). It does NOT decide WHAT runs — that is `compiler.compile_card`, the
single deterministic source of truth. Each compiled vendor step becomes one SSM
command; the step's argv (dataset flags, request-rate, goodput) comes straight
from the compiler. The old version built argv inline (synthetic→sonnet only) and
silently dropped every other dataset type — that logic is gone.

HyperPod provides deep health checks and auto-recovery; the artifact captures
that metadata so results compare against plain EKS deployments. An orchestrated
plan cannot run as a vendor SSM command and fails closed.
"""

import argparse
import json
import subprocess
import sys
import time
import yaml
from pathlib import Path

# runner/ is the parent of platforms/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compiler import compile_card  # noqa: E402
from registry import UnsupportedWorkload  # noqa: E402


TOOL_MODULE = {
    "vllm": "vllm.entrypoints.openai.bench_serving",
    "sglang": "sglang.bench_serving",
}


def build_step_cmd(tool: str, endpoint: str, model: str, step_argv: list[str]) -> str:
    """Build the shell command for one compiled step, run on the target node."""
    import shlex
    parts = ["python3", "-m", TOOL_MODULE[tool], "--base-url", endpoint]
    if tool == "vllm":
        parts += ["--model", model, "--save-result"]
    elif tool == "sglang":
        parts += ["--backend", "sglang", "--model", model,
                  "--output-file", "/tmp/bench_output.json"]
    parts += step_argv
    return " ".join(shlex.quote(p) for p in parts)


def resolve_instance(cluster_name: str, instance_group: str) -> str:
    """Resolve the target instance ID from the HyperPod cluster."""
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
    return instance_id


def submit_ssm(instance_id: str, bench_cmd: str, timeout: int) -> str:
    """Submit a benchmark command via SSM. Returns the command ID."""
    ssm_result = subprocess.run(
        ["aws", "ssm", "send-command",
         "--instance-ids", instance_id,
         "--document-name", "AWS-RunShellScript",
         "--parameters", json.dumps({"commands": [bench_cmd]}),
         "--timeout-seconds", str(timeout),
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
    parser = argparse.ArgumentParser(description="HyperPod platform — execute a compiled benchmark plan")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--sidecar", type=Path, help="benchmark.yaml sidecar (for overrides)")
    parser.add_argument("--tool", required=True, choices=list(TOOL_MODULE))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cluster", default="hp-inference-01", help="HyperPod cluster name")
    parser.add_argument("--instance-group", default="gpu-workers")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--duration", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    card = yaml.safe_load(open(args.workload))
    sidecar = yaml.safe_load(open(args.sidecar)) if args.sidecar else {}
    model = sidecar.get("model", {}).get("id", "default")

    # Compile — fail closed and loud on anything unmapped.
    try:
        plan = compile_card(card, sidecar, args.tool)
    except UnsupportedWorkload as e:
        print(f"ERROR: workload did not compile: {e}", file=sys.stderr)
        print("This is a card/sidecar problem. Fix the declaration or add a "
              "registry handler — do not hand-write a one-off driver.", file=sys.stderr)
        sys.exit(3)

    if plan.kind == "orchestrated":
        msg = (f"Card '{plan.catalog_id}' requires the '{plan.orchestrator}' "
               f"orchestrated executor.\nReason: {plan.reason}")
        if args.dry_run:
            print(f"[DRY RUN] {msg}")
            return
        print(f"ERROR: {msg}\n\nOrchestrated cards cannot run as a vendor SSM "
              f"command. Run on the 'local' platform where orchestrators.py is "
              f"dispatched, or implement a HyperPod-native executor.", file=sys.stderr)
        sys.exit(4)

    # Vendor plan: one SSM command per step.
    print(f"Compiled '{plan.catalog_id}' -> {len(plan.steps)} vendor step(s) [{args.tool}]")
    if args.dry_run:
        for i, step in enumerate(plan.steps):
            cmd = build_step_cmd(args.tool, args.endpoint, model, step.argv)
            print(f"  [step {i+1}/{len(plan.steps)}] {step.label}:")
            print(f"    {cmd}")
        print("Dry run complete. No SSM command submitted.")
        return

    instance_id = resolve_instance(args.cluster, args.instance_group)
    for i, step in enumerate(plan.steps):
        cmd = build_step_cmd(args.tool, args.endpoint, model, step.argv)
        print(f"  [step {i+1}/{len(plan.steps)}] {step.label}: submitting SSM command...")
        command_id = submit_ssm(instance_id, cmd, args.timeout)
        print(f"    Monitor: aws ssm get-command-invocation "
              f"--command-id {command_id} --instance-id {instance_id}")
    print(f"\nNote: Collect results from HyperPod FSx or SSM output "
          f"(/tmp/bench_output.json on the target instance).")


if __name__ == "__main__":
    main()
