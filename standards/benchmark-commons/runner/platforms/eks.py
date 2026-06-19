#!/usr/bin/env python3
"""EKS platform — executes a COMPILED plan as Kubernetes Job(s).

This platform decides WHERE the benchmark runs (a pod in the cluster). It does
NOT decide WHAT runs — that is `compiler.compile_card`, the single deterministic
source of truth. Each compiled vendor step becomes one Job; the step's argv
(dataset flags, request-rate, goodput) comes straight from the compiler. The old
version built argv inline and silently dropped unmapped dataset types — that
logic is gone.

A vendor plan with N sweep steps submits N Jobs (one per step). An orchestrated
plan cannot run as a vendor Job and fails closed.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
import yaml
from pathlib import Path

# runner/ is the parent of platforms/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compiler import compile_card  # noqa: E402
from registry import UnsupportedWorkload  # noqa: E402


TOOL_IMAGE = {
    "vllm": ("vllm/vllm-openai:latest", "vllm.entrypoints.openai.bench_serving"),
    "sglang": ("lmsysorg/sglang:latest", "sglang.bench_serving"),
}


JOB_TEMPLATE = """
apiVersion: batch/v1
kind: Job
metadata:
  name: benchmark-{run_id}
  namespace: {namespace}
  labels:
    app: benchmark-runner
    workload: {workload}
spec:
  ttlSecondsAfterFinished: 300
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: benchmark
        image: {image}
        command: ["python3", "-m", "{module}"]
        args: {args}
        env:
        - name: BENCHMARK_ENDPOINT
          value: "{endpoint}"
        resources:
          requests:
            cpu: "4"
            memory: "8Gi"
          limits:
            cpu: "8"
            memory: "16Gi"
        volumeMounts:
        - name: results
          mountPath: /results
      volumes:
      - name: results
        emptyDir: {{}}
"""


def generate_run_id() -> str:
    """Short unique run ID."""
    import hashlib
    from datetime import datetime
    return hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:8]


def step_args(tool: str, endpoint: str, model: str, step_argv: list[str]) -> list[str]:
    """Prepend connection flags to a compiled step's argv (k8s container args)."""
    args = ["--base-url", endpoint]
    if tool == "vllm":
        args += ["--model", model, "--save-result"]
    elif tool == "sglang":
        args += ["--backend", "sglang", "--model", model]
    return args + step_argv


def submit_job(endpoint: str, workload_stem: str, tool: str, args: list[str],
               namespace: str) -> str:
    """Submit one benchmark Job with the given vendor args. Returns job name."""
    run_id = generate_run_id()
    image, module = TOOL_IMAGE[tool]
    job_name = f"benchmark-{run_id}"
    job_yaml = JOB_TEMPLATE.format(
        run_id=run_id,
        namespace=namespace,
        workload=workload_stem,
        image=image,
        module=module,
        args=json.dumps(args),
        endpoint=endpoint,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(job_yaml)
        f.flush()
        result = subprocess.run(
            ["kubectl", "apply", "-f", f.name],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Failed to submit job: {result.stderr}")
            sys.exit(1)
    print(f"Submitted job: {job_name}")
    return job_name


def wait_for_job(job_name: str, namespace: str = "default", timeout: int = 600) -> bool:
    """Wait for job to complete."""
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(
            ["kubectl", "get", "job", job_name, "-n", namespace, "-o", "json"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            time.sleep(5)
            continue

        job = json.loads(result.stdout)
        status = job.get("status", {})

        if status.get("succeeded", 0) > 0:
            return True
        if status.get("failed", 0) > 0:
            print(f"Job failed. Logs:")
            subprocess.run(["kubectl", "logs", f"job/{job_name}", "-n", namespace])
            return False

        time.sleep(10)

    print(f"Timeout waiting for job ({timeout}s)")
    return False


def collect_results(job_name: str, output_path: Path, namespace: str = "default"):
    """Collect results from completed job pod."""
    # Get pod name
    result = subprocess.run(
        ["kubectl", "get", "pods", "-n", namespace,
         "-l", f"job-name={job_name}", "-o", "jsonpath={.items[0].metadata.name}"],
        capture_output=True, text=True
    )
    pod_name = result.stdout.strip()

    # Get logs (benchmark tools print JSON to stdout)
    result = subprocess.run(
        ["kubectl", "logs", pod_name, "-n", namespace],
        capture_output=True, text=True
    )

    # Try to parse JSON from logs
    for line in result.stdout.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                if "mean_ttft_ms" in data or "p50_ttft_ms" in data or "ttft_ms" in data:
                    output_path.write_text(json.dumps(data, indent=2))
                    print(f"Results collected: {output_path}")
                    return
            except json.JSONDecodeError:
                continue

    # Fallback: copy from pod volume
    subprocess.run(
        ["kubectl", "cp", f"{namespace}/{pod_name}:/results/output.json", str(output_path)],
        capture_output=True
    )
    if output_path.exists():
        print(f"Results collected (volume): {output_path}")
    else:
        print("Warning: could not collect results from job")


def main():
    parser = argparse.ArgumentParser(description="EKS platform — execute a compiled benchmark plan")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--sidecar", type=Path, help="benchmark.yaml sidecar (for overrides)")
    parser.add_argument("--tool", required=True, choices=list(TOOL_IMAGE))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--namespace", default="default")
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
        print(f"ERROR: {msg}\n\nOrchestrated cards cannot run as a vendor "
              f"Kubernetes Job. Run on the 'local' platform where "
              f"orchestrators.py is dispatched, or implement an EKS-native "
              f"executor.", file=sys.stderr)
        sys.exit(4)

    # Vendor plan: one Job per step.
    print(f"Compiled '{plan.catalog_id}' -> {len(plan.steps)} vendor step(s) [{args.tool}]")
    for i, step in enumerate(plan.steps):
        step_argv = step_args(args.tool, args.endpoint, model, step.argv)
        if args.dry_run:
            image, module = TOOL_IMAGE[args.tool]
            print(f"  [step {i+1}/{len(plan.steps)}] {step.label}:")
            print(f"    image={image} module={module}")
            print(f"    args={json.dumps(step_argv)}")
            continue
        print(f"  [step {i+1}/{len(plan.steps)}] {step.label}: submitting Job...")
        job_name = submit_job(
            args.endpoint, args.workload.stem, args.tool, step_argv, args.namespace)
        if not wait_for_job(job_name, args.namespace, args.timeout):
            sys.exit(1)
        step_out = (args.output if len(plan.steps) == 1
                    else args.output.with_name(f"{args.output.stem}__{step.label}.json"))
        collect_results(job_name, step_out, args.namespace)

    if args.dry_run:
        print("Dry run complete. No Job submitted.")


if __name__ == "__main__":
    main()
