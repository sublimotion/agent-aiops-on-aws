#!/usr/bin/env python3
"""
EKS platform — submits benchmark as a Kubernetes Job and collects results.

For in-cluster endpoints (ClusterIP services), runs the benchmark pod in the same
namespace. For external endpoints (LoadBalancer), can run from anywhere.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
import yaml
from pathlib import Path


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


def submit_job(endpoint: str, workload_path: Path, tool: str, namespace: str = "default") -> str:
    """Submit benchmark job to EKS and return job name."""
    run_id = generate_run_id()

    with open(workload_path) as f:
        workload = yaml.safe_load(f)

    # Build args based on tool
    if tool == "vllm":
        module = "vllm.entrypoints.openai.bench_serving"
        image = "vllm/vllm-openai:latest"
    elif tool == "sglang":
        module = "sglang.bench_serving"
        image = "lmsysorg/sglang:latest"
    else:
        raise ValueError(f"Unsupported tool for EKS: {tool}")

    # Basic args
    dataset = workload.get("dataset", {})
    load = workload.get("load", {})
    args = [
        "--base-url", endpoint,
        "--num-prompts", str(load.get("num_prompts", 100)),
    ]

    if load.get("request_rate"):
        args.extend(["--request-rate", str(load["request_rate"])])

    job_name = f"benchmark-{run_id}"
    job_yaml = JOB_TEMPLATE.format(
        run_id=run_id,
        namespace=namespace,
        workload=workload_path.stem,
        image=image,
        module=module,
        args=json.dumps(args),
        endpoint=endpoint,
    )

    # Apply job
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
    parser = argparse.ArgumentParser(description="EKS platform benchmark execution")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--duration", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DRY RUN] Would submit benchmark job to EKS:")
        print(f"  Endpoint:  {args.endpoint}")
        print(f"  Workload:  {args.workload}")
        print(f"  Tool:      {args.tool}")
        print(f"  Namespace: {args.namespace}")
        return

    # Submit
    job_name = submit_job(args.endpoint, args.workload, args.tool, args.namespace)

    # Wait
    success = wait_for_job(job_name, args.namespace, args.timeout)
    if not success:
        sys.exit(1)

    # Collect
    collect_results(job_name, args.output, args.namespace)


if __name__ == "__main__":
    main()
