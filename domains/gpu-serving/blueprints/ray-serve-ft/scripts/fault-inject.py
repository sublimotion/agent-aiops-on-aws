#!/usr/bin/env python3
"""
Fault injection test runner for Ray Serve FT.

Runs T1-T6 test scenarios sequentially:
  T1: Replica crash recovery
  T2: Worker node drain
  T3: Head node failure WITH GCS FT
  T4: Head node failure WITHOUT GCS FT (control)
  T5: HTTP proxy failover
  T6: ElastiCache connectivity disruption

Each test:
  1. Starts traffic-gen in background (50 req/s)
  2. Waits 60s warmup
  3. Injects fault
  4. Observes for 120s
  5. Stops traffic, collects metrics

Usage:
  python3 fault-inject.py --test T1
  python3 fault-inject.py --test all
  python3 fault-inject.py --test T3 --test T4  # compare FT vs no-FT
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


NAMESPACE = "ray-ft"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def kubectl(*args, capture=True, check=True):
    """Run kubectl command."""
    cmd = ["kubectl", "-n", NAMESPACE] + list(args)
    result = subprocess.run(cmd, capture_output=capture, text=True, check=check)
    return result.stdout.strip() if capture else None


def kubectl_json(*args):
    """Run kubectl command and parse JSON output."""
    output = kubectl(*args, "-o", "json")
    return json.loads(output)


def get_head_pod():
    """Get the head pod name."""
    pods = kubectl_json("get", "pods", "-l", "ray-node=head")
    items = pods.get("items", [])
    if items:
        return items[0]["metadata"]["name"]
    return None


def get_worker_pods():
    """Get worker pod names."""
    pods = kubectl_json("get", "pods", "-l", "ray-node=worker")
    return [p["metadata"]["name"] for p in pods.get("items", [])]


def get_worker_nodes():
    """Get nodes running worker pods."""
    pods = kubectl_json("get", "pods", "-l", "ray-node=worker")
    return list(set(
        p["spec"]["nodeName"]
        for p in pods.get("items", [])
        if p["spec"].get("nodeName")
    ))


def start_traffic(test_name: str, duration: int = 240) -> subprocess.Popen:
    """Start traffic generator in background."""
    output_path = str(RESULTS_DIR / f"traffic_{test_name}_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    proc = subprocess.Popen(
        [
            sys.executable, str(Path(__file__).parent / "traffic-gen.py"),
            "--rps", "50",
            "--duration", str(duration),
            "--output", output_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc, output_path


def analyze_traffic(traffic_path: str) -> dict:
    """Analyze traffic JSONL for error rates and latency."""
    records = []
    with open(traffic_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        return {"total": 0, "errors": 0, "error_rate": 0}

    total = len(records)
    errors = sum(1 for r in records if r.get("error") or r.get("status", 200) != 200)
    ok_latencies = [r["latency_ms"] for r in records if not r.get("error") and r.get("status") == 200]

    # Find error window (first and last error timestamps)
    error_records = [r for r in records if r.get("error") or r.get("status", 200) != 200]
    error_window_s = 0
    if len(error_records) >= 2:
        error_window_s = error_records[-1]["timestamp"] - error_records[0]["timestamp"]

    sorted_lat = sorted(ok_latencies) if ok_latencies else [0]
    return {
        "total": total,
        "successes": len(ok_latencies),
        "errors": errors,
        "error_rate_pct": round(100 * errors / total, 2),
        "success_rate_pct": round(100 * len(ok_latencies) / total, 2),
        "latency_p50_ms": sorted_lat[len(sorted_lat) // 2],
        "latency_p99_ms": sorted_lat[int(len(sorted_lat) * 0.99)],
        "error_window_s": round(error_window_s, 1),
    }


def wait_with_countdown(seconds: int, label: str):
    """Wait with periodic status output."""
    print(f"  {label} ({seconds}s)...")
    for i in range(0, seconds, 10):
        remaining = seconds - i
        print(f"    {remaining}s remaining...")
        time.sleep(min(10, remaining))


def test_t1_replica_crash():
    """T1: Kill one replica actor, verify recovery."""
    print("\n" + "=" * 60)
    print("T1: REPLICA CRASH RECOVERY")
    print("=" * 60)

    workers = get_worker_pods()
    if len(workers) < 2:
        print("ERROR: Need at least 2 worker pods")
        return None

    proc, traffic_path = start_traffic("T1", duration=240)
    wait_with_countdown(60, "Warmup")

    # Kill the Ray worker process in one worker pod (simulates replica crash)
    target = workers[0]
    print(f"  INJECTING FAULT: Killing ray worker process in {target}")
    kubectl("exec", target, "-c", "ray-worker", "--", "pkill", "-f", "ray::SERVE_REPLICA", check=False)

    wait_with_countdown(120, "Observing recovery")

    proc.terminate()
    proc.wait(timeout=10)
    time.sleep(2)

    metrics = analyze_traffic(traffic_path)
    metrics["test"] = "T1"
    metrics["fault"] = f"pkill ray::SERVE_REPLICA in {target}"
    print(f"\n  Results: {json.dumps(metrics, indent=2)}")
    return metrics


def test_t2_worker_drain():
    """T2: Drain one worker node, verify replica migration."""
    print("\n" + "=" * 60)
    print("T2: WORKER NODE DRAIN")
    print("=" * 60)

    nodes = get_worker_nodes()
    if len(nodes) < 2:
        print("ERROR: Need at least 2 worker nodes")
        return None

    proc, traffic_path = start_traffic("T2", duration=300)
    wait_with_countdown(60, "Warmup")

    target_node = nodes[0]
    print(f"  INJECTING FAULT: Draining node {target_node}")
    subprocess.run(
        ["kubectl", "drain", target_node, "--ignore-daemonsets", "--delete-emptydir-data",
         "--force", "--grace-period=30", "--timeout=120s"],
        check=False,
    )

    wait_with_countdown(120, "Observing recovery")

    # Uncordon the node for cleanup
    print(f"  Uncordoning {target_node}")
    subprocess.run(["kubectl", "uncordon", target_node], check=False)

    proc.terminate()
    proc.wait(timeout=10)
    time.sleep(2)

    metrics = analyze_traffic(traffic_path)
    metrics["test"] = "T2"
    metrics["fault"] = f"kubectl drain {target_node}"
    print(f"\n  Results: {json.dumps(metrics, indent=2)}")
    return metrics


def test_t3_head_failure_ft():
    """T3: Kill head pod WITH GCS FT enabled."""
    print("\n" + "=" * 60)
    print("T3: HEAD NODE FAILURE (GCS FT ENABLED)")
    print("=" * 60)

    head = get_head_pod()
    if not head:
        print("ERROR: No head pod found")
        return None

    proc, traffic_path = start_traffic("T3", duration=300)
    wait_with_countdown(60, "Warmup")

    print(f"  INJECTING FAULT: Force-deleting head pod {head}")
    subprocess.run(
        ["kubectl", "-n", NAMESPACE, "delete", "pod", head, "--force", "--grace-period=0"],
        check=False,
    )

    wait_with_countdown(180, "Observing recovery (head restart + GCS restore)")

    proc.terminate()
    proc.wait(timeout=10)
    time.sleep(2)

    metrics = analyze_traffic(traffic_path)
    metrics["test"] = "T3"
    metrics["fault"] = f"kubectl delete pod {head} --force"
    metrics["gcs_ft"] = True
    print(f"\n  Results: {json.dumps(metrics, indent=2)}")
    return metrics


def test_t4_head_failure_no_ft():
    """T4: Kill head pod WITHOUT GCS FT (control experiment)."""
    print("\n" + "=" * 60)
    print("T4: HEAD NODE FAILURE (NO GCS FT — CONTROL)")
    print("=" * 60)

    # Switch to no-FT deployment
    print("  Switching to no-FT RayService...")
    k8s_dir = Path(__file__).parent.parent / "k8s"

    # Delete FT service
    subprocess.run(
        ["kubectl", "-n", NAMESPACE, "delete", "rayservice", "yolo-ft", "--ignore-not-found"],
        check=False,
    )
    time.sleep(5)

    # Apply no-FT service
    subprocess.run(
        ["kubectl", "apply", "-f", str(k8s_dir / "ray-service-no-ft.yaml")],
        check=True,
    )

    print("  Waiting for no-FT service to be ready...")
    for i in range(60):
        try:
            status = kubectl("get", "rayservice", "yolo-no-ft", "-o",
                             "jsonpath={.status.serviceStatus}")
            if status == "Running":
                break
        except subprocess.CalledProcessError:
            pass
        time.sleep(10)

    head = get_head_pod()
    if not head:
        print("ERROR: No head pod found for no-FT service")
        return None

    proc, traffic_path = start_traffic("T4", duration=300)
    wait_with_countdown(60, "Warmup")

    print(f"  INJECTING FAULT: Force-deleting head pod {head}")
    subprocess.run(
        ["kubectl", "-n", NAMESPACE, "delete", "pod", head, "--force", "--grace-period=0"],
        check=False,
    )

    wait_with_countdown(180, "Observing recovery (full cluster restart expected)")

    proc.terminate()
    proc.wait(timeout=10)
    time.sleep(2)

    metrics = analyze_traffic(traffic_path)
    metrics["test"] = "T4"
    metrics["fault"] = f"kubectl delete pod {head} --force (no GCS FT)"
    metrics["gcs_ft"] = False

    # Restore FT service
    print("  Restoring FT RayService...")
    subprocess.run(
        ["kubectl", "-n", NAMESPACE, "delete", "rayservice", "yolo-no-ft", "--ignore-not-found"],
        check=False,
    )

    print(f"\n  Results: {json.dumps(metrics, indent=2)}")
    return metrics


def test_t5_http_proxy():
    """T5: Kill HTTP proxy on head node."""
    print("\n" + "=" * 60)
    print("T5: HTTP PROXY FAILOVER")
    print("=" * 60)

    head = get_head_pod()
    if not head:
        print("ERROR: No head pod found")
        return None

    proc, traffic_path = start_traffic("T5", duration=240)
    wait_with_countdown(60, "Warmup")

    print(f"  INJECTING FAULT: Killing HTTP proxy in head pod {head}")
    kubectl("exec", head, "-c", "ray-head", "--", "pkill", "-f", "ProxyActor", check=False)

    wait_with_countdown(120, "Observing proxy recovery")

    proc.terminate()
    proc.wait(timeout=10)
    time.sleep(2)

    metrics = analyze_traffic(traffic_path)
    metrics["test"] = "T5"
    metrics["fault"] = f"pkill ProxyActor in {head}"
    print(f"\n  Results: {json.dumps(metrics, indent=2)}")
    return metrics


def test_t6_redis_disruption():
    """T6: Block Redis connectivity temporarily."""
    print("\n" + "=" * 60)
    print("T6: ELASTICACHE CONNECTIVITY DISRUPTION")
    print("=" * 60)

    # Find the Redis security group
    print("  Finding Redis security group...")
    result = subprocess.run(
        ["aws", "ec2", "describe-security-groups",
         "--filters", "Name=group-name,Values=ray-ft-redis-*",
         "--query", "SecurityGroups[0].GroupId", "--output", "text",
         "--region", "us-west-2"],
        capture_output=True, text=True,
    )
    sg_id = result.stdout.strip()
    if not sg_id or sg_id == "None":
        print("ERROR: Could not find Redis security group")
        return None

    # Find the ingress rule to revoke/restore
    print(f"  Redis SG: {sg_id}")

    proc, traffic_path = start_traffic("T6", duration=240)
    wait_with_countdown(60, "Warmup")

    print(f"  INJECTING FAULT: Revoking Redis SG ingress (30s partition)")
    # Revoke all ingress (block Redis access)
    subprocess.run(
        ["aws", "ec2", "revoke-security-group-ingress",
         "--group-id", sg_id,
         "--protocol", "tcp", "--port", "6379",
         "--source-group", sg_id,  # Will be replaced with actual source SG
         "--region", "us-west-2"],
        check=False,
    )

    time.sleep(30)

    print("  RESTORING: Re-adding Redis SG ingress")
    # Get cluster SG to restore the rule
    cluster_sg = subprocess.run(
        ["aws", "eks", "describe-cluster", "--name", "qn-sglang-eks-cluster",
         "--query", "cluster.resourcesVpcConfig.clusterSecurityGroupId",
         "--output", "text", "--region", "us-west-2"],
        capture_output=True, text=True,
    ).stdout.strip()

    subprocess.run(
        ["aws", "ec2", "authorize-security-group-ingress",
         "--group-id", sg_id,
         "--protocol", "tcp", "--port", "6379",
         "--source-group", cluster_sg,
         "--region", "us-west-2"],
        check=False,
    )

    wait_with_countdown(90, "Observing reconnection")

    proc.terminate()
    proc.wait(timeout=10)
    time.sleep(2)

    metrics = analyze_traffic(traffic_path)
    metrics["test"] = "T6"
    metrics["fault"] = f"SG ingress revoke/restore on {sg_id} (30s)"
    print(f"\n  Results: {json.dumps(metrics, indent=2)}")
    return metrics


TESTS = {
    "T1": test_t1_replica_crash,
    "T2": test_t2_worker_drain,
    "T3": test_t3_head_failure_ft,
    "T4": test_t4_head_failure_no_ft,
    "T5": test_t5_http_proxy,
    "T6": test_t6_redis_disruption,
}


def main():
    parser = argparse.ArgumentParser(description="Ray Serve FT Fault Injection")
    parser.add_argument("--test", action="append", default=[],
                        help="Test to run (T1-T6 or 'all'). Can specify multiple.")
    parser.add_argument("--output", default=str(RESULTS_DIR / "ft_results.json"),
                        help="Summary output path")
    args = parser.parse_args()

    if not args.test:
        args.test = ["all"]

    tests_to_run = []
    if "all" in args.test:
        tests_to_run = list(TESTS.keys())
    else:
        tests_to_run = [t.upper() for t in args.test]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    print(f"Running tests: {', '.join(tests_to_run)}")
    print(f"Results directory: {RESULTS_DIR}")

    for test_name in tests_to_run:
        if test_name not in TESTS:
            print(f"Unknown test: {test_name}")
            continue
        result = TESTS[test_name]()
        if result:
            all_results.append(result)

    # Write summary
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary written to {args.output}")

    # Print comparison table
    if all_results:
        print(f"\n{'Test':<6} {'Success%':>9} {'Errors':>7} {'ErrWindow':>10} {'P50ms':>7} {'P99ms':>7}")
        print("-" * 50)
        for r in all_results:
            print(f"{r['test']:<6} {r['success_rate_pct']:>8.1f}% {r['errors']:>7} "
                  f"{r['error_window_s']:>9.1f}s {r['latency_p50_ms']:>7.0f} {r['latency_p99_ms']:>7.0f}")


if __name__ == "__main__":
    main()
