#!/usr/bin/env python3
"""
Benchmark: Config A (in-memory) vs Config B (S3 passthrough)

Runs both pipeline variants, collects per-stage latency from ResultWriter logs,
and produces a comparison report.

Usage (from local machine with kubectl access):
    python benchmark.py --num-messages 50
"""

import argparse
import json
import logging
import re
import subprocess
import statistics
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

NAMESPACE = "ray-video"
BLUEPRINT_DIR = "domains/gpu-serving/blueprints/ray-serve-video"


def kubectl(*args, capture=True):
    cmd = ["kubectl"] + list(args)
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.stdout.strip(), result.returncode
    else:
        subprocess.run(cmd, timeout=300)
        return "", 0


def wait_for_rayservice_running(timeout=600):
    """Wait until the video RayService application status is RUNNING."""
    logger.info("Waiting for RayService to be RUNNING...")
    for i in range(timeout // 10):
        out, rc = kubectl(
            "get", "rayservice", "video-pipeline", "-n", NAMESPACE,
            "-o", "jsonpath={.status.activeServiceStatus.applicationStatuses.video.status}"
        )
        if out == "RUNNING":
            logger.info(f"  RayService RUNNING after {(i+1)*10}s")
            return True
        time.sleep(10)
    raise TimeoutError(f"RayService not RUNNING after {timeout}s")


def swap_pipeline(pipeline_file: str):
    """Update the ConfigMap with a different pipeline file and restart."""
    logger.info(f"Swapping pipeline to {pipeline_file}...")
    script_dir = f"{BLUEPRINT_DIR}/scripts"

    # Update ConfigMap
    kubectl(
        "create", "configmap", "video-pipeline-app",
        f"--from-file=video_pipeline.py={script_dir}/{pipeline_file}",
        "-n", NAMESPACE, "--dry-run=client", "-o", "yaml",
    )
    # Apply via pipe
    cmd = (
        f"kubectl create configmap video-pipeline-app "
        f"--from-file=video_pipeline.py={script_dir}/{pipeline_file} "
        f"-n {NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -"
    )
    subprocess.run(cmd, shell=True, timeout=30)

    # Restart the RayService by patching an annotation
    ts = str(int(time.time()))
    kubectl(
        "annotate", "rayservice", "video-pipeline",
        "-n", NAMESPACE, f"benchmark-restart={ts}", "--overwrite"
    )

    # Wait for pods to restart and service to be running
    logger.info("  Waiting 30s for rollout to begin...")
    time.sleep(30)
    wait_for_rayservice_running(timeout=600)
    logger.info("  Pipeline swapped and running")


def produce_messages(num_messages: int, delay: float = 0.5):
    """Run produce_test.py inside the head pod."""
    head_pod, _ = kubectl(
        "get", "pods", "-n", NAMESPACE, "-l", "ray-node=head",
        "-o", "jsonpath={.items[0].metadata.name}"
    )
    if not head_pod:
        raise RuntimeError("No head pod found")

    logger.info(f"Producing {num_messages} messages via {head_pod}...")

    # Copy producer script to head pod
    kubectl(
        "cp", f"{BLUEPRINT_DIR}/scripts/produce_test.py",
        f"{NAMESPACE}/{head_pod}:/tmp/produce_test.py", "-c", "ray-head"
    )

    # Run producer (kafka-python is in runtime_env)
    kubectl(
        "exec", "-n", NAMESPACE, head_pod, "-c", "ray-head", "--",
        "pip", "install", "-q", "kafka-python"
    )
    kubectl(
        "exec", "-n", NAMESPACE, head_pod, "-c", "ray-head", "--",
        "python", "/tmp/produce_test.py",
        "--num-messages", str(num_messages),
        "--delay", str(delay),
        capture=False
    )


def collect_results(since_seconds: int = 600) -> list:
    """Parse RESULT| lines from worker pod logs."""
    results = []
    pods_out, _ = kubectl(
        "get", "pods", "-n", NAMESPACE, "-l", "ray-node=worker",
        "-o", "jsonpath={.items[*].metadata.name}"
    )
    if not pods_out:
        # Also check head pod (ResultWriter may run there)
        pods_out, _ = kubectl(
            "get", "pods", "-n", NAMESPACE, "-l", "ray-node=head",
            "-o", "jsonpath={.items[*].metadata.name}"
        )

    for pod in pods_out.split():
        for container in ["ray-worker", "ray-head"]:
            logs, rc = kubectl(
                "logs", "-n", NAMESPACE, pod, "-c", container,
                f"--since={since_seconds}s"
            )
            if rc != 0:
                continue
            for line in logs.split("\n"):
                match = re.search(r"RESULT\|(.+)$", line)
                if match:
                    try:
                        results.append(json.loads(match.group(1)))
                    except json.JSONDecodeError:
                        pass
    return results


def analyze_results(results: list, config_name: str) -> dict:
    """Compute latency statistics from result records."""
    if not results:
        return {"config": config_name, "count": 0, "error": "no results"}

    e2e_latencies = []
    decode_latencies = []
    pt_latencies = []
    tf_latencies = []
    pt_to_tf_latencies = []
    s3_write_durations = []
    s3_read_durations = []

    for r in results:
        t_pub = r.get("t_kafka_publish", 0)
        t_result = r.get("t_result_written", 0)
        if t_pub and t_result:
            e2e_latencies.append(t_result - t_pub)

        t_dec_s = r.get("t_decode_start", 0)
        t_dec_e = r.get("t_decode_end", 0)
        if t_dec_s and t_dec_e:
            decode_latencies.append(t_dec_e - t_dec_s)

        t_pt_s = r.get("t_pt_start", 0)
        t_pt_e = r.get("t_pt_end", 0)
        if t_pt_s and t_pt_e:
            pt_latencies.append(t_pt_e - t_pt_s)

        t_tf_s = r.get("t_tf_start", 0)
        t_tf_e = r.get("t_tf_end", 0)
        if t_tf_s and t_tf_e:
            tf_latencies.append(t_tf_e - t_tf_s)

        t_pt_e2 = r.get("t_pt_end", 0)
        t_tf_s2 = r.get("t_tf_start", 0)
        if t_pt_e2 and t_tf_s2:
            pt_to_tf_latencies.append(t_tf_s2 - t_pt_e2)

        # S3 overhead (Config B only)
        if "t_s3_write_duration" in r:
            s3_write_durations.append(r["t_s3_write_duration"])
        if "t_s3_read_duration" in r:
            s3_read_durations.append(r["t_s3_read_duration"])

    def stats(values):
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "p50": round(statistics.median(values) * 1000, 1),
            "p95": round(sorted(values)[int(len(values) * 0.95)] * 1000, 1) if len(values) >= 2 else None,
            "p99": round(sorted(values)[int(len(values) * 0.99)] * 1000, 1) if len(values) >= 2 else None,
            "mean": round(statistics.mean(values) * 1000, 1),
            "min": round(min(values) * 1000, 1),
            "max": round(max(values) * 1000, 1),
        }

    report = {
        "config": config_name,
        "message_count": len(results),
        "e2e_latency_ms": stats(e2e_latencies),
        "decode_latency_ms": stats(decode_latencies),
        "pt_inference_ms": stats(pt_latencies),
        "tf_inference_ms": stats(tf_latencies),
        "pt_to_tf_handoff_ms": stats(pt_to_tf_latencies),
    }

    if s3_write_durations:
        report["s3_write_ms"] = stats(s3_write_durations)
    if s3_read_durations:
        report["s3_read_ms"] = stats(s3_read_durations)

    return report


def print_comparison(report_a: dict, report_b: dict):
    """Print side-by-side comparison."""
    print("\n" + "=" * 70)
    print(" BENCHMARK RESULTS: Config A (in-memory) vs Config B (S3 passthrough)")
    print("=" * 70)

    def fmt(report, key):
        data = report.get(key, {})
        if not data or data.get("count", 0) == 0:
            return "N/A"
        return f"p50={data['p50']}ms  p95={data.get('p95', 'N/A')}ms  mean={data['mean']}ms"

    rows = [
        ("E2E Latency", "e2e_latency_ms"),
        ("Decode (S3→numpy)", "decode_latency_ms"),
        ("PT Inference", "pt_inference_ms"),
        ("TF Inference", "tf_inference_ms"),
        ("PT→TF Handoff", "pt_to_tf_handoff_ms"),
    ]

    for label, key in rows:
        print(f"\n{label}:")
        print(f"  Config A (in-memory):      {fmt(report_a, key)}")
        print(f"  Config B (S3 passthrough): {fmt(report_b, key)}")

    if "s3_write_ms" in report_b:
        print(f"\nS3 Write (PT→S3, Config B only):")
        print(f"  {fmt(report_b, 's3_write_ms')}")
    if "s3_read_ms" in report_b:
        print(f"S3 Read (S3→TF, Config B only):")
        print(f"  {fmt(report_b, 's3_read_ms')}")

    # Compute speedup
    a_e2e = report_a.get("e2e_latency_ms", {}).get("p50")
    b_e2e = report_b.get("e2e_latency_ms", {}).get("p50")
    if a_e2e and b_e2e and a_e2e > 0:
        speedup = b_e2e / a_e2e
        print(f"\n{'=' * 70}")
        print(f" SPEEDUP: Config A is {speedup:.2f}x faster (p50 E2E)")
        a_handoff = report_a.get("pt_to_tf_handoff_ms", {}).get("p50", 0)
        b_handoff = report_b.get("pt_to_tf_handoff_ms", {}).get("p50", 0)
        if b_handoff:
            print(f" PT→TF handoff: {a_handoff}ms (in-memory) vs {b_handoff}ms (S3)")
            print(f" S3 overhead per frame: ~{b_handoff - a_handoff:.0f}ms")
        print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark in-memory vs S3 passthrough")
    parser.add_argument("--num-messages", type=int, default=20)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between messages")
    parser.add_argument("--skip-config-a", action="store_true", help="Skip Config A (already deployed)")
    parser.add_argument("--skip-config-b", action="store_true")
    args = parser.parse_args()

    report_a = {}
    report_b = {}

    # --- Config A: In-memory ---
    if not args.skip_config_a:
        logger.info("=" * 50)
        logger.info("CONFIG A: In-memory (video_pipeline.py)")
        logger.info("=" * 50)

        # Ensure Config A is deployed
        swap_pipeline("video_pipeline.py")
        time.sleep(10)

        # Reset Kafka consumer group offset by restarting
        produce_messages(args.num_messages, delay=args.delay)

        logger.info("Waiting 30s for pipeline to process all messages...")
        time.sleep(30)

        results_a = collect_results(since_seconds=600)
        # Filter to Config A results (no s3_write_duration field)
        results_a = [r for r in results_a if "t_s3_write_duration" not in r]
        report_a = analyze_results(results_a, "A_in_memory")
        logger.info(f"Config A: collected {len(results_a)} results")

    # --- Config B: S3 passthrough ---
    if not args.skip_config_b:
        logger.info("")
        logger.info("=" * 50)
        logger.info("CONFIG B: S3 passthrough (video_pipeline_s3.py)")
        logger.info("=" * 50)

        swap_pipeline("video_pipeline_s3.py")

        # Ensure intermediate bucket exists
        head_pod, _ = kubectl(
            "get", "pods", "-n", NAMESPACE, "-l", "ray-node=head",
            "-o", "jsonpath={.items[0].metadata.name}"
        )
        # The S3 intermediate bucket will be created by the pipeline or needs to exist

        time.sleep(10)
        produce_messages(args.num_messages, delay=args.delay)

        logger.info("Waiting 30s for pipeline to process all messages...")
        time.sleep(30)

        results_b = collect_results(since_seconds=600)
        results_b = [r for r in results_b if r.get("config") == "B_s3_passthrough" or "t_s3_write_duration" in r]
        report_b = analyze_results(results_b, "B_s3_passthrough")
        logger.info(f"Config B: collected {len(results_b)} results")

    # --- Comparison ---
    if report_a and report_b:
        print_comparison(report_a, report_b)

        # Save raw reports
        output = {
            "config_a": report_a,
            "config_b": report_b,
            "timestamp": time.time(),
        }
        output_path = f"{BLUEPRINT_DIR}/results/benchmark_{int(time.time())}.json"
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    elif report_a:
        print(json.dumps(report_a, indent=2))
    elif report_b:
        print(json.dumps(report_b, indent=2))


if __name__ == "__main__":
    main()
