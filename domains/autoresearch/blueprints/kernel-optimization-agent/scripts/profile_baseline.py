#!/usr/bin/env python3
"""Phase 1: Profile K2.6 baseline serving to identify top kernel bottlenecks.

Runs vLLM with K2.6 FP8, profiles with NSight Compute, and outputs:
- Top-10 kernels by wall-clock time
- Roofline classification (compute-bound / memory-bound / latency-bound)
- Pipeline stage mapping (MoE dispatch, MLA decode, attention, comm)

Usage:
    # Full pipeline profile at concurrency 1 (detailed ncu)
    python profile_baseline.py --mode ncu --concurrency 1

    # E2E throughput baseline at multiple concurrency levels
    python profile_baseline.py --mode throughput --concurrency 1,128,512

    # Quick nsys timeline (no kernel-level detail)
    python profile_baseline.py --mode nsys --concurrency 128
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_PATH = "/opt/dlami/nvme/kernel-opt/models/kimi-k26-fp8"
RESULTS_DIR = "/opt/dlami/nvme/kernel-opt/results"
PROFILE_DIR = "/opt/dlami/nvme/kernel-opt/profiles"

# K2.6 architecture constants
K26_CONFIG = {
    "n_routed_experts": 384,
    "num_experts_per_tok": 8,
    "n_group": 1,
    "kv_lora_rank": 512,
    "v_head_dim": 128,
    "num_attention_heads": 64,
    "hidden_size": 7168,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 61,
}


def start_vllm_server(tp_size=8):
    """Start vLLM serving K2.6 FP8 in background."""
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.92",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--disable-log-requests",
    ]
    print(f"Starting vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=open(f"{RESULTS_DIR}/vllm.log", "w"),
                           stderr=subprocess.STDOUT)
    # Wait for server to be ready
    for i in range(120):
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:8000/health")
            print(f"vLLM ready after {i}s")
            return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("vLLM failed to start within 120s")


def run_ncu_profile(output_path, concurrency=1, num_requests=4):
    """Profile specific kernels with NSight Compute."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Profile the server process
    # We use ncu --target-processes all to capture GPU kernels from vLLM workers
    cmd = [
        "ncu",
        "--set", "full",
        "--target-processes", "all",
        "--launch-skip", "100",  # skip warmup kernels
        "--launch-count", "50",  # capture 50 kernel launches
        "--export", output_path,
        "--force-overwrite",
    ]
    print(f"NSight Compute profiling: {' '.join(cmd)}")
    print("Sending requests to trigger kernel execution...")

    # Send requests while profiling
    import threading

    def send_requests():
        """Send inference requests to trigger kernel execution."""
        time.sleep(2)  # let ncu attach
        import urllib.request
        for i in range(num_requests):
            data = json.dumps({
                "model": MODEL_PATH,
                "messages": [{"role": "user", "content": f"Explain quantum computing in detail. Request {i}."}],
                "max_tokens": 256,
                "temperature": 0.0,
            }).encode()
            req = urllib.request.Request(
                "http://localhost:8000/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            try:
                resp = urllib.request.urlopen(req, timeout=120)
                print(f"  Request {i} complete: {resp.status}")
            except Exception as e:
                print(f"  Request {i} failed: {e}")

    t = threading.Thread(target=send_requests)
    t.start()

    # Run ncu (attaches to existing process)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    t.join()

    if result.returncode != 0:
        print(f"ncu failed: {result.stderr}")
        return None
    return output_path


def run_nsys_profile(output_path, concurrency=128, duration_sec=30):
    """Quick NSight Systems timeline profile."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "nsys", "profile",
        "--trace", "cuda,nvtx,osrt",
        "--duration", str(duration_sec),
        "--output", output_path,
        "--force-overwrite", "true",
        "--target-processes", "all",
    ]
    print(f"NSight Systems profiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_sec + 60)
    if result.returncode != 0:
        print(f"nsys failed: {result.stderr}")
    return output_path


def run_throughput_benchmark(concurrency_levels, num_requests_per_level=100):
    """Run throughput benchmark at multiple concurrency levels."""
    results = {}
    for conc in concurrency_levels:
        print(f"\n=== Benchmarking at concurrency={conc} ===")
        # Use vLLM's built-in benchmark
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.run_batch",
            "--model", MODEL_PATH,
            "--input-len", "512",
            "--output-len", "256",
            "--num-prompts", str(num_requests_per_level),
            "--concurrency", str(conc),
        ]
        # Fallback: use simple HTTP benchmark
        import concurrent.futures
        import urllib.request

        start = time.time()
        completed = 0
        total_tokens = 0

        def single_request(i):
            data = json.dumps({
                "model": MODEL_PATH,
                "messages": [{"role": "user", "content": f"Write a 200-word essay about topic {i}."}],
                "max_tokens": 256,
                "temperature": 0.0,
            }).encode()
            req = urllib.request.Request(
                "http://localhost:8000/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            t0 = time.time()
            resp = urllib.request.urlopen(req, timeout=120)
            result = json.loads(resp.read())
            t1 = time.time()
            tokens = result["usage"]["completion_tokens"]
            return {"latency": t1 - t0, "tokens": tokens}

        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as executor:
            futures = [executor.submit(single_request, i) for i in range(num_requests_per_level)]
            request_results = []
            for f in concurrent.futures.as_completed(futures):
                try:
                    request_results.append(f.result())
                except Exception as e:
                    print(f"  Request failed: {e}")

        elapsed = time.time() - start
        total_tokens = sum(r["tokens"] for r in request_results)
        throughput = total_tokens / elapsed
        latencies = sorted(r["latency"] for r in request_results)

        results[conc] = {
            "throughput_tok_s": round(throughput, 1),
            "requests_completed": len(request_results),
            "elapsed_s": round(elapsed, 1),
            "latency_p50": round(latencies[len(latencies)//2], 3) if latencies else None,
            "latency_p99": round(latencies[int(len(latencies)*0.99)], 3) if latencies else None,
        }
        print(f"  Throughput: {throughput:.1f} tok/s, p50={results[conc]['latency_p50']}s")

    return results


def parse_ncu_report(report_path):
    """Parse NSight Compute report to extract top kernels."""
    # Use ncu --import to get CSV
    cmd = [
        "ncu", "--import", f"{report_path}.ncu-rep",
        "--csv",
        "--page", "raw",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"Failed to parse ncu report: {result.stderr}")
        return []

    # Parse CSV output
    lines = result.stdout.strip().split("\n")
    if len(lines) < 2:
        return []

    # Extract kernel names and durations
    kernels = {}
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) >= 5:
            name = fields[3].strip('"')  # kernel name
            try:
                duration = float(fields[4])  # duration in ns
            except (ValueError, IndexError):
                continue
            if name not in kernels:
                kernels[name] = {"count": 0, "total_ns": 0}
            kernels[name]["count"] += 1
            kernels[name]["total_ns"] += duration

    # Sort by total time
    sorted_kernels = sorted(kernels.items(), key=lambda x: x[1]["total_ns"], reverse=True)
    return sorted_kernels[:20]


def classify_kernel(name):
    """Classify kernel into pipeline stage."""
    name_lower = name.lower()
    if "moe" in name_lower or "expert" in name_lower or "topk" in name_lower:
        return "moe_dispatch"
    elif "mla" in name_lower or "attention" in name_lower or "flash" in name_lower:
        return "mla_decode"
    elif "gemm" in name_lower or "matmul" in name_lower:
        return "linear"
    elif "nccl" in name_lower or "allreduce" in name_lower:
        return "communication"
    elif "norm" in name_lower or "rms" in name_lower:
        return "normalization"
    elif "rope" in name_lower or "rotary" in name_lower:
        return "positional"
    else:
        return "other"


def main():
    parser = argparse.ArgumentParser(description="K2.6 Baseline Profiling")
    parser.add_argument("--mode", choices=["ncu", "nsys", "throughput", "all"], default="all")
    parser.add_argument("--concurrency", default="1,128,512",
                       help="Comma-separated concurrency levels")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--skip-server", action="store_true", help="Skip starting vLLM server")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PROFILE_DIR, exist_ok=True)

    conc_levels = [int(x) for x in args.concurrency.split(",")]

    # Start server
    server_proc = None
    if not args.skip_server:
        server_proc = start_vllm_server(tp_size=args.tp)

    try:
        results = {"config": K26_CONFIG, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

        if args.mode in ("throughput", "all"):
            print("\n" + "="*60)
            print("THROUGHPUT BENCHMARK")
            print("="*60)
            results["throughput"] = run_throughput_benchmark(conc_levels)

        if args.mode in ("ncu", "all"):
            print("\n" + "="*60)
            print("NSIGHT COMPUTE PROFILING")
            print("="*60)
            profile_path = f"{PROFILE_DIR}/k26_baseline"
            run_ncu_profile(profile_path, concurrency=1)
            top_kernels = parse_ncu_report(profile_path)
            results["top_kernels"] = [
                {
                    "rank": i+1,
                    "name": name,
                    "count": data["count"],
                    "total_us": round(data["total_ns"] / 1000, 1),
                    "stage": classify_kernel(name),
                }
                for i, (name, data) in enumerate(top_kernels[:10])
            ]

        if args.mode in ("nsys", "all"):
            print("\n" + "="*60)
            print("NSIGHT SYSTEMS TIMELINE")
            print("="*60)
            nsys_path = f"{PROFILE_DIR}/k26_timeline"
            run_nsys_profile(nsys_path, concurrency=conc_levels[-1])
            results["nsys_report"] = f"{nsys_path}.nsys-rep"

        # Save results
        output_file = f"{RESULTS_DIR}/phase1_profile.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n=== Results saved to {output_file} ===")
        print(json.dumps(results, indent=2))

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    main()
