#!/usr/bin/env python3
"""Gen0 OpenHands re-baseline.

Load the Qwen3.5-27B base + the LoRA adapter (SFT-D), serve via vLLM, point
OpenHands v0.54 at it, run SWE-bench Lite 300, Docker-eval every patch, emit
summary.json with gold_pass_rate. This is the "must beat" number for Arm A.

Architecture:
  1. Start vLLM OpenAI-compatible server on localhost:8000 with base+LoRA.
  2. For each of 300 instances: spawn OpenHands container pointing at vLLM.
  3. Collect patches, run Docker gold eval (local p4de or kicks off m7i).
  4. Tabulate pass/fail, write summary.json.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def start_vllm(base_model: str, adapter: str, port: int = 8000) -> subprocess.Popen:
    """Launch vLLM server with base + LoRA adapter.

    Uses --enable-lora + --lora-modules gen0=<adapter_path>.
    Returns the process handle; caller must terminate.
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--tensor-parallel-size", "8",
        "--max-model-len", "65536",
        "--enable-lora",
        "--lora-modules", f"gen0={adapter}",
        "--port", str(port),
        "--host", "0.0.0.0",
    ]
    print(f"launching vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    # Wait for ready
    import urllib.request, urllib.error
    for i in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=3)
            print(f"vllm ready after {i*5}s")
            return proc
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(5)
    proc.terminate()
    raise RuntimeError("vllm did not start within 5 minutes")


def run_openhands_batch(instances: list[dict], openhands_image: str, vllm_url: str,
                       output_dir: Path, max_concurrency: int = 4) -> list[dict]:
    """Spawn OpenHands containers per instance and collect patches.

    This is a thin wrapper — the real implementation would use OpenHands's
    official batch runner (openhands.sh or the Python CLI). For now, shell out.
    """
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    # The OpenHands batch runner expects a predictions.jsonl-style input.
    # See: https://github.com/All-Hands-AI/OpenHands/blob/main/evaluation/swe_bench/README.md
    # For this script, assume a helper `openhands-swebench` CLI is on PATH inside the
    # container. Minimal invocation sketched below.
    preds_file = output_dir / "openhands_predictions.jsonl"
    cmd = [
        "docker", "run", "--rm", "--network=host",
        "-v", f"{output_dir}:/workspace",
        "-e", f"LLM_BASE_URL={vllm_url}",
        "-e", "LLM_MODEL=gen0",
        "-e", "LLM_API_KEY=dummy",
        openhands_image,
        "python", "-m", "openhands.core.main",
        "--agent", "CodeActAgent",
        "--config", "/workspace/openhands_config.toml",
        "--eval-dataset", "princeton-nlp/SWE-bench_Lite",
        "--eval-output", f"/workspace/{preds_file.name}",
        "--eval-n-limit", str(len(instances)),
        "--eval-max-iterations", "30",
    ]
    print(f"openhands batch: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)

    # Parse predictions
    if preds_file.exists():
        with open(preds_file) as f:
            for line in f:
                results.append(json.loads(line))
    return results


def run_docker_gold_eval(predictions: list[dict], output_dir: Path) -> dict:
    """Run SWE-bench's official Docker gold evaluator on the predictions."""
    preds_path = output_dir / "predictions_for_eval.jsonl"
    with open(preds_path, "w") as f:
        for p in predictions:
            f.write(json.dumps({
                "instance_id": p["instance_id"],
                "model_patch": p.get("model_patch", ""),
                "model_name_or_path": "qwen35-27b-gen0-openhands",
            }) + "\n")

    cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "princeton-nlp/SWE-bench_Lite",
        "--predictions_path", str(preds_path),
        "--max_workers", "16",
        "--run_id", "gen0_rebaseline",
        "--namespace", "swebench",
    ]
    print(f"docker gold eval: {' '.join(cmd)}")
    subprocess.run(cmd, check=False, cwd=output_dir)

    report_files = list(output_dir.glob("*.gen0_rebaseline.json"))
    if not report_files:
        return {"error": "no eval report produced"}
    with open(report_files[0]) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--harness", default="openhands")
    p.add_argument("--harness-version", default="v0.54")
    p.add_argument("--dataset", default="swebench-lite-300")
    p.add_argument("--openhands-image", default="ghcr.io/all-hands-ai/openhands:0.54.0")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load instance list
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    instances = list(ds)
    print(f"loaded {len(instances)} instances from SWE-bench Lite")

    vllm = start_vllm(args.base_model, args.adapter)
    try:
        predictions = run_openhands_batch(
            instances, args.openhands_image, "http://localhost:8000/v1", out
        )
        print(f"got {len(predictions)} predictions")
        eval_report = run_docker_gold_eval(predictions, out)
    finally:
        vllm.send_signal(signal.SIGINT)
        try:
            vllm.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm.kill()

    resolved = eval_report.get("resolved_ids", []) if isinstance(eval_report, dict) else []
    n = len(predictions)
    summary = {
        "dataset": args.dataset,
        "harness": args.harness,
        "harness_version": args.harness_version,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "n_instances": n,
        "n_resolved": len(resolved),
        "gold_pass_rate": len(resolved) / n if n else 0,
        "eval_report_path": str(out / "eval_report.json"),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if isinstance(eval_report, dict):
        with open(out / "eval_report.json", "w") as f:
            json.dump(eval_report, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
