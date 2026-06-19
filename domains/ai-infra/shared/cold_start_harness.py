"""Cold-start measurement harness for ai-infra experiments.

Measures pod-create → first-token-streamed time across load-format variants
and storage backends. Output is one JSON artifact per run, conforming to
benchmark-commons enriched-artifact format.

Usage:
    python cold_start_harness.py \\
        --blueprint domains/gpu-serving/blueprints/glm5-fp8/ \\
        --variant load_format=runai_streamer \\
        --storage s3 \\
        --runs 5 \\
        --out results/glm5-s3-runai.json

Assumes:
- kubectl available, KUBECONFIG points at target cluster
- vLLM container image already pushed; image tag controlled via --image
- Model URI resolvable from the storage backend (S3 prefix, FSx mount, NVMe path)
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    run_id: str
    started_at: float
    pod_ready_at: float | None = None
    first_token_at: float | None = None
    pod_create_to_ready_s: float | None = None
    pod_create_to_first_token_s: float | None = None
    pod_ready_to_first_token_s: float | None = None
    error: str | None = None


@dataclass
class Artifact:
    schema: str = "benchmark-commons/v1"
    experiment: str = ""
    blueprint_fixture: str = ""
    variant: dict[str, str] = field(default_factory=dict)
    storage_backend: str = ""
    hardware: dict[str, str] = field(default_factory=dict)
    model: dict[str, str] = field(default_factory=dict)
    runs: list[RunResult] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)


def kubectl(*args: str, timeout: int = 30) -> str:
    return subprocess.check_output(["kubectl", *args], text=True, timeout=timeout)


def deploy_pod(manifest_path: Path, namespace: str) -> str:
    """Apply manifest, return pod name."""
    out = kubectl("apply", "-n", namespace, "-f", str(manifest_path))
    # parse "pod/<name> created" — caller must ensure manifest declares one Pod
    for line in out.splitlines():
        if line.startswith("pod/"):
            return line.split()[0].removeprefix("pod/")
    raise RuntimeError(f"could not parse pod name from: {out}")


def wait_pod_ready(pod: str, namespace: str, timeout: int = 1800) -> float:
    """Block until pod is Ready. Return wall-clock seconds elapsed."""
    start = time.monotonic()
    deadline = start + timeout
    while time.monotonic() < deadline:
        try:
            phase = kubectl(
                "get", "pod", pod, "-n", namespace,
                "-o", "jsonpath={.status.phase}",
            ).strip()
            ready = kubectl(
                "get", "pod", pod, "-n", namespace,
                "-o",
                "jsonpath={.status.conditions[?(@.type=='Ready')].status}",
            ).strip()
            if phase == "Running" and ready == "True":
                return time.monotonic() - start
            if phase in ("Failed", "Unknown"):
                raise RuntimeError(f"pod {pod} entered phase {phase}")
        except subprocess.CalledProcessError:
            pass
        time.sleep(2)
    raise TimeoutError(f"pod {pod} not ready within {timeout}s")


def time_first_token(endpoint: str, model: str, prompt: str = "Hello") -> float:
    """Send a streaming completion, return seconds to first SSE chunk with content."""
    import urllib.request

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": 8,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line.removeprefix("data:").strip()
            if body == "[DONE]" or not body:
                continue
            chunk = json.loads(body)
            text = chunk.get("choices", [{}])[0].get("text", "")
            if text:
                return time.monotonic() - start
    raise RuntimeError("stream ended with no token")


def delete_pod(pod: str, namespace: str) -> None:
    subprocess.run(
        ["kubectl", "delete", "pod", pod, "-n", namespace, "--wait=true"],
        check=False,
        timeout=120,
    )


def run_one(args: argparse.Namespace) -> RunResult:
    rid = str(uuid.uuid4())[:8]
    result = RunResult(run_id=rid, started_at=time.time())
    pod = None
    try:
        t_create = time.monotonic()
        pod = deploy_pod(Path(args.manifest), args.namespace)
        ready_elapsed = wait_pod_ready(pod, args.namespace, timeout=args.timeout)
        t_ready = time.monotonic()
        result.pod_ready_at = result.started_at + ready_elapsed
        result.pod_create_to_ready_s = ready_elapsed

        ft_elapsed = time_first_token(args.endpoint, args.model)
        t_ft = time.monotonic()
        result.first_token_at = result.started_at + (t_ft - t_create)
        result.pod_ready_to_first_token_s = ft_elapsed
        result.pod_create_to_first_token_s = (t_ft - t_create)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        if pod and not args.keep_pod:
            delete_pod(pod, args.namespace)
    return result


def summarize(runs: list[RunResult]) -> dict[str, float]:
    valid = [r.pod_create_to_first_token_s for r in runs if r.pod_create_to_first_token_s]
    if not valid:
        return {}
    return {
        "n": len(valid),
        "median_s": statistics.median(valid),
        "p10_s": statistics.quantiles(valid, n=10)[0] if len(valid) >= 10 else min(valid),
        "p90_s": statistics.quantiles(valid, n=10)[-1] if len(valid) >= 10 else max(valid),
        "min_s": min(valid),
        "max_s": max(valid),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="K8s pod manifest with the variant baked in")
    p.add_argument("--namespace", default="ai-infra")
    p.add_argument("--endpoint", required=True, help="vLLM service endpoint, e.g. http://glm5.svc:8000")
    p.add_argument("--model", required=True, help="Model id to send to /v1/completions")
    p.add_argument("--variant", action="append", default=[], help="key=value, repeatable")
    p.add_argument("--storage", required=True, choices=["s3", "fsx", "nvme", "hbm-p2p"])
    p.add_argument("--hardware", required=True, help="instance type, e.g. p6-b300.48xlarge")
    p.add_argument("--blueprint", required=True, help="path to gpu-serving blueprint used as fixture")
    p.add_argument("--experiment", required=True, help="experiment name, e.g. runai-streamer-cold-start")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--keep-pod", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    artifact = Artifact(
        experiment=args.experiment,
        blueprint_fixture=args.blueprint,
        variant=dict(kv.split("=", 1) for kv in args.variant),
        storage_backend=args.storage,
        hardware={"instance_type": args.hardware},
        model={"id": args.model},
    )

    for i in range(args.runs):
        print(f"[run {i + 1}/{args.runs}] starting", file=sys.stderr)
        r = run_one(args)
        artifact.runs.append(r)
        print(f"[run {i + 1}] {r.error or f'{r.pod_create_to_first_token_s:.1f}s'}", file=sys.stderr)

    artifact.summary = summarize(artifact.runs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(artifact), indent=2, default=str))
    print(f"wrote {out} — median {artifact.summary.get('median_s', 'n/a')}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
