"""Weight-sync measurement harness for RL actor-learner experiments.

Measures wall-clock time to propagate a fresh weight set from a learner pod
to N rollout pods, comparing baseline (each pod re-pulls from S3) against
ModelExpress P2P (`--load-format mx` with broadcast from learner HBM).

Hooks into vLLM's `update_weights_from_distributed` API on rollout side,
and times from `learner.publish()` to last rollout reporting `weights_ready`.

Usage:
    python weight_sync_harness.py \\
        --learner-pod learner-0 \\
        --rollout-pods 'rollout-{0..15}' \\
        --variant baseline   # or modelexpress
        --iterations 20 \\
        --out results/rl-sync-baseline.json
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class IterResult:
    iter_id: str
    publish_at: float
    last_ready_at: float | None = None
    sync_wall_s: float | None = None
    per_pod_ready_s: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Artifact:
    schema: str = "benchmark-commons/v1"
    experiment: str = "modelexpress-rl-weight-sync"
    variant: str = ""
    n_rollouts: int = 0
    model_size_gb: float = 0.0
    fabric: str = ""
    iterations: list[IterResult] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)


def expand_pod_range(spec: str) -> list[str]:
    """rollout-{0..15} -> [rollout-0, ..., rollout-15]"""
    if "{" not in spec:
        return [spec]
    prefix, rest = spec.split("{", 1)
    rng, suffix = rest.split("}", 1)
    lo, hi = (int(x) for x in rng.split(".."))
    return [f"{prefix}{i}{suffix}" for i in range(lo, hi + 1)]


def trigger_publish(learner_pod: str, namespace: str) -> float:
    """Tell the learner to publish current weights. Returns wall-clock t0."""
    t0 = time.monotonic()
    subprocess.run(
        ["kubectl", "exec", "-n", namespace, learner_pod, "--",
         "curl", "-sfX", "POST", "http://localhost:8000/admin/publish_weights"],
        check=True, timeout=30,
    )
    return t0


def poll_ready(pod: str, namespace: str, t0: float, deadline: float) -> float | None:
    """Poll a rollout pod until its /admin/weights_version increments. Return seconds since t0."""
    initial = subprocess.check_output(
        ["kubectl", "exec", "-n", namespace, pod, "--",
         "curl", "-sf", "http://localhost:8000/admin/weights_version"],
        text=True, timeout=10,
    ).strip()
    while time.monotonic() < deadline:
        try:
            cur = subprocess.check_output(
                ["kubectl", "exec", "-n", namespace, pod, "--",
                 "curl", "-sf", "http://localhost:8000/admin/weights_version"],
                text=True, timeout=10,
            ).strip()
            if cur != initial:
                return time.monotonic() - t0
        except subprocess.CalledProcessError:
            pass
        time.sleep(0.5)
    return None


def run_iteration(args: argparse.Namespace, rollouts: list[str]) -> IterResult:
    iid = str(uuid.uuid4())[:8]
    t0 = trigger_publish(args.learner_pod, args.namespace)
    result = IterResult(iter_id=iid, publish_at=time.time())
    deadline = time.monotonic() + args.timeout
    try:
        for pod in rollouts:
            elapsed = poll_ready(pod, args.namespace, t0, deadline)
            if elapsed is None:
                raise TimeoutError(f"{pod} did not roll over")
            result.per_pod_ready_s[pod] = elapsed
        result.last_ready_at = result.publish_at + max(result.per_pod_ready_s.values())
        result.sync_wall_s = max(result.per_pod_ready_s.values())
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def summarize(iters: list[IterResult]) -> dict[str, float]:
    valid = [i.sync_wall_s for i in iters if i.sync_wall_s]
    if not valid:
        return {}
    return {
        "n": len(valid),
        "median_s": statistics.median(valid),
        "p90_s": statistics.quantiles(valid, n=10)[-1] if len(valid) >= 10 else max(valid),
        "min_s": min(valid),
        "max_s": max(valid),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--learner-pod", required=True)
    p.add_argument("--rollout-pods", required=True, help="e.g. rollout-{0..15}")
    p.add_argument("--namespace", default="rl")
    p.add_argument("--variant", required=True, choices=["baseline", "modelexpress"])
    p.add_argument("--model-size-gb", type=float, required=True)
    p.add_argument("--fabric", required=True, choices=["efa", "ib", "tcp"])
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rollouts = expand_pod_range(args.rollout_pods)
    artifact = Artifact(
        variant=args.variant,
        n_rollouts=len(rollouts),
        model_size_gb=args.model_size_gb,
        fabric=args.fabric,
    )

    for i in range(args.iterations):
        print(f"[iter {i + 1}/{args.iterations}]", file=sys.stderr)
        r = run_iteration(args, rollouts)
        artifact.iterations.append(r)
        msg = r.error or f"{r.sync_wall_s:.1f}s"
        print(f"[iter {i + 1}] {msg}", file=sys.stderr)

    artifact.summary = summarize(artifact.iterations)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(artifact), indent=2, default=str))
    print(f"wrote {out} — median {artifact.summary.get('median_s', 'n/a')}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
