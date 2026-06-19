"""Unified cold-start profiler for the ai-infra lab.

Drives a single experiment run end-to-end:
  apply manifest -> watch k8s events -> stream pod logs -> probe /health
  -> stream first-token request -> emit canonical artifact JSON.

Every spec in the lab calls this with a different manifest and variant tag.
Stage boundaries and event names are canonical across specs (see
log_patterns.yaml and specs/profiler-validation.md).

Usage:
    python profiler.py \\
        --manifest manifests/glm5-soci.yaml \\
        --namespace ai-infra \\
        --endpoint http://glm5.ai-infra.svc:8000 \\
        --model glm-5-fp8 \\
        --experiment image-pull-acceleration \\
        --variant snapshotter=soci fuse_tuning=modal-default \\
        --fixture domains/gpu-serving/blueprints/glm5-fp8/ \\
        --vllm-version 0.18.0 \\
        --out results/run-001.json

Implementation notes:
- Uses subprocess + kubectl rather than the python-kubernetes client to avoid
  pulling in a heavyweight dep tree. The kubectl JSON output is stable enough.
- Runs three concurrent watchers (events, logs, probe) feeding a single
  asyncio queue, drained by a writer that records monotonic timestamps.
- All timestamps come from time.monotonic() captured by the orchestrator
  *at the moment we observe* the event (not from the source's wallclock).
  This is intentional: it means we measure the latency from event-occurred
  to event-observed as part of "system overhead" rather than introducing
  cross-source clock skew. Spec 0 validates that this overhead is small.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml  # PyYAML; install in harness venv


CANONICAL_EVENTS = [
    "T0_pod_create",
    "T1_node_assigned",
    "T2_image_pull_start",
    "T3_image_pull_complete",
    "T4_container_created",
    "T5_container_started",
    "T6_python_alive",
    "T7_weights_load_start",
    "T8_weights_loaded",
    "T9_jit_compile_start",
    "T10_jit_compile_done",
    "T11_cuda_graphs_done",
    "T12_health_200",
    "T13_first_token",
]


@dataclass
class Event:
    name: str
    t_mono_s: float
    source: str
    raw: str = ""


@dataclass
class Stage:
    start: str
    end: str
    elapsed_s: float | None


@dataclass
class Artifact:
    schema: str = "benchmark-commons/v1"
    run_id: str = ""
    started_at_wallclock: float = 0.0
    experiment: str = ""
    variant: dict[str, str] = field(default_factory=dict)
    fixture_blueprint: str = ""
    vllm_version: str = ""
    namespace: str = ""
    pod: str = ""
    events: list[Event] = field(default_factory=list)
    stages: dict[str, dict] = field(default_factory=dict)
    gaps: dict[str, float] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)


# -- regex helpers --------------------------------------------------------


def load_patterns(path: Path, vllm_version: str) -> dict[str, list[re.Pattern]]:
    """Pick the patterns block whose applies_to matches vllm_version."""
    raw = yaml.safe_load(path.read_text())
    chosen = None
    for block in raw["ranges"]:
        # Lazy interpretation of applies_to: just check if version is mentioned.
        # For real version-range matching, install packaging.requirements.
        if vllm_version in block["applies_to"] or block["applies_to"].endswith("*"):
            chosen = block
            break
    if chosen is None:
        chosen = raw["ranges"][0]
        print(
            f"[profiler] WARNING: no pattern range matches vllm={vllm_version}; "
            f"falling back to {chosen['applies_to']}",
            file=sys.stderr,
        )

    compiled: dict[str, list[re.Pattern]] = {}
    for evt, body in chosen["events"].items():
        if body.get("any_line"):
            compiled[evt] = []  # any line matches; handled specially
        else:
            compiled[evt] = [re.compile(p) for p in body.get("regex", [])]
    return compiled


# -- watchers -------------------------------------------------------------


async def watch_pod_events(
    pod: str, namespace: str, queue: asyncio.Queue, stop: asyncio.Event
) -> None:
    """Stream kubectl pod events, emit T1/T2/T3/T4/T5 + raw."""
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "get", "events", "-n", namespace,
        "--field-selector", f"involvedObject.name={pod}",
        "-w", "-o", "json", "--output-watch-events=true",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    reason_to_event = {
        "Scheduled": "T1_node_assigned",
        "Pulling": "T2_image_pull_start",
        "Pulled": "T3_image_pull_complete",
        "Created": "T4_container_created",
        "Started": "T5_container_started",
    }
    seen: set[str] = set()
    buf = b""
    assert proc.stdout is not None
    while not stop.is_set():
        try:
            chunk = await asyncio.wait_for(proc.stdout.read(4096), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        if not chunk:
            break
        buf += chunk
        # kubectl emits one JSON object per "watch event" but pretty-printed,
        # so split on '\n}\n{'-equivalent. Cheap parser: try decoding chunks
        # ending with '\n}\n'.
        while True:
            try:
                obj, idx = json.JSONDecoder().raw_decode(buf.decode(errors="replace"))
            except json.JSONDecodeError:
                break
            buf = buf[idx:].lstrip().encode()
            evt = obj.get("object", obj)
            reason = evt.get("reason", "")
            if reason in reason_to_event and reason_to_event[reason] not in seen:
                name = reason_to_event[reason]
                seen.add(name)
                await queue.put(Event(
                    name=name,
                    t_mono_s=time.monotonic(),
                    source="k8s_event",
                    raw=evt.get("message", ""),
                ))
    proc.terminate()


async def watch_pod_logs(
    pod: str,
    namespace: str,
    patterns: dict[str, list[re.Pattern]],
    queue: asyncio.Queue,
    stop: asyncio.Event,
) -> None:
    """Stream kubectl logs, match each line against canonical patterns."""
    # Wait briefly for pod to be at least Pending+Created so kubectl logs works.
    await asyncio.sleep(0.5)
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "logs", "-n", namespace, pod, "-f",
        "--all-containers=true", "--prefix=true",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    # Order matters: T6 fires on first stdout line; T7-T11 are regex-matched.
    # We fire each event id at most once.
    fired: set[str] = set()
    assert proc.stdout is not None
    while not stop.is_set():
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        if not line:
            break
        text = line.decode(errors="replace").rstrip()
        now = time.monotonic()

        if "T6_python_alive" not in fired:
            fired.add("T6_python_alive")
            await queue.put(Event(
                name="T6_python_alive", t_mono_s=now, source="workload_log", raw=text,
            ))

        for evt in [
            "T7_weights_load_start",
            "T8_weights_loaded",
            "T9_jit_compile_start",
            "T10_jit_compile_done",
            "T11_cuda_graphs_done",
        ]:
            if evt in fired:
                continue
            for pat in patterns.get(evt, []):
                if pat.search(text):
                    fired.add(evt)
                    await queue.put(Event(
                        name=evt, t_mono_s=now, source="workload_log", raw=text,
                    ))
                    break
    proc.terminate()


async def probe_health_and_first_token(
    endpoint: str, model: str, queue: asyncio.Queue, stop: asyncio.Event
) -> None:
    """Poll /health until 200, then issue a streaming completion and time first chunk."""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline and not stop.is_set():
        try:
            with urllib.request.urlopen(f"{endpoint}/health", timeout=5) as resp:
                if resp.status == 200:
                    await queue.put(Event(
                        name="T12_health_200",
                        t_mono_s=time.monotonic(),
                        source="harness_probe",
                    ))
                    break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        await asyncio.sleep(1.0)
    else:
        return

    payload = json.dumps({
        "model": model, "prompt": "Hello", "max_tokens": 4, "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/completions", data=payload,
        headers={"Content-Type": "application/json"},
    )
    sent_at = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line.removeprefix("data:").strip()
            if not body or body == "[DONE]":
                continue
            chunk = json.loads(body)
            text = chunk.get("choices", [{}])[0].get("text", "")
            if text:
                await queue.put(Event(
                    name="T13_first_token",
                    t_mono_s=time.monotonic(),
                    source="harness_sse",
                    raw=f"sent_at={sent_at:.3f}",
                ))
                break
    except Exception as exc:
        await queue.put(Event(
            name="T13_first_token",
            t_mono_s=time.monotonic(),
            source="harness_sse",
            raw=f"ERROR: {exc}",
        ))


# -- orchestration --------------------------------------------------------


async def run_profile(args: argparse.Namespace) -> Artifact:
    artifact = Artifact(
        run_id=str(uuid.uuid4())[:8],
        started_at_wallclock=time.time(),
        experiment=args.experiment,
        variant=dict(kv.split("=", 1) for kv in args.variant),
        fixture_blueprint=args.fixture,
        vllm_version=args.vllm_version,
        namespace=args.namespace,
    )

    patterns = load_patterns(Path(args.patterns), args.vllm_version)

    # T0: apply manifest, capture pod name.
    t0 = time.monotonic()
    artifact.events.append(Event("T0_pod_create", t0, "harness"))

    apply = await asyncio.create_subprocess_exec(
        "kubectl", "apply", "-n", args.namespace, "-f", args.manifest,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await apply.communicate()
    pod = None
    for line in out.decode().splitlines():
        if line.startswith("pod/"):
            pod = line.split()[0].removeprefix("pod/").rstrip().split()[0]
            break
    if pod is None:
        raise RuntimeError(f"could not parse pod name; out={out!r} err={err!r}")
    artifact.pod = pod

    queue: asyncio.Queue = asyncio.Queue()
    stop = asyncio.Event()

    watchers = [
        asyncio.create_task(watch_pod_events(pod, args.namespace, queue, stop)),
        asyncio.create_task(watch_pod_logs(pod, args.namespace, patterns, queue, stop)),
        asyncio.create_task(probe_health_and_first_token(
            args.endpoint, args.model, queue, stop,
        )),
    ]

    deadline = time.monotonic() + args.timeout
    seen: set[str] = {"T0_pod_create"}
    while time.monotonic() < deadline:
        try:
            evt = await asyncio.wait_for(queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            if "T13_first_token" in seen:
                break
            continue
        if evt.name in seen:
            continue
        seen.add(evt.name)
        artifact.events.append(evt)
        if evt.name == "T13_first_token":
            break

    stop.set()
    for w in watchers:
        w.cancel()
    await asyncio.gather(*watchers, return_exceptions=True)

    if not args.keep_pod:
        await asyncio.create_subprocess_exec(
            "kubectl", "delete", "pod", pod, "-n", args.namespace,
            "--wait=false",
        )

    compute_stages_and_gaps(artifact, Path(args.patterns))
    return artifact


def compute_stages_and_gaps(artifact: Artifact, patterns_path: Path) -> None:
    raw = yaml.safe_load(patterns_path.read_text())
    by_name = {e.name: e for e in artifact.events}
    for stage_name, (start, end) in raw["stages"].items():
        if start in by_name and end in by_name:
            artifact.stages[stage_name] = {
                "start": start, "end": end,
                "elapsed_s": by_name[end].t_mono_s - by_name[start].t_mono_s,
            }
        else:
            artifact.stages[stage_name] = {
                "start": start, "end": end, "elapsed_s": None,
            }
    for start, end in raw["gaps"]:
        if start in by_name and end in by_name:
            artifact.gaps[f"{start}__{end}"] = (
                by_name[end].t_mono_s - by_name[start].t_mono_s
            )
    if "T0_pod_create" in by_name and "T13_first_token" in by_name:
        artifact.totals["pod_create_to_first_token_s"] = (
            by_name["T13_first_token"].t_mono_s - by_name["T0_pod_create"].t_mono_s
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--namespace", default="ai-infra")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--variant", action="append", default=[], help="key=value, repeatable")
    p.add_argument("--fixture", required=True, help="path to gpu-serving blueprint used as fixture")
    p.add_argument("--vllm-version", required=True)
    p.add_argument("--patterns", default=str(Path(__file__).parent / "log_patterns.yaml"))
    p.add_argument("--timeout", type=int, default=2400)
    p.add_argument("--keep-pod", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    artifact = asyncio.run(run_profile(args))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(artifact), indent=2, default=str))
    total = artifact.totals.get("pod_create_to_first_token_s", "n/a")
    print(f"wrote {out} -- total {total}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
