#!/usr/bin/env python3
"""
Phase 2a: Concurrent Agent Swarm on g7e.

Runs N concurrent OpenCode agents against a single Qwen3.5-397B TP4 endpoint.
Measures wall time, per-issue latency, and whether concurrency degrades fix rate.

Usage:
  # Sequential baseline (N=1)
  python3 swarm_concurrent.py --concurrency 1 --endpoint http://localhost:8000

  # 4 concurrent agents
  python3 swarm_concurrent.py --concurrency 4 --endpoint http://localhost:8000

  # 8 concurrent (tests vLLM queuing)
  python3 swarm_concurrent.py --concurrency 8 --endpoint http://localhost:8000
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Import shared infrastructure
sys.path.insert(0, "/mnt/nvme/agent-harness-scripts")
from harness_eval import Issue, load_subset, setup_workspace, _get_git_diff


@dataclass
class ConcurrentResult:
    instance_id: str
    model: str
    harness: str
    concurrency: int
    fix_generated: bool = False
    turns_used: int = 0
    tokens_consumed: int = 0
    wall_time_s: float = 0
    queue_wait_s: float = 0  # estimated from vLLM metrics
    error: Optional[str] = None
    worker_id: int = 0


WORKSPACE_BASE = Path("/mnt/nvme/swarm-concurrent-workspaces")
ADAPTER_PATH = Path("/mnt/nvme/agent-harness/adapters/run_opencode.sh")


def run_opencode_agent(
    issue: Issue,
    workspace: str,
    endpoint: str,
    model: str,
    worker_id: int,
    concurrency: int,
) -> ConcurrentResult:
    """Run one OpenCode agent synchronously."""
    result = ConcurrentResult(
        instance_id=issue.instance_id,
        model=model,
        harness="opencode",
        concurrency=concurrency,
        worker_id=worker_id,
    )
    start = time.monotonic()

    env = os.environ.copy()
    env.update({
        "WORKSPACE": workspace,
        "ENDPOINT": endpoint,
        "MODEL": model,
        "ISSUE_ID": issue.instance_id,
        "PROBLEM_STATEMENT": issue.problem_statement[:10000],
        "TEST_CMD": issue.test_cmd,
        "REPO": issue.repo,
    })

    try:
        proc = subprocess.run(
            ["bash", str(ADAPTER_PATH)],
            capture_output=True, text=True,
            timeout=600,
            cwd=workspace,
            env=env,
        )

        lines = [l.strip() for l in proc.stdout.strip().split("\n") if l.strip()]
        if lines:
            try:
                output = json.loads(lines[-1])
                result.turns_used = output.get("turns", 0)
                result.tokens_consumed = output.get("tokens", 0)
                result.fix_generated = output.get("fix_generated", False)
            except json.JSONDecodeError:
                result.error = "parse_error"
        else:
            result.error = "no_output"

        if proc.returncode != 0 and not result.error:
            result.error = proc.stderr[:200]

    except subprocess.TimeoutExpired:
        result.error = "timeout_600s"
    except Exception as e:
        result.error = str(e)[:200]

    # Fallback: check git diff
    if not result.fix_generated:
        result.fix_generated = bool(_get_git_diff(workspace))

    result.wall_time_s = time.monotonic() - start
    return result


async def run_worker(
    worker_id: int,
    issue_queue: asyncio.Queue,
    endpoint: str,
    model: str,
    concurrency: int,
    results: list,
    output_path: str,
    completed: set,
):
    """Worker coroutine: pulls issues from queue, runs OpenCode, saves results."""
    loop = asyncio.get_event_loop()

    while True:
        try:
            idx, issue = issue_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        if issue.instance_id in completed:
            log.info(f"  W{worker_id} [{idx+1}] {issue.instance_id} — skipped (done)")
            issue_queue.task_done()
            continue

        log.info(f"  W{worker_id} [{idx+1}] {issue.instance_id} — starting")

        # Setup workspace (blocking, but fast)
        workspace_dir = str(WORKSPACE_BASE / f"w{worker_id}")
        os.makedirs(workspace_dir, exist_ok=True)

        try:
            workspace = setup_workspace(issue, workspace_dir)
        except Exception as e:
            log.warning(f"  W{worker_id} workspace setup failed: {e}")
            result = ConcurrentResult(
                instance_id=issue.instance_id,
                model=model,
                harness="opencode",
                concurrency=concurrency,
                worker_id=worker_id,
                error=f"workspace: {str(e)[:200]}",
            )
            results.append(asdict(result))
            with open(output_path, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")
            issue_queue.task_done()
            continue

        # Run OpenCode in thread pool (blocking subprocess)
        result = await loop.run_in_executor(
            None,
            run_opencode_agent,
            issue, workspace, endpoint, model, worker_id, concurrency,
        )

        status = "FIX" if result.fix_generated else ("ERR" if result.error else "NOFIX")
        log.info(
            f"  W{worker_id} [{idx+1}] {issue.instance_id} — "
            f"{status} ({result.wall_time_s:.0f}s, {result.turns_used} turns)"
        )

        results.append(asdict(result))

        # Incremental write
        with open(output_path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

        # Cleanup workspace
        import shutil
        shutil.rmtree(workspace, ignore_errors=True)

        issue_queue.task_done()


async def run_swarm(
    concurrency: int,
    endpoint: str,
    model: str,
    output_path: str,
):
    """Run the concurrent swarm."""
    issues = load_subset()
    log.info(f"Loaded {len(issues)} issues")
    log.info(f"Concurrency: {concurrency} workers")
    log.info(f"Endpoint: {endpoint}")
    log.info(f"Model: {model}")

    # Resume support
    completed = set()
    results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed.add(r["instance_id"])
                    results.append(r)
        if completed:
            log.info(f"Resuming from {len(completed)} completed")

    # Build queue
    queue = asyncio.Queue()
    for i, issue in enumerate(issues):
        queue.put_nowait((i, issue))

    WORKSPACE_BASE.mkdir(parents=True, exist_ok=True)

    swarm_start = time.monotonic()

    # Launch workers
    workers = []
    for wid in range(concurrency):
        w = asyncio.create_task(
            run_worker(wid, queue, endpoint, model, concurrency, results, output_path, completed)
        )
        workers.append(w)

    await asyncio.gather(*workers)

    swarm_wall = time.monotonic() - swarm_start

    # Summary
    n = len(results)
    fixes = sum(1 for r in results if r.get("fix_generated"))
    errors = sum(1 for r in results if r.get("error"))
    wall_times = [r["wall_time_s"] for r in results if r.get("wall_time_s", 0) > 0]
    avg_wall = sum(wall_times) / len(wall_times) if wall_times else 0
    p50_wall = sorted(wall_times)[len(wall_times)//2] if wall_times else 0
    p99_wall = sorted(wall_times)[int(len(wall_times)*0.99)] if wall_times else 0

    log.info(f"\n{'='*70}")
    log.info(f"PHASE 2a: CONCURRENT SWARM RESULTS (N={concurrency})")
    log.info(f"{'='*70}")
    log.info(f"  Issues:       {n}")
    log.info(f"  Fixes:        {fixes}/{n} ({100*fixes/n:.0f}%)")
    log.info(f"  Errors:       {errors}")
    log.info(f"  Wall time:    {swarm_wall:.0f}s ({swarm_wall/60:.1f} min)")
    log.info(f"  Per-issue:    avg={avg_wall:.0f}s  p50={p50_wall:.0f}s  p99={p99_wall:.0f}s")
    log.info(f"  Throughput:   {n/swarm_wall*60:.1f} issues/min")
    log.info(f"  Speedup:      {avg_wall*n/swarm_wall:.1f}x vs sequential")
    log.info(f"  Output:       {output_path}")

    # Write summary
    summary = {
        "concurrency": concurrency,
        "model": model,
        "total": n,
        "fixes": fixes,
        "fix_rate": round(100*fixes/n, 1) if n else 0,
        "errors": errors,
        "swarm_wall_s": round(swarm_wall, 1),
        "avg_wall_s": round(avg_wall, 1),
        "p50_wall_s": round(p50_wall, 1),
        "p99_wall_s": round(p99_wall, 1),
        "throughput_per_min": round(n/swarm_wall*60, 2) if swarm_wall else 0,
        "speedup": round(avg_wall*n/swarm_wall, 2) if swarm_wall else 0,
    }
    summary_path = output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"  Summary:      {summary_path}")


async def main():
    parser = argparse.ArgumentParser(description="Phase 2a: Concurrent Agent Swarm")
    parser.add_argument("--concurrency", "-n", type=int, default=1, help="Number of concurrent agents")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="vLLM endpoint")
    parser.add_argument("--model", default="qwen35-397b", help="Model name")
    parser.add_argument("--output-dir", default="results/swarm", help="Output directory")
    args = parser.parse_args()

    output_path = os.path.join(
        args.output_dir,
        f"swarm_phase2a_n{args.concurrency}_{args.model}_opencode.jsonl",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    await run_swarm(args.concurrency, args.endpoint, args.model, output_path)


if __name__ == "__main__":
    asyncio.run(main())
