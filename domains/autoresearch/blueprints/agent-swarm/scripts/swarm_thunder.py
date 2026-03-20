#!/usr/bin/env python3
"""
Phase 2b: Agent Swarm with ThunderAgent Scheduling.

Wraps swarm_concurrent.py but:
1. Starts thunder_proxy.py on port 9000
2. Points agents at the proxy instead of vLLM directly
3. Sets X-Program-ID headers per agent session
4. Collects ThunderAgent metrics alongside fix rates

Usage:
  # Config D: ThunderAgent at N=4 (same as Phase 2a sweet spot)
  python3 swarm_thunder.py --concurrency 4 --tool-coefficient 1.5

  # Config E: Oversubscribed N=8 with scheduling
  python3 swarm_thunder.py --concurrency 8 --tool-coefficient 2.0
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Import shared infrastructure
sys.path.insert(0, "/mnt/nvme/agent-harness-scripts")
from harness_eval import Issue, load_subset, setup_workspace, _get_git_diff


@dataclass
class ThunderResult:
    instance_id: str
    model: str
    harness: str
    concurrency: int
    tool_coefficient: float
    fix_generated: bool = False
    turns_used: int = 0
    tokens_consumed: int = 0
    wall_time_s: float = 0
    queue_wait_s: float = 0
    admit_type: str = ""  # direct, backfill, queued
    error: Optional[str] = None
    worker_id: int = 0


WORKSPACE_BASE = Path("/mnt/nvme/swarm-thunder-workspaces")
ADAPTER_PATH = Path("/mnt/nvme/agent-harness/adapters/run_opencode.sh")
THUNDER_SCRIPT = Path(__file__).parent / "thunder_proxy.py"


def run_opencode_agent(
    issue: Issue,
    workspace: str,
    proxy_endpoint: str,
    model: str,
    worker_id: int,
    concurrency: int,
    tool_coefficient: float,
    program_id: str,
) -> ThunderResult:
    """Run one OpenCode agent, pointed at ThunderAgent proxy."""
    result = ThunderResult(
        instance_id=issue.instance_id,
        model=model,
        harness="opencode+thunder",
        concurrency=concurrency,
        tool_coefficient=tool_coefficient,
        worker_id=worker_id,
    )
    start = time.monotonic()

    env = os.environ.copy()
    env.update({
        "WORKSPACE": workspace,
        "ENDPOINT": proxy_endpoint,  # Port 9000 (proxy), not 8000 (vLLM)
        "MODEL": model,
        "ISSUE_ID": issue.instance_id,
        "PROBLEM_STATEMENT": issue.problem_statement[:10000],
        "TEST_CMD": issue.test_cmd,
        "REPO": issue.repo,
        # ThunderAgent program ID — encoded as API key so OpenCode forwards it
        # Proxy extracts "prog-*" from Authorization header
        "THUNDER_PROGRAM_ID": program_id,
        "OPENCODE_API_KEY": program_id,
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


async def fetch_thunder_metrics(proxy_url: str) -> dict:
    """Fetch current metrics from ThunderAgent proxy."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{proxy_url}/thunder/metrics", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return await resp.json()
    except Exception:
        return {}


async def run_worker(
    worker_id: int,
    issue_queue: asyncio.Queue,
    proxy_endpoint: str,
    model: str,
    concurrency: int,
    tool_coefficient: float,
    results: list,
    output_path: str,
    completed: set,
):
    """Worker coroutine: pulls issues from queue, runs OpenCode via proxy."""
    loop = asyncio.get_event_loop()

    while True:
        try:
            idx, issue = issue_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        if issue.instance_id in completed:
            log.info("  W%d [%d] %s -- skipped (done)", worker_id, idx+1, issue.instance_id)
            issue_queue.task_done()
            continue

        # Unique program ID for ThunderAgent tracking
        program_id = "prog-w{}-{}".format(worker_id, issue.instance_id.replace("/", "-")[:40])

        log.info("  W%d [%d] %s -- starting (program=%s)", worker_id, idx+1, issue.instance_id, program_id[:20])

        # Setup workspace
        workspace_dir = str(WORKSPACE_BASE / "w{}".format(worker_id))
        os.makedirs(workspace_dir, exist_ok=True)

        try:
            workspace = setup_workspace(issue, workspace_dir)
        except Exception as e:
            log.warning("  W%d workspace setup failed: %s", worker_id, e)
            result = ThunderResult(
                instance_id=issue.instance_id,
                model=model,
                harness="opencode+thunder",
                concurrency=concurrency,
                tool_coefficient=tool_coefficient,
                worker_id=worker_id,
                error="workspace: {}".format(str(e)[:200]),
            )
            results.append(asdict(result))
            with open(output_path, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")
            issue_queue.task_done()
            continue

        # Run OpenCode via ThunderAgent proxy
        result = await loop.run_in_executor(
            None,
            run_opencode_agent,
            issue, workspace, proxy_endpoint, model,
            worker_id, concurrency, tool_coefficient, program_id,
        )

        status = "FIX" if result.fix_generated else ("ERR" if result.error else "NOFIX")
        log.info(
            "  W%d [%d] %s -- %s (%ds, %d turns)",
            worker_id, idx+1, issue.instance_id,
            status, result.wall_time_s, result.turns_used,
        )

        results.append(asdict(result))

        # Incremental write
        with open(output_path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

        # Cleanup workspace
        shutil.rmtree(workspace, ignore_errors=True)
        issue_queue.task_done()


async def run_swarm(
    concurrency: int,
    backend: str,
    proxy_port: int,
    model: str,
    tool_coefficient: float,
    output_path: str,
    max_active: int,
):
    """Start ThunderAgent proxy, then run the concurrent swarm through it."""
    issues = load_subset()
    log.info("Loaded %d issues", len(issues))
    log.info("Concurrency: %d workers", concurrency)
    log.info("Backend: %s (via ThunderAgent proxy :%d)", backend, proxy_port)
    log.info("Tool coefficient: %.1f (effective slots: %d)", tool_coefficient, int(max_active * tool_coefficient))

    # Start ThunderAgent proxy as subprocess
    proxy_cmd = [
        sys.executable, str(THUNDER_SCRIPT),
        "--backend", backend,
        "--port", str(proxy_port),
        "--max-active", str(max_active),
        "--tool-coefficient", str(tool_coefficient),
    ]
    log.info("Starting ThunderAgent proxy: %s", " ".join(proxy_cmd))
    proxy_log_path = output_path.replace(".jsonl", "_proxy.log")
    proxy_log_file = open(proxy_log_path, "w")
    proxy_proc = subprocess.Popen(
        proxy_cmd,
        stdout=proxy_log_file,
        stderr=subprocess.STDOUT,
    )
    log.info("Proxy log: %s", proxy_log_path)

    # Wait for proxy to be ready
    proxy_url = "http://localhost:{}".format(proxy_port)
    for attempt in range(20):
        await asyncio.sleep(1)
        metrics = await fetch_thunder_metrics(proxy_url)
        if metrics:
            log.info("ThunderAgent proxy ready on :%d", proxy_port)
            break
    else:
        log.error("ThunderAgent proxy failed to start")
        proxy_proc.kill()
        return

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
            log.info("Resuming from %d completed", len(completed))

    # Build queue
    queue = asyncio.Queue()
    for i, issue in enumerate(issues):
        queue.put_nowait((i, issue))

    WORKSPACE_BASE.mkdir(parents=True, exist_ok=True)

    swarm_start = time.monotonic()

    # Periodic metrics collection
    metrics_snapshots = []

    async def collect_metrics():
        while True:
            await asyncio.sleep(15)
            m = await fetch_thunder_metrics(proxy_url)
            if m:
                metrics_snapshots.append(m)

    metrics_task = asyncio.create_task(collect_metrics())

    # Launch workers — point at proxy, not backend
    proxy_endpoint = "http://localhost:{}".format(proxy_port)
    workers = []
    for wid in range(concurrency):
        w = asyncio.create_task(
            run_worker(
                wid, queue, proxy_endpoint, model, concurrency,
                tool_coefficient, results, output_path, completed,
            )
        )
        workers.append(w)

    await asyncio.gather(*workers)
    metrics_task.cancel()

    swarm_wall = time.monotonic() - swarm_start

    # Final ThunderAgent metrics
    final_metrics = await fetch_thunder_metrics(proxy_url)

    # Kill proxy
    proxy_proc.terminate()
    try:
        proxy_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proxy_proc.kill()
    proxy_log_file.close()

    # Summary
    n = len(results)
    fixes = sum(1 for r in results if r.get("fix_generated"))
    errors = sum(1 for r in results if r.get("error"))
    wall_times = [r["wall_time_s"] for r in results if r.get("wall_time_s", 0) > 0]
    avg_wall = sum(wall_times) / len(wall_times) if wall_times else 0
    p50_wall = sorted(wall_times)[len(wall_times)//2] if wall_times else 0

    # ThunderAgent-specific metrics
    backfill_rate = 0
    avg_util = 0
    if final_metrics:
        total_req = final_metrics.get("total_requests", 0)
        backfills = final_metrics.get("backfill_admits", 0)
        backfill_rate = round(100 * backfills / total_req, 1) if total_req else 0
        avg_util = final_metrics.get("gpu_utilization_pct", 0)

    if metrics_snapshots:
        avg_util = round(
            sum(m.get("gpu_utilization_pct", 0) for m in metrics_snapshots)
            / len(metrics_snapshots), 1
        )

    log.info("")
    log.info("=" * 70)
    log.info("PHASE 2b: THUNDERAGENT SWARM RESULTS (N=%d, tc=%.1f)", concurrency, tool_coefficient)
    log.info("=" * 70)
    log.info("  Issues:         %d", n)
    log.info("  Fixes:          %d/%d (%d%%)", fixes, n, 100*fixes//n if n else 0)
    log.info("  Errors:         %d", errors)
    log.info("  Wall time:      %ds (%.1f min)", swarm_wall, swarm_wall/60)
    log.info("  Per-issue:      avg=%ds  p50=%ds", avg_wall, p50_wall)
    log.info("  Throughput:     %.1f issues/min", n/swarm_wall*60 if swarm_wall else 0)
    log.info("  Speedup:        %.1fx vs sequential", avg_wall*n/swarm_wall if swarm_wall else 0)
    log.info("  --- ThunderAgent ---")
    log.info("  Backfill rate:  %.1f%%", backfill_rate)
    log.info("  Avg GPU util:   %.1f%%", avg_util)
    log.info("  Total requests: %d", final_metrics.get("total_requests", 0) if final_metrics else 0)
    log.info("  Direct admits:  %d", final_metrics.get("direct_admits", 0) if final_metrics else 0)
    log.info("  Backfill admits:%d", final_metrics.get("backfill_admits", 0) if final_metrics else 0)
    log.info("  Output:         %s", output_path)

    # Write summary
    summary = {
        "phase": "2b",
        "scheduler": "thunderagent",
        "concurrency": concurrency,
        "tool_coefficient": tool_coefficient,
        "model": model,
        "total": n,
        "fixes": fixes,
        "fix_rate": round(100*fixes/n, 1) if n else 0,
        "errors": errors,
        "swarm_wall_s": round(swarm_wall, 1),
        "avg_wall_s": round(avg_wall, 1),
        "p50_wall_s": round(p50_wall, 1),
        "throughput_per_min": round(n/swarm_wall*60, 2) if swarm_wall else 0,
        "speedup": round(avg_wall*n/swarm_wall, 2) if swarm_wall else 0,
        "backfill_rate_pct": backfill_rate,
        "avg_gpu_util_pct": avg_util,
        "thunder_metrics": final_metrics or {},
        "metrics_snapshots_count": len(metrics_snapshots),
    }
    summary_path = output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("  Summary:        %s", summary_path)


async def main():
    parser = argparse.ArgumentParser(description="Phase 2b: ThunderAgent Agent Swarm")
    parser.add_argument("--concurrency", "-n", type=int, default=4, help="Number of concurrent agents")
    parser.add_argument("--backend", default="http://localhost:8000", help="vLLM backend URL")
    parser.add_argument("--proxy-port", type=int, default=9000, help="ThunderAgent proxy port")
    parser.add_argument("--model", default="qwen35-397b", help="Model name")
    parser.add_argument("--tool-coefficient", type=float, default=1.5, help="Oversubscription factor")
    parser.add_argument("--max-active", type=int, default=4, help="vLLM max-num-seqs")
    parser.add_argument("--output-dir", default="results/swarm", help="Output directory")
    args = parser.parse_args()

    output_path = os.path.join(
        args.output_dir,
        "swarm_phase2b_n{}_tc{}_{}_{}.jsonl".format(
            args.concurrency,
            str(args.tool_coefficient).replace(".", ""),
            args.model, "opencode",
        ),
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    await run_swarm(
        concurrency=args.concurrency,
        backend=args.backend,
        proxy_port=args.proxy_port,
        model=args.model,
        tool_coefficient=args.tool_coefficient,
        output_path=output_path,
        max_active=args.max_active,
    )


if __name__ == "__main__":
    asyncio.run(main())
